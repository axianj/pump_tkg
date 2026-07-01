"""
Kùzu 嵌入式图数据库后端

Kùzu 是 Python 原生的嵌入式图数据库，无需独立服务进程，
pip install 即用，适合桌面应用场景。

与 Neo4j 对比:
- Kùzu: 嵌入式, pip install kuzu, 列式存储, Cypher 兼容, 零运维
- Neo4j: 独立服务, Java 运行时, 行式存储, 成熟生态

Phase 6 评估用途:
1. 桌面应用部署（无需安装 Neo4j Desktop）
2. 离线环境下的图存储替代方案

切换方式:
    from core.graph_store import get_store
    store = get_store("kuzu")  # 或 "neo4j"
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import kuzu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class KuzuGraphStore:
    """
    Kùzu 嵌入式图数据库后端。

    提供与 TemporalStore 相同的接口，以便无痛切换。
    """

    def __init__(self, db_path: str = "data/output/kuzu_db"):
        self._db_path = Path(PROJECT_ROOT) / db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)

        self._schema_initialized = False
        self._ensure_schema()

    def _ensure_schema(self):
        """创建图 schema（如果不存在）"""
        try:
            # 实体节点表
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity (
                    id STRING PRIMARY KEY,
                    name STRING,
                    type STRING,
                    description STRING,
                    source STRING,
                    degree INT64,
                    timestamp STRING
                )
            """)

            # TEMPORAL 关系表
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS TEMPORAL (
                    FROM Entity TO Entity,
                    relation_type STRING,
                    valid_from STRING,
                    valid_to STRING,
                    confidence DOUBLE,
                    source STRING,
                    description STRING
                )
            """)

            self._schema_initialized = True
        except Exception:
            # Schema 可能已存在
            self._schema_initialized = True

    # ── 实体操作 ────────────────────────────────────

    def insert_entity(self, entity_id: str, entity_type: str, name: str,
                      description: str = "", source: str = "",
                      timestamp: Optional[str] = None):
        self._conn.execute(
            """
            MERGE (e:Entity {id: $id})
            SET e.name = $name, e.type = $type,
                e.description = $description, e.source = $source,
                e.timestamp = $timestamp
            """,
            {"id": entity_id, "name": name, "type": entity_type,
             "description": description, "source": source, "timestamp": timestamp},
        )

    def insert_entities_batch(self, entities: List[Dict]):
        """批量导入实体（从 entities.json）"""
        for e in entities:
            try:
                self.insert_entity(
                    entity_id=e["id"],
                    entity_type=e["type"],
                    name=e["name"],
                    description=e.get("description", ""),
                    source=e.get("source", ""),
                    timestamp=e.get("measurement_attrs", {}).get("timestamp") if "measurement_attrs" in e else None,
                )
            except Exception:
                pass

    # ── 关系操作 ────────────────────────────────────

    def insert_temporal_edge(
        self, source: str, target: str, relation_type: str,
        valid_from: Optional[str] = None, valid_to: Optional[str] = None,
        confidence: float = 1.0, description: str = "",
    ):
        self._conn.execute(
            """
            MATCH (a:Entity {id: $source}), (b:Entity {id: $target})
            MERGE (a)-[r:TEMPORAL {relation_type: $rel_type}]->(b)
            SET r.valid_from = $valid_from,
                r.valid_to = $valid_to,
                r.confidence = $confidence,
                r.description = $description
            """,
            {
                "source": source, "target": target, "rel_type": relation_type,
                "valid_from": valid_from, "valid_to": valid_to,
                "confidence": confidence, "description": description,
            },
        )

    def insert_edges_batch(self, edges: List[Dict]):
        """批量导入关系（从 relationships.json）"""
        for r in edges:
            try:
                self.insert_temporal_edge(
                    source=r["source"], target=r["target"],
                    relation_type=r["relation"],
                    description=r.get("description", ""),
                    confidence=r.get("weight", 1.0),
                )
            except Exception:
                pass

    # ── 查询操作 ────────────────────────────────────

    def query_entities(self, entity_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        if entity_type:
            result = self._conn.execute(
                "MATCH (e:Entity) WHERE e.type = $type RETURN e.* LIMIT $limit",
                {"type": entity_type, "limit": limit},
            )
        else:
            result = self._conn.execute(
                "MATCH (e:Entity) RETURN e.* LIMIT $limit",
                {"limit": limit},
            )
        out = []
        while result.has_next():
            row = result.get_next()
            out.append({
                "id": row[0], "name": row[1], "type": row[2],
                "description": row[3], "source": row[4],
                "degree": row[5], "timestamp": row[6],
            })
        return out

    def query_edges(self, source_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        if source_id:
            result = self._conn.execute(
                "MATCH (a:Entity {id: $id})-[r:TEMPORAL]->(b:Entity) "
                "RETURN a.id, r.relation_type, b.id, r.* LIMIT $limit",
                {"id": source_id, "limit": limit},
            )
        else:
            result = self._conn.execute(
                "MATCH (a:Entity)-[r:TEMPORAL]->(b:Entity) "
                "RETURN a.id, r.relation_type, b.id, r.* LIMIT $limit",
                {"limit": limit},
            )
        out = []
        while result.has_next():
            row = result.get_next()
            out.append({
                "source": row[0], "relation": row[1], "target": row[2],
                "valid_from": row[3], "valid_to": row[4],
                "confidence": row[5], "description": row[6],
            })
        return out

    def trace_path(self, source_id: str, max_depth: int = 3) -> List[List[Dict]]:
        """时序链路径追踪"""
        paths = []
        result = self._conn.execute(
            f"""
            MATCH p = (a:Entity {{id: $id}})-[r:TEMPORAL*1..{max_depth}]->(b:Entity)
            RETURN relationships(p)
            LIMIT 10
            """,
            {"id": source_id},
        )
        while result.has_next():
            rels = result.get_next()[0]
            steps = []
            for r in rels:
                steps.append({
                    "source": str(r["_src"]) if "_src" in r else str(r.get("_src", "")),
                    "target": str(r["_dst"]) if "_dst" in r else str(r.get("_dst", "")),
                    "relation": str(r.get("relation_type", "")),
                    "description": str(r.get("description", "")),
                })
            if steps:
                paths.append(steps)
        return paths

    # ── 统计 ────────────────────────────────────────

    def count_entities(self) -> int:
        result = self._conn.execute("MATCH (e:Entity) RETURN count(e)")
        return result.get_next()[0] if result.has_next() else 0

    def count_edges(self) -> int:
        result = self._conn.execute("MATCH ()-[r:TEMPORAL]->() RETURN count(r)")
        return result.get_next()[0] if result.has_next() else 0

    def close(self):
        self._conn.close()
        self._db.close()


# ── 统一接口 ────────────────────────────────────────

def get_graph_store(backend: str = "auto", **kwargs) -> object:
    """
    获取图数据库实例。

    Args:
        backend: "neo4j" | "kuzu" | "auto" (自动选择)
    """
    if backend == "kuzu":
        return KuzuGraphStore(**kwargs)
    elif backend == "neo4j":
        from core.temporal.temporal_store import TemporalStore
        return TemporalStore(**kwargs)
    elif backend == "auto":
        # 优先尝试 Kùzu（不需要外部服务），失败则回退到 Neo4j
        try:
            s = KuzuGraphStore(**kwargs)
            print("[GraphStore] 使用 Kùzu 嵌入式图数据库")
            return s
        except Exception:
            print("[GraphStore] Kùzu 不可用，回退到 Neo4j")
            from core.temporal.temporal_store import TemporalStore
            return TemporalStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── 便捷导入 ────────────────────────────────────────

def import_from_json(store: KuzuGraphStore, entities_path: str, edges_path: str):
    """从 entities.json + relationships.json 导入到 Kùzu"""
    import json

    with open(entities_path, "r", encoding="utf-8") as f:
        entities = json.load(f)
    with open(edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    print(f"[Kuzu Import] {len(entities)} 实体, {len(edges)} 关系")
    store.insert_entities_batch(entities)
    store.insert_edges_batch(edges)
    print(f"[Kuzu Import] 完成: {store.count_entities()} 实体, "
          f"{store.count_edges()} 关系")
