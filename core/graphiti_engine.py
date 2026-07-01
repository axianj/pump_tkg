"""
Graphiti 集成模块 — 方案B 的时序知识图谱后端

Graphiti 是唯一原生支持时序 KG 的开源框架，提供:
- 双时态模型（事件时间 event_time + 摄入时间 ingest_time）
- 增量更新（无需重建）
- 矛盾消解
- 混合检索（语义嵌入 + 关键词 + 图遍历）

当前实现:
- 已通过 pip install -e 本地克隆安装
- 依赖 Neo4j 作为图存储后端
- 从 entities.json + relationships.json 迁移到 Graphiti 双时态格式
- 通过 measurement_id 与 Chroma 向量库桥接
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GraphitiEngine:
    """
    Graphiti 时序 KG 引擎封装。

    使用 Graphiti 的双时态模型:
    - event_time: 事件发生时间 (来自传感器数据)
    - ingest_time: 数据摄入时间 (系统处理时间)
    """

    def __init__(self, neo4j_uri="neo4j://127.0.0.1:7687",
                 neo4j_user="neo4j", neo4j_password="12345678"):
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._graphiti = None
        self._fallback_store = None
        self._available = False

        try:
            from graphiti_core import Graphiti

            self._graphiti = Graphiti(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
            )
            self._available = True
            print("[Graphiti] 已连接 Neo4j (双时态模型就绪)")
        except Exception as e:
            print(f"[Graphiti] Graphiti 无法连接 Neo4j: {e}")
            print("[Graphiti] 回退到 TemporalStore")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def graphiti(self):
        """直接访问底层 Graphiti 实例"""
        if not self._available:
            raise RuntimeError("Graphiti 不可用")
        return self._graphiti

    @property
    def fallback_store(self):
        """当 Graphiti 不可用时的回退实现"""
        if self._fallback_store is None:
            from core.temporal.temporal_store import TemporalStore
            self._fallback_store = TemporalStore(
                uri=self._uri, user=self._user, password=self._password
            )
        return self._fallback_store

    # ── 实体操作 ────────────────────────────────────

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        description: str = "",
        event_time: Optional[datetime] = None,
    ):
        """
        添加时序实体 (带 event_time 时间戳)。

        Graphiti 会自动记录 ingest_time。
        """
        et = event_time or datetime.now(timezone.utc)
        if self._available:
            from graphiti_core.nodes import EntityNode
            entity = EntityNode(
                uuid=entity_id,
                name=name,
                labels=[entity_type],
                summary=description[:300],
                attributes={"type": entity_type, "source": "pump_tkg"},
                created_at=et,
            )
            self._graphiti.add_entity(entity)
        else:
            with self.fallback_store.driver.session(
                database=self.fallback_store._database
            ) as session:
                session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.type = $type, e.name = $name,
                        e.description = $description,
                        e.event_time = $event_time
                    """,
                    id=entity_id, type=entity_type, name=name,
                    description=description,
                    event_time=et.isoformat(),
                )

    def add_entities_batch(self, entities: List[Dict]):
        """批量导入实体"""
        for e in entities:
            ts = None
            if "measurement_attrs" in e:
                ts_str = e["measurement_attrs"].get("timestamp")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str)
            self.add_entity(
                entity_id=e["id"],
                entity_type=e["type"],
                name=e["name"],
                description=e.get("description", ""),
                event_time=ts,
            )

    # ── 关系操作 ────────────────────────────────────

    def add_relationship(
        self,
        source: str,
        target: str,
        relation: str,
        description: str = "",
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
    ):
        """添加时序关系 (支持有效时间区间)"""
        vf = valid_from or datetime.now(timezone.utc)
        if self._available:
            from graphiti_core.nodes import EpisodicNode
            episode = EpisodicNode(
                uuid=f"{source}-{relation}-{target}",
                name=relation,
                content=description,
                source=source,
                source_description=relation,
                valid_at=vf,
                invalid_at=valid_to,
            )
            try:
                self._graphiti.add_episodic(episode)
            except Exception:
                pass
        else:
            with self.fallback_store.driver.session(
                database=self.fallback_store._database
            ) as session:
                session.run(
                    """
                    MATCH (a:Entity {id: $source}), (b:Entity {id: $target})
                    MERGE (a)-[r:TEMPORAL {type: $relation}]->(b)
                    SET r.description = $description,
                        r.valid_from = $valid_from,
                        r.valid_to = $valid_to
                    """,
                    source=source, target=target, relation=relation,
                    description=description,
                    valid_from=vf.isoformat(),
                    valid_to=valid_to.isoformat() if valid_to else None,
                )

    def add_relationships_batch(self, edges: List[Dict]):
        """批量导入关系"""
        for r in edges:
            self.add_relationship(
                source=r["source"],
                target=r["target"],
                relation=r["relation"],
                description=r.get("description", ""),
            )

    # ── 查询操作 ────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        混合检索: 语义嵌入 + 关键词 + 图遍历 (Graphiti 原生支持)
        """
        results = []

        if self._available:
            try:
                # Graphiti 的 search 方法
                graphiti_results = self._graphiti.search(query, limit=top_k)
                for node in graphiti_results:
                    results.append({
                        "id": node.uuid if hasattr(node, "uuid") else str(node),
                        "name": node.name if hasattr(node, "name") else "",
                        "type": node.labels[0] if hasattr(node, "labels") and node.labels else "",
                        "description": node.summary if hasattr(node, "summary") else "",
                    })
            except Exception:
                pass

        # 回退到 Cypher 关键词搜索
        if not results:
            try:
                with self.fallback_store.driver.session(
                    database=self.fallback_store._database
                ) as session:
                    result = session.run(
                        """
                        MATCH (e:Entity)
                        WHERE e.name CONTAINS $query
                           OR e.description CONTAINS $query
                        RETURN e.id AS id, e.name AS name, e.type AS type,
                               e.description AS description
                        LIMIT $limit
                        """,
                        query=query, limit=top_k,
                    )
                    for record in result:
                        results.append({
                            "id": record["id"],
                            "name": record["name"],
                            "type": record["type"],
                            "description": record.get("description", ""),
                        })
            except Exception:
                pass

        return results

    def search_temporal(
        self, query: str, start_time: datetime, end_time: datetime
    ) -> List[Dict]:
        """
        时序范围搜索 — Graphiti 的 temporal_between 核心 API。

        这是 Phase 5 对比实验的关键差异化测试点。
        """
        if self._available:
            try:
                from graphiti_core.search import search
                results = search(
                    self._graphiti,
                    query,
                    start_time=start_time,
                    end_time=end_time,
                )
                return [{"id": r.uuid, "name": r.name, "content": r.content} for r in results]
            except Exception:
                pass

        # 回退
        return self.fallback_store.query_time_range(start_time, end_time)

    # ── 数据迁移 ────────────────────────────────────

    def migrate_from_json(
        self, entities_path: str = "data/output/entities.json",
        edges_path: str = "data/output/relationships.json",
    ):
        """
        从 entities.json + relationships.json 迁移到 Graphiti。

        为每个实体和边添加双时态时间戳。
        """
        import json

        with open(Path(PROJECT_ROOT) / entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
        with open(Path(PROJECT_ROOT) / edges_path, "r", encoding="utf-8") as f:
            edges = json.load(f)

        print(f"[Graphiti Migrate] {len(entities)} 实体, {len(edges)} 关系")

        if self._available:
            # 批量导入实体
            self.add_entities_batch(entities)
            print(f"[Graphiti Migrate] 实体导入: {len(entities)} 个")

            # 批量导入关系
            for i, r in enumerate(edges):
                self.add_relationship(
                    source=r["source"], target=r["target"],
                    relation=r["relation"], description=r.get("description", ""),
                )
                if (i + 1) % 100 == 0:
                    print(f"  关系: {i + 1}/{len(edges)}")
            print(f"[Graphiti Migrate] 完成: {len(entities)} 实体, {len(edges)} 关系")
        else:
            print("[Graphiti Migrate] Graphiti 不可用，跳过迁移")

    # ── 统计 ────────────────────────────────────────

    def count_entities(self) -> int:
        if self._available:
            try:
                # Graphiti 内部查询
                return len(self._graphiti.search("", limit=1000))
            except Exception:
                pass
        with self.fallback_store.driver.session(
            database=self.fallback_store._database
        ) as session:
            return session.run("MATCH (e:Entity) RETURN count(e)").single()[0]

    def count_temporal_edges(self) -> int:
        if self._available:
            return 0  # Graphiti 使用 EpisodicNode 非边
        with self.fallback_store.driver.session(
            database=self.fallback_store._database
        ) as session:
            return session.run("MATCH ()-[r:TEMPORAL]->() RETURN count(r)").single()[0]

    def close(self):
        if self._graphiti:
            try:
                self._graphiti.close()
            except Exception:
                pass
        if self._fallback_store:
            try:
                self._fallback_store.close()
            except Exception:
                pass
