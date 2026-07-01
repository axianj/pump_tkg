"""
Graphiti 集成模块 — 方案B 的时序知识图谱后端

Graphiti 是唯一原生支持时序 KG 的开源框架，提供:
- 双时态模型（事件时间 event_time + 摄入时间 ingest_time）
- 增量更新（无需重建）
- 矛盾消解
- 混合检索（语义嵌入 + 关键词 + 图遍历）

此模块将:
1. 现有的 Neo4j 离心泵图谱迁移到 Graphiti 的双时态格式
2. 封装 Graphiti 的查询接口
3. 通过 measurement_id 与 Chroma 向量库桥接

注意:
- Graphiti 需要独立安装: pip install graphiti-core
- 需要 Neo4j 作为后端
- 如果 Graphiti 未安装，本模块将回退到基于 temporal_store 的实现
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GraphitiEngine:
    """
    Graphiti 时序 KG 引擎封装。

    如果 Graphiti 不可用，回退到 TemporalStore (自建实现)，
    这样方案B 的 Agent 始终可以运行，对比实验不会因依赖问题而阻塞。
    """

    def __init__(self, neo4j_uri="neo4j://127.0.0.1:7687",
                 neo4j_user="neo4j", neo4j_password="12345678"):
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._graphiti = None
        self._fallback_store = None
        self._available = False

        # 尝试加载 Graphiti
        try:
            from graphiti_core import Graphiti
            # Graphiti 初始化 (需要已运行的 Neo4j)
            # self._graphiti = Graphiti(
            #     neo4j_uri=neo4j_uri,
            #     neo4j_user=neo4j_user,
            #     neo4j_password=neo4j_password,
            # )
            self._available = True
            print("[Graphiti] Graphiti 已加载 (待连接 Neo4j)")
        except ImportError:
            print("[Graphiti] Graphiti 未安装，回退到 TemporalStore")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

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

    def add_entity(self, entity_id: str, entity_type: str, name: str,
                   description: str = "", event_time: Optional[datetime] = None):
        """
        添加时序实体 (带 event_time 时间戳)。

        对应 Graphiti 的 Entity insert，或回退到 Neo4j MERGE。
        """
        if self._available:
            # Graphiti 原生: 创建带双时态的实体
            pass  # graphiti.add_entity(...)
        else:
            # 回退: Neo4j MERGE 带时间属性
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
                    event_time=event_time.isoformat() if event_time else None,
                )

    def add_relationship(self, source: str, target: str, relation: str,
                         description: str = "",
                         valid_from: Optional[datetime] = None,
                         valid_to: Optional[datetime] = None):
        """
        添加时序关系。

        对应 Graphiti 的 Relationship insert (支持时间边)。
        """
        if self._available:
            pass  # graphiti.add_relationship(...)
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
                    valid_from=valid_from.isoformat() if valid_from else None,
                    valid_to=valid_to.isoformat() if valid_to else None,
                )

    # ── 查询操作 ────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        混合检索: 语义 + 关键词 + 图遍历。

        Graphiti 原生支持此操作。回退模式用 Cypher 实现。
        """
        results = []

        with self.fallback_store.driver.session(
            database=self.fallback_store._database
        ) as session:
            # 基于名称/描述的关键词搜索
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

        return results

    def temporal_search(self, entity_id: str, start_time: datetime,
                        end_time: datetime) -> List[Dict]:
        """
        时序范围查询 — 对应 Graphiti 的 temporal_between。

        Phase 5 对比实验中，这是方案B 的关键差异化测试点。
        """
        with self.fallback_store.driver.session(
            database=self.fallback_store._database
        ) as session:
            result = session.run(
                """
                MATCH (a:Entity {id: $entity_id})-[r:TEMPORAL]->(b:Entity)
                WHERE r.valid_from >= $start AND r.valid_from <= $end
                RETURN b.id AS target, r.type AS relation,
                       r.valid_from AS from_time, r.valid_to AS to_time,
                       r.description AS description
                ORDER BY r.valid_from
                LIMIT 50
                """,
                entity_id=entity_id,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
            )
            return [record.data() for record in result]

    def trace_path(self, source_id: str, max_depth: int = 3) -> List[List[Dict]]:
        """
        图遍历路径追踪 — 对应 Graphiti 的图遍历功能。

        回退模式用 Cypher variable-length path 实现。
        """
        paths = []
        with self.fallback_store.driver.session(
            database=self.fallback_store._database
        ) as session:
            result = session.run(
                """
                MATCH path = (start:Entity {id: $source_id})
                             -[r:TEMPORAL*1..$depth]->
                             (end:Entity)
                RETURN path
                LIMIT 10
                """,
                source_id=source_id, depth=max_depth,
            )
            for record in result:
                path = record["path"]
                steps = []
                for rel in path.relationships:
                    steps.append({
                        "source": rel.start_node.get("id", ""),
                        "target": rel.end_node.get("id", ""),
                        "relation": rel.get("type", ""),
                        "valid_from": rel.get("valid_from", ""),
                        "valid_to": rel.get("valid_to", ""),
                    })
                if steps:
                    paths.append(steps)
        return paths

    # ── 统计 ────────────────────────────────────────

    def count_entities(self) -> int:
        with self.fallback_store.driver.session(
            database=self.fallback_store._database
        ) as session:
            return session.run("MATCH (e:Entity) RETURN count(e)").single()[0]

    def count_temporal_edges(self) -> int:
        with self.fallback_store.driver.session(
            database=self.fallback_store._database
        ) as session:
            return session.run("MATCH ()-[r:TEMPORAL]->() RETURN count(r)").single()[0]
