"""
时序知识图谱 Neo4j 存储层

将时序四元组持久化到 Neo4j 图数据库，提供：
- 时间范围查询 (query_time_range)
- 时间点快照查询 (query_at_time)
- 时序链路径追踪 (trace_temporal_path)
- 时序冲突检测 (detect_temporal_conflicts)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.temporal.temporal_quad import TemporalQuad, TemporalRelation


# ── 默认连接配置 ────────────────────────────────────
DEFAULT_NEO4J_URI = "neo4j://127.0.0.1:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "12345678"
DEFAULT_DATABASE = "neo4j"


def _connect(uri=None, user=None, password=None):
    """建立 Neo4j 连接"""
    from neo4j import GraphDatabase, basic_auth
    driver = GraphDatabase.driver(
        uri or DEFAULT_NEO4J_URI,
        auth=basic_auth(user or DEFAULT_NEO4J_USER, password or DEFAULT_NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    return driver


class TemporalStore:
    """
    时序图谱 Neo4j 存储层。

    使用时序四元组模型 (h, r, t, [t₁, t₂])，在 Neo4j 中用带时间属性的关系表示。
    """

    def __init__(self, uri=None, user=None, password=None, database=None):
        self._uri = uri or DEFAULT_NEO4J_URI
        self._user = user or DEFAULT_NEO4J_USER
        self._password = password or DEFAULT_NEO4J_PASSWORD
        self._database = database or DEFAULT_DATABASE
        self._driver = None

    def connect(self):
        """建立 Neo4j 连接"""
        self._driver = _connect(self._uri, self._user, self._password)

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self.connect()
        return self._driver

    # ── 初始化 ──────────────────────────────────────

    def setup_indexes(self):
        """创建时序查询所需索引"""
        statements = [
            "CREATE INDEX temporal_from IF NOT EXISTS "
            "FOR ()-[r:TEMPORAL]->() ON (r.valid_from)",
            "CREATE INDEX temporal_type IF NOT EXISTS "
            "FOR ()-[r:TEMPORAL]->() ON (r.relation_type)",
        ]
        with self.driver.session(database=self._database) as session:
            for s in statements:
                try:
                    session.run(s)
                except Exception as e:
                    print(f"  [Temporal Index] {e}")

    # ── 写入 ────────────────────────────────────────

    def insert_quad(self, quad: TemporalQuad):
        """插入单条时序四元组"""
        props = quad.to_neo4j_props()
        with self.driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (h:Entity {id: $head})
                MATCH (t:Entity {id: $tail})
                MERGE (h)-[r:TEMPORAL {type: $rel_type}]->(t)
                SET r += $props
                """,
                head=quad.head_entity,
                tail=quad.tail_entity,
                rel_type=quad.relation.value,
                props=props,
            )

    def insert_batch(self, quads: List[TemporalQuad], batch_size: int = 200):
        """批量插入时序四元组"""
        total = len(quads)
        with self.driver.session(database=self._database) as session:
            for start in range(0, total, batch_size):
                batch = quads[start:start + batch_size]
                params = []
                for q in batch:
                    params.append({
                        "head": q.head_entity,
                        "tail": q.tail_entity,
                        "rel_type": q.relation.value,
                        "rel_enum": q.relation.name,
                        "valid_from": q.valid_from.isoformat(),
                        "valid_to": q.valid_to.isoformat() if q.valid_to else None,
                        "confidence": q.confidence,
                        "source": q.source,
                    })

                session.run(
                    """
                    UNWIND $batch AS props
                    MATCH (h:Entity {id: props.head})
                    MATCH (t:Entity {id: props.tail})
                    MERGE (h)-[r:TEMPORAL {type: props.rel_type}]->(t)
                    SET r.valid_from = props.valid_from,
                        r.valid_to = props.valid_to,
                        r.confidence = props.confidence,
                        r.source = props.source
                    """,
                    {"batch": params},
                )
                pct = min(100, (start + batch_size) * 100 // total)
                print(f"  时序四元组: {min(start + batch_size, total)}/{total} ({pct}%)")

    # ── 查询 ────────────────────────────────────────

    def query_at_time(self, timestamp: datetime) -> List[Dict]:
        """
        获取某时刻的图谱快照

        返回在该时刻活跃的所有时序关系。
        """
        ts_str = timestamp.isoformat()
        with self.driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (h:Entity)-[r:TEMPORAL]->(t:Entity)
                WHERE r.valid_from <= $ts
                  AND (r.valid_to IS NULL OR r.valid_to >= $ts)
                RETURN h.id AS head, r.type AS relation, t.id AS tail,
                       r.valid_from AS from_time, r.valid_to AS to_time,
                       r.confidence AS confidence
                ORDER BY r.valid_from DESC
                LIMIT 100
                """,
                ts=ts_str,
            )
            return [record.data() for record in result]

    def query_time_range(self, start: datetime, end: datetime) -> List[Dict]:
        """
        时间范围查询

        返回 valid_from 在 [start, end] 范围内的所有时序关系。
        """
        with self.driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (h:Entity)-[r:TEMPORAL]->(t:Entity)
                WHERE r.valid_from >= $start
                  AND r.valid_from <= $end
                RETURN h.id AS head, r.type AS relation, t.id AS tail,
                       r.valid_from AS from_time, r.valid_to AS to_time,
                       r.confidence AS confidence
                ORDER BY r.valid_from
                LIMIT 200
                """,
                start=start.isoformat(),
                end=end.isoformat(),
            )
            return [record.data() for record in result]

    def trace_temporal_path(
        self, entity_id: str, direction: str = "forward", max_depth: int = 5
    ) -> List[Dict]:
        """
        时序链路径追踪

        Args:
            entity_id: 起始实体 ID
            direction: 'forward' (沿 EVOLVES_TO/CAUSES/AFTER), 'backward' (沿 ORIGINATES_FROM/BEFORE)
            max_depth: 最大深度

        Returns:
            路径列表，每条路径是一组有序的时序关系
        """
        forward_rels = ["演化为", "导致", "后于"]
        backward_rels = ["源于", "先于"]

        rels = forward_rels if direction == "forward" else backward_rels
        # 构建关系列表字符串
        rel_list = "[" + ", ".join(f"'{r}'" for r in rels) + "]"

        with self.driver.session(database=self._database) as session:
            if direction == "forward":
                query = f"""
                    MATCH path = (start:Entity {{id: $entity_id}})
                                 -[r:TEMPORAL*1..{max_depth}]->
                                 (end:Entity)
                    WHERE all(rel IN relationships(path) WHERE rel.type IN {rel_list})
                    RETURN path
                    LIMIT 20
                """
            else:
                query = f"""
                    MATCH path = (start:Entity {{id: $entity_id}})
                                 <-[r:TEMPORAL*1..{max_depth}]-
                                 (end:Entity)
                    WHERE all(rel IN relationships(path) WHERE rel.type IN {rel_list})
                    RETURN path
                    LIMIT 20
                """

            result = session.run(query, entity_id=entity_id)
            paths = []
            for record in result:
                path = record["path"]
                steps = []
                for rel in path.relationships:
                    steps.append({
                        "source": rel.start_node["id"],
                        "target": rel.end_node["id"],
                        "relation": rel["type"],
                        "valid_from": rel.get("valid_from", ""),
                        "valid_to": rel.get("valid_to", ""),
                    })
                paths.append(steps)
            return paths

    # ── 冲突检测 ────────────────────────────────────

    def detect_temporal_conflicts(self, entity_id: str) -> List[Dict]:
        """
        检测时序冲突

        冲突类型:
        1. 同一实体在同一时刻被记录为不同状态
        2. 因果链中出现了时间倒序（effect 时间早于 cause）
        """
        conflicts = []

        with self.driver.session(database=self._database) as session:
            # 类型 2: 因果时间倒序
            result = session.run(
                """
                MATCH (cause:Entity)-[r_cause:TEMPORAL]->(effect:Entity)
                WHERE r_cause.type IN ['导致', '演化为']
                OPTIONAL MATCH (cause)-[r_other:TEMPORAL]->(effect)
                WHERE r_other.type IN ['导致', '演化为']
                  AND r_other.valid_from < r_cause.valid_from
                RETURN cause.id, effect.id, r_cause.valid_from AS from1,
                       r_other.valid_from AS from2
                """
            )
            for record in result:
                conflicts.append({
                    "type": "temporal_reversal",
                    "entity1": record["cause.id"],
                    "entity2": record["effect.id"],
                    "detail": f"因果关系时间倒序: {record['from1']} vs {record['from2']}",
                })

        return conflicts

    # ── 统计 ────────────────────────────────────────

    def count_temporal_relations(self) -> Dict:
        """统计时序关系统计"""
        with self.driver.session(database=self._database) as session:
            total = session.run("MATCH ()-[r:TEMPORAL]->() RETURN count(r)").single()[0]
            by_type = session.run(
                "MATCH ()-[r:TEMPORAL]->() RETURN r.type, count(r) ORDER BY count(r) DESC"
            ).data()
        return {
            "total_temporal_relations": total,
            "by_type": {r["r.type"]: r["count(r)"] for r in by_type},
        }
