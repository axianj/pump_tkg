"""
时序四元组数据模型

基于 Allen 区间代数和 Li et al. (Science, 2017) 的时序网络理论，
将时间从属性提升为关系维度。

时序四元组: (头实体, 时序关系, 尾实体, [valid_from, valid_to])
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict


class TemporalRelation(Enum):
    """时序关系类型 — 区分类似性、因果性和状态变化"""
    # 时序顺序
    BEFORE = "先于"
    AFTER = "后于"
    # 因果关系
    CAUSES = "导致"
    ORIGINATES_FROM = "源于"
    # 并发
    DURING = "期间并发"
    # 状态变化
    TERMINATES = "终止"
    EVOLVES_TO = "演化为"
    # 测量标注
    MEASURED_AT = "测量于"


# 关系传递性规则表 — 基于 Allen 区间代数
# (R1, R2) → R3 表示: 如果 A R1 B 且 B R2 C, 则 A R3 C
TRANSITIVITY_TABLE = {
    (TemporalRelation.BEFORE, TemporalRelation.BEFORE): TemporalRelation.BEFORE,
    (TemporalRelation.AFTER, TemporalRelation.AFTER): TemporalRelation.AFTER,
    (TemporalRelation.CAUSES, TemporalRelation.CAUSES): TemporalRelation.CAUSES,
    (TemporalRelation.CAUSES, TemporalRelation.BEFORE): TemporalRelation.BEFORE,
    (TemporalRelation.BEFORE, TemporalRelation.CAUSES): TemporalRelation.BEFORE,
    (TemporalRelation.EVOLVES_TO, TemporalRelation.EVOLVES_TO): TemporalRelation.EVOLVES_TO,
    (TemporalRelation.EVOLVES_TO, TemporalRelation.BEFORE): TemporalRelation.BEFORE,
    (TemporalRelation.BEFORE, TemporalRelation.EVOLVES_TO): TemporalRelation.BEFORE,
    # CAUSES + EVOLVES_TO → CAUSES（因果链向前推导）
    (TemporalRelation.CAUSES, TemporalRelation.EVOLVES_TO): TemporalRelation.CAUSES,
    # EVOLVES_TO + CAUSES → CAUSES（演化到达的状态触发新的因果）
    (TemporalRelation.EVOLVES_TO, TemporalRelation.CAUSES): TemporalRelation.CAUSES,
}


@dataclass
class TemporalQuad:
    """
    时序四元组: (头实体, 时序关系, 尾实体, 时间区间)

    Attributes:
        head_entity: 头实体 ID
        relation: 时序关系类型
        tail_entity: 尾实体 ID
        valid_from: 关系生效时间
        valid_to: 关系失效时间 (None = 持续至今)
        confidence: 置信度 [0, 1]
        source: 来源标注
    """
    head_entity: str
    relation: TemporalRelation
    tail_entity: str
    valid_from: datetime
    valid_to: Optional[datetime] = None
    confidence: float = 1.0
    source: str = ""

    def is_active_at(self, timestamp: datetime) -> bool:
        """判断该四元组在给定时间是否有效"""
        after_start = self.valid_from <= timestamp
        if self.valid_to is None:
            return after_start
        return after_start and timestamp <= self.valid_to

    def duration_seconds(self) -> Optional[float]:
        """返回时间区间的持续时间（秒）"""
        if self.valid_to is None:
            return None
        return (self.valid_to - self.valid_from).total_seconds()

    def overlaps(self, other: "TemporalQuad") -> bool:
        """判断与另一个四元组的时间区间是否重叠"""
        s1, e1 = self.valid_from, self.valid_to or datetime.max
        s2, e2 = other.valid_from, other.valid_to or datetime.max
        return s1 <= e2 and s2 <= e1

    def to_tuple(self) -> tuple:
        """转换为 (h, r, t, from, to)"""
        return (
            self.head_entity,
            self.relation.value,
            self.tail_entity,
            self.valid_from.isoformat(),
            self.valid_to.isoformat() if self.valid_to else None,
        )

    def to_neo4j_props(self) -> Dict:
        """转换为 Neo4j 关系属性"""
        return {
            "relation_type": self.relation.value,
            "relation_enum": self.relation.name,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "confidence": self.confidence,
            "source": self.source,
        }

    @staticmethod
    def infer_transitive(r1: TemporalRelation, r2: TemporalRelation) -> Optional[TemporalRelation]:
        """
        传递性推理: 如果 A r1 B 且 B r2 C, 返回 A ? C 的关系类型

        基于 Allen 区间代数的传递性闭包规则。
        返回 None 表示两个关系在此规则下没有传递性。
        """
        return TRANSITIVITY_TABLE.get((r1, r2))


class TemporalQuadStore:
    """时序四元组的内存存储（轻量实现，后续对接 temporal_store.py）"""

    def __init__(self):
        self._quads: List[TemporalQuad] = []
        self._index_by_head: Dict[str, List[int]] = {}
        self._index_by_tail: Dict[str, List[int]] = {}
        self._index_by_time: List[int] = []  # 按 valid_from 排序的索引

    def add(self, quad: TemporalQuad):
        idx = len(self._quads)
        self._quads.append(quad)
        self._index_by_head.setdefault(quad.head_entity, []).append(idx)
        self._index_by_tail.setdefault(quad.tail_entity, []).append(idx)
        self._index_by_time.append(idx)
        self._index_by_time.sort(key=lambda i: self._quads[i].valid_from)

    def __len__(self):
        return len(self._quads)

    def __iter__(self):
        return iter(self._quads)

    def find_by_head(self, entity_id: str) -> List[TemporalQuad]:
        return [self._quads[i] for i in self._index_by_head.get(entity_id, [])]

    def find_by_tail(self, entity_id: str) -> List[TemporalQuad]:
        return [self._quads[i] for i in self._index_by_tail.get(entity_id, [])]

    def find_active_at(self, timestamp: datetime) -> List[TemporalQuad]:
        return [q for q in self._quads if q.is_active_at(timestamp)]

    def find_in_range(self, start: datetime, end: datetime) -> List[TemporalQuad]:
        return [q for q in self._quads
                if q.valid_from <= end and (q.valid_to is None or q.valid_to >= start)]

    def trace_forward(self, entity_id: str, max_depth: int = 5) -> List[List[TemporalQuad]]:
        """
        从 entity_id 向前追踪时序链（BFS 遍历）

        Returns: 路径列表，每条路径是一个四元组序列
        """
        paths = []
        queue = [[q] for q in self.find_by_head(entity_id)]
        while queue:
            path = queue.pop(0)
            if len(path) > max_depth:
                paths.append(path)
                continue
            last = path[-1]
            next_quads = self.find_by_head(last.tail_entity)
            if not next_quads:
                paths.append(path)
            else:
                for q in next_quads:
                    queue.append(path + [q])
        return paths

    def to_neo4j_cypher(self) -> List[str]:
        """将四元组转换为 Neo4j MERGE 语句列表"""
        statements = []
        for quad in self._quads:
            props = quad.to_neo4j_props()
            props_str = ", ".join(
                f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}"
                for k, v in props.items() if v is not None
            )
            stmt = (
                f"MATCH (h:Entity {{id: '{quad.head_entity}'}}), "
                f"(t:Entity {{id: '{quad.tail_entity}'}}) "
                f"MERGE (h)-[r:TEMPORAL {{type: '{quad.relation.value}'}}]->(t) "
                f"SET r += {{{props_str}}}"
            )
            statements.append(stmt)
        return statements
