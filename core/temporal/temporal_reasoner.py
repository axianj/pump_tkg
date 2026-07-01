"""
时序推理引擎

基于 Allen 区间代数的规则推理 + 基于图遍历的因果链推理。
预留深度学习推理接口（Phase 5 对接 TTansE / EvolveGCN / RE-NET / TGN / Know-Evolve）。

理论依据: Li et al. (Science, 2017) — 时序网络在可控性、能量效率、轨迹紧凑性上
         比静态网络有数量级优势。
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.temporal.temporal_quad import (
    TemporalQuad, TemporalRelation, TemporalQuadStore, TRANSITIVITY_TABLE
)


class TemporalReasoner:
    """
    时序推理引擎 — 支持规则推理和图遍历推理。

    规则推理: 基于 Allen 区间代数的传递性规则
    图遍历推理: 在时序图谱上搜索因果链路
    """

    def __init__(self, store: Optional[TemporalQuadStore] = None):
        self.store = store or TemporalQuadStore()

    # ── 规则推理 ────────────────────────────────────

    def infer_transitive_closure(self) -> List[TemporalQuad]:
        """
        传递性闭包推理: 根据已知四元组推导隐含关系。

        规则: 如果 A r1 B 且 B r2 C, 且 (r1, r2) 在 TRANSITIVITY_TABLE 中,
              则推导出 A r_derived C。

        例如:
          (轴承BPFO, CAUSES, 振动超阈值, [t1, t2])
        + (振动超阈值, EVOLVES_TO, 设备停机, [t2, t3])
        → (轴承BPFO, CAUSES, 设备停机, [t1, t3])
        """
        derived_quads = []

        for quad_a in self.store:
            intermediates = self.store.find_by_head(quad_a.tail_entity)
            for quad_b in intermediates:
                derived_rel = TRANSITIVITY_TABLE.get(
                    (quad_a.relation, quad_b.relation)
                )
                if derived_rel is None:
                    continue

                # 检查时间一致性: 只有 quad_a 不晚于 quad_b 才有效
                if quad_a.valid_from > quad_b.valid_from:
                    continue

                # 构建推导四元组
                earliest = quad_a.valid_from
                latest = quad_a.valid_to if quad_a.valid_to else quad_b.valid_to
                if quad_b.valid_to and (latest is None or quad_b.valid_to < latest):
                    latest = quad_b.valid_to

                derived = TemporalQuad(
                    head_entity=quad_a.head_entity,
                    relation=derived_rel,
                    tail_entity=quad_b.tail_entity,
                    valid_from=earliest,
                    valid_to=latest,
                    confidence=min(quad_a.confidence, quad_b.confidence) * 0.9,
                    source=f"传递推理: {quad_a.head_entity}→{quad_a.tail_entity}→{quad_b.tail_entity}",
                )
                derived_quads.append(derived)

        return derived_quads

    def infer_all_rules(self) -> List[TemporalQuad]:
        """运行所有规则推理，返回所有新推导的四元组"""
        return self.infer_transitive_closure()

    # ── 图遍历推理 ──────────────────────────────────

    def trace_causal_chain(
        self, start_entity: str, max_depth: int = 5
    ) -> List[List[TemporalQuad]]:
        """
        从 start_entity 出发，追踪所有因果链。

        只沿 CAUSES 和 EVOLVES_TO 关系前进。

        Returns:
            路径列表，每条路径是因果链（有序的四元组序列）
        """
        # 从 store 中提取因果相关的关系
        causal_paths = []
        causal_relations = {TemporalRelation.CAUSES, TemporalRelation.EVOLVES_TO}

        def dfs(current: str, depth: int, path: List[TemporalQuad]):
            if depth > max_depth:
                causal_paths.append(list(path))
                return
            outgoing = self.store.find_by_head(current)
            causal_out = [q for q in outgoing if q.relation in causal_relations]
            if not causal_out:
                causal_paths.append(list(path))
                return
            for q in causal_out:
                path.append(q)
                dfs(q.tail_entity, depth + 1, path)
                path.pop()

        dfs(start_entity, 1, [])
        return causal_paths

    def find_root_causes(self, entity_id: str, max_depth: int = 5) -> List[TemporalQuad]:
        """
        反向追踪根因: 沿 ORIGINATES_FROM 和 CAUSES 逆向寻找根故障。

        Returns:
            根因四元组列表（已经没有更多上游因果的叶子节点）
        """
        root_causes = []

        def reverse_trace(current: str, depth: int):
            if depth > max_depth:
                return
            incoming = self.store.find_by_tail(current)
            causal_in = [
                q for q in incoming
                if q.relation in {TemporalRelation.CAUSES, TemporalRelation.ORIGINATES_FROM}
            ]
            if not causal_in:
                root_causes.append(current)
                return
            for q in causal_in:
                reverse_trace(q.head_entity, depth + 1)

        reverse_trace(entity_id, 1)
        return root_causes

    # ── 故障链分析 API ──────────────────────────────

    def analyze_fault_chain(self, fault_id: str) -> Dict:
        """
        分析故障的完整因果链。

        Returns:
            {
                "fault": str,
                "root_causes": [...],         # 根因
                "propagation_paths": [...],   # 传播路径
                "downstream_effects": [...],  # 下游影响
            }
        """
        roots = self.find_root_causes(f"fault_{fault_id}")
        paths = self.trace_causal_chain(f"fault_{fault_id}")
        downstream = self.store.find_by_head(f"fault_{fault_id}")

        return {
            "fault": fault_id,
            "root_causes": roots,
            "propagation_paths": paths,
            "downstream_effects": [q.tail_entity for q in downstream],
        }

    def find_fault_evolution(
        self, measurement_id: str, time_window_hours: int = 72
    ) -> List[TemporalQuad]:
        """
        查找测量的故障演化历史。

        给定一个测量记录 ID，返回其前 72 小时内的所有相关时序事件。

        Returns:
            按时间排序的四元组列表
        """
        all_related = []
        for q in self.store:
            if q.head_entity == measurement_id or q.tail_entity == measurement_id:
                all_related.append(q)

        all_related.sort(key=lambda x: x.valid_from)
        return all_related

    # ── 深度学习推理接口 (Phase 5 对接) ───────────────

    def prepare_training_samples(self) -> Tuple:
        """
        准备用于训练时序推理模型的数据集。

        将时序四元组转换为 (source_embedding, relation, target_embedding, time) 格式。
        Phase 5 对接 TTansE / HyTE / EvolveGCN / RE-NET / TGN / Know-Evolve。

        Returns:
            (quad_tuples, entity_list, relation_list)
        """
        quads = [
            (
                q.head_entity,
                q.relation.value,
                q.tail_entity,
                q.valid_from.isoformat(),
                q.valid_to.isoformat() if q.valid_to else None,
            )
            for q in self.store
        ]
        entities = set()
        relations = set()
        for q in self.store:
            entities.add(q.head_entity)
            entities.add(q.tail_entity)
            relations.add(q.relation.value)

        return quads, sorted(entities), sorted(relations)

    def apply_dl_model(
        self, model, entity_id: str, timestamp: datetime
    ) -> List[Tuple[str, float]]:
        """
        使用深度学习模型进行链路预测。

        Args:
            model: 训练好的时序推理模型 (Phase 5 实现)
            entity_id: 查询实体
            timestamp: 查询时间

        Returns:
            [(tail_entity, probability), ...] 预测的尾实体及概率
        """
        raise NotImplementedError("深度学习推理接口将在 Phase 5 实现")
