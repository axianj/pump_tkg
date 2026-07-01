"""
混合检索模块 — Graphiti + Chroma + RRF 融合

为 Phase 5 对比实验提供统一的检索接口。

方法:
1. 并行查询 Graphiti (图遍历 + 时序) 和 Chroma (向量相似度)
2. 使用 RRF (Reciprocal Rank Fusion) 合并去重
3. 返回融合后的 Top-K 结果
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def reciprocal_rank_fusion(
    results_list: List[List[Dict]],
    k: int = 60,
    top_n: int = 10,
) -> List[Dict]:
    """
    RRF 融合多个检索结果。

    Args:
        results_list: 多个检索源的结果列表，每项包含 {"id": str, "score": float, ...}
        k: RRF 常数（典型值 60）
        top_n: 返回前 N 个

    Returns:
        融合后的排序结果
    """
    scores = {}
    for results in results_list:
        for rank, item in enumerate(results, start=1):
            item_id = item.get("id", item.get("measurement_id", str(rank)))
            rrf_score = scores.get(item_id, 0) + 1.0 / (k + rank)
            scores[item_id] = rrf_score
            if "_meta" not in scores:
                scores["_meta"] = {}
            scores["_meta"][item_id] = item

    # 去掉内部元数据键
    if "_meta" in scores:
        meta = scores.pop("_meta")

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for item_id, rrf_score in sorted_items:
        item = meta.get(item_id, {})
        item["rrf_score"] = rrf_score
        results.append(item)

    return results


class HybridRetriever:
    """
    混合检索引擎。

    同时查询多个后端，用 RRF 融合结果。
    """

    def __init__(
        self,
        vector_store=None,
        temporal_store=None,
        graphiti_engine=None,
    ):
        self.vector_store = vector_store
        self.temporal_store = temporal_store
        self.graphiti_engine = graphiti_engine

    def search(
        self,
        query: str,
        top_k: int = 10,
        time_range: Optional[tuple] = None,
    ) -> List[Dict]:
        """
        混合检索主入口。

        Args:
            query: 查询文本
            top_k: 返回 Top-K
            time_range: 可选 (start, end) 时间范围

        Returns:
            RRF 融合后的 Top-K 结果
        """
        all_results = []

        # 1. Chroma 向量检索
        if self.vector_store is not None:
            try:
                vr = self.vector_store.search_documents(query, top_k=top_k)
                all_results.append([
                    {"id": r["id"], "score": 1.0 - r.get("distance", 0), "source": "vector", "text": r.get("text", "")}
                    for r in vr
                ])
            except Exception:
                all_results.append([])

        # 2. Graphiti / TemporalStore 图检索
        if self.graphiti_engine is not None:
            try:
                gr = self.graphiti_engine.search(query, top_k=top_k)
                all_results.append([
                    {"id": r["id"], "score": 0.9, "source": "graphiti",
                     "name": r.get("name", ""), "type": r.get("type", ""),
                     "description": r.get("description", "")}
                    for r in gr
                ])
            except Exception:
                all_results.append([])

        # 3. 时序路径追踪（时间范围查询）
        if time_range and self.temporal_store is not None:
            try:
                from datetime import datetime
                start, end = time_range
                tr = self.temporal_store.query_time_range(start, end)
                all_results.append([
                    {"id": r.get("head", ""), "score": 0.8, "source": "temporal",
                     "relation": r.get("relation", ""), "target": r.get("tail", ""),
                     "from_time": r.get("from_time", ""), "to_time": r.get("to_time", "")}
                    for r in tr
                ][:top_k])
            except Exception:
                all_results.append([])

        # RRF 融合
        return reciprocal_rank_fusion(all_results, top_n=top_k)


# ── 便捷 API ────────────────────────────────────────

def create_hybrid_retriever(
    vector_store=None,
    temporal_store=None,
    graphiti_engine=None,
) -> HybridRetriever:
    """工厂函数：创建混合检索引擎"""
    return HybridRetriever(
        vector_store=vector_store,
        temporal_store=temporal_store,
        graphiti_engine=graphiti_engine,
    )
