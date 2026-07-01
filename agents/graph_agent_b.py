"""
Langgraph Agent — 方案B (Graphiti + 混合检索 + TKG Core)

与方案A 的 Agent 保持相同的状态定义和路由逻辑，
差异仅在底层检索后端:
- 方案A: TemporalQuadStore (自建) + LightRAG
- 方案B: GraphitiEngine (原生时序) + Chroma

Phase 5 对比实验中，两个 Agent 使用相同的测试集和评估指标，
确保差异仅来自底层框架。
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import TypedDict, List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 与方案A 共享状态定义和路由逻辑
from agents.graph_agent_a import (
    AgentState, SharedRetrievalComponents,
    _classify_query, vector_retrieve, multimodal_retrieve, diagnose,
)


# ── 方案B 专用检索 ──────────────────────────────────

def temporal_retrieve_b(state: AgentState, components: SharedRetrievalComponents) -> AgentState:
    """
    方案B 时序检索 — 使用 GraphitiEngine 的 temporal_search 和 trace_path。

    与方案A 的 temporal_retrieve 相比:
    - 方案A: 用 TemporalReasoner + TemporalQuadStore (内存/Neo4j Cypher)
    - 方案B: 用 GraphitiEngine (原生时序 API, 支持 temporal_between)
    """
    results = []

    # 解析查询中的时间范围 (如 "过去24小时", "最近3天")
    import re
    from datetime import timedelta
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=72)  # 默认72小时

    time_patterns = [
        (r"过去\s*(\d+)\s*小时", "hours"),
        (r"最近\s*(\d+)\s*天", "days"),
        (r"过去\s*(\d+)\s*天", "days"),
    ]
    for pattern, unit in time_patterns:
        match = re.search(pattern, state["query"])
        if match:
            value = int(match.group(1))
            if unit == "hours":
                start_time = end_time - timedelta(hours=value)
            elif unit == "days":
                start_time = end_time - timedelta(days=value)
            break

    # 使用 GraphitiEngine (或回退 TemporalStore) 检索
    if components.temporal_store is not None:
        # 尝试提取查询中的故障名
        from core.pump_domain import FAULT_NAMES_ZH
        for code, zh in FAULT_NAMES_ZH.items():
            if code == "Healthy":
                continue
            if zh in state["query"] or code.lower().replace("_", "") in state["query"].lower().replace(" ", ""):
                entity_id = f"fault_{code}"
                # 时序范围查询 (Graphiti 原生或回退)
                temporal_results = (components.temporal_store.query_time_range(start_time, end_time) if hasattr(components.temporal_store, "query_time_range") else [])
                for r in temporal_results[:10]:
                    if r.get("head") == entity_id or r.get("tail") == entity_id:
                        results.append({
                            "type": "temporal_edge",
                            "source": r.get("head"),
                            "target": r.get("tail"),
                            "relation": r.get("relation"),
                            "from_time": r.get("from_time"),
                            "to_time": r.get("to_time"),
                        })

                # 路径追踪
                paths = (components.temporal_store.trace_temporal_path(entity_id, direction="forward", max_depth=3) if hasattr(components.temporal_store, "trace_temporal_path") else [])
                for path in paths[:3]:
                    results.append({
                        "type": "temporal_path",
                        "steps": path,
                    })

    # 叠加向量检索补充上下文
    if components.vector_store is not None:
        vr = components.vector_store.search_documents(state["query"], top_k=3)
        for r in vr:
            results.append({"type": "document", "content": r.get("text", "")[:500]})

    state["retrieval_results"] = results
    return state


# ── 构建 Graph B ────────────────────────────────────

def build_graph_b(components: SharedRetrievalComponents) -> StateGraph:
    """
    构建方案B 的 Langgraph Agent。

    路由逻辑与方案A 相同，但 temporal_retrieve 节点使用方案B 实现。
    """
    graph = StateGraph(AgentState)

    def router(state: AgentState) -> AgentState:
        state["route"] = _classify_query(state["query"])
        return state

    graph.add_node("router", router)
    graph.add_node("vector_retrieve", lambda s: vector_retrieve(s, components))
    graph.add_node("temporal_retrieve", lambda s: temporal_retrieve_b(s, components))
    graph.add_node("multimodal_retrieve", lambda s: multimodal_retrieve(s, components))
    graph.add_node("diagnose", lambda s: diagnose(s, components))

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda s: {
            "simple_fact": "vector_retrieve",
            "temporal": "temporal_retrieve",
            "multimodal": "multimodal_retrieve",
            "diagnosis": "temporal_retrieve",
        }[s["route"]],
        {
            "vector_retrieve": "vector_retrieve",
            "temporal_retrieve": "temporal_retrieve",
            "multimodal_retrieve": "multimodal_retrieve",
        },
    )

    for node in ["vector_retrieve", "temporal_retrieve", "multimodal_retrieve"]:
        graph.add_edge(node, "diagnose")
    graph.add_edge("diagnose", END)

    return graph


def create_agent_b(components: SharedRetrievalComponents):
    """创建方案B Agent (已编译)"""
    graph = build_graph_b(components)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ── 便捷 API ────────────────────────────────────────

class AgenticDiagnosisEngineB:
    """
    方案B 的 Agentic 故障诊断引擎。

    用法与方案A 的 AgenticDiagnosisEngine 完全一致，
    只是底层的检索后端不同。
    """

    def __init__(self, components: SharedRetrievalComponents):
        self.components = components
        self.agent = create_agent_b(components)

    def query(self, question: str, thread_id: str = "default") -> Dict:
        state = self.agent.invoke(
            {"query": question, "route": "", "retrieval_results": [],
             "diagnosis": "", "confidence": 0.0, "sources": []},
            {"configurable": {"thread_id": thread_id}},
        )
        return {
            "query": question,
            "route": state["route"],
            "diagnosis": state["diagnosis"],
            "confidence": state["confidence"],
            "sources": state["sources"],
        }
