"""
Langgraph Agent — 方案A (LightRAG + 混合检索 + TKG Core)

Agent 架构:
    Router Agent → 查询分类 →
        ├── vector_retrieve (简单事实 → Chroma)
        ├── temporal_retrieve (时序推理 → TKG Core)
        ├── multimodal_retrieve (多模态文档 → RAG-Anything)
        └── diagnose (综合诊断 → Ollama LLM)

状态定义 (shared):
    query: 用户原始问题
    route: 路由类型 (simple_fact / temporal / multimodal / diagnosis)
    retrieval_results: 检索到的上下文
    diagnosis_result: 最终诊断结论
    confidence: 置信度
"""

import sys
from pathlib import Path
from typing import TypedDict, List, Dict, Optional, Literal, Annotated
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ── Agent 状态 ──────────────────────────────────────


class AgentState(TypedDict):
    query: str
    route: str  # "simple_fact" | "temporal" | "multimodal" | "diagnosis"
    retrieval_results: List[Dict]
    diagnosis: str
    confidence: float
    sources: List[str]


# ── 共享组件 ────────────────────────────────────────

class SharedRetrievalComponents:
    """
    Agent 依赖的外部组件（方案A专用）。
    Phase 4 的方案B将使用相同的 State 定义和路由逻辑，
    但底层 TemporalRetrieval 和向量检索的后端不同。
    """

    def __init__(
        self,
        vector_store=None,
        temporal_store=None,
        temporal_reasoner=None,
        lightrag_engine=None,
        multimodal_engine=None,
        llm_model: str = "qwen3:14b",
    ):
        self.vector_store = vector_store
        self.temporal_store = temporal_store
        self.temporal_reasoner = temporal_reasoner
        self.lightrag_engine = lightrag_engine
        self.multimodal_engine = multimodal_engine
        self.llm_model = llm_model


# ── Router ──────────────────────────────────────────

def _classify_query(query: str) -> str:
    """
    基于关键词的路由分类。

    Phase 3 实现中，用简单规则做路由。
    后续可通过 Ollama LLM 的 function calling 增强。

    Returns:
        "temporal" | "multimodal" | "simple_fact" | "diagnosis"
    """
    query_lower = query.lower()

    # 时序关键词
    temporal_kw = [
        "趋势", "历史", "之前", "之后", "时间", "变化", "退化", "恶化",
        "过去", "最近", "昨天", "小时", "周", "月", "年",
        "trend", "history", "before", "after", "degradation", "evolution",
    ]
    # 多模态关键词
    multimodal_kw = [
        "图片", "图表", "图", "表格", "公式", "手册", "规格", "型号",
        "datasheet", "diagram", "figure", "table",
    ]
    # 故障诊断关键词
    diagnosis_kw = [
        "诊断", "故障", "原因", "为什么", "怎么办", "修复", "维修", "处理",
        "解决", "根因", "来源", "引起",
        "diagnose", "fault", "cause", "repair", "root cause",
    ]

    temporal_score = sum(1 for kw in temporal_kw if kw in query_lower)
    multimodal_score = sum(1 for kw in multimodal_kw if kw in query_lower)
    diagnosis_score = sum(1 for kw in diagnosis_kw if kw in query_lower)

    # 诊断+时序 → 时序推理（最重要的是时序因果链诊断）
    if temporal_score > diagnosis_score and temporal_score > multimodal_score:
        return "temporal"
    if diagnosis_score > 0:
        return "diagnosis"
    if multimodal_score > temporal_score and multimodal_score > diagnosis_score:
        return "multimodal"
    if temporal_score > 0:
        return "temporal"

    return "simple_fact"


# ── 检索节点 ────────────────────────────────────────

def vector_retrieve(state: AgentState, components: SharedRetrievalComponents) -> AgentState:
    """简单事实查询 → Chroma 向量检索"""
    if components.vector_store is None:
        state["retrieval_results"] = []
        return state

    results = components.vector_store.search_documents(state["query"], top_k=5)
    state["retrieval_results"] = results
    return state


def temporal_retrieve(state: AgentState, components: SharedRetrievalComponents) -> AgentState:
    """
    时序推理查询 → TKG Core 时序路径追踪

    在 Neo4j 中搜索时序链，获取故障演化路径。
    """
    results = []

    # 从图谱中找时序路径
    if components.temporal_reasoner is not None:
        # 尝试提取故障名称做检索
        from core.pump_domain import FAULT_NAMES_ZH
        for code, zh in FAULT_NAMES_ZH.items():
            if code == "Healthy":
                continue
            if zh in state["query"] or code.lower().replace("_", "") in state["query"].lower().replace(" ", ""):
                analysis = components.temporal_reasoner.analyze_fault_chain(code)
                if analysis["propagation_paths"]:
                    for path in analysis["propagation_paths"]:
                        results.append({
                            "type": "temporal_chain",
                            "fault": code,
                            "root_causes": analysis["root_causes"],
                            "propagation_path": [
                                {"source": q.head_entity, "target": q.tail_entity,
                                 "relation": q.relation.value}
                                for q in path
                            ],
                        })
                else:
                    results.append({
                        "type": "temporal_info",
                        "fault": code,
                        "downstream_effects": analysis["downstream_effects"],
                    })

    # 叠加 LightRAG 检索（如果可用）
    if components.lightrag_engine is not None:
        try:
            lr_results = [{"content": components.lightrag_engine.query_sync(state["query"], mode="hybrid")}]
            for r in lr_results[:3]:
                results.append({
                    "type": "lightrag",
                    "content": r if isinstance(r, str) else str(r),
                })
        except Exception:
            pass

    state["retrieval_results"] = results
    return state


def multimodal_retrieve(state: AgentState, components: SharedRetrievalComponents) -> AgentState:
    """多模态文档查询 → RAG-Anything"""
    results = []

    # 先用 Chroma 查文本
    if components.vector_store is not None:
        vr = components.vector_store.search_documents(state["query"], top_k=3)
        for r in vr:
            results.append({"type": "document", "content": r["text"][:500]})

    # RAG-Anything 多模态
    if components.multimodal_engine is not None:
        try:
            mm_results = components.multimodal_engine.query(state["query"])
            for r in mm_results:
                results.append({"type": "multimodal", "content": r if isinstance(r, str) else str(r)})
        except Exception:
            pass

    state["retrieval_results"] = results
    return state


# ── 诊断节点 ────────────────────────────────────────

def diagnose(state: AgentState, components: SharedRetrievalComponents) -> AgentState:
    """
    综合诊断节点 — 用 Ollama LLM 分析检索结果，输出故障诊断结论。
    """
    import ollama

    results = state.get("retrieval_results", [])
    query = state["query"]

    # 构建上下文
    context_parts = []
    for r in results:
        if r.get("type") == "temporal_chain":
            context_parts.append(
                f"故障演化链: {r.get('fault', '')}, "
                f"传播路径: {r.get('propagation_path', [])}"
            )
        elif r.get("type") == "temporal_info":
            context_parts.append(
                f"故障信息: {r.get('fault', '')}, "
                f"下游影响: {r.get('downstream_effects', [])}"
            )
        else:
            context_parts.append(r.get("content", str(r)))

    context = "\n".join(context_parts) if context_parts else "无额外检索信息"

    prompt = f"""你是离心泵故障诊断专家。请基于以下知识图谱检索结果，回答用户问题。

检索上下文:
{context}

用户问题: {query}

请给出:
1. 故障诊断结论
2. 判断依据（引用检索结果中的信息）
3. 建议的验证步骤
4. 推荐的维修/处理措施

限制200字以内，使用专业工程术语。"""

    try:
        response = ollama.chat(
            model=components.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"]
        state["diagnosis"] = answer
        state["confidence"] = 0.7  # 基础置信度
        state["sources"] = [f"Retrieval from {len(results)} results"]
    except Exception as e:
        state["diagnosis"] = f"[本地诊断模式] 无法连接 Ollama: {e}"
        state["confidence"] = 0.3
        state["sources"] = [f"Error: {e}"]

    return state


# ── 构建 Graph ──────────────────────────────────────

def build_graph_a(components: SharedRetrievalComponents) -> StateGraph:
    """
    构建方案A 的 Langgraph Agent。

    ```
    START → router → [vector_retrieve | temporal_retrieve | multimodal_retrieve] → diagnose → END
    ```
    """
    graph = StateGraph(AgentState)

    # 路由节点
    def router(state: AgentState) -> AgentState:
        route = _classify_query(state["query"])
        state["route"] = route
        return state

    graph.add_node("router", router)

    # 检索引擎节点
    graph.add_node("vector_retrieve", lambda s: vector_retrieve(s, components))
    graph.add_node("temporal_retrieve", lambda s: temporal_retrieve(s, components))
    graph.add_node("multimodal_retrieve", lambda s: multimodal_retrieve(s, components))

    # 诊断节点
    graph.add_node("diagnose", lambda s: diagnose(s, components))

    # 边
    graph.set_entry_point("router")

    # 根据路由选择检索引擎
    graph.add_conditional_edges(
        "router",
        lambda s: {
            "simple_fact": "vector_retrieve",
            "temporal": "temporal_retrieve",
            "multimodal": "multimodal_retrieve",
            "diagnosis": "temporal_retrieve",  # 诊断问题先做时序检索
        }[s["route"]],
        {
            "vector_retrieve": "vector_retrieve",
            "temporal_retrieve": "temporal_retrieve",
            "multimodal_retrieve": "multimodal_retrieve",
        },
    )

    # 所有检索完成后 → 诊断
    for node in ["vector_retrieve", "temporal_retrieve", "multimodal_retrieve"]:
        graph.add_edge(node, "diagnose")

    graph.add_edge("diagnose", END)

    return graph


def create_agent_a(components: SharedRetrievalComponents):
    """创建方案A Agent (已编译)"""
    graph = build_graph_a(components)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ── 便捷 API ────────────────────────────────────────

class AgenticDiagnosisEngine:
    """
    方案A 的 Agentic 故障诊断引擎。

    用法:
        engine = AgenticDiagnosisEngine(vector_store=vs, temporal_reasoner=reasoner)
        result = engine.query("轴承外圈故障的原因是什么？")
    """

    def __init__(self, components: SharedRetrievalComponents):
        self.components = components
        self.agent = create_agent_a(components)

    def query(self, question: str, thread_id: str = "default") -> Dict:
        """单次诊断查询"""
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
