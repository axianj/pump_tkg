"""
共享 Agent 状态定义

两个 Agent (A和B) 使用相同的状态结构，确保 Phase 5 对比实验的公平性。
"""

from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    query: str
    route: str
    retrieval_results: List[Dict]
    diagnosis: str
    confidence: float
    sources: List[str]
