"""
Chroma 向量库封装

用途:
- 设备文档块的语义检索
- 传感器测量记录的频域特征向量相似度检索
- metadata 支持 measurement_id 桥接到 Neo4j

Phase 3 Plan: Chroma 封装 + LightRAG + Langgraph Agent
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Chroma 向量库封装。

    两个 collection:
    - "documents": 设备知识文档块 (embedding 由 Chroma 内置模型生成)
    - "measurements": 传感器频域特征向量 (10 维浮点向量)
    """

    def __init__(self, persist_dir: str = "data/output/vector_store"):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._doc_collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"description": "设备知识文档块 - 用于语义检索"},
        )
        self._meas_collection = self._client.get_or_create_collection(
            name="measurements",
            metadata={"description": "传感器频域特征向量 - 用于相似故障检索"},
        )

    # ── 文档操作 ────────────────────────────────────

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """添加文档块"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        self._doc_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def search_documents(
        self, query: str, top_k: int = 5
    ) -> List[Dict]:
        """语义检索文档块"""
        results = self._doc_collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        docs = []
        if results["documents"] and results["documents"][0]:
            for i, text in enumerate(results["documents"][0]):
                docs.append({
                    "text": text,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "id": results["ids"][0][i] if results["ids"] else f"doc_{i}",
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })
        return docs

    # ── 测量向量操作 ─────────────────────────────────

    def add_measurements(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
    ):
        """添加传感器频域特征向量"""
        self._meas_collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def search_similar_measurements(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        fault_filter: Optional[str] = None,
    ) -> List[Dict]:
        """检索与目标特征最相似的历史测量记录"""
        where_filter = None
        if fault_filter:
            where_filter = {"fault_type": fault_filter}

        results = self._meas_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )
        docs = []
        if results["ids"] and results["ids"][0]:
            for i, mid in enumerate(results["ids"][0]):
                docs.append({
                    "measurement_id": mid,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })
        return docs

    # ── 统计 ────────────────────────────────────────

    def count_documents(self) -> int:
        return self._doc_collection.count()

    def count_measurements(self) -> int:
        return self._meas_collection.count()

    def clear(self):
        """清空所有 collection（用于重新索引）"""
        try:
            self._client.delete_collection("documents")
            self._client.delete_collection("measurements")
        except Exception:
            pass
        self._doc_collection = self._client.get_or_create_collection("documents")
        self._meas_collection = self._client.get_or_create_collection("measurements")
