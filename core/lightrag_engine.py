"""
LightRAG 集成模块

封装 LightRAG 的初始化、文档插入和混合检索。
支持 Ollama 本地 LLM 和 embedding 模型。

LightRAG 提供双层检索:
- global 模式: 处理全局/时序关系（社区摘要级）
- local 模式: 处理具体实体匹配（邻居节点级）
- hybrid 模式: 两者结合

Phase 3 Plan: 方案A 的核心检索后端
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class LightRAGEngine:
    """
    LightRAG 引擎封装。

    用法:
        engine = LightRAGEngine(working_dir="data/output/lightrag_storage")
        await engine.initialize()
        await engine.insert("离心泵故障诊断知识...")
        results = await engine.query("轴承故障的原因", mode="hybrid")
    """

    def __init__(
        self,
        working_dir: str = "data/output/lightrag_storage",
        llm_model: str = "qwen3:14b",
        embedding_model: str = "nomic-embed-text:latest",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self._working_dir = Path(PROJECT_ROOT) / working_dir
        self._working_dir.mkdir(parents=True, exist_ok=True)

        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._ollama_base_url = ollama_base_url
        self._rag = None
        self._initialized = False

    async def _ollama_llm(self, prompt: str, **kwargs) -> str:
        """Ollama LLM 回调函数"""
        import ollama
        response = ollama.chat(
            model=self._llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    async def _ollama_embed(self, texts: List[str]) -> List[List[float]]:
        """Ollama Embedding 回调函数 — 返回 numpy 数组"""
        import ollama
        import numpy as np
        embeddings = []
        for text in texts:
            response = ollama.embeddings(
                model=self._embedding_model,
                prompt=text,
            )
            embeddings.append(response["embedding"])
        return np.array(embeddings, dtype=np.float32)

    def initialize(self, llm_func: Optional[Callable] = None,
                   embedding_func: Optional[Callable] = None):
        """
        初始化 LightRAG 实例。

        使用异步执行方式绕过 async/await 兼容性问题。
        """
        from lightrag import LightRAG
        from lightrag.base import EmbeddingFunc

        llm = llm_func or self._ollama_llm
        embed = embedding_func or self._ollama_embed

        # 包装同步调用
        def sync_llm(prompt, **kwargs):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(asyncio.ensure_future(
                    llm(prompt, **kwargs) if asyncio.iscoroutinefunction(llm)
                    else asyncio.get_event_loop().run_in_executor(None, lambda: llm(prompt, **kwargs))
                ))
            finally:
                loop.close()

        self._rag = LightRAG(
            working_dir=str(self._working_dir),
            llm_model_func=llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=embed,
            ),
            enable_llm_cache=True,
        )
        self._initialized = True

    def initialize_sync(self):
        """
        同步初始化 (使用默认 Ollama 配置)。

        这是最简单的初始化方式，适合 CLI 工具。
        """
        import asyncio
        from lightrag import LightRAG
        from lightrag.base import EmbeddingFunc
        from lightrag.kg.shared_storage import initialize_pipeline_status

        async def _init():
            self._rag = LightRAG(
                working_dir=str(self._working_dir),
                llm_model_func=self._ollama_llm,
                embedding_func=EmbeddingFunc(
                    embedding_dim=768,
                    max_token_size=8192,
                    func=self._ollama_embed,
                ),
                enable_llm_cache=True,
            )
            await self._rag.initialize_storages()
            await initialize_pipeline_status()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_init())
        finally:
            loop.close()
        self._initialized = True

    # ── 文档操作 ────────────────────────────────────

    async def insert(self, text: str, ids: Optional[str] = None) -> str:
        """插入文档（异步）"""
        if not self._initialized:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        return await self._rag.ainsert(text, ids=ids)

    def insert_sync(self, text: str) -> str:
        """插入文档（同步封装）"""
        import asyncio
        if not self._initialized:
            self.initialize_sync()
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._rag.ainsert(text))
        finally:
            loop.close()

    async def insert_batch(self, texts: List[str]) -> List[str]:
        """批量插入文档"""
        results = []
        for i, text in enumerate(texts):
            result = await self.insert(text, ids=f"doc_{i}")
            results.append(result)
        return results

    def insert_batch_sync(self, texts: List[str]) -> List[str]:
        """批量插入（同步封装）"""
        import asyncio
        if not self._initialized:
            self.initialize_sync()
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.insert_batch(texts))
        finally:
            loop.close()

    # ── 查询操作 ────────────────────────────────────

    async def query(
        self, question: str, mode: str = "hybrid", top_k: int = 5
    ) -> List[Dict]:
        """
        查询 LightRAG。

        Args:
            question: 查询文本
            mode: "local" (精确) | "global" (全局) | "hybrid" (混合)
            top_k: 返回 Top-K
        """
        if not self._initialized:
            raise RuntimeError("LightRAG not initialized.")
        result = await self._rag.aquery(question, mode=mode)
        return result if isinstance(result, list) else [{"content": str(result)}]

    def query_sync(self, question: str, mode: str = "hybrid") -> str:
        """同步查询（返回文本）"""
        if not self._initialized:
            self.initialize_sync()
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._rag.aquery(question, mode=mode))
        finally:
            loop.close()

    # ── 知识库操作 ──────────────────────────────────

    def import_knowledge_base(self, knowledge_dir: str = "data/knowledge"):
        """从知识库目录导入所有文档"""
        kb_dir = Path(PROJECT_ROOT) / knowledge_dir
        if not kb_dir.exists():
            print(f"[LightRAG] 知识库目录不存在: {kb_dir}")
            return

        doc_files = []
        for pattern in ["*.md", "*.txt"]:
            doc_files.extend(kb_dir.glob(pattern))

        if not doc_files:
            print("[LightRAG] 知识库目录为空")
            return

        texts = []
        for f in doc_files:
            content = f.read_text(encoding="utf-8")
            texts.append(content)

        print(f"[LightRAG] 从 {len(texts)} 个文档初始化知识库...")
        self.insert_batch_sync(texts)
        print(f"[LightRAG] 完成: {len(texts)} 个文档已索引")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def rag(self):
        """直接访问底层 LightRAG 实例"""
        if not self._initialized:
            raise RuntimeError("LightRAG not initialized")
        return self._rag


def create_lightrag_engine(
    working_dir: str = "data/output/lightrag_storage",
    llm_model: str = "qwen3:14b",
    auto_import: bool = False,
) -> LightRAGEngine:
    """
    工厂函数: 创建并初始化 LightRAG 引擎。

    Args:
        working_dir: 工作目录
        llm_model: Ollama 模型名
        auto_import: 是否自动导入知识库
    """
    engine = LightRAGEngine(
        working_dir=working_dir,
        llm_model=llm_model,
    )
    engine.initialize_sync()

    if auto_import:
        engine.import_knowledge_base()

    return engine
