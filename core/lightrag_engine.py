"""
LightRAG 集成模块

封装 LightRAG 的初始化、文档插入和混合检索。
支持 Ollama 本地 LLM 和 embedding 模型。

用法:
    from core.lightrag_engine import LightRAGEngine
    engine = LightRAGEngine()
    engine.initialize()
    engine.insert_sync("离心泵故障诊断知识...")
    result = engine.query_sync("轴承故障的原因")

配置:
    默认使用 Ollama qwen3:14b (LLM) + nomic-embed-text (Embedding)
    working_dir: data/output/lightrag_storage
"""

import sys
import asyncio
import threading
from pathlib import Path
from typing import List, Dict, Optional, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class LightRAGEngine:
    """
    LightRAG 引擎封装。

    使用持久化 event loop 线程解决 sync/async 混合调用的问题。
    LLM: Ollama qwen3:14b
    Embedding: Ollama nomic-embed-text:latest
    """

    def __init__(
        self,
        working_dir: str = "data/output/lightrag_storage",
        llm_model: str = "qwen3:14b",
        embedding_model: str = "nomic-embed-text:latest",
    ):
        self._working_dir = Path(PROJECT_ROOT) / working_dir
        self._working_dir.mkdir(parents=True, exist_ok=True)

        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._rag = None
        self._initialized = False
        self._loop = None
        self._loop_thread = None

    # ── LLM / Embedding 回调 ──────────────────────────

    async def _ollama_llm(self, prompt, **kwargs) -> str:
        """Ollama LLM 回调 — 兼容 LightRAG 的调用格式"""
        import ollama

        # LightRAG 传入 messages 列表或纯文本
        if isinstance(prompt, list):
            msgs = prompt
        elif isinstance(prompt, str):
            msgs = [{"role": "user", "content": prompt}]
        else:
            msgs = [{"role": "user", "content": str(prompt)}]

        response = ollama.chat(model=self._llm_model, messages=msgs)
        return response["message"]["content"]

    async def _ollama_embed(self, texts: List[str]):
        """Ollama Embedding 回调 — 返回 numpy 数组"""
        import ollama
        import numpy as np
        embeddings = []
        for text in texts:
            resp = ollama.embeddings(model=self._embedding_model, prompt=text)
            embeddings.append(resp["embedding"])
        return np.array(embeddings, dtype=np.float32)

    # ── 初始化 ────────────────────────────────────────

    def _start_event_loop(self):
        """在独立线程中启动持久 event loop"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _init():
            from lightrag import LightRAG
            from lightrag.base import EmbeddingFunc
            from lightrag.kg.shared_storage import initialize_pipeline_status

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
            print(f"[LightRAG] 初始化完成: {self._working_dir}")
            self._initialized = True

        self._loop.run_until_complete(_init())
        self._loop.run_forever()

    def initialize(self):
        """启动 LightRAG (后台 event loop 线程)"""
        if self._initialized:
            return
        self._loop_thread = threading.Thread(
            target=self._start_event_loop, daemon=True
        )
        self._loop_thread.start()
        # 等待初始化完成
        import time
        deadline = time.time() + 30
        while not self._initialized and time.time() < deadline:
            time.sleep(0.1)
        if not self._initialized:
            raise RuntimeError("LightRAG 初始化超时 (30s)")

    def _run_async(self, coro, timeout=120):
        """在后台 loop 中执行异步操作"""
        if not self._initialized:
            self.initialize()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── 文档操作 ──────────────────────────────────────

    def insert_sync(self, text: str) -> str:
        """插入文档"""
        return self._run_async(self._rag.ainsert(text))

    def insert_batch_sync(self, texts: List[str]):
        """批量插入"""
        for i, text in enumerate(texts):
            self.insert_sync(text)
            if (i + 1) % 10 == 0:
                print(f"  [LightRAG] {i+1}/{len(texts)} 个文档已索引")

    # ── 查询操作 ──────────────────────────────────────

    def query_sync(self, question: str, mode: str = "hybrid") -> str:
        """
        同步查询。

        Args:
            question: 查询文本
            mode: local | global | hybrid | naive | mix
        """
        return self._run_async(
            self._rag.aquery(question, mode=mode)
        )

    # ── 知识库导入 ────────────────────────────────────

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

    # ── 状态 ──────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def rag(self):
        if not self._initialized:
            raise RuntimeError("LightRAG not initialized")
        return self._rag

    def shutdown(self):
        """关闭后台 event loop"""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread:
            self._loop_thread.join(timeout=5)


# ── 工厂函数 ──────────────────────────────────────────

def create_lightrag_engine(
    working_dir: str = "data/output/lightrag_storage",
    llm_model: str = "qwen3:14b",
    auto_import: bool = False,
) -> LightRAGEngine:
    """
    创建并初始化 LightRAG 引擎。

    Args:
        working_dir: 工作目录
        llm_model: Ollama 模型名
        auto_import: 是否自动导入知识库
    """
    engine = LightRAGEngine(working_dir=working_dir, llm_model=llm_model)
    engine.initialize()

    if auto_import:
        engine.import_knowledge_base()

    return engine
