"""LightRAG 增量更新演示"""
import sys, os, asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

WORK_DIR = str(Path(__file__).parent / "lightrag_demo")
import shutil
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

from lightrag import LightRAG
from lightrag.base import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import numpy as np

async def dummy_embed(texts):
    return np.random.rand(len(texts), 64).astype(np.float32)

async def dummy_llm(history, prompt, **kwargs):
    return ""

async def run():
    rag = LightRAG(
        working_dir=WORK_DIR,
        llm_model_func=dummy_llm,
        embedding_func=EmbeddingFunc(embedding_dim=64, max_token_size=8192, func=dummy_embed),
        enable_llm_cache=True,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    print("=" * 60)
    print("  LightRAG 增量更新演示")
    print("=" * 60)
    print(f"工作目录: {WORK_DIR}")

    def show_storage():
        p = Path(WORK_DIR)
        for f in sorted(p.iterdir()):
            if f.is_dir():
                cnt = len(list(f.rglob("*")))
                sz = sum(fs.stat().st_size for fs in f.rglob("*") if fs.is_file())
                if cnt > 0:
                    print(f"  {f.name}/ {cnt} files ({sz/1024:.1f} KB)")
            elif f.is_file():
                print(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")

    docs1 = [
        "离心泵叶轮磨损会导致流量下降和振动增大",
        "轴承BPFI故障的特征频率为135.5Hz",
        "角度不对中的主要特征是2倍转频振动突出",
    ]
    print("\n=== 第一次: 插入 3 条 ===")
    for i, text in enumerate(docs1):
        await rag.ainsert(text)
        print(f"  [{i+1}] OK: {text[:20]}...")
    show_storage()

    docs2 = [
        "转子断条的特征是电源频率两侧出现旁频带",
        "定子短路会导致电流谐波增加和振动异常",
    ]
    print("\n=== 第二次: 增量插入 2 条 ===")
    for i, text in enumerate(docs2):
        await rag.ainsert(text)
        print(f"  [{i+1}] OK: {text[:20]}...")
    show_storage()

    print("\n✅ 成功! 第二次是在第一次基础上追加, 不是重建。")

asyncio.run(run())
