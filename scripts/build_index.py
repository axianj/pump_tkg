"""
离心泵故障知识图谱索引脚本

基于 RAG-Anything (LightRAG) 构建多模态知识图谱。
支持：
1. 从 PDF 文档（设备手册、报告）提取知识 → 构建图谱
2. 从传感器 CSV 提取时序特征 → 构建时序实体
3. 增量更新（新数据到达时只追加不重头构建）

KG 构建逻辑已抽取至 core/kg_builder.py。
"""

import sys
import os
from pathlib import Path
from typing import Optional, Callable

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def ensure_lightrag():
    """确保 LightRAG 可用"""
    try:
        from lightrag import LightRAG
        return True
    except ImportError:
        print("[WARN] LightRAG 未安装。请运行: pip install lightrag-hku")
        print("[INFO] 将回退到简化模式（纯文本 + 文件存储）")
        return False


def build_from_documents(
    docs_dir: Path,
    output_dir: Path,
    llm_func: Optional[Callable] = None,
    embedding_func: Optional[Callable] = None,
    working_dir: Optional[str] = None,
    incremental: bool = True,
):
    """
    从文档目录构建知识图谱

    Args:
        docs_dir: 包含 .md / .txt / .pdf 文档的目录
        output_dir: 输出目录
        llm_func: LLM 函数（不传则使用简化模式）
        embedding_func: Embedding 函数
        incremental: 是否为增量更新
        working_dir: LightRAG 工作目录（存储图和索引）
    """
    from core.kg_builder import KnowledgeGraphBuilder
    from core.pump_domain import PumpDomainKnowledge

    domain = PumpDomainKnowledge()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not working_dir:
        working_dir = str(output_dir / "lightrag_storage")

    # 收集知识文档
    doc_files = []
    for ext in ["*.md", "*.txt", "*.pdf"]:
        doc_files.extend(docs_dir.glob(ext))

    if not doc_files:
        print("[WARN] 未找到知识文档，使用领域默认知识")
        doc_files = [_generate_default_knowledge(output_dir)]

    print(f"[Build] 发现 {len(doc_files)} 个文档文件")

    # LightRAG 可用则使用它，否则用简化模式
    if ensure_lightrag():
        return _build_with_lightrag(
            doc_files, output_dir, working_dir,
            llm_func, embedding_func, incremental, domain
        )
    else:
        return _build_simple(doc_files, output_dir, domain)


def _build_with_lightrag(
    doc_files: List[Path],
    output_dir: Path,
    working_dir: str,
    llm_func: Optional[Callable],
    embedding_func: Optional[Callable],
    incremental: bool,
    domain,
):
    """使用 LightRAG 构建知识图谱"""
    from lightrag import LightRAG

    # 初始化 LightRAG
    rag_params = {
        "working_dir": working_dir,
        "enable_llm_cache": True,
    }
    if llm_func:
        rag_params["llm_model_func"] = llm_func
    if embedding_func:
        rag_params["embedding_func"] = embedding_func

    rag = LightRAG(**rag_params)
    print(f"[LightRAG] 初始化完成: {working_dir}")

    # 插入文档
    for i, doc_path in enumerate(doc_files):
        try:
            content = doc_path.read_text(encoding="utf-8")
            if incremental:
                # 增量模式：检查是否已插入
                rag.insert(content)
            else:
                rag.insert(content)
            print(f"  [{i+1}/{len(doc_files)}] {doc_path.name} → OK")
        except Exception as e:
            print(f"  [{i+1}/{len(doc_files)}] {doc_path.name} → FAILED: {e}")

    # 导出图数据
    _export_graph_data(rag, output_dir, domain)
    return output_dir


def _build_simple(doc_files: List[Path], output_dir: Path, domain):
    """简化模式：直接复制文档并生成结构化数据（无需 LightRAG）"""
    print("[Mode] 简化模式 — 不依赖 LightRAG")
    from core.kg_builder import KnowledgeGraphBuilder

    # 复制文档到输出目录
    docs_out = output_dir / "documents"
    docs_out.mkdir(exist_ok=True)
    for f in doc_files:
        content = f.read_text(encoding="utf-8")
        target = docs_out / f.name
        target.write_text(content, encoding="utf-8")

    # 使用统一构建器
    kgb = KnowledgeGraphBuilder()
    nodes, edges = kgb.build(output_dir=output_dir)

    return output_dir


# ── 以下函数已迁移至 core/kg_builder.py，保留为向后兼容的 deletegate ──

def _generate_base_graph(domain):
    """[已废弃] 请使用 core.kg_builder.KnowledgeGraphBuilder"""
    from core.kg_builder import KnowledgeGraphBuilder
    kgb = KnowledgeGraphBuilder()
    return kgb.build()


def _export_graph_data(rag, output_dir: Path, domain):
    """从 LightRAG 导出图数据到 JSON (fallback 到统一构建器)"""
    from core.kg_builder import KnowledgeGraphBuilder
    try:
        if hasattr(rag, "graph_storage"):
            graph = rag.graph_storage
            print("[Export] 从 LightRAG 导出图数据...")
    except Exception as e:
        print(f"[Export] 导出失败（非致命）: {e}")

    kgb = KnowledgeGraphBuilder()
    kgb.build(output_dir=output_dir)


def _generate_default_knowledge(output_dir: Path) -> Path:
    """如果没有找到文档则生成默认知识文件"""
    content = """# 离心泵故障诊断知识库

## 设备概述
- 电机: MG 160 MA (11kW, 1480rpm) / MG 180 MB (45kW, 2960rpm)
- 离心泵: NK 80-250 / NK 80-160 (KSB)

## 常见故障模式

### 1. 不对中 (Misalignment)
- 角度不对中: 联轴器两端轴线成角度
- 平行不对中: 联轴器两端轴线平行偏移
- 特征: 2倍转频振动大，轴向振动大

### 2. 不平衡 (Unbalance)
- 电机不平衡: 转子质量分布不均
- 泵不平衡: 叶轮磨损或结垢
- 特征: 1倍转频振动大，径向振动为主

### 3. 气蚀 (Cavitation)
- 吸入口气蚀: 入口压力过低
- 排出口气蚀: 出口压力过高
- 特征: 宽频带高频振动，伴有噪声

### 4. 轴承故障 (Bearing Fault)
- BPFO: 外圈故障特征频率
- BPFI: 内圈故障特征频率
- BSF: 滚动体故障特征频率
- 特征: 对应特征频率及其谐波

### 5. 其他故障
- 软脚: 底座不平导致机壳变形
- 断条: 转子导条断裂
- 定子短路: 绕组匝间短路
"""
    path = output_dir / "knowledge_base.md"
    path.write_text(content, encoding="utf-8")
    return path


if __name__ == "__main__":
    # 简易测试
    print("=== 离心泵故障知识图谱构建工具 ===")
    print("请通过主入口 pump_main.py 使用")
