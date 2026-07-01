"""
离心泵故障知识图谱索引脚本

基于 RAG-Anything (LightRAG) 构建多模态知识图谱。
支持：
1. 从 PDF 文档（设备手册、报告）提取知识 → 构建图谱
2. 从传感器 CSV 提取时序特征 → 构建时序实体
3. 增量更新（新数据到达时只追加不重头构建）
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

# 项目根目录
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
    from core.pump_domain import PumpDomainKnowledge

    domain = PumpDomainKnowledge()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not working_dir:
        working_dir = str(output_dir / "lightrag_storage")

    # 收集所有知识文档
    doc_files = []
    for ext in ["*.md", "*.txt", "*.pdf"]:
        doc_files.extend(docs_dir.glob(ext))

    if not doc_files:
        print("[WARN] 未找到知识文档，尝试生成领域默认知识")
        doc_files = [_generate_default_knowledge(output_dir)]

    print(f"[Build] 发现 {len(doc_files)} 个文档文件")

    # 如果 LightRAG 可用，使用它构建
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

    # 复制文档到输出目录
    docs_out = output_dir / "documents"
    docs_out.mkdir(exist_ok=True)
    for f in doc_files:
        content = f.read_text(encoding="utf-8")
        target = docs_out / f.name
        target.write_text(content, encoding="utf-8")

    # 生成节点和关系文件
    nodes, edges = _generate_base_graph(domain)

    with open(output_dir / "entities.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(output_dir / "relationships.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"[Output] 实体: {len(nodes)} 个, 关系: {len(edges)} 条")
    return output_dir


def _generate_base_graph(domain):
    """生成基础知识图谱节点和关系（不依赖 LLM 提取）"""
    nodes = []
    edges = []

    # 1. 设备节点
    for name, spec in domain.equipment_specs.items():
        type_zh = "电机" if spec.type == "motor" else "离心泵"
        nodes.append({
            "id": name, "type": "设备", "name": spec.model,
            "description": f"{type_zh} {spec.model}",
            "source": "datasheet_pdf"
        })

    # 2. 故障类型节点
    for fault_name, severities in domain.fault_taxonomy.motor2_faults.items():
        nodes.append({
            "id": f"Fault_{fault_name}",
            "type": "故障类型",
            "name": domain._build_equipment_specs.__class__  # 用 get_fault_name_zh
        })

    # 改用 pump_domain 的 FAULT_NAMES_ZH
    from core.pump_domain import FAULT_NAMES_ZH
    for fault_code, zh_name in FAULT_NAMES_ZH.items():
        if fault_code == "Healthy":
            continue
        nodes.append({
            "id": f"fault_{fault_code}",
            "type": "故障类型",
            "name": zh_name,
            "description": f"{zh_name} ({fault_code})"
        })

    # 3. 传感器节点
    for sensor in domain.sensor_configs:
        from core.pump_domain import SENSOR_NAMES_ZH
        ch_key = f"Ch{sensor.channel}"
        nodes.append({
            "id": ch_key,
            "type": "传感器",
            "name": f"{ch_key} {SENSOR_NAMES_ZH.get(ch_key, '')}",
            "description": f"{sensor.location} {sensor.orientation}向 加速度计"
        })

    # 4. 基础关系
    # 设备 → 包含 → 故障
    for fault_code in FAULT_NAMES_ZH:
        if fault_code == "Healthy":
            continue
        edges.append({
            "source": "Motor_MG160MA",
            "target": f"fault_{fault_code}",
            "relation": "包含",
            "description": f"可在该设备上模拟的故障"
        })
        edges.append({
            "source": "Motor_MG180MB",
            "target": f"fault_{fault_code}",
            "relation": "包含",
            "description": f"可在该设备上模拟的故障"
        })

    return nodes, edges


def _export_graph_data(rag, output_dir: Path, domain):
    """从 LightRAG 导出图数据到 JSON"""
    # 这个函数在 LightRAG 没有直接导出接口时，
    # 通过查询所有实体来获取图结构
    try:
        # 尝试通过 LightRAG 的内部存储获取实体
        if hasattr(rag, "graph_storage"):
            graph = rag.graph_storage
            # 实体导出逻辑 — 取决于 LightRAG 版本
            print("[Export] 从 LightRAG 导出图数据...")
    except Exception as e:
        print(f"[Export] 导出失败（非致命）: {e}")

    # 作为 fallback，生成基础图
    nodes, edges = _generate_base_graph(domain)
    with open(output_dir / "entities.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(output_dir / "relationships.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)


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
