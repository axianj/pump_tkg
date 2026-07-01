"""
离心泵故障诊断时序知识图谱系统 — 主 CLI 入口

基于 RAG-Anything (LightRAG) 构建多模态+时序知识图谱。

用法:
    python pump_main.py build            # 从文档构建知识图谱
    python pump_main.py status           # 查看知识图谱状态
    python pump_main.py ingest-sensors   # 从传感器 CSV 提取时序实体
    python pump_main.py to-echarts       # 生成 ECharts HTML 可视化
    python pump_main.py to-neo4j         # 导入 Neo4j 图数据库
    python pump_main.py web              # 启动 Streamlit Web 界面
    python pump_main.py query            # 交互式查询

工作目录假设:
    data/knowledge/        — 领域知识文档（.md / .txt / .pdf）
    data/sensors/          — 传感器 CSV 数据
    data/output/           — 构建结果（图谱数据、索引）
"""

import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_command(args):
    """构建知识图谱"""
    from scripts.build_index import build_from_documents

    docs_dir = Path("data/knowledge")
    output_dir = Path(args.output) if args.output else Path("data/output")

    # 确保知识文档存在
    if not docs_dir.exists() or not list(docs_dir.glob("*")):
        print("[INFO] 知识文档目录为空，生成默认领域知识")
        _ensure_default_knowledge(docs_dir)

    build_from_documents(
        docs_dir=docs_dir,
        output_dir=output_dir,
        incremental=args.incremental,
    )


def status_command(args):
    """查看知识图谱状态"""
    output_dir = Path("data/output")

    entities_path = output_dir / "entities.json"
    rels_path = output_dir / "relationships.json"
    docs_dir = Path("data/knowledge")

    print("=" * 60)
    print("  离心泵故障知识图谱 — 系统状态")
    print("=" * 60)

    # 知识文档数
    doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))  + list(docs_dir.glob("*.pdf"))
    print(f"\n  [Doc] 知识文档: {len(doc_files)} 个")

    # 图谱实体/关系
    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
        print(f"  [KG]   实体: {len(entities)} 个")
    else:
        print(f"  [KG]   实体: 未构建")

    if rels_path.exists():
        with open(rels_path, "r", encoding="utf-8") as f:
            rels = json.load(f)
        print(f"  [KG]  关系: {len(rels)} 条")
    else:
        print(f"  [KG]  关系: 未构建")

    # 传感器数据
    sensor_dir = Path("data/sensors")
    csv_files = list(sensor_dir.rglob("*.csv"))
    print(f"  [Sens] 传感器数据文件: {len(csv_files)} 个")

    # 输出文件
    html_path = Path("pump_graph_viz.html")
    if html_path.exists():
        size = html_path.stat().st_size / 1024
        print(f"  [HTML] HTML 可视化: {html_path.name} ({size:.0f} KB)")

    print("\n" + "=" * 60)
    print("  可用命令:")
    print("    python pump_main.py build              # 构建图谱")
    print("    python pump_main.py to-echarts         # 生成可视化")
    print("    python pump_main.py to-neo4j           # 导入Neo4j")
    print("    python pump_main.py ingest-sensors     # 注入传感器数据")
    print("    python pump_main.py web                # 启动界面")
    print("=" * 60)


def ingest_sensors_command(args):
    """从传感器CSV提取时序实体并注入图谱"""
    from core.temporal_adapter import TemporalAdapter

    sensor_dir = Path(args.sensor_dir) if args.sensor_dir else Path("data/sensors")
    output_dir = Path(args.output) if args.output else Path("data/output")

    if not sensor_dir.exists():
        print(f"[ERROR] 传感器目录不存在: {sensor_dir}")
        print("从 TKG_Data 复制样本数据后重试")
        return

    csv_files = list(sensor_dir.rglob("*.csv"))
    print(f"[Sensors] 发现 {len(csv_files)} 个 CSV 文件")

    adapter = TemporalAdapter()

    # 解析每个CSV为测量记录
    measurements = []
    for csv_path in csv_files:
        rel = csv_path.relative_to(sensor_dir.parent if sensor_dir.parent != sensor_dir else sensor_dir)
        m = adapter.parse_measurement_from_path(str(rel))
        if m:
            measurements.append(m)

    print(f"[Sensors] 解析出 {len(measurements)} 条测量记录")

    # 导出为 LightRAG 可读格式
    adapter.export_for_lightrag(measurements, output_dir)

    # 合并到知识图谱
    entities_path = output_dir / "entities.json"
    rels_path = output_dir / "relationships.json"

    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            existing_entities = json.load(f)
    else:
        existing_entities = []

    if rels_path.exists():
        with open(rels_path, "r", encoding="utf-8") as f:
            existing_rels = json.load(f)
    else:
        existing_rels = []

    # 添加新实体
    existing_ids = {e["id"] for e in existing_entities}
    new_entities = []
    for m in measurements:
        if m.measurement_id not in existing_ids:
            new_entities.append({
                "id": m.measurement_id,
                "type": "测量记录",
                "name": f"测量_{m.timestamp.strftime('%m%d_%H%M')}_{m.fault_type}",
                "description": f"故障: {m.fault_type}, 严重度: {m.severity}, 转速: {m.motor_speed_pct}%",
                "source": m.source_file,
                "degree": 2,
            })
            existing_ids.add(m.measurement_id)

    # 添加时序关系
    temporal_rels = adapter.build_temporal_relations(measurements)
    existing_rel_keys = {(r["source"], r["target"]) for r in existing_rels}
    new_rels = []
    for r in temporal_rels:
        key = (r["source"], r["target"])
        if key not in existing_rel_keys:
            new_rels.append(r)

    # 输出
    all_entities = existing_entities + new_entities
    all_rels = existing_rels + new_rels

    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(all_entities, f, ensure_ascii=False, indent=2)
    with open(rels_path, "w", encoding="utf-8") as f:
        json.dump(all_rels, f, ensure_ascii=False, indent=2)

    print(f"[Merge] 新增 {len(new_entities)} 个实体, {len(new_rels)} 条关系")
    print(f"[Merge] 总计 {len(all_entities)} 个实体, {len(all_rels)} 条关系")


def query_command(args):
    """交互式查询（简化版 — 循环读取查询）"""
    print("=" * 60)
    print("  离心泵故障知识图谱 — 交互查询 (输入 exit 退出)")
    print("=" * 60)

    print("\n示例查询:")
    print("  > 角度不对中的故障特征")
    print("  > 列出所有传感器通道")
    print("  > 轴承故障有哪些类型")
    print("  > 气蚀和不对中有什么不同")

    while True:
        try:
            q = input("\n> ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break

            # 简单本地匹配
            answer = _local_query(q)
            print(f"\n{answer}")
        except (KeyboardInterrupt, EOFError):
            break

    print("\n再见!")


def _local_query(query: str) -> str:
    """离线查询（不依赖 LLM）"""
    from core.pump_domain import FAULT_NAMES_ZH

    query_lower = query.lower()

    # 检查是否问故障
    found_faults = []
    for code, zh in FAULT_NAMES_ZH.items():
        if code.lower() in query_lower or zh in query:
            found_faults.append((code, zh))

    if found_faults:
        results = []
        for code, zh in found_faults:
            results.append(f"**{zh}** ({code})")
        return "相关故障类型:\n" + "\n".join(f"- {r}" for r in results)

    if "传感器" in query or "通道" in query or "channel" in query_lower:
        return """传感器通道信息:
- Ch1: 电机非驱动端轴承 水平方向
- Ch2: 电机驱动端轴承 垂直方向
- Ch3: 电机驱动端轴承 轴向
- Ch4: 泵驱动端轴承 水平方向
- Ch5: 泵非驱动端轴承 垂直方向
采样率: 20 kHz, 振动单位: g"""

    if "设备" in query or "电机" in query or "泵" in query:
        return """设备信息:
- Motor MG 160 MA: 11kW, 1480rpm, Setup1
- Motor MG 180 MB: 45kW, 2960rpm, Setup2
- Pump NK 80-250: Setup1配套离心泵
- Pump NK 80-160: Setup2配套离心泵"""

    return f"关于「{query}」的知识图谱信息：\n请先构建知识图谱后查询，或使用 LightRAG 集成模式。\n当前已定义故障类型: {', '.join(FAULT_NAMES_ZH.values())}"


def _ensure_default_knowledge(docs_dir: Path):
    """确保知识文档目录存在并有默认文件"""
    docs_dir.mkdir(parents=True, exist_ok=True)

    content = """# 离心泵故障诊断与维护知识库

## 设备系统
离心泵系统由电机、联轴器、泵体、轴承、轴封等组成。

### 电机
MG 160 MA 电机: 额定功率 11kW, 额定转速 1480rpm, 轴承类型 NU311/6311
MG 180 MB 电机: 额定功率 45kW, 额定转速 2960rpm, 轴承类型 NU314/6314

### 离心泵
NK 80-250 离心泵: 适用于 Setup1 (与 MG160MA 配合)
NK 80-160 离心泵: 适用于 Setup2 (与 MG180MB 配合)

### 传感器配置
5个加速度计:
1. 电机非驱动端轴承(水平)
2. 电机驱动端轴承(垂直)
3. 电机驱动端轴承(轴向)
4. 泵驱动端轴承(水平)
5. 泵非驱动端轴承(垂直)
采样率 20kHz, 每样本 12秒

## 故障模式与特征

### 不对中故障
不对中分为角度不对中和平行不对中:
- 角度不对中: 联轴器两端轴线存在角度偏差
- 平行不对中: 联轴器两端轴线平行但偏移
- 特征: 2倍转频(2X)振动分量突出, 轴向振动显著增大

### 不平衡故障
- 电机不平衡: 转子质量分布不均
- 泵不平衡: 叶轮磨损、结垢或腐蚀
- 特征: 1倍转频(1X)振动分量突出, 径向振动为主

### 轴承故障
轴承故障分为外圈(BPFO)、内圈(BPFI)和滚动体(BSF)故障:
- BPFO: 外圈故障特征频率
- BPFI: 内圈故障特征频率
- BSF: 滚动体故障特征频率
- 特征: 对应特征频率及其谐波, 伴有边频带

### 气蚀
- 吸入口气蚀: 泵入口压力低于液体饱和蒸气压
- 排出口气蚀: 泵出口压力过高导致
- 特征: 宽频带高频振动, 伴有噪声和流量波动

### 断条故障
- 转子导条断裂
- 特征: 电源频率的旁频带, 极通过频率边带

### 定子短路
- 定子绕组匝间短路
- 特征: 电流增大, 振动异常, 谐波成分增加

## 诊断方法

### 时域分析
- RMS值: 反映总体振动水平
- 峰值: 反映冲击性故障
- 峭度: 反映信号的非高斯性, 对早期故障敏感

### 频域分析
- FFT频谱: 识别故障特征频率
- 包络谱: 对轴承故障敏感
- 倒频谱: 识别边频带模式

## 维护策略
- 定期对中检查与调整
- 动平衡校正
- 轴承定期更换与润滑
- 入口过滤器清洗
- 监测振动趋势, 设定报警阈值
"""
    path = docs_dir / "pump_fault_knowledge.md"
    path.write_text(content, encoding="utf-8")
    print(f"[Init] 生成默认知识文档: {path}")


def web_command(args):
    """启动 Streamlit Web 界面"""
    print("启动离心泵故障诊断系统界面...")
    import subprocess
    web_path = Path(__file__).parent / "web" / "app.py"
    subprocess.run(["streamlit", "run", str(web_path)])


def main():
    parser = argparse.ArgumentParser(
        description="Pump TKG - Centrifugal Pump Fault Diagnosis Temporal KG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

使用示例:
  python pump_main.py init              # 初始化（创建默认知识文档）
  python pump_main.py build             # 构建知识图谱
  python pump_main.py status            # 查看状态
  python pump_main.py ingest-sensors    # 注入传感器数据
  python pump_main.py to-echarts       # 生成 ECharts 可视化
  python pump_main.py to-neo4j         # 导入 Neo4j
  python pump_main.py web              # 启动 Web 界面
  python pump_main.py query            # 交互查询
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # init
    subparsers.add_parser("init", help="初始化项目（创建默认知识文档）")

    # build
    build_parser = subparsers.add_parser("build", help="从文档构建知识图谱")
    build_parser.add_argument("--output", "-o", default="data/output", help="输出目录")
    build_parser.add_argument("--incremental", action="store_true", default=True,
                              help="增量更新（默认启用）")
    build_parser.add_argument("--full", action="store_true", dest="full",
                              help="完全重建（禁用增量）")

    # status
    subparsers.add_parser("status", help="查看知识图谱状态")

    # ingest-sensors
    sensor_parser = subparsers.add_parser("ingest-sensors",
                                          help="从传感器 CSV 提取时序实体")
    sensor_parser.add_argument("--sensor-dir", default="data/sensors",
                               help="传感器数据目录")
    sensor_parser.add_argument("--output", "-o", default="data/output",
                               help="输出目录")

    # to-echarts
    echart_parser = subparsers.add_parser("to-echarts", help="生成 ECharts HTML 可视化")
    echart_parser.add_argument("--data-dir", default="data/output", help="数据目录")
    echart_parser.add_argument("--output", default="pump_graph_viz.html",
                               help="输出 HTML 文件")

    # to-neo4j
    neo4j_parser = subparsers.add_parser("to-neo4j", help="导入 Neo4j 图数据库")
    neo4j_parser.add_argument("--data-dir", default="data/output", help="数据目录")
    neo4j_parser.add_argument("--clear", action="store_true", help="清空后导入")
    neo4j_parser.add_argument("--temporal", action="store_true", help="导入时序关系")

    # query
    subparsers.add_parser("query", help="交互式查询")

    # web
    subparsers.add_parser("web", help="启动 Streamlit Web 界面")

    args = parser.parse_args()

    if args.command == "init":
        _ensure_default_knowledge(Path("data/knowledge"))
        print("[Init] 完成！可运行 python pump_main.py build 构建知识图谱")

    elif args.command == "build":
        if args.full:
            args.incremental = False
        build_command(args)

    elif args.command == "status":
        status_command(args)

    elif args.command == "ingest-sensors":
        ingest_sensors_command(args)

    elif args.command == "to-echarts":
        from visualize.echart_viz import load_data, build_graph_data, generate_html
        entities, relationships = load_data(args.data_dir)
        graph_data = build_graph_data(entities, relationships)
        generate_html(graph_data, args.output)

    elif args.command == "to-neo4j":
        from visualize.neo4j_import import main as neo4j_main
        sys.argv = ["neo4j_import.py"]
        if args.data_dir:
            sys.argv.extend(["--data-dir", args.data_dir])
        if args.clear:
            sys.argv.append("--clear")
        if args.temporal:
            sys.argv.append("--temporal")
        neo4j_main()

    elif args.command == "query":
        query_command(args)

    elif args.command == "web":
        web_command(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
