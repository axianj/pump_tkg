"""
一键生成离心泵基础知识图谱并可视化
使用统一的 KnowledgeGraphBuilder
"""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.kg_builder import KnowledgeGraphBuilder


def main():
    kbg = KnowledgeGraphBuilder()
    out = Path("data/output")
    entities, edges = kbg.build(output_dir=out)

    print(f"[Build] 完成! 实体: {len(entities)}, 关系: {len(edges)}")

    # 生成 ECharts HTML
    from visualize.echart_viz import build_graph_data, generate_html

    graph = build_graph_data(entities, edges)
    generate_html(graph, "pump_graph_viz.html")

    size = os.path.getsize("pump_graph_viz.html") / 1024
    print(f"[ECharts] pump_graph_viz.html ({size:.0f} KB)")

    # 测试 query 命令的信息
    from core.pump_domain import PumpDomainKnowledge, FAULT_NAMES_ZH
    d = PumpDomainKnowledge()
    print("\n[Query] 知识库内容预览:")
    print(f"  - 设备类型: {list(d.equipment_specs.keys())}")
    print(f"  - 故障类型: {len(FAULT_NAMES_ZH)-1} 个")
    print(f"  - 传感器: {len(d.sensor_configs)} 个通道")
    print(f"  - 工况: 4 种转速")
    print(f"\n  python pump_main.py to-neo4j   # 导入 Neo4j")
    print(f"  python pump_main.py web          # 启动 Web 界面")
    print(f"  python pump_main.py query        # 交互查询")


if __name__ == "__main__":
    main()
