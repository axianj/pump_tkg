"""
一键生成离心泵基础知识图谱并可视化
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.pump_domain import PumpDomainKnowledge, FAULT_NAMES_ZH, SENSOR_NAMES_ZH

def main():
    d = PumpDomainKnowledge()
    out = Path("data/output")
    out.mkdir(parents=True, exist_ok=True)

    nodes = []
    edges = []
    seen_ids = set()

    def add_node(id, type, name, description, degree=0):
        if id not in seen_ids:
            seen_ids.add(id)
            nodes.append({"id": id, "type": type, "name": name, "description": description, "degree": degree})

    def add_edge(src, tgt, relation, description, weight=1.0):
        edges.append({"source": src, "target": tgt, "relation": relation, "description": description, "weight": weight})

    # === 1. 设备节点 ===
    for ename, spec in d.equipment_specs.items():
        t = "电机" if spec.type == "motor" else "离心泵"
        add_node(ename, "设备", spec.model, f"{t} {spec.model}, {spec.rated_power_kw or '?'}kW")
        if spec.type == "motor":
            add_node(f"{ename}_DE_bearing", "部件", f"{spec.model}驱动端轴承", "Drive End bearing")
            add_node(f"{ename}_NDE_bearing", "部件", f"{spec.model}非驱动端轴承", "Non-Drive End bearing")
            add_edge(ename, f"{ename}_DE_bearing", "包含", f"{spec.model}包含驱动端轴承")
            add_edge(ename, f"{ename}_NDE_bearing", "包含", f"{spec.model}包含非驱动端轴承")

    # === 2. 故障节点 ===
    for code, zh in FAULT_NAMES_ZH.items():
        if code == "Healthy":
            continue
        add_node(f"fault_{code}", "故障类型", zh, f"{zh} ({code})")
        # 关联到设备
        add_edge("Motor_MG160MA", f"fault_{code}", "包含", "该设备可模拟此故障")
        add_edge("Motor_MG180MB", f"fault_{code}", "包含", "该设备可模拟此故障")

    # === 3. 故障之间的关联推理 ===
    # (基于领域知识的预定义推理规则)
    misalignment_faults = ["Angular_misalignment", "Parallel_misalignment", "Combined_misalignment"]
    for i, a in enumerate(misalignment_faults):
        for j, b in enumerate(misalignment_faults):
            if i < j:
                add_edge(f"fault_{a}", f"fault_{b}", "关联", "同类故障: 不对中类")

    cavitation_faults = ["Cavitation_suction", "Cavitation_discharge"]
    add_edge(f"fault_{cavitation_faults[0]}", f"fault_{cavitation_faults[1]}", "关联", "同类故障: 气蚀类")

    bearing_faults = ["Bearing_BPFO", "Bearing_BPFI", "Bearing_BSF", "Bearing_contaminated", "Pump_bearing"]
    for i, a in enumerate(bearing_faults):
        for j, b in enumerate(bearing_faults):
            if i < j:
                add_edge(f"fault_{a}", f"fault_{b}", "关联", "同类故障: 轴承类")

    # 因果推理: 不对中 → 轴承磨损
    for mf in misalignment_faults:
        for bf in bearing_faults:
            add_edge(f"fault_{mf}", f"fault_{bf}", "导致", f"{FAULT_NAMES_ZH.get(mf,'')}可能导致{FAULT_NAMES_ZH.get(bf,'')}")

    # 因果推理: 气蚀 → 叶轮故障
    add_edge("fault_Cavitation_suction", "fault_Impeller_fault", "导致", "吸入口气蚀可能导致叶轮损坏")
    add_edge("fault_Cavitation_discharge", "fault_Impeller_fault", "导致", "排出口气蚀可能导致叶轮损坏")

    # 因果推理: 联轴器损坏 → 不对中
    add_edge("fault_Coupling_damage", "fault_Angular_misalignment", "导致", "联轴器损坏可能导致角度不对中")
    add_edge("fault_Coupling_damage", "fault_Parallel_misalignment", "导致", "联轴器损坏可能导致平行不对中")

    # === 4. 传感器节点 ===
    for s in d.sensor_configs:
        ch_key = f"Ch{s.channel}"
        sensor_name = f"{ch_key}_{s.location}_{s.orientation}"
        zh_loc = SENSOR_NAMES_ZH.get(ch_key, s.location)
        add_node(ch_key, "传感器", f"{ch_key} {zh_loc}", f"{s.location} {s.orientation}向 加速度计 100mV/g")
        add_node(sensor_name, "监测点", zh_loc, f"{s.location} 位置")
        add_edge(sensor_name, ch_key, "安装于", f"{sensor_name}安装{ch_key}")
        # 关联传感器到设备
        if "Motor" in s.location:
            motor = "Motor_MG160MA"
            add_edge(motor, sensor_name, "包含", f"{motor}包含{sensor_name}")
        elif "Pump" in s.location:
            pump = "Pump_NK80-250"
            add_edge(pump, sensor_name, "包含", f"{pump}包含{sensor_name}")

    # === 5. 工况条件 ===
    for speed_name, speed_pct in [("Speed_50pct", 50), ("Speed_75pct", 75), ("Speed_100pct", 100)]:
        add_node(speed_name, "工况条件", f"{speed_pct}%转速", f"电机转速 {speed_pct}%")
        add_edge("Motor_MG160MA", speed_name, "工况", f"MG160MA可在{speed_pct}%转速运行")

    # === 6. 保存 ===
    with open(out / "entities.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(out / "relationships.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"[Build] 完成! 实体: {len(nodes)}, 关系: {len(edges)}")

    # === 7. 生成 ECharts HTML ===
    from visualize.echart_viz import build_graph_data, generate_html

    graph = build_graph_data(nodes, edges)
    generate_html(graph, "pump_graph_viz.html")

    import os
    size = os.path.getsize("pump_graph_viz.html") / 1024
    print(f"[ECharts] pump_graph_viz.html ({size:.0f} KB)")

    # === 8. 测试 query 命令 ===
    print("\n[Query] 知识库内容预览:")
    print(f"  - 设备类型: {list(d.equipment_specs.keys())}")
    print(f"  - 故障类型: {len(FAULT_NAMES_ZH)-1}个")
    print(f"  - 传感器: {len(d.sensor_configs)}个通道")
    print(f"  - 工况: 3种转速")
    print(f"\n  python pump_main.py to-neo4j   # 导入Neo4j")
    print(f"  python pump_main.py web          # 启动Web界面")
    print(f"  python pump_main.py query        # 交互查询")


if __name__ == "__main__":
    main()
