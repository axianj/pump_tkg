"""
从离心泵数据集 CSV 解析测量记录并构建时序知识图谱

支持两种模式：
1. --from-file PATH  从真实 CSV 文件解析
2. --simulate        生成模拟传感器数据用于演示

用法:
    python scripts/ingest_sensors.py --simulate
    python scripts/ingest_sensors.py --from-file "data/sensors/sample.csv"
"""

import sys, json, os, csv, math, random
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.pump_domain import FAULT_NAMES_ZH, SENSOR_NAMES_ZH


def simulate_measurements(count: int = 30, offset: int = 0) -> list:
    """
    生成模拟传感器测量数据，覆盖多种故障类型。

    每个测量记录包含:
    - 元数据: 时间戳、设备、故障类型、严重度、转速
    - 特征值: RMS, Peak, 1X, 2X, BPFO, BPFI 等频域分量
    """
    faults = [c for c in FAULT_NAMES_ZH.keys() if c != "Healthy"]
    base_time = datetime(2020, 7, 1, 8, 0, 0)

    measurements = []
    for i in range(count):
        fault_type = random.choice(faults)
        severity = random.randint(1, 4)
        speed_pct = random.choice([50, 75, 100])
        ts = base_time + timedelta(hours=i * 6)

        # 根据故障类型和严重度生成信号特征
        features = _simulate_features(fault_type, severity, speed_pct)

        mid = f"M_SIM_{i+1:03d}"

        measurements.append({
            "measurement_id": mid,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "setup_id": f"Set_{random.choice([1,2])}",
            "motor_speed_pct": speed_pct,
            "fault_type": fault_type,
            "fault_name_zh": FAULT_NAMES_ZH.get(fault_type, fault_type),
            "severity": severity,
            "features": features,
            "health_status": "faulty" if fault_type != "Healthy" else "healthy",
            "source_file": f"simulated/{mid}.csv",
        })

    return measurements


def _simulate_features(fault_type: str, severity: int, speed_pct: int) -> dict:
    """模拟特定故障类型的频谱特征"""
    base_rpm = 1480 * speed_pct / 100
    freq_1x = base_rpm / 60

    # 基线值（健康状态）
    rms_base = random.uniform(0.5, 1.5)
    peak_base = random.uniform(2.0, 5.0)

    # 根据故障类型调整特征
    features = {
        "rms": rms_base,
        "peak": peak_base,
        "freq_1x": random.uniform(0.1, 0.3),
        "freq_2x": random.uniform(0.05, 0.1),
        "freq_3x": random.uniform(0.02, 0.05),
        "freq_bpfo": random.uniform(0.01, 0.03),
        "freq_bpfi": random.uniform(0.01, 0.03),
        "freq_bsf": random.uniform(0.005, 0.015),
        "kurtosis": random.uniform(2.5, 3.5),
        "crest_factor": random.uniform(3.0, 5.0),
    }

    sev_factor = severity / 4.0  # 0.25 ~ 1.0

    if "misalignment" in fault_type:
        features["freq_2x"] *= 5 * sev_factor
        features["freq_1x"] *= 2 * sev_factor
        features["rms"] *= 2 * sev_factor
    elif "Unbalance" in fault_type:
        features["freq_1x"] *= 6 * sev_factor
        features["rms"] *= 2.5 * sev_factor
    elif "Cavitation" in fault_type:
        # 宽频带高频
        for k in features:
            if k.startswith("freq_"):
                features[k] *= 1.5 * sev_factor
        features["kurtosis"] *= 0.8
        features["rms"] *= 1.5 * sev_factor
    elif "Bearing" in fault_type or "bearing" in fault_type:
        features["freq_bpfo"] *= 8 * sev_factor
        features["freq_bpfi"] *= 6 * sev_factor
        features["freq_bsf"] *= 4 * sev_factor
        features["crest_factor"] *= 2 * sev_factor
        features["kurtosis"] *= 1.5 * sev_factor
    elif "Impeller" in fault_type:
        features["freq_1x"] *= 3 * sev_factor
        features["rms"] *= 2 * sev_factor
    elif "Stator" in fault_type or "Broken" in fault_type:
        features["freq_1x"] *= 2 * sev_factor
        features["rms"] *= 1.5 * sev_factor

    return features


def measurements_to_graph(measurements: list, output_dir: Path):
    """
    将测量记录转化为知识图谱实体和关系

    每个测量记录生成:
    - 一个 "测量记录" 实体
    - 关联到对应的故障类型
    - 关联到对应的设备
    - 按时间顺序建立 prev/next 关系链
    """
    nodes = []
    edges = []
    seen_ids = set()

    def N(id, type, name, desc, deg=0):
        if id not in seen_ids:
            seen_ids.add(id)
            nodes.append({"id": id, "type": type, "name": name, "description": desc, "degree": deg})

    def E(src, tgt, rel, desc, w=1.0):
        edges.append({"source": src, "target": tgt, "relation": rel, "description": desc, "weight": w})

    # 1. 加载已有的基础图谱（如果存在）
    entities_path = output_dir / "entities.json"
    rels_path = output_dir / "relationships.json"

    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            existing_nodes = json.load(f)
        for n in existing_nodes:
            seen_ids.add(n["id"])
        nodes = existing_nodes

    if rels_path.exists():
        with open(rels_path, "r", encoding="utf-8") as f:
            existing_edges = json.load(f)
        edges = existing_edges

    existing_edge_keys = {(e["source"], e["target"]) for e in edges}

    # 2. 添加测量记录实体
    new_measurements = 0
    for m in measurements:
        mid = m["measurement_id"]
        if mid in seen_ids:
            continue

        features = m.get("features", {})
        feat_str = ", ".join(f"{k}={v:.3f}" for k, v in list(features.items())[:5])
        desc = f"故障:{m['fault_name_zh']} 严重度:{m['severity']}/4 转速:{m['motor_speed_pct']}% | {feat_str}"

        N(mid, "测量记录", f"测量_{m['timestamp'][5:16].replace(' ', '_')}", desc, 2)
        new_measurements += 1

        # 关联到故障类型
        fault_id = f"fault_{m['fault_type']}"
        E(mid, fault_id, "表现为", f"该测量记录显示{m['fault_name_zh']}特征")

        # 关联到设备
        setup = m.get("setup_id", "Set_1")
        if setup == "Set_2" or m["motor_speed_pct"] == 100:
            E("Motor_MG180MB", mid, "监测", f"Motor4在{m['motor_speed_pct']}%转速下采集")
        else:
            E("Motor_MG160MA", mid, "监测", f"Motor2在{m['motor_speed_pct']}%转速下采集")

        # 关联到工况
        speed_id = f"Speed_{m['motor_speed_pct']}pct"
        E(mid, speed_id, "工况", f"转速{m['motor_speed_pct']}%")

        # 特征作为实体
        for feat_name, feat_val in features.items():
            feat_id = f"{mid}_{feat_name}"
            N(feat_id, "信号特征", f"{feat_name}={feat_val:.3f}", f"测量{mid}的{feat_name}特征值", 1)
            E(mid, feat_id, "特征", f"{feat_name}={feat_val:.3f}")

    # 3. 构建时序关系链（按故障类型分组后按时间排序）
    fault_groups = {}
    for m in measurements:
        key = m["fault_type"]
        if key not in fault_groups:
            fault_groups[key] = []
        fault_groups[key].append(m)

    for fault_type, group in fault_groups.items():
        group.sort(key=lambda x: x["timestamp"])
        for i in range(len(group) - 1):
            src, tgt = group[i]["measurement_id"], group[i+1]["measurement_id"]
            key = (src, tgt)
            if key not in existing_edge_keys:
                E(src, tgt, "后续测量", f"{src} → {tgt} (时序前驱)", 0.5)
                E(tgt, src, "前次测量", f"{tgt} ← {src} (时序后继)", 0.5)
                existing_edge_keys.add(key)

    # 4. 保存
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(rels_path, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    return {
        "entities_before": len(seen_ids) - new_measurements if new_measurements > 0 else len(existing_nodes) - new_measurements,
        "entities_added": new_measurements,
        "entities_total": len(nodes),
        "relations_total": len(edges),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="传感器数据注入 — 构建时序知识图谱")
    parser.add_argument("--simulate", action="store_true", help="生成模拟传感器数据")
    parser.add_argument("--count", type=int, default=30, help="模拟数据条数")
    parser.add_argument("--offset", type=int, default=0, help="ID偏移量（用于演示增量更新）")
    parser.add_argument("--from-file", type=str, help="从CSV文件解析")
    parser.add_argument("--output", default="data/output", help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.simulate:
        print(f"[Simulate] 生成 {args.count} 条模拟传感器数据 (offset={args.offset})...")
        measurements = simulate_measurements(args.count, offset=args.offset)
    elif args.from_file:
        print(f"[File] 从 {args.from_file} 解析...")
        # TODO: 实现真实 CSV 解析逻辑
        # 需要 7z 解压后才能读取真实数据
        measurements = []
        print("[File] 真实CSV解析需要先解压数据集。使用 --simulate 生成演示数据。")
        return
    else:
        print("请指定 --simulate 或 --from-file")
        return

    # 注入到图谱
    result = measurements_to_graph(measurements, output_dir)

    print(f"\n[Result] 传感器数据注入完成!")
    print(f"  - 原有实体: {result['entities_before']}")
    print(f"  - 新增实体: {result['entities_added']}")
    print(f"  - 总计实体: {result['entities_total']}")
    print(f"  - 总计关系: {result['relations_total']}")

    # 更新 HTML 可视化
    print("\n[Viz] 重新生成 ECharts 可视化...")
    from visualize.echart_viz import load_data, build_graph_data, generate_html
    entities, relationships = load_data(str(output_dir))
    graph_data = build_graph_data(entities, relationships)
    generate_html(graph_data, "pump_graph_viz.html")

    # 输出时序样本
    print(f"\n[Sample] 生成的测量记录 (共 {len(measurements)} 条, 显示前5条):")
    for m in measurements[:5]:
        feat_str = ", ".join(f"{k}={v:.2f}" for k, v in m["features"].items())
        print(f"  [{m['measurement_id']}] {m['timestamp']} | {m['fault_name_zh']}(sev={m['severity']}) | {feat_str[:80]}")

    if len(measurements) > 5:
        print(f"  ... 还有 {len(measurements)-5} 条")


if __name__ == "__main__":
    main()
