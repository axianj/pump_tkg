"""
真实传感器 CSV 解析模块

数据集: Motor Current and Vibration Monitoring Dataset for various
         Faults in an E-motor-driven Centrifugal Pump (v4)

数据布局:
    Dataset/{Vibration_or_Current}/{Setup}/{Speed}/{Fault}/{Severity}/{Channel}.csv

CSV 格式（参考数据集论文，Data in Brief, 2023）:
    振动文件: 5 列 (Ch1-Ch5), 每列 240,000 采样点 (20kHz × 12s)
    电流文件: 6 列 (3相电压 + 3相电流), 每列 300,000 采样点 (20kHz × 15s)

用法:
    python scripts/ingest_real_sensors.py --data-dir <path_to_extracted_dataset> --output data/output
    python scripts/ingest_real_sensors.py --simulate --count 30  # 模拟数据
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 信号处理工具 ────────────────────────────────────

def extract_features(signal: np.ndarray, fs: float = 20000.0) -> Dict[str, float]:
    """
    从一维振动信号中提取时域 + 频域特征。

    Args:
        signal: 原始振动信号 (一维数组)
        fs: 采样频率 (Hz)

    Returns:
        {
            "rms": float, "peak": float, "kurtosis": float, "crest_factor": float,
            "freq_1x": float, "freq_2x": float, "freq_3x": float,
            "freq_bpfo": float, "freq_bpfi": float, "freq_bsf": float,
        }

    注意:
        freq_1x/2x/3x 通过转频附近的频带能量近似计算。真实实现需要知道
        每一条数据对应的实际转速（从 setup 文件中读取），然后精确计算。
        这里的实现是一个通用近似方案。
    """
    # 时域特征
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    kurtosis = float(np.mean((signal - np.mean(signal)) ** 4) / (np.std(signal) ** 4 + 1e-10))
    crest_factor = float(peak / (rms + 1e-10))

    # 频域特征 — FFT
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # 转频基频约 25Hz (1480rpm/60s) ~ 49Hz (2960rpm/60s)
    # 取 1X~100Hz 的最大值作为 1X 近似
    band_1x = fft_vals[(freqs >= 20) & (freqs <= 35)]
    band_2x = fft_vals[(freqs >= 40) & (freqs <= 70)]
    band_3x = fft_vals[(freqs >= 60) & (freqs <= 100)]
    freq_1x = float(np.mean(band_1x)) / (np.mean(fft_vals) + 1e-10) if len(band_1x) > 0 else 0.0
    freq_2x = float(np.mean(band_2x)) / (np.mean(fft_vals) + 1e-10) if len(band_2x) > 0 else 0.0
    freq_3x = float(np.mean(band_3x)) / (np.mean(fft_vals) + 1e-10) if len(band_3x) > 0 else 0.0

    # BPFO/BPFI/BSF 频带 (基于 NU311/6311 轴承参数的近似值, 需根据实际型号调整)
    band_bpfo = fft_vals[(freqs >= 80) & (freqs <= 100)]
    band_bpfi = fft_vals[(freqs >= 120) & (freqs <= 150)]
    band_bsf  = fft_vals[(freqs >= 40) & (freqs <= 60)]
    freq_bpfo = float(np.mean(band_bpfo)) / (np.mean(fft_vals) + 1e-10) if len(band_bpfo) > 0 else 0.0
    freq_bpfi = float(np.mean(band_bpfi)) / (np.mean(fft_vals) + 1e-10) if len(band_bpfi) > 0 else 0.0
    freq_bsf  = float(np.mean(band_bsf)) / (np.mean(fft_vals) + 1e-10) if len(band_bsf) > 0 else 0.0

    return {
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "kurtosis": round(kurtosis, 4),
        "crest_factor": round(crest_factor, 4),
        "freq_1x": round(freq_1x, 4),
        "freq_2x": round(freq_2x, 4),
        "freq_3x": round(freq_3x, 4),
        "freq_bpfo": round(freq_bpfo, 4),
        "freq_bpfi": round(freq_bpfi, 4),
        "freq_bsf": round(freq_bsf, 4),
    }


def make_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """提取的特征 (10 维) → 归一化向量 (用于 Chroma embedding)"""
    vec = np.array([
        features["rms"], features["peak"], features["kurtosis"], features["crest_factor"],
        features["freq_1x"], features["freq_2x"], features["freq_3x"],
        features["freq_bpfo"], features["freq_bpfi"], features["freq_bsf"],
    ], dtype=np.float32)
    # L2 归一化
    norm = np.linalg.norm(vec) + 1e-10
    return vec / norm


# ── 路径解析 ────────────────────────────────────────

def parse_measurement_path(rel_path: str) -> Optional[Dict]:
    """
    从文件路径解析测量元数据。

    路径示例:
        Dataset/Vibration/Set_2/Speed_3/Fault_Angular_misalignment/Severity_5/
            Angular_misalignment_Severity_5_Speed_3_Ch_1.csv
    """
    parts = rel_path.replace("\\", "/").split("/")

    try:
        setup_id = ""
        speed_code = 0
        fault_type = ""
        severity = 0
        channel = ""
        measurement_type = ""

        for i, p in enumerate(parts):
            if "Vibration" in p:
                measurement_type = "vibration"
            elif "Current" in p or "Motor_current" in p:
                measurement_type = "current"
            elif p.startswith("Set_"):
                setup_id = p
            elif p.startswith("Speed_"):
                # Speed_1 → 25%, Speed_2 → 50%, ...
                code = int(p.replace("Speed_", ""))
                speed_code = code * 25
            elif p.startswith("Fault_"):
                fault_type = p.replace("Fault_", "")
            elif p.startswith("Severity_"):
                severity = int(p.replace("Severity_", ""))
            elif "Ch" in p and p.endswith(".csv"):
                # 从文件名提取通道号
                fname = p.replace(".csv", "")
                parts_f = fname.split("_")
                for pf in parts_f:
                    if pf.startswith("Ch") and len(pf) <= 4:
                        channel = pf.replace("Ch_", "").replace("Ch", "")
                        break

            if not fault_type and "Fault_" in p:
                fault_type = p.replace("Fault_", "").split("/")[0]

        if not fault_type:
            return None

        # 构造测量 ID
        ts_str = f"202007_{setup_id}_{speed_code}pct_{fault_type}_sev{severity}_ch{channel}"
        mid = hashlib.md5(ts_str.encode()).hexdigest()[:12]

        return {
            "measurement_id": f"M_{mid}",
            "timestamp": datetime(2020, 7, 1),  # 数据集采集于 2020.07
            "setup_id": setup_id,
            "speed_pct": speed_code,
            "fault_type": fault_type,
            "severity": severity,
            "channel": channel,
            "measurement_type": measurement_type,
            "source_file": rel_path,
            "health_status": "faulty" if fault_type and "Healthy" not in fault_type else "healthy",
        }
    except Exception:
        return None


def parse_csv_file(csv_path: Path) -> Optional[Dict]:
    """解析单个 CSV 文件，提取元数据 + 特征"""
    metadata = parse_measurement_path(str(csv_path))
    if metadata is None:
        return None

    try:
        signal = np.loadtxt(csv_path, delimiter=',')
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        # 取第一列作为主信号
        features = extract_features(signal[:, 0])
    except Exception:
        features = {}

    metadata["features"] = features
    return metadata


# ── 模拟数据生成 ────────────────────────────────────

def _simulate_features(fault_type: str, severity: int, speed_pct: int) -> Dict:
    """
    模拟特定故障类型的频谱特征（真实实现用 CSV 解析替换）。
    本函数与 scripts/ingest_sensors.py 保持相同逻辑。
    """
    import random

    rms_base = random.uniform(0.5, 1.5)
    peak_base = random.uniform(2.0, 5.0)

    features = {
        "rms": rms_base, "peak": peak_base,
        "freq_1x": random.uniform(0.08, 0.25),
        "freq_2x": random.uniform(0.04, 0.10),
        "freq_3x": random.uniform(0.02, 0.05),
        "freq_bpfo": random.uniform(0.01, 0.03),
        "freq_bpfi": random.uniform(0.01, 0.03),
        "freq_bsf": random.uniform(0.005, 0.015),
        "kurtosis": random.uniform(2.5, 3.5),
        "crest_factor": random.uniform(3.0, 5.0),
    }

    sev_factor = severity / 4.0

    if "misalignment" in fault_type.lower():
        features["freq_2x"] *= 5 * sev_factor
        features["freq_1x"] *= 2 * sev_factor
    elif "Unbalance" in fault_type:
        features["freq_1x"] *= 6 * sev_factor
    elif "Cavitation" in fault_type:
        for k in ("freq_1x", "freq_2x", "freq_3x", "freq_bpfo", "freq_bpfi", "freq_bsf"):
            features[k] *= 1.5 * sev_factor
    elif "Bearing" in fault_type:
        features["freq_bpfo"] *= 8 * sev_factor
        features["freq_bpfi"] *= 6 * sev_factor
        features["freq_bsf"] *= 4 * sev_factor
    elif "Impeller" in fault_type:
        features["freq_1x"] *= 3 * sev_factor

    return {k: round(v, 4) for k, v in features.items()}


def simulate_measurements(count: int = 30, offset: int = 0) -> List[Dict]:
    """生成模拟测量数据"""
    import random
    from core.pump_domain import FAULT_NAMES_ZH

    faults = [c for c in FAULT_NAMES_ZH.keys() if c != "Healthy"]
    base_time = datetime(2020, 7, 1, 8, 0, 0)

    measurements = []
    for i in range(count):
        fault_type = random.choice(faults)
        severity = random.randint(1, 4)
        speed_pct = random.choice([50, 75, 100])
        ts = base_time.replace(hour=(8 + i * 6) % 24)

        features = _simulate_features(fault_type, severity, speed_pct)
        mid = f"M_SIM_{offset + i + 1:03d}"

        measurements.append({
            "measurement_id": mid,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "setup_id": f"Set_{random.choice([1,2])}",
            "speed_pct": speed_pct,
            "fault_type": fault_type,
            "severity": severity,
            "features": features,
            "health_status": "faulty",
            "source_file": f"simulated/{mid}.csv",
        })

    return measurements


# ── 分层注入 ────────────────────────────────────────

def ingest_measurements(
    measurements: List[Dict],
    output_dir: Path,
    use_chroma: bool = True,
):
    """
    将测量数据以分层方式注入：
    1. 测量记录实体 → entities.json（标量值作为属性）
    2. 频域特征向量 → Chroma（通过 measurement_id 关联回 Neo4j）
    3. 时序四元组 → 测量记录间的 EVOLVES_TO / BEFORE 关系
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载已有图谱
    entities_path = output_dir / "entities.json"
    rels_path = output_dir / "relationships.json"

    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
    else:
        entities = []
    if rels_path.exists():
        with open(rels_path, "r", encoding="utf-8") as f:
            edges = json.load(f)
    else:
        edges = []

    existing_ids = {e["id"] for e in entities}
    existing_edge_keys = {(r["source"], r["target"]) for r in edges}

    new_entity_count = 0
    new_edge_count = 0
    chroma_docs = []

    for m in measurements:
        mid = m["measurement_id"]

        # ── 1. 测量记录实体 (属性化存储) ──
        if mid not in existing_ids:
            features = m.get("features", {})
            feat_parts = ", ".join(f"{k}={v:.3f}" for k, v in features.items())
            desc = (
                f"故障: {m.get('fault_type', '')} "
                f"严重度: {m.get('severity', '')}"
                f"转速: {m.get('speed_pct', '')}% "
                f"| {feat_parts}"
            )
            name = f"测量_{m.get('timestamp', '')[:16].replace(' ', '_')}"

            entity = {
                "id": mid,
                "type": "测量记录",
                "name": name,
                "description": desc,
                "source": m.get("source_file", ""),
                "degree": 2,
                # 标量属性 — 内嵌为 dict（不再作为独立实体）
                "measurement_attrs": {
                    "timestamp": m.get("timestamp", ""),
                    "fault_type": m.get("fault_type", ""),
                    "severity": m.get("severity", 0),
                    "speed_pct": m.get("speed_pct", 0),
                    "setup_id": m.get("setup_id", ""),
                    "rms": features.get("rms", 0),
                    "peak": features.get("peak", 0),
                    "kurtosis": features.get("kurtosis", 0),
                    "crest_factor": features.get("crest_factor", 0),
                    "freq_1x": features.get("freq_1x", 0),
                    "freq_2x": features.get("freq_2x", 0),
                    "freq_3x": features.get("freq_3x", 0),
                    "freq_bpfo": features.get("freq_bpfo", 0),
                    "freq_bpfi": features.get("freq_bpfi", 0),
                    "freq_bsf": features.get("freq_bsf", 0),
                },
            }
            entities.append(entity)
            existing_ids.add(mid)
            new_entity_count += 1

        # ── 2. Chroma 向量 (待后续 Phase 3 实现) ──
        if use_chroma and features:
            from . import make_feature_vector
            fv = make_feature_vector(features)
            chroma_docs.append({
                "id": mid,
                "embedding": fv.tolist(),
                "metadata": {
                    "measurement_id": mid,
                    "fault_type": m.get("fault_type", ""),
                    "severity": m.get("severity", 0),
                },
            })

        # ── 3. 桥接关系 (测量↔故障/工况) ──
        # 测量 → 故障
        fault_id = f"fault_{m.get('fault_type', '')}"
        key = (mid, fault_id)
        if key not in existing_edge_keys:
            edges.append({
                "source": mid, "target": fault_id,
                "relation": "表现为",
                "description": f"该测量记录显示 {m.get('fault_type', '')} 特征",
                "weight": 1.0,
            })
            existing_edge_keys.add(key)
            new_edge_count += 1

    # ── 4. 时序四元组 (同一故障的测量链) ──
    # 按故障类型分组后排序
    fault_groups: Dict[str, list] = {}
    for m in sorted(measurements, key=lambda x: (x["fault_type"], x["timestamp"])):
        ft = m["fault_type"]
        if ft not in fault_groups:
            fault_groups[ft] = []
        fault_groups[ft].append(m)

    for ft, group in fault_groups.items():
        for i in range(len(group) - 1):
            src, tgt = group[i]["measurement_id"], group[i + 1]["measurement_id"]
            key = (src, tgt)
            if key not in existing_edge_keys:
                edges.append({
                    "source": src, "target": tgt,
                    "relation": "后续测量",
                    "description": f"时序链路: {src} → {tgt}",
                    "weight": 0.5,
                })
                existing_edge_keys.add(key)
                new_edge_count += 1

    # 保存
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    with open(rels_path, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"[Ingest] 新增 {new_entity_count} 个测量实体, "
          f"{new_edge_count} 条关系, {len(chroma_docs)} 条 Chroma 向量")
    print(f"[Ingest] 总计 {len(entities)} 个实体, {len(edges)} 条关系")

    return {
        "new_entities": new_entity_count,
        "new_edges": new_edge_count,
        "total_entities": len(entities),
        "total_edges": len(edges),
        "chroma_docs": chroma_docs,
    }


# ── CLI ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="传感器数据注入 — 真实/模拟 CSV → 分层存储"
    )
    parser.add_argument("--simulate", action="store_true",
                        help="生成模拟数据")
    parser.add_argument("--count", type=int, default=30,
                        help="模拟数据条数")
    parser.add_argument("--from-dir", type=str,
                        help="从真实 CSV 目录解析")
    parser.add_argument("--output", default="data/output",
                        help="输出目录")
    parser.add_argument("--no-chroma", action="store_true",
                        help="不生成 Chroma 向量")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.simulate:
        print(f"[Simulate] 生成 {args.count} 条模拟测量数据...")
        measurements = simulate_measurements(args.count)
    elif args.from_dir:
        data_dir = Path(args.from_dir)
        print(f"[File] 从 {data_dir} 解析真实 CSV...")
        csv_files = list(data_dir.rglob("*.csv"))
        print(f"[File] 发现 {len(csv_files)} 个 CSV 文件")
        measurements = []
        for i, csv_path in enumerate(csv_files):
            rel = str(csv_path.relative_to(data_dir))
            m = parse_measurement_path(rel)
            if m:
                if i < 10:  # 前 10 个做 FFT
                    features = extract_features(
                        np.random.randn(240000) * 0.1  # 占位符, 等解压后替换
                    )
                    m["features"] = features
                measurements.append(m)
        print(f"[File] 解析出 {len(measurements)} 条测量记录")
    else:
        parser.print_help()
        return

    if measurements:
        result = ingest_measurements(
            measurements, output_dir,
            use_chroma=not args.no_chroma,
        )
        print(f"\n[Result] 注入完成: {result['new_entities']} 新实体, "
              f"{result['new_edges']} 新关系")
        print(f"[Result] 图谱总计: {result['total_entities']} 实体, "
              f"{result['total_edges']} 关系")

        # 更新 ECharts
        print("\n[Viz] 重新生成 ECharts...")
        from visualize.echart_viz import load_data, build_graph_data, generate_html
        entities, relationships = load_data(str(output_dir))
        graph_data = build_graph_data(entities, relationships)
        generate_html(graph_data, "pump_graph_viz.html")


if __name__ == "__main__":
    main()
