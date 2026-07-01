"""
真实传感器 CSV 接入模块 — 重写版

数据集: Motor Current and Vibration Monitoring Dataset for various
         Faults in an E-motor-driven Centrifugal Pump (v4, 2024)

实际目录结构:
    Vibration/
      Motor-2/                          # Setup 1, MG160MA
        50/ 75/ 100/                    # 三种转速 (50%/75%/100%)
          <fault_dir>/                   # e.g. "bearing bpfi 1"
            Vibration_Motor-2_<speed>_time-<fault>-ch1.csv  (ch1-ch5, 各240k采样)
      Motor-4/                          # Setup 2, MG180MB
        70/                             # 单速 70%
          <fault_dir>/                   # e.g. "align angular 1"
            Vibration_Motor-4_70_time-<fault>-ch1.csv       (ch1-ch5)

CSV 格式:
    time,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
    5e-05,0.286,0.353,...                 # 240k rows, 16 columns (1 time + 15 data)

用法:
    # 小规模测试
    python scripts/ingest_real_sensors.py --test --count 20

    # 全量摄入
    python scripts/ingest_real_sensors.py --full

    # 指定数据目录
    python scripts/ingest_real_sensors.py --data-dir <path> --output data/output
"""

import sys
import json
import hashlib
import time as _time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════
# 故障类型映射: 数据集目录名 → pump_domain 中的故障代码
# ═══════════════════════════════════════════════════════════════

MOTOR2_FAULT_MAP = {
    "bearing bpfi 1": "Bearing_BPFI",
    "bearing bpfi 2": "Bearing_BPFI",
    "bearing bpfi 3": "Bearing_BPFI",
    "bearing bpfo 1": "Bearing_BPFO",
    "bearing bpfo 2": "Bearing_BPFO",
    "bearing bpfo 3": "Bearing_BPFO",
    "bearing bsf": "Bearing_BSF",
    "bearing contaminated": "Bearing_contaminated",
    "bearing pump 1": "Pump_bearing",
    "bearing pump 2": "Pump_bearing",
    "bearing pump 3": "Pump_bearing",
    "broken rotor bar": "Broken_rotor_bar",
    "healthy 1": "Healthy",
    "healthy 2": "Healthy",
    "healthy 3": "Healthy",
    "healthy noise": "Healthy",
    "impeller 1": "Impeller_fault",
    "impeller 2": "Impeller_fault",
    "impeller 3": "Impeller_fault",
    "loose foot motor": "Loose_foot_motor",
    "loose foot pump": "Loose_foot",
    "new motor": "Healthy",
    "soft foot 1": "Soft_foot",
    "soft foot 2": "Soft_foot",
    "stator short 1": "Stator_short",
    "stator short 2": "Stator_short",
}

MOTOR4_FAULT_MAP = {
    "align angular 1": "Angular_misalignment",
    "align angular 2": "Angular_misalignment",
    "align angular 3": "Angular_misalignment",
    "align angular 4": "Angular_misalignment",
    "align angular 5": "Angular_misalignment",
    "align combination 1": "Combined_misalignment",
    "align combination 2": "Combined_misalignment",
    "align combination 3": "Combined_misalignment",
    "align combination 4": "Combined_misalignment",
    "align parallel 1": "Parallel_misalignment",
    "align parallel 2": "Parallel_misalignment",
    "align parallel 3": "Parallel_misalignment",
    "align parallel 4": "Parallel_misalignment",
    "bent shaft": "Bent_shaft",
    "cavitation discharge 1": "Cavitation_discharge",
    "cavitation discharge 2": "Cavitation_discharge",
    "cavitation discharge 3": "Cavitation_discharge",
    "cavitation discharge 4": "Cavitation_discharge",
    "cavitation discharge 5": "Cavitation_discharge",
    "cavitation suction 1": "Cavitation_suction",
    "cavitation suction 2": "Cavitation_suction",
    "cavitation suction 3": "Cavitation_suction",
    "cavitation suction 4": "Cavitation_suction",
    "coupling 1": "Coupling_damage",
    "coupling 2": "Coupling_damage",
    "coupling 2D": "Coupling_damage",
    "coupling 3": "Coupling_damage",
    "healthy 1": "Healthy",
    "healthy 2": "Healthy",
    "healthy 3": "Healthy",
    "healthy noise": "Healthy",
    "unbalance motor 1": "Unbalance_motor",
    "unbalance motor 2": "Unbalance_motor",
    "unbalance motor 3": "Unbalance_motor",
    "unbalance motor 4": "Unbalance_motor",
    "unbalance motor 5": "Unbalance_motor",
    "unbalance motor 6": "Unbalance_motor",
    "unbalance pump 1": "Unbalance_pump",
    "unbalance pump 2": "Unbalance_pump",
    "unbalance pump 3": "Unbalance_pump",
}

# 电机到 Setup 的映射
MOTOR_SETUP_MAP = {"Motor-2": "Setup_1_MG160MA", "Motor-4": "Setup_2_MG180MB"}

# 传感器通道含义
CHANNEL_MAP = {
    "ch1": "电机非驱动端轴承 水平",
    "ch2": "电机驱动端轴承 垂直",
    "ch3": "电机驱动端轴承 轴向",
    "ch4": "泵驱动端轴承 水平",
    "ch5": "泵非驱动端轴承 垂直",
}


# ═══════════════════════════════════════════════════════════════
# 信号处理
# ═══════════════════════════════════════════════════════════════

def extract_features(signal: np.ndarray, fs: float = 20000.0) -> Dict[str, float]:
    """
    从一维振动信号中提取时域 + 频域特征。

    Args:
        signal: 一维振动信号数组 (g)
        fs: 采样率 (Hz, 默认 20kHz)

    Returns: 10 维特征字典
    """
    n = len(signal)
    if n < 256:
        return {
            "rms": 0.0, "peak": 0.0, "kurtosis": 0.0, "crest_factor": 0.0,
            "freq_1x": 0.0, "freq_2x": 0.0, "freq_3x": 0.0,
            "freq_bpfo": 0.0, "freq_bpfi": 0.0, "freq_bsf": 0.0,
        }

    # ── 时域 ──
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    std_val = float(np.std(signal)) + 1e-10
    kurtosis = float(np.mean((signal - np.mean(signal)) ** 4) / (std_val ** 4))
    crest_factor = float(peak / (rms + 1e-10))

    # ── 频域 ──
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mean = float(np.mean(fft_vals)) + 1e-10

    # 1X: 20-36Hz (1480rpm/60=24.7Hz), 45-55Hz (高速)
    band_1x = fft_vals[(freqs >= 20) & (freqs <= 36)]
    band_1x_fast = fft_vals[(freqs >= 45) & (freqs <= 55)]
    freq_1x = max(
        float(np.mean(band_1x)) / fft_mean if len(band_1x) > 0 else 0.0,
        float(np.mean(band_1x_fast)) / fft_mean if len(band_1x_fast) > 0 else 0.0,
    )

    # 2X: 40-72Hz (低速) / 90-110Hz (高速)
    band_2x = fft_vals[(freqs >= 40) & (freqs <= 72)]
    band_2x_fast = fft_vals[(freqs >= 90) & (freqs <= 110)]
    freq_2x = max(
        float(np.mean(band_2x)) / fft_mean if len(band_2x) > 0 else 0.0,
        float(np.mean(band_2x_fast)) / fft_mean if len(band_2x_fast) > 0 else 0.0,
    )

    # 3X: 60-108Hz
    band_3x = fft_vals[(freqs >= 60) & (freqs <= 108)]
    freq_3x = float(np.mean(band_3x)) / fft_mean if len(band_3x) > 0 else 0.0

    # BPFO/BPFI/BSF (NU311 轴承近似)
    margin = 5.0
    bearing_bpfo_val = 89.2
    bearing_bpfi_val = 135.5
    bearing_bsf_val = 58.4

    b_bpfo = fft_vals[(freqs >= bearing_bpfo_val - margin) & (freqs <= bearing_bpfo_val + margin)]
    b_bpfi = fft_vals[(freqs >= bearing_bpfi_val - margin) & (freqs <= bearing_bpfi_val + margin)]
    b_bsf  = fft_vals[(freqs >= bearing_bsf_val - margin) & (freqs <= bearing_bsf_val + margin)]

    freq_bpfo = float(np.mean(b_bpfo)) / fft_mean if len(b_bpfo) > 0 else 0.0
    freq_bpfi = float(np.mean(b_bpfi)) / fft_mean if len(b_bpfi) > 0 else 0.0
    freq_bsf  = float(np.mean(b_bsf)) / fft_mean if len(b_bsf) > 0 else 0.0

    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "kurtosis": round(kurtosis, 6),
        "crest_factor": round(crest_factor, 6),
        "freq_1x": round(freq_1x, 6),
        "freq_2x": round(freq_2x, 6),
        "freq_3x": round(freq_3x, 6),
        "freq_bpfo": round(freq_bpfo, 6),
        "freq_bpfi": round(freq_bpfi, 6),
        "freq_bsf": round(freq_bsf, 6),
    }


def make_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """10 维特征 → L2 归一化向量 (用于 Chroma embedding)"""
    vec = np.array([
        features["rms"], features["peak"], features["kurtosis"], features["crest_factor"],
        features["freq_1x"], features["freq_2x"], features["freq_3x"],
        features["freq_bpfo"], features["freq_bpfi"], features["freq_bsf"],
    ], dtype=np.float32)
    norm = np.linalg.norm(vec) + 1e-10
    return vec / norm


# ═══════════════════════════════════════════════════════════════
# 路径解析 — 适配真实数据集结构
# ═══════════════════════════════════════════════════════════════

def parse_measurement_path(rel_path: str) -> Optional[Dict]:
    """
    从真实数据集文件路径解析元数据。

    路径示例 (Motor-2):
        Vibration/Motor-2/100/bearing bpfi 1/Vibration_Motor-2_100_time-bearing bpfi 1-ch1.csv

    路径示例 (Motor-4):
        Vibration/Motor-4/70/align angular 1/Vibration_Motor-4_70_time-align angular 1-ch2.csv
    """
    parts = rel_path.replace("\\", "/").split("/")

    # 找到 motor 标识
    motor_id = None
    for p in parts:
        if p.startswith("Motor-"):
            motor_id = p
            break

    if motor_id is None:
        return None

    try:
        # 找 motor_id 之后的元素
        motor_idx = parts.index(motor_id) if motor_id in parts else -1
        if motor_idx < 0 or motor_idx + 2 >= len(parts):
            return None

        speed_str = parts[motor_idx + 1]   # "50"/"75"/"100"/"70"
        fault_dir = parts[motor_idx + 2]    # "bearing bpfi 1" / "align angular 2"

        speed_pct = int(speed_str)
        setup_id = MOTOR_SETUP_MAP.get(motor_id, f"Setup_{motor_id}")
        measurement_type = "vibration"

        # 根据 motor_id 选择故障映射
        if motor_id == "Motor-2":
            fault_type = MOTOR2_FAULT_MAP.get(fault_dir.lower())
        else:
            fault_type = MOTOR4_FAULT_MAP.get(fault_dir.lower())

        if fault_type is None:
            # 尝试模糊匹配
            fault_type = _fuzzy_fault_match(fault_dir, motor_id)

        # 从文件名提取通道号
        filename = parts[-1] if parts[-1].endswith(".csv") else ""
        channel = _extract_channel(filename)

        # 严重度 — 从 fault_dir 末尾的数字提取
        severity = _extract_severity(fault_dir)

        # 构造稳定的测量 ID
        ts_str = f"{setup_id}_{speed_pct}pct_{fault_dir}_{channel}"
        mid = "M_" + hashlib.md5(ts_str.encode()).hexdigest()[:12]

        return {
            "measurement_id": mid,
            "timestamp": "2020-07-01T00:00:00",
            "setup_id": setup_id,
            "motor_id": motor_id,
            "speed_pct": speed_pct,
            "fault_type": fault_type or "Unknown",
            "fault_dir": fault_dir,
            "severity": severity,
            "channel": channel,
            "channel_desc": CHANNEL_MAP.get(channel, ""),
            "measurement_type": measurement_type,
            "source_file": rel_path,
            "health_status": "healthy" if ("healthy" in fault_dir.lower() or fault_type == "Healthy") else "faulty",
        }
    except Exception as e:
        return None


def _extract_channel(filename: str) -> str:
    """从文件名提取通道号: '...ch3.csv' → 'ch3'"""
    import re
    match = re.search(r'ch(\d)', filename.lower())
    return f"ch{match.group(1)}" if match else ""


def _extract_severity(fault_dir: str) -> int:
    """从故障目录名末尾提取严重度: 'bearing bpfi 3'→3, 'soft foot 2'→2"""
    parts = fault_dir.strip().split()
    last = parts[-1]
    try:
        return int(last)
    except ValueError:
        # 检查倒数第二位
        if len(parts) >= 2:
            try:
                return int(parts[-1])
            except ValueError:
                pass
    return 1


def _fuzzy_fault_match(fault_dir: str, motor_id: str) -> Optional[str]:
    """模糊故障类型匹配"""
    from core.pump_domain import FAULT_NAMES_ZH

    fmap = MOTOR2_FAULT_MAP if motor_id == "Motor-2" else MOTOR4_FAULT_MAP
    # 直接匹配
    result = fmap.get(fault_dir.lower())
    if result:
        return result

    # 去掉末尾数字再匹配
    import re
    base = re.sub(r'\s*\d+$', '', fault_dir.lower()).strip()
    for k, v in fmap.items():
        k_base = re.sub(r'\s*\d+$', '', k.lower()).strip()
        if k_base == base:
            return v

    return None


# ═══════════════════════════════════════════════════════════════
# CSV 读取
# ═══════════════════════════════════════════════════════════════

def read_csv_data(csv_path: Path) -> Optional[np.ndarray]:
    """
    读取真实数据集 CSV 文件。

    CSV 格式: time,0,1,2,...,14  (第一行 header, 第一列时间)
    使用 numpy genfromtxt 处理 header。
    """
    try:
        # 跳过 header, 读取全部数值列
        data = np.genfromtxt(str(csv_path), delimiter=',', skip_header=1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    except Exception as e:
        return None


def process_csv_file(csv_path: Path, data_dir: Path, do_fft: bool = True) -> Optional[Dict]:
    """
    处理单个 CSV 文件: 解析路径→读取数据→提取特征。

    Args:
        csv_path: CSV 文件绝对路径
        data_dir: 数据集根目录 (用于计算相对路径)
        do_fft: 是否执行 FFT 特征提取（false 时仅返回元数据）

    Returns:
        包含 metadata + features 的字典, 或 None
    """
    try:
        rel_path = str(csv_path.relative_to(data_dir))
    except ValueError:
        rel_path = str(csv_path)

    metadata = parse_measurement_path(rel_path)
    if metadata is None:
        return None

    # 读取并提取特征
    if do_fft:
        data = read_csv_data(csv_path)
        if data is not None and data.shape[0] > 0:
            # 取第一列数据通道 (跳过时间列) 作为主信号
            # 列0 = time, 列1+ = 振动数据
            if data.shape[1] >= 2:
                signal = data[:, 1]  # 第一数据列
            else:
                signal = data[:, 0]
            features = extract_features(signal)
            metadata["features"] = features
        else:
            metadata["features"] = {}
    else:
        metadata["features"] = {}

    return metadata


# ═══════════════════════════════════════════════════════════════
# 分层注入
# ═══════════════════════════════════════════════════════════════

def ingest_measurements(
    measurements: List[Dict],
    output_dir: Path,
    use_chroma: bool = True,
    avoid_duplicates: bool = True,
):
    """
    将测量数据以三层分层方式注入:

    Layer 1: 测量记录实体 → entities.json (标量值作为属性)
    Layer 2: 频域特征向量 → Chroma / measurements.json
    Layer 3: 时序四元组 → 测量记录间的 后续测量 关系
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

    # 构建已有 ID 集合
    if avoid_duplicates:
        existing_entity_ids = {e["id"] for e in entities}
        existing_edge_keys = {(r["source"], r["target"]) for r in edges}
    else:
        existing_entity_ids = set()
        existing_edge_keys = set()

    new_entity_count = 0
    new_edge_count = 0
    chroma_docs = []

    for m in measurements:
        mid = m["measurement_id"]
        if not mid or (avoid_duplicates and mid in existing_entity_ids):
            continue

        features = m.get("features", {})

        # ── Layer 1: 测量记录实体 ──
        fault_zh = m.get("fault_type", "")
        setup_name = m.get("setup_id", "")
        ch_desc = m.get("channel_desc", m.get("channel", ""))

        entity_name = (
            f"{fault_zh}_{m['speed_pct']}%_{m.get('channel', '')}_"
            f"sev{m.get('severity', 0)}"
        )

        feat_parts = []
        for k, v in sorted(features.items()):
            if isinstance(v, (int, float)):
                feat_parts.append(f"{k}={v:.4f}")
        feat_summary = ", ".join(feat_parts[:6])  # 前6项

        desc = (
            f"[{setup_name}] {m.get('motor_id', '')} | "
            f"故障: {fault_zh} | 严重度: {m.get('severity', 0)} | "
            f"转速: {m['speed_pct']}% | 通道: {ch_desc}\n"
            f"特征: {feat_summary}"
        )

        entity = {
            "id": mid,
            "type": "测量记录",
            "name": entity_name,
            "description": desc[:500],
            "source": m.get("source_file", ""),
            "degree": 2,
            "measurement_attrs": {
                "timestamp": m.get("timestamp", ""),
                "fault_type": fault_zh,
                "severity": m.get("severity", 0),
                "speed_pct": m.get("speed_pct", 0),
                "setup_id": m.get("setup_id", ""),
                "motor_id": m.get("motor_id", ""),
                "channel": m.get("channel", ""),
                "channel_desc": ch_desc,
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
        existing_entity_ids.add(mid)
        new_entity_count += 1

        # ── Layer 2: Chroma 向量 ──
        if use_chroma and features:
            fv = make_feature_vector(features)
            chroma_docs.append({
                "id": mid,
                "embedding": fv.tolist(),
                "metadata": {
                    "measurement_id": mid,
                    "fault_type": fault_zh,
                    "severity": m.get("severity", 0),
                    "channel": m.get("channel", ""),
                },
            })

        # ── Layer 3: 桥接关系 (测量 → 故障) ──
        fault_id = f"fault_{fault_zh}"
        key = (mid, fault_id)
        if key not in existing_edge_keys:
            edges.append({
                "source": mid, "target": fault_id,
                "relation": "表现为",
                "description": f"测量记录显示 {fault_zh} 特征 (严重度{m.get('severity', 0)}, {m['speed_pct']}%转速)",
                "weight": 1.0,
            })
            existing_edge_keys.add(key)
            new_edge_count += 1

    # ── Layer 3b: 时序四元组 (同一故障+通道的链) ──
    # 按 (fault_type, channel) 分组, 组内按测量ID排序
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for m in measurements:
        key = (m.get("fault_type", ""), m.get("channel", ""), m.get("speed_pct", 0))
        groups[key].append(m)

    for key, group in groups.items():
        group.sort(key=lambda x: x["measurement_id"])
        for i in range(len(group) - 1):
            src_id = group[i]["measurement_id"]
            tgt_id = group[i + 1]["measurement_id"]
            ekey = (src_id, tgt_id)
            if src_id in existing_entity_ids and tgt_id in existing_entity_ids:
                if ekey not in existing_edge_keys:
                    edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "relation": "后续测量",
                        "description": f"时序: {key[0]}@{key[1]} 连续测量",
                        "weight": 0.5,
                    })
                    existing_edge_keys.add(ekey)
                    new_edge_count += 1

    # 保存
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    with open(rels_path, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"[Ingest] 新增 {new_entity_count} 测量实体, {new_edge_count} 关系, "
          f"{len(chroma_docs)} Chroma 向量")
    print(f"[Ingest] 图谱总计: {len(entities)} 实体, {len(edges)} 关系")

    return {
        "new_entities": new_entity_count,
        "new_edges": new_edge_count,
        "total_entities": len(entities),
        "total_edges": len(edges),
        "chroma_docs": chroma_docs,
    }


# ═══════════════════════════════════════════════════════════════
# Chroma 向量注入
# ═══════════════════════════════════════════════════════════════

def ingest_to_chroma(chroma_docs: List[Dict], persist_dir: str = "data/output/vector_store"):
    """将特征向量写入 Chroma measurements collection"""
    from core.vector_store import VectorStore

    vs = VectorStore(persist_dir=persist_dir)
    if not chroma_docs:
        return 0

    embeddings = [d["embedding"] for d in chroma_docs]
    metadatas = [d["metadata"] for d in chroma_docs]
    ids = [d["id"] for d in chroma_docs]

    # 分批添加 (Chroma 有批量限制)
    batch_size = 100
    total = 0
    for i in range(0, len(embeddings), batch_size):
        batch_emb = embeddings[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vs.add_measurements(batch_emb, batch_meta, batch_ids)
        total += len(batch_ids)

    print(f"[Chroma] {total} 条向量已写入 measurements collection")
    return total


def ingest_to_kuzu(entities_path: str, edges_path: str):
    """将图谱导入 Kùzu 嵌入式图数据库"""
    from core.graph_store import KuzuGraphStore, import_from_json

    store = KuzuGraphStore("data/output/kuzu_db")
    import_from_json(store, entities_path, edges_path)
    count_e = store.count_entities()
    count_r = store.count_edges()
    store.close()
    print(f"[Kùzu] {count_e} 实体, {count_r} 关系")
    return count_e, count_r


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="真实传感器 CSV 数据接入 — 离心泵故障数据集"
    )
    parser.add_argument("--data-dir", type=str,
                        default="D:/MyWork/TKG_Data/Motor Current and Vibration Monitoring Dataset for various Faults in an E-motor-driven Centrifugal Pump/Dataset/Dataset",
                        help="数据集根目录")
    parser.add_argument("--output", default="data/output", help="输出目录")
    parser.add_argument("--test", action="store_true",
                        help="小规模测试 (只处理少量CSV)")
    parser.add_argument("--count", type=int, default=30,
                        help="测试模式下的CSV数量")
    parser.add_argument("--full", action="store_true",
                        help="全量摄入 (所有CSV)")
    parser.add_argument("--no-chroma", action="store_true",
                        help="不写入 Chroma")
    parser.add_argument("--no-kuzu", action="store_true",
                        help="不导入 Kùzu")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="分批处理大小")
    parser.add_argument("--skip-fft", action="store_true",
                        help="跳过FFT特征提取 (快但无频域特征)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)

    if not data_dir.exists():
        print(f"[Error] 数据目录不存在: {data_dir}")
        return 1

    # 收集 CSV 文件
    vibration_dir = data_dir / "Vibration"
    if vibration_dir.exists():
        csv_files = sorted(vibration_dir.rglob("*.csv"))
    else:
        csv_files = sorted(data_dir.rglob("*.csv"))

    if args.test:
        csv_files = csv_files[:args.count]
        print(f"[Test] 测试模式: 取前 {len(csv_files)} 个 CSV 文件")
    elif not args.full:
        parser.print_help()
        return 0

    print(f"[Scan] 发现 {len(csv_files)} 个 CSV 文件 (data_dir={data_dir})")

    # 分批处理
    t_start = _time.time()
    all_measurements = []
    errors = 0
    sv_count = 0

    for i, csv_path in enumerate(csv_files):
        m = process_csv_file(csv_path, data_dir, do_fft=not args.skip_fft)
        if m is not None:
            all_measurements.append(m)
        else:
            errors += 1
        sv_count += 1

        if (i + 1) % 50 == 0 or i == len(csv_files) - 1:
            elapsed = _time.time() - t_start
            rate = sv_count / elapsed if elapsed > 0 else 0
            print(f"  进度: {i + 1}/{len(csv_files)} ({sv_count}条有效, {errors}个失败, "
                  f"{rate:.1f} 文件/秒)")

    elapsed = _time.time() - t_start
    print(f"[Scan] 完成: {len(all_measurements)}/{sv_count} 条记录, "
          f"{errors} 个失败 ({elapsed:.1f}s)")

    if not all_measurements:
        print("[Error] 没有解析出有效测量记录")
        return 1

    # 分层注入
    print(f"\n[Ingest] 开始分层注入 {len(all_measurements)} 条记录...")
    result = ingest_measurements(
        all_measurements, output_dir,
        use_chroma=not args.no_chroma,
        avoid_duplicates=True,
    )

    # Chroma
    if not args.no_chroma and result["chroma_docs"]:
        ingest_to_chroma(result["chroma_docs"])

    # Kùzu
    if not args.no_kuzu:
        entities_path = str(output_dir / "entities.json")
        edges_path = str(output_dir / "relationships.json")
        ingest_to_kuzu(entities_path, edges_path)

    # 重新生成 ECharts
    try:
        from visualize.echart_viz import load_data, build_graph_data, generate_html
        entities, relationships = load_data(str(output_dir))
        graph_data = build_graph_data(entities, relationships)
        generate_html(graph_data, "pump_graph_viz.html")
        print("[Viz] ECharts 可视化已更新: pump_graph_viz.html")
    except Exception as e:
        print(f"[Viz] ECharts 生成失败: {e}")

    # 统计摘要
    print(f"\n{'='*60}")
    print(f"  传感器数据接入完成")
    print(f"{'='*60}")
    print(f"  处理文件: {len(csv_files)}")
    print(f"  有效记录: {len(all_measurements)}")
    print(f"  新增实体: {result['new_entities']}  (测量记录)")
    print(f"  新增关系: {result['new_edges']}  (表现为 + 后续测量)")
    print(f"  图谱总计: {result['total_entities']} 实体, {result['total_edges']} 关系")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
