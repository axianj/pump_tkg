"""
传感器-图谱桥接模块

将传感器 CSV 特征向量以三层方式注入系统:
1. 标量值（rms/peak/kurtosis）→ 测量记录实体属性（Neo4j）
2. 频域特征向量 → Chroma embedding（相似度检索）
3. 跨记录时序推理 → 时序四元组（EVOLVES_TO 等）
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.temporal.temporal_quad import TemporalQuad, TemporalRelation


class SensorBridge:
    """
    传感器 CSV → 三层注入

    使用方式:
        bridge = SensorBridge()
        for csv_file in sensors:
            features = bridge.extract_features(csv_file)
            mid = bridge.ingest_measurement(features, metadata)
            bridge.build_temporal_chains(by_fault_type)

        # 阈值事件
        events = bridge.detect_threshold_events("Ch2", hours=24, threshold_rms=3.0)
    """

    def __init__(self):
        self._measurements: List[Dict] = []
        self._quads: List[TemporalQuad] = []

    # ── Layer 1: 标量值 → 实体属性 ───────────────────

    def ingest_measurement(
        self,
        metadata: Dict,
        features: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        注入一次测量记录。

        Args:
            metadata: {measurement_id, timestamp, fault_type, severity, speed_pct, ...}
            features: {rms, peak, kurtosis, freq_1x, freq_2x, ...}

        Returns:
            measurement_id — 作为 Neo4j 实体 ID 和 Chroma doc ID
        """
        mid = metadata.get("measurement_id", "")
        ts = metadata.get("timestamp", datetime.now())
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        features = features or {}
        fault_type = metadata.get("fault_type", "")

        # 构建 Neo4j 实体属性（标量值内嵌）
        entity_attrs = {
            "timestamp": ts.isoformat(),
            "fault_type": fault_type,
            "severity": metadata.get("severity", 0),
            "speed_pct": metadata.get("speed_pct", 0),
            "setup_id": metadata.get("setup_id", ""),
            "channel": metadata.get("channel", ""),
            "measurement_type": metadata.get("measurement_type", "vibration"),
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
            "source_file": metadata.get("source_file", ""),
        }

        record = {
            "measurement_id": mid,
            "timestamp": ts,
            "fault_type": fault_type,
            "entity_attrs": entity_attrs,
            "features": features,
        }
        self._measurements.append(record)

        return mid

    def get_entity_attrs(self, measurement_id: str) -> Optional[Dict]:
        """获取测量记录的 Neo4j 实体属性"""
        for m in self._measurements:
            if m["measurement_id"] == measurement_id:
                return m["entity_attrs"]
        return None

    # ── Layer 2: 频域向量 → Chroma ──────────────────

    def generate_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        10 维频域特征 → L2 归一化向量

        用于 Chroma embedding 的相似度检索。
        """
        vec = np.array([
            features.get("rms", 0),
            features.get("peak", 0),
            features.get("kurtosis", 0),
            features.get("crest_factor", 0),
            features.get("freq_1x", 0),
            features.get("freq_2x", 0),
            features.get("freq_3x", 0),
            features.get("freq_bpfo", 0),
            features.get("freq_bpfi", 0),
            features.get("freq_bsf", 0),
        ], dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-10
        return vec / norm

    def to_chroma_documents(self) -> List[Dict]:
        """
        将所有测量记录转换为 Chroma 文档格式。

        Returns:
            [{
                "id": measurement_id,
                "embedding": [10 维浮点数组],
                "metadata": {"measurement_id", "fault_type", "severity"},
            }]
        """
        docs = []
        for m in self._measurements:
            fv = self.generate_feature_vector(m.get("features", {}))
            docs.append({
                "id": m["measurement_id"],
                "embedding": fv.tolist(),
                "metadata": {
                    "measurement_id": m["measurement_id"],
                    "fault_type": m["fault_type"],
                    "severity": m["entity_attrs"].get("severity", 0),
                    "timestamp": m["entity_attrs"].get("timestamp", ""),
                },
            })
        return docs

    def find_similar_faults(
        self, target_features: Dict[str, float], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        找与目标特征最相似的历史测量记录（基于余弦相似度）。

        Phase 3 完整实现将使用 Chroma 向量检索替代这里的暴力计算。

        Returns:
            [(measurement_id, similarity), ...]
        """
        target_vec = self.generate_feature_vector(target_features)
        similarities = []
        for m in self._measurements:
            fv = self.generate_feature_vector(m.get("features", {}))
            sim = float(np.dot(target_vec, fv))
            similarities.append((m["measurement_id"], sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    # ── Layer 3: 时序四元组 ─────────────────────────

    def build_temporal_chains(
        self,
        group_key: str = "fault_type",
    ) -> List[TemporalQuad]:
        """
        在同组（同一故障类型）测量记录之间构建 EVOLVES_TO 时序链。

        Args:
            group_key: 分组键（默认按 fault_type，也可按 channel + fault_type）

        Returns:
            时序四元组列表
        """
        # 分组并排序
        groups: Dict[str, list] = {}
        for m in self._measurements:
            key = str(m.get(group_key, "default"))
            if key not in groups:
                groups[key] = []
            groups[key].append(m)

        quads = []
        for key, group in groups.items():
            group.sort(key=lambda x: x["timestamp"])
            for i in range(len(group) - 1):
                a = group[i]
                b = group[i + 1]

                # 判断关系：特征恶化 → EVOLVES_TO
                # 简单规则: 如果 rms 升高超过 30% 则为恶化
                a_rms = a["entity_attrs"].get("rms", 0)
                b_rms = b["entity_attrs"].get("rms", 0)
                if b_rms > a_rms * 1.3:
                    rel = TemporalRelation.EVOLVES_TO
                    desc = f"恶化: rms {a_rms:.3f}→{b_rms:.3f} ({a['fault_type']})"
                else:
                    rel = TemporalRelation.BEFORE
                    desc = f"时序链: {a['measurement_id']} 之后 {b['measurement_id']}"

                quad = TemporalQuad(
                    head_entity=a["measurement_id"],
                    relation=rel,
                    tail_entity=b["measurement_id"],
                    valid_from=a["timestamp"],
                    valid_to=b["timestamp"],
                    confidence=0.8 if rel == TemporalRelation.EVOLVES_TO else 1.0,
                    source=f"sensor_bridge.{group_key}={key}",
                )
                quads.append(quad)

        self._quads = quads
        return quads

    def get_quads(self) -> List[TemporalQuad]:
        return self._quads

    # ── 阈值事件检测 ────────────────────────────────

    def detect_threshold_events(
        self,
        channel: Optional[str] = None,
        time_window_hours: int = 24,
        threshold_rms: float = 3.0,
    ) -> List[Dict]:
        """
        检测阈值跨越事件。

        Args:
            channel: 传感器通道过滤（Ch1~Ch5），None=所有通道
            time_window_hours: 时间窗口
            threshold_rms: RMS 报警阈值

        Returns:
            [{measurement_id, timestamp, rms_value, exceeded_threshold, ...}]
        """
        events = []
        prev_rms = {}
        for m in sorted(self._measurements, key=lambda x: x["timestamp"]):
            ch = m["entity_attrs"].get("channel", "")
            if channel and ch != channel:
                continue
            rms = m["entity_attrs"].get("rms", 0)
            mid = m["measurement_id"]
            ts = m["timestamp"]

            # 阈值穿越事件
            if ch in prev_rms:
                if prev_rms[ch] < threshold_rms <= rms:
                    events.append({
                        "measurement_id": mid,
                        "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else ts,
                        "event_type": "threshold_crossed",
                        "threshold": threshold_rms,
                        "rms_value": rms,
                        "prev_rms": prev_rms[ch],
                        "channel": ch,
                    })

                # 快速恶化事件 (>50% 增长)
                if rms > prev_rms[ch] * 1.5:
                    events.append({
                        "measurement_id": mid,
                        "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else ts,
                        "event_type": "rapid_degradation",
                        "rms_value": rms,
                        "prev_rms": prev_rms[ch],
                        "change_pct": (rms - prev_rms[ch]) / (prev_rms[ch] + 1e-10) * 100,
                        "channel": ch,
                    })

            prev_rms[ch] = rms

        return events

    # ── 统计 ────────────────────────────────────────

    @property
    def measurement_count(self) -> int:
        return len(self._measurements)

    @property
    def quad_count(self) -> int:
        return len(self._quads)

    def summary(self) -> Dict:
        """桥接模块统计摘要"""
        from collections import Counter
        fault_counter = Counter(m.get("fault_type", "") for m in self._measurements)

        return {
            "total_measurements": len(self._measurements),
            "total_temporal_quads": len(self._quads),
            "by_fault_type": dict(fault_counter.most_common()),
            "time_range": (
                (min(m["timestamp"] for m in self._measurements),
                 max(m["timestamp"] for m in self._measurements))
                if self._measurements else (None, None)
            ),
        }


# ── 信号处理工具 ────────────────────────────────────

def extract_signal_features(
    signal: np.ndarray,
    fs: float = 20000.0,
    bearing_bpfo: float = 89.2,
    bearing_bpfi: float = 135.5,
    bearing_bsf: float = 58.4,
) -> Dict[str, float]:
    """
    从一维振动信号中提取特征。

    Args:
        signal: 原始振动信号
        fs: 采样率 (Hz)
        bearing_bpfo / bearing_bpfi / bearing_bsf: 轴承故障特征频率 (Hz)
            默认值为 NU311 轴承近似值，实际需要从数据表中查找

    Returns:
        时域 + 频域特征字典
    """
    n = len(signal)
    if n < 256:
        return {}

    # ── 时域 ──
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    mean_val = np.mean(signal)
    std_val = np.std(signal) + 1e-10
    kurtosis = float(np.mean((signal - mean_val) ** 4) / (std_val ** 4))
    crest_factor = float(peak / (rms + 1e-10))

    # ── 频域 ──
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mean = np.mean(fft_vals) + 1e-10

    # 1X: 22-36Hz (1480rpm/60=24.7Hz, 2960rpm/60=49.3Hz)
    band_1x = fft_vals[(freqs >= 20) & (freqs <= 36)]
    band_1x_fast = fft_vals[(freqs >= 45) & (freqs <= 55)]
    # 2X: 44-72Hz (对于低速) 或 90-110Hz (对于高速)
    band_2x = fft_vals[(freqs >= 40) & (freqs <= 72)]
    band_2x_fast = fft_vals[(freqs >= 90) & (freqs <= 110)]
    # 3X
    band_3x = fft_vals[(freqs >= 60) & (freqs <= 108)]

    freq_1x = max(
        float(np.mean(band_1x)) / fft_mean if len(band_1x) > 0 else 0.0,
        float(np.mean(band_1x_fast)) / fft_mean if len(band_1x_fast) > 0 else 0.0,
    )
    freq_2x = max(
        float(np.mean(band_2x)) / fft_mean if len(band_2x) > 0 else 0.0,
        float(np.mean(band_2x_fast)) / fft_mean if len(band_2x_fast) > 0 else 0.0,
    )
    freq_3x = float(np.mean(band_3x)) / fft_mean if len(band_3x) > 0 else 0.0

    # BPFO/BPFI/BSF
    margin = 5.0
    b_bpfo = fft_vals[(freqs >= bearing_bpfo - margin) & (freqs <= bearing_bpfo + margin)]
    b_bpfi = fft_vals[(freqs >= bearing_bpfi - margin) & (freqs <= bearing_bpfi + margin)]
    b_bsf  = fft_vals[(freqs >= bearing_bsf - margin) & (freqs <= bearing_bsf + margin)]

    return {
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "kurtosis": round(kurtosis, 4),
        "crest_factor": round(crest_factor, 4),
        "freq_1x": round(freq_1x, 4),
        "freq_2x": round(freq_2x, 4),
        "freq_3x": round(freq_3x, 4),
        "freq_bpfo": round(float(np.mean(b_bpfo)) / fft_mean if len(b_bpfo) > 0 else 0.0, 4),
        "freq_bpfi": round(float(np.mean(b_bpfi)) / fft_mean if len(b_bpfi) > 0 else 0.0, 4),
        "freq_bsf": round(float(np.mean(b_bsf)) / fft_mean if len(b_bsf) > 0 else 0.0, 4),
    }
