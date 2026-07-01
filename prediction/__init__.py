"""
预测模块接口 — 传感器信号编码 + TKG图嵌入 + 融合预测

Phase 6 定义完整的数据管道接口，不立即实现 DL 模型。

Schema:
    sensor_signal (ndarray) → SignalEncoder → feature_vector (10维)
    temporal_graph (quadruples) → TKGEncoder → graph_embedding (64维)
    [feature_vector + graph_embedding] → FusionPredictor → fault_predictions

基线模型候选池:
    翻译模型: TTansE, HyTE
    GNN: EvolveGCN / RE-NET / TGN (GNN主模型待调研选定)
    点过程: Know-Evolve
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Signal Encoder ──────────────────────────────────

class SignalEncoder:
    """
    传感器信号 → 特征向量

    输入: 原始振动信号 (240,000 采样点 × 5 通道)
    输出: 10 维归一化频域特征向量
    """

    FEATURE_DIM = 10

    FEATURE_NAMES = [
        "rms", "peak", "kurtosis", "crest_factor",
        "freq_1x", "freq_2x", "freq_3x",
        "freq_bpfo", "freq_bpfi", "freq_bsf",
    ]

    def __init__(self, fs: float = 20000.0):
        self.fs = fs

    def encode(self, signal: np.ndarray, channel_mask: Optional[List[int]] = None) -> np.ndarray:
        """
        将多通道振动信号编码为特征向量。

        Args:
            signal: (n_samples, n_channels) 或 (n_samples,) 的原始信号
            channel_mask: 需要使用的通道索引列表，None = 所有通道
        """
        from core.temporal.sensor_bridge import extract_signal_features

        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        n_channels = signal.shape[1]
        if channel_mask is None:
            channel_mask = list(range(n_channels))

        all_features = []
        for ch in channel_mask:
            if ch < n_channels:
                sig_ch = signal[:, ch]
                features = extract_signal_features(sig_ch, fs=self.fs)
                all_features.append(features)

        if not all_features:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # 多通道特征平均
        avg_features = {}
        for feat_name in self.FEATURE_NAMES:
            vals = [f.get(feat_name, 0.0) for f in all_features]
            avg_features[feat_name] = float(np.mean(vals))

        vec = np.array([avg_features[n] for n in self.FEATURE_NAMES], dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-10
        return vec / norm

    def encode_batch(self, signals: List[np.ndarray]) -> np.ndarray:
        """批量编码 → (n_signals, FEATURE_DIM)"""
        return np.array([self.encode(s) for s in signals], dtype=np.float32)

    def to_dict(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """特征向量 → 可解释字典"""
        return {
            name: float(feature_vector[i])
            for i, name in enumerate(self.FEATURE_NAMES)
        }


# ── TKG Encoder ─────────────────────────────────────

class TKGEncoder:
    """
    时序知识图谱 → 图嵌入向量

    将时序四元组集合编码为固定长度的图嵌入。
    当前实现使用统计聚合（度分布、关系类型分布等），
    Phase 5+ 将对接真正的图嵌入模型（TTansE / EvolveGCN 等）。

    Output: 64 维 graph embedding
    """

    EMBEDDING_DIM = 64

    def __init__(self):
        self._entity_vocab: Dict[str, int] = {}
        self._relation_vocab: Dict[str, int] = {}

    def encode_graph_structure(
        self,
        entities: List[Dict],
        edges: List[Dict],
    ) -> np.ndarray:
        """
        从实体和关系列表编码图结构。

        Returns: 64 维 embedding（当前用统计聚合，后续替换为模型推理）
        """
        # ── 统计特征（占位实现） ──
        n_entities = len(entities)
        n_edges = len(edges)

        # 节点度分布特征
        degrees = {}
        for r in edges:
            src = r.get("source", "")
            tgt = r.get("target", "")
            degrees[src] = degrees.get(src, 0) + 1
            degrees.get(tgt, 0)  # 确保 tgt 在 dict 中

        deg_values = list(degrees.values()) if degrees else [0]
        deg_mean = np.mean(deg_values)
        deg_std = np.std(deg_values)
        deg_max = max(deg_values)

        # 关系类型分布
        from collections import Counter
        rel_counter = Counter(r.get("relation", "unknown") for r in edges)
        n_rel_types = len(rel_counter)

        # 实体类型分布
        type_counter = Counter(e.get("type", "unknown") for e in entities)
        n_entity_types = len(type_counter)

        # 组装 embedding
        embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        embedding[0] = float(np.log1p(n_entities))
        embedding[1] = float(np.log1p(n_edges))
        embedding[2] = float(deg_mean) / (deg_max + 1)
        embedding[3] = float(deg_std) / (deg_max + 1)
        embedding[4] = float(n_rel_types) / 20.0
        embedding[5] = float(n_entity_types) / 10.0

        # 归一化
        norm = np.linalg.norm(embedding) + 1e-10
        return embedding / norm

    def encode_temporal_quads(
        self,
        quads: List,
    ) -> np.ndarray:
        """
        从时序四元组列表编码。

        当前用 TemporalQuadStore 做结构统计，
        后续替换为 TTansE / EvolveGCN 的前向推理。
        """
        # 提取时序四元组的统计结构
        n_quads = len(quads)
        entities_in_quads = set()
        rel_types_in_quads = set()
        for q in quads:
            entities_in_quads.add(q.head_entity)
            entities_in_quads.add(q.tail_entity)
            rel_types_in_quads.add(q.relation.value)

        embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        embedding[0] = float(np.log1p(n_quads))
        embedding[1] = float(np.log1p(len(entities_in_quads)))
        embedding[2] = float(len(rel_types_in_quads)) / 8.0

        norm = np.linalg.norm(embedding) + 1e-10
        return embedding / norm


# ── Fusion Predictor ────────────────────────────────

class FusionPredictor:
    """
    融合预测器 — 特征向量 + 图嵌入 → 故障预测

    当前为接口定义，具体模型将在 Phase 5+ 实现。
    支持的模型接口:
    - TTansE / HyTE: 时序翻译模型
    - EvolveGCN / RE-NET / TGN: 图神经网络
    - Know-Evolve: 点过程模型
    """

    # 故障类型 → 索引映射
    FAULT_CLASSES = [
        "Angular_misalignment", "Parallel_misalignment", "Combined_misalignment",
        "Unbalance_motor", "Unbalance_pump",
        "Coupling_damage", "Cavitation_suction", "Cavitation_discharge",
        "Bent_shaft", "Impeller_fault",
        "Bearing_BPFO", "Bearing_BPFI", "Bearing_contaminated",
        "Bearing_BSF", "Soft_foot", "Loose_foot_motor", "Loose_foot_pump",
        "Broken_rotor_bar", "Stator_short", "Pump_bearing",
    ]

    def __init__(self, model=None):
        self._model = model
        self._fault_to_idx = {f: i for i, f in enumerate(self.FAULT_CLASSES)}
        self._idx_to_fault = {i: f for i, f in enumerate(self.FAULT_CLASSES)}

    @property
    def n_classes(self) -> int:
        return len(self.FAULT_CLASSES)

    def predict(
        self,
        feature_vector: np.ndarray,
        graph_embedding: np.ndarray,
    ) -> Dict:
        """
        故障预测主接口。

        Args:
            feature_vector: (10,) 传感器频域特征
            graph_embedding: (64,) 时序图谱嵌入

        Returns:
            {
                "top_faults": [(fault_name, probability), ...],
                "severity_estimate": float,  # 估计严重度 1-6
                "confidence": float,
                "temporal_trend": "steady" | "degrading" | "critical",
            }
        """
        if self._model is not None:
            # DL 模型推理 (Phase 5+ 实现)
            combined = np.concatenate([feature_vector, graph_embedding])
            # logits = self._model.predict(combined)
            # probs = softmax(logits)
            raise NotImplementedError("DL model inference not yet implemented")
        else:
            # 规则基线（当前实现）
            return self._rule_based_predict(feature_vector)

    def _rule_based_predict(self, fv: np.ndarray) -> Dict:
        """基于规则的基线预测（DL 模型未就绪时使用）"""
        # 从特征向量判断主要故障方向
        freq_1x = fv[4]
        freq_2x = fv[5]
        freq_bpfo = fv[7]
        freq_bpfi = fv[8]

        probs = {}

        # 2X增大 → 不对中
        if freq_2x > 0.3:
            probs["Angular_misalignment"] = min(0.9, freq_2x)

        # 1X增大 → 不平衡
        if freq_1x > 0.3:
            probs["Unbalance_motor"] = min(0.9, freq_1x)
            probs["Unbalance_pump"] = min(0.8, freq_1x * 0.8)

        # BPFO增大 → 轴承外圈故障
        if freq_bpfo > 0.1:
            probs["Bearing_BPFO"] = min(0.9, freq_bpfo * 3)

        # BPFI增大 → 轴承内圈故障
        if freq_bpfi > 0.1:
            probs["Bearing_BPFI"] = min(0.9, freq_bpfi * 3)

        # 全频带高 → 气蚀
        if np.mean(fv[4:]) > 0.2:
            probs["Cavitation_suction"] = min(0.7, np.mean(fv[4:]) * 2)

        # 排序
        sorted_faults = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

        # 趋势判断
        rms_val = fv[0]
        if rms_val > 0.6:
            trend = "critical"
        elif rms_val > 0.4:
            trend = "degrading"
        else:
            trend = "steady"

        return {
            "top_faults": [(f, round(p, 3)) for f, p in sorted_faults],
            "fault_names_zh": {f: self._get_fault_name_zh(f) for f, _ in sorted_faults},
            "severity_estimate": min(6, max(1, int(rms_val * 8))),
            "confidence": float(np.mean([p for _, p in sorted_faults])) if sorted_faults else 0.3,
            "temporal_trend": trend,
        }

    def _get_fault_name_zh(self, fault_code: str) -> str:
        from core.pump_domain import FAULT_NAMES_ZH
        return FAULT_NAMES_ZH.get(fault_code, fault_code)

    def predict_batch(
        self,
        feature_vectors: np.ndarray,
        graph_embeddings: np.ndarray,
    ) -> List[Dict]:
        """批量预测"""
        return [
            self.predict(fv, ge)
            for fv, ge in zip(feature_vectors, graph_embeddings)
        ]


# ── 便捷 API ────────────────────────────────────────

def create_prediction_pipeline() -> Tuple[SignalEncoder, TKGEncoder, FusionPredictor]:
    """工厂函数：创建完整的预测管道"""
    return SignalEncoder(), TKGEncoder(), FusionPredictor()
