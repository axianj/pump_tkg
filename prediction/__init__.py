"""
预测模块 — 传感器信号编码 + TKG图嵌入 + 机器学习分类器

Schema:
    sensor_features (10维) → RandomForest → fault_predictions (20类)
    + temporal_graph → TKGEncoder → graph_embedding (64维) [可选融合]

基线模型候选池:
    规则基线: _rule_based_predict (仅对 BPFO/BPFI/不平衡/不对中 有效)
    ML 分类器: RandomForest + SMOTE 处理类别不平衡 (575样本 × 21类)
    预留接口: TTansE, HyTE, EvolveGCN, RE-NET, TGN, Know-Evolve
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

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
        rel_counter = Counter(r.get("relation", "unknown") for r in edges)
        n_rel_types = len(rel_counter)

        # 实体类型分布
        type_counter = Counter(e.get("type", "unknown") for e in entities)
        n_entity_types = len(type_counter)

        # 时序边密度 (有 valid_from / valid_to 的边)
        temporal_edge_count = sum(
            1 for r in edges
            if r.get("valid_from") or r.get("from_time") or "后续测量" in r.get("relation", "")
        )

        # 组装 embedding (64维)
        embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        embedding[0] = float(np.log1p(n_entities))
        embedding[1] = float(np.log1p(n_edges))
        embedding[2] = float(deg_mean) / (deg_max + 1)
        embedding[3] = float(deg_std) / (deg_max + 1)
        embedding[4] = float(n_rel_types) / 20.0
        embedding[5] = float(n_entity_types) / 10.0
        embedding[6] = float(np.log1p(temporal_edge_count))
        # 填充关系分布
        for i, (rel, cnt) in enumerate(rel_counter.most_common(10)):
            embedding[7 + i] = float(cnt) / max(n_edges, 1)

        # 归一化
        norm = np.linalg.norm(embedding) + 1e-10
        return embedding / norm

    def encode_temporal_quads(self, quads: List) -> np.ndarray:
        """从时序四元组列表编码"""
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


# ── ML 分类器 ───────────────────────────────────────

class FaultClassifier:
    """
    基于 RandomForest + SMOTE 的故障类型分类器。

    输入: (n_samples, 10) 频域特征矩阵
    输出: (n_samples,) 故障类型预测 + 概率分布

    训练方式:
        clf = FaultClassifier()
        clf.fit(X_train, y_train)  # y_train 是故障代码列表
        preds = clf.predict(X_test)   # 返回 Top-K 故障 + 概率
    """

    # 支持的所有故障类型
    FAULT_CLASSES = [
        "Healthy",
        "Angular_misalignment", "Parallel_misalignment", "Combined_misalignment",
        "Unbalance_motor", "Unbalance_pump",
        "Coupling_damage", "Cavitation_suction", "Cavitation_discharge",
        "Bent_shaft", "Impeller_fault",
        "Bearing_BPFO", "Bearing_BPFI", "Bearing_contaminated", "Bearing_BSF",
        "Soft_foot", "Loose_foot_motor", "Loose_foot",
        "Broken_rotor_bar", "Stator_short", "Pump_bearing",
    ]

    def __init__(self, use_smote: bool = True, random_state: int = 42):
        self._use_smote = use_smote
        self._random_state = random_state
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._fitted_classes = None
        self._smote = None

    def fit(self, X: np.ndarray, y: List[str]):
        """
        训练分类器。

        Args:
            X: (n_samples, 10) 特征矩阵
            y: (n_samples,) 故障代码列表
        """
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier

        # 标签编码
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self._fitted_classes = self._label_encoder.classes_

        # 标准化
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # SMOTE 过采样 (处理类别不平衡)
        if self._use_smote:
            from imblearn.over_sampling import SMOTE
            # 统计最少样本数
            min_count = min(Counter(y).values())
            # SMOTE 的 k_neighbors 必须 <= min_count - 1
            k_neighbors = max(1, min(min_count - 1, 4))
            self._smote = SMOTE(
                k_neighbors=k_neighbors,
                random_state=self._random_state,
            )
            X_scaled, y_encoded = self._smote.fit_resample(X_scaled, y_encoded)

        # RandomForest
        self._model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=self._random_state,
            n_jobs=-1,
        )
        self._model.fit(X_scaled, y_encoded)

        return self

    def predict(self, X: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        单样本/批量预测。

        Returns:
            [{
                "top_faults": [(fault_code, probability), ...],
                "top_3_faults": [(fault_code, probability)],
                "confidence": float,
                "severity_estimate": int,
            }, ...]
        """
        if self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self._scaler.transform(X)
        proba = self._model.predict_proba(X_scaled)

        results = []
        for i in range(len(X)):
            probs = proba[i]
            # 取 Top-K
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_faults = [
                (self._label_encoder.inverse_transform([idx])[0], float(probs[idx]))
                for idx in top_indices
            ]

            # 严重度估计 (基于 RMS 特征)
            rms = X[i][0] if X.shape[1] > 0 else 0
            severity = min(6, max(1, int(rms * 8)))

            results.append({
                "top_faults": top_faults,
                "top_3_faults": top_faults[:3],
                "confidence": float(np.max(probs)),
                "severity_estimate": severity,
                "all_probabilities": {
                    self._label_encoder.inverse_transform([j])[0]: float(probs[j])
                    for j in top_indices
                },
            })

        return results

    def evaluate(self, X: np.ndarray, y: List[str]) -> Dict:
        """在测试集上评估，返回完整指标"""
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            accuracy_score, f1_score,
        )
        import json

        y_pred_encoded = self._model.predict(self._scaler.transform(X))
        y_pred = self._label_encoder.inverse_transform(y_pred_encoded)

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
            "classification_report": classification_report(
                y, y_pred, output_dict=True, zero_division=0,
            ),
            "n_samples": len(X),
            "n_classes": len(set(y)),
        }


# ── Fusion Predictor ────────────────────────────────

class FusionPredictor:
    """
    融合预测器 — ML 分类器 + 规则回退

    默认使用 FaultClassifier (RandomForest + SMOTE)。
    Feature Vector (10维) → FaultClassifier → Top-K Fault Predictions
    Graph Embedding (64维) 保留为可选特征融合输入。
    """

    def __init__(self):
        self._classifier = None
        self._trained = False

    def train(self, X: np.ndarray, y: List[str]):
        """训练 ML 分类器"""
        self._classifier = FaultClassifier(use_smote=True)
        self._classifier.fit(X, y)
        self._trained = True
        return self

    def predict(self, feature_vector: np.ndarray, graph_embedding: Optional[np.ndarray] = None) -> Dict:
        """统一预测接口"""
        if self._trained:
            result = self._classifier.predict(feature_vector)[0]
            # 添加故障中文名
            fault_names_zh = {}
            for fault_code, _ in result["top_faults"]:
                fault_names_zh[fault_code] = self._get_fault_name_zh(fault_code)
            result["fault_names_zh"] = fault_names_zh
            return result
        else:
            return self._rule_based_predict(feature_vector)

    def _rule_based_predict(self, fv: np.ndarray) -> Dict:
        """基于规则的基线预测"""
        freq_1x = fv[4]
        freq_2x = fv[5]
        freq_bpfo = fv[7]
        freq_bpfi = fv[8]
        rms_val = fv[0]

        probs = {}

        if freq_2x > 0.3:
            probs["Angular_misalignment"] = min(0.9, freq_2x)
        if freq_1x > 0.3:
            probs["Unbalance_motor"] = min(0.9, freq_1x)
        if freq_bpfo > 0.1:
            probs["Bearing_BPFO"] = min(0.9, freq_bpfo * 3)
        if freq_bpfi > 0.1:
            probs["Bearing_BPFI"] = min(0.9, freq_bpfi * 3)

        sorted_faults = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        fault_names_zh = {f: self._get_fault_name_zh(f) for f, _ in sorted_faults}

        if rms_val > 0.6:
            trend = "critical"
        elif rms_val > 0.4:
            trend = "degrading"
        else:
            trend = "steady"

        return {
            "top_faults": [(f, round(p, 3)) for f, p in sorted_faults],
            "fault_names_zh": fault_names_zh,
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
        graph_embeddings: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """批量预测"""
        return [self.predict(fv) for fv in feature_vectors]

    def evaluate(self, X: np.ndarray, y: List[str]) -> Optional[Dict]:
        """返回分类器评估指标 (仅 ML 模式)"""
        if self._trained:
            return self._classifier.evaluate(X, y)
        return None


# ── 便捷 API ────────────────────────────────────────

def create_prediction_pipeline() -> Tuple[SignalEncoder, TKGEncoder, FusionPredictor]:
    """工厂函数：创建完整的预测管道"""
    return SignalEncoder(), TKGEncoder(), FusionPredictor()
