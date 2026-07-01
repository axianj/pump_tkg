"""
离心泵故障诊断领域知识定义

基于 Dataset README.txt + Appendices 中的文档资料构建的领域知识体系。
包含故障分类体系、设备规格、实体/关系类型定义。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── 故障分类体系 ──────────────────────────────────────

@dataclass
class FaultTaxonomy:
    """离心泵故障分类体系 — 来源于实验设计中的故障类型列表"""

    # Setup1 (Motor 2) — 3种转速
    motor2_faults: Dict[str, List[int]] = field(default_factory=lambda: {
        "Impeller_fault":         [1, 2, 3],        # 叶轮故障 (3个严重度)
        "Bearing_BPFO":           [1, 2, 3],        # 外圈故障
        "Bearing_BPFI":           [1, 2, 3],        # 内圈故障
        "Bearing_contaminated":   [1],               # 轴承污染
        "Bearing_BSF":            [1],               # 滚动体故障
        "Soft_foot":              [1, 2],            # 软脚 (2个等级)
        "Loose_foot_motor":       [1],               # 电机底脚松动
        "Loose_foot_pump":        [1],               # 泵底脚松动
        "Broken_rotor_bar":       [1],               # 断条
        "Stator_short":           [1, 2],            # 定子短路
        "Pump_bearing":           [1, 2, 3],         # 泵轴承故障
    })

    # Setup2 (Motor 4) — 单转速，但故障更多
    motor4_faults: Dict[str, List[int]] = field(default_factory=lambda: {
        "Angular_misalignment":   [1, 2, 3, 4, 5],   # 角度不对中
        "Parallel_misalignment":  [1, 2, 3, 4],      # 平行不对中
        "Combined_misalignment":  [1, 2, 3, 4],      # 复合不对中
        "Unbalance_motor":        [1, 2, 3, 4, 5, 6],# 电机不平衡
        "Unbalance_pump":         [1, 2, 3],          # 泵不平衡
        "Coupling_damage":        [1, 2, 3, 4],       # 联轴器损坏
        "Cavitation_suction":     [1, 2, 3, 4],       # 吸入口气蚀
        "Cavitation_discharge":   [1, 2, 3, 4, 5],    # 排出口气蚀
        "Bent_shaft":             [1],                # 轴弯曲
    })


# ── 设备规格 ──────────────────────────────────────────

@dataclass
class EquipmentSpec:
    """设备规格 — 从 PDF 数据手册提取"""
    name: str
    type: str              # motor / pump
    model: str
    manufacturer: str
    rated_power_kw: Optional[float] = None
    rated_speed_rpm: Optional[float] = None
    rated_current_a: Optional[float] = None
    bearing_types: List[str] = field(default_factory=list)
    notes: str = ""


# ── 传感器配置 ─────────────────────────────────────────

@dataclass
class SensorConfig:
    """传感器配置"""
    channel: int
    location: str              # 安装位置
    orientation: str           # H=水平, V=垂直, A=轴向
    measurement_type: str      # vibration / current / voltage
    sensitivity: str = "100 mV/g"
    sample_rate_hz: int = 20000
    sample_duration_s: float = 12.0  # 振动12秒，电流15秒


# ── 领域实体/关系类型定义 ─────────────────────────────

# LightRAG / RAG-Anything 中使用的实体类型
PUMP_ENTITY_TYPES = [
    "设备",           # Motor_MG160MA, Pump_NK80-250
    "部件",           # Bearing, Impeller, Shaft, Seal
    "故障类型",       # Angular_misalignment, Cavitation
    "故障严重度",     # Severity_1 ~ Severity_6
    "传感器",         # Ch1~Ch5 加速度计, 电流钳
    "监测点",         # Motor_DE_bearing, Pump_NDE_bearing
    "测量记录",       # 具体某次实验测量
    "维修操作",       # 更换轴承、对中调整、平衡校正
    "工况条件",       # 转速(50%/75%/100%), 流量, 压力
    "信号特征",       # RMS, Peak, FFT分量, 频率峰值
]

PUMP_RELATION_TYPES = [
    "包含",           # 设备 → 部件
    "安装于",         # 传感器 → 监测点
    "监测",           # 传感器 → 测量记录
    "表现为",         # 故障 → 信号特征
    "导致",           # 故障 → 故障
    "严重度",         # 故障 → 严重度等级
    "测量于",         # 测量记录 → 时间
    "前次测量",       # 测量记录 → 前次测量(时序链)
    "后续测量",       # 测量记录 → 后续测量(时序链)
    "工况",           # 测量记录 → 工况条件
    "维修",           # 维修操作 → 解决 → 故障
]

# 中文故障显示名称
FAULT_NAMES_ZH: Dict[str, str] = {
    "Angular_misalignment":   "角度不对中",
    "Parallel_misalignment":  "平行不对中",
    "Combined_misalignment":  "复合不对中",
    "Unbalance_motor":        "电机不平衡",
    "Unbalance_pump":         "泵不平衡",
    "Coupling_damage":        "联轴器损坏",
    "Cavitation_suction":     "吸入口气蚀",
    "Cavitation_discharge":   "排出口气蚀",
    "Bent_shaft":             "轴弯曲",
    "Impeller_fault":         "叶轮故障",
    "Bearing_BPFO":           "轴承外圈故障",
    "Bearing_BPFI":           "轴承内圈故障",
    "Bearing_contaminated":   "轴承污染",
    "Bearing_BSF":            "轴承滚动体故障",
    "Soft_foot":              "软脚",
    "Loose_foot_motor":       "电机底脚松动",
    "Loose_foot_pump":        "泵底脚松动",
    "Broken_rotor_bar":       "断条",
    "Stator_short":           "定子短路",
    "Pump_bearing":           "泵轴承故障",
    "Healthy":                "正常状态",
}

SENSOR_NAMES_ZH: Dict[str, str] = {
    "Ch1": "电机非驱动端轴承水平",
    "Ch2": "电机驱动端轴承垂直",
    "Ch3": "电机驱动端轴承轴向",
    "Ch4": "泵驱动端轴承水平",
    "Ch5": "泵非驱动端轴承垂直",
}


def get_fault_name_zh(fault_code: str) -> str:
    """获取故障中文名称"""
    return FAULT_NAMES_ZH.get(fault_code, fault_code)


def get_sensor_name_zh(ch: str) -> str:
    """获取传感器中文名称"""
    return SENSOR_NAMES_ZH.get(ch, ch)


class PumpDomainKnowledge:
    """离心泵领域知识 — 供构建知识图谱使用"""

    def __init__(self):
        self.equipment_specs = self._build_equipment_specs()
        self.sensor_configs = self._build_sensor_configs()
        self.fault_taxonomy = FaultTaxonomy()

    def _build_equipment_specs(self) -> Dict[str, EquipmentSpec]:
        return {
            "Motor_MG160MA": EquipmentSpec(
                name="MG 160 MA",
                type="motor",
                model="MG 160 MA",
                manufacturer="ABB/VEM",
                rated_power_kw=11.0,
                rated_speed_rpm=1480,
                bearing_types=["NU 311", "6311"],
                notes="Setup 1 中使用; 可变速 50%/75%/100%"
            ),
            "Motor_MG180MB": EquipmentSpec(
                name="MG 180 MB",
                type="motor",
                model="MG 180 MB",
                manufacturer="ABB/VEM",
                rated_power_kw=45.0,
                rated_speed_rpm=2960,
                bearing_types=["NU 314", "6314"],
                notes="Setup 2 中使用; 固定转速 70%"
            ),
            "Pump_NK80-250": EquipmentSpec(
                name="NK 80-250",
                type="pump",
                model="NK 80-250",
                manufacturer="KSB",
                rated_power_kw=None,
                rated_speed_rpm=1480,
                notes="Setup 1 用离心泵"
            ),
            "Pump_NK80-160": EquipmentSpec(
                name="NK 80-160",
                type="pump",
                model="NK 80-160",
                manufacturer="KSB",
                rated_power_kw=None,
                rated_speed_rpm=2960,
                notes="Setup 2 用离心泵"
            ),
        }

    def _build_sensor_configs(self) -> List[SensorConfig]:
        return [
            SensorConfig(1, "Motor_NDE_bearing", "H", "vibration"),
            SensorConfig(2, "Motor_DE_bearing", "V", "vibration"),
            SensorConfig(3, "Motor_DE_bearing", "A", "vibration"),
            SensorConfig(4, "Pump_DE_bearing", "H", "vibration"),
            SensorConfig(5, "Pump_NDE_bearing", "V", "vibration"),
        ]

    def get_entity_types(self) -> List[str]:
        return PUMP_ENTITY_TYPES

    def get_relation_types(self) -> List[str]:
        return PUMP_RELATION_TYPES

    def get_system_prompt(self) -> str:
        """用于 LLM 的领域系统提示词"""
        return """你是离心泵故障诊断专家。你精通：
- 电动机驱动的离心泵设备结构（电机、轴承、叶轮、轴封等）
- 常见故障模式及其振动/电流特征（不对中、不平衡、气蚀、轴承故障等）
- 故障诊断方法（时域分析、频域分析、包络谱分析等）
- 维护策略（对中调整、平衡校正、轴承更换等）

请基于知识图谱中的信息，对离心泵故障进行专业诊断分析：
1. 识别故障类型和严重程度
2. 分析可能的故障原因
3. 给出诊断验证步骤
4. 提供维护和修复建议
5. 评估风险等级

使用专业工程术语，引用知识来源。"""
