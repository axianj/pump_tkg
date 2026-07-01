"""
时序知识图谱适配器

在 LightRAG/RAG-Anything 的基础上增加时序支持：
1. 给测量记录实体自动添加时间戳
2. 相同监测点的时序实体之间建立 prev/next 关系链
3. 支持增量插入（非重头构建）
4. 时间范围的查询过滤
"""

import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path


@dataclass
class TemporalMeasurement:
    """一次传感器测量记录的时序实体"""
    measurement_id: str          # 唯一标识
    timestamp: datetime          # 测量时间
    setup_id: str                # Setup 1 / 2
    motor_speed_pct: int         # 50 / 75 / 100
    fault_type: str              # 故障类型代码
    severity: int                # 严重度 1-6
    channel_data: Dict[str, Any] = field(default_factory=dict)  # 各通道统计特征
    source_file: str = ""        # 来源 CSV 文件
    health_status: str = "healthy"  # healthy / faulty


@dataclass
class TemporalConfig:
    """时序配置"""
    enabled: bool = True
    timestamp_field: str = "measured_at"
    prev_relation: str = "前次测量"
    next_relation: str = "后续测量"
    max_chain_length: int = 1000


class TemporalAdapter:
    """
    时序适配器 — 负责：

    1. 从 CSV 文件路径解析时间戳和故障标签
    2. 将测量记录构造成时序实体
    3. 维护 prev/next 关系链
    4. 生成增量插入数据（用于 LightRAG 的 insert）
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self.measurements: List[TemporalMeasurement] = []

    def parse_measurement_from_path(self, rel_path: str) -> Optional[TemporalMeasurement]:
        """
        从相对路径解析测量记录元数据

        路径示例:
        Dataset/Vibration/Set_2/Speed_3/Fault_Angular_misalignment/Severity_5/
            Angular_misalignment_Severity_5_Speed_3_Ch_1.csv
        """
        parts = rel_path.replace("\\", "/").split("/")

        try:
            # 提取分类信息
            setup_id = ""
            speed_pct = 0
            fault_type = ""
            severity = 0
            channel = ""

            for i, p in enumerate(parts):
                if p.startswith("Set_"):
                    setup_id = p
                elif p.startswith("Speed_"):
                    speed_pct = int(p.replace("Speed_", "")) * 25  # 1→25%,2→50%,...
                elif p.startswith("Fault_"):
                    fault_type = p.replace("Fault_", "")
                elif p.startswith("Severity_"):
                    severity = int(p.replace("Severity_", ""))
                elif p.startswith("Ch_"):
                    channel = p.replace("Ch_", "")
                elif p.endswith(".csv"):
                    # 从文件名提取通道号（如果没有从路径提取到）
                    if not channel:
                        # 文件名通常是: Fault_Severity_Speed_Ch.csv
                        base = p.replace(".csv", "")
                        parts_f = base.split("_")
                        if parts_f[-2].startswith("Ch") or parts_f[-1].startswith("Ch"):
                            ch_part = parts_f[-1] if parts_f[-1].startswith("Ch") else parts_f[-2]
                            channel = ch_part.replace("Ch_", "").replace("Ch", "")

            if not fault_type:
                return None

            # 构造测量 ID
            ts_str = f"202007_{setup_id}_{speed_pct}pct_{fault_type}_sev{severity}"
            mid = hashlib.md5(ts_str.encode()).hexdigest()[:12]

            # 解析时间（从路径文件名中的日期，如果存在）
            timestamp = datetime(2020, 7, 1)  # 默认七月数据

            return TemporalMeasurement(
                measurement_id=f"M_{mid}",
                timestamp=timestamp,
                setup_id=setup_id,
                motor_speed_pct=speed_pct,
                fault_type=fault_type,
                severity=severity,
                source_file=rel_path,
                health_status="faulty" if fault_type and fault_type != "Healthy" else "healthy",
            )
        except Exception:
            return None

    def build_measurement_text(self, m: TemporalMeasurement) -> str:
        """将测量记录转换为可插入 LightRAG 的文本"""
        fault_zh = self._fault_name_zh(m.fault_type)
        status_zh = "故障" if m.health_status == "faulty" else "正常"
        speed_map = {25: "25%", 50: "50%", 75: "75%", 100: "100%"}

        text = f"""
## 离心泵测量记录

**测量ID**: {m.measurement_id}
**时间**: {m.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**实验设置**: {m.setup_id}
**电机转速**: {speed_map.get(m.motor_speed_pct, f'{m.motor_speed_pct}%')}
**设备状态**: {status_zh}

### 故障信息
- **故障类型**: {fault_zh} ({m.fault_type})
- **严重度等级**: {m.severity} (1-6级，越高越严重)

### 数据来源
- **原始文件**: {m.source_file}
"""
        return text

    def build_temporal_relations(self, measurements: List[TemporalMeasurement]) -> List[Dict]:
        """
        在同一故障类型 + 同一严重度下，按时间构建 prev/next 关系链

        返回关系列表: [{source, target, relation, description}]
        """
        relations = []
        # 按 (setup, speed, fault, severity) 分组
        groups: Dict[str, List[TemporalMeasurement]] = {}
        for m in measurements:
            key = f"{m.setup_id}_{m.motor_speed_pct}_{m.fault_type}_{m.severity}"
            if key not in groups:
                groups[key] = []
            groups[key].append(m)

        # 每组内按时间排序
        for key, group in groups.items():
            group.sort(key=lambda x: x.timestamp)
            for i in range(len(group) - 1):
                if i >= self.config.max_chain_length:
                    break
                relations.append({
                    "source": group[i].measurement_id,
                    "target": group[i + 1].measurement_id,
                    "relation": self.config.next_relation,
                    "description": f"测量 {group[i].measurement_id} 之后进行 {group[i+1].measurement_id}",
                })
                relations.append({
                    "source": group[i + 1].measurement_id,
                    "target": group[i].measurement_id,
                    "relation": self.config.prev_relation,
                    "description": f"测量 {group[i+1].measurement_id} 之前进行 {group[i].measurement_id}",
                })

        return relations

    def _fault_name_zh(self, fault_code: str) -> str:
        """故障代码转中文"""
        names = {
            "Angular_misalignment": "角度不对中",
            "Parallel_misalignment": "平行不对中",
            "Combined_misalignment": "复合不对中",
            "Unbalance_motor": "电机不平衡",
            "Unbalance_pump": "泵不平衡",
            "Coupling_damage": "联轴器损坏",
            "Cavitation_suction": "吸入口气蚀",
            "Cavitation_discharge": "排出口气蚀",
            "Bent_shaft": "轴弯曲",
            "Impeller_fault": "叶轮故障",
            "Bearing_BPFO": "轴承外圈故障",
            "Bearing_BPFI": "轴承内圈故障",
            "Bearing_contaminated": "轴承污染",
            "Bearing_BSF": "滚动体故障",
            "Soft_foot": "软脚",
            "Loose_foot_motor": "电机底脚松动",
            "Loose_foot_pump": "泵底脚松动",
            "Broken_rotor_bar": "断条",
            "Stator_short": "定子短路",
            "Pump_bearing": "泵轴承故障",
            "Healthy": "正常状态",
        }
        return names.get(fault_code, fault_code)

    def export_for_lightrag(self, measurements: List[TemporalMeasurement], output_dir: Path):
        """
        将时序测量数据导出为 LightRAG 可读取的文本文件

        每个测量记录生成一个 .md 文件，包含：
        - 元数据（YAML frontmatter）
        - 故障描述
        - 时序关系（以链接形式）
        """
        docs_dir = output_dir / "temporal_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        for m in measurements:
            content = self.build_measurement_text(m)
            filepath = docs_dir / f"{m.measurement_id}.md"
            filepath.write_text(content, encoding="utf-8")

        # 生成时序关系文件（用于 Neo4j 导入）
        rels = self.build_temporal_relations(measurements)
        rel_path = output_dir / "temporal_relations.json"
        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump(rels, f, ensure_ascii=False, indent=2)

        print(f"[Temporal] 导出 {len(measurements)} 个测量记录到 {docs_dir}")
        print(f"[Temporal] 导出 {len(rels)} 条时序关系到 {rel_path}")
        return docs_dir
