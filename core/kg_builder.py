"""
离心泵故障知识图谱统一构建器

统一了 scripts/build_index.py 和 scripts/build_full.py 的图构建逻辑。
使用分层存储原则：
- 语义实体 → 图谱节点
- 传感器数值特征 → 实体属性（不作为独立节点）
- 因果链 + 时序关系 → 关系边

用法:
    from core.kg_builder import KnowledgeGraphBuilder
    kgb = KnowledgeGraphBuilder()
    kgb.build(output_dir="data/output")
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple


class KnowledgeGraphBuilder:
    """构建离心泵故障知识图谱（语义实体 + 因果链 + 时序关系）"""

    def __init__(self):
        from core.pump_domain import (
            PumpDomainKnowledge, FAULT_NAMES_ZH, SENSOR_NAMES_ZH
        )
        self.domain = PumpDomainKnowledge()
        self.FAULT_NAMES_ZH = FAULT_NAMES_ZH
        self.SENSOR_NAMES_ZH = SENSOR_NAMES_ZH

        self._entities: Dict[str, dict] = {}
        self._edges: List[dict] = []
        self._edge_keys: set = set()

    # ── 公共 API ──────────────────────────────────────

    def build(self, output_dir: str | Path = "data/output") -> Tuple[List, List]:
        """完整重建知识图谱 → 返回 (entities, edges)"""
        self._entities.clear()
        self._edges.clear()
        self._edge_keys.clear()

        self._add_equipment_nodes()
        self._add_component_nodes()
        self._add_fault_nodes()
        self._add_fault_causal_chains()
        self._add_fault_classification()
        self._add_sensor_nodes()
        self._add_operating_conditions()
        self._add_inference_rules()

        entities = list(self._entities.values())
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "entities.json", "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        with open(output_dir / "relationships.json", "w", encoding="utf-8") as f:
            json.dump(self._edges, f, ensure_ascii=False, indent=2)

        print(f"[KG Builder] 实体: {len(entities)} 个, 关系: {len(self._edges)} 条")
        return entities, self._edges

    # ── 实体构建 ──────────────────────────────────────

    def _add_node(self, eid: str, etype: str, name: str,
                  description: str = "", source: str = "", degree: int = 0):
        if eid not in self._entities:
            self._entities[eid] = {
                "id": eid, "type": etype, "name": name,
                "description": description, "source": source, "degree": degree,
            }

    def _add_edge(self, src: str, tgt: str, relation: str,
                  description: str = "", weight: float = 1.0):
        key = (src, tgt)
        if key not in self._edge_keys:
            self._edge_keys.add(key)
            self._edges.append({
                "source": src, "target": tgt,
                "relation": relation, "description": description,
                "weight": weight,
            })

    def _add_equipment_nodes(self):
        """设备节点"""
        for ename, spec in self.domain.equipment_specs.items():
            type_zh = "电机" if spec.type == "motor" else "离心泵"
            desc_parts = [f"{type_zh} {spec.model}"]
            if spec.rated_power_kw:
                desc_parts.append(f"{spec.rated_power_kw}kW")
            if spec.rated_speed_rpm:
                desc_parts.append(f"{spec.rated_speed_rpm}rpm")
            desc_parts.append(spec.notes)
            desc = ", ".join(desc_parts)

            self._add_node(
                ename, "设备", spec.model, description=desc,
                source="datasheet_pdf"
            )

    def _add_component_nodes(self):
        """部件节点 — 电机和泵的关键轴承位置"""
        component_map = {
            "Motor_MG160MA": ("Motor_MG160MA_DE", "Motor_MG160MA_NDE"),
            "Motor_MG180MB": ("Motor_MG180MB_DE", "Motor_MG180MB_NDE"),
            "Pump_NK80-250":  ("Pump_NK80-250_DE",  "Pump_NK80-250_NDE"),
            "Pump_NK80-160":  ("Pump_NK80-160_DE",  "Pump_NK80-160_NDE"),
        }
        for equip_id, (de_id, nde_id) in component_map.items():
            equip = self.domain.equipment_specs.get(equip_id)
            model = equip.model if equip else equip_id

            self._add_node(de_id, "部件", f"{model}驱动端轴承",
                           description=f"Drive End bearing ({model})")
            self._add_node(nde_id, "部件", f"{model}非驱动端轴承",
                           description=f"Non-Drive End bearing ({model})")

            self._add_edge(equip_id, de_id, "包含", f"{model}包含驱动端轴承")
            self._add_edge(equip_id, nde_id, "包含", f"{model}包含非驱动端轴承")

    def _add_fault_nodes(self):
        """故障类型节点 — 已修复 Bug（原来用了 __class__）"""
        for fault_code, zh_name in self.FAULT_NAMES_ZH.items():
            if fault_code == "Healthy":
                continue
            self._add_node(
                f"fault_{fault_code}",
                "故障类型",
                zh_name,
                description=f"{zh_name} ({fault_code})"
            )

    def _add_fault_causal_chains(self):
        """故障因果链 — 基于领域知识的推理规则"""
        F = "fault_"

        # ── 关联（同类故障内部） ──
        misalignment = ["Angular_misalignment", "Parallel_misalignment",
                        "Combined_misalignment"]
        cavitation = ["Cavitation_suction", "Cavitation_discharge"]
        bearing = ["Bearing_BPFO", "Bearing_BPFI", "Bearing_BSF",
                   "Bearing_contaminated", "Pump_bearing"]
        unbalance = ["Unbalance_motor", "Unbalance_pump"]

        for group, name in [(misalignment, "不对中类"),
                             (cavitation, "气蚀类"),
                             (bearing, "轴承类"),
                             (unbalance, "不平衡类")]:
            for i, a in enumerate(group):
                for j, b in enumerate(group):
                    if i < j:
                        self._add_edge(
                            f"{F}{a}", f"{F}{b}", "关联",
                            f"同类故障: {name}"
                        )

        # ── 导致（因果链） ──
        # 不对中 → 轴承磨损
        for mf in misalignment:
            for bf in bearing:
                self._add_edge(
                    f"{F}{mf}", f"{F}{bf}", "导致",
                    f"{self.FAULT_NAMES_ZH.get(mf, mf)} 可能导致 {self.FAULT_NAMES_ZH.get(bf, bf)}",
                    weight=0.7
                )

        # 不平衡 → 轴承磨损
        for uf in unbalance:
            for bf in bearing:
                self._add_edge(
                    f"{F}{uf}", f"{F}{bf}", "导致",
                    f"{self.FAULT_NAMES_ZH.get(uf, uf)} 可能导致 {self.FAULT_NAMES_ZH.get(bf, bf)}",
                    weight=0.6
                )

        # 气蚀 → 叶轮故障
        self._add_edge(f"{F}Cavitation_suction", f"{F}Impeller_fault",
                       "导致", "吸入口气蚀可能导致叶轮损坏", weight=0.8)
        self._add_edge(f"{F}Cavitation_discharge", f"{F}Impeller_fault",
                       "导致", "排出口气蚀可能导致叶轮损坏", weight=0.8)

        # 联轴器损坏 → 不对中
        self._add_edge(f"{F}Coupling_damage", f"{F}Angular_misalignment",
                       "导致", "联轴器损坏可能导致角度不对中", weight=0.8)
        self._add_edge(f"{F}Coupling_damage", f"{F}Parallel_misalignment",
                       "导致", "联轴器损坏可能导致平行不对中", weight=0.7)

        # 软脚 → 不对中
        self._add_edge(f"{F}Soft_foot", f"{F}Angular_misalignment",
                       "导致", "软脚可能导致角度不对中", weight=0.6)
        self._add_edge(f"{F}Soft_foot", f"{F}Parallel_misalignment",
                       "导致", "软脚可能导致平行不对中", weight=0.6)

        # 松动 → 不对中/不平衡
        for loose in ["Loose_foot_motor", "Loose_foot_pump"]:
            self._add_edge(f"{F}{loose}", f"{F}Angular_misalignment",
                           "导致", f"{self.FAULT_NAMES_ZH[loose]}可能导致不对中", weight=0.5)
            self._add_edge(f"{F}{loose}", f"{F}Unbalance_motor",
                           "导致", f"{self.FAULT_NAMES_ZH[loose]}可能导致不平衡", weight=0.5)

        # 定子短路 → 轴承故障
        self._add_edge(f"{F}Stator_short", f"{F}Bearing_BPFO",
                       "导致", "定子短路导致的异常振动可能加速轴承外圈磨损", weight=0.5)

    def _add_fault_classification(self):
        """故障层级分类"""
        # 6 大类故障
        categories = {
            "fault_cat_misalignment": ("不对中类故障", [
                "Angular_misalignment", "Parallel_misalignment", "Combined_misalignment"
            ]),
            "fault_cat_unbalance": ("不平衡类故障", [
                "Unbalance_motor", "Unbalance_pump"
            ]),
            "fault_cat_bearing": ("轴承类故障", [
                "Bearing_BPFO", "Bearing_BPFI", "Bearing_BSF",
                "Bearing_contaminated", "Pump_bearing"
            ]),
            "fault_cat_cavitation": ("气蚀类故障", [
                "Cavitation_suction", "Cavitation_discharge"
            ]),
            "fault_cat_structure": ("结构/安装类故障", [
                "Soft_foot", "Loose_foot_motor", "Loose_foot_pump",
                "Bent_shaft", "Coupling_damage"
            ]),
            "fault_cat_electrical": ("电气类故障", [
                "Broken_rotor_bar", "Stator_short"
            ]),
        }
        for cat_id, (cat_name, fault_list) in categories.items():
            self._add_node(cat_id, "故障类型", cat_name,
                           description=f"故障大类: {cat_name}")
            for fc in fault_list:
                self._add_edge(
                    f"fault_{fc}", cat_id, "属于",
                    f"{self.FAULT_NAMES_ZH[fc]}属于{cat_name}"
                )

        # 叶轮故障自成一类
        self._add_node("fault_cat_impeller", "故障类型", "叶轮类故障",
                       description="故障大类: 叶轮类故障")
        self._add_edge("fault_Impeller_fault", "fault_cat_impeller", "属于",
                       "叶轮故障属于叶轮类故障")

    def _add_sensor_nodes(self):
        """传感器和监测点节点"""
        for sensor in self.domain.sensor_configs:
            ch_key = f"Ch{sensor.channel}"
            loc_name = self.SENSOR_NAMES_ZH.get(ch_key, sensor.location)

            # 传感器实体
            self._add_node(
                ch_key, "传感器",
                f"{ch_key} {loc_name}",
                description=f"{sensor.location} {sensor.orientation}向 加速度计, {sensor.sample_rate_hz//1000}kHz, {sensor.sensitivity}",
                source="dataset_readme"
            )

            # 监测点实体
            mp_id = f"{ch_key}_{sensor.location}_{sensor.orientation}"
            self._add_node(
                mp_id, "监测点", loc_name,
                description=f"{sensor.location} ({sensor.orientation}向) 振动监测点"
            )
            self._add_edge(mp_id, ch_key, "安装于",
                           f"{sensor.location}安装{ch_key}传感器")

            # 关联到设备
            if "Motor" in sensor.location:
                # 判断是 NDE 还是 DE
                if "NDE" in sensor.location:
                    motor = "Motor_MG160MA_NDE" if sensor.orientation == "H" else "Motor_MG180MB_NDE"
                else:
                    motor = "Motor_MG160MA_DE" if sensor.orientation == "V" else "Motor_MG180MB_DE"
                self._add_edge(motor, mp_id, "监测点",
                               f"{sensor.location}位置设置监测点")
            elif "Pump" in sensor.location:
                self._add_edge("Pump_NK80-250", mp_id, "监测点",
                               f"{sensor.location}位置设置监测点")

    def _add_operating_conditions(self):
        """工况条件节点"""
        for speed_name, speed_pct in [("Speed_50pct", 50), ("Speed_75pct", 75),
                                       ("Speed_100pct", 100), ("Speed_70pct", 70)]:
            self._add_node(
                speed_name, "工况条件",
                f"{speed_pct}%额定转速",
                description=f"电机额定转速的 {speed_pct}%"
            )
            self._add_edge("Motor_MG160MA", speed_name, "工况",
                           f"MG160MA可在{speed_pct}%额定转速运行")

    def _add_inference_rules(self):
        """预定义推理规则 — 基于振动特征的故障诊断原则"""
        rules = [
            ("fault_Angular_misalignment", "freq_2x",
             "2倍转频(2X)振动分量突出",
             "角度不对中的核心特征：2X分量显著增大"),
            ("fault_Parallel_misalignment", "freq_2x",
             "2倍转频(2X)振动分量突出",
             "平行不对中同样引起2X增大（轴向分量相对角度不对中偏小）"),
            ("fault_Unbalance_motor", "freq_1x",
             "1倍转频(1X)振动分量突出",
             "不平衡的核心特征：1X径向振动为主导"),
            ("fault_Unbalance_pump", "freq_1x",
             "1倍转频(1X)振动分量突出",
             "泵不平衡同样表现为1X径向振动增大"),
            ("fault_Bearing_BPFO", "freq_bpfo",
             "BPFO特征频率增加",
             "轴承外圈故障的频域特征"),
            ("fault_Bearing_BPFI", "freq_bpfi",
             "BPFI特征频率增加",
             "轴承内圈故障的频域特征"),
            ("fault_Bearing_BSF", "freq_bsf",
             "BSF特征频率增加",
             "滚动体故障的频域特征"),
            ("fault_Cavitation_suction", "freq_broadband",
             "宽频带高频振动",
             "气蚀的频域特征：宽频带能量增加"),
            ("fault_Cavitation_discharge", "freq_broadband",
             "宽频带高频振动",
             "排出口气蚀同样表现为宽频振动"),
            ("fault_Impeller_fault", "freq_1x",
             "1X转频增大",
             "叶轮磨损/不平衡导致1X增大"),
        ]
        for fault_id, signal_type, signal_desc, desc in rules:
            # 创建一个"信号特征模式"节点（非数值，而是语义描述）
            signal_id = f"signal_pattern_{signal_type}"
            self._add_node(signal_id, "信号特征", signal_desc,
                           description=f"典型信号特征: {signal_desc}")
            self._add_edge(fault_id, signal_id, "表现为",
                           desc, weight=0.9)


def build_knowledge_graph(output_dir: str = "data/output") -> Tuple[List, List]:
    """便捷函数：一键构建知识图谱"""
    kgb = KnowledgeGraphBuilder()
    return kgb.build(output_dir=output_dir)
