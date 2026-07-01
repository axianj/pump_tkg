"""
离心泵故障诊断系统 — Streamlit Web 界面
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from core.pump_domain import (
    PumpDomainKnowledge, FAULT_NAMES_ZH, get_fault_name_zh, get_sensor_name_zh
)

st.set_page_config(
    page_title="离心泵故障诊断系统",
    page_icon="🔧",
    layout="wide",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def main():
    domain = PumpDomainKnowledge()

    with st.sidebar:
        st.title("🔧 离心泵故障诊断系统")
        st.markdown("---")
        st.subheader("设备概览")
        for name, spec in domain.equipment_specs.items():
            st.write(f"**{spec.model}** ({spec.type})")
            if spec.rated_power_kw:
                st.caption(f"{spec.rated_power_kw}kW / {spec.rated_speed_rpm}rpm")

        st.markdown("---")
        st.subheader("传感器通道")
        for sensor in domain.sensor_configs:
            st.write(f"Ch{sensor.channel}: {get_sensor_name_zh(f'Ch{sensor.channel}')}")
            st.caption(f"{sensor.orientation}向 | {sensor.sample_rate_hz/1000:.0f}kHz")

        st.markdown("---")
        st.subheader("故障类型")
        for zh in sorted(FAULT_NAMES_ZH.values()):
            st.write(f"- {zh}")

        st.markdown("---")
        st.caption("基于 LightRAG + RAG-Anything 构建")
        st.caption("数据来源: TKG_Data 离心泵数据集")

    st.title("🏭 离心泵设备故障智能诊断系统")
    st.caption("多模态知识图谱驱动的故障诊断与预测平台")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 智能诊断", "📊 传感器分析", "🗺️ 知识图谱", "📚 知识库"]
    )

    with tab1:
        st.subheader("故障诊断对话")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("描述设备异常现象..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("正在诊断分析..."):
                    response = _diagnose(prompt, domain)
                    st.markdown(response)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": response
                    })

    with tab2:
        st.subheader("传感器数据分析")
        st.info("选择传感器通道和故障类型查看振动/电流特征")
        col1, col2 = st.columns(2)
        with col1:
            channel = st.selectbox(
                "传感器通道",
                [f"Ch{i} - {get_sensor_name_zh(f'Ch{i}')}" for i in range(1, 6)],
            )
        with col2:
            fault = st.selectbox(
                "故障类型",
                list(FAULT_NAMES_ZH.values()),
            )

        st.write("---")
        st.write("**典型故障特征频率:**")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("1X 转频", "24.7 Hz")
            st.metric("2X 转频", "49.3 Hz")
        with col4:
            st.metric("BPFO (外圈)", "89.2 Hz")
            st.metric("BPFI (内圈)", "135.5 Hz")

        # 占位图
        st.write("**频谱分析 (示意)**")
        st.caption("注: 需要加载传感器CSV数据后显示真实频谱")
        chart_data = {
            "1X": 0.8, "2X": 0.3, "3X": 0.1,
            "BPFO": 0.05, "BPFI": 0.02, "BSF": 0.01
        }
        st.bar_chart(chart_data)

    with tab3:
        st.subheader("知识图谱可视化")
        st.info("运行 `python pump_main.py to-echarts` 生成交互式知识图谱")
        html_path = Path(PROJECT_ROOT) / "pump_graph_viz.html"
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
        else:
            st.write("知识图谱尚未生成。运行以下命令:")
            st.code("python pump_main.py build && python pump_main.py to-echarts",
                    language="bash")

    with tab4:
        st.subheader("知识库管理")
        knowledge_dir = PROJECT_ROOT / "data" / "knowledge"
        doc_files = list(knowledge_dir.glob("*.md"))
        doc_files += list(knowledge_dir.glob("*.txt"))
        if doc_files:
            for f in doc_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"- {f.name}")
                with col2:
                    if st.button("查看", key=f"view_{f.name}"):
                        content = f.read_text(encoding="utf-8")
                        st.text_area("文件内容", content, height=400)
        else:
            st.warning("知识库为空，请运行 `python pump_main.py init`")

        st.write("---")
        st.write("**系统命令:**")
        st.code("python pump_main.py build --full", language="bash")
        st.code("python pump_main.py to-neo4j", language="bash")
        st.code("python pump_main.py ingest-sensors", language="bash")


def _diagnose(query: str, domain) -> str:
    """简易诊断逻辑（离线模式，不依赖 LLM）"""
    query_lower = query.lower()

    # 匹配故障类型
    matched_faults = []
    for code, zh in FAULT_NAMES_ZH.items():
        if code.lower().replace("_", "") in query_lower.replace(" ", "") or zh in query:
            matched_faults.append((code, zh))

    # 匹配设备
    has_motor = any(kw in query for kw in ["电机", "motor", "MG"])
    has_pump = any(kw in query for kw in ["泵", "pump", "NK"])
    has_bearing = any(kw in query for kw in ["轴承", "bearing", "BPFO", "BPFI", "BSF"])

    lines = []
    lines.append("### 🔍 诊断分析结果")

    if matched_faults:
        lines.append(f"\n**识别到的故障类型:**")
        for code, zh in matched_faults:
            lines.append(f"- **{zh}** ({code})")

        if any("misalignment" in c or "不对中" in z for c, z in matched_faults):
            lines.append("\n**不对中故障特征:**")
            lines.append("- 2倍转频(2X)振动分量突出")
            lines.append("- 轴向振动显著增大")
            lines.append("- 建议: 检查联轴器对中")
            lines.append("- 验证: 激光对中仪检测")

        if any("cavitation" in c or "气蚀" in z for c, z in matched_faults):
            lines.append("\n**气蚀故障特征:**")
            lines.append("- 宽频带高频振动")
            lines.append("- 伴有噪声和流量波动")
            lines.append("- 建议: 检查入口压力和过滤器")
            lines.append("- 验证: 提高入口压力或降低转速")

        if any("bearing" in c or "轴承" in z for c, z in matched_faults):
            lines.append("\n**轴承故障特征:**")
            lines.append("- 对应特征频率及其谐波")
            lines.append("- 伴有边频带")
            lines.append("- 建议: 检查轴承状态，准备更换")
            lines.append("- 验证: 包络谱分析确认")

    else:
        lines.append("\n**未精确匹配到已知故障类型**")
        lines.append("\n基于当前症状的可能方向:")
        if "振动" in query or "vibration" in query_lower:
            lines.append("- 振动异常: 可能为不平衡、不对中或轴承故障")
        if "温度" in query or "温度" in query:
            lines.append("- 温度异常: 可能为轴承磨损或润滑不良")
        if "噪声" in query or "噪音" in query or "noise" in query_lower:
            lines.append("- 噪声异常: 可能为气蚀或机械松动")
        if not has_motor and not has_pump:
            lines.append("\n**建议:** 提供更详细的设备信息和故障现象")

    if has_motor or has_pump:
        lines.append(f"\n**涉及设备:**")
        if has_motor:
            lines.append("- Motor MG 160 MA (11kW, 1480rpm) / MG 180 MB (45kW, 2960rpm)")
        if has_pump:
            lines.append("- Pump NK 80-250 / NK 80-160")
        if has_bearing:
            lines.append("- 轴承: 电机DE/NDE + 泵DE/NDE")

    lines.append("\n---")
    lines.append("*此为离线诊断结果。完整分析需构建知识图谱后运行 LightRAG 查询。*")
    lines.append("*运行 `python pump_main.py build` 构建知识图谱*")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
