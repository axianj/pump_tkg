# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

离心泵故障诊断时序知识图谱平台 (Centrifugal Pump Fault Diagnosis Temporal Knowledge Graph Platform).

基于时序四元组 (h, r, t, [t₁, t₂]) 的 Agentic RAG 系统，用于离心泵设备故障诊断与预测。

两套并行方案用于框架选型对比：
- **方案A**：LightRAG + 自建 TKG Core + Langgraph Agent (推荐，低成本)
- **方案B**：Graphiti (原生时序KG) + Langgraph Agent

## Quick Start

```bash
# 构建基础知识图谱
python scripts/build_full.py

# 生成 ECharts 可视化
python pump_main.py to-echarts

# 导入 Neo4j
python pump_main.py to-neo4j --clear

# 启动 Web 界面
python pump_main.py web

# 运行方案对比评估
python evaluation/run_eval.py --compare
```

## Architecture

### Data Flow
```
设备文档 (Markdown/PDF)                传感器 CSV (振动/电流)
        │                                    │
        ▼                                    ▼
core/kg_builder.py                   core/temporal/sensor_bridge.py
(语义实体 + 因果链)                   (特征提取 + 三层注入)
        │                                    │
        ▼                                    ▼
entities.json + relationships.json    Neo4j (TKG) + Chroma (向量)
        │                                    │
        └────────────────┬───────────────────┘
                         ▼
              Langgraph Agent (Router)
              ├── vector_retrieve (Chroma)
              ├── temporal_retrieve (TKG Core)
              └── multimodal_retrieve (RAG-Anything)
                         │
                         ▼
              diagnosis (Ollama LLM)
```

### Three-Layer Storage Architecture
```
Neo4j (语义层)    — 实体+关系+时序四元组 (Cypher 图遍历)
Chroma (相似度层)  — 传感器频域特征向量 (相似故障检索)
CSV  (不常驻内存)  — 24万采样点/通道原始时序
```

### Key Modules
| 模块 | 文件 | 作用 |
|------|------|------|
| 领域本体 | `core/pump_domain.py` | 故障分类、设备规格、传感器配置 |
| KG 构建器 | `core/kg_builder.py` | 统一图谱构建 (实体+因果链+推理规则) |
| 时序四元组 | `core/temporal/temporal_quad.py` | (h,r,t,[t₁,t₂]) 数据模型 + Allen 区间代数 |
| 时序存储 | `core/temporal/temporal_store.py` | Neo4j 时序查询与路径追踪 |
| 时序推理 | `core/temporal/temporal_reasoner.py` | 传递性推理 + 因果链分析 |
| 传感器桥接 | `core/temporal/sensor_bridge.py` | CSV→特征提取→三层注入 |
| 向量库 | `core/vector_store.py` | Chroma 文档检索 + 测量向量相似度 |
| 混合检索 | `core/hybrid_search.py` | RRF 融合 (Chroma + Neo4j) |
| Graphiti | `core/graphiti_engine.py` | Graphiti 封装 (自动回退到 TemporalStore) |
| Kùzu | `core/graph_store.py` | 嵌入式图数据库 (桌面应用替代 Neo4j) |
| Agent A | `agents/graph_agent_a.py` | 方案A Agent (Router→检索→诊断) |
| Agent B | `agents/graph_agent_b.py` | 方案B Agent (相同逻辑，不同后端) |
| 预测管道 | `prediction/__init__.py` | SignalEncoder + TKGEncoder + FusionPredictor |

### LLM Layer (三层混合架构)
```
Layer 1: 规则闭合 (Allen代数 + 故障传导规则库) — 100%确定
Layer 2: KICGPT + Qwen3-14B — 语义补全，KG补全
Layer 3: DeepSeek-R1-14B — 复杂多步推理 (LangGraph)
```

## Key Commands

```bash
# 知识图谱
python scripts/build_full.py              # 一键构建+可视化
python pump_main.py to-echarts           # ECharts 可视化
python pump_main.py to-neo4j --clear     # Neo4j 导入

# 传感器数据
python scripts/ingest_real_sensors.py --simulate --count 30
python scripts/ingest_real_sensors.py --from-dir <path_to_csv>

# 评估
python evaluation/run_eval.py --compare         # 方案对比
python evaluation/run_eval.py --approach a      # 单独评估方案A

# Web
python pump_main.py web                         # Streamlit UI

# 预测管道测试
python -c "from prediction import create_prediction_pipeline; ..."
```

## Configuration

- Neo4j: `neo4j://127.0.0.1:7687`, user `neo4j`, password `12345678`
- Ollama: 本地服务, 模型 `qwen3:14b` / `deepseek-r1:14b`
- Chroma: 持久化到 `data/output/vector_store/`
- Kùzu: 嵌入式，目录 `data/output/kuzu_db/`
