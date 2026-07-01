# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

离心泵故障诊断时序知识图谱系统 (Centrifugal Pump Fault Diagnosis Temporal Knowledge Graph System). Builds a domain knowledge graph from pump fault documentation and sensor measurements, with ECharts visualization and Neo4j import. Integrates with LightRAG for RAG-based querying.

## Commands

```bash
# Initialize (creates default knowledge doc in data/knowledge/)
python pump_main.py init

# Build knowledge graph from docs in data/knowledge/ → data/output/
python pump_main.py build              # incremental (default)
python pump_main.py build --full       # full rebuild

# View system status (entity/relation counts, available commands)
python pump_main.py status

# Generate interactive ECharts HTML visualization
python pump_main.py to-echarts
python pump_main.py to-echarts --output custom.html

# Import into Neo4j (requires Neo4j running on neo4j://127.0.0.1:7687)
python pump_main.py to-neo4j
python pump_main.py to-neo4j --clear --temporal

# Inject simulated sensor measurements into the graph
python pump_main.py ingest-sensors

# Start Streamlit web UI
python pump_main.py web

# Interactive CLI query (local matching, no LLM needed)
python pump_main.py query
```

**One-shot full pipeline:** `python scripts/build_full.py` builds the base graph, generates ECharts HTML, and prints a summary in one step.

## Architecture

### Data Flow
```
data/knowledge/*.md|.txt|.pdf     data/sensors/*.csv
         │                                │
         ▼                                ▼
  scripts/build_index.py         core/temporal_adapter.py
  (LightRAG or simple mode)      (TemporalMeasurement entities
         │                         + prev/next relation chains)
         ▼                                │
  data/output/entities.json ◄────────────┘
  data/output/relationships.json
         │                    │
         ▼                    ▼
  visualize/echart_viz.py   visualize/neo4j_import.py
  → pump_graph_viz.html     → Neo4j graph database
         │
         ▼
  web/app.py (Streamlit — embeds the HTML, provides chat UI)
```

### Key Modules

- **`core/pump_domain.py`** — Domain ontology: `FaultTaxonomy` (all 20+ fault types with severity levels per setup), `EquipmentSpec` (motor/pump specs), `SensorConfig`, entity/relation type definitions, Chinese display name mappings (`FAULT_NAMES_ZH`, `SENSOR_NAMES_ZH`), and `PumpDomainKnowledge` (aggregates all domain knowledge, provides LLM system prompt).
- **`core/temporal_adapter.py`** — `TemporalAdapter`: parses measurement metadata from CSV file paths (directory naming convention like `Set_2/Speed_3/Fault_Angular_misalignment/Severity_5/`), builds `TemporalMeasurement` entities, constructs prev/next temporal relation chains within same fault+severity groups, and exports documents for LightRAG ingestion.
- **`scripts/build_index.py`** — Knowledge graph construction. Attempts LightRAG first; falls back to "simple mode" that generates structured JSON directly from `PumpDomainKnowledge` without requiring LLM/embeddings. Outputs `entities.json` and `relationships.json`.
- **`scripts/build_full.py`** — One-shot script that builds the base graph AND generates the ECharts HTML in a single run.
- **`scripts/ingest_sensors.py`** — Generates simulated sensor measurements with fault-specific spectral features (1X/2X/BPFO/BPFI/BSF amplitudes vary by fault type and severity). Supports `--simulate` and `--from-file` modes. Injects measurement entities, feature entities, and temporal chains into the existing graph.
- **`visualize/echart_viz.py`** — Converts `entities.json` + `relationships.json` into a self-contained ECharts HTML file with force-directed layout, type-colored nodes, click-to-inspect detail panel, and legend filtering. Usable standalone or via CLI.
- **`visualize/neo4j_import.py`** — Batch-imports entities and relationships into Neo4j using Cypher MERGE. Creates constraints, maps Chinese entity types to Neo4j labels. Supports temporal relation import. Default credentials: `neo4j://127.0.0.1:7687`, user `neo4j`, password `12345678`.
- **`web/app.py`** — Streamlit UI with 4 tabs: chat-based diagnosis (local pattern matching against fault names), sensor analysis (channel/fault selector with static metrics), knowledge graph viewer (embeds the ECharts HTML), and knowledge base browser. No LLM dependency in offline mode.
- **`demo_incremental.py`** — Standalone demo of LightRAG incremental insert (two rounds of document insertion, showing storage growth without rebuild). Uses dummy LLM/embedding functions.

### Domain Model

Two experimental setups with different motors and fault configurations:
- **Setup 1** (Motor MG 160 MA, 11kW, 1480rpm): 11 fault types (impeller, bearing BPFO/BPFI/BSF/contaminated, soft foot, loose foot, broken rotor bar, stator short, pump bearing) at 3 speeds (50%/75%/100%)
- **Setup 2** (Motor MG 180 MB, 45kW, 2960rpm): 9 fault types (angular/parallel/combined misalignment, motor/pump unbalance, coupling damage, suction/discharge cavitation, bent shaft) at fixed 70% speed

5 vibration sensors (Ch1-Ch5): Motor NDE horizontal, Motor DE vertical, Motor DE axial, Pump DE horizontal, Pump NDE vertical. Sampling at 20kHz, 12s per sample.

### Graph Schema

Entity types (in Chinese): 设备, 部件, 故障类型, 故障严重度, 传感器, 监测点, 测量记录, 维修操作, 工况条件, 信号特征

Relation types: 包含, 安装于, 监测, 表现为, 导致, 严重度, 测量于, 前次测量, 后续测量, 工况, 维修

Measurement entities are chained via `前次测量`/`后续测量` (prev/next) relations within the same fault type + severity group, enabling temporal traversal.

### Key Patterns
- **Incremental by default**: `build` uses `--incremental` by default; sensor ingestion merges new entities/relations into existing JSON without rebuilding.
- **LightRAG optional**: The `build` command tries LightRAG first but gracefully degrades to a simple mode that generates structured JSON from the domain ontology — no LLM or embedding API required for basic functionality.
- **Simulated data**: `ingest-sensors --simulate` generates realistic synthetic measurements with fault-specific spectral signatures (e.g., misalignment boosts 2X, unbalance boosts 1X, bearing faults boost BPFO/BPFI/BSF).
