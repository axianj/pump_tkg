"""
将离心泵故障知识图谱导入 Neo4j 图数据库

用法:
    python visualize/neo4j_import.py                          # 默认导入
    python visualize/neo4j_import.py --data-dir data/output    # 指定数据
    python visualize/neo4j_import.py --clear                   # 清空后导入
    python visualize/neo4j_import.py --temporal                # 导入时序关系
"""

import sys
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase, basic_auth

# ── 配置 ──────────────────────────────────────────────
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
DATABASE = "neo4j"
BATCH_SIZE = 200

TYPE_LABELS = {
    "设备": "Equipment",
    "部件": "Component",
    "故障类型": "Fault",
    "故障严重度": "Severity",
    "传感器": "Sensor",
    "监测点": "MeasurementPoint",
    "测量记录": "Measurement",
    "工况条件": "OperatingCondition",
    "信号特征": "SignalFeature",
    "维修操作": "Maintenance",
}


def connect():
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def clear_database(driver):
    print("[CLEAR] 清空图数据库...")
    with driver.session(database=DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("[CLEAR] 完成")


def create_constraints(driver):
    print("[SETUP] 创建约束和索引...")
    statements = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    ]
    with driver.session(database=DATABASE) as session:
        for s in statements:
            try:
                session.run(s)
            except Exception as e:
                print(f"  [WARN] {e}")
    print("[SETUP] 完成")


def import_entities(driver, entities):
    print(f"[IMPORT] 导入 {len(entities)} 个实体...")
    total = len(entities)

    with driver.session(database=DATABASE) as session:
        for start in range(0, total, BATCH_SIZE):
            batch = entities[start:start + BATCH_SIZE]
            params = []
            for e in batch:
                etype = e.get("type", "未知")
                label = TYPE_LABELS.get(etype, "Entity")
                params.append({
                    "id": str(e["id"]),
                    "name": str(e.get("name", e["id"])),
                    "type": etype,
                    "description": str(e.get("description", ""))[:5000],
                    "source": str(e.get("source", "")),
                    "degree": int(e.get("degree", 0)),
                })

            session.run("""
                UNWIND $batch AS props
                MERGE (e:Entity {id: props.id})
                SET e.name = props.name,
                    e.type = props.type,
                    e.description = props.description,
                    e.source = props.source,
                    e.degree = props.degree
            """, {"batch": params})

            pct = min(100, (start + BATCH_SIZE) * 100 // total)
            print(f"  实体: {min(start+BATCH_SIZE, total)}/{total} ({pct}%)")

    print(f"[IMPORT] 实体导入完成: {total} 个")


def import_relationships(driver, relationships):
    print(f"[IMPORT] 导入 {len(relationships)} 条关系...")
    total = len(relationships)
    skipped = 0

    with driver.session(database=DATABASE) as session:
        for start in range(0, total, BATCH_SIZE):
            batch = relationships[start:start + BATCH_SIZE]
            params = []
            for r in batch:
                params.append({
                    "source": str(r["source"]),
                    "target": str(r["target"]),
                    "relation": str(r.get("relation", "关联")),
                    "description": str(r.get("description", ""))[:2000],
                    "weight": float(r.get("weight", 1.0)),
                })

            result = session.run("""
                UNWIND $batch AS props
                MATCH (src:Entity {id: props.source})
                MATCH (tgt:Entity {id: props.target})
                MERGE (src)-[r:RELATES_TO {type: props.relation}]->(tgt)
                SET r.description = props.description,
                    r.weight = props.weight
                RETURN count(r) as created
            """, {"batch": params})

            created = result.single()["created"]
            batch_skipped = len(batch) - created
            skipped += batch_skipped

            pct = min(100, (start + BATCH_SIZE) * 100 // total)
            print(f"  关系: {min(start+BATCH_SIZE, total)}/{total} ({pct}%) "
                  f"[创建: {created}, 跳过: {batch_skipped}]")

    print(f"[IMPORT] 关系导入完成: {total - skipped} 创建, {skipped} 跳过")


def import_temporal_relations(driver, path: str):
    """导入时序关系文件（从 temporal_adapter 导出）"""
    if not os.path.exists(path):
        print(f"[INFO] 时序关系文件不存在: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        relations = json.load(f)

    if not relations:
        return

    print(f"[IMPORT] 导入 {len(relations)} 条时序关系...")
    with driver.session(database=DATABASE) as session:
        for start in range(0, len(relations), BATCH_SIZE):
            batch = relations[start:start + BATCH_SIZE]
            params = []
            for r in batch:
                params.append({
                    "source": r["source"],
                    "target": r["target"],
                    "relation": r.get("relation", "时序关联"),
                    "description": r.get("description", ""),
                })

            session.run("""
                UNWIND $batch AS props
                MATCH (src:Entity {id: props.source})
                MATCH (tgt:Entity {id: props.target})
                MERGE (src)-[r:TEMPORAL {type: props.relation}]->(tgt)
                SET r.description = props.description
            """, {"batch": params})

    print(f"[IMPORT] 时序关系导入完成: {len(relations)} 条")


def print_stats(driver):
    with driver.session(database=DATABASE) as session:
        node_count = session.run("MATCH (e:Entity) RETURN count(e) as cnt").single()["cnt"]
        rel_count = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as cnt").single()["cnt"]
        temp_count = session.run("MATCH ()-[r:TEMPORAL]->() RETURN count(r) as cnt").single()["cnt"]
        type_dist = session.run("""
            MATCH (e:Entity)
            RETURN e.type as type, count(e) as cnt
            ORDER BY cnt DESC
        """).data()

    print("\n" + "=" * 50)
    print("  Neo4j 导入完成 — 离心泵故障知识图谱")
    print("=" * 50)
    print(f"  实体节点:     {node_count}")
    print(f"  关系:         {rel_count}")
    print(f"  时序关系:     {temp_count}")
    print(f"\n  实体类型分布:")
    for row in type_dist:
        print(f"    {row['type']:12s}: {row['cnt']:4d}")
    print()
    print("  Cypher 查询示例:")
    print("    MATCH (n) RETURN n LIMIT 50")
    print("    MATCH (f:Fault)-[r]-(other) RETURN f, r, other")
    print("    MATCH (m:Measurement)-[r:TEMPORAL]->(next) RETURN m, r, next")
    print("    MATCH (e:Entity {type: '故障类型'})-[r]-(other) RETURN e, r, other")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="导入离心泵知识图谱到 Neo4j")
    parser.add_argument("--data-dir", default="data/output", help="数据目录")
    parser.add_argument("--clear", action="store_true", help="清空数据库后导入")
    parser.add_argument("--temporal", action="store_true", help="导入时序关系")
    parser.add_argument("--no-import", action="store_true", help="仅显示统计，不导入")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 加载数据
    entities_path = data_dir / "entities.json"
    rels_path = data_dir / "relationships.json"

    if not entities_path.exists():
        print(f"[ERROR] 未找到实体文件: {entities_path}")
        print("请先运行构建脚本: python pump_main.py build")
        sys.exit(1)

    with open(entities_path, "r", encoding="utf-8") as f:
        entities = json.load(f)
    with open(rels_path, "r", encoding="utf-8") as f:
        relationships = json.load(f)

    print(f"[LOAD] 实体: {len(entities)}, 关系: {len(relationships)}")

    if args.no_import:
        print("[INFO] 仅显示信息，不导入")
        return

    # 连接 Neo4j
    print(f"\n[CONNECT] 连接到 {NEO4J_URI}...")
    try:
        driver = connect()
        print("[CONNECT] 连接成功!")
    except Exception as e:
        print(f"[ERROR] 连接 Neo4j 失败: {e}")
        print("请确保 Neo4j Desktop 已运行")
        sys.exit(1)

    try:
        if args.clear:
            clear_database(driver)

        create_constraints(driver)
        import_entities(driver, entities)
        import_relationships(driver, relationships)

        if args.temporal:
            temporal_path = data_dir / "temporal_relations.json"
            import_temporal_relations(driver, str(temporal_path))

        print_stats(driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
