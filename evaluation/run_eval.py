"""
评估框架 — 自动化运行测试查询、计算指标、生成对比报告

Phase 5: 对比 Baseline（纯文本 KG）vs 方案A（LightRAG+自建TKG）vs 方案B（Graphiti+TKG）

指标:
- Hit@1: Top-1 结果包含正确答案的比例
- MRR: 平均倒数排名
- 时序推理准确率: 时序查询的正确率
- 查询延迟 P50/P95: 毫秒级
- 回答质量评分: LLM 输出的专业性、完整性

用法:
    python evaluation/run_eval.py --approach a    # 单独评估方案A
    python evaluation/run_eval.py --compare       # 对比所有方案
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 评估指标 ────────────────────────────────────────

def calculate_hit_at_k(
    results: List[Dict], expected_entities: List[str], k: int = 1
) -> bool:
    """
    计算 Hit@K。

    Args:
        results: 检索结果列表 [{id, ...}]
        expected_entities: 期望的实体ID列表
        k: Top-K

    Returns: True/False
    """
    top_k_ids = {r.get("id", r.get("measurement_id", "")) for r in results[:k]}
    expected_set = set(expected_entities)
    return bool(top_k_ids & expected_set) if expected_set else True


def calculate_mrr(
    results: List[Dict], expected_entities: List[str]
) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)。

    Returns: 1/rank (1.0 表示第一个结果就命中)
    """
    if not expected_entities:
        return 1.0
    expected_set = set(expected_entities)
    for rank, result in enumerate(results, start=1):
        rid = result.get("id", result.get("measurement_id", ""))
        if rid in expected_set:
            return 1.0 / rank
    return 0.0


def calculate_temporal_accuracy(
    diagnosis: str, expected_chain: List[str]
) -> float:
    """
    评估时序推理准确率。

    检查诊断回答中是否提到了期望的故障传播链中的关键实体。

    Returns: [0, 1] 比例分数
    """
    if not expected_chain:
        return 1.0

    hit_count = 0
    for entity in expected_chain:
        # 检查实体名称的简化形式是否在诊断中出现
        if entity.lower().replace("fault_", "").replace("_", " ") in diagnosis.lower():
            hit_count += 1

    return hit_count / len(expected_chain)


# ── 评估运行器 ──────────────────────────────────────

class Evaluator:
    """对比评估运行器"""

    def __init__(self, agent_a=None, agent_b=None):
        self.agent_a = agent_a
        self.agent_b = agent_b

    def load_test_queries(self, path: str = "evaluation/test_queries.json") -> List[Dict]:
        with open(Path(PROJECT_ROOT) / path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_approach(
        self, agent, test_queries: List[Dict], approach_name: str
    ) -> Dict:
        """
        评估单个方案。
        """
        metrics = {
            "name": approach_name,
            "total_queries": len(test_queries),
            "hit_at_1": 0,
            "hit_at_3": 0,
            "mrr_total": 0.0,
            "temporal_accuracy_total": 0.0,
            "query_times_ms": [],
            "by_level": {},
        }

        for i, q in enumerate(test_queries):
            level = q.get("level", "L1")
            if level not in metrics["by_level"]:
                metrics["by_level"][level] = {
                    "total": 0, "hit_at_1": 0, "mrr": 0.0, "temporal_acc": 0.0,
                }

            # 测量延迟
            start = time.time()
            try:
                result = agent.query(q["query"])
            except Exception as e:
                result = {"diagnosis": f"Error: {e}", "route": "error"}
            elapsed_ms = (time.time() - start) * 1000
            metrics["query_times_ms"].append(elapsed_ms)

            diagnosis = result.get("diagnosis", "")

            # 计算指标
            expected_entities = q.get("expected_entities", [])
            if q.get("expected_entity"):
                expected_entities = [q["expected_entity"]]

            # 简化结果构造（agents 不是纯检索，所以从 diagnosis 提取）
            # 实际上需要从 agent.agent.invoke 的 state 中获取 retrieval_results
            # 这里用 diagnosis 文本来近似评估

            # 检查回答中是否包含期望答案关键词
            expected_answer = q.get("expected_answer", "")
            has_answer = expected_answer.lower() in diagnosis.lower() if expected_answer else True

            # 检查期望实体
            entity_hit = any(
                e.lower().replace("fault_", "").replace("_", " ")
                in diagnosis.lower()
                for e in expected_entities
            ) if expected_entities else has_answer

            hit1 = 1 if entity_hit else 0
            metrics["hit_at_1"] += hit1

            temporal_acc = calculate_temporal_accuracy(
                diagnosis, q.get("expected_temporal_chain", [])
            )
            metrics["temporal_accuracy_total"] += temporal_acc

            # 层级统计
            lv = metrics["by_level"][level]
            lv["total"] += 1
            lv["hit_at_1"] += hit1
            lv["temporal_acc"] += temporal_acc

        n = len(test_queries)
        metrics["hit_at_1"] /= n
        metrics["temporal_accuracy_total"] /= n
        metrics["query_time_p50"] = float(np.percentile(metrics["query_times_ms"], 50))
        metrics["query_time_p95"] = float(np.percentile(metrics["query_times_ms"], 95))
        metrics["query_time_mean"] = float(np.mean(metrics["query_times_ms"]))

        for lv in metrics["by_level"]:
            data = metrics["by_level"][lv]
            if data["total"] > 0:
                data["hit_at_1"] /= data["total"]
                data["temporal_acc"] /= data["total"]

        return metrics

    def run_comparison(self) -> Dict:
        """运行完整对比"""
        test_queries = self.load_test_queries()
        print(f"[Eval] 测试查询: {len(test_queries)} 条 "
              f"(L1: {sum(1 for q in test_queries if q['level']=='L1')}, "
              f"L2: {sum(1 for q in test_queries if q['level']=='L2')}, "
              f"L3: {sum(1 for q in test_queries if q['level']=='L3')}, "
              f"L4: {sum(1 for q in test_queries if q['level']=='L4')})")

        results = []

        if self.agent_a:
            print(f"\n[Eval] 评估方案A...")
            r = self.evaluate_approach(self.agent_a, test_queries, "Approach A: LightRAG + 自建TKG")
            results.append(r)
            print(f"  Hit@1={r['hit_at_1']:.3f}, Time P50={r['query_time_p50']:.0f}ms, P95={r['query_time_p95']:.0f}ms")

        if self.agent_b:
            print(f"\n[Eval] 评估方案B...")
            r = self.evaluate_approach(self.agent_b, test_queries, "Approach B: Graphiti + TKG")
            results.append(r)
            print(f"  Hit@1={r['hit_at_1']:.3f}, Time P50={r['query_time_p50']:.0f}ms, P95={r['query_time_p95']:.0f}ms")

        comparison = self._generate_comparison_report(results)
        return comparison

    def _generate_comparison_report(self, results: List[Dict]) -> Dict:
        """生成对比报告"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "approaches": [],
            "recommendation": "",
            "decision_applied": "",
        }

        for r in results:
            report["approaches"].append({
                "name": r["name"],
                "hit_at_1": r["hit_at_1"],
                "temporal_accuracy": r["temporal_accuracy_total"],
                "query_time_p50_ms": r["query_time_p50"],
                "query_time_p95_ms": r["query_time_p95"],
                "by_level": r["by_level"],
            })

        # 决策标准应用
        if len(results) >= 2:
            a = results[0]
            b = results[1]
            diff = abs(a["hit_at_1"] - b["hit_at_1"]) / max(a["hit_at_1"], b["hit_at_1"], 0.01)
            temporal_diff = abs(
                a["temporal_accuracy_total"] - b["temporal_accuracy_total"]
            ) / max(a["temporal_accuracy_total"], b["temporal_accuracy_total"], 0.01)

            if diff < 0.15:
                report["recommendation"] = "差异 <15% — 选择部署成本更低的方案A"
                report["decision_applied"] = "cost_optimized"
            elif b.get("temporal_accuracy_total", 0) > a.get("temporal_accuracy_total", 0) * 1.2:
                report["recommendation"] = "方案B 时序推理领先 >20% — 选择方案B"
                report["decision_applied"] = "performance_optimized"
            else:
                report["recommendation"] = "各有优劣 — 考虑混合方案"
                report["decision_applied"] = "hybrid"

        return report


# ── 报告输出 ────────────────────────────────────────

def print_report(report: Dict):
    """格式化打印评估报告"""
    print("\n" + "=" * 70)
    print("  离心泵故障诊断系统 — 方案对比评估报告")
    print("=" * 70)
    print(f"  生成时间: {report['generated_at']}")
    print()

    for approach in report["approaches"]:
        print(f"\n  [{approach['name']}]")
        print(f"    Hit@1:           {approach['hit_at_1']:.3f}")
        print(f"    时序推理准确率:   {approach['temporal_accuracy_total']:.3f}")
        print(f"    查询延迟 P50:     {approach['query_time_p50']:.0f} ms")
        print(f"    查询延迟 P95:     {approach['query_time_p95']:.0f} ms")
        print()

    print(f"  推荐方案: {report.get('recommendation', 'N/A')}")
    print(f"  决策依据: {report.get('decision_applied', 'N/A')}")
    print("=" * 70)


# ── CLI ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="方案对比评估")
    parser.add_argument("--approach", choices=["a", "b", "baseline"], default="a")
    parser.add_argument("--compare", action="store_true", help="对比所有方案")
    parser.add_argument("--output", default="docs/comparison_report.json")
    args = parser.parse_args()

    from agents.graph_agent_a import SharedRetrievalComponents, AgenticDiagnosisEngine
    from agents.graph_agent_b import AgenticDiagnosisEngineB
    from core.temporal import TemporalQuadStore, TemporalReasoner, TemporalQuad, TemporalRelation

    # 构建时序图（两个 Agent 共享）
    tstore = TemporalQuadStore()
    t1 = datetime(2020, 7, 1, 8, 0, 0)
    t2 = datetime(2020, 7, 1, 12, 0, 0)
    t3 = datetime(2020, 7, 1, 16, 0, 0)

    # 载入因果链
    import json
    entities_path = Path("data/output/entities.json")
    rels_path = Path("data/output/relationships.json")
    if entities_path.exists() and rels_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
        with open(rels_path, "r", encoding="utf-8") as f:
            rels = json.load(f)
        for r in rels:
            if r["relation"] in ("导致", "关联"):
                try:
                    tstore.add(TemporalQuad(
                        r["source"], TemporalRelation.CAUSES, r["target"],
                        t1, t2, confidence=r.get("weight", 0.8),
                    ))
                except Exception:
                    pass

    reasoner = TemporalReasoner(tstore)
    components = SharedRetrievalComponents(
        temporal_reasoner=reasoner,
        llm_model="qwen3:14b",
    )

    agent_a = AgenticDiagnosisEngine(components)
    agent_b = AgenticDiagnosisEngineB(components)

    evaluator = Evaluator(agent_a=agent_a, agent_b=agent_b)

    if args.compare:
        report = evaluator.run_comparison()
        print_report(report)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[Eval] 报告已保存: {output_path}")
    elif args.approach == "a":
        queries = evaluator.load_test_queries()
        r = evaluator.evaluate_approach(agent_a, queries, "Approach A")
        print_report({"generated_at": datetime.now().isoformat(),
                       "approaches": [r], "recommendation": "", "decision_applied": ""})
    elif args.approach == "b":
        queries = evaluator.load_test_queries()
        r = evaluator.evaluate_approach(agent_b, queries, "Approach B")
        print_report({"generated_at": datetime.now().isoformat(),
                       "approaches": [r], "recommendation": "", "decision_applied": ""})


if __name__ == "__main__":
    main()
