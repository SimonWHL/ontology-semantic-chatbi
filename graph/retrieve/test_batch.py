#!/usr/bin/env python3
"""批量测试脚本：跑一组问题，输出实体识别结果 + 保存 JSON + 生成可视化 HTML。"""

import json
import re
import sys
import time
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from graph.loader import load_graph, Edge
from graph.index_builder import build_index
from extraction.entity_extractor import build_extractor
from retrieval.subgraph_builder import build_subgraph
from retrieval.subgraph_retriever_v2 import build_subgraph as build_subgraph_v2
from output.context_formatter import format_context

from output.html_visualizer import _generate_html


def _load_config() -> dict:
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_subgraph_builder():
    """根据 config 返回当前激活的 build_subgraph 函数和额外参数。"""
    config = _load_config()
    retriever = config.get("retriever", "v1")
    if retriever == "v2":
        v2_config = config.get("v2", {})
        return build_subgraph_v2, {
            "use_diffusion": v2_config.get("use_diffusion", True),
            "max_levels": v2_config.get("max_levels", 2),
            "min_cluster_size": v2_config.get("min_cluster_size", 2),
            "cross_cluster_threshold": v2_config.get("cross_cluster_threshold", 0.1),
            "use_llm": v2_config.get("use_llm", True),
            "force_rebuild_hg": v2_config.get("force_rebuild_hg", False),
        }, retriever
    return build_subgraph, {}, retriever

# 测试问题集
TEST_QUESTIONS = [
    # 场景1: 单城市维度 + 指标 + 函数
    "嘉兴市大项目商机数量和出库金额占比",
    # 场景2: 双维度(城市+行业) + 过滤器
    "杭州互联网行业在途商机总金额",
    # 场景3: 城市+产品类别维度
    "深圳地区服务器产品线赢单商机金额",
    # 场景4: 分组语义(各行业) + 过滤器
    "各行业的在途商机去重金额是多少",
    # 场景5: 无维度纯指标 + 函数(同比/增长率)
    "商机数量和出库金额的同比增长率",
    # 场景6: 单维度 + 事件过滤器(已签)
    "成都地区已签商机的出库金额",
]

RESULTS_DIR = BASE_DIR / "results"

def _safe_filename(text: str, max_len: int = 40) -> str:
    safe = re.sub(r'[\\/*?:"<>|\s]+', '_', text.strip())
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe.strip("_")


def _merge_sql_edges(subgraph: dict, all_edges_with_sql: list) -> dict:
    """子图生成完成后，合并相关的 SQL 逻辑边（仅 edge.sql_edge=True）。

    SQL 边只用于最终展示和上下文补充，不参与路径搜索。
    筛选规则：边必须是 SQL 边，且 from 和 to 节点都在子图中。

    同时生成 sql_analysis_paths：对于通过 SQL 边连接但无语义路径的
    维度-指标对，生成分析路径描述供下游使用。
    """
    sub_labels = {n.label for n in subgraph.get("nodes", [])}

    relevant_sql_edges = [
        e for e in all_edges_with_sql
        if e.sql_edge and e.from_label in sub_labels and e.to_label in sub_labels
    ]

    if not relevant_sql_edges:
        return dict(subgraph)

    paths = subgraph.get("paths", [])
    connected_pairs = set()
    for p in paths:
        between = p.get("between", [])
        if len(between) == 2:
            connected_pairs.add(tuple(sorted(between)))

    sql_analysis_paths = []
    seen_pairs = set()
    for e in relevant_sql_edges:
        pair_key = tuple(sorted([e.from_label, e.to_label]))
        if pair_key in connected_pairs or pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        sql_analysis_paths.append({
            "between": [e.from_label, e.to_label],
            "sql_clause": e.sql_clause or "WHERE",
            "condition_type": e.condition_type or "",
            "join_key": e.join_key or "",
            "display_label": e.display_label or "",
        })

    result = {
        **subgraph,
        "edges": subgraph.get("edges", []) + relevant_sql_edges,
        "sql_edge_count": len(relevant_sql_edges),
    }
    if sql_analysis_paths:
        result["sql_analysis_paths"] = sql_analysis_paths

    return result


def main():
    graph_path = BASE_DIR / "../data/商机.json"
    print(f"📂 加载图谱: {graph_path}")

    from main import _load, query
    from main import _load_config as _load_main_config

    graph, index, capability, sql_edges = _load(str(graph_path))
    extractor = build_extractor(
        node_labels=set(index.graph.node_map.keys()),
        node_map=index.graph.node_map,
        cache_path=BASE_DIR / "embeddings.pkl",
        graph_path=graph_path,
    )

    print("   ⏳ 初始化 Embedding...")
    extractor.initialize()
    print("   ✓ 就绪")

    cfg = _load_config()
    ext_cfg = cfg.get("extraction", {})
    use_embedding = ext_cfg.get("use_embedding", True)
    use_llm = ext_cfg.get("use_llm", True)
    max_hops = cfg.get("max_hops", 5)
    retriever_name = cfg.get("retriever", "v1")

    print(f"   节点: {len(graph.nodes)}  语义边: {len(graph.edges)}")
    print(f"   召回策略: {retriever_name}  Embedding: {'开启' if use_embedding else '关闭'}  LLM: {'开启(精筛)' if use_llm else '关闭'}")
    print()

    summary_rows = []

    for qi, q in enumerate(TEST_QUESTIONS, 1):
        print("=" * 70)
        print(f"[{qi}/{len(TEST_QUESTIONS)}] 🔍 问题: {q}")
        print("-" * 70)

        t0 = time.time()

        context = query(
            q, index, extractor,
            use_embedding=use_embedding,
            use_llm=use_llm,
            max_hops=max_hops,
            retriever=retriever_name,
            capability=capability,
            sql_edges=sql_edges,
            verbose=True,
        )

        elapsed = time.time() - t0
        print(f"   ⏱ 耗时: {elapsed:.2f}s")

        entities = context["entities"]
        attr_hits = [e for e in entities if index.graph.node_map.get(e) and index.graph.node_map[e].type == "Attribute"]
        safe_hits = [e for e in entities if e not in attr_hits]

        # ── 保存 JSON + HTML ──
        safe_name = _safe_filename(q)
        json_path = RESULTS_DIR / f"{qi:02d}_{safe_name}.json"
        html_path = RESULTS_DIR / f"{qi:02d}_{safe_name}.html"

        json_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        _generate_html(q, context["subgraph"], html_path)

        print(f"   💾 {json_path.name}  |  🌐 {html_path.name}")

        summary_rows.append({
            "id": qi,
            "question": q,
            "entities": entities,
            "attr": attr_hits,
            "safe": safe_hits,
            "nodes": context["meta"]["total_nodes"],
            "edges": context["meta"]["total_edges"],
            "paths": context["meta"]["total_paths"],
            "elapsed_s": round(elapsed, 2),
            "json": json_path.name,
            "html": html_path.name,
        })
        print()

    # ── 汇总表 ──
    print()
    print("=" * 80)
    print("📊 测试汇总")
    print("=" * 80)
    header = f"{'#':<3} {'问题':<32} {'维度':<16} {'固定概念':<30} {'N/E/P':<10} {'耗时':<8}"
    print(header)
    print("-" * 80)
    for r in summary_rows:
        attr_str = ",".join(r["attr"]) if r["attr"] else "-"
        safe_str = ",".join(r["safe"]) if r["safe"] else "-"
        nep = f"{r['nodes']}/{r['edges']}/{r['paths']}"
        elapsed_str = f"{r.get('elapsed_s', 0):.2f}s"
        q_display = r["question"][:30] + "…" if len(r["question"]) > 30 else r["question"]
        attr_display = attr_str[:14] + "…" if len(attr_str) > 14 else attr_str
        safe_display = safe_str[:28] + "…" if len(safe_str) > 28 else safe_str
        print(f"{r['id']:<3} {q_display:<32} {attr_display:<16} {safe_display:<30} {nep:<10} {elapsed_str:<8}")
    print("=" * 80)
    print(f"结果已保存到: {RESULTS_DIR.resolve()}")
    print(f"  共 {len(summary_rows)} 个 JSON + {len(summary_rows)} 个 HTML")

    summary_path = RESULTS_DIR / "_summary.json"
    summary_path.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  汇总: {summary_path.name}")


if __name__ == "__main__":
    main()
