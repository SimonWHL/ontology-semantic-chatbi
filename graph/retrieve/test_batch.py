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

from loader import load_graph, Edge
from index_builder import build_index
from entity_extractor import build_extractor
from subgraph_builder import build_subgraph
from subgraph_retriever_v2 import build_subgraph as build_subgraph_v2
from context_formatter import format_context

# ── 复用 test_interactive.py 的 HTML 模板 ──
from test_interactive import _generate_html, format_paths_text


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
    graph = load_graph(graph_path, include_sql_edges=False)

    # 加载含 SQL 边的图谱用于指标能力矩阵
    from metric_capability import build_capability
    graph_with_sql = load_graph(graph_path, include_sql_edges=True)
    capability = build_capability(graph_with_sql)

    index = build_index(graph)
    extractor = build_extractor(
        node_labels=set(index.graph.node_map.keys()),
        node_map=index.graph.node_map,
        cache_path=BASE_DIR / "embeddings.pkl",
        graph_path=graph_path,
    )

    # 启动时一次性初始化 Embedding（缓存命中→直接加载，否则编码+落盘）
    print("   ⏳ 初始化 Embedding...")
    extractor.initialize()
    print("   ✓ 就绪")

    cfg = _load_config()
    ext_cfg = cfg.get("extraction", {})
    use_embedding = ext_cfg.get("use_embedding", True)
    use_llm = ext_cfg.get("use_llm", True)
    max_hops = cfg.get("max_hops", 5)

    subgraph_fn, extra_kwargs, retriever_name = _get_subgraph_builder()
    print(f"   节点: {len(graph.nodes)}  语义边: {len(graph.edges)}")
    print(f"   召回策略: {retriever_name}  Embedding: {'开启' if use_embedding else '关闭'}  LLM: {'开启(精筛)' if use_llm else '关闭'}")
    print()

    summary_rows = []

    for qi, q in enumerate(TEST_QUESTIONS, 1):
        print("=" * 70)
        print(f"[{qi}/{len(TEST_QUESTIONS)}] 🔍 问题: {q}")
        print("-" * 70)

        t0 = time.time()

        # Phase 1: 实体抽取
        entities = extractor.extract(
            q, use_embedding=use_embedding, use_llm=use_llm,
            embedding_threshold=ext_cfg.get("embedding_threshold", 0.35),
            embedding_max_results=ext_cfg.get("embedding_max_results", 15),
        )
        print(f"📌 最终实体: {entities}")

        rule_entities = extractor.extract_by_rules(q)
        kw_results = extractor.extract_by_embedding_keywords(
            q,
            threshold=ext_cfg.get("embedding_threshold", 0.35),
            max_results=ext_cfg.get("embedding_max_results", 15),
        )
        dim_results = extractor.extract_by_embedding(q, top_k_attr=3, top_k_safe=3)

        # 候选池（排除规则已匹配的）
        rule_set = set(rule_entities)
        emb_candidates = [l for l, _ in kw_results if l not in rule_set]
        for d in dim_results:
            if d not in rule_set and d not in emb_candidates:
                emb_candidates.append(d)

        attr_hits = [e for e in entities if extractor.node_map.get(e) and extractor.node_map[e].type == "Attribute"]
        safe_hits = [e for e in entities if e not in attr_hits]

        print(f"   L1 规则匹配(确定保留): {rule_entities}")
        print(f"   L2 Embedding宽召回: {[(l, f'{s:.3f}') for l, s in kw_results]}")
        print(f"   L3 Embedding维度: {dim_results}")
        print(f"   → 候选池(去重后): {emb_candidates}")
        print(f"   → 维度节点: {attr_hits}")
        print(f"   → 固定概念: {safe_hits}")

        # Phase 2: 子图构建（SQL 边不参与路径搜索）
        subgraph = subgraph_fn(entities, index, max_hops=max_hops, **extra_kwargs)
        # 后处理：合并 SQL 边（仅 sql_edge=True 的边）
        subgraph_with_sql = _merge_sql_edges(subgraph, graph_with_sql.edges)
        paths = subgraph_with_sql.get("paths", subgraph.get("paths", []))
        isolated = subgraph_with_sql.get("isolated", subgraph.get("isolated", []))
        sql_edge_count = subgraph_with_sql.get("sql_edge_count", 0)
        print(f"   子图: {len(subgraph_with_sql.get('nodes', []))} 节点, {len(subgraph_with_sql.get('edges', []))} 边 (含 {sql_edge_count} SQL边), {len(paths)} 路径")
        if isolated:
            print(f"   ⚠ 孤立节点: {isolated}")

        for i, p in enumerate(paths[:3], 1):
            between = p.get("between", [])
            nodes = p.get("nodes", [])
            edges = p.get("edges", [])
            parts = []
            for j, nl in enumerate(nodes):
                parts.append(nl)
                if j < len(edges):
                    display = edges[j].get("display_label", edges[j].get("label", "?"))
                    parts.append(f"-[{display}]->")
            chain = " ".join(parts)
            between_str = " & ".join(between)
            print(f"   [{i}] {between_str}: {chain}")
        if len(paths) > 3:
            print(f"   ... 共 {len(paths)} 条路径")

        elapsed = time.time() - t0
        print(f"   ⏱ 耗时: {elapsed:.2f}s")

        # ── 保存 JSON + HTML ──
        # 生成指标分析能力描述（独立模块，不参与检索）
        cap_notes = capability.describe_for_entities(entities, graph.node_map)
        context = format_context(q, entities, subgraph_with_sql, meta={"max_hops": max_hops},
                                capability_notes=cap_notes)
        safe_name = _safe_filename(q)
        json_path = RESULTS_DIR / f"{qi:02d}_{safe_name}.json"
        html_path = RESULTS_DIR / f"{qi:02d}_{safe_name}.html"

        json_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        # _generate_html 需要的是序列化好的 subgraph dict（nodes/edges 是 dict 而非对象）
        _generate_html(q, context["subgraph"], html_path)

        print(f"   💾 {json_path.name}  |  🌐 {html_path.name}")

        # 记录汇总
        summary_rows.append({
            "id": qi,
            "question": q,
            "entities": entities,
            "attr": attr_hits,
            "safe": safe_hits,
            "nodes": len(subgraph_with_sql.get("nodes", [])),
            "edges": len(subgraph_with_sql.get("edges", [])),
            "sql_edges": sql_edge_count,
            "paths": len(paths),
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
    header = f"{'#':<3} {'问题':<32} {'维度':<16} {'固定概念':<30} {'N/E/SQL/P':<14} {'耗时':<8}"
    print(header)
    print("-" * 80)
    for r in summary_rows:
        attr_str = ",".join(r["attr"]) if r["attr"] else "-"
        safe_str = ",".join(r["safe"]) if r["safe"] else "-"
        nep = f"{r['nodes']}/{r['edges']}/{r.get('sql_edges',0)}/{r['paths']}"
        elapsed_str = f"{r.get('elapsed_s', 0):.2f}s"
        # 截断显示
        q_display = r["question"][:30] + "…" if len(r["question"]) > 30 else r["question"]
        attr_display = attr_str[:14] + "…" if len(attr_str) > 14 else attr_str
        safe_display = safe_str[:28] + "…" if len(safe_str) > 28 else safe_str
        print(f"{r['id']:<3} {q_display:<32} {attr_display:<16} {safe_display:<30} {nep:<14} {elapsed_str:<8}")
    print("=" * 80)
    print(f"结果已保存到: {RESULTS_DIR.resolve()}")
    print(f"  共 {len(summary_rows)} 个 JSON + {len(summary_rows)} 个 HTML")

    # 保存汇总 JSON
    summary_path = RESULTS_DIR / "_summary.json"
    summary_path.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  汇总: {summary_path.name}")


if __name__ == "__main__":
    main()
