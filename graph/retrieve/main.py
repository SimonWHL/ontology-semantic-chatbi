"""M4: CLI 入口 v2.0。

核心流程：输入自然语言问题 → 抽取实体 → 构建子图 → 输出 JSON 上下文。

用法：
    # 单次查询
    python main.py "帮我统计杭州市大项目商机数量和出库金额的占比"

    # 交互模式
    python main.py

    # 指定图谱文件
    python main.py --graph ../data/商机.json

    # 启用 embedding 增强
    python main.py --embedding

    # LLM / Embedding 参数由 config.yaml 控制
    python main.py

    # 覆盖跳数
    python main.py --hops 6 "嘉兴市大项目商机数量和出库金额占比"

    # 输出 JSON 文件
    python main.py "问题" -o result.json
"""

import json
import sys
from pathlib import Path
from typing import Optional

import yaml

from loader import load_graph, SemanticGraph
from index_builder import build_index, GraphIndex
from entity_extractor import build_extractor, EntityExtractor
from subgraph_builder import build_subgraph
from subgraph_retriever_v2 import build_subgraph as build_subgraph_v2
from context_formatter import format_context, format_context_md

BASE_DIR = Path(__file__).resolve().parent


def _load_config() -> dict:
    """加载 config.yaml。"""
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_subgraph_builder():
    """根据 config 返回当前激活的 build_subgraph 函数、额外参数和策略名。"""
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


def _load(graph_path: str) -> tuple[SemanticGraph, GraphIndex, object, list]:
    """加载图谱 + 构建能力矩阵。返回 (graph, index, capability, sql_edges)。"""
    path = Path(graph_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    # 语义图谱（不含 SQL 边）
    graph = load_graph(path, include_sql_edges=False)
    index = build_index(graph)
    # 能力矩阵（含 SQL 边）
    from metric_capability import build_capability
    graph_sql = load_graph(path, include_sql_edges=True)
    capability = build_capability(graph_sql)
    sql_edges = graph_sql.edges
    return graph, index, capability, sql_edges


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

    # 生成 sql_analysis_paths：找出通过 SQL 边连接但语义路径中未覆盖的维度-指标对
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


def query(
    question: str,
    index: GraphIndex,
    extractor: EntityExtractor,
    *,
    use_embedding: bool = False,
    use_llm: bool = False,
    max_hops: int = 5,
    retriever: Optional[str] = None,
    capability: object = None,
    sql_edges: list = None,
) -> dict:
    """端到端查询：问题 → 上下文 JSON。

    Args:
        question: 自然语言问题
        index: 图谱索引
        extractor: 实体抽取器
        use_embedding: 是否启用 embedding 增强
        use_llm: 是否启用 LLM 增强
        max_hops: 最大跳数
        retriever: 召回策略 ("v1" / "v2")，None 则读取 config
        capability: 指标能力矩阵（MetricCapability 实例），用于生成维度-指标分析描述

    Returns:
        格式化后的上下文 dict
    """
    # Phase 1: 实体抽取
    cfg = _load_config()
    ext_cfg = cfg.get("extraction", {})
    entities = extractor.extract(
        question, use_embedding=use_embedding, use_llm=use_llm,
        embedding_threshold=ext_cfg.get("embedding_threshold", 0.35),
        embedding_max_results=ext_cfg.get("embedding_max_results", 15),
    )

    # Phase 2: 子图构建（根据 config 切换）
    if retriever is None:
        subgraph_fn, extra_kwargs, retriever_name = _get_subgraph_builder()
    elif retriever == "v2":
        config = _load_config()
        v2_config = config.get("v2", {})
        subgraph_fn, extra_kwargs = build_subgraph_v2, {
            "use_diffusion": v2_config.get("use_diffusion", True),
            "max_levels": v2_config.get("max_levels", 2),
            "min_cluster_size": v2_config.get("min_cluster_size", 2),
            "cross_cluster_threshold": v2_config.get("cross_cluster_threshold", 0.1),
            "use_llm": v2_config.get("use_llm", True),
            "force_rebuild_hg": v2_config.get("force_rebuild_hg", False),
        }
        retriever_name = "v2"
    else:
        subgraph_fn, extra_kwargs = build_subgraph, {}
        retriever_name = "v1"

    subgraph = subgraph_fn(entities, index, max_hops=max_hops, **extra_kwargs)

    # Phase 2.5: SQL 边后处理合并（SQL 边不参与路径搜索，仅在子图完成后补充）
    if sql_edges:
        subgraph = _merge_sql_edges(subgraph, sql_edges)

    # Phase 3: 格式化（含指标能力矩阵）
    cap_notes = ""
    if capability is not None:
        cap_notes = capability.describe_for_entities(entities, index.graph.node_map)
    context = format_context(question, entities, subgraph, meta={"max_hops": max_hops},
                            capability_notes=cap_notes)

    return context


def main():
    import argparse

    parser = argparse.ArgumentParser(description="语义知识图谱检索系统 v2.0")
    parser.add_argument("question", nargs="?", help="自然语言问题（不提供则进入交互模式）")
    parser.add_argument("--graph", default="../data/商机.json", help="图谱 JSON 文件路径")
    parser.add_argument("--hops", type=int, default=None, help="最大跳数（默认从 config 读取）")
    parser.add_argument("--retriever", choices=["v1", "v2"], default=None,
                        help="召回策略: v1=BFS最短路径, v2=分层聚合（默认读取 config.yaml）")
    parser.add_argument("-o", "--output", help="输出 JSON 文件路径")
    parser.add_argument("--md", action="store_true", help="以 Markdown 格式输出（调试用）")
    args = parser.parse_args()

    # 加载图谱
    print(f"加载图谱: {args.graph}", file=sys.stderr)
    graph_path = Path(args.graph)
    if not graph_path.is_absolute():
        graph_path = BASE_DIR / graph_path
    graph, index, capability, sql_edges = _load(str(graph_path))
    extractor = build_extractor(
        node_labels=set(index.graph.node_map.keys()),
        node_map=index.graph.node_map,
        cache_path=BASE_DIR / "embeddings.pkl",
        graph_path=graph_path,
    )
    extractor.initialize()
    print(f"节点: {len(graph.nodes)}  语义边: {len(graph.edges)}", file=sys.stderr)

    # 从 config 读取参数
    cfg = _load_config()
    ext_cfg = cfg.get("extraction", {})
    use_embedding = ext_cfg.get("use_embedding", True)
    use_llm = ext_cfg.get("use_llm", True)
    max_hops = args.hops if args.hops is not None else cfg.get("max_hops", 5)

    retriever_name = args.retriever or cfg.get("retriever", "v1")
    print(f"召回策略: {retriever_name}  Embedding: {use_embedding}  LLM: {use_llm}", file=sys.stderr)

    if args.question:
        # 单次查询
        result = query(
            args.question, index, extractor,
            use_embedding=use_embedding,
            use_llm=use_llm,
            max_hops=max_hops,
            retriever=args.retriever,
            capability=capability,
            sql_edges=sql_edges,
        )

        if args.md:
            entities = result["entities"]
            # 需要重建 subgraph dict 格式
            subgraph_raw = {
                "nodes": [index.graph.node_map[n["label"]] for n in result["subgraph"]["nodes"] if n["label"] in index.graph.node_map],
                "edges": _rebuild_edges(result["subgraph"]["edges"], index),
                "paths": result["paths"],
                "isolated": result["isolated"],
            }
            print(format_context_md(args.question, entities, subgraph_raw))
        else:
            output = json.dumps(result, ensure_ascii=False, indent=2)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output)
                print(f"已输出到: {args.output}", file=sys.stderr)
            else:
                print(output)
    else:
        # 交互模式
        print("\n知识图谱检索系统 v2.0", file=sys.stderr)
        print(f"Embedding: {use_embedding}  LLM: {use_llm}  最大跳数: {max_hops}", file=sys.stderr)
        print("输入自然语言问题，输入 quit 退出\n", file=sys.stderr)

        while True:
            try:
                line = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出", file=sys.stderr)
                break

            if not line:
                continue
            if line.lower() in ("quit", "exit"):
                break

            result = query(
                line, index, extractor,
                use_embedding=use_embedding,
                use_llm=use_llm,
                max_hops=max_hops,
                retriever=args.retriever,
                capability=capability,
                sql_edges=sql_edges,
            )

            if args.md:
                entities = result["entities"]
                subgraph_raw = {
                    "nodes": [index.graph.node_map[n["label"]] for n in result["subgraph"]["nodes"] if n["label"] in index.graph.node_map],
                    "edges": _rebuild_edges(result["subgraph"]["edges"], index),
                    "paths": result["paths"],
                    "isolated": result["isolated"],
                }
                print(format_context_md(line, entities, subgraph_raw))
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))

            print()


def _rebuild_edges(edge_dicts: list, index: GraphIndex) -> list:
    """从 context dict 重建 Edge 对象列表（用于 Markdown 格式化）。"""
    edges = []
    for ed in edge_dicts:
        for e in index.graph.edges:
            if e.from_label == ed["from"] and e.to_label == ed["to"] and e.label == ed["label"]:
                edges.append(e)
                break
    return edges


if __name__ == "__main__":
    main()
