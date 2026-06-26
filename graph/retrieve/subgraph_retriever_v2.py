"""M2-v2: 子图召回方法 v2 — 分层知识图谱聚合（Hierarchical Graph Aggregation）。

策略：自底向上递归构建多层级连通语义网络

1. 语义聚类：对底层实体做 Embedding 编码，GMM 完成语义相似实体分簇
2. 聚合实体生成：LLM 为每个簇生成高层抽象概念节点
3. 跨簇关联生成：统计簇间实体关联强度，超阈值由 LLM 生成高层抽象关系
4. 递归迭代：生成多层图谱 H={G0, G1, ...Gk}

检索时：
- 在各层匹配实体
- 在最高匹配层找抽象路径（语义骨架）
- 展开回 G0 底层做细粒度路径填充

接口与 v1 的 build_subgraph 保持一致，方便切换。
"""

import heapq
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from index_builder import GraphIndex
from loader import Edge, Node

from hierarchical_aggregator import (
    HierarchicalGraph,
    build_hierarchical_graph,
    build_hierarchical_subgraph,
)

BASE_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════
# 分层图谱缓存
# ══════════════════════════════════════════════════════════════

_hg_cache: Optional[HierarchicalGraph] = None
_hg_cache_graph_hash: str = ""


def _get_or_build_hg(
    graph,
    *,
    max_levels: int = 2,
    min_cluster_size: int = 2,
    cross_cluster_threshold: float = 0.1,
    use_llm: bool = True,
    force_rebuild: bool = False,
) -> HierarchicalGraph:
    """获取或构建分层聚合图谱（带缓存）。"""
    global _hg_cache, _hg_cache_graph_hash

    # 用节点/边数量和结构生成简单哈希来判断图谱是否变化
    graph_hash = f"{len(graph.nodes)}:{len(graph.edges)}:{graph.domain}"
    if not force_rebuild and _hg_cache is not None and _hg_cache_graph_hash == graph_hash:
        return _hg_cache

    cache_dir = BASE_DIR / ".hg_cache"
    hg = build_hierarchical_graph(
        graph,
        max_levels=max_levels,
        min_cluster_size=min_cluster_size,
        cross_cluster_threshold=cross_cluster_threshold,
        cache_dir=cache_dir,
        use_llm=use_llm,
    )

    _hg_cache = hg
    _hg_cache_graph_hash = graph_hash
    return hg


# ══════════════════════════════════════════════════════════════
# 公共入口（与 v1 接口一致）
# ══════════════════════════════════════════════════════════════

def build_subgraph(
    entities: List[str],
    index: GraphIndex,
    *,
    max_hops: int = 5,
    max_paths_per_pair: int = 2,
    expand_neighbors: int = 0,
    include_sql_edges: bool = False,
    # v2 特有参数
    use_diffusion: bool = True,
    diffusion_depth: int = 3,
    diffusion_max_nodes: int = 50,
    # 分层聚合参数
    max_levels: int = 2,
    min_cluster_size: int = 2,
    cross_cluster_threshold: float = 0.1,
    use_llm: bool = True,
    force_rebuild_hg: bool = False,
    sql_edges: list = None,  # SQL 边列表，用于注入直接捷径
) -> dict:
    """构建子图（v2 分层聚合策略）。

    当 use_diffusion=True 时，使用分层知识图谱聚合检索：
    1. 构建/加载多层聚合图谱 H={G0, G1, ...}
    2. 在各层匹配问题实体
    3. 在抽象层找语义骨架路径
    4. 展开回 G0 做细粒度填充

    Args:
        entities: 实体 label 列表
        index: 图谱索引
        max_hops: 最大跳数
        max_paths_per_pair: 每对节点最多保留几条路径
        expand_neighbors: 额外展开邻居跳数（默认 0）
        include_sql_edges: 是否包含 SQL 逻辑边
        use_diffusion: True=分层聚合, False=传统 BFS 最短路径
        diffusion_depth: （保留，分层模式下不使用）
        diffusion_max_nodes: （保留，分层模式下不使用）
        max_levels: 最大层级数（含 G0）
        min_cluster_size: 每簇最少实体数
        cross_cluster_threshold: 跨簇关联强度阈值
        use_llm: 是否启用 LLM 生成抽象概念
        force_rebuild_hg: 强制重建分层图谱

    Returns:
        {
            "nodes": [Node, ...],
            "edges": [Edge, ...],
            "paths": [{"between": [a,b], "nodes": [...], "edges": [...]}, ...],
            "isolated": [label, ...],
            "hierarchy": {...}  # v2 专属：分层信息
        }
    """
    graph = index.graph
    node_map = graph.node_map

    # 过滤无效实体
    valid_entities = [e for e in entities if e in node_map]
    invalid_entities = [e for e in entities if e not in node_map]

    if not valid_entities:
        return {"nodes": [], "edges": [], "paths": [], "isolated": invalid_entities}

    if not use_diffusion:
        # 传统 BFS 最短路径模式（兼容旧行为）
        return _build_subgraph_bfs(
            valid_entities, invalid_entities, index,
            max_hops=max_hops,
            max_paths_per_pair=max_paths_per_pair,
            expand_neighbors=expand_neighbors,
            include_sql_edges=include_sql_edges,
        )

    # ── 分层聚合模式 ──
    print(f"  [v2-hierarchical] 构建分层图谱 (max_levels={max_levels})...")
    hg = _get_or_build_hg(
        graph,
        max_levels=max_levels,
        min_cluster_size=min_cluster_size,
        cross_cluster_threshold=cross_cluster_threshold,
        use_llm=use_llm,
        force_rebuild=force_rebuild_hg,
    )
    print(f"  [v2-hierarchical] 分层图谱: {hg.num_levels} 层, "
          f"{len(hg.abstract_nodes)} 个抽象概念, {len(hg.abstract_edges)} 条抽象边")

    result = build_hierarchical_subgraph(
        valid_entities, graph, hg,
        max_hops=max_hops,
        expand_to_g0=True,
        sql_edges=sql_edges,
    )

    # 展开邻居（可选）
    if expand_neighbors > 0:
        collected_nodes = {n.label for n in result["nodes"]}
        extra_nodes: Set[str] = set()
        for nl in list(collected_nodes):
            for _ in range(expand_neighbors):
                neighbors = set()
                for edge in graph.adjacency.get(nl, []):
                    if not include_sql_edges and edge.sql_edge:
                        continue
                    if edge.to_label not in collected_nodes:
                        neighbors.add(edge.to_label)
                for edge in graph.in_adjacency.get(nl, []):
                    if not include_sql_edges and edge.sql_edge:
                        continue
                    if edge.from_label not in collected_nodes:
                        neighbors.add(edge.from_label)
                extra_nodes.update(neighbors)
        collected_nodes.update(extra_nodes)
        result["nodes"] = [node_map[n] for n in collected_nodes if n in node_map]

    return result


# ══════════════════════════════════════════════════════════════
# 兼容模式：传统 BFS 最短路径
# ══════════════════════════════════════════════════════════════

_NOISY_INTERMEDIATE_TYPES = {"Function", "Event", "MetricCategory", "Concept/Filter"}

_MEASUREMENT_EDGE_TYPES = {"measured_by", "measured_as"}


def _is_measurement_bridge(edges: list, middle_idx: int) -> bool:
    """检查 Event 中间节点是否是度量桥接。

    Event → Metric 或 Metric → Event 之间存在 measured_by/measured_as 边，
    说明该 Event 是度量的语义锚点（如 商机创建→商机金额），不应视为噪音。
    """
    out_edge = middle_idx + 1
    if out_edge < len(edges) and edges[out_edge].get("label") in _MEASUREMENT_EDGE_TYPES:
        return True
    if middle_idx < len(edges) and edges[middle_idx].get("label") in _MEASUREMENT_EDGE_TYPES:
        return True
    return False


def _classify_entities(
    entities: List[str],
    node_map: Dict[str, Node],
) -> Tuple[List[str], List[str], List[str]]:
    _METRIC_TYPES = {"Metric", "MetricCategory"}
    _FUNCTION_TYPES = {"Function"}
    _TARGET_TYPES = _METRIC_TYPES | _FUNCTION_TYPES
    _CONSTRAINT_TYPES = {"Entity", "Attribute", "Filter", "Concept/Filter", "Event"}

    metrics = []
    constraints = []
    others = []
    for e in entities:
        node = node_map.get(e)
        if node is None:
            others.append(e)
        elif node.type in _TARGET_TYPES:
            metrics.append(e)
        elif node.type in _CONSTRAINT_TYPES:
            constraints.append(e)
        else:
            others.append(e)
    return metrics, constraints, others


def _score_path(path: dict, entity_set: set = None) -> Tuple[int, int]:
    """评分路径：(跳数, 噪音数)。
    
    泛化规则：中间节点如果在 entity_set 中，不算噪音（因为用户明确要了它）。
    Concept/Filter 节点若不在 entity_set 中且为中间节点则算噪音，防止无关约束被 BFS 反拉。
    """
    nodes = path.get("nodes", [])
    edges = path.get("edges", [])
    hops = len(nodes) - 1
    entity_set = entity_set or set()
    noise = 0
    for i, n in enumerate(nodes[1:-1]):  # 只检查中间节点
        if n in entity_set:
            continue
        if n in _NOISY_INTERMEDIATE_TYPES:
            if _is_measurement_bridge(edges, i):
                continue
            noise += 1
    return (hops, noise)


def _filter_relevant_paths(paths: List[dict], entity_set: set = None) -> List[dict]:
    if not paths:
        return []
    entity_set = entity_set or set()
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for p in paths:
        between = p.get("between", [])
        if len(between) == 2:
            key = tuple(sorted(between))
            groups.setdefault(key, []).append(p)
        else:
            groups.setdefault(("?", "?"), []).append(p)

    filtered = []
    for key, group in groups.items():
        group.sort(key=lambda p: _score_path(p, entity_set))
        best_score = _score_path(group[0], entity_set)
        has_clean_path = any(_score_path(p, entity_set)[1] == 0 for p in group)
        for p in group:
            s = _score_path(p, entity_set)
            if s[0] > best_score[0] and s[1] > best_score[1]:
                continue
            if has_clean_path and s[1] > 0:
                continue
            filtered.append(p)
    return filtered


def _bfs_shortest_paths(
    index: GraphIndex,
    start: str,
    end: str,
    max_hops: int = 5,
    max_paths: int = 3,
    include_sql_edges: bool = False,
) -> List[dict]:
    graph = index.graph
    if start not in graph.node_map or end not in graph.node_map:
        return []
    if start == end:
        return [{"nodes": [start], "edges": []}]

    paths = []
    pq = []
    seq = 0
    heapq.heappush(pq, (0, seq, start, [start], []))
    seq += 1
    visited_paths = set()
    best_weight = float("inf")

    while pq and len(paths) < max_paths:
        weight, _, current, node_path, edge_path = heapq.heappop(pq)
        if len(node_path) > max_hops + 1:
            continue
        if weight > best_weight:
            continue
        if current == end:
            paths.append({
                "nodes": list(node_path),
                "edges": [
                    {"from": e.from_label, "to": e.to_label,
                     "label": e.label, "display_label": e.display_label}
                    for e in edge_path
                ],
            })
            best_weight = min(best_weight, weight)
            continue

        next_steps = []
        for edge in graph.adjacency.get(current, []):
            if not include_sql_edges and edge.sql_edge:
                continue
            next_steps.append((edge.to_label, edge))
        for edge in graph.in_adjacency.get(current, []):
            if not include_sql_edges and edge.sql_edge:
                continue
            next_steps.append((edge.from_label, edge))

        for next_node, edge in next_steps:
            if next_node in node_path:
                continue
            new_path = tuple(node_path + [next_node])
            if new_path in visited_paths:
                continue
            visited_paths.add(new_path)
            next_node_obj = graph.node_map.get(next_node)
            penalty = 4 if (next_node_obj and next_node_obj.type == "MetricCategory") else 0
            new_weight = weight + 1 + penalty
            heapq.heappush(pq, (new_weight, seq, next_node,
                                list(node_path) + [next_node],
                                edge_path + [edge]))
            seq += 1

    return paths


def _build_subgraph_bfs(
    valid_entities: List[str],
    invalid_entities: List[str],
    index: GraphIndex,
    *,
    max_hops: int = 5,
    max_paths_per_pair: int = 2,
    expand_neighbors: int = 0,
    include_sql_edges: bool = False,
) -> dict:
    """传统 BFS 最短路径子图构建（兼容 use_diffusion=False）。"""
    graph = index.graph
    node_map = graph.node_map

    metric_nodes, constraint_nodes, _ = _classify_entities(valid_entities, node_map)

    collected_nodes: Set[str] = set()
    collected_edges: Dict[Tuple[str, str], Edge] = {}
    all_paths: List[dict] = []

    def _collect_path(path: dict):
        nonlocal collected_nodes, collected_edges
        all_paths.append(path)
        for nl in path["nodes"]:
            collected_nodes.add(nl)
        for ei in path["edges"]:
            key = (ei["from"], ei["to"])
            if key not in collected_edges:
                for e in graph.edges:
                    if (e.from_label == ei["from"] and
                        e.to_label == ei["to"] and
                        e.label == ei["label"]):
                        collected_edges[key] = e
                        break

    # Phase A: 指标间路径
    if len(metric_nodes) >= 2:
        for i in range(len(metric_nodes)):
            for j in range(i + 1, len(metric_nodes)):
                a, b = metric_nodes[i], metric_nodes[j]
                paths = _bfs_shortest_paths(
                    index, a, b, max_hops=max_hops,
                    max_paths=max_paths_per_pair,
                    include_sql_edges=include_sql_edges,
                )
                for p in paths:
                    _collect_path({"between": [a, b], "nodes": p["nodes"], "edges": p["edges"]})

    # Phase B: 约束→指标
    for c in constraint_nodes:
        for m in metric_nodes:
            paths = _bfs_shortest_paths(
                index, c, m, max_hops=max_hops,
                max_paths=max_paths_per_pair,
                include_sql_edges=include_sql_edges,
            )
            for p in paths:
                _collect_path({"between": [c, m], "nodes": p["nodes"], "edges": p["edges"]})

    # Phase C: 无指标退化
    if not metric_nodes and len(constraint_nodes) >= 2:
        for i in range(len(constraint_nodes)):
            for j in range(i + 1, len(constraint_nodes)):
                a, b = constraint_nodes[i], constraint_nodes[j]
                paths = _bfs_shortest_paths(
                    index, a, b, max_hops=max_hops,
                    max_paths=max_paths_per_pair,
                    include_sql_edges=include_sql_edges,
                )
                for p in paths:
                    _collect_path({"between": [a, b], "nodes": p["nodes"], "edges": p["edges"]})

    all_paths = _filter_relevant_paths(all_paths, set(valid_entities))

    if expand_neighbors > 0:
        extra_nodes: Set[str] = set()
        for nl in list(collected_nodes):
            for _ in range(expand_neighbors):
                neighbors = set()
                for edge in graph.adjacency.get(nl, []):
                    if not include_sql_edges and edge.sql_edge:
                        continue
                    if edge.to_label not in collected_nodes:
                        neighbors.add(edge.to_label)
                for edge in graph.in_adjacency.get(nl, []):
                    if not include_sql_edges and edge.sql_edge:
                        continue
                    if edge.from_label not in collected_nodes:
                        neighbors.add(edge.from_label)
                extra_nodes.update(neighbors)
        collected_nodes.update(extra_nodes)

    nodes_in_paths: Set[str] = set()
    for p in all_paths:
        nodes_in_paths.update(p.get("nodes", []))
    isolated = [e for e in valid_entities if e not in nodes_in_paths]

    return {
        "nodes": [node_map[n] for n in collected_nodes if n in node_map],
        "edges": list(collected_edges.values()),
        "paths": all_paths,
        "isolated": isolated,
    }
