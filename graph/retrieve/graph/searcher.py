"""M3: 检索核心模块。

基于 GraphIndex 实现 4 种检索能力：
1. search_nodes  — 节点检索（精确/模糊/多字段）
2. search_edges  — 边检索（按源/目标/类型/条件过滤）
3. search_paths  — 路径检索（BFS 多跳路径）
4. get_neighbors — 邻居检索（N 跳子图）
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from graph.index_builder import GraphIndex, _tokenize
from graph.loader import Edge, Node


# ============================================================
# 1. 节点检索
# ============================================================

def search_nodes(
    index: GraphIndex,
    query: str = "",
    *,
    node_type: Optional[str] = None,
    cube: Optional[str] = None,
    has_warning: Optional[bool] = None,
    has_synonyms: Optional[bool] = None,
    exact_label: Optional[str] = None,
    top_k: int = 20,
) -> List[Node]:
    """检索节点。

    Args:
        index: 图谱索引
        query: 关键词/自然语言查询
        node_type: 按节点类型过滤
        cube: 按所属 cube 过滤
        has_warning: 是否只返回有 warning 的节点
        has_synonyms: 是否只返回有同义词的节点
        exact_label: 精确匹配 label
        top_k: 返回最多节点数

    Returns:
        按相关度排序的节点列表
    """
    # 精确 label 匹配（最高优先级）
    if exact_label:
        node = index.graph.node_map.get(exact_label)
        return [node] if node else []

    candidates: Dict[str, float] = {}  # label → score

    # 倒排索引检索
    if query:
        query_tokens = _tokenize(query)
        for token in query_tokens:
            if token in index.inverted_index:
                for label in index.inverted_index[token]:
                    candidates[label] = candidates.get(label, 0) + 1

    # 无 query 时返回全部
    if not query:
        candidates = {n.label: 0.0 for n in index.graph.nodes}

    # 过滤
    result = []
    for label, score in candidates.items():
        node = index.graph.node_map.get(label)
        if node is None:
            continue

        # 类型过滤
        if node_type and node.type != node_type:
            continue

        # cube 过滤
        if cube and node.cube != cube:
            continue

        # warning 过滤
        if has_warning is True and not node.warning:
            continue
        if has_warning is False and node.warning:
            continue

        # synonyms 过滤
        if has_synonyms is True and not node.synonyms:
            continue
        if has_synonyms is False and node.synonyms:
            continue

        # 额外加分：同义词匹配
        if query:
            for syn in node.synonyms:
                for token in query_tokens:
                    if token in syn:
                        score += 1.5
                        break

            # 类型名称匹配加分
            if any(token in node.type for token in query_tokens):
                score += 0.5

        result.append((node, score))

    # 按分数降序排列
    result.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in result[:top_k]]


# ============================================================
# 2. 边检索
# ============================================================

def search_edges(
    index: GraphIndex,
    *,
    from_label: Optional[str] = None,
    to_label: Optional[str] = None,
    edge_label: Optional[str] = None,
    include_sql_edges: bool = False,
    sql_clause: Optional[str] = None,
    condition_type: Optional[str] = None,
) -> List[Edge]:
    """检索边。

    Args:
        index: 图谱索引
        from_label: 按源节点过滤
        to_label: 按目标节点过滤
        edge_label: 按边关系类型过滤 (relates_to / has_attribute / measured_by / ...)
        include_sql_edges: 是否包含 SQL 逻辑边
        sql_clause: SQL 子句类型 (WHERE / GROUP BY / HAVING)
        condition_type: 条件类型 (dimension_filter / concept_filter / time_filter / ...)

    Returns:
        匹配的边列表
    """
    results = []

    for edge in index.graph.edges:
        if not include_sql_edges and edge.sql_edge:
            continue
        if from_label and edge.from_label != from_label:
            continue
        if to_label and edge.to_label != to_label:
            continue
        if edge_label and edge.label != edge_label:
            continue
        if sql_clause and edge.sql_clause != sql_clause:
            continue
        if condition_type and edge.condition_type != condition_type:
            continue
        results.append(edge)

    return results


# ============================================================
# 3. 路径检索
# ============================================================

def search_paths(
    index: GraphIndex,
    start_label: str,
    end_label: str,
    *,
    max_hops: int = 4,
    include_sql_edges: bool = False,
    max_paths: int = 10,
) -> List[dict]:
    """BFS 查找两个节点之间的所有路径。

    Args:
        index: 图谱索引
        start_label: 起始节点 label
        end_label: 目标节点 label
        max_hops: 最大跳数
        include_sql_edges: 是否包含 SQL 逻辑边
        max_paths: 最多返回路径数

    Returns:
        路径列表，每条路径格式:
        {
            "nodes": ["A", "B", "C"],
            "edges": [{"from": "A", "to": "B", "label": "..."}, ...],
            "hops": 2
        }
    """
    if start_label not in index.graph.node_map or end_label not in index.graph.node_map:
        return []

    if start_label == end_label:
        return [{"nodes": [start_label], "edges": [], "hops": 0}]

    paths = []
    # BFS queue: (current_node, [node_path], [edge_path])
    queue = deque()
    queue.append((start_label, [start_label], []))
    visited_paths = set()  # 用于去重：记录经过的节点序列

    while queue and len(paths) < max_paths:
        current, node_path, edge_path = queue.popleft()

        if len(node_path) > max_hops + 1:
            continue

        if current == end_label:
            paths.append({
                "nodes": node_path,
                "edges": [{"from": edge_path[i].from_label, "to": edge_path[i].to_label,
                           "label": edge_path[i].label, "display_label": edge_path[i].display_label}
                          for i in range(len(edge_path))],
                "hops": len(edge_path),
            })
            continue

        # 获取当前节点的所有出边
        out_edges = index.graph.adjacency.get(current, [])
        for edge in out_edges:
            if not include_sql_edges and edge.sql_edge:
                continue
            next_node = edge.to_label
            if next_node in node_path:
                continue  # 避免环
            new_path = tuple(node_path + [next_node])
            if new_path in visited_paths:
                continue
            visited_paths.add(new_path)
            queue.append((next_node, node_path + [next_node], edge_path + [edge]))

    return paths


# ============================================================
# 4. 邻居检索
# ============================================================

def get_neighbors(
    index: GraphIndex,
    center_label: str,
    *,
    hops: int = 1,
    include_sql_edges: bool = False,
    direction: str = "both",
) -> dict:
    """获取以指定节点为中心的 N 跳邻居子图。

    Args:
        index: 图谱索引
        center_label: 中心节点 label
        hops: 跳数
        include_sql_edges: 是否包含 SQL 逻辑边
        direction: "out"(出边) / "in"(入边) / "both"(双向)

    Returns:
        {
            "center": Node,
            "nodes": [Node, ...],      # 含中心节点
            "edges": [Edge, ...],
            "hops": int
        }
    """
    center = index.graph.node_map.get(center_label)
    if center is None:
        return {"center": None, "nodes": [], "edges": [], "hops": hops}

    visited_nodes: Set[str] = {center_label}
    collected_edges: List[Edge] = []
    current_frontier: Set[str] = {center_label}

    for _ in range(hops):
        next_frontier: Set[str] = set()
        for node_label in current_frontier:
            # 出边
            if direction in ("out", "both"):
                for edge in index.graph.adjacency.get(node_label, []):
                    if not include_sql_edges and edge.sql_edge:
                        continue
                    if edge.to_label not in visited_nodes:
                        visited_nodes.add(edge.to_label)
                        next_frontier.add(edge.to_label)
                    collected_edges.append(edge)

            # 入边
            if direction in ("in", "both"):
                for edge in index.graph.in_adjacency.get(node_label, []):
                    if not include_sql_edges and edge.sql_edge:
                        continue
                    if edge.from_label not in visited_nodes:
                        visited_nodes.add(edge.from_label)
                        next_frontier.add(edge.from_label)
                    collected_edges.append(edge)

        current_frontier = next_frontier

    # 收集节点
    nodes = [index.graph.node_map[lbl] for lbl in visited_nodes if lbl in index.graph.node_map]

    return {
        "center": center,
        "nodes": nodes,
        "edges": collected_edges,
        "hops": hops,
    }
