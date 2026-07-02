"""M2: 子图构建模块（v2.1 核心路径优先）。

策略变更：不再做「所有实体两两 BFS 路径并集 + 邻居展开」，
改为「核心路径优先」——

1. 区分三类节点：
   - 指标节点 (Metric / Function)：问题的查询目标
   - 约束节点 (Entity / Filter / Attribute / Event)：问题的查询条件
   - 中间节点：路径上经过的其他节点

2. 子图 = 指标间的推导路径 ∪ 指标到 Function 的计算路径
           ∪ 约束到指标的约束路径（按需纳入）

3. 不纳入与指标无关的约束节点间路径，不盲目展开邻居。
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from graph.index_builder import GraphIndex
from graph.loader import Edge, Node


# ── 节点分类 ────────────────────────────────────────────────

# 指标类节点类型：它们是查询"想算什么"的目标
_METRIC_TYPES = {"Metric", "MetricCategory"}
_FUNCTION_TYPES = {"Function"}
_TARGET_TYPES = _METRIC_TYPES | _FUNCTION_TYPES

# 约束类节点类型：它们是查询"按什么筛选/分组/限制"的条件
_CONSTRAINT_TYPES = {"Entity", "Attribute", "Filter", "Concept/Filter", "Event"}

# 高频指标分类枢纽：这些节点连接了大量指标，容易在路径搜索中形成语义捷径。
# 不禁用，但当它们作为中间节点出现时显著降权。
_LOW_PRIORITY_BRIDGE_LABELS = {"金额指标", "数量指标"}
_LOW_PRIORITY_BRIDGE_PENALTY = 8


def _is_metric_node(label: str, node_map: Dict[str, Node]) -> bool:
    node = node_map.get(label)
    return bool(node and node.type == "Metric")


def _is_low_priority_metric_shortcut(
    prev_node: str,
    bridge_node: str,
    next_node: str,
    node_map: Dict[str, Node],
) -> bool:
    return (
        bridge_node in _LOW_PRIORITY_BRIDGE_LABELS
        and _is_metric_node(prev_node, node_map)
        and _is_metric_node(next_node, node_map)
    )


def _path_has_low_priority_metric_shortcut(nodes: List[str], node_map: Dict[str, Node]) -> bool:
    return any(
        _is_low_priority_metric_shortcut(nodes[i - 1], nodes[i], nodes[i + 1], node_map)
        for i in range(1, len(nodes) - 1)
    )


def _classify_entities(
    entities: List[str],
    node_map: Dict[str, Node],
) -> Tuple[List[str], List[str], List[str]]:
    """将实体分类为 (指标, 约束, 其他)。"""
    metrics: List[str] = []
    constraints: List[str] = []
    others: List[str] = []

    for e in entities:
        node = node_map.get(e)
        if node is None:
            others.append(e)
            continue
        if node.type in _TARGET_TYPES:
            metrics.append(e)
        elif node.type in _CONSTRAINT_TYPES:
            constraints.append(e)
        else:
            others.append(e)

    return metrics, constraints, others


# ── 子图构建 ────────────────────────────────────────────────

def build_subgraph(
    entities: List[str],
    index: GraphIndex,
    *,
    max_hops: int = 5,
    max_paths_per_pair: int = 2,
    expand_neighbors: int = 0,   # 默认不展开邻居
    include_sql_edges: bool = False,
) -> dict:
    """核心路径优先：构建以指标为中心的精简子图。

    算法：
    1. 将 entities 分为指标节点和约束节点
    2. 指标间：两两找最短推导路径（derived_from / classified_as / supports_function）
    3. 约束→指标：每个约束节点找最短路径到达任意指标节点
    4. 没有指标时退化为：约束→约束 路径
    5. 所有路径取并集形成子图

    Args:
        entities: 实体 label 列表
        index: 图谱索引
        max_hops: 最大跳数
        max_paths_per_pair: 每对节点最多保留几条路径
        expand_neighbors: 额外展开邻居跳数（默认 0）
        include_sql_edges: 是否包含 SQL 逻辑边

    Returns:
        {
            "nodes": [Node, ...],
            "edges": [Edge, ...],
            "paths": [{"between": [a,b], "nodes": [...], "edges": [...]}, ...],
            "isolated": [label, ...]
        }
    """
    graph = index.graph
    node_map = graph.node_map

    # 过滤无效实体
    valid_entities = [e for e in entities if e in node_map]
    invalid_entities = [e for e in entities if e not in node_map]

    if not valid_entities:
        return {"nodes": [], "edges": [], "paths": [], "isolated": invalid_entities}

    # 分类
    metric_nodes, constraint_nodes, _ = _classify_entities(valid_entities, node_map)

    collected_nodes: Set[str] = set()
    collected_edges: Dict[Tuple[str, str], Edge] = {}
    all_paths: List[dict] = []

    # 辅助：收集路径中的节点和边
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

    # ── Phase A: 指标节点间路径 ──
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

    # ── Phase B: 约束节点 → 指标节点路径 ──
    if metric_nodes and constraint_nodes:
        for c_node in constraint_nodes:
            for m_node in metric_nodes:
                # 约束节点出发 → 指标节点（约束是"条件"，指标是"目标"）
                # 方向可能是 c→m 或 m→c（取决于边方向，BFS 已双向）
                paths = _bfs_shortest_paths(
                    index, c_node, m_node, max_hops=max_hops,
                    max_paths=max_paths_per_pair,
                    include_sql_edges=include_sql_edges,
                )
                for p in paths:
                    _collect_path({"between": [c_node, m_node], "nodes": p["nodes"], "edges": p["edges"]})

    # ── Phase C: 退化：没有指标时，约束节点间路径 ──
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

    # ── Phase D: 单一指标/约束 → 展开上游路径 ──
    # 如果只有一个指标或约束，展开它与其他节点的关联
    if not all_paths and len(valid_entities) == 1:
        solo = valid_entities[0]
        # 找所有 1-2 跳可达节点
        paths = _bfs_all_reachable(index, solo, max_hops=min(max_hops, 3), include_sql_edges=include_sql_edges)
        for p in paths:
            end_node = p["nodes"][-1] if p["nodes"] else solo
            _collect_path({"between": [solo, end_node], "nodes": p["nodes"], "edges": p["edges"]})

    # ── 路径相关性过滤：去掉经过无关 Function/Event 的绕路路径 ──
    all_paths = _filter_relevant_paths(all_paths, valid_entities, node_map)

    # 根据过滤后的路径重建 collected_nodes / collected_edges
    collected_nodes.clear()
    collected_edges.clear()
    for p in all_paths:
        for nl in p["nodes"]:
            collected_nodes.add(nl)
        for ei in p["edges"]:
            key = (ei["from"], ei["to"])
            if key not in collected_edges:
                for e in graph.edges:
                    if (e.from_label == ei["from"] and
                        e.to_label == ei["to"] and
                        e.label == ei["label"]):
                        collected_edges[key] = e
                        break

    # ── Phase E: 额外邻居展开（默认关闭） ──
    if expand_neighbors > 0:
        for _ in range(expand_neighbors):
            current_labels = list(collected_nodes)
            for label in current_labels:
                for edge in graph.adjacency.get(label, []):
                    if not include_sql_edges and edge.sql_edge:
                        continue
                    key = (edge.from_label, edge.to_label)
                    if key not in collected_edges:
                        collected_edges[key] = edge
                    collected_nodes.add(edge.to_label)
                for edge in graph.in_adjacency.get(label, []):
                    if not include_sql_edges and edge.sql_edge:
                        continue
                    key = (edge.from_label, edge.to_label)
                    if key not in collected_edges:
                        collected_edges[key] = edge
                    collected_nodes.add(edge.from_label)

    # ── 收集节点对象 ──
    nodes = [node_map[lbl] for lbl in collected_nodes if lbl in node_map]
    edges = list(collected_edges.values())

    # 孤立节点
    connected = set()
    for p in all_paths:
        for nl in p["nodes"]:
            connected.add(nl)
    isolated = [e for e in valid_entities if e not in connected]

    return {
        "nodes": nodes,
        "edges": edges,
        "paths": all_paths,
        "isolated": isolated,
    }


# ── BFS 路径查找 ────────────────────────────────────────────

def _bfs_shortest_paths(
    index: GraphIndex,
    start: str,
    end: str,
    max_hops: int = 5,
    max_paths: int = 3,
    include_sql_edges: bool = False,
) -> List[dict]:
    """双向 BFS 查找两个节点间的最短路径（多条），优先避开 MetricCategory 桥接节点。

    使用加权优先队列：经过 MetricCategory 的路径被降权（等效跳数+2），
    确保不经过 MetricCategory 的直接路径优先被探索和返回。

    Returns:
        [{"nodes": [...], "edges": [{"from","to","label","display_label"}, ...]}, ...]
    """
    import heapq

    graph = index.graph

    if start not in graph.node_map or end not in graph.node_map:
        return []
    if start == end:
        return [{"nodes": [start], "edges": []}]

    paths = []
    # 优先队列: (weight, seq, current, node_path, edge_path)
    # weight = 实际跳数 + MetricCategory 惩罚分
    # seq 用于 tie-break，确保 FIFO 稳定性
    pq = []
    seq = 0
    heapq.heappush(pq, (0, seq, start, [start], []))
    seq += 1
    visited_paths = set()
    # 记录已找到路径的最小权重，不再探索更重的路径
    best_weight = float('inf')

    while pq and len(paths) < max_paths:
        weight, _, current, node_path, edge_path = heapq.heappop(pq)

        if len(node_path) > max_hops + 1:
            continue

        # 剪枝：当前权重已经超过已找到的最优权重，跳过
        if weight > best_weight:
            continue

        if current == end:
            paths.append({
                "nodes": list(node_path),
                "edges": [
                    {
                        "from": e.from_label,
                        "to": e.to_label,
                        "label": e.label,
                        "display_label": e.display_label,
                    }
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
            if (
                len(node_path) >= 2
                and _is_low_priority_metric_shortcut(
                    node_path[-2], current, next_node, graph.node_map
                )
            ):
                continue
            new_path = tuple(node_path + [next_node])
            if new_path in visited_paths:
                continue
            visited_paths.add(new_path)

            # 计算惩罚：MetricCategory 是跨域桥接，权重+4 确保不产生语义捷径。
            # 金额指标/数量指标是高频分类枢纽，作为中间节点时更容易制造捷径，额外降权。
            next_node_obj = graph.node_map.get(next_node)
            penalty = 0
            if next_node_obj and next_node_obj.type == "MetricCategory":
                penalty += 4
            if next_node in _LOW_PRIORITY_BRIDGE_LABELS and next_node not in {start, end}:
                penalty += _LOW_PRIORITY_BRIDGE_PENALTY

            new_weight = weight + 1 + penalty
            heapq.heappush(pq, (new_weight, seq, next_node, list(node_path) + [next_node], edge_path + [edge]))
            seq += 1

    return paths


# ── 路径相关性过滤 ──────────────────────────────────────────

# 这些节点类型如果在路径中间出现，且不在用户问题中，说明是绕路路径
_NOISY_INTERMEDIATE_TYPES = {"Function", "Event", "MetricCategory", "Concept/Filter"}

# 度量桥接边类型：Event 通过这些边连接 Metric 时，是必要的语义锚点而非噪音
_MEASUREMENT_EDGE_TYPES = {"measured_by", "measured_as"}


def _is_measurement_bridge(edges: list, middle_idx: int) -> bool:
    """检查 Event 中间节点是否是度量桥接。

    Event → Metric 或 Metric → Event 之间存在 measured_by/measured_as 边，
    说明该 Event 是度量的语义锚点（如 商机创建→商机金额），不应视为噪音。

    Args:
        edges: 路径的边列表
        middle_idx: 中间节点在 middle 数组中的索引
    """
    # 边 edges[middle_idx+1] 连接 middle[middle_idx] 到 middle[middle_idx+1]
    out_edge = middle_idx + 1
    if out_edge < len(edges) and edges[out_edge].get("label") in _MEASUREMENT_EDGE_TYPES:
        return True
    # 边 edges[middle_idx] 连接 middle[middle_idx-1] 到 middle[middle_idx]
    if middle_idx < len(edges) and edges[middle_idx].get("label") in _MEASUREMENT_EDGE_TYPES:
        return True
    return False


def _filter_relevant_paths(
    paths: List[dict],
    valid_entities: List[str],
    node_map: Dict[str, Node],
) -> List[dict]:
    """过滤掉冗余/绕路路径。

    规则：
    1. 同端点对（between 相同）去重，保留最短路径
    2. 路径中间包含无关 Function/Event 节点（不在 valid_entities 中）且
       存在更短替代路径的，过滤掉。
    """
    entity_set = set(valid_entities)
    paths = [
        p for p in paths
        if not _path_has_low_priority_metric_shortcut(p.get("nodes", []), node_map)
    ]

    # group by (between tuple)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for path in paths:
        key = tuple(sorted(path.get("between", [])))
        groups[key].append(path)

    def _score(p: dict) -> tuple:
        """评分：越低越好。
        (路径长度, 高频分类枢纽数, 中间无关Function数, 中间无关节点总数)
        """
        nodes = p.get("nodes", [])
        edges = p.get("edges", [])
        middle = nodes[1:-1]
        low_priority_bridges = sum(
            1 for mn in middle
            if mn not in entity_set and mn in _LOW_PRIORITY_BRIDGE_LABELS
        )
        noisy_fn = 0
        for i, mn in enumerate(middle):
            if mn in entity_set:
                continue
            node_obj = node_map.get(mn)
            if not node_obj or node_obj.type not in _NOISY_INTERMEDIATE_TYPES:
                continue
            # 泛化规则：Event 若通过 measured_by/measured_as 连接 Metric，
            #           是必要的度量桥接，不视为噪音
            if node_obj.type == "Event" and _is_measurement_bridge(edges, i):
                continue
            noisy_fn += 1
        noisy_all = sum(1 for mn in middle if mn not in entity_set)
        return (len(nodes), low_priority_bridges, noisy_fn, noisy_all)

    filtered = []
    for key, group in groups.items():
        group.sort(key=_score)
        best_score = _score(group[0])

        # 检查是否存在完全不经过噪音节点（MetricCategory/Function/Event）的干净路径
        has_clean_path = any(_score(p)[2] == 0 for p in group)

        for p in group:
            s = _score(p)
            # 规则1: 如果存在更干净的同端点路径（长度≤当前 且 无关节点更少），跳过当前
            if s[0] > best_score[0] and (s[1], s[2]) > (best_score[1], best_score[2]):
                continue
            # 规则2: 如果存在完全不经过 MetricCategory/Function/Event 的干净路径，
            #         则过滤掉经过了这些噪音节点的路径（等长也过滤）
            if has_clean_path and s[2] > 0:
                continue
            filtered.append(p)

    return filtered


def _bfs_all_reachable(
    index: GraphIndex,
    start: str,
    max_hops: int = 3,
    include_sql_edges: bool = False,
) -> List[dict]:
    """BFS 从起点展开所有可达节点（用于单实体情况）。

    Returns:
        每个可达节点一条路径。
    """
    graph = index.graph
    if start not in graph.node_map:
        return []

    paths = []
    queue = deque()
    queue.append((start, [start], []))
    visited = {start}

    while queue:
        current, node_path, edge_path = queue.popleft()

        if len(node_path) > max_hops + 1:
            continue

        # 记录路径（除了起点自身）
        if len(node_path) > 1:
            paths.append({
                "nodes": list(node_path),
                "edges": [
                    {
                        "from": e.from_label,
                        "to": e.to_label,
                        "label": e.label,
                        "display_label": e.display_label,
                    }
                    for e in edge_path
                ],
            })

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
            if next_node in visited:
                continue
            visited.add(next_node)
            queue.append((next_node, list(node_path) + [next_node], edge_path + [edge]))

    return paths
