"""M1: 知识图谱加载模块。

读取 JSON 图谱文件，构建内存中的 SemanticGraph 数据结构。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    """图谱节点。"""
    label: str
    type: str
    color: str = "#888"
    # 语义层字段（通用）
    cube: Optional[str] = None
    dataset: Optional[str] = None
    dimension: Optional[str] = None
    column: Optional[str] = None
    data_type: Optional[str] = None
    unit: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    subject_domains: List[str] = field(default_factory=list)
    description: Optional[str] = None
    warning: Optional[str] = None
    # Entity 专属
    primary_entity: Optional[str] = None
    primary_column: Optional[str] = None
    foreign_entity: Optional[str] = None
    # Metric 专属
    metric: Optional[str] = None
    metric_type: Optional[str] = None
    measure: Optional[str] = None
    agg: Optional[str] = None
    expr: Optional[str] = None
    based_on_metric: Optional[str] = None
    label_alias: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    related_measure: Optional[str] = None
    # Filter 专属
    filter: Optional[str] = None
    filter_type: Optional[str] = None
    # Function 专属
    category: Optional[str] = None
    # 原始数据
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """图谱边。"""
    from_label: str
    to_label: str
    label: str               # 关系类型: relates_to / has_attribute / measured_by 等
    display_label: Optional[str] = None
    # 语义关系标记
    custom: bool = False
    join_key: Optional[str] = None
    # SQL 逻辑边专属
    sql_edge: bool = False
    sql_clause: Optional[str] = None      # WHERE / GROUP BY / HAVING
    condition_type: Optional[str] = None  # dimension_filter / dimension_group / concept_filter / time_filter / time_group / metric_filter
    cube: Optional[str] = None
    dimension: Optional[str] = None
    column: Optional[str] = None
    requires_join: bool = False
    join_path: List[str] = field(default_factory=list)
    filter: Optional[str] = None
    expr: Optional[str] = None
    based_on_metric: Optional[str] = None
    default_grain: Optional[str] = None
    # 原始数据
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticGraph:
    """语义知识图谱。"""
    domain: str
    description: str
    nodes: List[Node]
    edges: List[Edge]
    # 快速索引（由 index_builder 填充）
    node_map: Dict[str, Node] = field(default_factory=dict)
    adjacency: Dict[str, List[Edge]] = field(default_factory=dict)
    in_adjacency: Dict[str, List[Edge]] = field(default_factory=dict)


def load_graph(json_path: Path, include_sql_edges: bool = False) -> SemanticGraph:
    """从 JSON 文件加载知识图谱。

    Args:
        json_path: JSON 图谱文件路径
        include_sql_edges: 是否包含 SQL 逻辑边（默认不包含）

    Returns:
        SemanticGraph 实例
    """
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    domain = data.get("domain", "")
    description = data.get("description", "")

    nodes = [_parse_node(item) for item in data.get("nodes", [])]
    edges = _parse_edges(data.get("edges", []), include_sql_edges)

    graph = SemanticGraph(domain=domain, description=description, nodes=nodes, edges=edges)

    # 构建快速索引
    graph.node_map = {n.label: n for n in nodes}
    graph.adjacency = _build_adjacency(edges)
    graph.in_adjacency = _build_in_adjacency(edges)

    return graph


def _parse_node(raw: Dict[str, Any]) -> Node:
    """将原始 JSON 节点转为 Node 对象。"""
    return Node(
        label=raw.get("label", ""),
        type=raw.get("type", ""),
        color=raw.get("color", "#888"),
        cube=raw.get("cube"),
        dataset=raw.get("dataset"),
        dimension=raw.get("dimension"),
        column=raw.get("column"),
        data_type=raw.get("data_type"),
        unit=raw.get("unit"),
        synonyms=raw.get("synonyms", []) or [],
        subject_domains=raw.get("subject_domains", []) or [],
        description=raw.get("description"),
        warning=raw.get("warning"),
        primary_entity=raw.get("primary_entity"),
        primary_column=raw.get("primary_column"),
        foreign_entity=raw.get("foreign_entity"),
        metric=raw.get("metric"),
        metric_type=raw.get("metric_type"),
        measure=raw.get("measure"),
        agg=raw.get("agg"),
        expr=raw.get("expr"),
        based_on_metric=raw.get("based_on_metric"),
        label_alias=raw.get("label_alias"),
        depends_on=raw.get("depends_on", []) or [],
        related_measure=raw.get("related_measure"),
        filter=raw.get("filter"),
        filter_type=raw.get("filter_type"),
        category=raw.get("category"),
        raw=raw,
    )


def _parse_edges(raw_edges: List[Dict[str, Any]], include_sql_edges: bool) -> List[Edge]:
    """解析边列表，可选过滤 SQL 逻辑边。"""
    edges = []
    for e in raw_edges:
        sql_edge = e.get("sql_edge", False)
        if sql_edge and not include_sql_edges:
            continue
        edges.append(Edge(
            from_label=e.get("from", ""),
            to_label=e.get("to", ""),
            label=e.get("label", ""),
            display_label=e.get("display_label"),
            custom=e.get("custom", False),
            join_key=e.get("join_key"),
            sql_edge=sql_edge,
            sql_clause=e.get("sql_clause"),
            condition_type=e.get("condition_type"),
            cube=e.get("cube"),
            dimension=e.get("dimension"),
            column=e.get("column"),
            requires_join=e.get("requires_join", False),
            join_path=e.get("join_path", []) or [],
            filter=e.get("filter"),
            expr=e.get("expr"),
            based_on_metric=e.get("based_on_metric"),
            default_grain=e.get("default_grain"),
            raw=e,
        ))
    return edges


def _build_adjacency(edges: List[Edge]) -> Dict[str, List[Edge]]:
    """构建出边邻接表。"""
    adj: Dict[str, List[Edge]] = {}
    for e in edges:
        adj.setdefault(e.from_label, []).append(e)
    return adj


def _build_in_adjacency(edges: List[Edge]) -> Dict[str, List[Edge]]:
    """构建入边邻接表。"""
    adj: Dict[str, List[Edge]] = {}
    for e in edges:
        adj.setdefault(e.to_label, []).append(e)
    return adj
