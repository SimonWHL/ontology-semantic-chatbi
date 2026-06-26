"""M3: 上下文格式化模块。

将实体抽取结果和子图格式化为 JSON，供下游 LLM Agent 消费。
"""

from typing import Any, Dict, List, Optional

from loader import Edge, Node


def format_context(
    question: str,
    entities: List[str],
    subgraph: dict,
    *,
    meta: Optional[Dict[str, Any]] = None,
    capability_notes: str = "",
) -> dict:
    """格式化最终上下文 JSON。

    Args:
        question: 用户原始问题
        entities: 抽取的实体 label 列表
        subgraph: build_subgraph 的返回结果
        meta: 额外元信息
        capability_notes: 维度-指标分析能力描述（由 MetricCapability 生成）

    Returns:
        结构化 JSON（dict）
    """
    result = {
        "question": question,
        "entities": entities,
        "subgraph": {
            "nodes": [_node_to_context(n) for n in subgraph.get("nodes", [])],
            "edges": [_edge_to_context(e) for e in subgraph.get("edges", [])],
        },
        "paths": subgraph.get("paths", []),
        "isolated": subgraph.get("isolated", []),
    }

    # 维度-指标分析能力矩阵（独立模块，不参与路径检索）
    if capability_notes:
        result["capability"] = capability_notes

    # 分层信息（v2 专属）
    hierarchy = subgraph.get("hierarchy")
    if hierarchy:
        result["hierarchy"] = hierarchy

    # 统计 meta
    result["meta"] = {
        "total_nodes": len(subgraph.get("nodes", [])),
        "total_edges": len(subgraph.get("edges", [])),
        "total_paths": len(subgraph.get("paths", [])),
        "isolated_count": len(subgraph.get("isolated", [])),
        **(meta or {}),
    }

    return result


def format_context_md(
    question: str,
    entities: List[str],
    subgraph: dict,
) -> str:
    """格式化为 Markdown（人类可读，调试用）。"""
    lines = [f"## 问题: {question}", ""]

    lines.append(f"### 识别实体")
    lines.append(", ".join(f"`{e}`" for e in entities) if entities else "无")
    lines.append("")

    # 子图概览
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])
    lines.append(f"### 子图概览")
    lines.append(f"- 节点: {len(nodes)}")
    lines.append(f"- 边: {len(edges)}")
    lines.append("")

    # 节点列表
    lines.append("### 节点")
    for node in nodes:
        lines.append(f"- **{node.label}** ({node.type}): {node.description or ''}")
    lines.append("")

    # 边列表
    lines.append("### 边")
    for edge in edges:
        display = edge.display_label or edge.label
        lines.append(f"- {edge.from_label} --[{display}]--> {edge.to_label}")
    lines.append("")

    # 路径
    paths = subgraph.get("paths", [])
    if paths:
        lines.append("### 路径详情")
        for i, path in enumerate(paths, 1):
            between = path.get("between", [])
            chain = " → ".join(path.get("nodes", []))
            lines.append(f"**路径 {i}** ({' ↔ '.join(between)}): `{chain}`")
        lines.append("")

    # 孤立节点
    isolated = subgraph.get("isolated", [])
    if isolated:
        lines.append("### 孤立节点（无路径可达）")
        lines.append(", ".join(f"`{e}`" for e in isolated))
        lines.append("")

    # 分层信息（v2 专属）
    hierarchy = subgraph.get("hierarchy")
    if hierarchy:
        lines.append("### 分层图谱信息")
        lines.append(f"- 层级数: {hierarchy.get('num_levels', 0)}")
        matched_levels = hierarchy.get("matched_levels", {})
        if matched_levels:
            lines.append(f"- 匹配层级: {matched_levels}")
        abstract_nodes = hierarchy.get("abstract_nodes", [])
        if abstract_nodes:
            lines.append("- 抽象概念:")
            for an in abstract_nodes[:10]:
                lines.append(f"  - **{an['label']}** (L{an['level']}): {an.get('description', '')} [{an.get('member_count', 0)}成员]")
        lines.append("")

    return "\n".join(lines)


def _node_to_context(node: Node) -> dict:
    """Node → 供 LLM 消费的精简 dict。"""
    d = {
        "label": node.label,
        "type": node.type,
        "color": node.color,
    }
    if node.description:
        d["description"] = node.description
    if node.cube:
        d["cube"] = node.cube
    if node.dataset:
        d["dataset"] = node.dataset
    if node.dimension:
        d["dimension"] = node.dimension
    if node.column:
        d["column"] = node.column
    if node.data_type:
        d["data_type"] = node.data_type
    if node.unit:
        d["unit"] = node.unit
    if node.metric:
        d["metric"] = node.metric
    if node.metric_type:
        d["metric_type"] = node.metric_type
    if node.agg:
        d["agg"] = node.agg
    if node.expr:
        d["expr"] = node.expr
    if node.filter:
        d["filter"] = node.filter
    if node.filter_type:
        d["filter_type"] = node.filter_type
    if node.depends_on:
        d["depends_on"] = node.depends_on
    if node.category:
        d["category"] = node.category
    if node.warning:
        d["warning"] = node.warning
    if node.primary_entity:
        d["primary_entity"] = node.primary_entity
    if node.foreign_entity:
        d["foreign_entity"] = node.foreign_entity
    return d


def _edge_to_context(edge: Edge) -> dict:
    """Edge → 供 LLM 消费的精简 dict。"""
    d = {
        "from": edge.from_label,
        "to": edge.to_label,
        "label": edge.label,
    }
    if edge.display_label:
        d["display_label"] = edge.display_label
    if edge.join_key:
        d["join_key"] = edge.join_key
    if edge.custom:
        d["custom"] = True
    if edge.sql_edge:
        d["sql_edge"] = True
        if edge.sql_clause:
            d["sql_clause"] = edge.sql_clause
        if edge.condition_type:
            d["condition_type"] = edge.condition_type
        if edge.expr:
            d["expr"] = edge.expr
    return d
