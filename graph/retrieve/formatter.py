"""M4: 输出格式化模块。

将检索结果格式化为 JSON 或 Markdown，方便人类阅读和 Agent 消费。
"""

import json
from typing import Any, Dict, List

from loader import Edge, Node


def format_nodes_json(nodes: List[Node]) -> str:
    """将节点列表格式化为 JSON 字符串。"""
    return json.dumps([_node_to_dict(n) for n in nodes], ensure_ascii=False, indent=2)


def format_edges_json(edges: List[Edge]) -> str:
    """将边列表格式化为 JSON 字符串。"""
    return json.dumps([_edge_to_dict(e) for e in edges], ensure_ascii=False, indent=2)


def format_node_detail_md(node: Node) -> str:
    """将单个节点格式化为 Markdown 详情。"""
    lines = [f"## {node.label}", f"", f"- **类型**: {node.type}", f"- **颜色**: {node.color}"]
    if node.description:
        lines.append(f"- **说明**: {node.description}")
    if node.cube:
        lines.append(f"- **Cube**: `{node.cube}`")
    if node.dataset:
        lines.append(f"- **数据集**: `{node.dataset}`")
    if node.dimension:
        lines.append(f"- **维度**: `{node.dimension}`")
    if node.column:
        lines.append(f"- **字段列**: `{node.column}`")
    if node.data_type:
        lines.append(f"- **数据类型**: {node.data_type}")
    if node.unit:
        lines.append(f"- **单位**: {node.unit}")
    if node.synonyms:
        lines.append(f"- **同义词**: {', '.join(node.synonyms)}")
    if node.metric:
        lines.append(f"- **指标**: `{node.metric}`")
    if node.metric_type:
        lines.append(f"- **指标类型**: {node.metric_type}")
    if node.measure:
        lines.append(f"- **度量**: `{node.measure}`")
    if node.agg:
        lines.append(f"- **聚合**: {node.agg}")
    if node.expr:
        lines.append(f"- **表达式**: `{node.expr}`")
    if node.filter:
        lines.append(f"- **过滤器**: `{node.filter}`")
    if node.filter_type:
        lines.append(f"- **过滤类型**: {node.filter_type}")
    if node.depends_on:
        lines.append(f"- **依赖项**: {', '.join(node.depends_on)}")
    if node.category:
        lines.append(f"- **分类**: {node.category}")
    if node.warning:
        lines.append(f"- **⚠️ 注意**: {node.warning}")
    return "\n".join(lines)


def format_paths_md(paths: List[dict]) -> str:
    """将路径列表格式化为 Markdown。"""
    if not paths:
        return "未找到路径。"
    lines = [f"共找到 {len(paths)} 条路径：", ""]
    for i, path in enumerate(paths, 1):
        node_chain = " → ".join(path["nodes"])
        lines.append(f"### 路径 {i}（{path['hops']} 跳）")
        lines.append(f"```")
        lines.append(node_chain)
        if path["edges"]:
            lines.append("")
            lines.append("边详情：")
            for e in path["edges"]:
                display = e.get("display_label") or e["label"]
                lines.append(f"  {e['from']} --[{display}]--> {e['to']}")
        lines.append(f"```")
        lines.append("")
    return "\n".join(lines)


def format_neighbors_md(result: dict) -> str:
    """将邻居检索结果格式化为 Markdown。"""
    center = result["center"]
    if center is None:
        return "中心节点不存在。"

    lines = [
        f"## {center.label} 的 {result['hops']} 跳邻居子图",
        "",
        f"- 节点数: {len(result['nodes'])}",
        f"- 边数: {len(result['edges'])}",
        "",
        "### 邻居节点",
    ]
    for node in result["nodes"]:
        prefix = "★" if node.label == center.label else "  "
        lines.append(f"{prefix} **{node.label}** ({node.type})")

    lines.extend(["", "### 边关系"])
    for edge in result["edges"]:
        display = edge.display_label or edge.label
        sql_tag = f" [{edge.sql_clause}]" if edge.sql_edge and edge.sql_clause else ""
        lines.append(f"- {edge.from_label} --**{display}**{sql_tag}--> {edge.to_label}")

    return "\n".join(lines)


def format_stats_md(stats: Dict[str, Any]) -> str:
    """将统计信息格式化为 Markdown。"""
    lines = [
        f"## 图谱统计 — {stats.get('domain', '')}",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 总节点数 | {stats.get('total_nodes', 0)} |",
        f"| 总边数 | {stats.get('total_edges', 0)} |",
        f"| 语义边 | {stats.get('semantic_edges', 0)} |",
        f"| SQL逻辑边 | {stats.get('sql_edges', 0)} |",
    ]

    if stats.get("nodes_by_type"):
        lines.extend(["", "### 节点类型分布", ""])
        for t, cnt in sorted(stats["nodes_by_type"].items(), key=lambda x: -x[1]):
            lines.append(f"- {t}: {cnt}")

    if stats.get("edges_by_label"):
        lines.extend(["", "### 边类型分布", ""])
        for lbl, cnt in sorted(stats["edges_by_label"].items(), key=lambda x: -x[1]):
            lines.append(f"- {lbl}: {cnt}")

    if stats.get("nodes_with_warning"):
        lines.extend(["", "### ⚠️ 存在 warning 的节点", ""])
        for name in stats["nodes_with_warning"]:
            lines.append(f"- {name}")

    if stats.get("nodes_missing_metric"):
        lines.extend(["", "### 缺少 metric 定义的节点", ""])
        for name in stats["nodes_missing_metric"]:
            lines.append(f"- {name}")

    if stats.get("nodes_missing_filter"):
        lines.extend(["", "### 缺少 filter 定义的节点", ""])
        for name in stats["nodes_missing_filter"]:
            lines.append(f"- {name}")

    return "\n".join(lines)


# ---- 内部辅助 ----

def _node_to_dict(node: Node) -> dict:
    """Node → 精简 dict。"""
    d = {
        "label": node.label,
        "type": node.type,
        "color": node.color,
    }
    if node.description:
        d["description"] = node.description
    if node.cube:
        d["cube"] = node.cube
    if node.dimension:
        d["dimension"] = node.dimension
    if node.column:
        d["column"] = node.column
    if node.synonyms:
        d["synonyms"] = node.synonyms
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
    if node.warning:
        d["warning"] = node.warning
    if node.depends_on:
        d["depends_on"] = node.depends_on
    if node.data_type:
        d["data_type"] = node.data_type
    if node.unit:
        d["unit"] = node.unit
    return d


def _edge_to_dict(edge: Edge) -> dict:
    """Edge → 精简 dict。"""
    d = {
        "from": edge.from_label,
        "to": edge.to_label,
        "label": edge.label,
    }
    if edge.display_label:
        d["display_label"] = edge.display_label
    if edge.sql_edge:
        d["sql_edge"] = True
        if edge.sql_clause:
            d["sql_clause"] = edge.sql_clause
        if edge.condition_type:
            d["condition_type"] = edge.condition_type
    if edge.join_key:
        d["join_key"] = edge.join_key
    if edge.custom:
        d["custom"] = True
    return d
