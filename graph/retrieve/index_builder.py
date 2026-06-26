"""M2: 索引构建模块。

基于加载的 SemanticGraph 构建检索索引：
- 倒排索引（关键词 → 节点集合）
- 按类型分组的节点索引
- 边类型分布索引
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

from loader import Edge, Node, SemanticGraph


@dataclass
class GraphIndex:
    """图谱检索索引。"""
    graph: SemanticGraph

    # 倒排索引: 词 → 匹配的节点 label 集合
    inverted_index: Dict[str, Set[str]] = field(default_factory=dict)

    # 按 type 分组的节点
    nodes_by_type: Dict[str, List[Node]] = field(default_factory=dict)

    # 按边 label 分组的边
    edges_by_label: Dict[str, List[Edge]] = field(default_factory=dict)

    # 节点 label → 出边 label 集合
    out_edge_labels: Dict[str, Set[str]] = field(default_factory=dict)

    # 统计信息
    stats: Dict = field(default_factory=dict)


def build_index(graph: SemanticGraph) -> GraphIndex:
    """为图谱构建全套索引。"""
    index = GraphIndex(graph=graph)

    _build_inverted_index(index)
    _build_type_index(index)
    _build_edge_index(index)
    _build_stats(index)

    return index


def _build_inverted_index(index: GraphIndex) -> None:
    """构建倒排索引。

    对每个节点的以下字段提取关键词并建索引：
    - label（权重最高）
    - synonyms
    - type
    - description
    - cube / dataset / dimension / column
    - metric / filter / category
    """
    idx: Dict[str, Set[str]] = defaultdict(set)

    for node in index.graph.nodes:
        # 提取该节点的所有可检索文本
        texts = _extract_node_texts(node)
        for text in texts:
            words = _tokenize(text)
            for w in words:
                idx[w].add(node.label)

    index.inverted_index = dict(idx)


def _extract_node_texts(node: Node) -> List[str]:
    """从节点提取所有可检索文本片段。"""
    texts = [
        node.label,
        node.type,
        node.description or "",
        node.cube or "",
        node.dataset or "",
        node.dimension or "",
        node.column or "",
        node.metric or "",
        node.measure or "",
        node.filter or "",
        node.category or "",
        node.label_alias or "",
        node.agg or "",
        node.filter_type or "",
        node.metric_type or "",
        node.unit or "",
    ]
    # 同义词
    for s in node.synonyms:
        texts.append(s)
    # 主题域
    for d in node.subject_domains:
        texts.append(d)
    # 依赖项
    for dep in node.depends_on:
        texts.append(dep)

    return [t for t in texts if t]


def _tokenize(text: str) -> List[str]:
    """中文混合分词：按中文词组 + 英文单词 + 数字切分。

    策略：中文按单字/双字切分，英文按空白和标点切分，数字保留原样。
    """
    tokens = []

    # 提取中文词组（连续的汉字）
    chinese_chunks = re.findall(r'[\u4e00-\u9fff]+', text)
    for chunk in chinese_chunks:
        # 单字 + 双字滑动窗口
        for i in range(len(chunk)):
            tokens.append(chunk[i])
            if i + 1 < len(chunk):
                tokens.append(chunk[i:i + 2])

    # 提取英文/数字/下划线 token
    non_chinese = re.sub(r'[\u4e00-\u9fff]+', ' ', text)
    for token in re.findall(r'[a-zA-Z0-9_]+', non_chinese):
        tokens.append(token.lower())

    # 去重
    return list(set(tokens))


def _build_type_index(index: GraphIndex) -> None:
    """按节点类型分组。"""
    by_type: Dict[str, List[Node]] = defaultdict(list)
    for node in index.graph.nodes:
        by_type[node.type].append(node)
    index.nodes_by_type = dict(by_type)


def _build_edge_index(index: GraphIndex) -> None:
    """按边 label 分组，并构建出边类型索引。"""
    by_label: Dict[str, List[Edge]] = defaultdict(list)
    out_labels: Dict[str, Set[str]] = defaultdict(set)

    for edge in index.graph.edges:
        by_label[edge.label].append(edge)
        out_labels[edge.from_label].add(edge.label)

    index.edges_by_label = dict(by_label)
    index.out_edge_labels = {k: v for k, v in out_labels.items()}


def _build_stats(index: GraphIndex) -> None:
    """生成图谱统计信息。"""
    graph = index.graph
    stats = {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "sql_edges": sum(1 for e in graph.edges if e.sql_edge),
        "semantic_edges": sum(1 for e in graph.edges if not e.sql_edge),
        "nodes_by_type": {t: len(ns) for t, ns in index.nodes_by_type.items()},
        "edges_by_label": {lbl: len(es) for lbl, es in index.edges_by_label.items()},
        "nodes_with_warning": [n.label for n in graph.nodes if n.warning],
        "nodes_missing_metric": [n.label for n in graph.nodes if n.type == "Metric" and not n.metric],
        "nodes_missing_filter": [n.label for n in graph.nodes if "Filter" in n.type and not n.filter],
        "domain": graph.domain,
        "description": graph.description,
    }
    index.stats = stats
