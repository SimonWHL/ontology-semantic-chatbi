"""指标分析能力矩阵 (MetricCapability).

提取图谱中 SQL 边（where/group_by/having），构建维度-指标分析能力索引。
独立于语义图谱检索，在子图构建完成后作为附属信息注入，不参与路径搜索。

SQL 边映射的三种操作类型：
  - group_by: 维度可作为分组维度分析该指标（如 城市→商机数）  
  - where:   维度/概念可作为过滤条件筛选该指标（如 城市→出库金额, 赢单商机→商机数）
  - having:  概念可作为聚合后条件筛选该指标（如 规上商机→商机数）
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from loader import Edge, SemanticGraph


# ── 数据模型 ──

@dataclass
class CapabilityEntry:
    """一条维度-指标分析能力记录。"""
    dimension: str        # 维度/概念节点 label
    metric: str           # 指标节点 label
    op_type: str          # "group_by" | "where" | "having"
    display: str          # 中文描述


# ── 能力矩阵 ──

class MetricCapability:
    """从 SQL 边构建维度-指标分析能力矩阵。

    提供两种查询方向：
    1. 维度→指标: 某个维度/概念能对哪些指标做什么操作
    2. 指标→维度: 某个指标能被哪些维度/概念分析
    """

    def __init__(self, graph: SemanticGraph):
        self._entries: List[CapabilityEntry] = []
        self._dim_to_metrics: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._metric_to_dims: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._all_dims: Set[str] = set()
        self._all_metrics: Set[str] = set()
        self._build(graph)

    def _build(self, graph: SemanticGraph):
        for edge in graph.edges:
            if not edge.sql_edge:
                continue
            entry = CapabilityEntry(
                dimension=edge.from_label,
                metric=edge.to_label,
                op_type=edge.label,
                display=edge.display_label or edge.label,
            )
            self._entries.append(entry)
            self._dim_to_metrics[edge.from_label][edge.to_label].append(edge.label)
            self._metric_to_dims[edge.to_label][edge.from_label].append(edge.label)
            self._all_dims.add(edge.from_label)
            self._all_metrics.add(edge.to_label)

    # ── 查询接口 ──

    def query_dim_metrics(self, dim_label: str) -> Dict[str, List[str]]:
        """维度能分析哪些指标 → {metric_label: [op_types]}"""
        return dict(self._dim_to_metrics.get(dim_label, {}))

    def query_metric_dims(self, metric_label: str) -> Dict[str, List[str]]:
        """指标能被哪些维度分析 → {dim_label: [op_types]}"""
        return dict(self._metric_to_dims.get(metric_label, {}))

    def can_analyze(self, dim_label: str, metric_label: str) -> bool:
        """维度是否能分析该指标（任意操作类型）。"""
        return metric_label in self._dim_to_metrics.get(dim_label, {})

    # ── 批量描述（注入下游 LLM）──

    def describe_for_entities(
        self,
        entities: List[str],
        node_map: Dict[str, Any],
    ) -> str:
        """给定抽取出的实体列表，生成维度-指标分析能力的自然语言描述。

        只输出 entities 中出现的维度/概念节点与指标节点之间的关系。

        Returns:
            格式化的分析能力描述文本，可直接拼接进下游 LLM 的 context。
            entities 中没有维度或指标时返回空字符串。
        """
        entity_set = set(entities)
        dims_in_entities = [
            e for e in entities
            if e in self._all_dims and e in entity_set
        ]
        metrics_in_entities = [
            e for e in entities
            if e in self._all_metrics and e in entity_set
        ]

        if not dims_in_entities or not metrics_in_entities:
            return ""

        OP_LABELS = {
            "group_by": "可作为分组维度",
            "where": "可作为过滤条件",
            "having": "可作为聚合后条件",
        }

        lines = ["\n## 维度-指标分析能力"]
        for dim in sorted(dims_in_entities):
            dim_metrics = self._dim_to_metrics.get(dim, {})
            relevant = {m: ops for m, ops in dim_metrics.items() if m in metrics_in_entities}
            if not relevant:
                continue
            parts = []
            for metric in sorted(relevant):
                ops = relevant[metric]
                op_descs = sorted({OP_LABELS.get(op, op) for op in ops})
                parts.append(f"   {metric} ({'、'.join(op_descs)})")
            if parts:
                node = node_map.get(dim)
                type_tag = f" [{node.type}]" if node and hasattr(node, 'type') else ""
                lines.append(f"  {dim}{type_tag}:")
                lines.extend(parts)

        # 反过来：指标能被哪些维度分析
        lines.append("")
        for metric in sorted(metrics_in_entities):
            dims_for_metric = self._metric_to_dims.get(metric, {})
            relevant_dims = [
                d for d in dims_for_metric
                if d in dims_in_entities
            ]
            if not relevant_dims:
                # 也检查 entities 中的所有维度（不仅是 classification 出来的）
                all_relevant = [
                    d for d in dims_for_metric if d in entity_set
                ]
                if not all_relevant:
                    continue
            # 只输出"能被哪些维度分析"的正面信息
            analyzer_dims = sorted(set(dims_for_metric) & entity_set)
            if analyzer_dims and len(analyzer_dims) <= 8:
                lines.append(f"  {metric} 可被以下维度/概念分析: {', '.join(analyzer_dims)}")

        return "\n".join(lines) if len(lines) > 2 else ""


def build_capability(graph: SemanticGraph) -> MetricCapability:
    """工厂函数。"""
    return MetricCapability(graph)
