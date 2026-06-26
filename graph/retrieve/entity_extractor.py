"""M1: 实体抽取模块。

重新设计的三级策略：
1. 规则匹配：仅匹配图谱中「固定概念」节点（Metric/Function/ConceptFilter/Event/MetricCategory）
   —— 这些节点有明确的名称，别名映射是有效的。
2. Embedding 语义匹配（BAAI/bge-base-zh-v1.5）：
   —— 识别 Attribute 类型（城市、行业等）的「维度引用」，以及补充遗漏的固定概念。
3. LLM 抽取（DeepSeek API）：
   —— 主力识别 Attribute 值→类型的映射（如"嘉兴"→"城市"、"互联网"→"行业"），
   以及复杂语义场景下的一步到位抽取。

关键改进：
- 不再手工维护城市名/行业名等动态值别名（虚假繁荣）
- Attribute 节点的匹配交给 Embedding + LLM
- 规则匹配退守到真正可靠的「固定概念名」匹配
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loader import Node

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# 规则匹配配置
# ============================================================

# 节点类型分类：哪些类型适合规则匹配（固定概念名），哪些不适合
# 注意：Function/MetricCategory 不纳入规则匹配！
# 因为它们的别名太通用（"计数""最大值""占比""同比"），在规则阶段命中会造成大量误匹配。
# 这些泛化操作符应只在 Embedding+LLM 阶段按需召回。
RULE_SAFE_TYPES = {
    "Metric",          # 商机数、出库金额、去重商机金额...
    "Concept/Filter",  # 赢单商机、规上商机、在途商机...
    "Event",           # 商机创建、首次签章、中标...
}

# Attribute 类型不适合规则匹配（值太多，如嘉兴、互联网...），交给 Embedding+LLM
# "Attribute": 城市、行业、子行业、产品类别...

# 别名映射表：只映射图谱中「固定概念」节点的同义说法
# 格式：别名 → 图谱节点 label（目标必须是 RULE_SAFE_TYPES 中的节点）
ALIAS_MAP: Dict[str, str] = {
    # ── Concept/Filter 别名 ──
    "大项目": "规上商机", "重大项目": "规上商机",
    "大商机": "规上商机", "规模以上商机": "规上商机",
    "10万以上": "规上商机", "出库额10万以上": "规上商机",
    "规上": "规上商机",

    "赢单": "赢单商机",
    "未中标": "未中标商机", "丢单": "未中标商机",
    "在途": "在途商机", "进行中": "在途商机",
    "新建": "新建商机",
    "已签": "已签商机", "已签约": "已签商机",

    # ── Metric 别名 ──
    "商机数量": "商机数", "机会数量": "商机数",
    "商机总数": "商机数", "机会总数": "商机数",
    "商机个数": "商机数", "机会个数": "商机数",

    "商机总额": "总商机金额",
    "商机总金额": "总商机金额", "未去重金额": "总商机金额",

    "去重金额": "去重商机金额", "去重商机总额": "去重商机金额",
    "去重后金额": "去重商机金额",

    "出库额": "出库金额", "交货金额": "出库金额",
    "发货金额": "出库金额", "出库总额": "出库金额",

    "去重商机数量": "去重商机数", "去重数量": "去重商机数",
    "重复商机数量": "重复商机数", "重复数量": "重复商机数",

    # ── Event 别名 ──
    "创建日期": "商机创建", "创建时间": "商机创建",
    "预计签单日期": "预计签单", "预签日期": "预计签单",
    "签章日期": "首次签章",
    "下单日期": "首次下单", "首次下单日期": "首次下单",
    "出库日期": "首次出库", "首次出库日期": "首次出库",
    "中标时间": "中标", "中标日期": "中标",
    "签章": "首次签章",

    # ── Function 别名 ──
    "占比": "占比", "比例": "占比", "百分比": "占比",
    "占总体比例": "占比",
    "缺口": "缺口", "差额": "缺口", "差距": "缺口",
    "增长率": "增长率", "增长": "增长率",
    "同比": "同比", "年同比": "同比",
    "环比": "环比", "月环比": "环比",
    "排名": "排名", "排序": "排名", "Top": "排名",
    "求和": "求和", "合计": "求和", "总和": "求和",
    "均值": "均值", "平均": "均值", "平均值": "均值",
    "计数": "计数", "数量": "计数",
    "最大值": "最大值", "最大值": "最大值",
    "最小值": "最小值", "最小值": "最小值",

    # ── MetricCategory 别名 ──
    "金额类指标": "金额指标",
    "数量类指标": "数量指标",
}

class EntityExtractor:
    """实体抽取器：重新设计的三级策略。

    - L1 规则匹配：仅匹配 RULE_SAFE_TYPES 的固定概念节点
    - L2 Embedding：主力匹配 Attribute 节点 + 补充遗漏的固定概念
    - L3 LLM：复杂语义场景的精准抽取（含 Attribute 值→类型映射）
    """

    def __init__(self, node_labels: Set[str], node_map: Dict[str, Node],
                 cache_path: Optional[Path] = None, graph_path: Optional[Path] = None):
        self.node_labels = node_labels
        self.node_map = node_map

        # 分离两类节点
        self._safe_labels: Set[str] = set()       # 规则可匹配的固定概念
        self._attr_labels: Set[str] = set()        # Attribute 节点（需 Embedding/LLM）
        for label, node in node_map.items():
            if node.type in RULE_SAFE_TYPES:
                self._safe_labels.add(label)
            elif node.type == "Attribute":
                self._attr_labels.add(label)
            # Entity 节点（商机、客户、出库明细）也适合规则匹配
            elif node.type == "Entity":
                self._safe_labels.add(label)

        # 构建别名 → label 映射（仅覆盖 RULE_SAFE_TYPES + Entity 节点）
        self._alias_to_label: Dict[str, str] = {}
        for label in self._safe_labels:
            node = node_map[label]
            self._alias_to_label[label] = label  # label 自身也是别名
            for syn in node.synonyms:
                if syn not in self._alias_to_label:
                    self._alias_to_label[syn] = label

        # 合并手工别名表（只合并目标在 safe_labels 中的）
        for alias, target in ALIAS_MAP.items():
            if target in self._safe_labels:
                self._alias_to_label[alias] = target

        # 按别名长度降序排列，优先长匹配
        self._sorted_aliases = sorted(
            self._alias_to_label.keys(),
            key=lambda x: -len(x)
        )

        # 构建 Concept/Filter 节点的触发词反向映射（用于 LLM 后处理去噪）
        # concept_label → [触发词列表]，问题文字中至少出现一个触发词才保留该概念
        self._concept_triggers: Dict[str, List[str]] = {}
        for alias, target in ALIAS_MAP.items():
            node = node_map.get(target)
            if node and node.type in ("Concept/Filter",):
                self._concept_triggers.setdefault(target, []).append(alias)
        # 每个 Concept/Filter 自身 label 也是触发词
        for label, node in node_map.items():
            if node.type in ("Concept/Filter",):
                self._concept_triggers.setdefault(label, [])
                if label not in self._concept_triggers[label]:
                    self._concept_triggers[label].append(label)

        # Embedding 模型（延迟加载）
        self._embedding_model = None
        self._node_embeddings: Optional[Dict[str, List[float]]] = None

        # ── 缓存路径 ──
        self._cache_path = cache_path
        self._graph_path = graph_path
        self._cache_tried = False

        # 构建派生指标映射（泛化：从图谱边的"衍生依赖"关系自动提取）
        # derived_metric -> {base_metric1, base_metric2, ...}，用于去重
        self._derived_map: Dict[str, Set[str]] = {}
        if graph_path and Path(graph_path).exists():
            try:
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.loads(f.read())
                for edge in graph_data.get('edges', []):
                    dl = edge.get('display_label', '') or edge.get('label', '')
                    if dl in ('衍生依赖', 'derived_from', 'depends_on'):
                        if edge['from'] not in self._derived_map:
                            self._derived_map[edge['from']] = set()
                        self._derived_map[edge['from']].add(edge['to'])
            except Exception:
                pass

    # ── 缓存加载 ────────────────────────────────────────

    def _try_load_from_cache(self):
        """尝试从持久化缓存加载节点和别名向量。"""
        if self._cache_tried:
            return
        self._cache_tried = True

        if not self._cache_path or not self._graph_path:
            return

        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
        from embedding_store import load_cache
        cache = load_cache(self._cache_path, self._graph_path, model_name)
        if cache is None:
            return

        self._node_embeddings = cache["node_embeddings"]
        self._alias_embeddings = cache["alias_embeddings"]
        self._alias_id_to_label = cache["alias_id_to_label"]
        print(f"[CACHE] ✓ 命中: {len(self._node_embeddings)} 节点向量, "
              f"{len(self._alias_embeddings)} 别名向量")

    # ── L1: 规则匹配（仅固定概念） ─────────────────────

    def _is_chinese_char(self, ch: str) -> bool:
        """判断是否为中文字符。"""
        return '\u4e00' <= ch <= '\u9fff'

    def _has_word_boundary(self, question: str, idx: int, alias: str) -> bool:
        """检查别名匹配是否有词边界问题。

        对于3字及以上的中文别名，防止跨词匹配。
        核心场景："在途商机总金额"中，q.find("商机总金额")会匹配到跨词位置
        ——它从"在途商机"的"商机"开始，跨越到后面的"总金额"。

        策略：如果匹配起始位置的前一个字符是中文，往前扩展最多4字：
        - 如果找到同 label 的更长别名 → 拒绝（长匹配会处理）
        - 如果找到不同 label 的更长别名 → 允许（独立概念拼接）
        - 如果没找到任何更长别名 → 允许（说明别名本身是独立词的起始）
        """
        if len(alias) < 3:
            return True  # 短别名不检查
        if idx > 0 and self._is_chinese_char(question[idx - 1]):
            current_label = self._alias_to_label.get(alias)
            for lookback in range(1, 5):
                if idx - lookback < 0:
                    break
                extended = question[idx - lookback:idx + len(alias)]
                extended_label = self._alias_to_label.get(extended)
                if extended_label is not None:
                    if extended_label == current_label:
                        return False  # 同 label，长匹配会处理
                    else:
                        return True   # 不同 label，独立概念，允许
            # 没找到任何更长别名 → 允许（是独立词的起始）
            return True
        return True

    def extract_by_rules(self, question: str) -> List[str]:
        """规则匹配：仅匹配 RULE_SAFE_TYPES 的固定概念节点。

        不再尝试匹配城市名/行业名等动态值。
        返回去重后的节点 label 列表，按在问题中出现顺序排列。
        """
        matched: List[Tuple[int, int, str]] = []  # (start, end, label)

        # 遍历所有别名，找在 question 中的所有出现位置
        for alias in self._sorted_aliases:
            # 跳过单字别名（太短容易误匹配）
            if len(alias) <= 1:
                continue
            # 跳过2字短别名（如"金额""数量"），太容易在其他长词中误匹配
            # 它们会通过 Embedding + LLM 阶段被正确召回
            if len(alias) == 2 and self._is_chinese_char(alias[0]):
                continue
            label = self._alias_to_label[alias]
            start = 0
            while True:
                idx = question.find(alias, start)
                if idx == -1:
                    break
                # 词边界检查：防止"在途商机总金额"中匹配到"商机总金额"这种跨词情况
                if self._has_word_boundary(question, idx, alias):
                    matched.append((idx, idx + len(alias), label))
                start = idx + 1

        # 按起始位置排序，同位置优先长匹配
        matched.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # 消歧：重叠匹配的处理
        # - 同 label 重叠 → 保留长的（标准最长匹配）
        # - 不同 label 重叠 → 都保留（如"在途商机"和"商机总金额"是独立概念）
        filtered = []
        for m in matched:
            conflict_same_label = False
            for f in filtered:
                if not (m[1] <= f[0] or m[0] >= f[1]):
                    # 重叠了
                    if m[2] == f[2]:
                        # 同 label：保留长的
                        if (m[1] - m[0]) > (f[1] - f[0]):
                            filtered.remove(f)
                        else:
                            conflict_same_label = True
                            break
                    # 不同 label 重叠：两个都保留（它们是独立概念）
            if not conflict_same_label:
                filtered.append(m)

        # 去重，保持顺序
        seen = set()
        result = []
        for _, _, label in filtered:
            if label not in seen:
                seen.add(label)
                result.append(label)

        return result

    # ── L2: Embedding 语义匹配 ──────────────────────────

    def _ensure_embedding_model(self):
        """延迟加载 embedding 模型。"""
        if self._embedding_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.environ.get(
                "EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5"
            )
            self._embedding_model = SentenceTransformer(model_name)
        except ImportError:
            print("[WARN] sentence-transformers 未安装，embedding 匹配不可用")
            self._embedding_model = None
        except Exception as e:
            print(f"[WARN] 加载 embedding 模型失败: {e}")
            self._embedding_model = None

    def _ensure_node_embeddings(self):
        """预计算所有节点的向量。"""
        if self._node_embeddings is not None:
            return

        # 尝试从缓存加载
        self._try_load_from_cache()
        if self._node_embeddings is not None:
            return  # 已从缓存加载，跳过实时编码

        self._ensure_embedding_model()
        if self._embedding_model is None:
            self._node_embeddings = {}
            return

        texts = []
        labels = []
        for label, node in self.node_map.items():
            # 拼接节点描述文本（Attribute 节点重点描述其语义角色）
            parts = [node.label]
            if node.description:
                parts.append(node.description)
            if node.synonyms:
                parts.extend(node.synonyms)
            # Attribute 节点：附加上下文帮助 Embedding 理解其维度含义
            if node.type == "Attribute":
                parts.append(f"按{node.label}维度筛选")
                parts.append(f"{node.label}名称")
            texts.append(" | ".join(parts))
            labels.append(label)

        embeddings = self._embedding_model.encode(texts, normalize_embeddings=True)
        self._node_embeddings = {label: emb.tolist() for label, emb in zip(labels, embeddings)}

    def extract_by_embedding(self, question: str,
                              top_k_attr: int = 1,
                              top_k_safe: int = 2) -> List[str]:
        """Embedding 语义匹配实体。

        对问题做向量化，与所有节点描述向量计算 cosine 相似度。
        核心用途：
        - 识别 Attribute 节点（用户说"嘉兴"→与"城市"节点语义相近）
        - 补充规则遗漏的固定概念节点

        策略：
        - Attribute 节点：只保留超过阈值(0.45)的维度。一个问题通常涉及1个维度类型。
        - 固定概念节点：阈值 0.50，防止把无关概念拉进来

        Args:
            question: 用户问题
            top_k_attr: Attribute 节点最多取几个
            top_k_safe: 固定概念节点最多取几个

        Returns:
            Attribute 节点优先的 label 列表
        """
        self._ensure_embedding_model()
        if self._embedding_model is None:
            return []

        self._ensure_node_embeddings()
        if not self._node_embeddings:
            return []

        # 编码问题
        q_emb = self._embedding_model.encode(
            [question], normalize_embeddings=True
        )[0]

        # 计算所有节点相似度
        import numpy as np
        all_scores = []
        for label, emb in self._node_embeddings.items():
            sim = np.dot(q_emb, emb)
            all_scores.append((label, float(sim)))

        all_scores.sort(key=lambda x: -x[1])

        # Attribute 节点：只保留超过阈值 0.45 的
        attr_scores = [(l, s) for l, s in all_scores
                       if self.node_map.get(l) and self.node_map[l].type == "Attribute"]
        attr_scores.sort(key=lambda x: -x[1])
        ATTR_THRESHOLD = 0.45
        attr_results = [l for l, s in attr_scores[:top_k_attr] if s > ATTR_THRESHOLD]

        # 固定概念节点：绝对阈值过滤
        safe_scores = [(l, s) for l, s in all_scores
                       if self.node_map.get(l) and self.node_map[l].type != "Attribute"]
        SAFE_THRESHOLD = 0.50
        safe_results = [l for l, s in safe_scores[:top_k_safe] if s > SAFE_THRESHOLD]

        return attr_results + safe_results

    # ── L2.5: Embedding 关键词提取 ────────────────────────

    def _ensure_alias_embeddings(self):
        """预计算所有别名+节点描述的向量，用于关键词级语义匹配。

        覆盖范围包括：
        - _safe_labels 节点的所有别名（规则匹配的固定概念）
        - Function / MetricCategory 节点的别名（不参与规则匹配但需要被 Embedding 召回）
        - 手工 ALIAS_MAP 中的别名
        """
        if hasattr(self, '_alias_embeddings') and self._alias_embeddings is not None:
            return

        # 尝试从缓存加载
        self._try_load_from_cache()
        if hasattr(self, '_alias_embeddings') and self._alias_embeddings is not None:
            return  # 已从缓存加载

        self._ensure_embedding_model()
        if self._embedding_model is None:
            self._alias_embeddings = {}
            self._alias_id_to_label = {}
            return

        # 收集所有需要 Embedding 索引的别名
        alias_to_label_emb: Dict[str, str] = dict(self._alias_to_label)

        # 补充 Function / MetricCategory 节点的别名
        for label, node in self.node_map.items():
            if node.type in ("Function", "MetricCategory"):
                alias_to_label_emb[label] = label
                for syn in node.synonyms:
                    if syn not in alias_to_label_emb:
                        alias_to_label_emb[syn] = label

        # 补充 ALIAS_MAP 中 Function/MetricCategory 相关的别名
        for alias, target in ALIAS_MAP.items():
            if target in self.node_map and self.node_map[target].type in ("Function", "MetricCategory"):
                if alias not in alias_to_label_emb:
                    alias_to_label_emb[alias] = target

        texts = []
        alias_ids = []  # 用 (alias, label) 作为 id
        for alias, label in alias_to_label_emb.items():
            # 别名文本：别名本身 + 目标节点的描述信息
            node = self.node_map.get(label)
            parts = [alias]
            if node:
                parts.append(node.label)
                if node.description:
                    parts.append(node.description)
                if node.type:
                    parts.append(f"类型:{node.type}")
            texts.append(" | ".join(parts))
            alias_ids.append((alias, label))

        embeddings = self._embedding_model.encode(texts, normalize_embeddings=True)
        self._alias_embeddings = {
            alias_id: emb.tolist() for alias_id, emb in zip(alias_ids, embeddings)
        }
        self._alias_id_to_label = {aid: lbl for aid, lbl in alias_ids}

    def extract_by_embedding_keywords(self, question: str,
                                       threshold: float = 0.55,
                                       max_results: int = 10) -> List[Tuple[str, float]]:
        """Embedding 语义关键词提取：用整个问题的向量 vs 所有别名/节点向量。

        与 extract_by_embedding 的区别：
        - extract_by_embedding：问题 → 节点描述向量，侧重 Attribute 维度识别
        - extract_by_embedding_keywords：问题 → 所有别名向量，语义级关键词召回

        优势：不依赖精确字符串匹配，能捕获"总金额"→"总商机金额"、"占比"→"占比"等
        同义表达，同时用整个问题向量做语义检索，比词级匹配更准确。

        Args:
            question: 用户问题
            threshold: 相似度阈值（默认 0.55，比节点级更高因为别名更精确）
            max_results: 最多返回结果数

        Returns:
            [(label, similarity_score), ...] 按相似度降序排列
        """
        self._ensure_embedding_model()
        if self._embedding_model is None:
            return []

        self._ensure_alias_embeddings()
        if not self._alias_embeddings:
            return []

        import numpy as np
        q_emb = self._embedding_model.encode(
            [question], normalize_embeddings=True
        )[0]

        # 计算问题向量与所有别名向量的相似度
        all_scores: List[Tuple[str, float]] = []
        for (alias, label), emb in self._alias_embeddings.items():
            sim = np.dot(q_emb, emb)
            all_scores.append((label, float(sim)))

        # 按相似度降序排列，去重保留最高分
        all_scores.sort(key=lambda x: -x[1])
        seen = set()
        results = []
        for label, score in all_scores:
            if label not in seen and score >= threshold:
                seen.add(label)
                results.append((label, score))
                if len(results) >= max_results:
                    break

        return results

    def extract_function_terms(self, question: str,
                                threshold: float = 0.45,
                                max_results: int = 5) -> List[Tuple[str, float]]:
        """专门针对 Function 类短词的逐词 Embedding 召回。

        整个问题向量 vs 短别名（占比/同比/环比等）的相似度通常很低，
        因为这些功能词在长问题中被语义稀释。这里改用逐词（2-3字滑动窗口）
        编码后 vs 别名向量做匹配，大幅提升短功能词的召回率。

        Args:
            question: 用户问题
            threshold: 相似度阈值
            max_results: 最多返回结果数

        Returns:
            [(label, similarity_score), ...]
        """
        self._ensure_embedding_model()
        if self._embedding_model is None:
            return []

        self._ensure_alias_embeddings()
        if not self._alias_embeddings:
            return []

        import numpy as np

        # 提取问题中的 2-3 字中文片段作为滑动窗口
        fragments = []
        for win_size in (2, 3):
            for i in range(len(question) - win_size + 1):
                frag = question[i:i + win_size]
                # 只保留纯中文片段
                if all(self._is_chinese_char(ch) for ch in frag):
                    fragments.append(frag)

        if not fragments:
            return []

        # 编码所有片段
        frag_embs = self._embedding_model.encode(
            fragments, normalize_embeddings=True
        )

        # 对每个别名计算与所有片段的最大相似度
        label_max_scores: Dict[str, float] = {}
        for (alias, label), emb in self._alias_embeddings.items():
            sims = np.dot(frag_embs, np.array(emb))
            max_sim = float(np.max(sims))
            if max_sim >= threshold:
                if label not in label_max_scores or max_sim > label_max_scores[label]:
                    label_max_scores[label] = max_sim

        # 按相似度降序排列
        results = sorted(label_max_scores.items(), key=lambda x: -x[1])
        return results[:max_results]

    # ── L3: LLM 抽取 ─────────────────────────────────────

    def _load_llm_config(self) -> dict:
        """加载 LLM 配置（优先 config.yaml，fallback 环境变量）。"""
        config = {}
        config_path = BASE_DIR / "config.yaml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            except ImportError:
                with open(config_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or ":" not in line:
                            continue
                        key, _, val = line.partition(":")
                        key = key.strip().strip('"').strip("'")
                        val = val.strip().strip('"').strip("'")
                        config[key] = val
        ds = config.get("deepseek", config)
        api_key = ds.get("api_key", "") or os.environ.get("DEEPSEEK_API_KEY", "")
        base_url = ds.get("base_url", "https://api.deepseek.com/chat/completions")
        model = ds.get("model", "deepseek-v4-flash")
        return {"api_key": api_key, "base_url": base_url, "model": model}

    def extract_by_llm(self, question: str) -> List[str]:
        """调用 LLM 抽取实体（全量模式，独立工作）。

        核心用途：
        - 识别 Attribute 值→类型的映射（"嘉兴"→"城市"、"互联网"→"行业"）
        - 复杂语义场景的精准抽取
        """
        llm_cfg = self._load_llm_config()
        api_key = llm_cfg["api_key"]
        if not api_key or api_key == "your-deepseek-api-key-here":
            print("[WARN] 请先在 config.yaml 中配置 deepseek.api_key")
            return []

        # 构建 prompt：分类列出节点，帮助 LLM 理解哪些是维度
        safe_nodes = []
        attr_nodes = []
        entity_nodes = []
        for node in sorted(self.node_map.values(), key=lambda n: n.label):
            desc = node.description or ""
            syns = f"（同义词：{'、'.join(node.synonyms)}）" if node.synonyms else ""
            line = f"- {node.label}{syns}: {desc}"
            if node.type == "Attribute":
                attr_nodes.append(line)
            elif node.type == "Entity":
                entity_nodes.append(line)
            else:
                safe_nodes.append(line)

        prompt = f"""你是知识图谱实体抽取助手。从用户问题中识别涉及的知识图谱节点。

## 维度节点（用户问题中的具体值需映射到这些节点）：
{chr(10).join(attr_nodes) if attr_nodes else '(无)'}

## 实体节点：
{chr(10).join(entity_nodes) if entity_nodes else '(无)'}

## 固定概念节点（指标、函数、过滤器等）：
{chr(10).join(safe_nodes) if safe_nodes else '(无)'}

## 规则说明：
1. 如果用户提到具体地名（如"嘉兴""北京"），映射到维度节点「城市」
2. 如果用户提到具体行业名（如"互联网""金融"），映射到维度节点「行业」
3. 如果用户提到具体产品（如"服务器""数据库"），映射到维度节点「产品类别」
4. 其他固定概念（指标名、函数名、过滤器名）直接匹配节点名
5. 不要输出维度节点本身的 label（如不要单独输出「城市」），除非用户的问题直接提到了维度这个概念本身（如"按城市分组"）

用户问题：{question}

请严格按 JSON 数组格式输出匹配的节点 label，只输出 label 列表，不要其他内容。
示例：["商机数", "出库金额", "占比", "赢单商机"]
直接输出："""

        return self._call_llm(prompt)

    def extract_by_llm_filter(self, question: str, candidates: List[str]) -> List[str]:
        """Embedding 召回 → LLM 筛选模式。

        Embedding 先做宽召回（高阈值），把候选节点列表给 LLM 做精准筛选。
        LLM 只从候选列表中勾选真正相关的节点，不做开放式抽取。

        优势：
        - Embedding 保召回率（多拉候选）
        - LLM 保准确率（精准过滤噪音）
        - 候选集远小于全量节点，LLM 判断更准确、更快

        Args:
            question: 用户问题
            candidates: Embedding 召回的候选节点 label 列表

        Returns:
            经过 LLM 筛选后的节点 label 列表
        """
        if not candidates:
            return []

        llm_cfg = self._load_llm_config()
        api_key = llm_cfg["api_key"]
        if not api_key or api_key == "your-deepseek-api-key-here":
            print("[WARN] 请先在 config.yaml 中配置 deepseek.api_key")
            return candidates  # 没有 LLM 时直接返回候选

        # 构建候选节点列表（Metric 节点带上 cube/域标签帮助 LLM 区分）
        attr_names = []
        safe_names_with_domain = []
        for label in candidates:
            node = self.node_map.get(label)
            if not node:
                continue
            if node.type == "Attribute":
                attr_names.append(node.label)
            elif node.type == "Metric":
                # 给 Metric 节点标注所属域，帮助 LLM 区分商机金额 vs 出库金额
                cube = getattr(node, 'cube', '') or ''
                if 'opportunity' in cube:
                    domain = '[商机域]'
                elif 'deliver' in cube or 'outbound' in cube:
                    domain = '[出库域]'
                else:
                    domain = ''
                safe_names_with_domain.append(f"{node.label}{domain}")
            else:
                safe_names_with_domain.append(node.label)

        attr_str = "、".join(attr_names) if attr_names else "无"
        safe_str = "、".join(safe_names_with_domain) if safe_names_with_domain else "无"

        prompt = f"""你是知识图谱实体筛选器。从候选节点中选出问题明确需要的节点，严格筛选。

维度节点(问题有地名/行业名/产品名就必须选): {attr_str}
候选固定概念节点: {safe_str}

严格筛选规则:
1. 维度节点: 候选池中的维度节点（城市/行业/产品类别/子行业）必须全部保留！问题有地名→城市必须选；有行业名→行业必须选；有产品名→产品类别必须选；说"各行业"→行业必须选。
2. 指标(Metric)域区分【关键 - 精准匹配，不跨界】:
   - 候选节点后缀 [商机域] 表示该指标来自商机表(cube_kpi_opportunity)，如商机金额、总商机金额、商机数等
   - 候选节点后缀 [出库域] 表示该指标来自出库明细表(cube_gn_sales_deliver_detail)，如出库金额
   - 「商机金额」和「出库金额」是不同域的指标，不要混淆！
   - 问题中明确要"商机金额"/"商机总金额"/"金额"(上下文与商机相关)→只选商机域指标，不要多选出库域指标
   - 问题中明确要"出库金额"/"交货金额"/"发货金额"→只选出库域指标，不要多选商机域指标
   - 问题只说"金额"时，看上下文：前面有"商机"→商机域；前面有"出库"/"交货"/"发货"→出库域
   - 注意：除非问题同时提到两个域的金额（如"对比商机金额和出库金额"），否则不要跨域选取
   - 问题没提的指标一律不选！
3. 指标具体映射:
   - "商机数量"→「商机数」; "去重金额"→「去重商机金额」; "未去重金额"→「总商机金额」
4. 状态过滤器(Concept/Filter)—核心原则【一字不差，互斥不跨界】:
   Concept/Filter 节点代表互不相同的业务状态，每个状态只能通过问题中出现的对应关键词触发。候选池里可能出现多个状态过滤器（如赢单、中标、在途、已签、规上等），它们语义再相近也不是同一个东西！
   选取铁律: 只有问题文字中出现该关键词或其别名时，才选对应的节点。例如: "赢单"→赢单商机; "中标"→中标商机; "在途"→在途商机; "已签"→已签商机; "规上/大项目"→规上商机; "新建"→新建商机; "未中标/丢单"→未中标商机。
   禁止行为: 问题说了"赢单"但选中标商机、问题说了"在途"但选规上商机、问题没说任何状态但瞎选一个——这些都属于跨界误选，严格禁止！
5. 实体(Entity): 只选「商机」（所有指标的基础实体）。除非问题明确提到其他实体。
6. 函数/计算(Function): 问题文字中出现才选。如"占比"→选占比；"同比增长率"→选同比+增长率。

问题: {question}
直接输出JSON数组:"""

        result = self._call_llm(prompt)
        # 只保留在候选集中的结果
        candidate_set = set(candidates)
        return [e for e in result if e in candidate_set]

    def _call_llm(self, prompt: str, max_tokens: int = 4096) -> List[str]:
        """统一的 LLM 调用封装。"""
        llm_cfg = self._load_llm_config()
        api_key = llm_cfg["api_key"]
        if not api_key or api_key == "your-deepseek-api-key-here":
            return []

        try:
            import requests
            resp = requests.post(
                llm_cfg["base_url"],
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm_cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "thinking": {"type": "disabled"},
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            # deepseek-v4 等推理模型返回在 reasoning_content 而非 content
            content = msg.get("content") or msg.get("reasoning_content") or ""
            finish = choice.get("finish_reason", "?")

            # finish_reason=length 说明输出被截断，扩大上限重试
            if finish == "length" and max_tokens < 8192:
                print(f"[LLM_RETRY] 输出被截断，增大 max_tokens 重试")
                return self._call_llm(prompt, max_tokens=8192)

            content = content.strip()
            # 尝试解析 JSON
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            entities = json.loads(content)
            if isinstance(entities, list):
                return [e for e in entities if e in self.node_labels]
        except Exception as e:
            print(f"[WARN] LLM 调用失败: {e}")

        return []

    def _filter_derived_metrics(self, entities: List[str], question: str) -> List[str]:
        """泛化去重：如果同时选了基础指标和派生指标，且问题未明确提派生指标，则去派生。
        
        例如: 问题"商机数量"→选"商机数"不选"去重商机数"
              问题"去重商机数"→两者都保留(问题明确提了)
        
        原理: 从图谱边的"衍生依赖"关系自动构建映射表(支持多父依赖)，无硬编码。
        """
        if not self._derived_map:
            return entities
        entity_set = set(entities)
        result = []
        for e in entities:
            bases = self._derived_map.get(e, set())
            # 如果 e 的任一基础指标在选中列表中，且问题文字中没出现 e 自身 → 去重
            if bases and bases & entity_set and e not in question:
                continue
            result.append(e)
        return result

    def _filter_concept_by_question(self, concepts: List[str], question: str) -> List[str]:
        """确定性后处理：移除问题文字中未出现触发词的 Concept/Filter 节点。

        LLM 精筛有时会跨界误选（如问题说「赢单」但 LLM 选了「中标商机」）。
        本方法用触发词硬匹配确保：问题没提的中标→不会选中标商机。
        触发词来源：ALIAS_MAP 反查 + 节点 label 自身。

        不受模型质量影响、不受 prompt 调优影响、对任意规模的图谱均可工作。
        """
        result = []
        for label in concepts:
            node = self.node_map.get(label)
            if node and node.type == "Concept/Filter":
                triggers = self._concept_triggers.get(label, [label])
                if not any(t in question for t in triggers):
                    continue  # 问题文字中没出现触发词，丢弃
            result.append(label)
        return result

    # ── 综合抽取 ──────────────────────────────────────────

    def extract(self, question: str, use_embedding: bool = True, use_llm: bool = False,
                embedding_threshold: float = 0.35, embedding_max_results: int = 15) -> List[str]:
        """实体抽取主流程。

        流水线：
        1. L1 规则匹配：字符串精确匹配已知别名，确定性强，直接保留
        2. L2 Embedding 宽召回：整个问题向量 vs 所有别名向量，低阈值多拉候选
        3. L2.5 LLM 筛选：从 Embedding 召回的候选中做精准过滤（如果开启 LLM）
        4. L3 Embedding 维度识别：专门识别 Attribute 节点（城市/行业等）

        核心思路：
        - 规则匹配保底（确定性强）
        - Embedding 保召回（语义匹配，宽进）
        - LLM 保精度（从候选中筛选，严出）

        Args:
            question: 用户自然语言问题
            use_embedding: 是否启用 embedding
            use_llm: 是否启用 LLM 筛选

        Returns:
            去重后的节点 label 列表
        """
        entities = []
        rule_set = set()

        # ── Phase 1: 规则匹配（并行跑，确定性结果直接保留）──
        rule_entities = self.extract_by_rules(question)
        entities.extend(rule_entities)
        rule_set = set(rule_entities)

        # ── Phase 2: Embedding 宽召回 + LLM 筛选 ──
        if use_embedding:
            # 2a: Embedding 关键词宽召回（低阈值多拉候选，LLM 精筛兜底）
            kw_results = self.extract_by_embedding_keywords(
                question, threshold=embedding_threshold, max_results=embedding_max_results
            )
            # 提取候选 label（排除规则已匹配的）
            emb_candidates = [label for label, _ in kw_results if label not in rule_set]

            # 2a2: 逐词 Embedding 召回 Function 类短词（占比/同比/环比等）
            func_results = self.extract_function_terms(question, threshold=0.45, max_results=5)
            for label, _ in func_results:
                if label not in rule_set and label not in emb_candidates:
                    emb_candidates.append(label)

            # 2b: Embedding 维度识别 → 维度节点直接保留，不交给 LLM 筛选
            dim_results = self.extract_by_embedding(question, top_k_attr=2, top_k_safe=3)
            for d in dim_results:
                node = self.node_map.get(d)
                if node and node.type == "Attribute":
                    # Attribute 维度节点直接加入最终结果
                    if d not in rule_set and d not in entities:
                        entities.append(d)
                else:
                    # 非维度节点加入候选池给 LLM 筛选
                    if d not in rule_set and d not in emb_candidates:
                        emb_candidates.append(d)

            if use_llm and emb_candidates:
                # 2c: LLM 从候选池中精准筛选
                filtered = self.extract_by_llm_filter(question, emb_candidates)
                # 2d: 确定性后处理 — 去掉 LLM 跨界误选的 Concept/Filter
                filtered = self._filter_concept_by_question(filtered, question)
                # 2e: 派生指标去重 — 基础指标和派生指标同时命中时去派生
                filtered = self._filter_derived_metrics(filtered, question)
                for e in filtered:
                    if e not in rule_set:
                        entities.append(e)
            else:
                # 不开 LLM 时直接用 Embedding 结果
                for e in emb_candidates:
                    if e not in rule_set:
                        entities.append(e)

        return entities


def build_extractor(node_labels: Set[str], node_map: Dict[str, Node],
                    cache_path: Optional[Path] = None,
                    graph_path: Optional[Path] = None) -> EntityExtractor:
    """工厂函数：构建实体抽取器。"""
    return EntityExtractor(node_labels, node_map,
                           cache_path=cache_path, graph_path=graph_path)
