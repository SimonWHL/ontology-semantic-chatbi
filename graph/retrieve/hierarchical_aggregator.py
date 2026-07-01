"""分层知识图谱聚合引擎(Hierarchical Graph Aggregation).

自底向上递归构建多层级连通语义网络:

1. 语义聚类:对底层实体做 Embedding 编码,GMM 完成语义相似实体分簇
2. 聚合实体生成:LLM 为每个簇生成高层抽象概念节点
3. 跨簇关联:统计簇间实体关联强度,超阈值由 LLM 生成高层抽象关系
4. 递归迭代:生成多层图谱 H={G0, G1, ...Gk}

层级图谱结构:
    G0: 原始细粒度实体(原图谱的节点/边)
    G1: 第一层抽象概念 + 跨簇关系
    G2: 更高层抽象(可选)
    ...

每层都包含 nodes,edges,以及到下一层的映射关系.
"""

from __future__ import annotations

import json
import hashlib
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from loader import Edge, Node, SemanticGraph


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class AbstractNode:
    """抽象层节点(G1+ 层的聚合概念)."""
    label: str                          # 抽象概念名(LLM 生成)
    description: str                    # 概念描述(LLM 生成)
    level: int                          # 所在层级(1, 2, ...)
    member_labels: List[str]            # 该抽象节点包含的底层实体 label
    member_relations: List[dict]        # 簇内关系摘要
    cluster_id: int                     # 所属簇 ID
    embedding: Optional[List[float]] = None  # 抽象节点的 embedding(成员均值)


@dataclass
class AbstractEdge:
    """抽象层边(G1+ 层的跨簇关系)."""
    from_label: str
    to_label: str
    label: str                          # 关系类型(LLM 生成)
    display_label: str                  # 关系描述
    strength: float                     # 簇间关联强度(0~1)
    level: int                          # 所在层级


@dataclass
class HierarchicalGraph:
    """多层聚合图谱.

    layers[0] = G0 原始图谱
    layers[1] = G1 第一层抽象
    layers[2] = G2 第二层抽象(如有)
    ...
    """
    layers: List[dict] = field(default_factory=list)
    # 每层: {"nodes": [...], "edges": [...], "level": int}
    abstract_nodes: List[AbstractNode] = field(default_factory=list)
    abstract_edges: List[AbstractEdge] = field(default_factory=list)

    # G0 的 embedding 和簇分配(持久化用)
    g0_embeddings: Optional[Dict[str, List[float]]] = None
    g0_clusters: Optional[Dict[int, List[str]]] = None

    # 可导航层级索引：用于 ancestor / LCA 检索。
    # parent_of: child label -> parent abstract labels
    # children_of: abstract label -> direct children labels
    # level_of: every known label -> hierarchy level
    parent_of: Dict[str, List[str]] = field(default_factory=dict)
    children_of: Dict[str, List[str]] = field(default_factory=dict)
    level_of: Dict[str, int] = field(default_factory=dict)

    @property
    def num_levels(self) -> int:
        return len(self.layers)


# ══════════════════════════════════════════════════════════════
# Embedding 编码
# ══════════════════════════════════════════════════════════════

_EMBEDDING_MODEL = None


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL


def _encode_nodes(nodes: List[Node]) -> Dict[str, np.ndarray]:
    """对节点列表做 embedding 编码.

    每个节点的编码文本 = label + type + description + cube + dimension
    """
    model = _get_embedding_model()
    texts = []
    for n in nodes:
        parts = [n.label, n.type]
        if n.description:
            parts.append(n.description)
        if n.cube:
            parts.append(n.cube)
        if n.dimension:
            parts.append(n.dimension)
        texts.append(" ".join(parts))

    embeddings = model.encode(texts, normalize_embeddings=True)
    return {n.label: emb for n, emb in zip(nodes, embeddings)}


# ══════════════════════════════════════════════════════════════
# GMM 语义聚类
# ══════════════════════════════════════════════════════════════

def _cluster_by_gmm(
    embeddings: Dict[str, np.ndarray],
    n_components: Optional[int] = None,
    min_clusters: int = 3,
    max_clusters: int = 15,
) -> Dict[int, List[str]]:
    """使用 GMM 对实体 embedding 做软聚类.

    Args:
        embeddings: {label: embedding_array}
        n_components: 簇数,None 则自动选择(BIC 准则)
        min_clusters: 最小簇数
        max_clusters: 最大簇数

    Returns:
        {cluster_id: [label, ...]}
    """
    labels = list(embeddings.keys())
    X = np.array([embeddings[l] for l in labels])

    n_samples = len(labels)

    if n_samples < min_clusters:
        # 样本太少,全部归为一个簇
        return {0: labels}

    if n_components is None:
        # 自动选择最佳簇数(BIC 准则),用 diagonal 协方差加速
        best_bic = float("inf")
        best_n = min(min_clusters, n_samples)
        effective_max = min(max_clusters, n_samples)

        for n in range(min_clusters, effective_max + 1):
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type="diag",
                    random_state=42,
                    max_iter=100,
                    n_init=1,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
            except Exception:
                continue
        n_components = best_n

    # 最终聚类
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=42,
        max_iter=200,
        n_init=3,
    )
    gmm.fit(X)
    y_pred = gmm.predict(X)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for label, cid in zip(labels, y_pred):
        clusters[int(cid)].append(label)

    return dict(clusters)


# ══════════════════════════════════════════════════════════════
# LLM 调用
# ══════════════════════════════════════════════════════════════

def _load_deepseek_config() -> dict:
    """加载 DeepSeek API 配置."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config.get("deepseek", {})
    return {}


def _call_llm(prompt: str, system_prompt: str = "") -> str:
    """调用 DeepSeek API."""
    import urllib.request
    import urllib.error

    config = _load_deepseek_config()
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "https://api.deepseek.com/chat/completions")
    model = config.get("model", "deepseek-v4-flash")

    if not api_key:
        return ""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
    }).encode("utf-8")

    req = urllib.request.Request(
        base_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[WARN] LLM 调用失败: {e}")
        return ""


def _generate_abstract_concept(
    cluster_id: int,
    member_labels: List[str],
    node_details: List[dict],
    level: int,
) -> dict:
    """LLM 为一个簇生成抽象概念.

    Returns:
        {"label": "抽象概念名", "description": "概念描述", "key_relations": [...]}
    """
    members_text = "\n".join(
        f"- {d['label']} ({d['type']}): {d.get('description', '')}"
        for d in node_details
    )

    prompt = f"""你是一个知识图谱专家.以下是一组语义相似的实体节点(层级 {level}):

{members_text}

请为这组实体生成一个**高层抽象概念**,要求:
1. label(概念名):简洁的 2-6 字中文名称,概括这组实体的共同语义
2. description(描述):一句话说明这组实体是什么,约 20-50 字
3. key_relations(关键关系):列出这组实体内部最重要的 1-3 种关系类型

请严格按以下 JSON 格式输出(不要输出其他内容):
{{"label": "概念名", "description": "描述", "key_relations": ["关系1", "关系2"]}}"""

    result = _call_llm(prompt)
    if not result:
        # LLM 不可用时,用最常见的 label 作为概念名
        best = max(member_labels, key=len, default=f"cluster_{cluster_id}")
        return {
            "label": f"抽象_{best}",
            "description": f"包含 {len(member_labels)} 个相关实体的语义簇",
            "key_relations": [],
        }

    # 尝试解析 JSON
    try:
        # 提取 JSON 块
        import re
        json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return {
                "label": parsed.get("label", f"cluster_{cluster_id}"),
                "description": parsed.get("description", ""),
                "key_relations": parsed.get("key_relations", []),
            }
    except (json.JSONDecodeError, Exception):
        pass

    return {
        "label": result[:20].strip(),
        "description": result[:100].strip(),
        "key_relations": [],
    }


def _generate_cross_cluster_relations(
    from_abstract: str,
    from_desc: str,
    to_abstract: str,
    to_desc: str,
    link_count: int,
    sample_links: List[dict],
) -> dict:
    """LLM 生成跨簇抽象关系.

    Returns:
        {"label": "关系类型", "display_label": "关系描述"}
    """
    samples_text = "\n".join(
        f"- {l['from']} --[{l['edge_label']}]--> {l['to']}"
        for l in sample_links[:5]
    )

    prompt = f"""你是知识图谱专家.两个抽象概念簇之间存在 {link_count} 条底层关联:

簇A: {from_abstract} — {from_desc}
簇B: {to_abstract} — {to_desc}

底层关联示例:
{samples_text}

请为这两个簇生成一条**高层抽象关系**,概括它们之间的语义联系.
要求:
1. label(关系类型):1-4字中文,如"业务依赖""数据关联""语义相似"
2. display_label(关系描述):10-25字描述

严格按 JSON 格式输出:
{{"label": "关系类型", "display_label": "关系描述"}}"""

    result = _call_llm(prompt)
    if not result:
        return {"label": "relates_to", "display_label": f"存在{link_count}条底层关联"}

    try:
        import re
        json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return {
                "label": parsed.get("label", "relates_to"),
                "display_label": parsed.get("display_label", f"存在{link_count}条关联"),
            }
    except Exception:
        pass

    return {"label": "relates_to", "display_label": f"存在{link_count}条底层关联"}


# ══════════════════════════════════════════════════════════════
# 跨簇关联强度计算
# ══════════════════════════════════════════════════════════════

def _compute_cross_cluster_links(
    clusters: Dict[int, List[str]],
    graph: SemanticGraph,
    threshold: float = 0.1,
) -> List[dict]:
    """统计簇间实体关联强度,返回超过阈值的跨簇关联.

    Returns:
        [{
            "from_cluster": cid_a,
            "to_cluster": cid_b,
            "strength": 0.0~1.0,
            "link_count": int,
            "sample_links": [{"from":..., "to":..., "edge_label":...}, ...]
        }, ...]
    """
    # 建立 label → cluster_id 映射
    label_to_cluster: Dict[str, int] = {}
    for cid, members in clusters.items():
        for label in members:
            label_to_cluster[label] = cid

    # 统计簇间边数
    cross_links: Dict[Tuple[int, int], dict] = {}

    for edge in graph.edges:
        if edge.sql_edge:
            continue
        c_from = label_to_cluster.get(edge.from_label)
        c_to = label_to_cluster.get(edge.to_label)
        if c_from is None or c_to is None:
            continue
        if c_from == c_to:
            continue

        key = tuple(sorted([c_from, c_to]))
        if key not in cross_links:
            cross_links[key] = {
                "from_cluster": c_from,
                "to_cluster": c_to,
                "link_count": 0,
                "sample_links": [],
            }
        cross_links[key]["link_count"] += 1
        if len(cross_links[key]["sample_links"]) < 5:
            cross_links[key]["sample_links"].append({
                "from": edge.from_label,
                "to": edge.to_label,
                "edge_label": edge.display_label or edge.label,
            })

    # 计算强度(归一化:link_count / max_possible_links)
    results = []
    for link in cross_links.values():
        c_from = link["from_cluster"]
        c_to = link["to_cluster"]
        max_possible = len(clusters[c_from]) * len(clusters[c_to])
        strength = link["link_count"] / max(max_possible, 1)
        if strength >= threshold:
            link["strength"] = min(strength, 1.0)
            results.append(link)

    return results


# ══════════════════════════════════════════════════════════════
# 层级图谱构建
# ══════════════════════════════════════════════════════════════

def build_hierarchical_graph(
    graph: SemanticGraph,
    *,
    max_levels: int = 2,
    min_cluster_size: int = 2,
    cross_cluster_threshold: float = 0.1,
    cache_dir: Optional[Path] = None,
    use_llm: bool = True,
) -> HierarchicalGraph:
    """从原始图谱自底向上构建多层聚合图谱.

    Args:
        graph: 原始语义图谱(G0)
        max_levels: 最大层级数(含 G0,即最多构建到 G{max_levels-1})
        min_cluster_size: 每簇最少实体数
        cross_cluster_threshold: 跨簇关联强度阈值
        cache_dir: 缓存目录(避免重复 LLM 调用)
        use_llm: 是否启用 LLM 生成抽象概念

    Returns:
        HierarchicalGraph 多层图谱
    """
    # 尝试从缓存加载(pickle 格式, 含指纹验证)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "hierarchical_graph.pkl"
        fingerprint_path = cache_dir / "graph_fingerprint.txt"

        # 计算当前图的指纹: node数:edge数:domain:labels_hash
        fp_parts = [
            str(len(graph.nodes)),
            str(len(graph.edges)),
            graph.domain or "",
            hashlib.md5(",".join(sorted(n.label for n in graph.nodes)).encode()).hexdigest(),
        ]
        current_fp = ":".join(fp_parts)

        if cache_path.exists() and fingerprint_path.exists():
            try:
                cached_fp = fingerprint_path.read_text(encoding="utf-8").strip()
                if cached_fp == current_fp:
                    import time
                    t0 = time.time()
                    with open(cache_path, "rb") as f:
                        hg = pickle.load(f)
                    elapsed = time.time() - t0
                    print(f"  ⚡ [Cache] 从磁盘加载分层图谱, 耗时 {elapsed:.1f}s"
                          f" (节点 {fp_parts[0]}, 边 {fp_parts[1]})")
                    return hg
                else:
                    print(f"  [Cache] 图指纹已变化, 重新构建分层图谱")
            except Exception as e:
                print(f"  [Cache] 缓存加载失败 ({e}), 重新构建分层图谱")
        else:
            print(f"  [Build] 缓存不存在, 首次构建分层图谱（约需 60~95 秒）...")

    hg = HierarchicalGraph()
    hg.level_of = {n.label: 0 for n in graph.nodes}
    hg.parent_of = {n.label: [] for n in graph.nodes}
    hg.children_of = {}
    
    # ── G0: 原始图谱 ──
    hg.layers.append({
        "level": 0,
        "nodes": graph.nodes,
        "edges": graph.edges,
        "node_map": graph.node_map,
    })

    current_nodes = graph.nodes
    current_graph = graph

    # 用于持久化的 G0 数据
    g0_embeddings: Optional[Dict[str, List[float]]] = None
    g0_clusters: Optional[Dict[int, List[str]]] = None

    for level in range(1, max_levels):
        if len(current_nodes) < min_cluster_size * 2:
            break

        # Step 1: Embedding 编码
        print(f"  [Level {level}] 编码 {len(current_nodes)} 个节点...")
        embeddings = _encode_nodes(current_nodes)

        # Step 2: 分层聚类 — 先按节点 type 分组,再在每组内 GMM 细分
        # 这样不同类型的节点(Entity vs Metric vs Function)不会混在一起
        type_groups: Dict[str, List[str]] = defaultdict(list)
        for n in current_nodes:
            type_groups[n.type].append(n.label)

        clusters: Dict[int, List[str]] = {}
        global_cid = 0
        for tname, member_labels in sorted(type_groups.items()):
            if len(member_labels) < 2:
                # 单节点组,直接作为簇
                clusters[global_cid] = member_labels
                global_cid += 1
                continue

            # 只取该组的 embeddings
            group_embs = {l: embeddings[l] for l in member_labels if l in embeddings}
            if len(group_embs) < 2:
                clusters[global_cid] = member_labels
                global_cid += 1
                continue

            # 在该组内做 GMM 细分
            n_sub = max(1, min(len(member_labels) // 3, 5))
            sub_clusters = _cluster_by_gmm(
                group_embs,
                n_components=n_sub,  # 强制指定,不做 BIC
                min_clusters=1,
                max_clusters=n_sub,
            )
            for sub_cid, sub_members in sub_clusters.items():
                clusters[global_cid] = sub_members
                global_cid += 1

        print(f"  [Level {level}] 分层聚类: {len(clusters)} 个簇 (按 {len(type_groups)} 个类型分组)")

        # 持久化 G0 的 embedding 和簇分配
        if level == 1:
            g0_embeddings = {label: emb.tolist() if isinstance(emb, np.ndarray) else emb
                           for label, emb in embeddings.items()}
            g0_clusters = dict(clusters)

        # Step 3: LLM 生成抽象概念
        abstract_nodes: List[AbstractNode] = []
        cluster_abstract_map: Dict[int, str] = {}  # cluster_id → abstract_label

        for cid, member_labels in sorted(clusters.items()):
            member_details = []
            for label in member_labels:
                node = current_graph.node_map.get(label)
                if node:
                    member_details.append({
                        "label": node.label,
                        "type": node.type,
                        "description": node.description or "",
                    })

            if use_llm:
                concept = _generate_abstract_concept(
                    cid, member_labels, member_details, level
                )
            else:
                # 无 LLM 模式:取最常见的 type 作为概念
                types = [d["type"] for d in member_details]
                common_type = max(set(types), key=types.count)
                concept = {
                    "label": f"{common_type}簇{cid}",
                    "description": f"{len(member_labels)}个{common_type}节点的语义聚合",
                    "key_relations": [],
                }

            abstract_label = concept["label"]
            # 确保标签唯一
            if abstract_label in cluster_abstract_map.values():
                abstract_label = f"{concept['label']}_{cid}"

            cluster_abstract_map[cid] = abstract_label

            # 计算抽象节点的 embedding(成员均值)
            member_embs = [embeddings[l] for l in member_labels if l in embeddings]
            avg_emb = np.mean(member_embs, axis=0).tolist() if member_embs else None

            an = AbstractNode(
                label=abstract_label,
                description=concept["description"],
                level=level,
                member_labels=member_labels,
                member_relations=concept.get("key_relations", []),
                cluster_id=cid,
                embedding=avg_emb,
            )
            abstract_nodes.append(an)
            hg.abstract_nodes.append(an)
            hg.level_of[abstract_label] = level
            hg.children_of[abstract_label] = list(member_labels)
            for child_label in member_labels:
                hg.parent_of.setdefault(child_label, [])
                if abstract_label not in hg.parent_of[child_label]:
                    hg.parent_of[child_label].append(abstract_label)

        print(f"  [Level {level}] 生成 {len(abstract_nodes)} 个抽象概念")

        # Step 4: 跨簇关联
        cross_links = _compute_cross_cluster_links(
            clusters, current_graph, threshold=cross_cluster_threshold
        )

        abstract_edges: List[AbstractEdge] = []
        edge_set: Set[Tuple[str, str]] = set()

        for link in cross_links:
            c_from = link["from_cluster"]
            c_to = link["to_cluster"]
            a_from = cluster_abstract_map.get(c_from)
            a_to = cluster_abstract_map.get(c_to)
            if not a_from or not a_to:
                continue

            # 去重(无向)
            key = tuple(sorted([a_from, a_to]))
            if key in edge_set:
                continue
            edge_set.add(key)

            if use_llm:
                from_desc = next(
                    (n.description for n in abstract_nodes if n.label == a_from), ""
                )
                to_desc = next(
                    (n.description for n in abstract_nodes if n.label == a_to), ""
                )
                rel = _generate_cross_cluster_relations(
                    a_from, from_desc, a_to, to_desc,
                    link["link_count"], link["sample_links"],
                )
            else:
                rel = {
                    "label": "relates_to",
                    "display_label": f"存在{link['link_count']}条底层关联",
                }

            ae = AbstractEdge(
                from_label=a_from,
                to_label=a_to,
                label=rel["label"],
                display_label=rel["display_label"],
                strength=link["strength"],
                level=level,
            )
            abstract_edges.append(ae)
            hg.abstract_edges.append(ae)

        print(f"  [Level {level}] 生成 {len(abstract_edges)} 条跨簇边")

        # Step 5: 构建该层图谱
        layer_nodes = []
        for an in abstract_nodes:
            layer_nodes.append(Node(
                label=an.label,
                type="AbstractConcept",
                color="#c084fc",  # 紫色表示抽象层
                description=an.description,
                raw={"level": an.level, "cluster_id": an.cluster_id,
                     "member_labels": an.member_labels},
            ))

        layer_edges = []
        for ae in abstract_edges:
            layer_edges.append(Edge(
                from_label=ae.from_label,
                to_label=ae.to_label,
                label=ae.label,
                display_label=ae.display_label,
                raw={"level": ae.level, "strength": ae.strength},
            ))

        layer_node_map = {n.label: n for n in layer_nodes}
        hg.layers.append({
            "level": level,
            "nodes": layer_nodes,
            "edges": layer_edges,
            "node_map": layer_node_map,
            "clusters": clusters,
            "cluster_abstract_map": cluster_abstract_map,
        })

        # 准备下一层迭代:抽象节点成为新的 current_nodes
        current_nodes = layer_nodes
        current_graph = SemanticGraph(
            domain=f"{graph.domain}_L{level}",
            description=f"层级 {level} 抽象图谱",
            nodes=layer_nodes,
            edges=layer_edges,
            node_map=layer_node_map,
            adjacency={n.label: [e for e in layer_edges if e.from_label == n.label]
                       for n in layer_nodes},
            in_adjacency={n.label: [e for e in layer_edges if e.to_label == n.label]
                          for n in layer_nodes},
        )

    # 挂载 G0 embedding 和 clusters 到 hg
    hg.g0_embeddings = g0_embeddings
    hg.g0_clusters = g0_clusters

    # 缓存: pickle 序列化整个 hg(含 g0 embeddings 和 clusters)
    if cache_dir:
        import time
        t0 = time.time()
        with open(cache_dir / "hierarchical_graph.pkl", "wb") as f:
            pickle.dump(hg, f)
        # 写入图指纹
        fp_parts = [
            str(len(graph.nodes)),
            str(len(graph.edges)),
            graph.domain or "",
            hashlib.md5(",".join(sorted(n.label for n in graph.nodes)).encode()).hexdigest(),
        ]
        (cache_dir / "graph_fingerprint.txt").write_text(":".join(fp_parts), encoding="utf-8")
        elapsed = time.time() - t0
        print(f"  💾 [Cache] 分层图谱已保存, 耗时 {elapsed:.1f}s")

    return hg


def _save_hierarchical_to_cache(hg: HierarchicalGraph, cache_path: Path):
    """序列化分层图谱到 JSON 缓存."""
    data = {
        "num_levels": hg.num_levels,
        "abstract_nodes": [
            {
                "label": an.label,
                "description": an.description,
                "level": an.level,
                "member_labels": an.member_labels,
                "member_relations": an.member_relations,
                "cluster_id": an.cluster_id,
                "embedding": an.embedding,
            }
            for an in hg.abstract_nodes
        ],
        "abstract_edges": [
            {
                "from_label": ae.from_label,
                "to_label": ae.to_label,
                "label": ae.label,
                "display_label": ae.display_label,
                "strength": ae.strength,
                "level": ae.level,
            }
            for ae in hg.abstract_edges
        ],
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_hierarchical_from_cache(cache_path: Path, g0_graph: SemanticGraph) -> HierarchicalGraph:
    """从缓存加载分层图谱."""
    data = json.loads(cache_path.read_text(encoding="utf-8"))

    hg = HierarchicalGraph()

    # G0
    hg.layers.append({
        "level": 0,
        "nodes": g0_graph.nodes,
        "edges": g0_graph.edges,
        "node_map": g0_graph.node_map,
    })

    # 抽象节点
    for an_data in data.get("abstract_nodes", []):
        an = AbstractNode(
            label=an_data["label"],
            description=an_data["description"],
            level=an_data["level"],
            member_labels=an_data["member_labels"],
            member_relations=an_data.get("member_relations", []),
            cluster_id=an_data["cluster_id"],
            embedding=an_data.get("embedding"),
        )
        hg.abstract_nodes.append(an)

    # 抽象边
    for ae_data in data.get("abstract_edges", []):
        ae = AbstractEdge(
            from_label=ae_data["from_label"],
            to_label=ae_data["to_label"],
            label=ae_data["label"],
            display_label=ae_data["display_label"],
            strength=ae_data["strength"],
            level=ae_data["level"],
        )
        hg.abstract_edges.append(ae)

    # 按层级组织 layers
    max_level = max((an.level for an in hg.abstract_nodes), default=0)
    for level in range(1, max_level + 1):
        layer_nodes = [
            Node(
                label=an.label,
                type="AbstractConcept",
                color="#c084fc",
                description=an.description,
                raw={"level": an.level, "cluster_id": an.cluster_id,
                     "member_labels": an.member_labels},
            )
            for an in hg.abstract_nodes if an.level == level
        ]
        layer_edges = [
            Edge(
                from_label=ae.from_label,
                to_label=ae.to_label,
                label=ae.label,
                display_label=ae.display_label,
                raw={"level": ae.level, "strength": ae.strength},
            )
            for ae in hg.abstract_edges if ae.level == level
        ]
        layer_node_map = {n.label: n for n in layer_nodes}
        hg.layers.append({
            "level": level,
            "nodes": layer_nodes,
            "edges": layer_edges,
            "node_map": layer_node_map,
        })

    return hg


# ══════════════════════════════════════════════════════════════
# 分层检索:在多层图谱上匹配实体并查找路径
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# 分层检索:可导航索引、LCA 和路径搜索
# ══════════════════════════════════════════════════════════════

def _ensure_navigation_index(hg: HierarchicalGraph, g0_graph: SemanticGraph) -> None:
    """确保分层图谱具备 child/parent/level 导航索引。

    旧缓存中可能没有这些字段，因此在检索入口统一补齐。
    """
    if not getattr(hg, "level_of", None):
        hg.level_of = {}
    if not getattr(hg, "parent_of", None):
        hg.parent_of = {}
    if not getattr(hg, "children_of", None):
        hg.children_of = {}

    for n in g0_graph.nodes:
        hg.level_of.setdefault(n.label, 0)
        hg.parent_of.setdefault(n.label, [])

    for an in hg.abstract_nodes:
        hg.level_of[an.label] = an.level
        hg.children_of[an.label] = list(an.member_labels)
        hg.parent_of.setdefault(an.label, [])
        for child in an.member_labels:
            hg.parent_of.setdefault(child, [])
            if an.label not in hg.parent_of[child]:
                hg.parent_of[child].append(an.label)


def _ancestor_distances(label: str, hg: HierarchicalGraph) -> Dict[str, int]:
    """返回 label 到所有祖先的最短上行距离，包含自身距离 0。"""
    distances = {label: 0}
    queue = [(label, 0)]
    while queue:
        current, dist = queue.pop(0)
        for parent in hg.parent_of.get(current, []):
            if parent not in distances or dist + 1 < distances[parent]:
                distances[parent] = dist + 1
                queue.append((parent, dist + 1))
    return distances


def _find_lca_for_entities(entities: List[str], hg: HierarchicalGraph) -> Optional[str]:
    """寻找一组实体的最低公共祖先。"""
    if not entities:
        return None
    ancestor_maps = [_ancestor_distances(e, hg) for e in entities]
    common = set(ancestor_maps[0])
    for amap in ancestor_maps[1:]:
        common &= set(amap)
    if not common:
        return None
    return min(
        common,
        key=lambda a: (-hg.level_of.get(a, 0), sum(amap.get(a, 10**6) for amap in ancestor_maps), a),
    )


def _find_group_lcas(entities: List[str], hg: HierarchicalGraph) -> List[dict]:
    """生成全局 LCA 与 pair LCA，用于跨社区结构化检索。"""
    groups: List[dict] = []
    global_lca = _find_lca_for_entities(entities, hg)
    if global_lca and global_lca not in entities:
        groups.append({"lca": global_lca, "entities": list(entities), "scope": "global"})

    seen = set()
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            pair = [entities[i], entities[j]]
            lca = _find_lca_for_entities(pair, hg)
            if not lca or lca in pair:
                continue
            key = (lca, tuple(sorted(pair)))
            if key in seen:
                continue
            seen.add(key)
            groups.append({"lca": lca, "entities": pair, "scope": "pair"})
    return groups


def _semantic_distance(a: str, b: str, hg: HierarchicalGraph) -> int:
    """基于祖先链的语义距离，用于 LCA-guided pair 排序。"""
    aa = _ancestor_distances(a, hg)
    bb = _ancestor_distances(b, hg)
    common = set(aa) & set(bb)
    if not common:
        return 10**6
    return min(aa[x] + bb[x] for x in common)


def _match_in_hierarchical(
    entities: List[str],
    hg: HierarchicalGraph,
    g0_node_map: Dict[str, Node],
) -> Dict[int, List[str]]:
    """在多层图谱中匹配实体。

    使用 parent_of 逐级上溯，而不是只看一层 member_labels，保证 G2/G3 等高层也可被检索利用。
    """
    matched: Dict[int, List[str]] = defaultdict(list)
    seen: Dict[int, Set[str]] = defaultdict(set)

    for e in entities:
        if e in g0_node_map:
            matched[0].append(e)
            seen[0].add(e)
        for anc in _ancestor_distances(e, hg):
            level = hg.level_of.get(anc, 0)
            if level <= 0 or anc in seen[level]:
                continue
            matched[level].append(anc)
            seen[level].add(anc)

    return dict(matched)


def _find_paths_in_layer(
    from_label: str,
    to_label: str,
    layer: dict,
    max_hops: int = 3,
) -> List[dict]:
    """在单层图谱中 BFS 找路径."""
    node_map = layer["node_map"]
    edges_list = layer["edges"]

    # 构建邻接表
    adjacency: Dict[str, List[Edge]] = defaultdict(list)
    in_adjacency: Dict[str, List[Edge]] = defaultdict(list)
    for e in edges_list:
        adjacency[e.from_label].append(e)
        in_adjacency[e.to_label].append(e)

    if from_label not in node_map or to_label not in node_map:
        return []
    if from_label == to_label:
        return [{"nodes": [from_label], "edges": []}]

    paths = []
    queue = [(from_label, [from_label], [])]
    visited_paths = set()

    while queue and len(paths) < 3:
        current, node_path, edge_path = queue.pop(0)

        if len(node_path) > max_hops + 1:
            continue

        if current == to_label:
            paths.append({
                "nodes": list(node_path),
                "edges": [
                    {"from": e.from_label, "to": e.to_label,
                     "label": e.label, "display_label": e.display_label}
                    for e in edge_path
                ],
            })
            continue

        next_steps = []
        for edge in adjacency.get(current, []):
            next_steps.append((edge.to_label, edge))
        for edge in in_adjacency.get(current, []):
            next_steps.append((edge.from_label, edge))

        for next_node, edge in next_steps:
            if next_node in node_path:
                continue
            new_path = tuple(node_path + [next_node])
            if new_path in visited_paths:
                continue
            visited_paths.add(new_path)
            queue.append((next_node, list(node_path) + [next_node],
                          edge_path + [edge]))

    return paths


# ══════════════════════════════════════════════════════════════
# 路径去重 & 降噪
# ══════════════════════════════════════════════════════════════

# 噪音中间节点类型：这些类型的节点在路径中间时，通常不提供有价值的语义信息
_NOISY_INTERMEDIATE_TYPES = {"Function", "Event", "MetricCategory", "Concept/Filter"}

_LOW_PRIORITY_BRIDGE_LABELS = {"金额指标", "数量指标"}


def _is_metric_node(label: str, node_map: Dict[str, Any]) -> bool:
    node = node_map.get(label) if node_map else None
    return bool(node and getattr(node, "type", None) == "Metric")


def _is_low_priority_metric_shortcut(
    prev_node: str,
    bridge_node: str,
    next_node: str,
    node_map: Dict[str, Any],
) -> bool:
    return (
        bridge_node in _LOW_PRIORITY_BRIDGE_LABELS
        and _is_metric_node(prev_node, node_map)
        and _is_metric_node(next_node, node_map)
    )


def _path_has_low_priority_metric_shortcut(nodes: List[str], node_map: Dict[str, Any]) -> bool:
    return any(
        _is_low_priority_metric_shortcut(nodes[i - 1], nodes[i], nodes[i + 1], node_map)
        for i in range(1, len(nodes) - 1)
    )

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


def deduplicate_paths(
    paths: List[dict],
    max_per_pair: int = 2,
    node_map: Optional[Dict[str, Any]] = None,
    entities: Optional[List[str]] = None,
) -> List[dict]:
    """多策略路径去重与降噪。

    策略流水线：
    1. 精确结构去重            — 相同节点序列只保留一条
    2. 每对实体 Top-N 排序      — 按质量评分，每对保留 top N
    3. 边集去重                — 两端相同 + 边集合相同 → 只保留最短
    4. 子路径消除              — 去掉被更长路径完全包含的
    5. 噪音过滤                — 去掉中间节点全是无意义类型的路径
    6. 语义相关性过滤           — 只保留路径两端点都在问题实体中的路径

    Args:
        paths: 原始路径列表, 每条格式:
               {"between": [a, b], "nodes": [...], "edges": [...], "level": int}
        max_per_pair: 每对实体最多保留几条路径
        node_map: 节点映射表，用于噪音判定(node.type)
        entities: 问题实体列表，用于语义相关性过滤

    Returns:
        去重后的路径列表
    """
    if not paths:
        return []

    entity_set = set(entities or [])

    if node_map:
        paths = [
            p for p in paths
            if not _path_has_low_priority_metric_shortcut(p.get('nodes', []), node_map)
        ]
        if not paths:
            return []

    # ── 策略1: 精确结构去重 ──
    seen_signatures: Set[str] = set()
    unique_paths: List[dict] = []
    for p in paths:
        between = tuple(sorted(p.get("between", [])))
        nodes_tuple = tuple(p.get("nodes", []))
        sig = str((p.get("level", 0), between, nodes_tuple))
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_paths.append(p)
    dup_removed = len(paths) - len(unique_paths)

    # ── 辅助: 路径质量评分 ──
    def _score_path(p: dict) -> tuple:
        """评分: (跳数, 噪音节点数, 路径长度)。越小越好。
        
        泛化规则：中间节点如果在 entities 中，不算噪音。
        Event 若通过 measured_by/measured_as 连接 Metric，是必要的度量桥接，也不计噪音。
        """
        nodes = p.get("nodes", [])
        edges = p.get("edges", [])
        hops = len(nodes) - 1
        if node_map:
            noise_count = 0
            for i, n in enumerate(nodes[1:-1]):
                if n in entity_set:
                    continue
                node_obj = node_map.get(n)
                if not (node_obj and hasattr(node_obj, 'type')):
                    continue
                if node_obj.type not in _NOISY_INTERMEDIATE_TYPES:
                    continue
                if node_obj.type == "Event" and _is_measurement_bridge(edges, i):
                    continue
                noise_count += 1
        else:
            noise_count = sum(
                1 for n in nodes[1:-1]
                if n not in entity_set and n in _NOISY_INTERMEDIATE_TYPES
            )
        return (hops, noise_count, len(nodes))

    # ── 策略2: 每对实体 Top-N ──
    pair_groups: Dict[Tuple[str, str], List[dict]] = {}
    for p in unique_paths:
        between = p.get("between", [])
        if len(between) == 2:
            key = tuple(sorted(between))
        else:
            key = ("_single_", str(id(p)))
        pair_groups.setdefault(key, []).append(p)

    for key in pair_groups:
        pair_groups[key].sort(key=_score_path)

    top_paths: List[dict] = []
    for group in pair_groups.values():
        # 方案A: 如果该实体对存在0噪音路径，过滤掉同对有噪音的路径
        if any(_score_path(p)[1] == 0 for p in group):
            group = [p for p in group if _score_path(p)[1] == 0]
        kept = group[:max_per_pair]
        # 额外检查：如果第3条路径的边集合与前两条完全不同，也保留
        if len(group) > max_per_pair:
            existing_edge_sets = {
                frozenset((e["from"], e["to"], e.get("label", ""))
                          for e in p.get("edges", []))
                for p in kept
            }
            for p in group[max_per_pair:max_per_pair + 1]:
                p_edge_set = frozenset(
                    (e["from"], e["to"], e.get("label", ""))
                    for e in p.get("edges", [])
                )
                if p_edge_set and p_edge_set not in existing_edge_sets:
                    kept.append(p)
        top_paths.extend(kept)
    pair_dedup_removed = len(unique_paths) - len(top_paths)

    # ── 策略3: 边集去重 ──
    # 同一对端点，边集合相同 → 语义等价，只保留跳数最短的
    edge_grouped: Dict[Tuple, List[dict]] = {}
    for p in top_paths:
        between = (
            tuple(sorted(p.get("between", [])))
            if len(p.get("between", [])) == 2
            else ("_", "_")
        )
        edge_set = frozenset(
            (e["from"], e["to"], e.get("label", ""))
            for e in p.get("edges", [])
        )
        key = (between, edge_set)
        edge_grouped.setdefault(key, []).append(p)

    edge_deduped: List[dict] = []
    for group in edge_grouped.values():
        if len(group) == 1:
            edge_deduped.append(group[0])
        else:
            best = min(group, key=_score_path)
            edge_deduped.append(best)
    edge_dedup_removed = len(top_paths) - len(edge_deduped)

    # ── 策略4: 子路径消除 ──
    # 如果路径A的节点序列是路径B的子序列(两端点相同) → 删除更长的
    sorted_indices = sorted(
        range(len(edge_deduped)),
        key=lambda i: -len(edge_deduped[i].get("nodes", [])),
    )

    eliminated: Set[int] = set()
    for i in sorted_indices:
        if i in eliminated:
            continue
        pi = edge_deduped[i]
        ni = tuple(pi.get("nodes", []))
        for j in sorted_indices:
            if i == j or j in eliminated:
                continue
            pj = edge_deduped[j]
            nj = tuple(pj.get("nodes", []))
            if len(ni) >= len(nj):
                for k in range(len(ni) - len(nj) + 1):
                    if ni[k:k + len(nj)] == nj:
                        eliminated.add(j)
                        break

    subpath_filtered: List[dict] = []
    for idx in range(len(edge_deduped)):
        if idx not in eliminated:
            subpath_filtered.append(edge_deduped[idx])
    subpath_removed = len(edge_deduped) - len(subpath_filtered)

    # ── 策略5: 噪音过滤 ──
    # 去掉中间节点全是无意义类型(且不在entities中)的路径。
    # 例外: 如果路径两端点都在 entity_set 中(用户明确要的)，保留即使中间节点全是噪音。
    clean_paths: List[dict] = []
    for p in subpath_filtered:
        nodes = p.get("nodes", [])
        if len(nodes) <= 2:
            clean_paths.append(p)
            continue
        # 两端点都是用户要的 → 不按噪音过滤
        between = p.get("between", [])
        if len(between) == 2 and between[0] in entity_set and between[1] in entity_set:
            clean_paths.append(p)
            continue
        intermediates = nodes[1:-1]
        if node_map:
            all_noise = all(
                n not in entity_set
                and hasattr(node_map.get(n), 'type')
                and node_map[n].type in _NOISY_INTERMEDIATE_TYPES
                for n in intermediates
            )
        else:
            all_noise = all(
                n not in entity_set and n in _NOISY_INTERMEDIATE_TYPES
                for n in intermediates
            )
        if not all_noise:
            clean_paths.append(p)
    noise_removed = len(subpath_filtered) - len(clean_paths)

    # ── 策略6: 语义相关性过滤 ──
    if entity_set:
        relevant_paths: List[dict] = []
        for p in clean_paths:
            between = p.get("between", [])
            if len(between) == 2:
                if between[0] in entity_set and between[1] in entity_set:
                    relevant_paths.append(p)
            else:
                # 单端点路径也保留(如抽象层路径)
                relevant_paths.append(p)
        semantic_removed = len(clean_paths) - len(relevant_paths)
        clean_paths = relevant_paths
    else:
        semantic_removed = 0

    # ── 统计输出 ──
    total_removed = len(paths) - len(clean_paths)
    if total_removed > 0:
        print(
            f"  [PathDedup] 原始 {len(paths)} →"
            f" 结构去重-{dup_removed} → 配对TopN-{pair_dedup_removed} →"
            f" 边集去重-{edge_dedup_removed} → 子路径-{subpath_removed} →"
            f" 噪音过滤-{noise_removed} → 语义过滤-{semantic_removed}"
            f" = {len(clean_paths)} 条"
            f" (减少 {total_removed}/{len(paths)}"
            f" = {100 * total_removed // len(paths)}%)"
        )

    return clean_paths


# ══════════════════════════════════════════════════════════════
# 构建分层子图(对外接口)
# ══════════════════════════════════════════════════════════════

def build_hierarchical_subgraph(
    entities: List[str],
    g0_graph: SemanticGraph,
    hg: HierarchicalGraph,
    *,
    max_hops: int = 5,
    expand_to_g0: bool = True,
) -> dict:
    """LCA-guided v2 子图检索。

    核心策略：
    1. 先补齐可导航的 parent/children/level 层级索引；
    2. 对查询实体计算全局 LCA 和 pair LCA，形成结构化语义骨架；
    3. 根据 LCA 骨架选择少量高价值 G0 实体对；
    4. 只在基础知识图谱 G0 的非 SQL 语义边上搜索路径；
    5. SQL 边不参与任何路径构建，留给子图完成后的后处理合并。
    """
    node_map = g0_graph.node_map
    valid_entities = [e for e in entities if e in node_map]
    invalid_entities = [e for e in entities if e not in node_map]

    if not valid_entities:
        return {"nodes": [], "edges": [], "paths": [], "isolated": invalid_entities}

    _ensure_navigation_index(hg, g0_graph)
    matched = _match_in_hierarchical(valid_entities, hg, node_map)
    lca_groups = _find_group_lcas(valid_entities, hg)

    from subgraph_builder import _classify_entities as _classify_v1
    from subgraph_builder import _bfs_shortest_paths as _bfs_v1

    metric_nodes, constraint_nodes, other_nodes = _classify_v1(valid_entities, node_map)

    def _add_pair(pairs: List[Tuple[str, str]], a: str, b: str) -> None:
        if a == b:
            return
        if a not in node_map or b not in node_map:
            return
        key = tuple(sorted([a, b]))
        if key not in {tuple(sorted(x)) for x in pairs}:
            pairs.append((a, b))

    candidate_pairs: List[Tuple[str, str]] = []

    # 查询目标优先：指标间、约束到指标。
    for i in range(len(metric_nodes)):
        for j in range(i + 1, len(metric_nodes)):
            _add_pair(candidate_pairs, metric_nodes[i], metric_nodes[j])

    for c in constraint_nodes + other_nodes:
        for m in metric_nodes:
            _add_pair(candidate_pairs, c, m)

    # 若没有指标，退化为 LCA-guided 约束间结构搜索。
    if not metric_nodes:
        for i in range(len(valid_entities)):
            for j in range(i + 1, len(valid_entities)):
                _add_pair(candidate_pairs, valid_entities[i], valid_entities[j])

    # LCA 补充：对同一 LCA 语义社区内的实体保留结构化连接。
    for group in lca_groups:
        group_entities = [e for e in group.get("entities", []) if e in node_map]
        group_metrics = [e for e in group_entities if e in metric_nodes]
        group_context = [e for e in group_entities if e not in group_metrics]
        if group_metrics:
            for c in group_context:
                for m in group_metrics:
                    _add_pair(candidate_pairs, c, m)
        else:
            for i in range(len(group_entities)):
                for j in range(i + 1, len(group_entities)):
                    _add_pair(candidate_pairs, group_entities[i], group_entities[j])

    candidate_pairs.sort(key=lambda ab: (_semantic_distance(ab[0], ab[1], hg), ab[0], ab[1]))

    # 控制扁平搜索退化：LCA 负责排序与剪枝，只展开最有结构价值的 pair。
    max_pairs = max(8, len(valid_entities) * 4)
    pairs_to_search = candidate_pairs[:max_pairs]

    all_paths: List[dict] = []
    semantic_index = _build_simple_index(g0_graph)

    for a, b in pairs_to_search:
        paths = _bfs_v1(
            semantic_index,
            a,
            b,
            max_hops=max_hops,
            max_paths=2,
            include_sql_edges=False,
        )
        for pth in paths:
            all_paths.append({
                "between": [a, b],
                "nodes": pth["nodes"],
                "edges": pth["edges"],
                "level": 0,
                "guided_by_lca": _find_lca_for_entities([a, b], hg),
            })

    deduped_paths = deduplicate_paths(
        all_paths,
        max_per_pair=2,
        node_map=node_map,
        entities=valid_entities,
    )

    collected_nodes: Set[str] = set()
    collected_edges: Dict[Tuple[str, str, str], Edge] = {}

    for pth in deduped_paths:
        for nl in pth.get("nodes", []):
            if nl in node_map:
                collected_nodes.add(nl)
        for ei in pth.get("edges", []):
            key = (ei["from"], ei["to"], ei.get("label", ""))
            if key in collected_edges:
                continue
            for edge in g0_graph.edges:
                if edge.sql_edge:
                    continue
                if (edge.from_label == ei["from"]
                        and edge.to_label == ei["to"]
                        and edge.label == ei.get("label")):
                    collected_edges[key] = edge
                    break

    nodes_in_paths: Set[str] = set()
    for pth in deduped_paths:
        nodes_in_paths.update(n for n in pth.get("nodes", []) if n in node_map)
    isolated = [e for e in valid_entities if e not in nodes_in_paths]

    return {
        "nodes": [node_map[n] for n in collected_nodes if n in node_map],
        "edges": list(collected_edges.values()),
        "paths": deduped_paths,
        "isolated": isolated + invalid_entities,
        "hierarchy": {
            "strategy": "lca_guided_v2_no_sql_path",
            "num_levels": hg.num_levels,
            "matched_levels": {str(k): v for k, v in matched.items()},
            "lca_groups": lca_groups,
            "searched_pairs": [list(p) for p in pairs_to_search],
            "sql_policy": "SQL edges are excluded during path construction and should be merged after subgraph retrieval.",
            "abstract_nodes": [
                {"label": an.label, "description": an.description,
                 "level": an.level, "member_count": len(an.member_labels)}
                for an in hg.abstract_nodes
            ],
            "path_dedup_stats": {
                "original_count": len(all_paths),
                "deduped_count": len(deduped_paths),
                "reduction_ratio": (
                    f"{100 * (len(all_paths) - len(deduped_paths)) // len(all_paths)}%"
                    if all_paths else "0%"
                ),
            },
        },
    }

def _build_simple_index(g0_graph: SemanticGraph):
    """为 G0 图构建简化索引(供 BFS 使用)."""
    from index_builder import GraphIndex
    idx = GraphIndex(graph=g0_graph)
    return idx


# ══════════════════════════════════════════════════════════════
# Embedding 和簇分配的持久化
# ══════════════════════════════════════════════════════════════

def _save_embeddings_cache(
    embeddings: Dict[str, List[float]],
    clusters: Dict[int, List[str]],
    cache_path: Path,
):
    """Persist G0 node embeddings and GMM cluster assignments to JSON file."""
    # 将 cluster key 转为 int(JSON 不支持 int key)
    clusters_str = {str(k): v for k, v in clusters.items()}
    data = {
        "embeddings": embeddings,
        "clusters": clusters_str,
        "num_nodes": len(embeddings),
        "num_clusters": len(clusters),
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"  [Cache] embedding + clusters 已保存: {cache_path}")


def _load_embeddings_cache(cache_path: Path) -> Optional[Tuple[Dict[str, List[float]], Dict[int, List[str]]]]:
    """Load embeddings and cluster assignments from cache. Returns (embeddings, clusters) or None."""
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        embeddings = data.get("embeddings", {})
        clusters_str = data.get("clusters", {})
        clusters = {int(k): v for k, v in clusters_str.items()}
        return embeddings, clusters
    except Exception:
        return None
