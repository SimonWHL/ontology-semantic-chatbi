"""Embedding 持久化模块。

预计算图谱中所有节点和别名的语义向量并持久化到磁盘。
运行时直接反序列化，避免每次启动都重新编码。
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional


def _compute_graph_fingerprint(graph_path: Path) -> str:
    """计算图谱文件 MD5 指纹，用于缓存失效自动判定。"""
    if not graph_path.exists():
        return ""
    with open(graph_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def build_cache(
    graph_path: Path,
    cache_path: Path,
    model_name: str = "BAAI/bge-base-zh-v1.5",
) -> dict:
    """离线构建 embedding 缓存并保存到 .pkl 文件。

    步骤：
    1. 加载图谱，提取所有节点的文本描述
    2. 加载 SentenceTransformer 模型
    3. 批量编码所有节点 → node_embeddings dict
    4. 构建别名列表并批量编码 → alias_embeddings dict
    5. 附上图谱指纹 + 模型名作为校验元数据
    6. pickle 序列化写入磁盘
    """
    from sentence_transformers import SentenceTransformer
    from graph.loader import load_graph
    from extraction.entity_extractor import ALIAS_MAP, RULE_SAFE_TYPES

    print(f"📂 加载图谱: {graph_path}")
    graph = load_graph(graph_path, include_sql_edges=True)
    node_map = graph.node_map

    print(f"🧠 加载模型: {model_name}")
    model = SentenceTransformer(model_name)

    # ── 节点向量 ──
    print(f"📊 编码节点向量 ({len(graph.nodes)} 个节点)...")
    node_texts = []
    node_labels = []
    for node in graph.nodes:
        parts = [node.label]
        if node.description:
            parts.append(node.description)
        if node.synonyms:
            parts.extend(node.synonyms)
        if node.type == "Attribute":
            parts.append(f"按{node.label}维度筛选")
            parts.append(f"{node.label}名称")
        node_texts.append(" | ".join(parts))
        node_labels.append(node.label)

    node_vecs = model.encode(
        node_texts, normalize_embeddings=True, show_progress_bar=True
    )
    node_embeddings: Dict[str, List[float]] = {
        label: vec.tolist() for label, vec in zip(node_labels, node_vecs)
    }

    # ── 构建别名列表（与 EntityExtractor._ensure_alias_embeddings 逻辑一致）──
    safe_labels: set = set()
    for label, node in node_map.items():
        if node.type in RULE_SAFE_TYPES or node.type == "Entity":
            safe_labels.add(label)

    alias_to_label: Dict[str, str] = {}
    for label in safe_labels:
        node = node_map[label]
        alias_to_label[label] = label
        for syn in node.synonyms:
            if syn not in alias_to_label:
                alias_to_label[syn] = label
    for alias, target in ALIAS_MAP.items():
        if target in safe_labels:
            alias_to_label[alias] = target
    for label, node in node_map.items():
        if node.type in ("Function", "MetricCategory"):
            alias_to_label[label] = label
            for syn in node.synonyms:
                if syn not in alias_to_label:
                    alias_to_label[syn] = label
    for alias, target in ALIAS_MAP.items():
        if target in node_map and node_map[target].type in ("Function", "MetricCategory"):
            if alias not in alias_to_label:
                alias_to_label[alias] = target

    # ── 别名向量 ──
    print(f"📊 编码别名向量 ({len(alias_to_label)} 个别名)...")
    alias_texts = []
    alias_keys = []
    for alias, label in alias_to_label.items():
        node = node_map.get(label)
        parts = [alias]
        if node:
            parts.append(node.label)
            if node.description:
                parts.append(node.description)
            if node.type:
                parts.append(f"类型:{node.type}")
        alias_texts.append(" | ".join(parts))
        alias_keys.append((alias, label))

    alias_vecs = model.encode(
        alias_texts, normalize_embeddings=True, show_progress_bar=True
    )
    alias_embeddings = {
        key: vec.tolist() for key, vec in zip(alias_keys, alias_vecs)
    }
    alias_id_to_label = {key: label for key, label in alias_keys}

    # ── 组装 ──
    cache = {
        "model_name": model_name,
        "graph_fingerprint": _compute_graph_fingerprint(graph_path),
        "graph_path": str(graph_path.resolve()),
        "node_embeddings": node_embeddings,
        "alias_embeddings": alias_embeddings,
        "alias_id_to_label": alias_id_to_label,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"\n💾 缓存已保存: {cache_path} ({size_mb:.1f} MB)")
    print(f"   节点向量: {len(node_embeddings)} 个")
    print(f"   别名向量: {len(alias_embeddings)} 个")
    return cache


def load_cache(
    cache_path: Path,
    graph_path: Path,
    model_name: str = "BAAI/bge-base-zh-v1.5",
) -> Optional[dict]:
    """从 .pkl 文件加载 embedding 缓存，自动校验有效性。

    失效条件（任一触发则返回 None）：
    - 缓存文件不存在
    - 模型名变更
    - 图谱内容变更（MD5 指纹不一致）
    """
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except Exception as e:
        print(f"[CACHE] 缓存文件损坏: {e}")
        return None

    if cache.get("model_name") != model_name:
        print(f"[CACHE] 模型变更 ({cache.get('model_name')} → {model_name})，需重建")
        return None

    current_fp = _compute_graph_fingerprint(graph_path)
    if cache.get("graph_fingerprint") != current_fp:
        print(f"[CACHE] 图谱已变更，需重建缓存")
        return None

    return cache
