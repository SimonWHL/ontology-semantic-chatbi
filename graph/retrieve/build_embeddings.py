#!/usr/bin/env python3
"""一次性构建 embedding 缓存。

用法:
    python3 build_embeddings.py            # 首次构建
    python3 build_embeddings.py --force    # 强制重建（覆盖已有缓存）
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from embedding_store import build_cache

GRAPH_PATH = BASE_DIR / "../data/商机.json"
CACHE_PATH = BASE_DIR / "embeddings.pkl"


def main():
    if CACHE_PATH.exists() and "--force" not in sys.argv:
        import pickle
        with open(CACHE_PATH, "rb") as f:
            meta = pickle.load(f)
        print(f"⚠ 缓存已存在: {CACHE_PATH}")
        print(f"   模型: {meta.get('model_name')}")
        print(f"   节点向量: {len(meta.get('node_embeddings', {}))} 个")
        print(f"   别名向量: {len(meta.get('alias_embeddings', {}))} 个")
        print(f"   使用 --force 强制重建")
        return

    build_cache(GRAPH_PATH, CACHE_PATH)


if __name__ == "__main__":
    main()
