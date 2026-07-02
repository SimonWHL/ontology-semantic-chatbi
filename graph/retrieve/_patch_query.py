import sys

path = r'd:\cursor\git-model\ontology-semantic-chatbi\graph\retrieve\main.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find "def query(" line
start_idx = None
for i, line in enumerate(lines):
    if 'def query(' in line and 'question' in line:
        start_idx = i
        break

if start_idx is None:
    print("ERROR: could not find query()")
    sys.exit(1)

# Find end: next top-level def
end_idx = None
for i in range(start_idx + 1, len(lines)):
    stripped = lines[i].lstrip()
    if stripped.startswith('def ') and not lines[i].startswith(' ' * 4 + ' '):
        end_idx = i
        break

if end_idx is None:
    # Try finding 'def main():'
    for i in range(start_idx + 1, len(lines)):
        if 'def main():' in lines[i]:
            end_idx = i
            break

if end_idx is None:
    print("ERROR: could not find end of query()")
    sys.exit(1)

print(f"Found query() at lines {start_idx+1}-{end_idx}")

new_query = '''def query(
    question: str,
    index: GraphIndex,
    extractor: EntityExtractor,
    *,
    use_embedding: bool = False,
    use_llm: bool = False,
    max_hops: int = 5,
    retriever: Optional[str] = None,
    capability: object = None,
    sql_edges: list = None,
    verbose: bool = False,
) -> dict:
    """端到端查询：问题 -> 上下文 JSON。"""
    cfg = _load_config()
    ext_cfg = cfg.get("extraction", {})

    # Phase 1: 实体抽取（直接调用 extractor.extract 统一入口）
    entities = extractor.extract(
        question, use_embedding=use_embedding, use_llm=use_llm,
        embedding_threshold=ext_cfg.get("embedding_threshold", 0.35),
        embedding_max_results=ext_cfg.get("embedding_max_results", 15),
    )

    if verbose:
        # 读取 extractor 内部中间状态做打印
        rule_entities = extractor.extract_by_rules(question)
        low_conf = getattr(extractor, '_low_confidence_rules', [])
        emb_candidates = []
        emb_dim_results = []
        llm_filtered = []
        if use_embedding:
            kw_results = extractor.extract_by_embedding_keywords(
                question,
                threshold=ext_cfg.get("embedding_threshold", 0.35),
                max_results=ext_cfg.get("embedding_max_results", 15),
            )
            rule_set = set(rule_entities)
            emb_candidates = [label for label, _ in kw_results if label not in rule_set]
            emb_dim_results = extractor.extract_by_embedding(question, top_k_attr=2, top_k_safe=3)

        print(f"\\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        print(f"[Phase 1] 实体抽取")
        print(f"  L1 规则匹配: {rule_entities if rule_entities else '(无命中)'}")
        if low_conf:
            print(f"  L1 低置信度(括号内): {low_conf} → 降级到候选池")
        if use_embedding:
            print(f"  L2a Embedding别名召回: {emb_candidates if emb_candidates else '(无命中 - 模型可能加载失败)'}")
            print(f"  L2b Embedding-label召回: {emb_dim_results if emb_dim_results else '(无命中)'}")
        else:
            print(f"  L2 Embedding: (未启用)")
        if use_llm:
            # LLM 结果从最终 entities 里反推
            non_rule = [e for e in entities if e not in set(rule_entities) and e not in set(emb_dim_results)]
            print(f"  L3 LLM精筛: {non_rule if non_rule else '(无命中 - API可能调用失败)'}")
        else:
            print(f"  L3 LLM: (未启用)")
        print(f"  -> 最终实体: {entities}")

    # Phase 2: 子图构建
    if retriever is None:
        subgraph_fn, extra_kwargs, retriever_name = _get_subgraph_builder()
    elif retriever == "v2":
        config = _load_config()
        v2_config = config.get("v2", {})
        subgraph_fn, extra_kwargs = build_subgraph_v2, {
            "use_diffusion": v2_config.get("use_diffusion", True),
            "max_levels": v2_config.get("max_levels", 2),
            "min_cluster_size": v2_config.get("min_cluster_size", 2),
            "cross_cluster_threshold": v2_config.get("cross_cluster_threshold", 0.1),
            "use_llm": v2_config.get("use_llm", True),
            "force_rebuild_hg": v2_config.get("force_rebuild_hg", False),
        }
        retriever_name = "v2"
    else:
        subgraph_fn, extra_kwargs = build_subgraph, {}
        retriever_name = "v1"

    subgraph = subgraph_fn(entities, index, max_hops=max_hops, **extra_kwargs)

    if verbose:
        print(f"\\n[Phase 2] 子图构建 (策略={retriever_name}, max_hops={max_hops})")
        sub_nodes = [n.label for n in subgraph.get("nodes", [])]
        paths = subgraph.get("paths", [])
        isolated = subgraph.get("isolated", [])
        print(f"  节点({len(sub_nodes)}): {sub_nodes}")
        print(f"  路径数: {len(paths)}")
        for i, p in enumerate(paths[:5], 1):
            chain = " -> ".join(p.get("nodes", []))
            print(f"    [{i}] {' & '.join(p.get('between',[]))}: {chain}")
        if len(paths) > 5:
            print(f"    ... 共 {len(paths)} 条")
        if isolated:
            print(f"  ! 孤立节点: {isolated}")

    # Phase 2.5: SQL 边后处理合并
    if sql_edges:
        subgraph = _merge_sql_edges(subgraph, sql_edges)

    if verbose:
        sql_count = subgraph.get("sql_edge_count", 0)
        sql_paths = subgraph.get("sql_analysis_paths", [])
        print(f"\\n[Phase 2.5] SQL边合并")
        if sql_count:
            print(f"  合并了 {sql_count} 条SQL边")
        else:
            print(f"  (无相关SQL边)")
        if sql_paths:
            print(f"  SQL分析路径({len(sql_paths)}):")
            for sp in sql_paths[:5]:
                print(f"    {sp['between'][0]} --[{sp['sql_clause']}]--> {sp['between'][1]}")

    # Phase 3: 格式化
    cap_notes = ""
    if capability is not None:
        cap_notes = capability.describe_for_entities(entities, index.graph.node_map)
    context = format_context(question, entities, subgraph, meta={"max_hops": max_hops},
                            capability_notes=cap_notes)

    if verbose:
        print(f"\\n[Phase 3] 最终输出")
        print(f"  总节点: {context['meta']['total_nodes']}")
        print(f"  总边: {context['meta']['total_edges']}")
        print(f"  总路径: {context['meta']['total_paths']}")
        print(f"  孤立: {context['meta']['isolated_count']}")
        print(f"{'='*60}\\n")

    return context


'''

lines[start_idx:end_idx] = [new_query]

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'OK - replaced query() (was lines {start_idx+1}-{end_idx})')
