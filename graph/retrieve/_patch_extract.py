import sys

path = r'd:\cursor\git-model\ontology-semantic-chatbi\graph\retrieve\entity_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_block_lines = [
    '        # Phase 1: 规则匹配\n',
    '        rule_entities = self.extract_by_rules(question)\n',
    '        low_conf_rules = getattr(self, "_low_confidence_rules", [])\n',
    '        entities.extend(rule_entities)\n',
    '        rule_set = set(rule_entities)\n',
    '\n',
    '        # Phase 2: Embedding 宽召回 + LLM 筛选\n',
    '        if use_embedding:\n',
    '            kw_results = self.extract_by_embedding_keywords(\n',
    '                question, threshold=embedding_threshold, max_results=embedding_max_results\n',
    '            )\n',
    '            emb_candidates = [label for label, _ in kw_results if label not in rule_set]\n',
    '\n',
    '            func_results = self.extract_function_terms(question, threshold=0.45, max_results=5)\n',
    '            for label, _ in func_results:\n',
    '                if label not in rule_set and label not in emb_candidates:\n',
    '                    emb_candidates.append(label)\n',
    '\n',
    '            # 低置信度 L1（括号内匹配）加入候选池交给 LLM 裁决\n',
    '            for lc in low_conf_rules:\n',
    '                if lc not in rule_set and lc not in emb_candidates:\n',
    '                    emb_candidates.append(lc)\n',
    '\n',
    '            dim_results = self.extract_by_embedding(question, top_k_attr=2, top_k_safe=3)\n',
    '            for d in dim_results:\n',
    '                node = self.node_map.get(d)\n',
    '                if node and node.type == "Attribute":\n',
    '                    if d not in rule_set and d not in entities:\n',
    '                        entities.append(d)\n',
    '                else:\n',
    '                    if d not in rule_set and d not in emb_candidates:\n',
    '                        emb_candidates.append(d)\n',
    '\n',
    '            if use_llm and emb_candidates:\n',
    '                filtered = self.extract_by_llm_filter(question, emb_candidates, rule_entities=rule_entities)\n',
    '                filtered = self._filter_concept_by_question(filtered, question)\n',
    '                filtered = self._filter_derived_metrics(filtered, question)\n',
    '                for e in filtered:\n',
    '                    if e not in rule_set:\n',
    '                        entities.append(e)\n',
    '            else:\n',
    '                for e in emb_candidates:\n',
    '                    if e not in rule_set:\n',
    '                        entities.append(e)\n',
    '\n',
    '        return entities\n',
]

# Replace lines 954-1003 (0-indexed 953:1003)
lines[953:1003] = new_block_lines

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('OK - replaced lines 954-1003')
