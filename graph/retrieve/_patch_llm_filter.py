import sys

path = r'd:\cursor\git-model\ontology-semantic-chatbi\graph\retrieve\entity_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "def extract_by_llm_filter"
start_idx = None
for i, line in enumerate(lines):
    if 'def extract_by_llm_filter(self, question: str, candidates: List[str])' in line:
        start_idx = i
        break

if start_idx is None:
    print("ERROR: could not find extract_by_llm_filter")
    sys.exit(1)

# Find end of function (next def at same indent or class-level)
end_idx = None
for i in range(start_idx + 1, len(lines)):
    line = lines[i]
    # Next method at same indent level
    if line.startswith('    def ') or line.startswith('class '):
        end_idx = i
        break

if end_idx is None:
    print("ERROR: could not find end of extract_by_llm_filter")
    sys.exit(1)

print(f"Found extract_by_llm_filter at lines {start_idx+1}-{end_idx}")

new_method = '''    def extract_by_llm_filter(self, question: str, candidates: List[str], rule_entities: List[str] = None) -> List[str]:
        """Embedding 召回 + LLM 精筛。

        LLM 从候选列表中选出问题真正需要的节点。
        同时展示 L1 规则已确认的实体，帮助 LLM 理解完整上下文。
        """
        if not candidates:
            return []

        llm_cfg = self._load_llm_config()
        api_key = llm_cfg["api_key"]
        if not api_key or api_key == "your-deepseek-api-key-here":
            print("[WARN] 请先在 config.yaml 中配置 deepseek.api_key")
            return candidates

        rule_entities = rule_entities or []

        # 构建候选节点展示（Metric 带域标签）
        candidate_display = []
        for label in candidates:
            node = self.node_map.get(label)
            if not node:
                continue
            if node.type == "Metric":
                cube = getattr(node, 'cube', '') or ''
                if 'opportunity' in cube:
                    candidate_display.append(f"{node.label}[商机域]")
                elif 'deliver' in cube or 'outbound' in cube:
                    candidate_display.append(f"{node.label}[出库域]")
                else:
                    candidate_display.append(node.label)
            elif node.type == "Attribute":
                candidate_display.append(f"{node.label}[维度]")
            else:
                candidate_display.append(node.label)

        # L1 已确认展示
        rule_display = []
        for label in rule_entities:
            node = self.node_map.get(label)
            if not node:
                continue
            if node.type == "Metric":
                cube = getattr(node, 'cube', '') or ''
                if 'opportunity' in cube:
                    rule_display.append(f"{node.label}[商机域]")
                elif 'deliver' in cube or 'outbound' in cube:
                    rule_display.append(f"{node.label}[出库域]")
                else:
                    rule_display.append(node.label)
            else:
                rule_display.append(node.label)

        rule_str = "、".join(rule_display) if rule_display else "无"
        cand_str = "、".join(candidate_display) if candidate_display else "无"

        prompt = f"""你是知识图谱实体筛选器。根据用户问题，从候选节点中选出真正需要的节点。

## 已确认节点（规则匹配，仅供参考上下文）:
{rule_str}

## 待筛选候选节点:
{cand_str}

## 筛选规则:
1. 维度[维度]: 问题提到地名→选城市；提到行业名或说"各行业"→选行业；提到产品名→选产品类别
2. 指标域区分: [商机域]和[出库域]是不同表的指标。问题说"商机金额/金额"(商机上下文)→选商机域；说"出库金额/交货额"→选出库域。不要跨域多选
3. 状态过滤器互斥: 赢单/中标/在途/已签/规上/新建/未中标 是不同状态，只选问题明确提到的那个
4. 函数: 问题提到"同比/环比/占比/排名"等才选对应函数节点
5. 实体: 只在需要基础实体时选「商机」「出库明细」「客户」
6. 注意括号内容: 括号内通常是解释说明，不是查询意图。如"规上（出库额10万以上）"中"出库额"是在解释规上的定义，用户并不是要查出库金额

## 重要提醒:
- 问题说"商机数和金额"→需要「商机金额[商机域]」，因为上下文是商机
- 只输出候选列表中的节点名（不带后缀标签）
- 宁缺毋滥，不确定的不选

问题: {question}
直接输出JSON数组（只含节点名，不带[域]后缀）:"""

        result = self._call_llm(prompt)
        candidate_set = set(candidates)
        return [e for e in result if e in candidate_set]

'''

lines[start_idx:end_idx] = [new_method]

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'OK - replaced extract_by_llm_filter (was lines {start_idx+1}-{end_idx})')
