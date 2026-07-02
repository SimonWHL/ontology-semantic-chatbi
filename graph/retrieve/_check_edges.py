import json
data = json.load(open(r'd:\cursor\git-model\ontology-semantic-chatbi\graph\data\商机.json', 'r', encoding='utf-8'))
print("=== 行业相关的语义边（非SQL）===")
for e in data['edges']:
    if e.get('sql_edge', False):
        continue
    if '行业' in e.get('from', '') or '行业' in e.get('to', ''):
        print(f"  {e['from']} --[{e.get('display_label', e['label'])}]--> {e['to']}")

print("\n=== 出库明细相关的语义边（非SQL）===")
for e in data['edges']:
    if e.get('sql_edge', False):
        continue
    if '出库明细' in e.get('from', '') or '出库明细' in e.get('to', ''):
        print(f"  {e['from']} --[{e.get('display_label', e['label'])}]--> {e['to']}")
