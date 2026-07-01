#!/usr/bin/env python3
"""交互式测试脚本：输入问题 → 输出路径文本 + JSON + 可选 LLM 评判。

用法:
    python test_interactive.py                    # 纯检索模式（config.yaml 控制参数）
    python test_interactive.py --judge            # 启用 LLM 评判（需在 config.yaml 中配置 api_key）
    python test_interactive.py --hops 6           # 覆盖最大跳数
    python test_interactive.py -o results/        # 指定输出目录
"""

import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

from loader import load_graph
from index_builder import build_index
from entity_extractor import build_extractor
from subgraph_builder import build_subgraph as build_subgraph_v1
from subgraph_retriever_v2 import build_subgraph as build_subgraph_v2
from context_formatter import format_context


# ═══════════════════════════════════════════════════════════════
# 读取配置文件
# ═══════════════════════════════════════════════════════════════

def _load_config() -> dict:
    """加载 config.yaml，fallback 到环境变量。"""
    config = {}
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except ImportError:
            # 没有 pyyaml，简单手动解析
            with open(config_path, encoding="utf-8") as f:
                content = f.read()
            config = _parse_simple_yaml(content)
    return config


def _parse_simple_yaml(content: str) -> dict:
    """极简 YAML 解析器（仅支持单层嵌套）。"""
    result = {}
    current_section = result
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and ":" in stripped and not line.startswith("-"):
            # 顶级 key
            key = stripped.split(":")[0].strip()
            current_section = {}
            result[key] = current_section
        elif line.startswith("  ") and ":" in stripped and not line.startswith("  -"):
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            current_section[key] = val
    return result


CONFIG = _load_config()

DS_API_KEY = (
    CONFIG.get("deepseek", {}).get("api_key", "")
    or os.environ.get("DEEPSEEK_API_KEY", "")
)
DS_BASE_URL = CONFIG.get("deepseek", {}).get("base_url", "https://api.deepseek.com/chat/completions")
DS_MODEL = CONFIG.get("deepseek", {}).get("model", "deepseek-v4-flash")


def _get_subgraph_builder():
    """根据 config 返回当前激活的 build_subgraph 函数和额外参数。"""
    retriever = CONFIG.get("retriever", "v1")
    if retriever == "v2":
        v2_config = CONFIG.get("v2", {})
        return build_subgraph_v2, {
            "use_diffusion": v2_config.get("use_diffusion", True),
            "max_levels": v2_config.get("max_levels", 2),
            "min_cluster_size": v2_config.get("min_cluster_size", 2),
            "cross_cluster_threshold": v2_config.get("cross_cluster_threshold", 0.1),
            "use_llm": v2_config.get("use_llm", True),
            "force_rebuild_hg": v2_config.get("force_rebuild_hg", False),
        }, retriever
    return build_subgraph_v1, {}, retriever


def _merge_sql_edges(subgraph: dict, all_sql_edges: list) -> dict:
    """子图生成完成后再合并相关 SQL 边。

    SQL 边只用于最终展示和上下文补充，不参与 v2 路径搜索，避免连接爆炸。
    """
    sub_labels = {n.label if hasattr(n, "label") else n.get("label") for n in subgraph.get("nodes", [])}
    relevant_sql_edges = [
        e for e in all_sql_edges
        if e.from_label in sub_labels and e.to_label in sub_labels
    ]
    if not relevant_sql_edges:
        return dict(subgraph)
    return {
        **subgraph,
        "edges": subgraph.get("edges", []) + relevant_sql_edges,
        "sql_edge_count": len(relevant_sql_edges),
    }


# ═══════════════════════════════════════════════════════════════
# 1. 路径文本格式化（人类可读）
# ═══════════════════════════════════════════════════════════════

def format_paths_text(entities: list, subgraph: dict, node_map: dict) -> str:
    """将子图格式化为紧凑的路径文本。格式: 节点A -[边名]-> 节点B -[边名]-> 节点C"""
    lines = []
    lines.append("=" * 60)

    paths = subgraph.get("paths", [])
    if not paths:
        lines.append("(无路径)")
    else:
        for i, p in enumerate(paths, 1):
            between = p.get("between", [])
            nodes = p.get("nodes", [])
            edges = p.get("edges", [])

            # 构建链式文本: A -[label]-> B -[label]-> C
            parts = []
            for j, nl in enumerate(nodes):
                parts.append(nl)
                if j < len(edges):
                    display = edges[j].get("display_label", edges[j].get("label", "?"))
                    parts.append(f"-[{display}]->")
            chain = " ".join(parts)

            between_str = " & ".join(between)
            lines.append(f"[{i}] {between_str}: {chain}")

    # 孤立节点
    isolated = subgraph.get("isolated", [])
    if isolated:
        lines.append(f"\n⚠ 孤立节点(无路径): {', '.join(isolated)}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_nodes_summary(subgraph: dict) -> str:
    """节点摘要表格。"""
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])
    paths = subgraph.get("paths", [])

    lines = []
    lines.append(f"\n📊 子图统计: {len(nodes)} 节点, {len(edges)} 边, {len(paths)} 路径")

    # 按类型分组
    from collections import Counter
    type_counts = Counter(n.label if hasattr(n, 'label') else n.get('label', '?') for n in nodes)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 2. LLM 评判
# ═══════════════════════════════════════════════════════════════

JUDGE_PROMPT = """你是一个知识图谱检索质量评估专家。请根据以下信息评判检索结果的质量。

## 用户问题
{question}

## 检索到的子图上下文（路径列表）
{paths_text}

## 图谱中所有可用节点
{all_nodes}

## 评判标准
请从以下维度评分（1-5分）：
1. **实体覆盖度**：问题中提到的关键实体是否都被识别？
2. **路径相关性**：检索到的路径是否与问题直接相关？有无无关绕路？
3. **路径完整性**：路径是否足够支撑回答该问题？有无关键路径缺失？
4. **噪音控制**：是否有多余的、不相关的节点/路径混入？

请给出 JSON 格式的评判结果：
{{
    "实体覆盖度": 分数(1-5),
    "路径相关性": 分数(1-5),
    "路径完整性": 分数(1-5),
    "噪音控制": 分数(1-5),
    "综合评分": 分数(1-5),
    "优点": "简要列出优点",
    "不足": "简要列出不足或遗漏的实体/路径",
    "建议": "改进建议"
}}

直接输出 JSON："""


def judge_with_llm(question: str, paths_text: str, all_nodes: list) -> dict | None:
    """调用 DeepSeek 评判检索质量。"""
    if not DS_API_KEY or DS_API_KEY == "your-deepseek-api-key-here":
        print("\n[评判] 请先在 config.yaml 中配置 deepseek.api_key")
        return None

    nodes_str = "\n".join(f"- {n}" for n in sorted(all_nodes))
    prompt = JUDGE_PROMPT.format(
        question=question,
        paths_text=paths_text,
        all_nodes=nodes_str,
    )

    try:
        import requests
        resp = requests.post(
            DS_BASE_URL,
            headers={
                "Authorization": f"Bearer {DS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DS_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2000,
                "thinking": {"type": "disabled"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or msg.get("reasoning_content") or "").strip()

        # 去掉可能的 markdown 包裹
        import re
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        return json.loads(content)
    except Exception as e:
        print(f"\n[评判] 调用失败: {e}")
        return None


def format_judge_result(judge: dict) -> str:
    """格式化评判结果。"""
    lines = ["\n" + "=" * 60]
    lines.append("🤖 LLM 评判结果")
    lines.append("=" * 60)

    dims = ["实体覆盖度", "路径相关性", "路径完整性", "噪音控制", "综合评分"]
    for d in dims:
        score = judge.get(d, "?")
        bar = "█" * int(score) + "░" * (5 - int(score)) if isinstance(score, (int, float)) else "?"
        lines.append(f"  {d}: {bar} {score}/5")

    for key in ["优点", "不足", "建议"]:
        val = judge.get(key)
        if val:
            lines.append(f"\n  {key}: {val}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 3. 主循环
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="交互式图谱检索测试")
    parser.add_argument("--graph", default="../data/商机.json", help="图谱文件")
    parser.add_argument("--judge", action="store_true", help="启用 LLM 评判")
    parser.add_argument("--hops", type=int, default=None, help="最大跳数（默认从 config 读取）")
    parser.add_argument("-o", "--output-dir", default=None, help="JSON 输出目录（可选）")
    args = parser.parse_args()

    # ── 加载配置 ──
    cfg = _load_config()
    ext_cfg = cfg.get("extraction", {})
    use_llm = ext_cfg.get("use_llm", True)
    use_embedding = ext_cfg.get("use_embedding", True)
    max_hops = args.hops if args.hops is not None else cfg.get("max_hops", 5)

    subgraph_fn, extra_kwargs, retriever_name = _get_subgraph_builder()

    # 加载
    graph_path = Path(args.graph)
    if not graph_path.is_absolute():
        graph_path = BASE_DIR / graph_path

    print(f"📂 加载图谱: {graph_path}", file=sys.stderr)
    graph = load_graph(graph_path, include_sql_edges=False)
    graph_with_sql = load_graph(graph_path, include_sql_edges=True)
    index = build_index(graph)
    extractor = build_extractor(
        node_labels=set(index.graph.node_map.keys()),
        node_map=index.graph.node_map,
        cache_path=BASE_DIR / "embeddings.pkl",
        graph_path=graph_path,
    )
    extractor.initialize()
    print(f"   节点: {len(graph.nodes)}  语义边: {len(graph.edges)}", file=sys.stderr)
    print(f"   召回策略: {retriever_name}  LLM: {use_llm}  Embedding: {use_embedding}  评判: {args.judge}", file=sys.stderr)

    all_node_labels = sorted(index.graph.node_map.keys())

    print("\n" + "=" * 60)
    print("  知识图谱检索交互测试")
    print("  输入自然语言问题，输入 quit 退出")
    print("  结果自动保存到 results/ 目录")
    print("=" * 60)

    # 确保 results 目录存在
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _safe_filename(text: str, max_len: int = 40) -> str:
        """将问题文本转为安全的文件名（保留中文）。"""
        # 去掉不适合文件名的字符
        safe = re.sub(r'[\\/*?:"<>|\s]+', '_', text.strip())
        if len(safe) > max_len:
            safe = safe[:max_len]
        return safe.strip("_")

    while True:
        try:
            question = input("\n🔍 问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        # ── Phase 1: 实体抽取
        entities = extractor.extract(
            question,
            use_embedding=use_embedding,
            use_llm=use_llm,
            embedding_threshold=ext_cfg.get("embedding_threshold", 0.35),
            embedding_max_results=ext_cfg.get("embedding_max_results", 15),
        )
        print(f"\n📌 识别实体: {entities}")

        # ── Phase 2: 子图构建
        if retriever_name == "v2":
            subgraph = subgraph_fn(entities, index, max_hops=max_hops, sql_edges=graph_with_sql.edges, **extra_kwargs)
        else:
            subgraph = subgraph_fn(entities, index, max_hops=max_hops, **extra_kwargs)
        subgraph = _merge_sql_edges(subgraph, graph_with_sql.edges)

        # ── 输出路径 1: 紧凑路径文本 ──
        paths_text = format_paths_text(entities, subgraph, index.graph.node_map)
        print(paths_text)
        print(format_nodes_summary(subgraph))

        # ── 输出路径 2: JSON ──
        context = format_context(question, entities, subgraph, meta={"max_hops": max_hops})
        json_str = json.dumps(context, ensure_ascii=False, indent=2)
        print(f"\n📋 JSON 上下文 ({len(json_str)} 字符):")
        # 折叠路径详情，只显示概览
        context_compact = {
            "question": context["question"],
            "entities": context["entities"],
            "subgraph": {
                "nodes": [{"label": n["label"], "type": n["type"]} for n in context["subgraph"]["nodes"]],
                "edges": [{"from": e["from"], "to": e["to"], "label": e.get("display_label", e["label"])} for e in context["subgraph"]["edges"]],
            },
            "paths_count": len(context["paths"]),
            "paths": context["paths"],
            "isolated": context["isolated"],
            "meta": context["meta"],
        }
        print(json.dumps(context_compact, ensure_ascii=False, indent=2))

        # ── Phase 3: LLM 评判 ──
        judge_result = None
        if args.judge:
            print("\n⏳ 正在调用 LLM 评判...")
            judge_result = judge_with_llm(question, paths_text, all_node_labels)
            if judge_result:
                print(format_judge_result(judge_result))

        # ── 自动保存: JSON → results/ + 生成 HTML ──
        safe_name = _safe_filename(question)
        json_path = RESULTS_DIR / f"{safe_name}.json"
        html_path = RESULTS_DIR / f"{safe_name}.html"

        # 保存完整 JSON（包含评判结果）
        save_data = dict(context)
        if judge_result:
            save_data["judge"] = judge_result
        json_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 生成可视化 HTML
        _generate_html(question, context["subgraph"], html_path)

        print(f"\n💾 已保存: {json_path.name}")
        print(f"🌐 可视化: {html_path.name}")


# ═══════════════════════════════════════════════════════════════
# 4. 可视化 HTML 生成
# ═══════════════════════════════════════════════════════════════

VIS_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Microsoft YaHei', sans-serif; background: #1a1a2e; overflow: hidden; }
#container { width: 100vw; height: 100vh; position: relative; }
svg { width: 100%; height: 100%; cursor: grab; }
svg:active { cursor: grabbing; }
.title { position: absolute; top: 16px; left: 16px; color: #eee; font-size: 18px; font-weight: bold; text-shadow: 0 1px 4px rgba(0,0,0,0.5); max-width: 50%; }
.subtitle { position: absolute; top: 46px; left: 16px; color: #9aa; font-size: 12px; max-width: 480px; }
.legend { position: absolute; bottom: 16px; left: 16px; background: rgba(30,30,60,0.92); border-radius: 10px; padding: 12px 16px; color: #eee; font-size: 13px; box-shadow: 0 2px 12px rgba(0,0,0,0.4); }
.legend-item { display: flex; align-items: center; margin-bottom: 6px; }
.legend-item:last-child { margin-bottom: 0; }
.legend-dot { width: 14px; height: 14px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
.panel { position: absolute; top: 16px; right: 16px; width: 340px; max-height: calc(100vh - 32px); overflow-y: auto; background: rgba(30,30,60,0.96); border-radius: 12px; padding: 18px 20px; color: #eee; font-size: 13px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); display: none; }
.panel.show { display: block; }
.panel h2 { font-size: 18px; margin-bottom: 4px; display: flex; align-items: center; }
.panel .type-badge { font-size: 11px; padding: 2px 8px; border-radius: 8px; margin-left: 8px; color: #1a1a2e; font-weight: bold; }
.panel-close { position: absolute; top: 12px; right: 14px; cursor: pointer; color: #aaa; font-size: 18px; line-height: 1; }
.panel-close:hover { color: #fff; }
.field { margin-top: 10px; }
.field-key { color: #8ab4ff; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.field-val { color: #eee; margin-top: 2px; word-break: break-word; }
.field-val.code { font-family: Consolas, Monaco, monospace; background: rgba(0,0,0,0.35); padding: 6px 8px; border-radius: 6px; font-size: 12px; }
.field-val.warn { color: #ffb86b; }
.rel-list { margin-top: 4px; }
.rel-item { padding: 3px 0; color: #cdd; }
.rel-item .arrow { color: #888; }
.hint { position: absolute; bottom: 16px; right: 16px; color: #667; font-size: 12px; }
.node-label { pointer-events: none; }
.toggle-bar { position: absolute; top: 16px; right: 360px; display: flex; align-items: center; gap: 10px; color: #eee; font-size: 13px; z-index: 10; background: rgba(30,30,60,0.88); padding: 8px 14px; border-radius: 8px; }
.toggle-bar input { display: none; }
.toggle-bar .slider { position: relative; width: 40px; height: 22px; background: #4ec9b0; border-radius: 11px; cursor: pointer; transition: background 0.3s; flex-shrink: 0; }
.toggle-bar .slider::before { content: ''; position: absolute; width: 18px; height: 18px; background: #fff; border-radius: 50%; top: 2px; left: 2px; transition: transform 0.3s; }
.toggle-bar input:checked + .slider { background: #4ec9b0; }
.toggle-bar input:not(:checked) + .slider { background: #555; }
.toggle-bar input:not(:checked) + .slider::before { transform: translateX(18px); }
.toggle-bar .toggle-label { user-select: none; cursor: pointer; }
</style>
</head>
<body>
<div id="container">
  <svg id="graph"></svg>
  <div class="title">__TITLE__</div>
  <div class="subtitle">__DESC__</div>
  <div class="legend" id="legend"></div>
  <div class="panel" id="panel"></div>
  <div class="toggle-bar">
    <label class="toggle-label" for="sqlToggle">SQL边</label>
    <input type="checkbox" id="sqlToggle" checked>
    <label class="slider" for="sqlToggle"></label>
  </div>
  <div class="hint">滚轮缩放 · 拖拽画布平移 · 拖拽节点移动 · 点击节点查看详情</div>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const graphData = __GRAPH_JSON__;

const typeColors = {};
graphData.nodes.forEach(n => { if (n.type && !(n.type in typeColors)) typeColors[n.type] = n.color; });

const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select('#graph').attr('width', width).attr('height', height);

const legendEl = d3.select('#legend');
Object.entries(typeColors).forEach(([type, color]) => {
  legendEl.append('div').attr('class', 'legend-item')
    .html('<div class="legend-dot" style="background:' + color + '"></div>' + type);
});
legendEl.append('div').attr('class', 'legend-item')
  .html('<div class="legend-dot" style="background:transparent; border:2px dashed #4ec9b0"></div>SQL逻辑边 (WHERE/GROUP BY/HAVING)');

const nodeMap = {};
graphData.nodes.forEach((n, i) => { n.id = i; nodeMap[n.label] = i; });
const links = graphData.edges
  .map(e => ({ source: nodeMap[e.from], target: nodeMap[e.to], label: e.label, display_label: e.display_label, from: e.from, to: e.to, sql_edge: e.sql_edge, sql_clause: e.sql_clause }))
  .filter(l => l.source !== undefined && l.target !== undefined);

const outRel = {}, inRel = {};
graphData.nodes.forEach(n => { outRel[n.label] = []; inRel[n.label] = []; });
graphData.edges.forEach(e => {
  if (outRel[e.from]) outRel[e.from].push({ label: e.label, display_label: e.display_label, other: e.to });
  if (inRel[e.to]) inRel[e.to].push({ label: e.label, display_label: e.display_label, other: e.from });
});

const g = svg.append('g');
const zoom = d3.zoom().scaleExtent([0.15, 4]).on('zoom', (event) => g.attr('transform', event.transform));
svg.call(zoom);
svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(0.8));

svg.append('defs').append('marker').attr('id', 'arrowhead').attr('viewBox', '0 -5 10 10')
  .attr('refX', 26).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
  .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#888');
svg.select('defs').append('marker').attr('id', 'arrowhead-sql').attr('viewBox', '0 -5 10 10')
  .attr('refX', 26).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
  .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#4ec9b0');

const simulation = d3.forceSimulation(graphData.nodes)
  .force('link', d3.forceLink(links).id(d => d.id).distance(110).strength(0.4))
  .force('charge', d3.forceManyBody().strength(-600))
  .force('center', d3.forceCenter(0, 0))
  .force('collision', d3.forceCollide(38));

const linkGroup = g.append('g');
const linkLines = linkGroup.selectAll('line').data(links).join('line')
  .attr('class', d => d.sql_edge ? 'sql-edge' : 'semantic-edge')
  .attr('stroke', d => d.sql_edge ? '#4ec9b0' : '#555')
  .attr('stroke-width', d => d.sql_edge ? 1.8 : 1.2)
  .attr('stroke-dasharray', d => d.sql_edge ? '5,4' : 'none')
  .attr('marker-end', d => d.sql_edge ? 'url(#arrowhead-sql)' : 'url(#arrowhead)');
const linkLabels = linkGroup.selectAll('text').data(links).join('text')
  .attr('class', d => d.sql_edge ? 'sql-edge' : 'semantic-edge')
  .text(d => (d.display_label || d.label) + (d.sql_clause ? ' [' + d.sql_clause + ']' : ''))
  .attr('font-size', d => d.sql_edge ? 8 : 9)
  .attr('fill', d => d.sql_edge ? '#4ec9b0' : '#9aa')
  .attr('text-anchor', 'middle').attr('dy', -3);

const nodeGroup = g.append('g');
const nodeGs = nodeGroup.selectAll('g').data(graphData.nodes).join('g')
  .style('cursor', 'pointer')
  .call(d3.drag()
    .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

nodeGs.append('circle')
  .attr('r', 16)
  .attr('fill', d => d.color || '#888')
  .attr('stroke', '#fff').attr('stroke-width', 1.5).attr('opacity', 0.92);

nodeGs.append('text').attr('class', 'node-label').text(d => d.label)
  .attr('font-size', 11).attr('fill', '#fff').attr('text-anchor', 'middle').attr('dy', 30)
  .attr('paint-order', 'stroke').attr('stroke', '#1a1a2e').attr('stroke-width', 3);

const panel = d3.select('#panel');
const META_KEYS = ['label', 'color', 'type', 'id', 'x', 'y'];
const KEY_LABELS = {
  cube: '所属Cube', dataset: '数据集', dimension: '维度', column: '字段列',
  data_type: '数据类型', unit: '单位', synonyms: '同义词', subject_domains: '主题域',
  primary_entity: '主键实体', primary_column: '主键列', foreign_entity: '外键实体',
  filter: '过滤器', filter_type: '过滤类型', expr: '表达式', metric: '指标',
  metric_type: '指标类型', measure: '度量', agg: '聚合方式', based_on_metric: '依赖指标',
  label_alias: '语义层名称', depends_on: '依赖项', related_measure: '关联度量',
  category: '算子分类', description: '说明', warning: '注意'
};

function renderPanel(d) {
  let html = '<span class="panel-close" id="panelClose">&times;</span>';
  html += '<h2>' + d.label + '<span class="type-badge" style="background:' + (d.color || '#888') + '">' + (d.type || '') + '</span></h2>';
  Object.keys(d).forEach(k => {
    if (META_KEYS.includes(k)) return;
    let v = d[k];
    if (v === null || v === undefined || v === '') return;
    if (Array.isArray(v)) v = v.join('、');
    const isCode = (k === 'expr');
    const isWarn = (k === 'warning');
    html += '<div class="field"><div class="field-key">' + (KEY_LABELS[k] || k) + '</div>';
    html += '<div class="field-val' + (isCode ? ' code' : '') + (isWarn ? ' warn' : '') + '">' + v + '</div></div>';
  });
  const outs = outRel[d.label] || [], ins = inRel[d.label] || [];
  if (outs.length) {
    html += '<div class="field"><div class="field-key">出边关系</div><div class="rel-list">';
    outs.forEach(r => { const name = (r.display_label ? r.display_label + '/' : '') + r.label; html += '<div class="rel-item"><span class="arrow">—' + name + '→</span> ' + r.other + '</div>'; });
    html += '</div></div>';
  }
  if (ins.length) {
    html += '<div class="field"><div class="field-key">入边关系</div><div class="rel-list">';
    ins.forEach(r => { const name = (r.display_label ? r.display_label + '/' : '') + r.label; html += '<div class="rel-item">' + r.other + ' <span class="arrow">—' + name + '→</span></div>'; });
    html += '</div></div>';
  }
  panel.html(html).classed('show', true);
  document.getElementById('panelClose').onclick = () => {
    panel.classed('show', false);
    nodeGs.selectAll('circle').attr('stroke', '#fff').attr('stroke-width', 1.5);
  };
}

nodeGs.on('click', (event, d) => {
  event.stopPropagation();
  nodeGs.selectAll('circle').attr('stroke', '#fff').attr('stroke-width', 1.5);
  d3.select(event.currentTarget).select('circle').attr('stroke', '#ffd166').attr('stroke-width', 3.5);
  renderPanel(d);
});

svg.on('click', () => {
  panel.classed('show', false);
  nodeGs.selectAll('circle').attr('stroke', '#fff').attr('stroke-width', 1.5);
});

simulation.on('tick', () => {
  linkLines.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  linkLabels.attr('x', d => (d.source.x + d.target.x) / 2).attr('y', d => (d.source.y + d.target.y) / 2);
  nodeGs.attr('transform', d => 'translate(' + d.x + ',' + d.y + ')');
});

// SQL边开关
document.getElementById('sqlToggle').addEventListener('change', function() {
  const visible = this.checked;
  linkLines.style('display', d => d.sql_edge && !visible ? 'none' : null);
  linkLabels.style('display', d => d.sql_edge && !visible ? 'none' : null);
});
</script>
</body>
</html>
"""


def _generate_html(question: str, subgraph: dict, output_path: Path) -> None:
    """根据子图数据生成交互式可视化 HTML。

    数据格式: {"nodes": [...], "edges": [...]}
    其中 nodes 含 label/type/color 等，edges 含 from/to/label/display_label 等。
    """
    graph_json = json.dumps(subgraph, ensure_ascii=False)

    desc = f"实体: {len(subgraph.get('nodes', []))} 节点, {len(subgraph.get('edges', []))} 边"

    html = (
        VIS_HTML_TEMPLATE
        .replace("__GRAPH_JSON__", graph_json)
        .replace("__TITLE__", question)
        .replace("__DESC__", desc)
    )
    output_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
