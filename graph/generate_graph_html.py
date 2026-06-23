"""读取语义图谱 JSON 并生成交互式可视化 HTML。

用法：
    python generate_graph_html.py [输入json] [输出html]

不带参数时默认读取同目录下的 商机.json，输出 商机图谱可视化.html。
改了 JSON 后重新运行本脚本即可刷新 HTML。
"""
import json
import sys
from pathlib import Path

HTML_TEMPLATE = r"""<!DOCTYPE html>
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
.title { position: absolute; top: 16px; left: 16px; color: #eee; font-size: 20px; font-weight: bold; text-shadow: 0 1px 4px rgba(0,0,0,0.5); }
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
</style>
</head>
<body>
<div id="container">
  <svg id="graph"></svg>
  <div class="title">__TITLE__</div>
  <div class="subtitle">__DESC__</div>
  <div class="legend" id="legend"></div>
  <div class="panel" id="panel"></div>
  <div class="hint">滚轮缩放 · 拖拽画布平移 · 拖拽节点移动 · 点击节点查看语义信息</div>
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

const nodeMap = {};
graphData.nodes.forEach((n, i) => { n.id = i; nodeMap[n.label] = i; });
const links = graphData.edges
  .map(e => ({ source: nodeMap[e.from], target: nodeMap[e.to], label: e.label, display_label: e.display_label, custom: e.custom, from: e.from, to: e.to }))
  .filter(l => l.source !== undefined && l.target !== undefined);

// 邻接关系，供详情面板展示
const outRel = {}, inRel = {};
graphData.nodes.forEach(n => { outRel[n.label] = []; inRel[n.label] = []; });
graphData.edges.forEach(e => {
  if (outRel[e.from]) outRel[e.from].push({ label: e.label, display_label: e.display_label, sql_clause: e.sql_clause, condition_type: e.condition_type, other: e.to });
  if (inRel[e.to]) inRel[e.to].push({ label: e.label, display_label: e.display_label, sql_clause: e.sql_clause, condition_type: e.condition_type, other: e.from });
});

const g = svg.append('g');
const zoom = d3.zoom().scaleExtent([0.15, 4]).on('zoom', (event) => g.attr('transform', event.transform));
svg.call(zoom);
svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(0.6));

svg.append('defs').append('marker').attr('id', 'arrowhead').attr('viewBox', '0 -5 10 10')
  .attr('refX', 26).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
  .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#888');

const simulation = d3.forceSimulation(graphData.nodes)
  .force('link', d3.forceLink(links).id(d => d.id).distance(110).strength(0.4))
  .force('charge', d3.forceManyBody().strength(-600))
  .force('center', d3.forceCenter(0, 0))
  .force('collision', d3.forceCollide(38));

const linkGroup = g.append('g');
const linkLines = linkGroup.selectAll('line').data(links).join('line')
  .attr('stroke', '#555').attr('stroke-width', 1.2).attr('marker-end', 'url(#arrowhead)');
const linkLabels = linkGroup.selectAll('text').data(links).join('text')
  .text(d => d.display_label || d.label).attr('font-size', 9).attr('fill', '#9aa').attr('text-anchor', 'middle').attr('dy', -3);

const nodeGroup = g.append('g');
const nodeGs = nodeGroup.selectAll('g').data(graphData.nodes).join('g')
  .style('cursor', 'pointer')
  .call(d3.drag()
    .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

nodeGs.append('circle')
  .attr('r', d => d.label === '商机' ? 22 : 16)
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
    outs.forEach(r => { const name = (r.display_label ? r.display_label + '/' : '') + r.label + (r.sql_clause ? '[' + r.sql_clause + ']' : ''); html += '<div class="rel-item"><span class="arrow">—' + name + '→</span> ' + r.other + '</div>'; });
    html += '</div></div>';
  }
  if (ins.length) {
    html += '<div class="field"><div class="field-key">入边关系</div><div class="rel-list">';
    ins.forEach(r => { const name = (r.display_label ? r.display_label + '/' : '') + r.label + (r.sql_clause ? '[' + r.sql_clause + ']' : ''); html += '<div class="rel-item">' + r.other + ' <span class="arrow">—' + name + '→</span></div>'; });
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
</script>
</body>
</html>
"""


def generate(input_path: Path, output_path: Path) -> None:
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    title = data.get("domain", "语义") + " 语义图谱"
    if data.get("domain") == "opportunity":
        title = "商机语义图谱"
    desc = data.get("description", "")

    graph_json = json.dumps(data, ensure_ascii=False)

    html = (
        HTML_TEMPLATE
        .replace("__GRAPH_JSON__", graph_json)
        .replace("__TITLE__", title)
        .replace("__DESC__", desc)
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"已生成: {output_path}")
    print(f"节点数: {len(data.get('nodes', []))}  边数: {len(data.get('edges', []))}")


def main() -> None:
    base = Path(__file__).resolve().parent
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else base / "商机.json"
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "商机图谱可视化.html"

    if not input_path.exists():
        print(f"错误: 找不到输入文件 {input_path}")
        sys.exit(1)

    generate(input_path, output_path)


if __name__ == "__main__":
    main()
