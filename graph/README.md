# 商机语义知识图谱检索系统

基于语义知识图谱的自然语言查询系统，将用户的业务分析问题转化为结构化图谱上下文，为 Text-to-SQL 提供精确的语义支撑。

## 环境准备

```bash
conda activate graph
cd ontology-semantic-chatbi/graph/retrieve
pip install -r ../require.txt
```

## 运行方式

```bash
conda activate graph
cd ontology-semantic-chatbi/graph/retrieve
```

### 单次查询

```bash
python main.py "嘉兴市大项目商机数量和出库金额占比"
```

### 交互模式

```bash
python main.py
```

### 批量测试（6 个预设问题 + HTML 可视化）

```bash
python test_batch.py
```

### 输出到文件

```bash
python main.py "问题" -o result.json
```

### 覆盖参数

```bash
python main.py --hops 6 "各行业的在途商机去重金额"
python main.py --retriever v2 "杭州互联网行业在途商机总金额"
```

## 输出格式

运行时按阶段打印各层抽取和构建结果：

```
============================================================
问题: 各行业的在途商机去重金额是多少
============================================================
[Phase 1] 实体抽取
  L1 规则匹配: ['在途商机', '去重商机金额']
  L2a Embedding别名召回: ['商机金额', '商机', ...]
  L2b Embedding-label召回: ['行业', '城市', ...]
  L3 LLM精筛: ['行业', '子行业']
  → 最终实体: ['在途商机', '去重商机金额', '行业', '子行业']

[Phase 2] 子图构建 (策略=v2, max_hops=5)
  节点(10): [...]
  路径数: 3
    [1] 在途商机 & 去重商机金额: 在途商机 → 预计签单 → ...
    [2] 行业 & 去重商机金额: 行业 → 子行业 → 客户 → ...

[Phase 2.5] SQL边合并
  合并了 20 条SQL边

[Phase 3] 最终输出
  总节点: 10  总边: 30  总路径: 3  孤立: 0
============================================================
```

如果某阶段未生效会有明确提示（如模型加载失败、API 调用失败）。

## 核心流程

```
自然语言问题
    |
    v
[Phase 1] 实体抽取 (entity_extractor.py)
    L1: 规则匹配 — 精确别名字符串匹配
    L2a: Embedding别名召回 — 问题向量 vs 别名向量，宽召回候选池
    L2b: Embedding-label召回 — 问题向量 vs 节点描述向量，识别维度
    L3: LLM精筛 — 从 L2a 候选中精选（DeepSeek API）
    |
    v
[Phase 2] 子图构建 (hierarchical_aggregator.py)
    仅在语义边上搜索路径，SQL 边不参与
    未连接到指标的实体自动扩大跳数重试（max_hops+3，上限10）
    |
    v
[Phase 2.5] SQL 边后处理合并
    子图完成后补充 sql_edge=True 的边
    生成 sql_analysis_paths（维度-指标无语义路径时的 SQL 分析关系）
    |
    v
[Phase 3] 上下文格式化 → JSON 输出
```

## SQL 边策略

**SQL 边不参与路径搜索，仅作为后处理补充。**

原因：SQL 边不带语义，路径搜索走 SQL 边会跳过有意义的中间节点，产生捷径。

流程：
1. 加载图谱 `include_sql_edges=False`，路径搜索只用语义边
2. 子图完成后筛选 `sql_edge=True` 且两端都在子图中的边追加
3. 对于通过 SQL 边连接但无语义路径的维度-指标对，生成 `sql_analysis_paths`

## 子图连通性保证

如果某个实体（如跨域维度节点）在正常跳数内找不到到指标的语义路径，系统会自动扩大搜索半径（+3跳，上限10跳）重试，确保子图连通。

判定标准：实体是否出现在"包含指标节点"的路径中，而不仅仅是 isolated 列表。

## 缓存机制

| 缓存文件 | 内容 | 失效条件 |
|----------|------|----------|
| `embeddings.pkl` | 节点+别名语义向量 | 图谱 MD5 变化 / 模型名变化 |
| `.hg_cache/hierarchical_graph.pkl` | 分层聚合图谱 | 节点数/边数/domain 变化 |

正常启动应看到：
```
[INIT] ✓ Embedding 缓存命中: 41 节点, 103 别名向量
⚡ [Cache] 从磁盘加载分层图谱, 耗时 0.1s
```

手动重建：
```bash
python build_embeddings.py          # 重建 embedding
rm -rf .hg_cache/ && python main.py "test"  # 重建分层图谱
```

## 项目结构

```
graph/
├── data/
│   └── 商机.json                    # 语义知识图谱
├── retrieve/
│   ├── config.yaml                  # 统一配置
│   ├── main.py                      # CLI 入口 + query() 核心接口
│   ├── loader.py                    # 图谱加载
│   ├── entity_extractor.py          # 实体抽取（L1规则/L2Embedding/L3LLM）
│   ├── embedding_store.py           # Embedding 持久化
│   ├── build_embeddings.py          # 离线构建 embedding 缓存
│   ├── index_builder.py             # 索引构建
│   ├── subgraph_builder.py          # 子图 v1（BFS）
│   ├── subgraph_retriever_v2.py     # 子图 v2（分层聚合入口）
│   ├── hierarchical_aggregator.py   # 分层聚合引擎 + 子图构建核心
│   ├── metric_capability.py         # 指标能力矩阵
│   ├── context_formatter.py         # 上下文格式化
│   ├── test_batch.py                # 批量测试（调用 main.query）
│   ├── test_interactive.py          # 交互测试 + HTML 可视化
│   ├── embeddings.pkl               # [缓存] 语义向量
│   ├── .hg_cache/                   # [缓存] 分层图谱
│   └── results/                     # 测试输出
└── require.txt                      # 依赖
```

## 配置 (`config.yaml`)

```yaml
extraction:
  use_embedding: true
  use_llm: true
  embedding_threshold: 0.35
  embedding_max_results: 15

retriever: "v2"
max_hops: 5

v2:
  use_diffusion: true
  max_levels: 2
  min_cluster_size: 2
  cross_cluster_threshold: 0.1
  use_llm: true
  force_rebuild_hg: false

deepseek:
  api_key: "your-api-key"
  base_url: "https://api.deepseek.com/chat/completions"
  model: "deepseek-v4-flash"
```
