# 商机语义知识图谱检索系统

基于语义知识图谱的自然语言查询系统，将用户的业务分析问题转化为结构化图谱上下文，为 Text-to-SQL 提供精确的语义支撑。

## 环境准备

```bash
# 激活 conda 环境
conda activate graph

# 进入工作目录
cd ontology-semantic-chatbi/graph/retrieve

# 首次安装依赖
pip install -r ../require.txt
```

## 运行方式

所有命令都在 `retrieve/` 目录下执行，确保已激活 `graph` 环境。

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
# 输入问题回车，输入 quit 退出
```

### 批量测试（6 个预设问题 + HTML 可视化）

```bash
python test_batch.py
# 结果保存在 results/ 目录
```

### 交互测试 + LLM 评判

```bash
python test_interactive.py --judge
```

### 预构建 Embedding 缓存

```bash
python build_embeddings.py
# 生成 embeddings.pkl，后续启动直接加载
```

### 常用参数覆盖

```bash
python main.py --hops 6 "各行业的在途商机去重金额"
python main.py --retriever v2 "杭州互联网行业在途商机总金额"
python main.py "问题" -o result.json
python main.py --md "问题"
```

> 所有参数统一在 `config.yaml` 中配置，命令行参数仅用于临时覆盖。

## 缓存机制

系统有两层持久化缓存，避免每次启动重新编码：

| 缓存文件 | 内容 | 失效条件 |
|----------|------|----------|
| `embeddings.pkl` | 节点+别名的语义向量 | 图谱文件 MD5 变化 / 模型名变化 |
| `.hg_cache/hierarchical_graph.pkl` | 分层聚合图谱 | 节点数/边数/domain 变化 |

正常启动时应看到：

```
[INIT] ✓ Embedding 缓存命中: 41 节点, 168 别名向量
  ⚡ [Cache] 从磁盘加载分层图谱, 耗时 0.1s
```

如果看到「缓存未命中」或「图指纹已变化」，说明缓存失效会自动重建（首次约 60-90 秒）。

手动重建：

```bash
# 重建 embedding
python build_embeddings.py

# 重建分层图谱（删除旧缓存即可）
rm -rf .hg_cache/
python test_batch.py
```

## 核心流程

```
自然语言问题
    |
    v
[Phase 1] 实体抽取 (entity_extractor.py)
    L1: 规则匹配（精确别名）
    L2: Embedding 语义召回（bge-base-zh-v1.5）
    L3: LLM 精筛（DeepSeek API）
    |
    v
[Phase 2] 子图构建 (subgraph_builder / subgraph_retriever_v2)
    仅在语义边上搜索路径，不走 SQL 边
    v1: BFS 最短路径
    v2: LCA-guided 分层聚合
    |
    v
[Phase 2.5] SQL 边后处理合并
    子图完成后补充 sql_edge=True 的边
    SQL 边仅用于展示，不参与路径搜索
    |
    v
[Phase 3] 上下文格式化 (context_formatter.py)
    输出 JSON（nodes/edges/paths/capability）
```

## SQL 边策略

SQL 边（`sql_edge: true`）代表维度/指标在 SQL 中的 WHERE/GROUP BY/HAVING 关系。

设计原则：**SQL 边不参与路径搜索，仅作为后处理补充。**

原因：SQL 边不带语义，路径搜索走 SQL 边会产生捷径，跳过有意义的中间节点。

流程：
1. 加载图谱 `include_sql_edges=False`，路径搜索只用语义边
2. 子图完成后筛选 `sql_edge=True` 且两端都在子图中的边
3. 追加到子图 edges 中供下游使用

## 召回策略

| 策略 | 文件 | 特点 |
|------|------|------|
| v1 | `subgraph_builder.py` | BFS 最短路径，适合简单查询 |
| v2 | `subgraph_retriever_v2.py` | LCA-guided 分层聚合，适合复杂多实体查询 |

通过 `config.yaml` 的 `retriever` 键切换。

## 项目结构

```
graph/
├── data/
│   └── 商机.json                    # 语义知识图谱
├── retrieve/
│   ├── config.yaml                  # 统一配置
│   ├── main.py                      # CLI 入口
│   ├── loader.py                    # 图谱加载
│   ├── entity_extractor.py          # 实体抽取
│   ├── embedding_store.py           # Embedding 持久化
│   ├── build_embeddings.py          # 离线构建 embedding 缓存
│   ├── index_builder.py             # 索引构建
│   ├── subgraph_builder.py          # 子图 v1（BFS）
│   ├── subgraph_retriever_v2.py     # 子图 v2（分层聚合）
│   ├── hierarchical_aggregator.py   # 分层聚合引擎
│   ├── metric_capability.py         # 指标能力矩阵
│   ├── context_formatter.py         # 上下文格式化
│   ├── test_batch.py                # 批量测试
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

## 知识图谱节点类型

| 类型 | 示例 | 说明 |
|------|------|------|
| Entity | 商机、客户、出库明细 | 业务实体 |
| Event | 商机创建、预计签单 | 业务事件 |
| Attribute | 城市、行业、产品类别 | 维度属性 |
| Metric | 商机金额、出库金额 | 度量指标 |
| Concept/Filter | 在途商机、已签商机 | 概念/过滤器 |
| Function | 同比、占比、增长率 | 聚合函数 |
| MetricCategory | 金额指标、数量指标 | 指标分类 |

## 边类型

| 边类型 | 含义 | SQL边 |
|--------|------|-------|
| `has_attribute` | 实体→属性 | 否 |
| `has_event` | 实体→事件 | 否 |
| `measured_by` | 实体→度量 | 否 |
| `derived_from` | 衍生关系 | 否 |
| `relates_to` | 跨域关联 | 否 |
| `constrains` | 过滤约束 | 否 |
| `where`/`group_by`/`having` | SQL 条件 | 是 |
