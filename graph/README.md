# 商机语义知识图谱检索系统

基于语义知识图谱的自然语言查询系统，将用户的业务分析问题转化为结构化图谱上下文，为 Text-to-SQL 提供精确的语义支撑。

## 核心流程

```
自然语言问题  →  [Phase 1] 实体抽取  →  [Phase 2] 子图构建  →  [Phase 3] 上下文格式化  →  JSON 输出
                   (规则/Embedding/LLM)   (BFS/分层聚合)          (结构化上下文)
```

### Phase 1: 实体抽取 (`entity_extractor.py`)

三级策略：规则匹配（精确）→ Embedding 语义（泛化）→ LLM 精筛（严出）。识别问题中的指标、维度、过滤器等实体。

> 所有参数（Embedding 阈值、LLM 开关等）统一在 `config.yaml` 的 `extraction` 区块配置。

### Phase 2: 子图构建 (`subgraph_builder.py` / `subgraph_retriever_v2.py`)

在知识图谱中查找实体间的最优路径，构建最小相关子图。支持两种策略：

| 策略 | 文件 | 特点 |
|------|------|------|
| **v1** | `subgraph_builder.py` | 核心路径优先 BFS，区分指标/约束/中间节点 |
| **v2** | `subgraph_retriever_v2.py` | 分层聚合（GMM 聚类 + LLM 抽象），支持 SQL 边捷径注入、Event 度量桥接识别 |

> 通过 `config.yaml` 的 `retriever` 键切换。v2 模式下自动启用 SQL 直接边捷径注入，避免 BFS 绕路拉入无关领域节点。

### Phase 3: 上下文格式化 (`context_formatter.py`)

将子图格式化为结构化 JSON，包含 nodes、edges、paths、capability 等字段，供下游消费。

## 快速开始

```bash
cd retrieve

# 安装依赖
pip install -r require.txt

# 单次查询（LLM/Embedding 等参数由 config.yaml 控制）
python main.py "嘉兴市大项目商机数量和出库金额占比"

# 交互模式
python main.py

# 批量测试（6 个预设问题）
python test_batch.py

# 交互测试 + LLM 评判 + HTML 可视化
python test_interactive.py --judge

# 临时覆盖最大跳数
python main.py --hops 6 "各行业的在途商机去重金额"
```

> **所有参数统一在 `config.yaml` 中配置**，无需在命令行传递 `--llm` / `--embedding` 等参数。

## 知识图谱 (`data/商机.json`)

### 节点类型

| 类型 | 示例 | 说明 |
|------|------|------|
| **Entity** | 商机、客户、出库明细 | 业务实体 |
| **Event** | 商机创建、预计签单、中标 | 业务事件 |
| **Attribute** | 城市、行业、产品类别 | 维度属性 |
| **Metric** | 商机金额、出库金额、去重商机金额 | 度量指标 |
| **Concept/Filter** | 在途商机、已签商机、大项目 | 概念/过滤器 |
| **Function** | 同比、占比、增长率 | 聚合函数 |
| **MetricCategory** | 金额指标、数量指标 | 指标分类 |

### 边类型

| 边类型 | 含义 | 示例 |
|--------|------|------|
| `has_attribute` | 实体→属性 | 商机→城市 |
| `has_event` | 实体→事件 | 商机→商机创建 |
| `measured_by` | 实体→度量 | 商机→商机金额 |
| `derived_from` | 衍生关系 | 总商机金额→商机金额 |
| `relates_to` | 跨域关联 | 商机→出库明细 |
| `constrains` | 过滤约束 | 在途商机→预计签单 |
| `where` / `group_by` | SQL 条件（SQL 边） | 行业→去重商机金额 |

每个节点关联数据库元数据：`cube`（数据立方体）、`column`、`metric`、`filter` 等，为 SQL 生成提供基础。

## 项目结构

```
graph/
├── data/
│   └── 商机.json                   # 核心：商机语义知识图谱
├── retrieve/
│   ├── main.py                     # CLI 入口（单次查询 + 交互模式）
│   ├── loader.py                   # 图谱加载（Node/Edge/SemanticGraph）
│   ├── entity_extractor.py         # 实体抽取（规则+Embedding+LLM）
│   ├── index_builder.py            # 索引构建（倒排+邻接表）
│   ├── subgraph_builder.py         # 子图构建 v1（BFS）
│   ├── subgraph_retriever_v2.py    # 子图召回 v2（分层聚合）
│   ├── hierarchical_aggregator.py  # 分层聚合引擎（GMM+LLM抽象+SQL捷径）
│   ├── context_formatter.py        # 上下文格式化
│   ├── metric_capability.py        # 指标能力矩阵
│   ├── config.yaml                 # 统一配置文件
│   ├── test_batch.py               # 批量测试
│   ├── test_interactive.py         # 交互测试 + HTML 可视化
│   └── results/                    # 测试结果（JSON + HTML）
├── add_sql_edges.py                # 图谱预处理脚本
└── require.txt                     # Python 依赖
```

## 配置 (`config.yaml`)

所有参数统一在此配置，修改后所有入口（`main.py` / `test_batch.py` / `test_interactive.py`）同步生效：

```yaml
# 实体抽取策略
extraction:
  use_embedding: true           # 是否启用 Embedding 语义匹配
  use_llm: true                 # 是否启用 LLM 精筛（需要 DeepSeek API）
  embedding_threshold: 0.35     # Embedding 宽召回相似度阈值
  embedding_max_results: 15     # Embedding 最大候选数

# 子图召回策略
retriever: "v2"                 # "v1" = BFS最短路径, "v2" = 分层聚合
max_hops: 5                     # BFS 最大跳数

# v2 分层聚合参数
v2:
  use_diffusion: true
  max_levels: 2
  min_cluster_size: 2
  cross_cluster_threshold: 0.1
  use_llm: true
  force_rebuild_hg: false

# DeepSeek API
deepseek:
  api_key: "sk-xxx"
  base_url: "https://api.deepseek.com/chat/completions"
  model: "deepseek-v4-flash"
```
