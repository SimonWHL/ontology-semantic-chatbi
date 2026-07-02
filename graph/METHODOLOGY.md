# 子图构建方法论演进记录

> 本文档记录图谱检索系统子图构建的设计思路、关键决策和每次改动的原因。
> 每次修改子图构建逻辑时，必须在本文档末尾追加更新记录。

---

## 1. 图谱拓扑分析

### 1.1 域（Cube）划分

图谱中存在两个核心数据域，它们的表在物理上是独立的（通过 opportunity_id JOIN）：

| 域 | Cube ID | 核心实体 | 核心指标 |
|---|---|---|---|
| 商机域 | `cube_kpi_opportunity` | 商机、商机创建 | 商机金额、商机数、去重商机数... |
| 出库明细域 | `cube_gn_sales_deliver_detail` | 出库明细、客户 | 出库金额 |

### 1.2 跨域桥接节点

`商机`（Entity）是商机域和出库明细域之间的唯一语义桥接：
- `商机 → 出库明细`（通过 opportunity_id 关联）
- `客户 → 商机`（relates_to）

这意味着：**任何 BFS 从商机域出发，最多 2 跳就能到达出库明细域的全部节点**。
这就是盲目 BFS 的污染根因。

---

## 2. 方法论核心原则

### 2.1 域准入：LLM 实体提取 = 动态白名单

**核心规则：LLM 实体提取结果决定哪些域可以进入子图。LLM 没提的域，BFS 一概不入。**

```
LLM 提取 → 各实体归属的 cube 集合 = 准入域列表
BFS 展开 → 邻居节点属于准入域外的 cube → 阻断（整个节点不纳入）
```

### 2.2 三级节点分类（准入门内）

准入域确定后，对域内和域外节点分类：

| 级别 | 定义 | 判定依据 | 示例 |
|---|---|---|---|
| **MUST-NOT** | 准入域外的**任何节点** | 该节点所在 cube 不在 LLM 提取的实体覆盖范围内 | 商机问题：出库明细域全部节点 |
| **MUST** | LLM 提取到的 Metric / Function | LLM 直接输出，承载答案语义 | 商机数、商机金额、同比 |
| **NEUTRAL** | 所有非 Metric 节点 | Entity / Attribute / Event / MetricCategory，仅作为语义脚手架，不产生答案级内容 | 产品类别、行业、商机创建… |

### 2.3 关键洞察：唯一会"产生答案"的是 Metric

```
Metric      → 答案载体（LLM 看到就认为是可查询数值）→ MUST 或 MUST-NOT
非-Metric   → 语义脚手架（帮助理解计算口径、维度拆分、业务链路）→ 永远是 NEUTRAL
```

这意味着：
- 只有 Metric/Function 有资格进入 MUST 和 MUST-NOT 两档
- Entity/Attribute/Event/MetricCategory 永远只是 NEUTRAL——进不进入子图都不构成答案级污染

### 2.4 两个典型场景

**场景 A：用户同时提了两个域 → 都准入**

```
问题: "商机数量和出库金额的同比增长率"
LLM 提取: {商机数, 出库金额, 同比}
准入域:   cube_kpi_opportunity + cube_gn_sales_deliver_detail ✅
BFS 可自由穿越两个域
```

**场景 B：用户只提了一个域 → 另一域完全阻断**

```
问题: "今年规上（出库额10万以上）商机数和金额、及同比？"
LLM 提取:  {商机数, 商机金额, 同比, 规上商机}
准入域:    cube_kpi_opportunity ✅
          cube_gn_sales_deliver_detail ❌（零实体，不入场）

关键：括号内"出库额10万以上"是规上商机的约束定义，
      不是对出库金额的查询诉求。
      LLM 正确选择了"规上商机"而非"出库金额"。
```

### 2.5 不可协商的硬约束

**① 子图构建阶段不使用 SQL 边**

```
SQL 边（where / group_by）只在子图构建完成后附加。
BFS 遍历时只走语义边（measured_by / derived_from / relates_to / has_attribute / constrains 等）。
```

原因：SQL 边会让任何 Attribute 直接连到所有 Metric，导致 BFS 1 跳即可遍历全图，完全失去了路径筛选的意义。SQL 边是"结果"，不是"构建过程"。

**② 子图必须全连通**

```
子图 = 所有节点的语义路径闭合图
不允许存在孤立节点或孤立组件。
```

原因：断开连接的子图意味着"这些信息之间没有关系"，但用户问题天然假设它们有关联。如果 BFS 找不到连通路径，说明图谱缺失了关键语义边。

### 2.6 跨域污染的严重性

> **出库明细域的任何节点出现在纯商机域问题中 = 不可接受的错误。**

原因：
- 下游 LLM 看到出库域节点（尤其是出库金额）会误以为需要 JOIN 出库表
- SQL 生成可能错误地引入跨域关联
- 同环比等派生指标的上下文发生混淆

---

## 3. 实现方向

### 3.1 域准入流程

```
LLM 实体提取 → 实体列表
       ↓
每个实体查 node_map → 获取 cube 字段
       ↓
汇总 = 准入域集合 (admitted_cubes)
       ↓
BFS 展开时：邻居的 cube ∉ admitted_cubes → 跳过（不入子图）
       ↓
特殊情况：实体无 cube 字段 → 视为无域限制，退化为全图 BFS
```

### 3.2 伪代码

```python
def build_subgraph(question, extracted_entities, graph):
    # Step 1: 确定准入域
    admitted_cubes = set()
    for entity_name in extracted_entities:
        node = graph.node_map.get(entity_name)
        if node and node.cube:
            admitted_cubes.add(node.cube)
    
    # Step 2: 域感知 BFS（仅语义边，不含 SQL 边）
    #  BFS 只遍历 node.cube 为空或 node.cube ∈ admitted_cubes 的节点
    #  跨域节点一律不入图
    #  边类型仅包含：measured_by, derived_from, relates_to, has_attribute,
    #               constrains, has_event, classified_as, supports_function 等语义边
    subgraph = bfs_semantic_only(entities, admitted_cubes)

    # Step 3: 连通性保证
    #  子图必须全连通，存在孤立组件则修正（放宽跳数或准入约束）
    assert is_fully_connected(subgraph), "子图存在孤立节点"

    # Step 4: 附加 SQL 边
    subgraph = attach_sql_edges(subgraph)

    return subgraph
```

### 3.3 构建流程

```
LLM 提取实体 → 准入域列表
       ↓
语义 BFS（不含 SQL 边，域准入过滤）
       ↓
子图必须全连通（失败则修正）
       ↓
附加上 SQL 边（where / group_by）
       ↓
输出最终子图
```

### 3.4 与现状的关键区别

| | 现状 (v2.x) | v3.0 |
|---|---|---|
| guardrail 位置 | BFS 内部：metric_core 过滤、跳数限制 | BFS 入口：域准入 |
| 跨域 Metric | 通过 constrains 边绕进去 | 直接阻断，不进子图 |
| 跨域 Entity | 可能作为桥接进入 | 不属于准入域 → 阻断 |
| SQL 边 | BFS 遍历时使用 | 构建完成后附加 |
| 连通性 | 允许孤立节点 | 必须全连通 |
| 净效果 | 出库金额通过"产品类别→出库明细→商机"污染 | 产品类别被拒绝进域，整条链断掉 |

---

## 4. 改动记录

| 日期 | 版本 | 改动内容 | 触发原因 |
|---|---|---|---|
| - | v1.0 | 初始：全对全 BFS 路径并集 | - |
| - | v2.0 | V2 分层聚合（GMM + LCA 引导） | 节点过多 |
| 2026-07-02 | v2.1 | 贪心链：Phase A 指标链式接入，Phase B 约束只连一次 | O(n²) BFS 太慢且冗余 |
| 2026-07-02 | v2.2 | metric_core：约束只连 Metric/Function 节点，不连 Entity 桥接 | 产品类别 → 出库明细 → 商机 污染 |
| 2026-07-02 | v2.2 | 约束 3 跳天花板 + retry 只做指标 | 行业通过扩大跳数重连拖回污染链 |
| **待实施** | **v3.0** | **域准入 BFS：LLM 提取实体 → 准入域集合 → 跨域节点阻断** | 上述所有补丁都没有解决"跨域污染"的根因 |
| 2026-07-02 | 方法论 | 确立核心原则：<br>① LLM 实体提取决定域准入，没提的域一概不进<br>② 只有 Metric/Function 产生答案级影响<br>③ 非 Metric 节点永远 NEUTRAL，不构成污染<br>④ SQL 边不参与子图构建，构建完成后附加<br>⑤ 子图必须全连通，不允许孤立节点 | v2.x 补丁越打越多，需要从方法论层面重新定义问题边界 |
