# 神经符号 FSAD 核心代码审阅指南

## 1. 本轮实现要证明什么

第一阶段不以 AUROC 提升作为完成标准，而是验证以下可证伪命题：

1. 所有类别使用同一条处理链，不写类别名称分支；
2. 不同正常 support 的统计结构可以自动产生不同规则结构；
3. 系统能区分“规则可执行”“规则未实现”“当前图像证据不足”；
4. 没有 length executor 时绝不把长度候选映射为 count 或 aspect；
5. INP 视觉残差不再是逻辑违反成立的必要条件；
6. 每条逻辑违反必须返回可追溯的实例、槽位或关系证据及空间图。

这套实现目前是一个**最小可信核心**，不是已经完成全部 LOCO 类别的最终系统。
当前注册了 4 类可执行原子规则：`length_range`、`aspect_range`、
`color_range`、`count_eq`。关系类 schema 已预留，但没有 executor 的关系候选必须
如实报告为 `UNSUPPORTED`，不能假装已经实现。

## 2. 推荐阅读顺序

### 2.1 数据协议与能力边界

先读：

- `symbolicAD/rules/rule_schema.py`
- `symbolicAD/rules/rule_registry.py`
- `symbolicAD/reasoning/violation_result.py`

`RuleSpec` 是候选规则的唯一协议。它记录谓词、实体类型、参数、依赖、来源、
grounding 策略和 support 证据，但不保存 query 标签或 GT mask。

`RuleRegistry` 是系统真实能力清单。候选空间可以很大，但只有精确注册的谓词才能
执行。`resolve()` 不做 alias 或 fuzzy matching，因此 `length_range` 不会退化成
`aspect_range`。

`ViolationResult` 是每个 executor 的统一输出：

```text
rule_id
status: SATISFIED | VIOLATED | UNKNOWN
violation_score
evidence_confidence
witness_instances / witness_relations / missing_slots
grounding_strategy
anomaly_map
attribution_uncertainty
details
```

编译状态与运行状态必须分开理解：

| 层次 | 状态 | 含义 |
|---|---|---|
| 编译期 | `EXECUTABLE` | 有 executor 且正常支持证据通过验证 |
| 编译期 | `UNSUPPORTED` | 候选被保留，但系统没有对应 executor |
| 编译期 | `REJECTED` | 正常 support 已反驳候选或规则格式无效 |
| 编译期 | `INSUFFICIENT_EVIDENCE` | K-shot 证据不足，不能可靠接受 |
| query 期 | `SATISFIED` | 证据充分且约束满足 |
| query 期 | `VIOLATED` | 证据充分且约束违反 |
| query 期 | `UNKNOWN` | 感知、属性或对应证据不足，不能判断 |

### 2.2 感知事实如何建立

再读：

- `symbolicAD/perception/component_instances.py`
- `symbolicAD/perception/part_attributes.py`
- `symbolicAD/perception/spatial_alignment.py`
- `symbolicAD/perception/scene_graph.py`

`ComponentInstance` 隔离了感知 backend。人工 mask、二值连通域、未来的
GroundingDINO/SAM 都必须转换成同一结构。正式定位优先使用实例 mask，bbox 只是
元数据和低置信度 fallback。

`part_attributes.py` 从 mask 点云提取 PCA 主轴长度、宽度、aspect、方向、面积和
端点。关键点是 length 与 aspect 分别存储：长度是主轴投影跨度，aspect 是长度与
宽度之比。弯曲目标若需要测量中心线/geodesic length，应新增明确属性和执行器，
不能修改现有 length 的含义来适配某个类别。

`spatial_alignment.py` 用 Hungarian assignment 建立 support/query 实例对应，但把
identity、shape、position 三项代价分开保存。这样可以判断失败来自实体认错、形状
变化还是位置变化。

### 2.3 规则如何自动产生并决定是否可执行

核心文件：

- `symbolicAD/rules/rule_compiler.py`
- `symbolicAD/rules/rule_validator.py`

执行路径为：

```text
normal support SceneGraph
  -> SupportRuleInducer：逐实体统计 count/length/aspect/color 的稳定性
  -> InductionDecision：每个候选都记录提出或拒绝及原因
  -> RuleRegistry：精确检查是否有 executor
  -> RuleValidator：在正常 support 上执行候选
  -> CompiledRuleSet：保留全部成功/失败 CompileRecord
```

`SupportRuleInducer` 没有类别参数；类别差异只来自 support 事实。属性相对跨度超过
阈值时得到 `UNSTABLE_ON_SUPPORT`，数量不恒定时 count 被拒绝。K-shot 不足或属性
缺失不会被解释成“满足”，而会进入证据不足状态。

必须注意：support 稳定只说明“这是一个可检验的正常候选”，不能说明它一定覆盖
真实异常。真实覆盖只能由严格隔离的评价集分析。

### 2.4 query 上如何执行和组合规则

核心文件：

- `symbolicAD/reasoning/predicate_executor.py`
- `symbolicAD/reasoning/constraint_executor.py`

每个原子执行器遵循同一顺序：

1. 验证规则参数；
2. 验证实体身份、检测完整性和必需属性；
3. 证据不足立即返回 `UNKNOWN`；
4. 计算规则真值与违反强度；
5. 只有 `VIOLATED` 才产生非零规则图。

`constraint_executor.py` 使用 Strong Kleene 三值逻辑组合正常性约束：

| 运算 | 可以确定 SATISFIED | 可以确定 VIOLATED | 其他情况 |
|---|---|---|---|
| normality AND | 全部满足 | 任一违反 | UNKNOWN |
| normality OR | 任一满足 | 全部违反 | UNKNOWN |

最终状态不是 `VIOLATED` 时，总逻辑图为零。逐规则结果仍完整保留，便于定位是哪条
规则无法判断。

### 2.5 空间责任如何产生

核心文件：

- `symbolicAD/grounding/instance_grounder.py`
- `symbolicAD/grounding/slot_grounder.py`
- `symbolicAD/grounding/shape_grounder.py`
- `symbolicAD/grounding/relation_grounder.py`
- `symbolicAD/grounding/counterfactual_repair.py`

不同规则使用不同空间责任：

| 规则类型 | 空间证据 |
|---|---|
| 属性超界 | 违反实例的 mask |
| 长度超界 | 把长度修到最近合法边界时的端点差异 |
| 数量过多 | 多余实例 mask |
| 数量不足 | normal support 对齐得到的缺失 slot 概率图 |
| 二元关系 | 两个 witness 实例 mask 与可选关系连线 |

`counterfactual_repair.py` 生成的是规则参数空间中的最小修复责任图，不是生成式 RGB
修复图。缺失端点属于反事实预测，所以 uncertainty 高于已经可见的多余端点。

三层 grounding 实验共用同一个 executor，仅替换证据来源：

1. 人工确认规则 + 人工组件 mask：空间归因理论上界；
2. 人工确认规则 + 自动视觉感知：测量 perception/匹配损失；
3. 自动规则 + 自动视觉感知：端到端系统。

人工或 GT mask 只允许用于第 1 层诊断，`InstanceGrounding` 会显式记录
`uses_ground_truth_during_inference=True`。

### 2.6 与 INP-Former baseline 如何连接

核心文件：

- `symbolicAD/entity_bridge.py`
- `symbolicAD/fusion/fusion.py`

桥接被拆成两个独立问题：

- `EntityVerifier`：INP/DINO 特征是否支持实体身份与实例对应。该置信度决定规则能否
  执行，低置信度导致 `UNKNOWN`。
- `AnomalyConfirmer`：INP reconstruction residual 是否支持局部外观异常。它是辅助
  诊断和视觉异常图，不修改逻辑真值。

最终融合使用 `max` 或 probabilistic OR，都是单调操作。已确认的逻辑异常不会因
INP residual 低而被清除；视觉热点也不会被逻辑分支压低。`UNKNOWN` 则逐位保持
baseline 不变。

### 2.7 规则集选择与定位评价

核心文件：

- `symbolicAD/rules/rule_optimizer.py`
- `symbolicAD/evaluation/spro.py`

规则集优化先使用可解释贪心基线：验证依赖图、对依赖闭包、按干预响应去冗余、再
最大化新增诊断覆盖。`diagnostic_intervention_coverage` 不能叫真实异常覆盖；报告把
`real_anomaly_coverage` 固定为 `None`。

LOCO 评价模块按原始灰度值读取缺陷区域和官方饱和阈值，分别在 logical 与
structural 子集上计算 sPRO 曲线，并归一化积分到 5% FPR。评价 map family 包括：

- `rule/<rule_id>`：逐规则空间图；
- `logic_total`：三态约束组合后的总逻辑图；
- `fused`：与 INP baseline 单调融合后的最终图。

GT mask 只进入 `evaluation/spro.py`，不进入任何 inference API。

## 3. 具体训练/归纳实例：length-only 类别

可执行回归位于 `tests/test_typed_rule_pipeline.py`。

构造 4 张正常 support，wire 实例数分别为 1、2、3、4；颜色与 aspect 大幅变化；
每个实例的显式 length 约为 0.4。系统对同一实体类型尝试四种统计：

```text
count_eq      -> [1, 2, 3, 4]，不稳定，拒绝
color_range   -> 跨多个 hue，不稳定，拒绝
aspect_range  -> 0.25/0.65/1.2/3.4，不稳定，拒绝
length_range  -> 约 0.398~0.402，稳定，提出候选
```

候选随后经 registry 精确解析到 `LengthRangeExecutor`，再在四张正常 support 上逐张
执行；已知结果均为 `SATISFIED`，因此编译为 `EXECUTABLE`。编译结果必须满足：

```text
structure_signature = ("length_range",)
substitutions = ()
```

异常 query 的 length 改为 0.20，但 aspect=0.65、color_h=0.25 都取正常 support
出现过的值。预期只有长度规则返回：

```text
status = VIOLATED
witness_instances = ["wire_0"]
grounding_strategy = "instance_mask_endpoint_difference"
anomaly_map.max() > 0
```

同一 induction pipeline 另构造一个 count 恒为 2、length/aspect/color 均不稳定的
类别，得到 `structure_signature=("count_eq",)`。两组 fixture 没有类别名称分支，
因此直接证明不同支持结构会产生不同规则结构。

额外给出未注册候选 `curvature_profile`，预期被保留为：

```text
disposition = UNSUPPORTED
reason_code = NO_REGISTERED_EXECUTOR
substitutions = ()
```

这项断言防止未来又把缺失能力静默映射到 aspect/count。

## 4. 实验审计入口

优先执行最小核心回归：

```bash
conda run -n H2VLR python -m pytest -q \
  tests/test_typed_rule_pipeline.py \
  tests/test_entity_bridge_v13.py \
  tests/test_spro_evaluation.py
```

审阅时建议逐一核对：

1. `induced.decisions` 是否包含每个未提出候选的原因；
2. `compiled.audit()` 是否保留 unsupported/rejected 规则；
3. `ConstraintExecution.rule_results` 是否能回溯到 witness 和 UNKNOWN 原因；
4. 低 INP residual 是否仍保留可靠逻辑违反；
5. UNKNOWN 融合是否与原 baseline map 逐位相等；
6. sPRO 是否分别报告逐规则、逻辑总图和融合图。

## 5. 当前仍需继续验证的部分

- 当前 length 是 PCA 主轴长度，对弯曲线缆不等于 geodesic length；需要独立能力。
- 当前自动实例感知接口已统一，但仍需在五个 LOCO 类别上验证 mask、实体身份和
  slot 对齐质量。
- relation schema 已存在，但未注册的关系 executor 仍应报告 `UNSUPPORTED`。
- support 稳定阈值目前是透明基线，需在 K=1/2/4 下做敏感性和校准实验。
- 复杂搜索尚不能作为主贡献；必须先在同协议、多 seed 下稳定超过贪心基线。
- 正式结论必须同时报告 image AUROC 与 logical/structural sPRO@5% FPR，避免只提升
  图像级分数却无法定位逻辑异常区域。
