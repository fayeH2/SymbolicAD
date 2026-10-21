# SymbolicAD 当前技术路线、知识分支与规则账本

> 文档状态：当前方案技术基线（可审计版本）  
> 更新日期：2026-10-21  
> 项目路径：`/home/hjh/Desktop/SymbolicAD`  
> 主要验证数据集：MVTec LOCO，K=4，support seed=1  
> 证据范围：当前仓库代码、`results_core_loco` 实验产物和 91 项回归测试  
> 重要边界：本文严格区分“已经实现”“已经产生候选”“已经编译”“已经完成测试评测”和“计划实现”。

---

## 1. 当前结论

当前项目已经建立了一条可审计的 Few-Shot Anomaly Detection 基线与单实体神经符号控制链，但尚未完成可宣称为“通用高阶神经符号推理”的多实体系统。

已经成立的部分：

1. 使用干净、train-only、可复现的 K-shot 正常支持集训练 INP-Former。
2. INP-Former 的训练 loss、梯度覆盖、checkpoint hash 和测试缓存均有完整记录。
3. Qwen3-VL 只从正常支持图提出闭集语义候选，不在测试时参与推理。
4. GroundingDINO/几何统计可把单一重复实体转换为候选逻辑原子和局部 witness map。
5. 当前逻辑搜索仅允许正单调 `AND/OR` normality tree，避免 XOR/IMPLIES/内部 NOT 导致多个违规相互抵消。
6. Entity Bridge 使用 INP 特征对规则局部区域二次确认，符号分支在证据不足时 abstain。
7. MVTec LOCO 的 all/logical/structural 图像指标、配对热图及固定选样 manifest 已生成。

尚未成立的部分：

1. 当前 schema 最多表示一个实体类型，不能表达 `washer–nut–screw`、`connector–cable` 等跨实体关系。
2. 当前 pushpins 只有一条 `count_eq` 语义提案，属于基数控制，不是高阶规则。
3. 当前 `results_core_loco` 尚无新 C1 规则库及融合评测产物，因此当前核心实验“已评测生效规则数”为 **0**。
4. 当前 loss 是纯 INP-Former loss，符号分支为 inference-only；不能称为“神经符号 loss 已收敛”。
5. 尚未完成冻结代码后的跨数据集、多 K、多 seed 证伪式验证。

---

## 2. 总体技术路线

当前维护流程为：

```text
MVTec-like Dataset
        │
        ├── Dataset/support protocol
        │     ├── train/good only
        │     ├── exclude flattened *_val
        │     ├── SHA-256 seed token
        │     └── nested K={1,2,4,8} prefixes
        │
        ├── C0 neural baseline
        │     ├── frozen DINOv2-reg ViT-B/14 encoder
        │     ├── trainable prototype/aggregation/decoder
        │     ├── encoder–decoder cosine residual
        │     └── pixel map + top-1% image score
        │
        ├── Normal-support knowledge induction
        │     ├── Qwen3-VL per-support semantic proposal
        │     ├── 3/4 support consensus
        │     └── fail-closed abstention
        │
        ├── C1 open-vocabulary grounding
        │     ├── GroundingDINO prompt/threshold/NMS search
        │     ├── count/layout/size/aspect facts
        │     ├── support-calibrated upper bounds
        │     └── atom-local witness maps
        │
        ├── C2 simple logic controls
        │     ├── best single atom
        │     ├── all atoms AND
        │     └── all atoms OR
        │
        ├── C3 monotone program search
        │     ├── support-only single/pair/triple counterfactuals
        │     ├── group-robust genetic search
        │     ├── positive AND/OR normality AST
        │     └── exhaustive monotonicity validation
        │
        ├── Entity-token bridge
        │     ├── GroundingDINO boxes ↔ INP feature-grid ROI tokens
        │     ├── normal slot/entity-set prototypes
        │     ├── rule-local INP evidence gate
        │     └── rescue / suppression / abstention
        │
        └── Evaluation and audit
              ├── all/logical/structural image AUROC and AP
              ├── common good controls
              ├── fixed-scale paired heatmaps
              ├── checkpoint/support/source hashes
              └── official sPRO deferred until official config is restored
```

最终目标不是让每个数据集强行使用相同视觉谓词，而是让所有数据集遵循相同阶段接口：

```text
DatasetSpec
  → normal supports
  → entity/attribute observations
  → typed atoms
  → support calibration or abstention
  → monotone normality program
  → witness maps
  → conservative fusion
  → unified evaluator and visualization manifest
```

对象类可以使用实体与关系 grounder；纹理或医学类别如果无法稳定实体化，应当自动 abstain 并退化为 baseline。

---

## 3. 分支、知识来源与当前状态

| 分支 | 状态 | 使用的知识 | 关键入口 | 当前作用 |
|---|---|---|---|---|
| C0 INP-Former baseline | 当前维护 | DINOv2视觉表征、正常支持集、encoder–decoder重构残差 | `scripts/train_generic_inpformer.py`, `inpformer_adapter.py` | 提供空间异常主干、feature grids、pixel map和image score |
| V2 reliable static rules | 历史 | Qwen语义候选、正常扰动、伪局部腐蚀、map拓扑统计 | `run_reliable_neurosymbolic_fsad.py`, `reliable_rules.py` | 历史诊断；旧LOCO五类均选0条规则 |
| V3 relational | 历史正向PoC | 实体颜色prototype、包含、属性多重集、相对位置、LOO refit | `run_relational_neurosymbolic_fsad.py`, `cable_relations.py` | 证明关系规则可能有效，但grounder为cable类专用 |
| V7 executable logic | 历史机制证据 | transistor部件知识、product fuzzy logic、LOO/bootstrap | `run_executable_logic_fsad.py`, `transistor_logic.py` | 证明同一AST可产生image score与pixel map；不具跨类通用性 |
| V9 grounded counterfactual | 历史思想保留 | 图像级detach/cut/bend/extra干预、normal nuisance | `grounded_counterfactuals.py` | 用真实视觉干预替代纯truth篡改；当前实现仍类专用 |
| SymbolicDet evolution | 当前保留搜索机制 | 单/双/三原子反事实、遗传搜索、group robustness、复杂度惩罚 | `symbolicdet_evolution.py` | 搜索正单调normality AST；不是NSGA-II |
| V11 orthogonal fusion | 已废弃 | 特征层调制、规则去相关；但使用NOT/XOR/IMPLIES | `run_generic_orthogonal_fsad.py` | 非单调规则可产生违规抵消，不进入维护路径 |
| V12 OVD | 当前C1组件 | Qwen单实体语义、GroundingDINO、PCA几何、robust support envelope | `run_ovd_symbolic_fsad.py`, `ovd_grounder.py` | 生成单实体候选原子和witness maps |
| V13 Entity Bridge | 当前C1组件 | INP ROI token、正常prototype、规则局部残差证据 | `run_entity_bridge_fsad.py`, `entity_bridge.py` | 阻止无神经证据的规则直接覆盖baseline |
| 多实体 typed atoms | 计划中 | 多实体集合、属性、配对、组成、比例、跨实体空间关系 | 尚未实现 | 解决LOCO高阶逻辑GAP的核心下一步 |

### 3.1 神经视觉知识

INP-Former 使用 DINOv2-reg ViT-B/14 作为视觉 encoder。当前训练器配置为：

- embedding dimension：768；
- attention heads：12；
- encoder target layers：2–9，共8层；
- encoder feature fusion：`[0,1,2,3]` 与 `[4,5,6,7]` 两组；
- decoder：8个 `Prototype_Block`；
- prototype tokens：6；
- 输入：Resize至448×448，再CenterCrop至392×392；
- anomaly map：插值至256×256并Gaussian平滑；
- image score：anomaly map最高1%像素均值。

encoder 不在 optimizer 的 trainable 参数集合中；训练的是 bottleneck、aggregation、decoder 和 prototype tokens。

### 3.2 正常支持集统计知识

所有阈值和原型应来自正常支持集，而不是测试标签：

- 支持图选择使用完整 SHA-256 派生seed；
- 对固定seed，K=1/2/4/8是嵌套前缀；
- MVTec LOCO 中后缀为 `_val` 的扁平化验证图不进入support；
- 每张support保存SHA-256；
- Grounder阈值使用原始support与normal-preserving augmentation估计；
- 测试图和测试标签不参与规则生成、阈值校准或AST搜索。

### 3.3 语言模型语义知识

Qwen3-VL-8B-Thinking 的作用是提出可检验的语义候选，不是输出最终规则：

1. 每张正常support独立调用，避免把拼图数量误认为单图数量。
2. 最多提出一个重复、可视觉分离的实体类型。
3. 每次可提出1–5条闭集谓词候选。
4. K=4时要求至少75%，即3/4 support一致。
5. count不一致则整体abstain；arrangement不一致则退化为`free`。
6. 纹理、字母、阴影、反光和背景pattern被提示为不可选实体。
7. LLM不决定数值阈值、fusion gain或测试时判定。

### 3.4 开放词汇检测与几何知识

GroundingDINO把实体描述转换为box集合。当前support-only搜索空间包括：

- prompt候选；
- confidence threshold：0.12–0.40；
- NMS IoU：0.15–0.55；
- 目标：正常support上的count match、count MAE、增强一致性与检测置信度；
- 正常support count match低于0.75时fail closed。

正确计数的support用于拟合：

- absolute slot center/size；
- PCA主轴；
- 主轴方向上的normalized gaps；
- line deviation；
- relative log area；
- relative aspect。

当前局限：`grid/row/ring` 尚未拥有各自独立执行器；现阶段主要共享PCA排序逻辑。`similar_orientation` 实际由relative aspect代理，不是显式角度或关键点方向。

### 3.5 模糊逻辑知识

对每个规则原子，1表示正常条件满足，0表示违反。当前normality program只允许：

```text
ATOM(name)
AND(child_1, ..., child_n)
OR(child_1, ..., child_n)
```

product fuzzy semantics：

```text
AND(x_1,...,x_n) = product(x_i)
OR(x_1,...,x_n)  = 1 - product(1-x_i)
anomaly           = NOT(normality) = 1-normality
```

虽然解析器为旧trace兼容仍能读取 `NOT/IMPLIES/XOR`，但当前normality validator会拒绝：

- 内部NOT；
- IMPLIES；
- XOR；
- 重复原子；
- 未知原子；
- 超过16个原子的穷举单调性审计。

对每个Boolean赋值边，validator验证“把某个原子从违规0提升到满足1，normality不能下降”。

### 3.6 反事实与进化搜索知识

当前搜索不读取异常测试图，而从正常support truth vector生成伪违规事件：

- 每原子2个single-atom variants；
- 每原子对1个pair variant；
- 每原子三元组1个triple variant；
- 原始正常support rows作为negative/normal group。

若4个OVD原子均存在，K=4且每张support生成5个detector augmentation，则共有24个normal truth rows。由此可生成：

- normal：24；
- single counterfactual：4 × 24 × 2 = 192；
- pair counterfactual：C(4,2) × 24 = 144；
- triple counterfactual：C(4,3) × 24 = 96；
- 合计：456 rows，其中432个为符号反事实异常。

OVD当前搜索配置：

- population size：48；
- generations：35；
- tournament size：5；
- elite count：6；
- crossover rate：0.5；
- mutation rate：0.3；
- complexity penalty：0.004；
- atom cost：0.015；
- normal false-positive penalty：0.2；
- worst-group weight：0.30；
- normal-invariance weight：0.25；
- early-stop patience：10。

当前是scalarized group-robust genetic search，不是NSGA-II。只有C3超过best-single/all-AND/all-OR这些C2简单控制后，才有理由增加NSGA-II或MCTS。

### 3.7 Entity Bridge知识

Entity Bridge建立GroundingDINO实体与INP特征之间的桥：

1. 在每个box内对每层INP feature grid计算mean和standard deviation。
2. 拼接并L2 normalize，得到entity token。
3. 用正常support拟合slot token prototypes、leave-one-out cosine distance threshold和scale。
4. 用同图其他实体构造permutation-invariant entity-set reference，区分真实缺失与detector miss。
5. 对每条规则，在其witness map区域内统计正常support的INP局部top score上界。
6. 非count规则只有在自己的局部区域超过该support上界时才能激活。
7. missing count可由entity-token关系支持；extra detection仍必须存在局部INP热点。

当前固定gain：

- rescue gain：0.035；
- suppression gain：0.020。

它们曾在两个开发类别上选择，尚不能当作跨数据集最优常数。Bridge还要求expected count至少为2、每个slot至少3个正常token，因此目前不能完整支持K=1/2和单对象类别。

---

## 4. Rule规则账本

### 4.1 为什么不能只报告一个“总规则数”

规则存在四个不同层级：

1. **schema predicate**：LLM被允许提出的谓词种类；
2. **executable atom**：grounder真正有代码计算的视觉事实；
3. **proposal rule**：某一类别通过support语义共识的候选；
4. **active deployed rule**：完成grounding、校准、搜索、冻结并经过测试评测的规则。

把这些数量相加会产生错误结论。

### 4.2 当前规则数量

| 层级 | 数量 | 当前内容 | 状态 |
|---|---:|---|---|
| Qwen schema允许的谓词 | 6类 | count, arrangement, size, orientation, spacing, appearance | 声明完成 |
| OVD真正实现的视觉原子 | 4类 | count_error, layout_error, size_error, aspect_error | 代码已实现 |
| pushpins共识proposal | 1条 | `count_eq(pushpin, 12)` | 已生成语义候选 |
| 当前核心编译后rule library | 0个产物 | 尚无新V12/V13规则库落盘 | 尚未运行验证 |
| 当前核心完成融合评测的active rule | 0条 | baseline metrics不能冒充C1结果 | 尚未运行验证 |

因此，当前最准确的说法是：

> 项目定义了6类候选谓词，当前grounder实际实现4类单实体视觉原子；pushpins建立了1条语义候选规则，但当前核心实验尚无完成编译和融合评测的active规则。

### 4.3 六类schema谓词

| 谓词 | family | 意义 | 当前执行状态 |
|---|---|---|---|
| `count_eq` | cardinality | 实体数量等于正常期望值 | 已实现为`count_error` |
| `arrangement_eq` | spatial_configuration | 实体排列与正常结构一致 | 部分实现为PCA layout error |
| `similar_size` | scale_geometry | 实体相对面积一致 | 已实现为`size_error` |
| `similar_orientation` | orientation_geometry | 实体方向一致 | 仅以aspect proxy实现 |
| `ordered_spacing` | spatial_configuration | 相邻实体间距顺序一致 | schema存在，未编译执行 |
| `appearance_consistency` | appearance | 外观/颜色/纹理一致 | schema存在，未编译执行 |

### 4.4 四个OVD可执行原子

| 原子ID | visual fact | 正常阈值 | witness定位 |
|---|---|---|---|
| `ovd_count_eq` | `count_error` | upper=0.005；scale=max(0.5/N,0.05) | 缺失slot或extra box |
| `ovd_arrangement_eq` | `layout_error` | max(original support + floor, augmented q90 + 0.5 robust sigma) | 每个检测box |
| `ovd_similar_size` | `size_error` | 同上 | 偏离正常relative area的box |
| `ovd_similar_orientation` | `aspect_error` | 同上 | 偏离正常relative aspect的box |

违规程度：

```text
excess    = max(0, fact_value - upper)
violation = tanh(excess / soft_scale)
truth     = 1 - violation
```

### 4.5 当前必须修复的rule compiler偏差

当前 `compile_ovd_program` 固定遍历4个 `OVD_PREDICATES`，没有依据 `proposal["rules"]` 做交集。因此即使pushpins的Qwen proposal只有1条`count_eq`，编译器仍可能创建4个候选原子。

这属于当前已确认缺陷，不能把潜在4原子写成Qwen建立了4条规则。下一次C1实验前必须：

1. 仅编译proposal明确提出且grounder支持的谓词；
2. 未知或未实现谓词fail closed；
3. compiler audit分别保存pre-search candidates与post-search active rules；
4. 不得用active rules覆盖candidate inventory字段。

### 4.6 C2简单规则控制

当4个原子均有效时，搜索必须同时比较6个简单控制：

- 4个single-atom程序；
- 1个all-AND；
- 1个all-OR。

C3搜索树必须超过这些控制，才能证明“高阶组合”而非“某一个强原子”贡献了增益。

### 4.7 历史规则数量，仅作背景

这些历史数量不能与当前核心数量相加：

| 分支 | 规则数量 | 结果与边界 |
|---|---:|---|
| V3 MVTec cable | 最终2条 | color identity multiset + relative position；类专用grounder |
| V7 transistor k30 | 32 raw，2 accepted | accepted为lead count和lead geometry；单类开发 |
| V4 MVTec/VisA/MPDD 33类 | 89 candidates，6 survived | 宏观增益接近0，不构成广泛泛化证据 |
| 旧MVTec LOCO五类 | 每类0条 | fusion disabled，结果就是baseline |
| V13旧全MVTec | 仅2/15类可ground | 说明单实体机制覆盖率不足 |

---

## 5. Baseline训练与loss

### 5.1 训练参数

当前五类MVTec LOCO联合训练配置：

| 参数 | 值 |
|---|---|
| dataset | `mvtec_loco` |
| classes | breakfast_box, juice_bottle, pushpins, screw_bag, splicing_connectors |
| shot | 每类4，共20张正常图 |
| support seed | 1 |
| epochs | 200 |
| batch size | 16 |
| encoder | `dinov2reg_vit_base_14` |
| input / crop | 448 / 392 |
| prototype tokens | 6 |
| optimizer | StableAdamW |
| initial / final LR | 1e-3 / 1e-4 |
| warmup | 100 iterations |
| weight decay | 1e-4 |
| gradient clipping | global norm 0.1 |

20张support经过22-way augmentation形成440个内存训练样本。`drop_last=True`时每epoch为27个batch，即200 epochs约5400个optimizer steps。

### 5.2 Loss定义

训练总损失：

```text
L_total = L_reconstruction + 0.2 * L_global
```

`L_reconstruction`是多层encoder/decoder特征的平均cosine reconstruction loss。encoder feature被detach，只训练decoder侧。对patch级距离使用hard-mining梯度调制：

```text
d_patch = 1 - cosine(F_encoder, F_decoder)
gradient_factor = (d_patch / mean(d_patch))^3
```

因此异常样式较强、重构距离较大的patch获得更大的decoder梯度，但当前没有任何rule truth或logic loss进入该目标。

### 5.3 两种22-view augmentation

**official22**：

- 原图1；
- ±11.25°、±22.5°、±33.75°、±45°旋转，共8；
- ±10%和±20%组合平移，共8；
- 水平翻转1；
- 灰度1；
- 90°/180°/270°旋转，共3。

**loco_safe22**：

- 原图1；
- ±2.5°、±3.75°、±5°小旋转，共6；
- 不超过3%的小平移，共8；
- 亮度factor 0.94/0.97/1.03/1.06，共4；
- 对比度factor 0.94/1.02/1.06，共3。

safe版本避免翻转、灰度和大幅平移破坏left/right、颜色身份和计数，但它是实验消融，不是默认已胜出的策略。

### 5.4 收敛结果

| 配置 | Epoch 1 total / reconstruction / global | Epoch 200 total / reconstruction / global | first10 mean | last10 mean | ratio |
|---|---|---|---:|---:|---:|
| official22 | 0.734385 / 0.565535 / 0.844250 | 0.182693 / 0.081414 / 0.506393 | 0.423203 | 0.182980 | 0.432370 |
| loco_safe22 | 0.717767 / 0.550363 / 0.837023 | 0.161430 / 0.063559 / 0.489357 | 0.402671 | 0.161616 | 0.401359 |

两支均满足：

- 200 epochs全部finite；
- gradient parameter fraction=1.0；
- finite gradient fraction=1.0；
- tail slope为负；
- official22 checkpoint SHA-256：`fa5b0dd2e2eea1702aa716e36c2cf5668d95c7edcd8695914f792f4098d1ca28`；
- safe22 checkpoint SHA-256：`d8468e423df4de5fe39bb7390a73a75e8a2f80d1d52c556614e3d43b2aac0aca`。

safe22有更低训练loss，但测试AUROC并不全面更高，因此不能根据训练loss选择最终模型。

---

## 6. 完整类别实例：MVTec LOCO pushpins，K=4

### 6.1 实例性质

当前checkpoint是五个LOCO类别、共20张support联合训练的模型，不是pushpins单类模型。本节把pushpins作为“类别级support选择、规则生成、独立评测和可视化”的完整实例。

### 6.2 支持集

协议：

- dataset：MVTec LOCO；
- class：pushpins；
- K=4；
- support seed=1；
- selector schema：`mvtec_loco_train_only_sha256_nested_v2`；
- 仅使用`train/good`；
- 排除`*_val`。

| 样本 | SHA-256 |
|---|---|
| `pushpins/train/good/370.png` | `623cd2c93a61ca88756b8365c3d7a9826ac5caf68c7ceba487ec7eaf15d5d4df` |
| `pushpins/train/good/197.png` | `8fc742dfb528ae40215776055b0d4b5231494e155a2b90004226c659346189ff` |
| `pushpins/train/good/179.png` | `a0a0381558ad6ff15d631c7611a18a87d4a1c97448d334a85561cf880512b19a` |
| `pushpins/train/good/356.png` | `0fa0d3ede771255c18529863fc5211f63c1e217ff460a50093768aa9026826c8` |

pushpins贡献4×22=88个增强训练样本；模型总训练集仍为五类的440个增强样本。

### 6.3 Qwen语义提案

四张支持图分别独立提议，结果为：

| support proposal | entity count | predicate |
|---:|---:|---|
| 1 | 15 | `count_eq` |
| 2 | 12 | `count_eq` |
| 3 | 12 | `count_eq` |
| 4 | 12 | `count_eq` |

3/4共识得到：

```json
{
  "entity": "pushpin",
  "expected_count": 12,
  "arrangement": "grid",
  "rule": {
    "id": "pushpin_count_eq",
    "predicate": "count_eq",
    "family": "cardinality",
    "importance": 0.9,
    "confidence": 0.9
  }
}
```

当前建立的是 **1条语义候选规则**：

```text
Normality rule: count(pushpin) == 12
Anomaly event: count(pushpin) != 12
```

`grid`目前只是entity元数据；当前proposal没有建立真正的grid关系规则。一次support误数15也说明感知层尚不稳定。

### 6.4 这条规则要成为active rule仍需经过什么

1. GroundingDINO在原始4张support上至少3/4正确计数。
2. prompt、threshold、NMS仅在support及其增强上选择。
3. 对每张正常support和增强计算`count_error`。
4. 校准`upper=0.005`与`soft_scale=max(0.5/12,0.05)=0.05`。
5. 生成缺失slot/extra box witness map。
6. Entity Bridge验证缺失slot是否也得到INP token或局部残差支持。
7. 与pure baseline及single-atom C1做配对评测。
8. 报告正常误触发率、logical/structural分组性能和定位图。

这些步骤当前尚未形成新的核心C1实验产物，所以不能写成“pushpins已有1条验证有效规则”。

### 6.5 Pushpins baseline测试结果

测试样本：

- normal：138；
- logical anomaly：91；
- structural anomaly：81；
- all：310；
- logical subset：138 normal + 91 logical = 229；
- structural subset：138 normal + 81 structural = 219。

| augmentation | subset | Image AUROC | Image AP |
|---|---|---:|---:|
| official22 | all | 0.575497 | 0.618635 |
| official22 | logical | 0.500557 | 0.369181 |
| official22 | structural | 0.659689 | 0.552201 |
| loco_safe22 | all | 0.555190 | 0.615347 |
| loco_safe22 | logical | 0.475315 | 0.375271 |
| loco_safe22 | structural | 0.644928 | 0.536824 |

结论：

- safe22训练loss更低，但pushpins的all/logical/structural AUROC均下降；
- logical AUROC约0.50，说明baseline几乎无法区分正常与逻辑异常；
- 因此不能根据loss选择augmentation，也不能把safe增强当作逻辑推理替代品；
- pushpins是验证count/slot grounding的合适C1对象，但不是证明多实体高阶推理的充分对象。

### 6.6 Pushpins可视化实例

代表样本按official22 image score的类别内中位数自动选择：

| family | 样本 | official22 score | safe22 score |
|---|---|---:|---:|
| good | `test/good/090.png` | 0.192180 | 0.232136 |
| logical | `test/logical_anomalies/001.png` | 0.193860 | 0.215164 |
| structural | `test/structural_anomalies/061.png` | 0.211284 | 0.248010 |

good与logical的official score仅差约0.00168，直观解释了logical AUROC接近随机。所有图使用：

- Resize(448,448)；
- CenterCrop(392,392)；
- Resize map至256×256；
- pooled q99.5统一heatmap上限0.315560；
- signed difference统一范围±0.099091。

可视化文件：

- `results_core_loco/visual_comparison/auroc_comparison.png`；
- `results_core_loco/visual_comparison/representative_heatmaps.png`；
- `results_core_loco/visual_comparison/representative_heatmaps_manifest.json`。

![五类normal/logical/structural代表样本的配对热图](results_core_loco/visual_comparison/representative_heatmaps.png)

---

## 7. 当前五类baseline结果与GAP

### 7.1 Official22

| class | all AUROC | logical AUROC | structural AUROC |
|---|---:|---:|---:|
| breakfast_box | 0.841891 | 0.781479 | 0.897603 |
| juice_bottle | 0.718040 | 0.684747 | 0.768334 |
| pushpins | 0.575497 | 0.500557 | 0.659689 |
| screw_bag | 0.612583 | 0.490726 | 0.816174 |
| splicing_connectors | 0.712979 | 0.596639 | 0.860801 |
| macro | 0.692198 | 0.610830 | 0.800520 |

### 7.2 LOCO-safe22

| class | all AUROC | logical AUROC | structural AUROC |
|---|---:|---:|---:|
| breakfast_box | 0.862858 | 0.860855 | 0.864706 |
| juice_bottle | 0.718716 | 0.688193 | 0.764826 |
| pushpins | 0.555190 | 0.475315 | 0.644928 |
| screw_bag | 0.595928 | 0.494795 | 0.764894 |
| splicing_connectors | 0.734576 | 0.633131 | 0.863470 |
| macro | 0.693454 | 0.630458 | 0.780565 |

safe22相对official22：

- macro all AUROC：+0.001256；
- macro logical AUROC：+0.019628；
- macro structural AUROC：-0.019955。

这是一种明显trade-off，不是全面改进。热图还显示部分safe响应是大面积背景抬高，不能解释为已经学习逻辑结构。

![Official22与LOCO-safe22的五类AUROC对比](results_core_loco/visual_comparison/auroc_comparison.png)

---

## 8. 评测、可视化与泄漏控制

### 8.1 图像指标

当前独立evaluator只报告可辩护的image-level AUROC/AP：

- all：全部good + 全部anomaly；
- logical：全部good + logical anomaly；
- structural：全部good + structural anomaly。

logical与structural必须复用同一批good controls。只在异常样本上计算AUROC是无效协议，代码会对single-class subset fail closed。

### 8.2 定位指标

MVTec LOCO官方定位指标是sPRO@5% FPR。当前本地扁平化数据缺少官方defect configuration，项目明确：

- 不报告伪官方sPRO；
- 不用global pixel AUROC替代sPRO；
- 待官方配置/评测包恢复后补算。

### 8.3 可视化协议

每次维护实验必须至少输出：

```text
Original
Pixel GT
Baseline heatmap/overlay
Detected entities
Per-atom witness maps
Symbolic map
Fused map
Signed delta map
Triggered rule + support threshold
```

样本必须按固定分层规则选择并保存manifest，不能只展示“前N张”或“所有triggered样本”，因为这会引入选择偏差。

### 8.4 Provenance

当前记录包括：

- support绝对路径和SHA-256；
- checkpoint绝对路径和SHA-256；
- support selector schema；
- 训练源代码SHA-256；
- cache中的checkpoint/input/crop/map metadata；
- 每图path、score、defect type；
- 可视化选样规则和全局色标。

---

## 9. 已删除、废弃或不能进入核心的部分

### 9.1 已删除的LOCO错误入口

以下旧脚本因硬编码缺失路径、使用无关transistor checkpoint、support污染或无good control的AUROC而删除：

- `run_loco_experiment.py`；
- `run_loco_enhanced.py`；
- `train_loco_inpformer.py`。

### 9.2 已删除的误导性可视化入口

- `run_all_mvtec_categories.py`：错误地按4个panel切分5/6联图，并用original冒充GT；
- `run_all_datasets_master.py`：把整个comparison拼图直接重命名为DHA，且依赖失效绝对路径。

### 9.3 历史版本处理原则

V2–V13是实验演进记录，不是可同时打开的模块集合。历史代码只能用于：

- 复现实验；
- 提取已经验证的机制；
- 分析失败原因。

组件只有在paired ablation中超过baseline和simple control、无test-label选模且没有结构异常显著退化后，才能进入maintained core。

---

## 10. 当前已知缺陷与风险

1. **单实体限制**：无法表达跨类型组成、比例和配对关系。
2. **规则声明/执行不一致**：schema有6类，grounder只执行4类。
3. **proposal/compiler不一致**：compiler尚未按proposal rules过滤。
4. **arrangement语义不足**：grid/row/ring未独立执行。
5. **orientation代理不充分**：aspect不等价于方向。
6. **检测确认偏差**：threshold以expected count为目标，可能找到“数量正确但实体错误”的blob。
7. **K=4估计不稳定**：pushpins已有1/4 support计数为15。
8. **Bridge覆盖不足**：expected count<2或每slot正常token不足3时不能工作。
9. **gain未跨数据集冻结验证**：0.035/0.020来自开发类别。
10. **符号分支无梯度路径**：不能声称logic loss收敛。
11. **无当前C1最终指标**：proposal存在不等于规则有效。
12. **官方sPRO缺失**：定位结论仍不完整。

---

## 11. 下一阶段唯一核心实现

不再新增V14/V15式平行runner；应在当前维护路径加入一个typed multi-entity compiler。

### 11.1 新schema

```json
{
  "entities": [
    {"id": "washer", "prompt": "..."},
    {"id": "nut", "prompt": "..."},
    {"id": "long_screw", "prompt": "..."}
  ],
  "atoms": [
    {"predicate": "count_eq", "entity": "washer", "value": 2},
    {"predicate": "relative_position", "left": "nut", "right": "washer"},
    {"predicate": "count_ratio", "left": "washer", "right": "nut"},
    {"predicate": "attribute_compatible", "entity": "connector", "attribute": "cable"}
  ]
}
```

### 11.2 第一批typed atoms

- `count_eq(entity, n)`；
- `count_ratio(entity_a, entity_b, r)`；
- `one_to_one(entity_a, entity_b)`；
- `relative_position(entity_a, entity_b, relation)`；
- `inside(entity_a, entity_b)`；
- `attribute_multiset_eq(entity, attributes)`；
- `attribute_compatible(entity_a, entity_b)`；
- `slot_occupancy(entity, slot)`。

### 11.3 强制实验阶梯

| 级别 | 内容 | 通过条件 |
|---|---|---|
| C0 | official22 INP baseline | 固定基线 |
| C0-safe | relation-preserving augmentation | 仅作消融 |
| C1 | 单实体原子+固定gate | 证明grounding有效 |
| C2 | multi-entity atoms + best single/all-AND/all-OR | 证明原子有信息且简单聚合已充分比较 |
| C3 | searched monotone AST | 必须超过C2且不明显损害structural |
| C4 | differentiable feature gating/logic loss | 仅在C3通过后考虑 |

### 11.4 泛化验证

在宣称普适性前，必须使用同一代码、冻结超参数完成：

1. MVTec LOCO五类；
2. 一个非LOCO工业关系类；
3. 一个应该主动abstain的纹理或医学类；
4. K={1,2,4,8}；
5. 至少5个nested support seeds；
6. paired confidence intervals；
7. logical与structural分开报告；
8. C3超过C2而不只是超过C0。

---

## 12. 复现入口与产物索引

### 12.1 核心代码

- 数据与support：`neurosymbolic_fsad/data.py`
- baseline训练：`scripts/train_generic_inpformer.py`
- augmentation/loss：`neurosymbolic_fsad/training_utils.py`
- INP推理：`neurosymbolic_fsad/inpformer_adapter.py`
- Qwen候选：`neurosymbolic_fsad/qwen_generic_relations.py`
- OVD grounder：`neurosymbolic_fsad/ovd_grounder.py`
- AST搜索：`neurosymbolic_fsad/symbolicdet_evolution.py`
- Qwen AST advisor：`neurosymbolic_fsad/qwen_evolution_advisor.py`
- Entity Bridge：`neurosymbolic_fsad/entity_bridge.py`
- LOCO evaluator：`scripts/evaluate_loco_checkpoint.py`
- 可视化：`scripts/visualize_loco_checkpoint_comparison.py`

### 12.2 Baseline产物

- official checkpoint：`results_core_loco/full200/official22/model.pth`
- official history/manifest：同目录下`training_history.json`、`training_manifest.json`
- safe checkpoint：`results_core_loco/full200/loco_safe22/model.pth`
- safe history/manifest：同目录下`training_history.json`、`training_manifest.json`
- 评测：`results_core_loco/evaluation/{official22,loco_safe22}`

### 12.3 Rule proposal产物

- pushpins proposal：`results_core_loco/proposals_single_entity_control/mvtec_loco/pushpins/k4/qwen_generic_relational_proposal.json`
- Qwen原始输出：同目录`qwen_generic_relational_raw.txt`
- support montage：同目录`normal_support_relational_montage.jpg`
- proposal summary：`results_core_loco/proposals_single_entity_control/mvtec_loco/k4_proposal_summary.json`

### 12.4 审计与图表

- protocol/GAP审计：`results_core_loco/PROTOCOL_AND_GAP_AUDIT_20261020.md`
- AUROC对比图：`results_core_loco/visual_comparison/auroc_comparison.png`
- paired heatmaps：`results_core_loco/visual_comparison/representative_heatmaps.png`
- 可视化manifest：`results_core_loco/visual_comparison/representative_heatmaps_manifest.json`

---

## 13. 最终口径

当前项目可以作出的结论是：

> INP-Former baseline已经在无support泄漏的MVTec LOCO K=4协议下稳定收敛，并完成五类logical/structural分组评测。项目已经建立Qwen正常语义候选、单实体开放词汇grounding、support-only单调逻辑搜索与INP局部证据bridge的核心组件。当前规则schema包含6类谓词，OVD实际实现4类视觉原子，pushpins生成1条`count_eq`候选；但尚无新核心规则完成编译与融合评测，因此当前验证生效规则数仍为0。下一阶段必须完成多实体typed atoms并按C0–C3阶梯验证，之后才能讨论显式高阶推理和跨数据集普适性。

该口径同时保留正向成果、负向结果和当前证据边界，不把语义候选、代码能力或训练loss误写成已经验证的神经符号性能。

### 13.1 当前验证状态

- 全量回归测试：91 passed；
- `git diff --check`：通过；
- 两个200-epoch训练历史：全部epoch finite；
- 两套evaluation cache：五类路径一一对应；
- 可视化：已按模型坐标系、固定选样和统一色标目视检查；
- 未完成项：新C1规则编译/融合评测、官方sPRO、多实体C2/C3和跨数据集冻结验证。
