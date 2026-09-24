# PROTOCOL-v2 冻结前科学评审

## 1. 评审信息与范围 / Review Information

- Template: ccfa-review-1
- Mode: scientific
- Detail: detailed
- Rubric: generic-7
- Source version: `docs/PROTOCOL-v2.md` at commit `35568e0`, with its plan `docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md` (Amendments 5–6)
- Contribution type: empirical benchmark / evaluation protocol, reviewed as a Stage-1 registered report before `v2-preregistered`
- Venue: none named. Judged against generic ML evaluation-paper standards.
- Materials inspected:
  - the protocol itself;
  - `configs/v2.yaml` and `configs/v2-freeze.json`;
  - `docs/public-results.csv`;
  - the candidate class counts of each dataset in the pinned cache: Emotion, SMS, Civil and AG News, counted locally with the repository loaders;
  - the approved assessment `docs/exec-plans/active/2026-09-24-benchmark-assessment.md`.
- Not available:
  - results (no benchmark inference has run);
  - the final manifest (`prepare` has not run);
  - the study Rmd.
- Evidence is therefore `not assessed`. Every judgment concerns design, not outcomes.
- The review is a single-agent role simulation, not an independent review panel. It is separate from the claudex-loop Codex code inspections.

## 2. 总体结论与关键理由 / Expected Review Outcome

**结论：边界偏正（borderline positive）；修复 C001–C002 后可以冻结。**

- **主要长处：**
  - 统计与溯源纪律高于同类公开评测：预注册、按内容分组划分数据、失败计入分母、策略配对的 A/B 校准比较、冻结 k 的 Holm 校正、快照一致性检查。
  - 头条问题（给本地模型同样约 200 条标签后，Jev 零样本的优势还剩多少）有实际决策价值。
- **决定性顾虑（都是可修复的界定问题，不是设计错误）：**
  - [C001] 类别平衡抽样改变了估计对象，但协议没有说明。头条指标"5% 错误预算下可自动化的流量比例"依赖类别先验。
  - [C002] 头条问题里"用同样的标签"对应哪一种比较没有写明。只有 B-vs-B 两边都用了那 200 条标签。
- **新颖性低**（与已批准评估一致）。协议已把自己定位为同批次受控复核，这一定位是准确的。
- 置信度 4/5。

## 3. 预审与投稿适配 / Desk Rejection Assessment

Not assessed：未指定投稿会议，页数、匿名与格式规则不适用。可评审性：pass，协议足以让第三方重建设计。

## 4. 论文摘要与贡献拆解 / Summary And Contributions

**研究问题。** 在同样的类型化决策上（choice / noul / score，9 个数据集），比较以下几方：

| 类别 | 参赛者 |
|---|---|
| 零样本 | OpenRouter 托管的 Jev 1.13；本地 Laya（base 与 multilingual）；本地 Qwen3-1.7B 末层标签 logit 打分 |
| 少标签（Amendment 5） | 用每数据集 200 条校准标签训练的 `qwen_probe`、`tfidf_lr`、`prior` |

**核心主张（均为待检验的设计，不是结果）：**
1. 确认性检验：C1–C3 零样本配对比较，指标为准确率、Brier A-vs-A、Brier B-vs-B，数据集为 AG News、Emotion、SMS、Civil，k=9，Holm 校正。
2. 头条估计：零样本差距与少标签差距并排报告，指标含准确率、Brier、5% 错误下的覆盖率，只做估计。
3. 次要分析：多语（MASSIVE）、score 与 noul 数据集、AG News 顺序敏感性。

**贡献类型。** 受控的同批次复核评测。核心贡献是一组控制手段的组合，而不是新方法：
- 对称校准；
- 扣除重复噪声后的顺序效应；
- Laya 不截断的 head 设置；
- 少标签对照组。

## 5. 主要优势 / Strengths

1. **预注册与哈希冻结。** Status 段与 `v2-freeze.json` 构成"先冻结、后推理"的可核查链条。探针只用合成字符串（Artifacts 段）。
2. **失败计入与最差惩罚。** Outcomes 段加 Amendment 6 的失败规则：失败调用计 Brier 2、NLL −log 0.005、覆盖率 0，不会出现"只算成功调用"的乐观偏差。这一点公开同类评测普遍没有做到。
3. **策略配对的校准比较。** 确认性 Brier 检验只做 A-vs-A 和 B-vs-B，bootstrap 每次重抽都重新拟合温度 T。这正面检验了 Jev 的"内置校准"主张，避免公开材料中"refit 后的 ECE 对比 raw ECE"那种不对等比较。
4. **Laya 公平设置。** head 规则经 `build_sequence` 合同测试验证，不会截断选项。Banking77 如实标为 N/A。
5. **数据许可审查。** 托管推理会把原文发给第三方，所以替换了 ToxicChat 和 Yelp（Amendment 4）。
6. **头条估计与确认性检验的衔接。** 主集上的零样本行直接复用 C1 的结果与 Holm 决定，避免同一比较有两套数字。

## 6. 主要问题与严重程度 / Major Concerns

### C001: 类别平衡抽样定义了一个与部署先验不同的估计对象，协议未声明

- Type / 类型: unsupported_claim
- Severity / 严重程度: major
- Location / 位置: Data and exclusions 第 2 段："class-balanced selection"；Amendment 6 头条指标"coverage at 5% error"；Amendment 5 Framing，与 `docs/public-results.csv` 的对照。
- Evidence / 证据: pinned 数据集的候选池类别计数：SMS Spam 为 4,827 条正常、747 条 spam（13.4%）；Civil Comments 为 1,660,540 条非毒、144,334 条毒（8.0%）；Emotion 最小类 66 条；平衡抽样后，测试集的正负比约为 50/50；Brier、ECE、温度拟合和选择性自动化阈值都随先验变化。"能自动化多少流量"在 50/50 与 8% 先验下含义不同；公开对照中，elcronos 使用自然分布，并指出 Laya 报告的 0.595 对 0.480 用的是类别平衡样本。
- Countercheck / 反证复核: 在协议与计划中搜索 prevalence、natural distribution、base rate，均无结果；协议只写了"class-balanced selection"，没有说明它对估计对象的含义。问题成立。
- Judgment / 判断: 确认性检验本身仍然有效，它比较的是同一平衡样本上的配对差；头条覆盖率与"对照公开结果"的解读可能被误读为部署层面的结论。
- Criterion / 维度: Soundness, Clarity
- Resolution / 解决或改判条件: 在协议中明确：估计对象是类别平衡的测试分布；覆盖率与校准指标不代表自然先验下的部署表现；可选的描述性补充：按候选池的类别先验对每条测试样本重新加权，给出 Brier 和覆盖率（不做推断）；与公开结果对照时注明分布差异。
- Status / 状态: unresolved

### C002: 头条问题"同样约 200 条标签"对应的是 B-vs-B，协议没有写明

- Type / 类型: clarification
- Severity / 严重程度: major
- Location / 位置: Amendment 6 首段（"how much of zero-shot Jev's advantage is left when a local model gets the same ~200 labels"）及其指标列表。
- Evidence / 证据: 在 condition B 下，Jev 的温度 T 用同样的 200 条校准标签拟合；少标签分类器则直接在这 200 条上训练；condition A 下，Jev 完全不用标签，少标签组仍用标签；因此只有 B-vs-B（以及 B 下的覆盖率）是"双方都拿到这 200 条标签"的比较。A-vs-A 衡量的是"零样本原样输出"对"有监督"。
- Countercheck / 反证复核: Amendment 6 同时列出 A 与 B 的行，但没有指明哪一行回答头条问题。问题成立。
- Judgment / 判断: 读者可能用 A-vs-A 的差距回答头条问题，从而高估本地少标签方案相对于"同样拿到标签的 Jev"的优势。
- Criterion / 维度: Clarity, Soundness
- Resolution / 解决或改判条件: 在 Amendment 6 中写明：头条问题的对等读数是 B-vs-B 的 Brier 与覆盖率，以及准确率。准确率不受 T 影响，A 与 B 相同；A-vs-A 作为"原样零样本对有监督"的补充读数；报告和 Rmd 按这个顺序呈现。
- Status / 状态: unresolved

## 7. 次要问题与写作表达 / Minor And Presentation Concerns

### C003: macro-F1 被称为 primary，但不在确认性检验家族中

- Type / 类型: confirmed_flaw
- Severity / 严重程度: minor
- Location / 位置: Outcomes 第 1 段（"Primary ... accuracy, macro-F1 and Brier"）；Confirmatory inference 第 1 段（只检验 accuracy 与 Brier）。
- Evidence / 证据: 两段直接矛盾。
- Countercheck / 反证复核: not needed：两段原文直接对照即可确认。
- Judgment / 判断: 读者会误以为 macro-F1 有确认性地位。
- Criterion / 维度: Clarity
- Resolution / 解决或改判条件: 把 macro-F1 改为 secondary/descriptive，或说明它是主要报告指标但不做推断。
- Status / 状态: unresolved

### C004: 冻结文本含已被取代的表述，修订顺序倒置

- Type / 类型: confirmed_flaw
- Severity / 严重程度: minor
- Location / 位置: Exploratory 段仍列 "latency scaling"，并把 "out-of-domain typed-decision checkpoint" 列为局限；Deviations 段写 "Amendment 3 adds GLiNER2.5 as a descriptive contender"；Amendment 6 出现在 Amendment 5 之前。
- Evidence / 证据: Amendment 6 已删除 latency 套件、laya_typed，以及 v2 中的 GLiNER 运行。
- Countercheck / 反证复核: not needed：删除事实在 Amendment 6 段与 `configs/v2.yaml` 中可直接核对。
- Judgment / 判断: 冻结后的协议必须能单独读懂，残留表述会让预注册文本自相矛盾。
- Criterion / 维度: Clarity
- Resolution / 解决或改判条件: 冻结前把 Amendment 5–6 并入正文对应章节；被取代的句子标为 superseded 或删除；修订史保留在 Deviations 或计划文件里。
- Status / 状态: unresolved

### C005: 跨数据集等权平均的 Brier 差量纲不同

- Type / 类型: clarification
- Severity / 严重程度: minor
- Location / 位置: Confirmatory inference 第 2 段（"equal-weight average of its frozen dataset effects"）。
- Evidence / 证据: 多分类 Brier 的取值范围与类别数有关。四个数据集分别为 K=4、6、2、2。等权平均时，K 较大的数据集可能主导数值。
- Countercheck / 反证复核: not needed：这是解读层面的问题，不影响检验的有效性。
- Judgment / 判断: 检验仍然有效，但合并效应量的单位不直观。
- Criterion / 维度: Clarity
- Resolution / 解决或改判条件: 报告中同时给出每个数据集的差值，并说明合并值是跨 K 的等权平均。
- Status / 状态: unresolved

### C006: "类别平衡"受类别可得量限制

- Type / 类型: clarification
- Severity / 严重程度: minor
- Location / 位置: Data and exclusions 第 2 段。
- Evidence / 证据: Emotion 的 surprise 类只有 66 条候选；按 pilot 5、calibration 33、test 50 的顺序配额，最多需要 88 条；test 最多只能拿到约 28 条，其余由其他类别补足。
- Countercheck / 反证复核: 协议写明会报告 per-class support，所以事后可见，但协议没有说明这一限制。问题成立。
- Judgment / 判断: "class-balanced" 对 Emotion 不成立。C001 讨论的是先验，这里是实际的样本构成。
- Criterion / 维度: Clarity, Reproducibility
- Resolution / 解决或改判条件: 协议写明"在类别可得量范围内平衡"；`prepare` 之后在冻结记录中列出各 split 的实际类别计数。
- Status / 状态: unresolved

### C007: 冻结记录不覆盖探针证据

- Type / 类型: clarification
- Severity / 严重程度: minor
- Location / 位置: Artifacts and capability gate 第 2 段；`configs/v2-freeze.json`。
- Evidence / 证据: 协议以 `results/runs/jev-laya-v2/probe-results.json` 为能力证据，但该文件被 `.gitignore` 排除；冻结记录只含 config、manifest、protocol 三个哈希。
- Countercheck / 反证复核: 探针结果只含合成字符串的聚合形状、长度与计数，不含数据集文本，按仓库规则可以作为聚合报告提交。问题成立。
- Judgment / 判断: 第三方无法核对协议中关于探针结果的每一条陈述。
- Criterion / 维度: Reproducibility
- Resolution / 解决或改判条件: 把探针结果作为聚合报告提交到 `results/reports/`，或在 `v2-freeze.json` 里加入它的 sha256。
- Status / 状态: unresolved

## 8. 新颖性与相关工作 / Novelty And Positioning

| Work / source | What it already establishes | Overlap and remaining difference | Consequence / concern ID |
| --- | --- | --- | --- |
| elcronos jev-vs-open-decision-models | 冻结协议下 Jev、Laya、PrismNLI 零样本比较，含 ECE/Brier/NLL、McNemar 检验、监督 LR 基线与标签效率曲线 | 重叠：Jev 对 Laya、校准指标、监督基线。本研究多出：LLM logit 路线、noul/score 题型、对称温度校准、同批次三方比较 | 新颖性低；定位为复核是恰当的；自然分布与平衡分布的差异见 [C001] |
| open-alternative-jev | Qwen3-1.7B logit 打分对 Jev、Laya（typed-decisions 基准） | 重叠：logit 路线。本研究多出：公开数据集、同批次 Jev、校准拆分 | 已列入 public-results.csv |
| AnyJev | Qwen3-1.7B 加约 300 标签的 head、flip rate、5% 风险覆盖率 | 重叠：少标签臂、覆盖率指标。本研究多出：与同批次 Jev 在公开数据集上配对比较 | 头条问题的新增量有限，需要 [C002] 界定清楚 |
| nibzard decision-model-benchmark | Jev 置换 flip 13%、Banking/SMS | 本研究多出：扣除重复噪声后的顺序效应 | 次要分析 |

覆盖范围：已检索并抓取一手页面（沿用已批准评估 §2）；未系统检索同行评审文献。

## 9. 方法正确性与主张支撑 / Soundness And Claim Support

| Claim / location | Inspected support | Judgment | Consequence / concern ID |
| --- | --- | --- | --- |
| 确认性配对差可推断（Confirmatory inference） | 数据集分层配对 bootstrap，每次重拟合 T；Holm 冻结 k=9；p 下限 1/2000，低于首步阈值 0.0056 | 设计成立 | — |
| B-vs-B 公平衡量"内置校准" | 双方都用同样 200 条标签拟合 T；T 无法拟合时有明确规则（T=1 或标为不可用） | 成立 | 头条解读见 [C002] |
| 头条"可自动化流量"具有部署含义 | 平衡测试集；阈值在平衡的校准集上选取 | 不能直接外推到部署 | [C001] |
| 少标签臂不泄漏测试标签 | 5 折交叉拟合；测试只用全 200 条重训的模型；合同测试"改变测试标签不改变任何概率" | 成立 | — |
| 失败不美化结果 | Brier 2、NLL −log ε、覆盖率 0；全失败的 bootstrap 抽样有定义或 T=1 | 成立 | — |
| 合并效应可解释 | 跨 K 的等权平均 | 有效但单位不直观 | [C005] |

## 10. 实验、证明与可复核性 / Evaluation And Reproducibility

- **样本与检出能力：** 每数据集 test 300 条。已批准评估 §13 的敏感度表给出：分歧率 0.1–0.3 时，最小可检出差约 3.3–5.7 pp（Holm 首步、80% 功效）；更小的差被报告为"未分辨"，处理得当。
- **可复核：** 所有版本 pin 到固定 revision；运行与报告都校验哈希；按 attempt 隔离命名空间；预算账本在每次调用前落盘并 fsync。不足：
  - 数据准备必须联网，协议已说明；
  - 探针证据未进入冻结记录 [C007]；
  - 类别实际计数需要在 `prepare` 之后记录 [C006]。
- Evidence 维度：not assessed，没有结果。

## 11. 多视角评审与综合意见 / Reviewer Perspectives And Synthesis

单一 agent 的角色模拟，不是独立评审。

| View | Decisive basis / concern IDs | Stance | What would change it |
| --- | --- | --- | --- |
| Best-supported assessment | §5 的优势 1–6；[C001] [C002] 可修复 | 边界偏正，修复后可冻结 | 若平衡先验无法说明，头条外推会被高估 |
| Strongest favorable reading | 失败感知、对称校准、预注册，严谨度高于全部公开对照 | 作为复核研究有明确价值 | 依赖 [C002] 写清对等读数 |
| Strongest critical reading | 新颖性低（§8）；头条估计全部是描述性的 | 贡献主要是工程与决策层面 | 结果如果显著偏离公开先验并能解释，意义会上升 |

综合：没有致命缺陷。影响冻结的是 [C001] 与 [C002] 两处估计对象界定，其余属于冻结前的文本清理。

## 12. 维度评分与置信度 / Critical Reviewer Ratings

### Scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty / 新颖性 | 2 | 4 | §8；public-results.csv | 核心比较已被公开工作覆盖。结果若揭示公开结论之外的可解释差异，可升到 3 |
| Soundness / 正确性 | 4 | 4 | §9；Confirmatory inference；Amendment 6 失败规则 | 扣分项 [C001] [C005]。补充先验说明与每个数据集的明细后可升到 5 |
| Evidence / 证据 | not assessed | not assessed | 尚无结果 | 协议阶段，没有可审的证据 |
| Significance / 意义 | 3 | 3 | Amendment 6 头条问题 | 决策价值明确，学术新增量有限 |
| Clarity / 清晰度 | 3 | 4 | [C002] [C003] [C004] [C006] | 并入修订、清理被取代的文本、写明对等读数后可升到 4 |
| Reproducibility / 可复核性 | 4 | 4 | Artifacts 段；v2-freeze.json；.gitignore | 扣分项 [C007] [C006]。探针证据进入冻结记录后可升到 5 |
| Ethics / Limitations / 伦理与局限 | 4 | 4 | Amendment 4 许可说明；Limitations 段 | 污染与 GPT-4 标注已声明；平衡先验这一局限缺失 [C001] |

**Overall:** 6  | **Scholarly Confidence:** 4

**Recommendation:** borderline（作为 Stage-1 协议：修复 [C001] [C002] 后可以冻结）

**Verdict:** 设计严谨，没有致命问题。决定分数高低的是新颖性（受限于公开工作）和两处估计对象的界定。在 6 与 7 之间取 6，决定因素是 [C001]：未声明的平衡先验会直接影响头条指标的解读。

| Change | Condition | Likely affected dimensions | Expected movement |
| --- | --- | --- | --- |
| Raise score | 写明平衡估计对象与 B-vs-B 对等读数，并入修订、清理被取代的文本，探针证据进入冻结记录 | Soundness, Clarity, Reproducibility | 可能从 6 升到 7 |
| Lower score | `prepare` 后发现排除或类别失衡严重改变主集构成且未报告 | Soundness, Evidence | 可能降到 5 |
| No quick change | 新颖性 | Novelty | 冻结前无法改变 |

## 13. 作者关键问题与改判条件 / Questions And Decision Conditions

1. 头条问题要回答的是"部署先验下"的问题，还是"平衡测试分布下"的问题？[C001] 如果是前者，需要冻结前加入按先验重新加权的描述性分析。
2. 头条的对等读数是否确定为 B-vs-B 加准确率？[C002]
3. macro-F1 的地位：secondary 还是 descriptive？[C003]
4. 冻结时是否把探针结果作为聚合报告提交，或把它的哈希写进冻结记录？[C007]

## 14. 修改优先级与复审记录 / Action Priorities And Re-Review

| ID | Priority / severity | Required change | Why it matters | Status / version |
| --- | --- | --- | --- | --- |
| [C001] | 高 / major | 协议声明平衡估计对象；可选按先验重新加权的描述性 Brier 与覆盖率；与公开结果对照时注明分布差异 | 头条"可自动化流量"的解读 | unresolved / 35568e0 |
| [C002] | 高 / major | 指定 B-vs-B 加准确率为头条对等读数，A-vs-A 为补充 | 头条问题的正确回答 | unresolved / 35568e0 |
| [C004] | 中 / minor | 冻结前并入修订，清理被取代的文本 | 预注册文本必须自洽 | unresolved / 35568e0 |
| [C003] | 中 / minor | 统一 macro-F1 的地位 | 避免误读确认性范围 | unresolved / 35568e0 |
| [C006] | 低 / minor | 注明"在可得量内平衡"；`prepare` 后记录实际类别计数 | 样本构成透明 | unresolved / 35568e0 |
| [C007] | 低 / minor | 探针证据进入冻结记录 | 能力证据可核对 | unresolved / 35568e0 |
| [C005] | 低 / minor | 报告每个数据集的差值，并说明跨 K 等权平均 | 效应量可解释 | unresolved / 35568e0 |
