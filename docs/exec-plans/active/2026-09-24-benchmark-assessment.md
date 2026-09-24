---
status: in-progress
created: 2026-09-24
---

# 评估：Jev / Laya / Qwen-logit benchmark 思路（ccf-common 路由）

被评对象：`docs/exec-plans/active/2026-09-23-jev-laya-benchmark.md`（locked plan v2，含 Amendment 3/4）与 `docs/PROTOCOL-v2.md`（draft，尚未 `v2-preregistered`）。
路由：ccf-humanization → ccf-common → `ccf-idea-reviewer`（思路靠谱/值得做/创新/评分排序，主责）＋ `ccf-literature-searcher`（公开 benchmark 与结果）＋ `ccf-experiment-designer`（用户明确要求的实验设计扩展，单列，不计入概念分）。
共识机制：claudex-loop，Claude 起草，Codex 只读复审至 APPROVED。单 agent 起草，非独立多审团。

## Goal / 验收标准

- 回答四问：思路靠谱吗、值得做吗、创新够不够、评分排序；附公开 benchmark 与结果对照表（带链接与出处性质）。
- 每个关键判断有来源或仓库文件锚点；不把公开数字当本研究结果。
- Codex 对本文件最终 SHA256 给出 APPROVED（或达到 5 轮上限时如实列出分歧）。
- 不改 benchmark 代码；设计修改只作为建议，待用户拍板。

## 1. 结论先行

| 问题 | 判断 | 置信度 |
| --- | --- | --- |
| 靠谱吗 | 方法学上靠谱：预注册、钉版本、calibration/test 分离、失败计入、策略配对对比、Holm 冻结 k。执行层 bug（D1 Laya option 截断、D2 Qwen 前缀、Laya 计时包装器参数）已在冻结前修复（`4becb91`、`8720bb5`）。 | 中高 |
| 值得做吗 | 对用户自己的决策（本地能否平替 Jev）值得：成本 < $2，M1 可跑，得到同批次、同条目、受控的三方证据，这是公开材料里缺的。作为可发表贡献价值有限。 | 中 |
| 创新够不够 | 不够（概念新颖度 2/5）。Jev vs Laya 零样本+校准（elcronos）、Qwen3-1.7B logit vs Jev vs Laya（open-alternative-jev）、logit+少标签 head、flip rate、5% 风险覆盖率（AnyJev）、Jev flip/Banking/SMS（nibzard）均已有公开结果。剩余差异是"同一预注册运行里的三方×三题型×对称校准"组合，属于整合与对照控制，不是新问题或新机制。 | 中（公开来源多为项目自报 README/博客） |
| 评分排序 | 概念加权分 3.20/5 → `pivot-with-rescue-route`。救法排序：V2（加少标签臂，有条件：需特征产物与 cross-fitting 契约）> V1（加参照基线）≈ V0（现状）> V3（只复现 elcronos）。 | 中 |

## 2. 公开 benchmark 与结果（ccf-literature-searcher）

覆盖：`searched`（单轮 web 检索 + 8 个一手页面抓取 + 本地 wiki 已入库来源）。未做同行评审文献的系统检索；无 MDPI 来源。表中数字均为**公开来源自报**，不同协议、样本、提示词，不可横向当排行。

| 来源 | 性质 | 参赛者 | 任务 / 规模 | 关键公开数字 | 与本研究重叠 |
| --- | --- | --- | --- | --- | --- |
| [elcronos/jev-vs-open-decision-models](https://github.com/elcronos/jev-vs-open-decision-models) | 第三方，冻结协议，原始预测公开 | Jev 1.13（OpenRouter）、Laya、PrismNLI-0.4B；LR/TF-IDF 监督基线 | choice only；emotion 2000、tweet_topic 1693、fin_topic 4117、daily_dialog 7740 | emotion：Jev 0.587 / Laya 0.587 / PrismNLI 0.725 acc；Jev–Laya McNemar p=1.000；监督 LR 在每个数据集 acc 高 5.5–15.8 点；MPS p50 Laya 31 ms、Jev 349 ms | **最高**：同一 Jev 通道、Laya、校准指标、配对检验、监督基线。缺：LLM logit、score/noul、零样本 T-scaling、置换 |
| [ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) | logit 方案实现者自报 | Qwen3-0.6B…27B logit、Laya base/微调、Jev 1.13 | typed-decisions 400 例 / 2000 决策；RACE-H；MMLU | Qwen3-1.7B 45.9% acc、ECE 0.509；Qwen3.6-27B 73.7%、ECE 0.020；Jev 72.7%、ECE 0.144；Laya base 36.0%、ECE 0.176 | **高**：正是"最后层 option token logit + softmax"路线，且含 1.7B。Jev 数字疑似转引（与 Laya 报告同值），非同批次 |
| [nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev) | 项目自报 | Qwen3-1.7B…32B + 闭式 head（~300 标签） | typed-decisions、BANKING77 20-way 300 例 | Qwen3-1.7B+head 0.730（≈Jev 0.727）；BANKING77 flip 0.230→0.073、ECE 0.240→0.095、5% 风险可自动覆盖 7.7%→52.0% | **高**：flip rate、5% 风险覆盖率、少标签 head 均已做 |
| [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark) | 第三方，花费与日志公开 | Jev + 8 个约束 LLM + 确定性基线 | Banking77、SMS Spam、255 选项、置换 | Banking：Jev 76.3%、GPT-oss-120b 81.3%；Jev 置换改答 13%（LLM 最差 37%）；Jev ECE 0.246 | 中：置换、Banking/SMS。无本地小模型、无 Laya |
| [Anthus: Jev vs Laya](https://anth.us/blog/jev-vs-laya/) | 第三方 | Jev、Laya（laya-mlx 0.1.0）、DistilBERT | 自建情感 600 例；noul/choice/score | Jev 0.768、Laya 0.722（零样本）；微调 Laya 0.896；指出 Laya 静默截断、192 token 共享 option 预算 | 中：三题型、Laya 截断问题（与本研究 D1 同源） |
| [JevBench v1.0](https://benchmarkheaven.com/jev-models/v1) | 第三方排行 | Jev、GPT-5.6 Luna、DeepSeek、开源 Jev 仿制品等 | 242 决策/系统 | GPT-5.6 Luna 97.1% 最高；Jev 最便宜 $0.027/1k | 低：无本地小 logit、无 Laya base |
| [Luni/laya-jev-benchmark](https://huggingface.co/datasets/Luni/laya-jev-benchmark) | 数据集 | Laya、Jev、Claude Haiku 4.5 | 钓鱼 2000、typed-decisions | 钓鱼校准后 Laya 0.611 vs Jev 0.626 | 低 |
| [BTZSC（ICLR 2026）](https://arxiv.org/abs/2603.11991) | 同行评审 benchmark | 38 模型：NLI、embedding、reranker、4–12B LLM | 22 数据集零样本分类 | reranker macro-F1 0.72 最高；LLM 4–12B 至 0.67 | 数据来源（本研究用其 AG News/Emotion/Banking77）；不含 Jev/Laya |
| upstream pilot（本仓库 `results/reports/btzsc-pilot-v1.json`） | 本 fork 上游 | Jev、GLiNER2.5 | BTZSC 3×100 | Jev acc AG 0.91 / Banking 0.87 / Emotion 0.48；Emotion 上 16% 真标签概率为 0 | 直接前身 |
| Laya 自报（wiki：Flowtivity、Artifilog、dev.to） | 项目/博客转述 | Laya 三检查点 vs Jev | typed-decisions、AG、Emotion、Banking77、MASSIVE | typed 微调 0.766 vs Jev 0.727；零样本 base 0.362；ECE raw 0.466→refit 0.081；Banking77 0.425 vs Jev 0.870 | 背景；"Jev 无公开多语 benchmark"（Artifilog） |

方法学先例（非 Jev 领域）：[Batch Calibration, ICLR 2024](https://arxiv.org/abs/2309.17249) 等说明 label-token logit 分类存在上下文偏置、需校准；本研究的 condition B（温度缩放）是其中最简单的一种，未测 BC/contextual calibration。

**由公开证据得到的先验（非本研究结果）**：零样本下 Jev 领先 Laya-base 与 Qwen3-1.7B-logit（typed-decisions 同表：Jev 72.7%、Qwen3-1.7B 45.9%、Laya base 36.0%；但 elcronos 的 emotion 上 Jev 与 Laya 持平 0.587）。Laya-base 与 Qwen3-1.7B 之间的排序**未定**：唯一同表证据（typed-decisions）是 Qwen 高于 Laya，本研究的英文 choice 数据集上无公开对照；原始校准 Qwen3-1.7B 最差（ECE≈0.51）；温度缩放后差距预计缩小最多的是 Qwen；少量标签（≈200–300）后，Qwen-1.7B+head 或 LR 基线可能追平/超过零样本 Jev（AnyJev、elcronos）。

## 3. 问题定义与研究价值

问题具体：同一条目上，托管 Jev、开源 Laya、本地小 LLM logit 三条"typed decision + 概率"路线，在判别、概率质量、选择性自动化、顺序敏感、重复方差、延迟上的差别。受益者：需要决定自托管还是调用 Jev 的开发者（首先是用户本人），以及 wiki Q165–Q192 的开放问题。价值主要是**决策证据**而非新科学知识：结论对硬件、提示词、数据集选择敏感。

## 4. 核心洞察与贡献拆解

问题 → 洞察 → 机制 → 贡献：
- 洞察 1（有价值）：比较"内置校准"不能看 T，要看同一 test 上 A vs B 的差（R2#3 政策配对）。公开 benchmark 多把零样本 ECE 与微调/refit 后 ECE 混比（Laya 0.081 vs Jev 0.246）。
- 洞察 2（有价值）：Jev 本身非确定，置换 flip 需减去同顺序重复 flip 才是顺序效应（excess order effect）。nibzard 的 13% 未扣除重复噪声。
- 洞察 3（执行层）：Laya 必须在不截断 option 的 head 下比较（D1），Anthus 已观察到静默截断。
- 贡献是这三个控制的**组合**加上同批次三方运行，而非新机制。

## 5. 最近工作与创新差异

扣除 §2 已知贡献后剩余：
1. 同一预注册运行、同条目的三方（Jev 托管 / Laya 本地 / Qwen-1.7B logit）比较，并覆盖 choice + noul + score。公开来源最多两方同批，或 Jev 数字转引。
2. 所有参赛者（含 Jev）对称的 calibration split 温度缩放，bootstrap 内重拟 T。
3. 扣除重复噪声的置换效应；Jev 在 zh/km 的多语结果（公开空白）。
4. 不截断 Laya head 的受控设置。
判断：`crowded but open`。中心问题"logit+softmax 能否平替 Jev"已被 open-alternative-jev/AnyJev 在 typed-decisions 上部分回答（零样本 1.7B 明显落后；加 head 后接近），本研究的新增是**在公开数据集、同批次、对称校准下复核**。

## 6. 机制、假设与逻辑

- 自洽：preregistration + 冻结 k + 失败惩罚 + 策略配对，无内部矛盾。
- 假设风险 A（D3）：Qwen 路线只有 1.7B 单尺寸、单提示，零样本。公开证据显示尺寸与少标签 head 是主导因素（0.6B→27B：29%→74%；1.7B+head 0.730）。因此"Qwen-1.7B 零样本输"不能推出"logit 路线不能平替"。
- 假设风险 B（D5）：UltraFeedback 标签是 GPT-4 评分，LLM 系参赛者有同源偏差优势；score 结论只能读作"与 GPT-4 评分一致度"。
- 假设风险 C：Laya 在 MASSIVE 用 head≈587–591，高于出厂 256，属分布外设置。
- 非矛盾、仅未验证：Jev 在 OpenRouter 上的快照稳定性（已有单快照规则）。

## 7. 主要优势与可保留成分

1. 协议纪律（`v2-probe`/`v2-preregistered` 两道 tag、hash 校验、attempt 命名空间）高于公开同类，可直接复用。
2. 失败计入分母并给最差惩罚（Brier 2、NLL −log ε），避免"只算成功调用"的乐观偏差。
3. A/B 策略配对 + T 重拟 bootstrap，正面回答"内置校准"这一 Jev 核心卖点。
4. 数据许可审查（ToxicChat、Yelp 替换）——托管推理把原文发给第三方，公开 benchmark 普遍未处理。
5. 探针已抓到三个真实执行 bug（D1、D2、计时包装器），说明分阶段探针有效。

## 8. 主要问题（稳定 ID）

| ID | 问题 | 依据 | 影响 | 严重度 | 最小修复 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 新颖度低：中心比较已被公开覆盖 | §2 elcronos / open-alternative-jev / AnyJev | 无法作为"首次"结果陈述 | 中 | 定位改为"同批次受控复核 + 三个控制"；报告中逐条对照公开数字 | 待用户确认定位 |
| C2 | Qwen 路线只测零样本 1.7B（=D3） | open-alternative-jev 尺寸曲线；AnyJev head | 对"平替"问题可能给出过悲观答案 | 中 | 结论措辞限定到"Qwen3-1.7B 零样本"；或加 V2 少标签臂 | 待决 |
| C3 | 缺参照基线（=D4） | elcronos：LR 每个数据集赢零样本 5.5–15.8 点 | 平替问题的实际答案可能是"200 标签+线性模型" | 中 | 加解析先验基线（Brier 1−1/K）与 calibration 200 条上的 LR/embedding 基线（描述性） | 待决 |
| C4 | UltraFeedback 标签源混淆（=D5） | Amendment 4：GPT-4 标注 | score 对比偏向 LLM 系 | 低-中 | UltraFeedback 不入检验家族，仅描述 | 待决 |
| C5 | probe 的 GLiNER `excluded_datasets` 预览按全池计（=D6，已收窄） | `probe.py:208-225` 全池；但权威判定在 `data.py:429-469` 按抽中条目写入 manifest summary，`v2_runner.py:208-216`、`v2_report.py:33-42` 均读该 summary | 仅 probe 输出会误显示 Civil/UltraFeedback 的 GLiNER N/A；正式运行不受影响 | 低 | probe 字段改名或注明"全池预览"，PROTOCOL 引用 manifest summary | 待修（文档级） |
| C6 | 组件偏多：5 参赛者×9 数据集×3 附加套件＋GLiNER anchor | plan P1–P5 | 工期与多重比较负担，稀释核心问题 | 低 | 保留检验家族；latency-10、typed-decisions 检查点、GLiNER 明确标为描述性并可裁剪 | 建议 |
| D1 | Laya option 静默截断 | `laya/common.py:91-95` | 主设置违反"不截断" | 高 | `laya_head_fits` + 真 `build_sequence` 契约测试 | **已修** `4becb91` |
| D2 | Qwen 字母 ID 前缀不兼容 | Qwen 分词实测 | 延迟虚高 K 倍 | 中 | `qwen_option_ids` 分前缀 | **已修** `4becb91`，待 probe 复核 |
| D7 | Laya 计时包装器参数数错误 | 首次真实 probe TypeError | 所有 Laya 调用崩溃 | 高 | `*args, **kwargs` 透传 | **已修** `8720bb5`，probe 复跑中 |

## 9. 多视角综合（单 agent 角色视角，非独立评审）

- 领域视角：问题对实践者有用，答案具有部署特定性。
- 先行工作视角：C1 决定性；不能宣称首次。
- 机制视角：协议无硬伤；C2/C4 是构念效度问题，可用措辞或小改修复。
- 贡献视角：最有信息量的增量是 V2（零样本 vs 少标签，同批次对 Jev），而不是再多一个零样本数据集。
综合：最强未决问题是 C1+C2，二者共同指向同一救法（加少标签臂、调整定位）。

## 10. 六维评分（概念，1–5）

| 维度 | 权重 | 分 | 依据 | 改判条件 |
| --- | ---: | ---: | --- | --- |
| 问题重要性与具体性 | 20 | 4 | §3 | 若结论被证明高度依赖硬件/提示而不可迁移则降 |
| 相对最近工作的新颖度 | 25 | 2 | C1，§2、§5 | 加 V2 且公开无同批次对照 → 3 |
| 概念洞察 | 20 | 3 | §4 洞察 1–2 | 若 excess order / 策略配对校准结论与公开结论相反并可解释 → 4 |
| 机制与逻辑自洽 | 20 | 4 | §6 | C2/C4 未处理维持 4；出现未修执行 bug 降 |
| 简洁与组件必要性 | 10 | 3 | C6 | 裁剪描述性套件 → 4 |
| 受众与贡献匹配 | 5 | 4 | §3 | — |

加权 = (4·20+2·25+3·20+4·20+3·10+4·5)/100 = **3.20**，覆盖 100%。按 calibration.md 落入 `pivot-with-rescue-route`（3.0–3.7）。发展潜力：**高**（一个具体修改即可保留中心问题）。置信度：**中**（公开证据多为项目自报；单轮检索）。

注意：该分数衡量"作为研究贡献"的概念质量；"对用户自身决策是否值得跑"见 §1，答案为值得。

## 11. 评分排序：候选方案（同一 rubric、同一范围）

| 排名 | 方案 | 内容 | 新颖 | 洞察 | 简洁 | 加权 | 推荐 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | V2 零样本 + 少标签臂（有条件） | 现设计 + 每数据集少标签分类器（Qwen-1.7B 答案位隐藏状态 linear probe；TF-IDF/embedding LR），与零样本 Jev 同 test 对比 | 3 | 4 | 3 | 3.65 | **首选**（仍在 pivot 带上沿；主要提升在决策相关性） |
| 2 | V1 现设计 + 参照基线 | 加解析先验基线 + LR 基线（描述性），UltraFeedback 出检验家族 | 2 | 3 | 3 | 3.20 | 次选，最小改动 |
| 3 | V0 现设计 | 按 locked plan 执行 | 2 | 3 | 3 | 3.20 | 可行但对平替问题信息量最低 |
| 4 | V3 只复现 elcronos | 仅 Jev vs Laya choice | 1 | 2 | 4 | 2.85 | 不推荐 |

其余三维（问题 4、机制 4、受众 4）四方案相同，加权按 §10 公式计算。V1 与 V0 概念分相同；V1 排前是因为 C3 直接影响"平替"答案的可解释性（决策相关性差异，非分数差异）。V2 的少标签臂属于计划修订，需要用户批准、单独一轮计划复审，并在 `v2-preregistered` 前写入 PROTOCOL。现有产物无法直接得到 V2：Qwen 适配器只返回 option 概率，`Prediction` 不存隐藏特征（`src/jev_benchmarks/adapters/qwen_logit.py`、`src/jev_benchmarks/models.py`），且 PROTOCOL 把 200 条 calibration 标签留给温度拟合（`docs/PROTOCOL-v2.md` Conditions 段）。V2 的最小契约：

1. 特征：固定层、固定位置（`Answer:` 后预测位）的隐藏状态，另存为带 hash 的版本化特征产物（不进 git）。
2. 数据：200 条 calibration 做 5 折 cross-fitting：probe/LR 在 4 折训练，温度在折外预测上拟合；test 不参与任何拟合。
3. 推理/报告：新增 `condition=C_fewshot`，只作描述性对比（或另开检验家族并重新冻结 k），不改 C1–C3。
4. 成本：一次额外前向取特征（与 letter 模式同一次前向可复用）。

若不愿承担上述改动，V1（TF-IDF/embedding LR 同样用 cross-fitting，无需模型特征）是 V2 的低成本子集。

## 12. 发展优先级

| ID | 优先级 | 修改 | 保留/修复 | 改判条件 |
| --- | --- | --- | --- | --- |
| P1 | 高 | 报告定位改为"同批次受控复核"，逐条对照 §2 公开数字 | 修 C1 表述风险 | — |
| P2 | 高 | 决定是否采用 V2 少标签臂（冻结前） | 修 C2/C3 | 采用 → 新颖度 3 |
| P3 | 中 | UltraFeedback 出检验家族；检验家族 = AG News、Emotion、SMS、Civil，k=9 | 修 C4 | — |
| P4 | 低 | probe 的 GLiNER 排除字段注明为全池预览；正式判定已在抽样后 | 修 C5 | — |
| P5 | 中 | 把 §2 先验写进 PROTOCOL 的"预期"段（不改双侧检验） | 让结果可对照先验 | — |
| P6 | 低 | 裁剪或明确降级描述性套件 | 修 C6 | — |

## 13. 实验设计扩展（用户要求，单列，不计入概念分）

检验家族建议：三方交集英文数据集 AG News、Emotion、SMS Spam、Civil Comments（Laya 在 Banking77 N/A：72 个完整 option 需约 1,280 token head，超过 1,024 上限）。C1–C3 保留，k=9。UltraFeedback 描述性。敏感度（替代原"可检出 ≥4 pp"表述）：配对准确率差的单数据集 SE = √(d/300)（d = 两模型答案分歧率，忽略 δ² 项），4 个数据集等权平均 SE = 单集 SE/2。Holm 家族 k=9 时最小 p 需 < 0.05/9，双侧 z≈2.77；80% 功效的最小可检出差 ≈ (2.77+0.84)·SE（首个检验；后续 Holm 步骤阈值放宽）。

| 分歧率 d | 平均 SE | MDE（Holm 首检，80% 功效） | MDE（未校正 α=0.05） |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.9 pp | 3.3 pp | 2.6 pp |
| 0.20 | 1.3 pp | 4.7 pp | 3.6 pp |
| 0.30 | 1.6 pp | 5.7 pp | 4.4 pp |
| 0.50 | 2.0 pp | 7.4 pp | 5.7 pp |
| 1.00（上界） | 2.9 pp | 10.4 pp | 8.1 pp |

本研究参赛者间的分歧率未知（公开来源未报告），表格给出范围；数值为正态近似，实际推断用 bootstrap。结论：约 4–6 pp 以下的平均准确率差应表述为"未分辨"，而非"无差异"。分歧率在 pilot 后可观测，但不得据此改设计。

## Assumptions / 风险

- 公开数字来自 README/博客自报，未复算；Jev 数字在多个仓库中疑似转引同一来源。
- 检索为单轮；可能漏掉未索引的仓库或论文。
- 评分为单 agent 判断，Codex 复审是唯一外部校验。

## Verification

- Codex review 结果文件（runner `review` 模式）verdict=APPROVED，绑定本文件 SHA256。
- 所有 §2 链接可访问（Codex 可抽查仓库本地文件锚点；web 链接由 Claude 已抓取，见 review log）。

## Progress Checklist

- [x] ccf 预检：humanization → common，路由到 idea-reviewer + literature-searcher + experiment-designer 扩展
- [x] 公开 benchmark 检索与一手页面抓取（elcronos、nibzard、Anthus、JevBench、Luni、AnyJev、open-alternative-jev、BTZSC、Batch Calibration）
- [x] 起草评估文档
- [x] Codex 复审第 1 轮（REVISE，R1–R4 全部接受并修订）
- [ ] 处理 Codex 意见并复审至 APPROVED（≤5 轮）
- [ ] 向用户呈现共识结论与待决项（P2 V2 少标签臂、P3 UltraFeedback、C1 定位）
