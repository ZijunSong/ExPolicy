# A：何时保留经验，以及采用哪种同化方式

> 版本：2026-10-09；性质：新立项实验方案，尚未执行。依据《ExPolicy_Progress_Report_20261009.md》及本次拆分讨论。目标是尽快决定一个局部命题是否值得写成论文，不要求完成联合 ExPolicy。

## 1. 唯一主张与研究边界

**拟验证：固定经验采集和同化算子后，当前能力状态与经验内容能够预测同化方式的未来效用；这种预测在同一环境内部优于固定动作，并改善未见任务表现。**

最小动作集合为 `DISCARD / EPISODIC_REPLAY / SEMANTIC_MEMORY`。不训练 LLM，不学习 Acquisition，不引入新 Skill 生成器，不做联合策略。SkillOpt 保留在历史 R9 复核中；只有它能真实改变状态并产生正效用后，才允许进入新方法的动作空间。

成功需要同时回答：有选择空间吗？决策前可见信息能预测吗？预测能够带来独立未来任务上的收益吗？有多个动作成为 argmax，单独不能回答这三个问题。

本文件建立新协议 `A-v1`。原 R9 H1–H6 与 Gate J 保持原定义、原结论；本方案不把删掉动作后的结果补写为“原 R9 通过”。用户当前已经授权拆分独立方向，原报告“不改成单轴论文”的旧项目约束不再限制新立项，但历史实验不得重新命名为成功。

## 2. 可复用资产和当前限制

| 资产 | 当前已知状态 | 本轮使用方式 |
| --- | --- | --- |
| 主代码 | 报告路径 `/data/ppnm/expolicy` | 在用户机器上检查后复用；本文件没有审查仓库源码 |
| 模型 | Qwen3-14B，bf16，关闭 thinking，temperature=0 | 首轮固定；不因结果不理想临时换模型 |
| 环境 | LifelongAgentBench DB / OS / KG | 固定官方 verifier 和环境版本 |
| R9 batch | 36 个，每批 8 条；21 个完成四分支 | 只用完成且审计通过的真实产物 |
| SkillOpt | 已完成的 21 个分支全部拒绝更新 | 不是已验证有效动作；不能按它的偶然最高分解释编辑收益 |
| DB no-op | DISCARD 的 U 多次为 −0.049877 | 优先审计；不是先默认代码错误 |
| KG / OS | 已完成格子偏向 episodic | 检查是否仅有环境级选择空间 |

原模型清单、split、冻结 batch 分别位于 `outputs/r9_lab/model_manifest.json`、`split.json`、`FROZEN_BATCHES.json`。报告记录 split seed 为 `20260822`，SHA256 为 `63ca95237468378f5eb3a74df0875b23c38f00a2c4bf58200d4473df350bf989`。须验证文件实值，不用文档中的字符串代替现场检查。

## 3. 第一道门：评分是否可信

先抽取每环境 2 个既有 batch，执行以下审计；不补跑完整矩阵、不训练 Router。

1. 从同一个不可变 S0 创建 `BASE / DISCARD / REJECTED_EDIT`；逐项比较 memory、skill、retriever index、system prompt、工具配置、环境快照、模型与 tokenizer 版本。只比较一个 state.json 的 hash 不够。
2. 记录最终渲染 prompt hash、检索返回顺序、环境 reset ID、任务 ID、推理 seed、max tokens、服务并发和缓存键。排除分支名被写进 prompt、过期 cache、DB/OS/KG 状态串扰。
3. 在同一组 shadow / retention 题上配对评估；先确定性串行复放，再在正式并发下复放。temperature=0 不是可复现性的充分条件。
4. 同一输入重复 3 次只用于测量运行噪声，不能当作 3 个独立实验 seed。SkillOpt 拒绝后的部署状态如确实等价 no-op，性能应可比，但生成与验证成本可以更大。
5. 分别存储原始 `delta_shadow`、`delta_retention`、最终 U；检查基线是否错误地取了 calibration / foundation 的不同题集。

沿用 R9 的描述性效用：

\[
U=\Delta J_{shadow}-0.5\max(0,-\Delta J_{retention}).
\]

**重要：随机评估时，即使真实 no-op 没有效果，正部遗忘惩罚也可能使估计 U 的均值小于零。** 因而不能只用“平均 U 不等于零”判 bug。必须同时看配对原始差值、噪声分布和非线性惩罚偏差。不要逐行强减 DISCARD 值把结果校正成零；那会改变指标并可能掩盖错配。

通过条件：确定性重放得到完全相同的 no-op 行为和分数，或者已定位不可消除的随机性，且原始 no-op 差值与预注册等效区间相容。默认实用等效界为 ±0.01 成功率；若当前样本量无法检验，记 `MEASUREMENT_INCONCLUSIVE`，不能宣称通过。必须先修复系统性偏移。

历史产物仅在模型、prompt、任务列表、状态、评分器、环境版本一致时续用。若修复改变这些语义，要重跑受影响的配对分支，并保留旧结果为 legacy；不能把新旧列拼成一张“完整矩阵”。

## 4. 数据边界与样本单位

| 数据 | R9 DB / OS / KG 数量 | 允许用途 |
| --- | --- | --- |
| acquisition_pool | 183 / 183 / 140 | 收集固定批次经验；不按未来分数挑 batch |
| assim_validation | 55 / 55 / 43 | 算子候选接受与超参选择 |
| shadow_reward_probe | 55 / 55 / 43 | 开发期效用标签；允许指导策略训练，因此不是最终测试 |
| retention_probe | 37 / 37 / 29 | 开发期遗忘约束；与 shadow 不重叠 |
| future_test | 73 / 73 / 60 | 冻结策略后的最后评估，不参与选择 |

原 36×8=288 条是经验使用次数，不保证 288 个唯一任务；先查 batch 间重复、模板重合和数据来源。相同或同模板任务不得跨 Router train/validation/test 泄漏。共享 probe 的多个 batch 不构成独立 probe 样本。

R9 矩阵均从 S0 开始，因此它只能验证“给定经验 batch 的单次选择”。若没有多个不同能力状态，不能把结果表述为能力状态自适应。先完成 S0 局部结论；通过后再添加由固定历史序列生成的 2–3 个状态快照，不能用候选 Router 自己的成功轨迹选择快照。

历史 36 batch 仅作为 Pilot。新 Router 用按来源任务/模板分组的外层留出；拟合、归一化、动作阈值全部在内层完成。样本太少时优先 3 折分组交叉验证，不使用复杂网络。最终确认需要另收未见 batch；同一批经验换一个随机 seed 不是新增独立样本。

## 5. 第二道门：存在可利用的环境内选择空间吗

对每个 `x=(S,B)` 收集全部动作的配对逐题结果。平局或在噪声范围内的差异标为 `TIE`，不强行生成 hard label。

同时输出三个层次：

- `naive_hindsight_gap`：同一测量矩阵上先取 max 再平均，仅为描述，往往乐观。
- `crossfit_selector_gain`：在 probe 一半选择每个 x 的动作，在另一半评价，再交换；环境固定动作也必须在选择半上确定。这是独立评价的选择器收益，不是真实 oracle 上界。
- `feature_policy_gain`：在未见 batch 上，仅用决策前特征的 Router 收益；这是论文真正需要的量。

每环境 probe 小，拆半后约 21–28 题；无法稳健辨识 1–2pp 差异。Pilot 采用实际分辨率，默认实用门槛 `delta_space=max(0.02, 2/n_probe_eval)`，用每环境独立评价部分的题数计算，再按环境宏平均。此门槛是投入决策，不是统计定理。

`A-SPACE-GO` 建议条件：至少两个非 DISCARD 动作分别在 ≥4 个有效 batch 中有正效用；至少两个环境出现超出噪声的动作交叉；独立评价的选择收益超过预注册实用门槛且方向稳定。原 R9 的 H1–H6 另表报告。

`A-SPACE-NO_GO`：环境固定策略已基本达到可达到的水平；或“交叉”随独立重评消失。`INCONCLUSIVE`：区间太宽、环境重置不稳定、有效 batch 不足。只允许一次事先预算好的补样，不因希望出现正结果而反复扩样。

## 6. 最小方法：条件效用回归 + 保守选择

输入仅使用动作发生前可获得的信息：经验批成功率、错误类别、轨迹长度、重复率、与已有记忆相似度、冲突率、记忆规模、已知状态的历史能力摘要。能力摘要来自历史训练侧评估，不能在线读取未计费的新 probe。

先做每动作一个 ridge 回归，以连续 U 为监督；有充分样本才尝试浅层树。固定 embedding 时记录模型版本。batch ID、未来任务答案、候选动作执行后的 changed/accepted、shadow 分数、最终 test 指标均不得作为特征。

拟合 \(\hat U(x,a)\)，并用分组 bootstrap 估计动作相对于 fallback 的不确定性。fallback 是训练折选出的 env-only 最佳动作；它可能是 DISCARD。只有候选动作相对 fallback 的保守收益超过验证集冻结阈值时才切换，否则回退。若 fallback 本身在验证中没有可靠正效用，使用 DISCARD。

该选择器刻意轻量；贡献必须来自可泛化的决策信息，不能来自网络复杂度。主结果是效用/成功率而非 argmax 分类准确率。

## 7. 必须运行的对照

| 组 | 定义 | 排除的问题 |
| --- | --- | --- |
| B0 | 固定 DISCARD | 更新是否值得 |
| B1/B2 | 总是 episodic / semantic | 全局固定载体是否足够 |
| B3 | 每环境固定最佳动作，训练集选择 | 是否只在识别环境 |
| B4 | 按训练动作频率采样 | 是否只是动作频率优势 |
| B5 | 预先冻结的简单规则选择器 | 是否需要学习复杂效用 |
| M | 条件效用回归 + 回退 | 主方法 |
| A1 | M 去掉能力状态，仅经验特征 | 状态信息是否必要 |
| A2 | M 去掉经验内容，仅环境/长度等 | 是否真正利用经验差异 |
| A3 | 在环境内置乱效用标签 | 是否有可学习监督 |
| A4 | M 取消保守回退 | 收益是否来自避免负更新 |

先跑 B0–B3 与 M，其他为通过 Pilot 后的归因实验。对照组具有相同记忆容量、检索预算、推理 token 上限及算子版本。Router 的标签生成、模型训练和推理成本单独计入，不能只报部署便宜。

与 MCMA 的直接比较应在官方接口可兼容、训练数据和教师预算匹配后进行。若采用简化适配，标明 `MCMA-style adaptation`；不要以无法运行的残缺实现代表原论文。ASSAY 对应任务时 skill 抑制，并不自动等同于写入时的载体选择。

## 8. 从单次决策到状态更新

S0 Pilot 通过后，固定 Acquisition 序列，做 4 轮×每轮 8 条经验；所有策略从相同 S0 开始，允许持久状态随后自然分化。在第 0/2/4 轮离线评估；测试分数不回流。

优先比较 M、env-only、best-global、DISCARD。训练中若仅见 S0，先报告分布外状态退化；不得把 teacher-forced 矩阵中的收益当成真实顺序运行收益。必要时新增由固定策略产生的训练状态，不启动 Acquisition 联合优化。

主要指标：future-test 宏平均成功率、相对 env-only 的配对差、retention 原始差值、更新接受率、负迁移率、选择成本与累计推理资源。对连续成功率报告 pp；所有原始 U 与带惩罚 U 同时保存。

## 9. 统计、预算和决策

按来源任务/模板与 probe 任务做分组或交叉聚类重采样，不能把共享 probe 的全部格子当独立样本。正式顺序实验至少 3 个真实不同的经验序列；5 个为资源允许时的建议，确定性重复不算新 seed。主比较 M vs env-only 预先指定；次要多重比较做校正或明确探索性。

Pilot 估算：36 batch×3 动作×约 80–92 个 shadow+retention 任务，约 8,640–9,936 次 episode；不含获取经验、重复测量和算子验证。拒绝更新与 DISCARD 在严格等价时可共享性能缓存，但仍记录真实算子成本。先实测 20 episode 的 tokens、工具调用和墙钟，再批准剩余预算；不凭 GPU 数给出耗时承诺。

正式方法 GO 的建议标准：在独立 batch/test 上相对 env-only ≥2pp，配对区间下界 >0，遗忘不超过预注册 1pp 容忍度，并超过简单规则。现有约 200 个 test 任务可能不足以证明 2pp；用 Pilot 的配对方差评估精度，只扩未见任务，不反复查看 test 调策略。达不到精度记证据不足，不能将 p>0.05 等同无效。

| 时间盒 | 动作 | 交付 |
| --- | --- | --- |
| 第1–3天 | no-op 审计和产物清点 | `audit.md`、有效格子索引、资源估计 |
| 第4–7天 | 补必要矩阵、独立选择评估 | 原 R9 Gate H 与 A-v1 判定分开 |
| 第8–10天 | ridge Router + 最强固定对照 | 分组留出结果 |
| 第11–14天 | 若通过，做短序列确认 | GO / NO_GO / INCONCLUSIVE |

两周是可行性决策窗口，不保证完成投稿规模。若无环境内选择空间，停止 A，保留结果供评测方向 D；不要继续换更强 Router。

## 10. 交给代码助手的实施清单

以下是待实现接口与产物建议，不声称仓库已有这些文件或命令。先读仓库实际入口，复用现有 clone/eval/日志设施，不重写整个 runtime。

- [ ] 新建独立实验配置 `configs/paper_a_v1.yaml`，锁定模型、split、动作、metric、预算和预注册门槛。
- [ ] 实现 `audit_noop(snapshot, tasks)`，输出逐题配对差、状态与 prompt hash、惩罚分解。
- [ ] 实现 `build_utility_matrix(batches, actions)`，以完整配置 hash 为缓存键；失败标为 missing，不填 0。
- [ ] 实现 `split_by_source_group()`，先做来源去重，再做外层/内层划分。
- [ ] 实现 `crossfit_action_selection()` 和 `fit_ridge_router()`，保留平局，不泄漏 probe 标签。
- [ ] 实现 `run_fixed_acquisition_sequence()`，所有策略共享 acquisition 顺序，状态独立。
- [ ] 每个 run 生成 manifest、逐题结果、费用台账、失败清单和不可变 summary。

推荐行字段：`protocol_id, model_hash, env, source_group, batch_id, state_hash_before, action, state_hash_after, changed, accepted, probe_task_id, eval_split, y_before, y_after, delta_shadow, delta_retention, utility, tokens, tool_calls, elapsed_s, infra_status`。

完成物：`A_MEASUREMENT_AUDIT.md`、`A_UTILITY_MATRIX.csv`、`A_CROSSFIT_RESULTS.csv`、`A_ROUTER_RESULTS.csv`、`A_COST_LEDGER.jsonl`、`A_DECISION.md`。最后一个文件必须解释现有证据能支持哪一级主张，而不是只写 PASS。

## 11. 相关工作核查与新颖性边界

截至 2026-10-09 核查的官方页面：

- [LifelongAgentBench](https://arxiv.org/abs/2505.11942)：实验环境与 lifelong 评测背景。
- [MCMA](https://aclanthology.org/2026.findings-acl.1535/)：已有可学习记忆管理与抽象；“学习如何管理记忆”不能单独作为新颖性。
- [ASSAY](https://arxiv.org/abs/2606.15390)：已有任务条件 skill 效用估计与抑制；需比较决策时点、状态、动作空间及监督成本。
- [SkillOpt](https://microsoft.github.io/SkillOpt/)：已有轨迹驱动技能编辑与验证门控。

本文件提出的独立价值仍是假设：在写入经验时预测不同持久化干预的净未来效用，且这种优势无法被 env-only 或简单规则解释。正式投入前用半天阅读全文和官方代码，形成逐项差异表；仅靠命名或将多个已有模块串联，不构成已确认的新贡献。
