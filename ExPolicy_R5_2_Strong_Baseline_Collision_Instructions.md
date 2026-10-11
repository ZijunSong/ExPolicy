# ExPolicy 下一轮实验指令：R5.2 Strong-Baseline Collision Test

> 目标：在继续优化方法之前，先回答一个最关键的问题：**当前 `frontier acquisition → experience-gated executable skill activation` 是否能显著优于最新、最接近的主动采样方法？**
>
> 当前阶段不要继续扩展 learned carrier router、LoRA assimilation 或更多 Helion 内部消融。R5.1 已经足够说明：当前有效机制主要是 **experience-gated skill activation**，而不是 content-dependent assimilation。
>
> Repo：`/data/ppnm/expolicy`
>
> GPU：如果 H20 `4,5,6,7` 当前仍空闲，则仅使用这四张卡；运行前先检查占用情况，不要抢占其他任务。

---

## 0. 本轮必须保持不变的结论与冻结项

先阅读并复用：

- `outputs/r5_acquisition_skill/`
- `outputs/r5_1_robustness/`
- `scripts/run_r5_suite.py`
- `scripts/run_r5_1_robustness.py`
- `scripts/analyze_r5_1_robustness.py`

冻结以下设定，不允许为了新 baseline 调整：

1. Base model：`Qwen2.5-3B-Instruct`
2. Helion world seed：13
3. R5/R5.1 seed 0–9 的 frozen split
4. Future probe：36 tasks
5. Retention：24 S0 tasks
6. Acquisition budget：48 rollouts
7. R2 已验证的三个 executable procedure primitive
8. Skill activation threshold：primary 固定 `K=8`
9. Skill activation 机制：**CountOnly / evidence-count gating**
10. 任何 future/test task、gold answer、gold doc id 不得进入 acquisition scoring 或 gating
11. 不训练 R6 learned carrier router
12. 不重新做 episodic / semantic / LoRA carrier 主实验
13. 不把 RealFit 重新包装成主要贡献

本轮 primary question 只有一个：

> 在完全相同 rollout budget 与完全相同 skill activation 后端下，当前 `frontier` acquisition 是否优于最新 strongest acquisition baseline？

---

# 1. 第一优先级：实现 Ace-Skill Prioritized Sampler baseline

这是本轮必须完成的 baseline。

论文：

- Ace-Skill: Bootstrapping Multimodal Agents with Prioritized and Clustered Evolution
- arXiv: https://arxiv.org/abs/2605.08887
- code: https://github.com/AMAP-ML/Ace-Skill

不要把整个 Ace-Skill 的 clustered memory/skill organizer 搬进来，因为那会同时改变 assimilation backend，失去 acquisition-only 公平比较。

只复现它的 **Prioritized Sampler**，后端仍统一接我们的 frozen CountOnly skill gating。

## 1.1 精确实现 sampler

对 acquisition pool 中每个 task \(x_i\)，维护 proficiency posterior。

使用论文定义：

\[
\tilde{\alpha}_i = \alpha_i \rho^{t-t_i},\qquad
\tilde{\beta}_i = \beta_i \rho^{t-t_i}
\]

\[
v_i = \frac{1+\tilde{\alpha}_i}
{2+\tilde{\alpha}_i+\tilde{\beta}_i}
\]

sampling weight：

\[
w_i \propto
\sqrt{v_i(1-v_i)}
+\gamma(1-v_i)
+\epsilon
\]

使用论文默认超参数：

```text
rho = 0.95
gamma = 0.4
epsilon = 0.1
```

rollout 得到 reward \(r_i\in[0,1]\) 后：

\[
\alpha_i \leftarrow \tilde{\alpha}_i+r_i
\]

\[
\beta_i \leftarrow \tilde{\beta}_i+(1-r_i)
\]

\[
t_i \leftarrow t
\]

### 初始化

按论文语义初始化 Beta proficiency statistics。

如果论文正文与官方代码在初值写法上存在细微差异：

1. 以官方代码为最高优先级；
2. 将差异写入 `ACE_SKILL_IMPLEMENTATION_AUDIT.md`；
3. 不自行调参选择更有利版本。

---

# 2. 公平比较协议

实现以下 acquisition 方法：

```text
passive
frontier
ace_skill_prioritized
reward_seeking        # 只作为已有负控
uncertainty           # 复用已有结果/实现
failure_replay        # 复用已有结果/实现
```

其中本轮真正需要新跑和比较的是：

```text
frontier
vs.
ace_skill_prioritized
vs.
passive
```

已有 R5/R5.1 的 `frontier/passive` 若 protocol 完全兼容，应尽量复用；不要无意义重复耗算力。

---

## 2.1 Ace-Skill 的重复采样公平性

Ace-Skill 原方法允许同一 sample 在不同 step 被再次访问，因此不能强行改成永久 no-replacement，否则就不是 faithful baseline。

但我们的 skill gating 不能因为重复刷同一道题而虚假累计 evidence。

因此区分：

```text
rollout_budget = 48
unique_evidence_count = 按 capability 去重后的 task-id 数量
```

### 强制规定

Skill activation 的 `K=8` 必须按：

```text
>= 8 distinct acquired task IDs from that capability
```

而不是：

```text
>= 8 rollout observations
```

这样：

- frontier/passive 原本基本无重复，不受影响；
- Ace-Skill 允许 faithful revisit；
- 但不能靠重复一个 task 人为快速解锁 skill。

必须同时记录：

```text
total_rollouts
unique_tasks
duplicate_rollouts
unique_tasks_per_capability
```

---

# 3. Rollout 组织

统一将 budget=48 分成：

```text
6 rounds × batch size 8
```

每个 round 后更新 acquisition statistics。

评测 checkpoint：

```text
budget = 0 / 8 / 16 / 24 / 32 / 40 / 48
```

如果完整 Future probe 每 8 rollout 的成本过高：

- `0 / 16 / 32 / 48` 做 full future probe；
- 其余 checkpoint 至少记录 skill activation、capability coverage 和 acquisition statistics。

禁止改变最终 `budget=48` 主比较。

---

# 4. Seeds 与并行

必须使用与 R5.1 相同的：

```text
seed = 0,1,2,3,4,5,6,7,8,9
```

不得筛 seed。

建议 GPU 分配：

```bash
CUDA_VISIBLE_DEVICES=4  # seeds 0,4,8
CUDA_VISIBLE_DEVICES=5  # seeds 1,5,9
CUDA_VISIBLE_DEVICES=6  # seeds 2,6
CUDA_VISIBLE_DEVICES=7  # seeds 3,7
```

实际脚本应支持 stall resume，不覆盖已有 R5/R5.1 产物。

输出目录：

```text
outputs/r5_2_baseline_collision/
```

建议新建：

```text
scripts/run_r5_2_baselines.py
scripts/analyze_r5_2_baselines.py
```

不要修改旧结果文件。

---

# 5. Primary evaluation

Primary 后端固定为：

```text
acquisition policy
    ↓
distinct capability evidence count
    ↓
K=8 eligibility gate
    ↓
frozen executable primitive
    ↓
future evaluation
```

也就是说，本轮比较的是 **acquisition policy 本身**。

不要重新启用 RealFit。

---

# 6. Dispatch

为了隔离 acquisition：

### Primary

使用 R5/R5.1 的：

```text
oracle capability dispatch
```

### Secondary sanity check

复用：

```text
text_router
```

只需要确认：

```text
frontier > baseline
```

的排序在 text-router 下不反转。

不要再花大量实验研究 Helion text-router，因为 R5.1 已经显示其模板可分、接近 oracle，这不是当前贡献重点。

---

# 7. 本轮必须新增的指标

不要只输出 final Future EM。

## 7.1 最重要的主指标

每个 method × seed 输出：

```text
Base Future EM
Final Future EM
ΔFuture
Retention EM
ΔRetention
```

比较：

```text
frontier - ace_skill_prioritized
frontier - passive
ace_skill_prioritized - passive
```

使用 paired seed comparison。

---

## 7.2 Sample-efficiency 指标

必须输出：

### A. Area Under Learning Curve

基于：

```text
budget 0 / 8 / 16 / 24 / 32 / 40 / 48
```

计算 normalized AULC / AUC。

如果 Future 只在 `0/16/32/48` full eval，则用这四点计算，并在结果中说明。

### B. Time-to-activation

分别记录：

```text
budget_to_first_target_skill
budget_to_second_target_skill
```

若始终未激活：

```text
>48
```

### C. Capability coverage

记录：

```text
unique MH evidence
unique temporal evidence
unique S4 evidence
# activated target skills
```

### D. Acquisition efficiency

记录：

```text
ΔFuture / 100k generated tokens
ΔFuture / 100 tool calls
ΔFuture / rollout
ΔFuture / unique acquired task
```

---

# 8. Statistical reporting

对 10 seed paired differences 做 bootstrap：

```text
mean gap
std
paired bootstrap 95% CI
win rate
```

至少报告：

```text
frontier - Ace-Skill
frontier - passive
Ace-Skill - passive
```

Future probe grain：

```text
1/36 ≈ 0.0278
```

小于约一个题翻转的差异不得写成实质优势。

---

# 9. 本轮最关键的 Go / No-Go

本轮不要在看到结果前继续改方法。

先跑完 strongest baseline，然后按以下决策执行。

## Case A：Frontier 明确优于 Ace-Skill

建议判据：

```text
mean(frontier - ace_skill) >= 0.05
AND
paired bootstrap 95% CI lower bound > 0
AND
frontier wins >= 7/10 seeds
```

同时最好满足至少一个 sample-efficiency 优势：

```text
AULC frontier > Ace-Skill
OR
time_to_second_target_skill frontier < Ace-Skill
```

### 若 Case A

结论：

> 当前 acquisition 机制已具有独立于 generic prioritized sampling 的实证价值。

下一步：

**不要继续优化 Helion。**

直接进入第二 benchmark / external generalization。

优先考虑 Search-Agent 场景，并尽量选择能与 SESA / SkillRL / SLIM 等公开代码对接的 benchmark/protocol。

---

## Case B：Frontier 与 Ace-Skill 基本持平

例如：

```text
|mean gap| < 0.05
或 CI 跨 0
```

结论：

> 当前 frontier rule 很可能只是 generic prioritized curriculum 的一个特例，方法创新性不足。

### 若 Case B

停止继续堆 baseline。

下一阶段才允许升级方法为：

```text
Evidence-Gated Skill Activation
```

重点不是优化 `frontier score` 的小权重，而是替换当前：

```text
count >= K
```

这一简单规则。

候选方向：

```text
evidence sufficiency
= quantity
+ diversity
+ repeated verified failure
+ cross-instance consistency
+ estimated marginal capability need
```

要求 activation evidence 必须来自 acquisition side，不能看 future/test probe。

在设计完成前不要直接跑大量 GPU 实验。

---

## Case C：Ace-Skill 明确优于 Frontier

结论：

> 当前 acquisition 方法本身不够强，不能以 frontier sampling 作为论文核心方法贡献。

下一步：

1. 保留 R5/R5.1 作为 mechanism finding；
2. 不再把 frontier sampler 当 novelty；
3. 设计新的 evidence/utility-aware acquisition；
4. 新方法必须至少重新打赢 Ace-Skill 后再进入外部 benchmark。

---

# 10. 本轮暂时不要做的事情

除非 R5.2 结果明确要求，否则禁止：

```text
- R6 learned carrier router
- learned assimilation router
- 新 LoRA / SFT sweep
- 更多 episodic memory prompt engineering
- 再做 RealFit vs shuffle 的类似消融
- 大规模调 K
- 为了解决 temporal 人工堆 capability heuristic
- 只增加 seed 而不增加 strongest baseline
- 直接宣称完整 ExPolicy
```

R5.1 已经对这些问题给出了足够的信息。

---

# 11. 关于 SESA / SkillRL / SLIM / HASP：本轮如何处理

这些工作必须进入 Related Work 和后续 baseline planning，但本轮不要强行做不公平的“简化版复现”。

## Ace-Skill

**本轮必须跑。**

原因：它与当前问题直接同构——有限 rollout budget 下主动把算力分配给 informative / under-mastered samples。

---

## SESA

SESA 是：

```text
self-play task generation
+ frontier shaping
+ failure distillation
+ evolving skill memory
```

它改变的是整个 task-generation/training loop。

不要把其中一句 frontier heuristic 抽出来就称为 “SESA baseline”。

如果后续进入第二 search benchmark，再优先考虑按官方代码做 system-level comparison。

---

## SkillRL

SkillRL 包含：

```text
experience-based skill distillation
+ hierarchical SkillBank
+ adaptive retrieval
+ recursive skill evolution
+ RL
```

同样不是 acquisition-only baseline。

后续外部 benchmark 可以比较完整方法。

---

## SLIM

SLIM 管理：

```text
skill expansion
skill retention
skill retirement
marginal external contribution
```

其核心变量是 active skill lifecycle，而不是固定 pool 中的 acquisition sampler。

它与我们下一阶段可能的 `Evidence-Gated Skill Activation` 高度相关，因此必须重点阅读。

如果 R5.2 得到 Case B/C，在设计新 gating 前先系统对照 SLIM，避免把 marginal-contribution lifecycle 重新实现一遍。

---

## HASP

HASP 已经证明：

```text
textual skill → executable Program Function
```

并支持 runtime activation/intervention。

因此我们不能把“skill 是 executable”本身当主要 novelty。

我们的潜在区分点应放在：

```text
what evidence should be acquired
→ when an external capability becomes justified / eligible
```

而不是：

```text
how to write an executable skill
```

---

# 12. 最终必须生成的结果文件

至少生成：

```text
outputs/r5_2_baseline_collision/
├── R5_2_MASTER_SUMMARY.md
├── R5_2_DECISION.json
├── R5_2_FINAL_METRICS.csv
├── R5_2_LEARNING_CURVES.csv
├── R5_2_ACTIVATION_TIMES.csv
├── R5_2_ACQUISITION_DISTRIBUTION.csv
├── R5_2_EFFICIENCY.csv
├── R5_2_PAIRED_BOOTSTRAP.json
├── ACE_SKILL_IMPLEMENTATION_AUDIT.md
└── per_seed/
```

`R5_2_MASTER_SUMMARY.md` 必须明确回答：

1. Frontier 是否显著优于 Ace-Skill？
2. 优势是否跨 10 seed？
3. 优势来自 final performance，还是更快 skill activation？
4. Ace-Skill 是否因为重复采样损失 unique evidence coverage？
5. Temporal 是否仍是所有 acquisition policy 的盲区？
6. 当前贡献还能否写成独立 acquisition 方法？
7. 下一步属于 Case A / B / C 中哪一个？

---

# 13. 实现完成后的运行入口

先检查：

```bash
cd /data/ppnm/expolicy

nvidia-smi

python scripts/run_r5_2_baselines.py --help
python scripts/analyze_r5_2_baselines.py --help
```

runner 必须支持：

```text
--methods
--seeds
--budget
--gpu
--resume
--output-dir
```

建议统一入口示例：

```bash
cd /data/ppnm/expolicy

CUDA_VISIBLE_DEVICES=4 python scripts/run_r5_2_baselines.py \
  --methods ace_skill_prioritized \
  --seeds 0 4 8 \
  --budget 48 \
  --resume \
  --output-dir outputs/r5_2_baseline_collision &

CUDA_VISIBLE_DEVICES=5 python scripts/run_r5_2_baselines.py \
  --methods ace_skill_prioritized \
  --seeds 1 5 9 \
  --budget 48 \
  --resume \
  --output-dir outputs/r5_2_baseline_collision &

CUDA_VISIBLE_DEVICES=6 python scripts/run_r5_2_baselines.py \
  --methods ace_skill_prioritized \
  --seeds 2 6 \
  --budget 48 \
  --resume \
  --output-dir outputs/r5_2_baseline_collision &

CUDA_VISIBLE_DEVICES=7 python scripts/run_r5_2_baselines.py \
  --methods ace_skill_prioritized \
  --seeds 3 7 \
  --budget 48 \
  --resume \
  --output-dir outputs/r5_2_baseline_collision &

wait

python scripts/analyze_r5_2_baselines.py \
  --input-dir outputs/r5_2_baseline_collision \
  --reuse-r5 outputs/r5_acquisition_skill \
  --reuse-r5-1 outputs/r5_1_robustness
```

注意：以上 CLI 是本轮要求的统一接口。如果当前 repo 尚无这些参数，先实现并完成 smoke test，再正式运行；不要用 mock rollout 或伪造结果替代真实实验。

---

# 14. 一句话执行原则

> **这一轮不要继续证明“我们的方法有效”，而要证明“它不是 Ace-Skill 类 generic prioritized sampling 的换壳”。**

只有 strongest-baseline collision test 通过后，才值得进入第二 benchmark；如果打平或失败，再升级为真正的 Evidence-Gated Skill Activation。
