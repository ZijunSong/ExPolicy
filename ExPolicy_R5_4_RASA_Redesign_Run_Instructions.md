# ExPolicy R5.4 — RASA 方法重设计与下一轮运行指令

> 日期：2026-08-15  
> Repo：`/data/ppnm/expolicy`  
> 目标：**不再修补已经在公开 benchmark 上失效的 Helion frontier heuristic，而是做一次真正的方法级 redesign。**  
> 新方法暂名：**RASA — Remediability-Aware Sequential Acquisition**。  
> 本文档可以直接交给 Cursor Agent 执行。不得使用 mock、伪造检索、gold-as-hit、未来集标签、未来集 outcome 或手工挑 seed。

---

## 0. 这轮实验的研究决策

R5.3 已经关闭了原来的 `frontier acquisition` 主线：

- 在 Helion 上，`frontier > Ace-Skill = passive`，而且 10/10 seed 稳定；
- 但在 HotpotQA / 2Wiki / MuSiQue + Wikipedia 2018 E5 的公开环境中：
  - `frontier ΔFuture = -0.0067`
  - `Ace-Skill ΔFuture = +0.0811`
  - `passive ΔFuture = +0.0822`
  - frontier 0/3 seed 胜出；
  - 48/48 acquisition 全部塌缩到 HotpotQA C1；
  - 同时 skill backend 并没有失效：C2 executable skill validation `ΔEM = +0.417`。

因此：

### 必须 discard 的东西

1. **discard 当前 Helion-derived `frontier_score` 作为候选主方法**；
2. 不调其系数；
3. 不调 `K=8` 来“救”它；
4. 不重抽 R5.3 split 来让旧 frontier 重新变好；
5. 不把论文 claim 缩成“Helion 上有效”；
6. 不把 paper 改写成“我们发现 frontier heuristic 有时会失败”的弱 observation paper；
7. 不进入 learned carrier router / LoRA / RealFit；
8. `legacy_frontier` 以后只作为**失败 baseline**保留。

### 保留的基础设施

可以保留，因为它们已经分别通过了机制验证：

- 真实 Wikipedia 2018 + E5 retriever；
- HotpotQA / 2Wiki / MuSiQue 环境；
- C1/C2/C3 public capability taxonomy；
- R5.3 已验证的 executable skill primitives；
- CountOnly K=8 作为 baseline backend；
- Ace-Skill Prioritized Sampler；
- passive；
- deployable text router；
- acquisition / future / validation 分离；
- R5.3 全部结果，作为 redesign 的历史负结果，**不得覆盖**。

---

# 1. 新的研究问题：不是“哪些题难”，而是“哪些能力缺口值得被修复”

旧 frontier 的核心缺陷是把“当前表现处于某个 frontier”近似成“值得继续投入经验”。

R5.3 说明这两个量不是一回事：

- C1 当前表现低，但对应 executable skill 几乎不能稳定修复它；
- C2 同样表现低，却存在一个能显著修复失败的 skill；
- 原 frontier 只看 capability 的表现形状，因而把预算持续压给 C1；
- passive / Ace 因为保留覆盖，反而碰到了 C2。

下一版不再估计 generic difficulty / learnability，而估计：

> **Repairable Capability Deficit：当前能力缺口有多大 × 已有 intervention 对该缺口有多大概率真正修复。**

对于 capability \(c\)，维护两个在线后验：

\[
q_c=P(\text{base fails}\mid c)
\]

\[
p_c=P(\text{skill repairs}\mid \text{base failed},c)
\]

并定义：

\[
z_c=q_c p_c
\]

其中 \(z_c\) 不是 skill 的“质量分”，而是：

> **在当前任务流上，激活这个已存在、已验证 executable skill 的期望可修复缺口。**

这直接区分：

- hard + repairable：应该优先确认并激活；
- hard + unrepairable：应该尽快停止浪费 acquisition budget；
- easy + repairable：收益有限；
- easy + unrepairable：无需投入。

这也是本轮最重要的机制假设。

---

# 2. Novelty Guard：先排除已经被抢占的 claim

截至 2026-08-15，以下方向已经有人做，**禁止把它们写成我们的核心 novelty**：

| 已有方向 | 代表工作 | 本项目禁止 claim |
|---|---|---|
| frontier / ZPD task selection | AgentFrontier, arXiv:2510.24695 | “在模型能力 frontier 选择训练任务” |
| instance proficiency prioritized sampling | Ace-Skill（R5.2 已 collision test） | “根据 mastery / difficulty 主动采样” |
| skill creation/refinement/selection 联合学习 | SkillMaster, arXiv:2605.08693；Skill1, arXiv:2605.06130 | “agent 自己学会管理 skill” |
| counterfactual probe utility for skill edits | SkillMaster, arXiv:2605.08693 | “用 counterfactual utility 判断 skill 是否有效” |
| dynamic retain / retire / expand lifecycle | SLIM, arXiv:2605.10923 | “根据 marginal utility 动态维护 skill 生命周期” |
| evidence-backed positive-gain skill crystallization | MSCE, arXiv:2607.16621 | “evidence-gated skill promotion” |
| search self-play + evolving skill memory | SESA, arXiv:2607.29468 | “失败改变未来练习分布并与 skill memory 共演化” |
| proposer/solver/skill-controller co-evolution | Skill Self-Play, arXiv:2607.22529 | “skill 与 task generation 共演化” |
| executable skill / program function | HASP, arXiv:2605.17734 | “把 textual skill 变成 executable procedure” |
| evidence-guided hypothesis testing for skill revision | SkillHEX, arXiv:2608.05628 | “通过证据测试修订 skill” |
| cross-batch evidence accumulation | MAA, arXiv:2606.20475 | “跨 batch 累积 evidence 再做 skill decision” |
| progressive RL skill generation | Skill-α, arXiv:2608.01678 | “根据 downstream execution signal 生成 skill” |

## 2.1 本轮允许验证的独立贡献

RASA 只能把贡献放在下面这个问题上：

> **Given a fixed bank of already validated executable capabilities and a limited stream-level acquisition budget, actively choose new experiences to estimate whether each capability deficit is both present and remediable, and sequentially decide ACTIVATE / CONTINUE / PARK.**

与上面工作必须保持以下边界：

- **不创建 skill**；
- **不修改 skill 内容**；
- **不学习 skill text**；
- **不把 evidence 汇总成新 skill**；
- **不研究 skill retirement after policy learning**；
- **不做 self-play task generation**；
- **不把 generic failure frequency 当 novelty**；
- **不把 generic evidence gate 当 novelty**；
- 核心必须是：  
  **active experience acquisition for estimating a repairable capability deficit under distribution shift**。

## 2.2 运行前的强制 literature collision audit

新增：

```text
outputs/r5_4_rasa/NOVELTY_AUDIT.md
```

至少检查上述工作，并额外搜索 2026-08-01 至当前日期的新论文，关键词：

```text
"remediability" agent skill
"repairable capability deficit" LLM agent
"active experience acquisition" skill activation
"sequential evidence" skill activation agent
"active skill eligibility" LLM agent
"Bayesian skill activation" agent
"capability acquisition" executable skill
```

若发现已有工作同时满足下面四项：

1. 固定/已有 skill bank；
2. 主动选择新的 experience；
3. 显式估计 base failure 与 skill repairability；
4. 用该估计决定 skill activation / defer / reject；

则：

> **立即 STOP R5.4，不跑 GPU 主实验。**

不要通过换名词继续做。

---

# 3. 不复用已经看过的 R5.3 future 作为 R5.4 主结果

RASA 的设计已经看到了 R5.3 的失败，因此不能把同一个 R5.3 future split 再当成新的独立 test。

R5.3 保留为：

```text
historical development / failure analysis
```

R5.4 必须重新建立一个**完全未被 R5.3 使用过**的 holdout protocol。

这不是“重抽 R5.3 split 来隐藏失败”：

- R5.3 原 split 原封不动保留；
- R5.3 的 Case C 仍然写进论文；
- R5.4 是一个新方法、一个新预注册 protocol；
- 新 test 必须排除 R5.3 calibration / acquisition / skill_validation / future 的所有 task id 和 exact question。

---

# 4. R5.4 新 frozen split

新增：

```text
scripts/build_r5_4_holdout_split.py
outputs/r5_4_rasa/split_r5_4.json
outputs/r5_4_rasa/SPLIT_AUDIT.md
```

固定：

```text
split_seed = 20260815
datasets = HotpotQA / 2WikiMultiHopQA / MuSiQue

per dataset:
    acquisition_pool = 300
    future_test = 150
```

要求：

1. 三个 dataset 分别抽取，不允许某一 dataset 数量不足后偷偷由其它 dataset 回填；
2. 与 R5.3 的所有 split 零 task-id overlap；
3. 与 R5.3 的所有 split 零 exact-question overlap；
4. R5.4 acquisition 与 R5.4 future 零 overlap；
5. split 一旦生成，写 SHA256；
6. 后续任何条件和 seed 共用同一个 frozen split；
7. 不因为结果差而重抽。

如果某 dataset 没有足够未使用样本：

```text
FAIL_CLOSED
```

并打印真实剩余数量；不要自动缩小或换数据集。

---

# 5. Capability assignment：主实验禁止 oracle label dispatch

R5.4 主实验从一开始就使用 deployable capability assignment。

复用/重训 R5.3 的 TF-IDF + multinomial LR router，但训练数据只能来自：

```text
R5.3 calibration + R5.3 acquisition 的 question text + capability label
```

允许 capability label，因为它属于 development-side annotation。

禁止：

- R5.4 future label；
- R5.4 future task id；
- dataset filename；
- gold answer；
- gold docs；
- future outcome。

必须输出：

```text
outputs/r5_4_rasa/ROUTER_AUDIT.md
```

至少报告：

- train size；
- class distribution；
- R5.3 heldout/dev accuracy；
- R5.4 question 中是否含 task-id leakage；
- vocabulary leakage audit；
- 对 R5.4 acquisition/future 只保存 prediction，不看 true capability outcome 来修改 router。

真实 capability metadata 只允许在实验结束后用于 diagnostic，不进入主方法。

---

# 6. RASA 的算法定义

新增：

```text
expolicy/acquisition/rasa.py
expolicy/assimilation/readiness_gate.py
```

不要把 RASA 写成若干手工权重的线性 score。

## 6.1 Bayesian state

对每个 predicted capability \(c\)：

```python
q_c ~ Beta(alpha_q, beta_q)  # P(base failure)
p_c ~ Beta(alpha_p, beta_p)  # P(skill repairs | base failure)
```

统一 prior：

```text
Beta(1, 1)
```

每 acquisition task：

### Step A：clean diagnostic execution

必须先关闭对应 skill，运行 clean base：

```text
success -> beta_q += 1
failure -> alpha_q += 1
```

这里 `q` 始终描述 **unassisted capability deficit**。

### Step B：failure-side remediation test

仅当 clean base 失败，而且 capability 尚未 terminal：

使用该 capability 的 frozen R5.3 executable skill 在**同一 acquisition-side task**上重跑一次。

```text
failure -> skill success:
    alpha_p += 1

failure -> skill failure:
    beta_p += 1
```

注意：

- 该 replay 不能访问 future；
- 不能改 skill；
- 不能从 replay 结果生成新 skill；
- 它只是 estimation probe。

### Step C：repairable deficit

固定 4096 次 posterior Monte Carlo：

```python
q = Beta(alpha_q, beta_q)
p = Beta(alpha_p, beta_p)
z = q * p
```

每个 acquisition seed 使用固定 RNG seed，结果可复现。

---

# 7. Evidence diversity：防止“同一种失败刷够数量”

不能再用 `distinct task id >= 8` 等价于 sufficient evidence。

在任何 outcome 产生之前，用 acquisition question text 的 E5 embedding 做 frozen clustering。

每个 predicted capability：

```text
KMeans k=4
random_state=13
```

如果该 cap 样本不足以稳定形成 4 cluster，则：

```text
k = min(4, max(2, floor(n_cap / 20)))
```

cluster 仅用于 evidence diversity，不使用 dataset name / answer / outcome。

记录：

```text
cluster_coverage_c = number of distinct clusters represented by tested base failures
```

ACTIVATE 至少要求：

```text
cluster_coverage_c >= 2
```

within-cap instance selection：

1. 优先最少覆盖 cluster；
2. cluster 内选与已 acquired evidence 最不相似的 candidate；
3. cosine distance 基于 frozen E5 embedding；
4. 不允许 replacement。

这部分只解决 redundant evidence，不作为单独 novelty。

---

# 8. Sequential Readiness Test：ACTIVATE / CONTINUE / PARK

固定 meaningful effect：

```text
delta = 0.10
confidence = 0.90
```

定义：

```python
P_good = P(z_c >= 0.10)
```

### ACTIVATE

```text
P_good >= 0.90
AND n_tested_base_failures >= 4
AND cluster_coverage >= 2
```

则：

```text
state[c] = ACTIVE
```

此后 future/deployment 可以调用对应 frozen skill。

### PARK

满足任一：

```text
P_good <= 0.10 AND n_tested_base_failures >= 4
```

或预注册 futility rule：

```text
n_tested_base_failures >= 8 AND repair_count == 0
```

则：

```text
state[c] = PARKED
```

PARKED 后本轮不再为该 capability 花 acquisition budget。

### CONTINUE

其余均为：

```text
UNRESOLVED
```

不允许根据 R5.4 结果修改：

```text
delta
confidence
4-failure minimum
8-zero-repair futility
cluster_coverage=2
```

若这些固定值导致方法失败，按失败处理，不现场调参。

---

# 9. RASA acquisition policy

RASA 不再使用：

```text
4*s*(1-s)
Helion transfer_potential
frontier coefficient
hand-tuned weighted sum
```

## 9.1 Forced exploration

开始时，每个 capability 最多先获取 2 个 unique tasks：

- 尽量来自两个不同 frozen semantic clusters；
- 总 warm start 上限 6 个 tasks；
- 如果 router 预测某 cap 在 pool 中不足 2 个，记录并 fail audit。

## 9.2 Thompson acquisition over repairable deficit

warm start 后，对每个 `UNRESOLVED` capability：

```python
q_sample ~ Beta(alpha_q, beta_q)
p_sample ~ Beta(alpha_p, beta_p)
z_sample = q_sample * p_sample
```

选择：

```python
c* = argmax z_sample
```

然后按照 §7 的 diversity rule 从该 capability 选择一个新 task。

对于：

```text
ACTIVE
PARKED
```

均不再主动 acquisition。

若所有 capability 都 terminal：

```text
early stop
```

不得为了凑够 48 个 rollout 继续刷题。

记录真实使用 budget。

---

# 10. Budget 定义

Primary experience budget：

```text
B_acq = 48 unique new task instances
```

同一个 acquisition-side failure 上执行 skill remediation replay：

- **不增加 unique experience count**；
- 但必须单独计入：
  - tokens；
  - tool calls；
  - wall time；
  - `diagnostic_replay_count`；
  - total executor episodes。

所以论文必须同时报告：

```text
new-experience efficiency
compute-normalized efficiency
```

不能把 remediation replay 当成免费。

checkpoint：

```text
0 / 16 / 32 / 48 acquired unique experiences
```

若 RASA early stop，则后续 checkpoint 使用 terminal agent，不再新增 experience。

---

# 11. Stage-1 条件：一次性跑完整因果矩阵

主实验不要只跑 RASA vs passive。

必须同时跑下面 8 个条件：

| condition | acquisition | activation | 目的 |
|---|---|---|---|
| `passive_k8` | passive | CountOnly K=8 | 原始随机 baseline |
| `ace_k8` | Ace-Skill | CountOnly K=8 | strongest generic prioritized baseline |
| `legacy_frontier_k8` | old frontier | CountOnly K=8 | 保留 R5.3 failed method |
| `rasa_k8` | RASA q×p acquisition | CountOnly K=8 | 只测新 acquisition |
| `passive_srt` | passive | Sequential Readiness Test | 只测新 gate |
| `rasa_full` | RASA q×p acquisition | SRT | 完整方法 |
| `rasa_qonly` | q-only acquisition | SRT | 去掉 remediability |
| `rasa_nodiversity` | RASA q×p | SRT，但 within-cap random | 去掉 diversity control |

所有条件：

```text
seeds = 0,1,2
B_acq = 48
same frozen R5.4 split
same Qwen2.5-3B-Instruct
same Wikipedia 2018 + E5 retriever
same executable skills
same deployable text router
T = 0
```

`qonly` 定义：

```python
z_sample = q_sample
```

但 SRT 的 activation decision 仍使用完整 `q*p`，从而只隔离 acquisition priority 中的 remediability。

---

# 12. 主指标

## 12.1 Primary outcome

每个 checkpoint 在：

```text
HotpotQA future = 150
2Wiki future = 150
MuSiQue future = 150
```

上评测。

Primary：

```text
macro future accuracy
ΔFuture vs clean base
```

Secondary：

```text
HotpotQA accuracy
2Wiki accuracy
MuSiQue accuracy
per-capability accuracy
```

## 12.2 Learning efficiency

报告：

```text
AULC over checkpoints 0/16/32/48
budget_to_first_ACTIVE
budget_to_C2_ACTIVE
number of ACTIVE / PARKED / UNRESOLVED caps
unique evidence per cap
semantic-cluster coverage
```

## 12.3 Mechanism state

每个 checkpoint 输出：

```text
alpha_q / beta_q
alpha_p / beta_p
posterior mean(q)
posterior mean(p)
posterior mean(z)
P(z >= 0.10)
state = ACTIVE / PARKED / UNRESOLVED
```

## 12.4 Cost

报告：

```text
acquired unique tasks
diagnostic skill replays
total executor episodes
tokens
tool calls
wall clock
future-eval cost
```

---

# 13. Stage-1 预注册 Go / No-Go

这轮不能出现“结果差一点，所以改成只 claim sample efficiency”。

`rasa_full` 必须同时满足：

### Gate A：最终能力必须真的更强

```text
mean final macro(rasa_full)
    >= max(passive_k8, ace_k8) + 0.02
```

并且：

```text
3/3 seeds rasa_full > strongest baseline
```

2 个百分点在 450-task future 上约等于 9 道题，不是单题噪声。

### Gate B：必须有正 absolute gain

```text
mean ΔFuture(rasa_full) > 0
```

### Gate C：不能只靠新 gate

```text
rasa_full - passive_srt >= 0.01
```

否则 acquisition policy 没有被证明。

### Gate D：不能只靠新 acquisition + 旧 K

```text
rasa_full - rasa_k8 >= 0.01
```

否则 SRT 没有被证明。

### Gate E：remediability 必须是因果变量

至少满足：

```text
rasa_full - rasa_qonly >= 0.01
```

或者在最终 macro 接近时，RASA full 必须在：

```text
AULC
或 total acquisition needed before terminal decision
```

上有明确优势，且差异大于一个 checkpoint 的单题粒度。

如果 q-only 与 full 几乎完全一样：

> **不要改写成“failure-aware acquisition”论文。该方向与已有 difficulty/failure sampling 太接近，直接判定当前 novelty 未成立。**

### Gate F：不能再次 collapse

到 B=48 时：

```text
至少 2/3 capability 已经达到 ACTIVE 或 PARKED 的 terminal decision
```

不允许出现：

```text
48/48 都花在一个仍然 UNRESOLVED 的 capability
```

### 总判

```text
PASS = A & B & C & D & E & F
```

---

# 14. 失败后的处理规则

## Case A：完整 PASS

进入 Stage-2：

- 扩 seed 3–9；
- full RASA / Ace / passive 共 10 seed；
- paired bootstrap；
- 最终再做 Bamboogle zero-shot；
- 方法及所有阈值冻结，不再调。

## Case B：RASA > baseline，但 `rasa_full ≈ rasa_k8`

说明 acquisition 有效、SRT 没贡献。

处理：

> **不发表“joint acquisition + readiness gating”版本。**

只有在 `rasa_k8` 同时：

- 显著优于 Ace；
- 显著优于 passive；
- remediability ablation 成立；
- 10-seed 稳健；

时，才允许把项目**重新定义**为：

```text
Remediability-Aware Active Experience Acquisition
```

并删除 SRT，而不是把 SRT 留在 paper 里凑模块。

如果做不到，discard。

## Case C：`passive_srt ≈ rasa_full`

说明 acquisition 不重要，只是 gate 有用。

因为 evidence-gated promotion / lifecycle 已有 MSCE / SLIM / related work：

> **不把它包装成新 paper。直接 discard 当前 RASA acquisition 主线。**

## Case D：`rasa_qonly ≈ rasa_full`

说明 `p(skill repairs | failure)` 并没有提供独立价值。

> **不要退化成 generic hard-example / failure sampling。直接 redesign 或停止。**

## Case E：Full 不优于 Ace / passive

> **RASA method falsified。停止这条论文线。**

不得：

- 改阈值；
- 改 seed；
- 改未来集；
- 改成“Helion + public average”；
- 只报最好 dataset；
- 只报最好 seed；
- 只 claim AULC；
- 把 final failure 降级成 caveat 后继续写主论文。

---

# 15. Stage-2（只有 Stage-1 PASS 才允许执行）

## 15.1 10-seed

新增 seed：

```text
3,4,5,6,7,8,9
```

只跑：

```text
rasa_full
ace_k8
passive_k8
```

最终合并 seed 0–9。

PASS：

```text
mean(rasa - strongest_baseline) >= 0.02
paired bootstrap 95% CI lower > 0
>= 8/10 seeds win
```

否则停止，不做 Bamboogle。

## 15.2 Bamboogle zero-shot

只有 10-seed PASS 后：

- 不在 Bamboogle acquisition；
- 不调 router；
- 不修改 skills；
- 使用每个 seed 在 public acquisition stream 最终冻结的 ACTIVE skill set；
- 用 deployable router；
- 与 clean base / Ace-final / passive-final 比较；
- 报全部 125 题；
- 不删失败题。

Bamboogle 只验证 transfer，不参与任何选择。

---

# 16. 必须实现的文件

```text
expolicy/acquisition/rasa.py
expolicy/assimilation/readiness_gate.py

scripts/build_r5_4_holdout_split.py
scripts/audit_r5_4_protocol.py
scripts/run_r5_4_condition.py
scripts/analyze_r5_4.py
scripts/run_r5_4_stage2.py

outputs/r5_4_rasa/
├── NOVELTY_AUDIT.md
├── SPLIT_AUDIT.md
├── ROUTER_AUDIT.md
├── PROTOCOL_AUDIT.md
├── split_r5_4.json
├── R5_4_STAGE1_MASTER.csv
├── R5_4_LEARNING_CURVES.csv
├── R5_4_POSTERIORS.csv
├── R5_4_DECISIONS.csv
├── R5_4_COST.csv
├── R5_4_GATE_DECISION.json
├── R5_4_MASTER_SUMMARY.md
├── per_seed/
└── logs/
```

禁止覆盖：

```text
outputs/r5_acquisition_skill/
outputs/r5_1_robustness/
outputs/r5_2_baseline_collision/
outputs/r5_3_external_generalization/
```

---

# 17. Protocol audit

`scripts/audit_r5_4_protocol.py` 必须在任何 GPU rollout 前检查：

```text
[ ] R5.3 output untouched
[ ] R5.4 split hash exists
[ ] R5.4 vs R5.3 task-id overlap == 0
[ ] R5.4 vs R5.3 exact-question overlap == 0
[ ] R5.4 acq vs future overlap == 0
[ ] no gold answer in acquisition scorer
[ ] no gold doc in acquisition scorer
[ ] no future outcome in RASA
[ ] no true future capability label in RASA
[ ] text router frozen before R5.4 evaluation
[ ] executable skills byte/hash frozen
[ ] legacy frontier coefficients frozen
[ ] Ace-Skill implementation frozen
[ ] RASA thresholds frozen
[ ] Beta priors frozen
[ ] KMeans fitted without outcomes
[ ] no replacement within method
[ ] all seeds retained
[ ] real retriever only
[ ] no mock LLM
```

只有：

```text
PASS all
```

才允许启动 Stage-1。

---

# 18. 服务器实际运行顺序

进入 repo：

```bash
cd /data/ppnm/expolicy
set -euo pipefail
mkdir -p outputs/r5_4_rasa/logs
```

## 18.1 第一步：只做 CPU / protocol 工作

Cursor Agent 先完成 RASA 与 runner，实现后执行：

```bash
python scripts/build_r5_4_holdout_split.py \
  --seed 20260815 \
  --acq-per-dataset 300 \
  --future-per-dataset 150 \
  --output outputs/r5_4_rasa/split_r5_4.json

python scripts/audit_r5_4_protocol.py \
  --split outputs/r5_4_rasa/split_r5_4.json \
  --out outputs/r5_4_rasa/PROTOCOL_AUDIT.md
```

如果 audit 非 0：

```text
STOP
```

不得启动 GPU。

---

# 19. Retriever

必须复用 R5.3 的 Wikipedia 2018 + E5-base-v2 真实 retriever。

目标端口仍为：

```text
127.0.0.1:18080
```

不要新造一个简化 retriever。

`run_r5_4_condition.py` 应复用 `scripts/orchestrate_r5_3.py` 内已经验证的 retriever lifecycle / health check 代码。

运行前：

```bash
curl -fsS http://127.0.0.1:18080/health || true
```

若服务未启动：

> 从 `scripts/orchestrate_r5_3.py` 抽取/复用原有启动路径并正常启动；  
> 不允许用 mock server 或小型替代 corpus。

---

# 20. 四张 GPU 的 Stage-1 分配

先设置：

```bash
cd /data/ppnm/expolicy
mkdir -p outputs/r5_4_rasa/logs
```

### GPU 4：两个 strongest baseline

```bash
nohup env CUDA_VISIBLE_DEVICES=4 \
python scripts/run_r5_4_condition.py \
  --condition passive_k8 \
  --seeds 0,1,2 \
  --budget 48 \
  --resume \
  > outputs/r5_4_rasa/logs/gpu4_passive_k8.log 2>&1

nohup env CUDA_VISIBLE_DEVICES=4 \
python scripts/run_r5_4_condition.py \
  --condition ace_k8 \
  --seeds 0,1,2 \
  --budget 48 \
  --resume \
  > outputs/r5_4_rasa/logs/gpu4_ace_k8.log 2>&1
```

Cursor Agent 应把这两个条件写成**顺序执行**的 GPU4 job wrapper，而不是让两个模型同时抢一张卡。

推荐创建：

```text
scripts/run_r5_4_gpu4.sh
```

内容按顺序执行 `passive_k8 -> ace_k8`。

---

### GPU 5：旧方法 + 新 acquisition-only

顺序：

```text
legacy_frontier_k8
rasa_k8
```

执行：

```bash
nohup bash -lc '
set -euo pipefail
cd /data/ppnm/expolicy

CUDA_VISIBLE_DEVICES=5 python scripts/run_r5_4_condition.py \
  --condition legacy_frontier_k8 \
  --seeds 0,1,2 \
  --budget 48 \
  --resume

CUDA_VISIBLE_DEVICES=5 python scripts/run_r5_4_condition.py \
  --condition rasa_k8 \
  --seeds 0,1,2 \
  --budget 48 \
  --resume
' > outputs/r5_4_rasa/logs/gpu5.log 2>&1 &
echo $! > outputs/r5_4_rasa/logs/gpu5.pid
```

---

### GPU 6：gate-only + full method

顺序：

```text
passive_srt
rasa_full
```

执行：

```bash
nohup bash -lc '
set -euo pipefail
cd /data/ppnm/expolicy

CUDA_VISIBLE_DEVICES=6 python scripts/run_r5_4_condition.py \
  --condition passive_srt \
  --seeds 0,1,2 \
  --budget 48 \
  --resume

CUDA_VISIBLE_DEVICES=6 python scripts/run_r5_4_condition.py \
  --condition rasa_full \
  --seeds 0,1,2 \
  --budget 48 \
  --resume
' > outputs/r5_4_rasa/logs/gpu6.log 2>&1 &
echo $! > outputs/r5_4_rasa/logs/gpu6.pid
```

---

### GPU 7：两个核心 mechanism ablation

顺序：

```text
rasa_qonly
rasa_nodiversity
```

执行：

```bash
nohup bash -lc '
set -euo pipefail
cd /data/ppnm/expolicy

CUDA_VISIBLE_DEVICES=7 python scripts/run_r5_4_condition.py \
  --condition rasa_qonly \
  --seeds 0,1,2 \
  --budget 48 \
  --resume

CUDA_VISIBLE_DEVICES=7 python scripts/run_r5_4_condition.py \
  --condition rasa_nodiversity \
  --seeds 0,1,2 \
  --budget 48 \
  --resume
' > outputs/r5_4_rasa/logs/gpu7.log 2>&1 &
echo $! > outputs/r5_4_rasa/logs/gpu7.pid
```

---

# 21. 注意：GPU4 不要用两个独立 nohup 并发

GPU4 也应和 5/6/7 一样创建顺序 wrapper：

```bash
nohup bash -lc '
set -euo pipefail
cd /data/ppnm/expolicy

CUDA_VISIBLE_DEVICES=4 python scripts/run_r5_4_condition.py \
  --condition passive_k8 \
  --seeds 0,1,2 \
  --budget 48 \
  --resume

CUDA_VISIBLE_DEVICES=4 python scripts/run_r5_4_condition.py \
  --condition ace_k8 \
  --seeds 0,1,2 \
  --budget 48 \
  --resume
' > outputs/r5_4_rasa/logs/gpu4.log 2>&1 &
echo $! > outputs/r5_4_rasa/logs/gpu4.pid
```

以这一版为准。

---

# 22. Watchdog

每 60 秒检查：

```bash
watch -n 60 '
echo "=== GPU ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true

echo "=== JOBS ==="
for f in /data/ppnm/expolicy/outputs/r5_4_rasa/logs/gpu{4,5,6,7}.log; do
  echo "----- $f -----"
  tail -n 8 "$f" 2>/dev/null || true
done
'
```

`--resume` 必须：

- 只跳过已经完整写完且 hash / metrics 完整的 `(condition, seed)`；
- 不能因为目录存在就跳过半成品；
- 不能重写已完成 seed；
- crash 后可恢复。

---

# 23. Stage-1 完成后的唯一分析命令

四卡全部完成后：

```bash
cd /data/ppnm/expolicy

python scripts/analyze_r5_4.py \
  --root outputs/r5_4_rasa \
  --stage 1 \
  --bootstrap 10000 \
  --output outputs/r5_4_rasa/R5_4_MASTER_SUMMARY.md
```

分析脚本必须自动生成：

```text
R5_4_STAGE1_MASTER.csv
R5_4_LEARNING_CURVES.csv
R5_4_POSTERIORS.csv
R5_4_DECISIONS.csv
R5_4_COST.csv
R5_4_GATE_DECISION.json
R5_4_MASTER_SUMMARY.md
```

并自动给出：

```text
CASE_A_PASS
CASE_B_ACQ_ONLY
CASE_C_GATE_ONLY
CASE_D_REMEDIABILITY_NOT_CAUSAL
CASE_E_METHOD_FAIL
```

不要让人手看结果后再选择 Case。

---

# 24. 若 Stage-1 PASS，Stage-2 四卡分配

**只有 `CASE_A_PASS` 才执行。**

seed 3–9：

### GPU4

```text
rasa_full seeds 3,7
```

### GPU5

```text
rasa_full seeds 4,8
```

### GPU6

```text
rasa_full seeds 5,9
```

### GPU7

```text
rasa_full seed 6
```

然后同样用四卡补：

```text
ace_k8 seeds 3–9
passive_k8 seeds 3–9
```

推荐让 `scripts/run_r5_4_stage2.py` 自动排队，不要手工复制命令：

```bash
python scripts/run_r5_4_stage2.py \
  --root outputs/r5_4_rasa \
  --gpus 4,5,6,7 \
  --seeds 3,4,5,6,7,8,9 \
  --conditions rasa_full,ace_k8,passive_k8 \
  --resume
```

Stage-2 最终分析：

```bash
python scripts/analyze_r5_4.py \
  --root outputs/r5_4_rasa \
  --stage 2 \
  --bootstrap 20000 \
  --output outputs/r5_4_rasa/R5_4_FINAL_SUMMARY.md
```

---

# 25. 论文 framing 的硬约束

如果 RASA 成功，论文问题应该是：

> **Under distribution shift, which experiences should an agent acquire before deciding that an external capability is worth activating?**

核心贡献只能来自：

1. 把 acquisition target 从 generic difficulty/frontier 改成 **repairable capability deficit**；
2. 用真实 acquired failures 上的 intervention outcomes 区分：
   - need；
   - remediability；
3. 在有限 experience budget 下主动决定：
   - acquire more evidence；
   - activate；
   - park；
4. 在公开 search benchmarks 上比 passive / Ace / legacy frontier 获得更高 future capability，而不是只在 Helion 合成环境赢。

不能写：

```text
We propose a general skill lifecycle framework.
We propose evidence-gated skill promotion.
We propose executable skills.
We propose self-evolving skill memory.
We propose counterfactual skill optimization.
We propose frontier curriculum learning.
```

这些 novelty 已经不安全。

---

# 26. 最后一个原则

本轮最重要的不是“把 RASA 跑赢”。

而是要回答一个足够锋利、且现有结果真正能证伪的问题：

> **Active acquisition 是否只有在同时估计“缺口存在”与“缺口可被当前 intervention 修复”时，才能在分布变化后避免把有限预算浪费在 hard-but-unremediable capabilities 上？**

如果答案是否定的：

> **discard RASA。**

不要把失败改成：

- “在某个 dataset 上略有优势”；
- “更省一点 token”；
- “AULC 好一点但 final 不好”；
- “只对 C2 有效”；
- “Helion 仍然很强”。

那些都会把研究从一个方法贡献降成局部 observation。

只有在新方法能在新的 frozen public holdout 上同时取得：

```text
stronger final future capability
+
causal remediability ablation
+
active acquisition contribution
+
sequential decision contribution
+
strong-baseline advantage
```

时，才值得继续扩成论文主线。
