# ExPolicy 下一步运行指令：R6 Assimilation-Validated Acquisition（AVA）可行性与 Novelty Gate

> 日期：2026-08-15  
> 基于：`expolicy-result-record(7).md` 最新 R5.3 / R5.4 结果  
> Repo：`/data/ppnm/expolicy`  
> 当前结论：**停止 legacy frontier 与 RASA 主线；不调阈值、不改 K、不重抽 R5.4 holdout。下一步不是 narrow claim，而是重设计 experience acquisition 的目标函数。**

---

## 0. 本轮研究决策

R5.3 已证明：

- Helion 上的 `frontier > Ace-Skill / passive` 是真实且稳定的；
- 但冻结 frontier score 到公开 HotpotQA / 2Wiki / MuSiQue 后，48/48 acquisition 全塌缩到 C1；
- 真正有用的 C2 没被采到；
- 因而 frontier 是 **Helion-specific heuristic**，不能继续作为通用方法。

R5.4 已进一步证明：

- 将 acquisition target 改成 `q=P(base fails|c)` 与 `p=P(skill repairs|failed,c)` 的 `q×p`，仍不能超过 Ace/passive；
- `rasa_full = rasa_k8 = rasa_qonly`，remediability 没有独立贡献；
- C3 上 1 次偶然 repair 足以阻止 PARK，导致约 35/48 budget 被继续投入几乎不可修的 long-chain；
- 因此 **RASA 直接 discard，不再做 delta/confidence/K/zero-repair threshold sweep**。

下一轮改写研究问题：

> **Experience value is not intrinsic. An experience is valuable only if, after being processed by the current assimilation/update operator, it produces measurable future capability gain.**

工作名暂定：

> **AVA — Assimilation-Validated Acquisition**

核心目标不是再预测“难不难”“失败多不多”“skill 能不能偶尔修复”，而是直接估计：

\[
V(B\mid S,A)
=
J\!\left(A(S,B);D_{\text{shadow}}\right)
-
J\!\left(S;D_{\text{shadow}}\right)
-
\lambda C(B)
\]

其中：

- \(S\)：当前 agent / skill-bank / evidence state；
- \(B\)：待 acquisition 的 experience micro-batch；
- \(A\)：**实际部署时真正使用的 assimilation operator**；
- \(D_{\text{shadow}}\)：与最终 future test 严格隔离的 shadow probe；
- \(J\)：future capability metric；
- \(C(B)\)：额外 acquisition / assimilation 成本。

即：**先问“如果我真的学这批经验，未来能力会涨多少”，再决定是否采。**

本轮只验证这个核心 principle 是否成立。若 exact shadow utility 本身都不能打败 passive / Ace，则直接 discard AVA，不继续训练 surrogate，不通过缩窄 claim 自救。

---

# 1. GPU 前置 Gate 0：最新 Novelty Collision Audit

## 1.1 必须重新检索截至 2026-08-15 的最近工作

禁止只依赖旧 `NOVELTY_AUDIT.md`。

至少逐篇阅读 Method / Algorithm / Experimental protocol，并建立 collision matrix：

1. Ace-Skill — arXiv:2605.08887
2. Bayesian-Agent — arXiv:2606.08348
3. SLIM / Dynamic Skill Lifecycle Management — arXiv:2605.10923
4. EvoLib / Test-Time Learning with an Evolving Library — arXiv:2605.14477
5. SkillRL — arXiv:2602.08234
6. SESA — arXiv:2607.29468
7. SkillHEX — arXiv:2608.05628
8. Harness-R1 — arXiv:2608.02276
9. SkillSV / What Is a Skill Worth? — arXiv:2608.04562
10. Rethinking Self-Evolving Agent Skills — arXiv:2608.02636
11. PROGRESS — arXiv:2506.01085
12. SkillRevise — arXiv:2606.01139
13. HASP — arXiv:2605.17734
14. EvoSkill — arXiv:2603.02766
15. 任何 2026-08-15 前新出现的：
    - experience/data acquisition for agents
    - learning-progress curriculum for agents
    - data valuation for self-evolving agents
    - future-gain / information-gain skill evolution
    - bilevel experience selection
    - shadow-update / temporary-update data valuation

### 对每篇工作必须回答

| 项 | 必答问题 |
|---|---|
| Selection object | 选择 raw task / experience / trajectory / skill / skill revision / abstraction / data point 中哪一个？ |
| Update object | memory / skill / executable harness / model weight / curriculum？ |
| Score | difficulty / uncertainty / proficiency / failure / information gain / marginal skill value / realized post-update gain？ |
| Assimilator-conditioned | score 是否显式依赖“当前实际 update operator”？ |
| Shadow update | 是否临时 clone 当前 agent，真正 apply candidate experience 后再评估？ |
| Eval separation | 是否用与最终 test 隔离的 shadow/validation probe？ |
| Online acquisition | 是否用该值决定**下一批 environment experience**采什么？ |
| Capability labels | 是否依赖预定义 taxonomy / oracle cap label？ |
| Learned selector | 是否把昂贵真实 value 蒸馏成廉价 selector？ |
| Generalization | 是否跨 benchmark / environment 验证？ |

## 1.2 Novelty Hard Stop

如果找到任何已发表/公开工作同时满足以下四点：

1. selection object 是 **未来要获取的 agent experience / task batch**；
2. 对 candidate experience 做 **temporary/shadow assimilation**；
3. 用 **post-assimilation held-out future gain** 作为 acquisition value；
4. 再据此决定后续 acquisition；

则：

> **立即停止 AVA 实现。**

不要通过把 skill bank、Search-Agent、CountOnly、某个公式换名来声称 novelty。

必须输出：

```text
outputs/r6_ava/audit/NOVELTY_COLLISION.md
outputs/r6_ava/audit/NOVELTY_COLLISION.csv
```

并给出：

```text
NOVELTY_STATUS = PROCEED / COLLISION / UNCERTAIN
```

只有 `PROCEED` 才能进入 GPU 实验。

---

# 2. Gate 1：代码审计，先确认 R5.3/R5.4 的失败不是实现问题

本轮允许并要求做一次代码级 audit。

```bash
cd /data/ppnm/expolicy
mkdir -p outputs/r6_ava/audit

git status --short
git rev-parse HEAD | tee outputs/r6_ava/audit/GIT_HEAD.txt
git diff --stat | tee outputs/r6_ava/audit/GIT_DIFF_STAT.txt
```

## 2.1 审计 acquisition → assimilation → future eval 的真实因果路径

重点 grep：

```bash
rg -n "frontier|rasa|qonly|passive|ace_skill|CountOnly|K=8|skill.*activ|fit\(|future|probe|router|capability" \
  expolicy scripts tests \
  > outputs/r6_ava/audit/RG_PIPELINE.txt
```

必须人工/Agent 阅读并回答：

1. R5.3 frontier 的 48/48 C1 是否确实由 score 排序导致，而不是候选池过滤 bug；
2. R5.4 `rasa_full` 是否真的在每次 acquisition 后更新 q/p；
3. `repair_count` 是否只在 base failure 上更新；
4. C3 的唯一 repair 是否来自真实 skill execution；
5. ACTIVE / PARKED 后是否真的停止该 cap acquisition；
6. `qonly` 是否只改变 acquisition score，而 activation 仍用 q·p；
7. `rasa_k8` 是否只替换 gate，而未偷偷改变 candidate pool；
8. final future test 是否从未进入任何 acquisition / threshold / router fitting；
9. text-router 是否没有使用 dataset id / task id / split filename / gold；
10. seed 0/1/2 三次 RASA 结果同构是否来自 deterministic pipeline，而不是 seed 没传进去。

新增：

```text
scripts/audit_r6_pipeline.py
```

输出：

```text
outputs/r6_ava/audit/R6_PIPELINE_AUDIT.json
outputs/r6_ava/audit/R6_PIPELINE_AUDIT.md
```

要求所有 critical checks PASS。

若发现会改变 R5.3/R5.4 结论的 bug：

> **先修 bug 并只重跑受影响的旧实验。不要直接进入 AVA。**

---

# 3. Gate 2：必须先做 case-level failure analysis

不要直接写新 acquisition policy。

新增：

```text
scripts/analyze_r5_4_cases_for_r6.py
```

## 3.1 读取的数据

读取：

```text
outputs/r5_3_external_generalization/
outputs/r5_4_rasa/
```

针对 C1 / C2 / C3，分别抽取至少：

- 20 个 acquisition cases；
- 20 个 shadow/skill-replay cases（若已有）；
- 20 个 final future cases；

优先分析：

- base fail + skill fail
- base fail + skill repair
- base success + skill regression
- C3 唯一 repair case
- C2 repair 成功 cases
- passive/Ace 激活 C3 后 MuSiQue 新增答对 cases

## 3.2 每个 case 输出

至少记录：

```json
{
  "task_id": "...",
  "dataset": "...",
  "router_cap": "...",
  "question": "...",
  "base_correct": false,
  "skill_correct": true,
  "base_trace": "...",
  "skill_trace": "...",
  "retrieved_docs": [],
  "failure_mode": "...",
  "repair_mode": "...",
  "skill_helped": true,
  "skill_harmed": false,
  "estimated_q": 0.0,
  "estimated_p": 0.0
}
```

禁止只看 aggregate number。

## 3.3 必须回答的核心问题

### Q1

R5.4 C3 的问题究竟是：

A. skill 对 C3 基本无效；

还是

B. skill 只对某个稀疏子簇有效，因此粗粒度 capability-level `p` 把异质性压平；

还是

C. base/skill replay 的单题 repair probability 与 deployment marginal utility 本来就不一致。

### Q2

为什么 CountOnly 激活 C3 能让 MuSiQue 从约 0.007 到 0.033，而 RASA posterior 仍判断 C3 几乎不可修？

计算：

```text
per-task skill delta
per-cluster skill delta
P(skill helps | base fail)
P(skill harms | base success)
net deployment delta
```

### Q3

计算现有 proxy 与真实 downstream utility 的排序相关性：

- frontier score
- failure rate q
- repair probability p
- q×p
- Ace proficiency score（如可映射）
- raw uncertainty
- actual held-out skill gain / downstream gain

输出 Spearman / Kendall。

## 3.4 Case Gate

如果：

```text
q×p 与真实 downstream utility Spearman >= 0.60
```

且错误主要能用单一阈值 bug 解释，

则：

> **不要做 AVA。RASA 的问题只是工程阈值，不足以支撑新方法贡献；直接 discard 当前项目分支。**

如果：

- proxy ranking 明显失真；
- utility 在 task/cluster 层面存在真实异质性；
- 且这种异质性不能由 dataset/capability label 直接解释；

则进入 R6。

输出：

```text
outputs/r6_ava/case_analysis/CASE_REPORT.md
outputs/r6_ava/case_analysis/CASE_TABLE.jsonl
outputs/r6_ava/case_analysis/PROXY_VS_UTILITY.csv
```

---

# 4. R6 核心原则：不要再设计新的 hand-crafted proxy

R6 不允许：

- 改写 frontier 系数；
- 加一个新的 `q×p×novelty×uncertainty`；
- 调 RASA PARK threshold；
- 用 capability frequency 做补丁；
- 人工规定 C2 比 C1 更值得采；
- 用 future-test label 训练 acquisition；
- 用 dataset name 作为 capability；
- 使用 oracle capability label 做主方法。

下一步只测试：

> **post-assimilation realized future gain 是否能成为 experience acquisition 的有效 supervision。**

---

# 5. R6.0 数据切分：从 R5.4 acquisition pool 内再切 shadow probe

**不得触碰 R5.4 final future_test。**

R5.4 每数据集已有 300 acquisition tasks。

在每个 dataset 内、对现有 acquisition pool 固定拆分：

```text
candidate_pool = 240
shadow_probe   = 60
```

三数据集总计：

```text
candidate_pool = 720
shadow_probe   = 180
```

要求：

- 不改变 R5.4 final future_test 450；
- candidate / shadow / final future 三者 task-id overlap = 0；
- exact question overlap = 0；
- split seed 固定为 `20260816`；
- 一旦写盘，不允许重抽。

新增：

```text
scripts/build_r6_ava_split.py
```

运行：

```bash
cd /data/ppnm/expolicy

python scripts/build_r6_ava_split.py \
  --source-split outputs/r5_4_rasa/split_r5_4.json \
  --seed 20260816 \
  --candidate-per-dataset 240 \
  --shadow-per-dataset 60 \
  --out outputs/r6_ava/split_r6_ava.json
```

输出审计：

```text
outputs/r6_ava/SPLIT_AUDIT.md
outputs/r6_ava/SPLIT_HASH.txt
```

---

# 6. R6.1 Exact AVA：先做“昂贵 oracle”，不急着训练 selector

这是整个重设计最重要的一步。

如果 exact post-assimilation utility 本身都不能指导 acquisition，则没有必要再设计 learned policy。

## 6.1 Candidate batch

每轮 acquisition budget 仍按 48 unique tasks，总计 6 round：

```text
6 rounds × 8 unique experiences
```

不要使用 oracle C1/C2/C3 分桶作为 candidate。

对 remaining candidate pool：

1. 用 question embedding；
2. 做在线 unsupervised clustering / nearest-neighbor grouping；
3. 每轮生成 `M=8` 个 candidate micro-batches；
4. 每个 batch 含 8 个 unique tasks；
5. batch 之间允许部分候选竞争，但最终 commit batch 不允许重复 task。

第一版建议直接采用：

```text
E5 embedding
k-means / balanced nearest-cluster sampling
M = 8 candidate batches / round
batch size = 8
```

candidate construction 只负责保证覆盖，不赋予任何“价值”分数。

---

# 7. Exact Shadow Assimilation

对每个 candidate batch \(B_j\)：

```text
clone current agent state S_t
        ↓
actually acquire / execute B_j
        ↓
apply the exact same production assimilation operator A
        ↓
evaluate cloned updated agent on shadow_probe
        ↓
compute ΔShadowFuture
        ↓
destroy clone
```

正式定义：

\[
V_j
=
\mathrm{MacroAcc}(A(S_t,B_j),D_\text{shadow})
-
\mathrm{MacroAcc}(S_t,D_\text{shadow})
\]

初期**不要加复杂 cost penalty**。

先独立报告：

- ΔShadowFuture
- acquisition tokens
- diagnostic tokens
- tool calls
- wall-clock

主 selector 先只用 ΔShadowFuture，避免又人为混入 λ。

---

# 8. Assimilation operator：第一阶段必须保持冻结

为了把变量严格限制在 acquisition：

第一阶段复用已经验证、完全冻结的公开 skill bank + deployable text router + CountOnly unique-evidence activation。

即：

```text
candidate experiences
    ↓
text-router
    ↓
unique evidence state
    ↓
frozen activation rule
    ↓
frozen executable skill bank
```

注意：

- **不允许**在 R6.1 同时重写 skill generator；
- 不允许修改 R5.4 executable skill；
- 不允许调 K；
- 不允许用 oracle cap dispatch；
- text-router weights 冻结；
- shadow clone 与真实 state 必须完全隔离。

这一阶段不是最终 paper configuration，而是验证：

> “如果 acquisition value 定义为真实 post-update capability gain，它是否能修掉 R5.3/R5.4 的 proxy failure？”

若这一最小验证都失败，则 AVA 直接停止。

---

# 9. R6.1 Baselines

同一 split、同一 48 unique experience budget、同一 frozen backend：

```text
passive
ace_skill_prioritized
legacy_frontier
rasa_full
exact_ava
```

可额外保留：

```text
random_cluster
```

但不要增加一堆弱 baseline。

Primary comparison：

```text
exact_ava vs max(passive, ace_skill_prioritized)
```

legacy frontier / RASA 只是历史失败对照。

---

# 10. R6.1 Stage-1 Gate（跑前冻结）

seeds：

```text
0, 1, 2
```

final metric：

```text
macro accuracy on untouched R5.4 final future_test
```

Exact AVA 必须同时满足：

### Gate A — Final capability

```text
mean(exact_ava) >= max(mean(passive), mean(ace)) + 0.02
```

### Gate B — Seed consistency

```text
exact_ava > max(passive, ace) on 3/3 seeds
```

### Gate C — Sample efficiency

至少一个：

```text
AULC_exact >= best_baseline_AULC + 0.02
```

或：

```text
达到 best baseline final score 所需 unique experiences 至少少 8 条
```

### Gate D — Shadow validity

每轮 candidate batch 的：

```text
Spearman(shadow Δ, eventual committed-batch downstream Δ) >= 0.40
```

或者 pooled candidate ranking top-1 hit rate 显著高于 random。

### Gate E — 不依赖 dataset shortcut

对 AVA 选择的 batch 做 dataset/capability distribution audit。

若结果实际上退化成：

```text
always choose 2Wiki
```

或：

```text
always choose router-C2
```

则必须进一步检查是否 shadow probe 本身泄漏了固定 benchmark composition。

### Gate F — 无 final-test leakage

严格 PASS。

总判：

```text
CASE_A_EXACT_AVA_PASS
```

仅当 A∧B∧C∧D∧E∧F 全过。

---

# 11. GPU 分配

先完成 CPU audit / case analysis。

通过 Gate 0–2 后：

## GPU 4

```bash
cd /data/ppnm/expolicy
CUDA_VISIBLE_DEVICES=4 python scripts/run_r6_ava_stage1.py \
  --condition exact_ava \
  --seed 0 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --batch-size 8 \
  --num-candidates 8 \
  --resume
```

## GPU 5

```bash
cd /data/ppnm/expolicy
CUDA_VISIBLE_DEVICES=5 python scripts/run_r6_ava_stage1.py \
  --condition exact_ava \
  --seed 1 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --batch-size 8 \
  --num-candidates 8 \
  --resume
```

## GPU 6

```bash
cd /data/ppnm/expolicy
CUDA_VISIBLE_DEVICES=6 python scripts/run_r6_ava_stage1.py \
  --condition exact_ava \
  --seed 2 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --batch-size 8 \
  --num-candidates 8 \
  --resume
```

## GPU 7

GPU7 顺序跑 baselines，优先复用能严格复用的旧轨迹，但必须在 R6 新 split 上重新评测：

```bash
cd /data/ppnm/expolicy

CUDA_VISIBLE_DEVICES=7 python scripts/run_r6_ava_stage1.py \
  --condition passive \
  --seeds 0 1 2 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --resume

CUDA_VISIBLE_DEVICES=7 python scripts/run_r6_ava_stage1.py \
  --condition ace_skill_prioritized \
  --seeds 0 1 2 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --resume

CUDA_VISIBLE_DEVICES=7 python scripts/run_r6_ava_stage1.py \
  --condition legacy_frontier \
  --seeds 0 1 2 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --resume

CUDA_VISIBLE_DEVICES=7 python scripts/run_r6_ava_stage1.py \
  --condition rasa_full \
  --seeds 0 1 2 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --resume
```

如果 GPU7 baseline 总耗时影响较大，可先只跑：

```text
passive + Ace-Skill
```

legacy/RASA 后补；**不要为了省时间省掉 primary strongest baseline。**

---

# 12. 必须实现的脚本

```text
scripts/audit_r6_pipeline.py
scripts/analyze_r5_4_cases_for_r6.py
scripts/build_r6_ava_split.py
scripts/build_r6_candidate_batches.py
scripts/eval_r6_shadow_utility.py
scripts/run_r6_ava_stage1.py
scripts/analyze_r6_ava_stage1.py
```

建议新增模块：

```text
expolicy/acquisition/ava.py
expolicy/evaluation/shadow_probe.py
expolicy/state/isolated_clone.py
```

---

# 13. Clone isolation 是本轮最高优先级工程审计项

必须写单测确保 shadow candidate 评估不会污染真实 agent：

```text
tests/test_r6_clone_isolation.py
tests/test_r6_no_future_leakage.py
tests/test_r6_candidate_unique.py
tests/test_r6_shadow_determinism.py
tests/test_r6_router_freeze.py
```

至少验证：

1. shadow assimilation 后真实 evidence count 不变；
2. skill activation mask 不变；
3. memory / cache / adapter 不变；
4. RNG state 不串；
5. candidate B1 的评测不影响 B2；
6. final future test id 从未出现在 shadow evaluator；
7. final future answers/gold/doc ids 不进入 selector；
8. shadow probe 只来自 R6 split 内固定 180 题。

运行：

```bash
pytest -q \
  tests/test_r6_clone_isolation.py \
  tests/test_r6_no_future_leakage.py \
  tests/test_r6_candidate_unique.py \
  tests/test_r6_shadow_determinism.py \
  tests/test_r6_router_freeze.py
```

任何一项失败都禁止开 Stage-1。

---

# 14. R6.1 结果分析

运行：

```bash
python scripts/analyze_r6_ava_stage1.py \
  --root outputs/r6_ava \
  --out outputs/r6_ava/R6_STAGE1_MASTER.md
```

必须输出：

```text
R6_STAGE1_MASTER.csv
R6_STAGE1_MASTER.md
R6_GATE_DECISION.json
R6_LEARNING_CURVES.csv
R6_SHADOW_RANK_CORRELATION.csv
R6_CANDIDATE_SELECTIONS.jsonl
R6_COST.csv
R6_DATASET_CAP_DISTRIBUTION.csv
```

图/表至少包括：

- final macro；
- per-dataset accuracy；
- learning curve @ 0/8/16/24/32/40/48；
- AULC；
- shadow value vs actual downstream delta；
- selected batch composition；
- skill activation timeline；
- tokens / tools / wall；
- exact AVA 相对 Ace/passive 的 paired per-seed gap。

---

# 15. CASE_A：Exact AVA 通过后才做 learned surrogate

如果 Exact AVA PASS，**不要直接把昂贵 shadow evaluator 当最终方法**。

下一步才进入：

> **AVA-Surrogate：用少量 exact shadow-assimilation observations 学一个廉价 value predictor。**

输入可包括：

```text
task/question embedding
current evidence state
router probabilities
base outcome statistics
skill-bank state
candidate-batch diversity
historical realized AVA values
```

禁止使用：

```text
future-test labels
dataset one-hot
oracle capability label
gold answer
gold document
```

输出：

```text
predicted post-assimilation future gain
uncertainty
```

建议先做：

```text
ridge / small MLP / LightGBM
```

而不是立刻 RL。

目的：

> 判断昂贵 exact value 是否可被一个 deployable selector 近似。

Surrogate 必须用 leave-one-seed-out 或 chronological split，禁止在同一 candidate observation 上训练再测试。

---

# 16. CASE_B：Exact AVA 强，但 surrogate 失败

如果：

```text
Exact AVA PASS
Surrogate FAIL
```

不要把论文 claim 缩成：

> “oracle AVA shows potential”。

这是不可部署的，不足以作为主方法。

允许一次 method-level redesign，例如：

- contextual bandit / UCB；
- calibrated uncertainty；
- low-rank value model；

但只能基于明确误差分析选择一种。

若仍失败：

> **discard AVA method。**

可保留 exact-utility 结果作为内部 insight，不单独弱化成论文。

---

# 17. CASE_C：Exact AVA 本身失败

如果 Exact AVA：

- final ≤ passive/Ace；
- 或 0/3、1/3 seed 胜；
- 或 shadow ranking 与 downstream utility 无相关；
- 或只能通过 dataset/capability shortcut 工作；

则：

> **立即停止 ExPolicy acquisition-policy 主线。**

不要：

- 改 shadow probe composition；
- 改 M=8；
- 改 batch=8 去追结果；
- 调 λ；
- 重新聚类直到 C2 被选中；
- 重新定义 macro；
- 换 seed；
- 改成“在部分能力上有效”。

此时应直接 change framing / discard method。

---

# 18. CASE_D：Exact + Surrogate 均通过

只有此时才进入正式扩展：

1. seed 3–9，扩到 10 seed；
2. Bamboogle / 新公开 zero-shot benchmark；
3. 第二个 assimilation operator；
4. 验证 **assimilation-conditioned** 是否真的必要。

最关键的第二阶段 ablation：

```text
AVA(current assimilator)
AVA(score learned under another assimilator)
assimilation-agnostic utility
Ace-Skill
passive
```

如果：

```text
V(x | A1) ranking != V(x | A2) ranking
```

且 matched-AVA 显著优于 mismatched-AVA，

则才真正支持更大的研究贡献：

> **Experience value is learner/update-operator dependent, and acquisition policies should optimize post-assimilation capability gain rather than intrinsic sample difficulty.**

这比“RASA 在某些数据集更好”要强得多，也更接近一个可长期成立的 Experience Policy 问题。

---

# 19. 与最新工作的 novelty boundary

本轮必须明确避开以下已被抢占的贡献：

### Ace-Skill

已覆盖：

```text
proficiency-aware prioritized sampling + clustered skill organization
```

AVA 不能写成“更好的 prioritized sampler”。

### Bayesian-Agent

已覆盖：

```text
posterior-guided skill/SOP evolution
```

AVA 不能写成“Bayesian skill utility / posterior skill activation”。

### SLIM

已覆盖：

```text
leave-one-skill-out marginal contribution
retain / retire / expand skill lifecycle
```

AVA 不能把“skill marginal utility”当 novelty。

### EvoLib

已覆盖：

```text
IG / FutureIG weighting of skills and abstractions
```

AVA 的 selection object 必须是 **future experience batch**，且 value 必须是 **actual post-assimilation state delta**，不能只是 abstraction FutureIG 改名。

### SkillSV

已覆盖：

```text
held-out task distribution上的 skill / skill-unit marginal valuation
```

AVA 不能写成“what is a skill worth”。

### Harness-R1

已覆盖：

```text
realized downstream utility → train harness editor
```

AVA 不训练 patch editor；它决定 **which environment experience to acquire next**。

### SkillHEX

已覆盖：

```text
hypothesis-driven self-verification + evidence-guided tree search for skill revision
```

不要做 failure hypothesis tree / patch search。

### SESA

已覆盖：

```text
search self-play task generation ↔ skill memory co-evolution
```

AVA 不以 self-play / challenger 作为贡献。

### PROGRESS

已覆盖：

```text
learning-progress-based sample selection for VLM training
```

因此论文若最终成立，必须证明 agentic setting 中关键对象不是 generic learning progress，而是：

```text
assimilation-conditioned realized future utility
```

并通过 matched vs mismatched assimilator 实验证明这一点。

---

# 20. 本轮明确禁止

- 不再跑 R6 learned carrier router；
- 不再做 RASA threshold sweep；
- 不重启 RealFit；
- 不补 Helion coefficient；
- 不继续证明 Helion frontier；
- 不把 negative result 改成弱 claim；
- 不用 mock LLM；
- 不用 gold-as-hit retrieval；
- 不重抽 R5.4 final holdout；
- 不允许 shadow evaluator 看 final future；
- 不允许 oracle capability dispatch 成为主结果；
- 不在 exact AVA 未通过前训练 RL acquisition policy。

---

# 21. 执行顺序

严格按以下顺序：

```text
P0  Novelty collision audit
    ↓
P1  R5.3/R5.4 code audit
    ↓
P2  C1/C2/C3 case-level failure analysis
    ↓
Gate: proxy ≠ real post-update utility?
    ↓ yes
P3  Frozen R6 split + clone-isolation tests
    ↓
P4  Exact AVA 3-seed Stage-1
    ↓
Gate: exact AVA > Ace/passive?
    ├─ no  → DISCARD AVA / STOP ExPolicy acquisition line
    └─ yes
         ↓
P5  Learned AVA surrogate
         ↓
Gate: deployable surrogate retains gain?
    ├─ no  → one justified redesign, then discard if still fail
    └─ yes
         ↓
P6  10 seed + Bamboogle + second assimilator
         ↓
P7  matched-vs-mismatched assimilator test
```

---

# 22. 最重要的研究判断

当前实验已经否定了两个直觉：

```text
difficulty/frontier ≠ learning value
repair probability ≠ deployment value
```

下一轮不应继续发明第三个 hand-crafted proxy。

真正值得验证的更大命题是：

> **The utility of an experience is defined by the capability change produced after the current agent actually assimilates it. Therefore, experience acquisition and assimilation cannot be optimized independently.**

如果这一命题在 exact AVA 上都没有实证支撑，就应彻底放弃 ExPolicy acquisition 主线。

如果 exact AVA、deployable surrogate、跨 benchmark、matched-vs-mismatched assimilator 四层都通过，则它重新形成一个足够强的研究故事，而不是把失败的 RASA claim 越缩越小。
