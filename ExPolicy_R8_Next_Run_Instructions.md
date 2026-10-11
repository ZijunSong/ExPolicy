# ExPolicy R8：Structural Redesign + Heterogeneous Persistent-Intervention Policy

> 日期：2026-08-21  
> 基于：`expolicy-result-record(20260820-235842).md` 最新 R7 结果  
> 目标：在 **不削弱 ExPolicy 核心贡献** 的前提下，决定下一轮是否值得继续训练 controller；如果当前实验结构不支持该问题，则先 redesign benchmark / assimilator substrate，而不是继续调 Linear TS、缩小 claim 或围绕单一 C2 做局部优化。

---

## 0. 本轮先冻结的科学结论

R7 已经完成了此前最重要的工程目标：

```text
acquisition
→ real environment interaction
→ observed experience
→ assimilation action
→ persistent state mutation
→ shadow capability reward
→ policy posterior update
→ next-round acquisition / assimilation
```

Gate P 全部通过，因此 **closed-loop Experience Policy runtime 已经成立**。

但是 R7 的 method gate 没有通过：

- `M0 expolicy` final macro = **0.0474**；
- `B2 CountOnly` = **0.0911**；
- `B4 all_skills` = **0.0911**；
- `A3 assimilation_fixed` = **0.0681**；
- M0 只比 NoFeedback / RewardShuffle 高约 **0.004**；
- acquisition distribution 的最大 TV 只有 **0.024**；
- 终局几乎由“是否激活 C2 comparison skill”决定。

因此，**下一步禁止把问题简单解释成“Linear TS 不够强，所以换一个更强 bandit”**。

当前更根本的问题是：

> 在冻结 C1/C2/C3 skill bank 下，`all_skills = CountOnly = 0.0911`，说明当前 future utility 基本已经退化成 activation-mask lookup。只要最终把三个 skill 打开，固定策略就达到当前 substrate 的上界；learned assimilation 几乎没有可辨识的终局优势空间。

换句话说：

```text
当前 R7 SearchEnv 是一个很好的 closed-loop plumbing benchmark，
但已经不是一个足够好的 learned Experience Policy method benchmark。
```

**本轮必须先验证“policy 是否有可学的决策空间”，再训练 policy。**

---

# 1. 本轮真正要回答的问题

ExPolicy 的核心问题保持不变，不允许缩成 memory policy、skill router、sample-efficiency trick 或 C2-specific curriculum：

\[
\pi_{\mathrm{ExPolicy}}
(a_t^{\mathrm{acq}}, a_t^{\mathrm{assim}}\mid S_t,H_t,e_{1:t})
\]

其中：

1. `acquisition`：下一步应该主动获得哪类真实环境经验；
2. `assimilation`：获得这批经验后，应当通过哪种 **真实、可持久、会改变未来行为的 intervention** 利用它；
3. 学习目标：最大化未参与 controller update 的未来任务分布上的 capability gain。

R8 需要建立比 R7 更强的条件：

> **不同 experience 在不同 agent state 下，最优 assimilation operator 必须真实不同；且不同 acquisition 选择必须改变之后可实现的 capability improvement。**

如果这个条件不存在，就不训练 controller。

---

# 2. Novelty collision audit（2026-08-21）

在写任何新代码前，先把以下工作读完并写 `R8_NOVELTY_AUDIT.md`。禁止只看摘要后写“没有相关工作”。

## 2.1 必查近邻

### AgeMem — Agentic Memory

- ACL 2026 long paper。
- 把 long-term / short-term memory operations 直接变成 policy actions，并用 RL 学习何时 store / retrieve / update / summarize / discard。
- 已经抢占：**“learned policy controls memory operations”**。
- ExPolicy 不能把“动态决定存不存 memory”作为 novelty。

Paper: <https://aclanthology.org/2026.acl-long.981/>

### MemoPilot

- 2026，训练 memory update process，使 frozen player 在 sequential interactions 中 test-time learning。
- 使用 multi-turn GRPO 与 turn-wise credit。
- 已经抢占：**“通过 RL 学习 how to update memory over time”**。

Paper: <https://arxiv.org/abs/2606.08656>

### HiMPO

- 2026，针对 long-horizon memory-writing action 的 causally entangled credit assignment。
- 使用 local counterfactual utility + hindsight relevance 给 memory action 分配更干净的 credit。
- 已经抢占：**“counterfactual/local credit assignment for memory updates”**。

Paper: <https://arxiv.org/abs/2606.16285>

### MemRL

- 2026，冻结 LLM，在 episodic memory 上通过环境反馈学习 Q-value / utility，进行 runtime self-evolution。
- 已经抢占：**“learn utility of retrieved experience/memory from feedback”**。

Paper: <https://arxiv.org/abs/2601.03192>

Code: <https://github.com/MemTensor/MemRL>

### SkillOpt

- 2026，把 skill document 当成 frozen agent 的可训练外部参数；从 rollout feedback 进行 bounded edit，并通过 held-out validation gate 接受更新。
- 已经抢占：**“trajectory-driven validated skill editing”**。

Paper: <https://arxiv.org/abs/2605.23904>

Code: <https://github.com/microsoft/SkillOpt>

### Mem2Evolve

- ACL 2026。
- Experience Memory 与 Asset Memory 共演化；经验指导新 tool / expert asset 的创建。
- 已经抢占：**“experience-guided capability asset creation / co-evolution”**。

Paper: <https://aclanthology.org/2026.acl-long.952/>

### MAGE

- 2026。
- 明确提出 self-evolving agent 要决定 “what to learn next and how to preserve what it learned”；维护 co-evolutionary graph、task-level search bandit、skill-level routing bandit，并由同一 reward stream 更新。
- 这是 **ExPolicy 当前最危险的 conceptual collision**。
- 已经抢占：**“同时存在 task-side policy 与 skill-side routing policy，并共同从 feedback 演化”** 这一宽泛表述。

Paper: <https://arxiv.org/abs/2605.10064>

### SESA

- 2026。
- self-play challenger 生成未来练习任务，solver skill memory 从 informative failures 演化；更新后的 memory 又改变未来 problem distribution，形成 bidirectional loop。
- 已经抢占：**“task distribution 与 skill memory co-evolve”**。

Paper: <https://arxiv.org/abs/2607.29468>

### ProactAgent

- 2026。
- 在 factual / episodic / behavioral skill experience base 上学习何时、检索什么经验，通过 paired-branch process reward 判断 retrieval 是否有价值。
- 已经抢占：**“主动决定何时调用 memory/skill experience”**。

Paper: <https://arxiv.org/abs/2604.20572>

### MemEvolve / MemSkill / Meta Context Engineering

这些工作分别让 memory architecture、memory skills、context engineering skill 本身发生 meta-evolution。

- MemEvolve: <https://arxiv.org/abs/2512.18746>
- MemSkill: <https://arxiv.org/abs/2602.02474>
- Meta Context Engineering: <https://arxiv.org/abs/2601.21557>

因此 ExPolicy 也不能把“学习 how to learn / meta-evolution”本身作为 novelty。

## 2.2 当前仍有机会成立的独特点

截至本轮检索，没有发现一篇已发表/公开工作同时满足以下四点：

1. **主动选择下一批真实 environment experience**，而不是仅选择 retrieval item / memory entry / generated self-play problem；
2. 对同一 acquired experience，在 **多个异构 persistent update operator** 之间进行决策，例如 `discard / episodic-memory update / procedural-skill update / ...`，而不是只优化一种 substrate；
3. experience 的价值由 **该 experience 经当前可用 assimilator 实际 intervention 后的 held-out future capability change** 定义，而不是 immediate reward、difficulty、similarity 或固定 curriculum score；
4. acquisition 与 assimilation policy 在同一个 stateful loop 中更新，并且更新后的 persistent state 会改变下一轮应该采什么经验。

因此 R8 的 novelty 只能建立在这个 **交叉区域**：

> **Learning a policy over both what the agent should experience next and which persistent learning operator should consume that experience, under realized downstream capability feedback.**

禁止把以下单点写成 novelty：

- memory policy；
- skill editing；
- counterfactual memory credit；
- task curriculum；
- skill routing；
- self-evolution；
- experience distillation。

### Novelty hard-stop

如果代码/论文审计发现已有工作已经同时做到了上面 1–4，立即输出：

```text
NOVELTY_COLLISION_HARD_STOP
```

停止 R8 method implementation。不要通过改名或缩 claim 继续。

---

# 3. R8-A：先对 R7 做 Structural Identifiability Audit

**这一阶段不训练新模型，不改 controller。**

目标：判断 R7 失败到底是 controller 弱，还是当前 substrate 本身没有足够的 optimal-policy gap。

## 3.1 必须阅读的真实 cases

至少完整读取并对照以下四类 R7 trajectory：

1. `M0 seed1`：round_02 采到 C2，并选择 `ACTIVATE_EXISTING_SKILL`，shadow Δ≈+0.105；
2. `M0 seed2`：采到 C2 但连续 `DISCARD` 的 rounds；
3. `M0 seed0`：激活 C3、最终仍低分的轨迹；
4. `A1 NoFeedback seed1`：posterior 完全不更新但随机撞到 C2，最终≈0.089 的轨迹。

不能只读 summary CSV。需要读取 round-level：

- state features；
- sampled action logits / probabilities；
- posterior / model state；
- experience batch IDs；
- assimilation choice；
- reward decomposition；
- shadow before / after；
- skill mask before / after；
- next-round policy input。

输出：

```text
outputs/r8_preflight/R7_CASE_AUDIT.md
outputs/r8_preflight/R7_CASE_TRACE.csv
```

## 3.2 对同一 state 做真实 counterfactual assimilation replay

对 M0 18 个 `(state, acquired_batch)`，从**完全相同的 pre-assimilation snapshot** clone 出四份：

```text
DISCARD
EPISODIC
SEMANTIC
ACTIVATE_EXISTING_SKILL
```

分别执行一次真实 assimilation，然后在同一 `shadow_reward_probe` 上评估。

得到：

\[
U_t(a)=J_{shadow}(A_a(S_t,e_t))-J_{shadow}(S_t)-cost(a)
\]

本阶段的 cost 只沿用 R7 已冻结定义，**不得新调 λ**。

输出 18×4 utility matrix：

```text
outputs/r8_preflight/R7_COUNTERFACTUAL_ASSIM_MATRIX.csv
```

需要回答：

- 最优 action 是否几乎总是 ACTIVATE / DISCARD？
- 在采到 C2 时，ACTIVATE 的 margin 是否显著大于其它 action？
- controller 当时的 feature 是否足以区分这些 state？
- assimilation regret 主要来自“没看到信号”，还是“看到了但 policy 不会用”？

## 3.3 Structural ceiling audit

计算以下三项：

### S1. Oracle assimilation gap

对 18 个真实 acquired batch，用 hindsight oracle 每轮选择当轮最高 `U_t(a)`，commit 后继续下一轮，得到完整 oracle trajectory。

比较：

```text
oracle_joint
best_fixed_assim
CountOnly
all_skills
M0
```

### S2. Final-state ceiling

枚举当前冻结 C1/C2/C3 skill mask 8 种状态，并复核 R6/R7 的 final future utility。

如果：

```text
max achievable final under frozen substrate
- strongest fixed baseline
< 0.02
```

则判定当前 benchmark 对 learned assimilation **STRUCTURALLY_NON_IDENTIFIABLE**。

### S3. Feature separability

把 18×4 counterfactual matrix 当监督，不更新在线 policy，只做 offline diagnostic：

- Linear classifier / ridge；
- 2-layer MLP；
- leave-one-seed-out；
- label = best assimilation action；
- regression target = per-action U。

如果 MLP 也无法稳定预测最优 action，则说明 state representation 也有问题。

## 3.4 R8-A Gate

输出：

```text
outputs/r8_preflight/R8_STRUCTURAL_AUDIT.json
outputs/r8_preflight/R8_STRUCTURAL_AUDIT.md
```

判定：

### `R7_ENV_REUSABLE`

只有同时满足：

- oracle_joint final ≥ strongest fixed + **0.03**；
- 至少 2 个 assimilation action 在 ≥20% state 上 unique-best；
- best fixed 与 oracle 的平均 utility gap ≥ **0.02**；
- MLP leave-one-seed-out action accuracy 明显高于 majority baseline（≥ +15 pp）。

才允许直接在 R7 SearchEnv 上训练新 controller。

### `R7_ENV_STRUCTURALLY_DEGENERATE`

任何一项不满足，则 **禁止继续在当前 frozen-mask setting 上换 controller**。

基于现有结果，预期大概率命中这一分支。

---

# 4. R8-B：如果 R7 structurally degenerate，改用真实 heterogeneous assimilation testbed

## 4.1 为什么选择 ALFWorld 作为第一候选

不要人工制造“错误 skill 会伤害”“只能开一个 skill”之类的负迁移来创造 routing 难度。

优先使用 **同一个公开环境中已经有两类成熟、公开、机制不同的 experience assimilation backend**：

### Backend A：MemRL episodic-memory learning

MemRL 官方代码支持 ALFWorld，并通过环境反馈学习 episodic memory utility。

### Backend B：SkillOpt procedural-skill optimization

SkillOpt 官方代码支持 ALFWorld，把 trajectory feedback 转成 held-out-gated skill edits。

因此 ALFWorld 可以提供一个更干净的实验问题：

> 对同一类真实 embodied experience，有时应该写入/更新 episodic memory，有时应该用于 procedural skill update，有时应该丢弃；同时 agent 还要决定下一批应该练哪类 task。

这里 **MemRL 和 SkillOpt 都只是 assimilator backend / baseline，不是 ExPolicy novelty**。

---

# 5. R8-B0：先独立复现两个 assimilator，不写 ExPolicy

建立独立 checkout，禁止直接魔改第三方代码后称为原方法：

```text
/data/ppnm/third_party/memrl
/data/ppnm/third_party/skillopt
```

记录：

```text
repo URL
commit hash
local patch diff
license
config hash
```

## 5.1 Foundation model

第一选择继续：

```text
Qwen2.5-3B-Instruct
```

保持 frozen actor。

先跑 ALFWorld no-learning baseline。目标区间：

```text
10% <= success <= 70%
```

- <10%：floor 太低，assimilation gain 难解释；停止，不静默换 prompt。
- >70%：ceiling 太高；停止。

如果失败，输出 `MODEL_FOUNDATION_GATE_FAIL.md`，不要自动换更大模型。

## 5.2 Backend validation gate

在严格 train / validation / future split 下，分别验证：

```text
B_mem: passive experience → MemRL update
B_skill: passive experience → SkillOpt update
B_none: no persistent update
```

要求：

1. MemRL 相对 B_none 在至少一类 ALFWorld task family 上有 ≥ +0.05 future success；
2. SkillOpt 相对 B_none 在至少一类 task family 上有 ≥ +0.05；
3. 两者不能完全由同一个 backend 全局统治：至少存在两个 task family，最优 backend 不同；
4. `oracle per-family backend` 必须比 `best fixed backend` ≥ +0.03 macro。

只有满足 1–4 才进入 R8 joint policy。

否则：

```text
HETEROGENEOUS_ASSIMILATION_GATE_FAIL
```

**不要训练 ExPolicy，不要把论文缩成“SkillOpt 调度器”。**

下一步应换一个能自然产生 heterogeneous persistent interventions 的公开环境；优先 WebShop / ScienceWorld / LifelongAgentBench，而不是给 ALFWorld 人工加惩罚。

---

# 6. 数据划分协议

先枚举 ALFWorld 官方可用 gamefiles 与 task types，再一次性冻结 split。

禁止假定每类数量足够；脚本必须 fail-closed。

建议目标：

```text
calibration            20 / task family
acquisition_pool       >=60 / task family
assim_validation       20 / task family
shadow_reward_probe    20 / task family
retention_probe        20 total from initially-mastered/easier families
future_test             >=30 / task family
```

约束：

- gamefile / task id 零 overlap；
- exact instruction 零 overlap；
- future_test 不进入 SkillOpt validation；
- future_test 不进入 MemRL Q update；
- future_test 不进入 controller feature fitting / reward / early stopping；
- acquisition 无 replacement；
- 禁止因某 seed 不好而重抽 split。

写：

```text
outputs/r8_hetero/audit/SPLIT_AUDIT.md
outputs/r8_hetero/audit/split.json
outputs/r8_hetero/audit/split.sha256
```

---

# 7. R8 assimilation action space

第一版只放 **真正经过 B0 gate 验证的 operator**。

默认候选：

```text
A0 DISCARD
A1 MEMRL_UPDATE
A2 SKILLOPT_UPDATE
```

不要为了“动作更多”继续放 R7 已验证无效的 old episodic / semantic implementation。

`BOTH(MEMRL+SKILLOPT)` 暂时不作为 action；只有先证明 A1/A2 都有独立 positive utility 后，才可以作为 R8.1 extension。

每个 action 必须修改独立 persistent state：

```text
S_t = {
  memory_state,
  skill_document/version,
  controller_state,
  provenance/audit,
  cost_state
}
```

SkillOpt update 必须使用 acquired experience；不能读取 future_test。  
MemRL update 同理。

---

# 8. R8 controller redesign：不再使用 Linear TS

## 8.1 核心思想

R7 的 reward 太稀疏，且 assimilation action credit 严重混在一起。

R8 使用 **同-state counterfactual intervention utility** 给 assimilation action 提供 dense supervision，但不允许它变成 test-time oracle。

对实际采到的 batch `e_t`：

1. clone 当前 persistent state；
2. 分别应用每个合法 assimilation operator；
3. 在 `shadow_reward_probe` 上评估；
4. 得到：

\[
U_t(e_t,a)
=
J_{shadow}(A_a(S_t,e_t))
-J_{shadow}(S_t)
-\lambda_r R_{ret}
-\lambda_c C(a,e_t)
\]

其中 λ 沿跑前协议冻结。

这张 action-utility vector 用于训练 **assimilation critic**。

注意：

- 这里只对**已经真实采到的 batch**计算所有 assimilation counterfactual；
- 不允许对所有未采 candidate 先跑 shadow oracle 再挑 acquisition；
- 因而它不是 R6 Exact AVA 的“枚举 candidate → exact shadow 选最大”做法。

## 8.2 Acquisition reward

对已经采到的 batch 定义：

\[
V_t(e_t)=\max_a U_t(e_t,a)
\]

它表示：

> **这批经验在当前 agent state 下，经最合适的当前 assimilator 处理后，最多能带来多少未来能力提升。**

用 `(state, acquisition family, observed experience features, V_t)` 更新 acquisition critic。

这正面解决 R7 的问题：不能再用“最终有没有碰巧打开 C2”给 acquisition 分配稀疏 credit。

## 8.3 Policy architecture

禁止再用独立线性 bandit。

实现：

```text
shared state encoder
    ├── acquisition value ensemble Q_acq(S, family)
    └── assimilation utility head Q_assim(S, e, action)
```

建议：

- shared MLP：2–3 层，hidden 128/256；
- acquisition 使用 bootstrap ensemble（5 heads）提供 epistemic uncertainty；
- online selection 用 Thompson / UCB on learned nonlinear critic；
- assimilation 用 predicted `argmax_a Q_assim`；
- 每轮用真实 counterfactual U 向量监督更新 assimilation head；
- acquisition 只用被选 family 的真实 `V_t` 更新，保持 bandit feedback 语义。

不需要正式训练大 LLM controller；先验证结构化 critic 能否解决 R7 暴露的 credit problem。

## 8.4 State features

至少包括：

### capability/task-family state

- per-family observed success / failure rate；
- uncertainty / posterior variance；
- acquired unique count；
- recent improvement slope；
- last-k utility statistics；
- time since last sampled；

### memory state

- memory size；
- MemRL Q utility summary；
- retrieval hit / use statistics；

### skill state

- current skill version；
- validation score；
- accepted / rejected edit count；
- recent skill marginal gain；

### budget / cost

- remaining acquisition budget；
- cumulative tool / token / environment steps；

### acquired experience features

assimilation head 可以额外读取：

- task family；
- success/failure；
- failure type；
- trajectory length；
- tool/action histogram；
- trajectory embedding；
- novelty vs persistent stores。

禁止放：future label、future task id、gold action sequence。

---

# 9. R8 Stage-1 conditions

同一 frozen split，seeds `0,1,2`。

建议：

```text
6 rounds × batch 8 = 48 acquired experiences
full future eval @ 0 / 16 / 32 / 48
```

条件至少包括：

| ID | condition | acquisition | assimilation |
|---|---|---|---|
| M0 | `expolicy_r8` | learned nonlinear acquisition critic | learned heterogeneous assimilator |
| B0 | `passive_discard` | passive | discard |
| B1 | `passive_memrl` | passive | always MemRL |
| B2 | `passive_skillopt` | passive | always SkillOpt |
| B3 | `passive_best_fixed` | passive | calibration 上最强 fixed operator |
| B4 | `random_joint` | random family | random legal assimilator |
| A1 | `learn_acq_fixed_assim` | learned | best fixed assimilator |
| A2 | `passive_learn_assim` | passive | learned assimilator |
| A3 | `no_feedback` | same architecture | no controller update |
| A4 | `reward_shuffle` | learned | shuffle utility↔action labels |
| O0 | `hindsight_oracle` | actual acquired stream | per-step best counterfactual assimilation；diagnostic only |

如果第三方实现允许，增加：

```text
MAGE-style dual-bandit baseline
```

要求在**同样 action space / same persistent substrate**上实现，而不是把 MAGE 整个图框架搬来后做不可比对照。

---

# 10. 必须先验证的三个 Gate

## Gate H — Heterogeneous Assimilation Identifiability

在 calibration / pilot counterfactual matrix 上：

1. 至少两个 operator 各自 unique-best ≥20% states；
2. oracle routing > best fixed ≥0.03 macro；
3. 至少两个 task family 的 best operator 不同；
4. best operator 不能在 >80% states 上统治；
5. utility 差异不能主要来自人工 cost penalty。

**Gate H 不过：停止。**

## Gate C — Credit Learnability

leave-one-seed-out / leave-one-family-out：

- assimilation predicted utility Spearman ≥0.40；
- top-1 action accuracy ≥ majority +15 pp；
- acquisition critic 对 observed `V_t` 的 rank correlation ≥0.30；
- reward_shuffle 显著破坏上述指标。

Gate C 不过，不准直接上更大 seed；先检查 representation / state aliasing。

只允许一次 representation redesign：

```text
structured features
→ + trajectory embedding + recurrent/history encoder
```

第二次仍不过则 **discard current controller family**。

## Gate M — Method

主指标：untouched `future_test` macro success。

Stage-1 必须同时满足：

1. `mean(M0) >= strongest_non_oracle + 0.03`；
2. 3/3 seed 均高于 strongest non-oracle；
3. `M0 > A1` ≥0.02：learned assimilation 有独立贡献；
4. `M0 > A2` ≥0.02：learned acquisition 有独立贡献；
5. `M0 > no_feedback` ≥0.02；
6. `M0 > reward_shuffle` ≥0.02；
7. oracle-joint 与 M0 之间仍有合理 headroom，但 M0 至少关闭 ≥40% oracle gap；
8. improvement 不由单一 task family 独占。

如果只赢 AULC、不赢 final future，不算 PASS。

如果只赢 passive、但输给 best fixed，不算 PASS。

如果只在某一个 family 上赢，不缩 claim，直接 FAIL。

---

# 11. Stage-2

只有 Stage-1 `CASE_A_METHOD_PASS` 才允许：

1. seeds 扩到 0–9；
2. paired bootstrap 95% CI；
3. randomize task order / stream order；
4. 至少一种 distribution shift；
5. 第二公开环境。

最近关于 self-improving agent fragility 的工作指出，多 run 与 task-order shuffle 会显著改变结论，因此 Stage-2 必须把顺序敏感性当主协议，而不是附录 sanity check。

第二环境不能只是 ALFWorld 的另一个 split。优先：

```text
WebShop
ScienceWorld
LifelongAgentBench
```

选择标准仍然是：必须先存在 ≥2 个经过验证、非全局统治的 persistent assimilation operator。

---

# 12. Code audit：开跑前必须完成

建立：

```text
scripts/r8/audit_r7_structural.py
scripts/r8/build_alfworld_splits.py
scripts/r8/audit_r8_leakage.py
scripts/r8/audit_assimilator_isolation.py
scripts/r8/audit_counterfactual_clone.py
scripts/r8/run_backend_validation.py
scripts/r8/run_r8_stage1.py
scripts/r8/analyze_r8_stage1.py
```

测试至少覆盖：

### clone isolation

- counterfactual A1 不能污染 A2；
- skill file/version、memory DB、RNG、cost state 均独立；
- commit 只允许 chosen action 写回 live state。

### leakage

- future_test IDs / instructions 不进入 acquisition critic；
- 不进入 SkillOpt update / gate；
- 不进入 MemRL update；
- 不进入 shadow reward；
- 不进入 early stopping。

### resume

- split hash / code hash / third-party commit 不一致时拒绝 resume；
- seed RNG 状态可复现；
- 已完成 round 不重复 commit persistent update。

### backend fidelity

- MemRL adapter 的 update / retrieve 路径与官方实现一致；
- SkillOpt 的 bounded edit / validation gate 语义不被静默改掉；
- 所有 local patch 进入 `THIRD_PARTY_PATCH_AUDIT.md`。

### real environment

禁止：

```text
mock ALFWorld
直接根据 gold plan 构造成功 trajectory
把 failure 改写成 success
用 future task 做 SkillOpt validation
```

所有环境交互必须走真实 environment step。

---

# 13. GPU / 工程约束

沿用当前约束：

- 优先只用 GPU **4 / 5 / 6 / 7**；
- 不杀其他用户进程；
- 每个 `(condition, seed)` 独立 lock；
- `--resume` 必须 hash-safe；
- watchdog 只杀本轮自身 PID；
- 第三方 backend reproduction 与 ExPolicy 主跑目录分开。

建议目录：

```text
outputs/r8_preflight/
outputs/r8_backend_validation/
outputs/r8_hetero/
```

---

# 14. 运行顺序

严格按下面顺序执行，不准并行跳 gate。

## Phase 1 — R7 structural audit

```bash
python scripts/r8/audit_r7_structural.py \
  --r7_dir outputs/r7_closed_loop \
  --out outputs/r8_preflight
```

生成并人工/Agent 阅读：

```text
R7_CASE_AUDIT.md
R7_COUNTERFACTUAL_ASSIM_MATRIX.csv
R8_STRUCTURAL_AUDIT.md
R8_STRUCTURAL_AUDIT.json
```

如果 `R7_ENV_REUSABLE`，可以在原 SearchEnv 上进入 controller redesign，但仍需先跑 Gate H/C。

如果 `R7_ENV_STRUCTURALLY_DEGENERATE`，进入 Phase 2。

## Phase 2 — third-party backend validation

先完成官方复现 smoke，再通过 adapter 接入统一 ALFWorld runner。

```bash
python scripts/r8/build_alfworld_splits.py \
  --seed 20260821 \
  --out outputs/r8_backend_validation/split

python scripts/r8/run_backend_validation.py \
  --conditions no_update,memrl,skillopt \
  --seeds 0,1,2 \
  --gpus 4,5,6,7
```

分析：

```bash
python scripts/r8/analyze_backend_validation.py \
  --input outputs/r8_backend_validation \
  --out outputs/r8_backend_validation/BACKEND_GATE
```

只有 `HETEROGENEOUS_ASSIMILATION_PASS` 才进入 Phase 3。

## Phase 3 — R8 Stage-1

```bash
python scripts/r8/audit_r8_protocol.py \
  --config configs/r8_hetero.yaml

pytest -q tests/r8

python scripts/r8/run_r8_stage1.py \
  --config configs/r8_hetero.yaml \
  --seeds 0,1,2 \
  --gpus 4,5,6,7 \
  --resume

python scripts/r8/analyze_r8_stage1.py \
  --run_dir outputs/r8_hetero
```

---

# 15. 输出文件

至少生成：

```text
R8_NOVELTY_AUDIT.md
R7_CASE_AUDIT.md
R8_STRUCTURAL_AUDIT.md
THIRD_PARTY_PATCH_AUDIT.md
SPLIT_AUDIT.md
BACKEND_VALIDATION.md
R8_COUNTERFACTUAL_UTILITY.csv
R8_CONTROLLER_DIAGNOSTICS.csv
R8_STAGE1_MASTER.csv
R8_LEARNING_CURVES.csv
R8_PER_FAMILY.csv
R8_ABLATIONS.csv
R8_COST.csv
R8_GATE_DECISION.json
R8_MASTER_SUMMARY.md
```

每个主结果必须记录：

```text
seed
split hash
git hash
third-party commit
model path/hash
condition
round
selected acquisition family
acquired IDs
assimilation action
counterfactual U vector
committed state delta
shadow reward
retention delta
future score only at eval checkpoint
tokens/tools/env steps/wall
```

---

# 16. 决策树：效果不好时怎么处理

## CASE A — `METHOD_PASS`

条件：Gate H/C/M 全部通过。

下一步：

- 扩 10 seed；
- task-order shuffle；
- 第二公开环境；
- 加 MAGE-style closest baseline；
- 写正式 paper method。

可主张：

> A learned Experience Policy jointly controls which environment experience to acquire and which persistent learning operator should consume it, improving future capability beyond passive acquisition and any fixed assimilator under the same interaction budget.

## CASE B — `SUBSTRATE_DEGENERATE`

如果 assimilation oracle ≈ best fixed，或者一个 operator 全局统治：

- **停止该 benchmark**；
- 不训练 controller；
- 换 environment / assimilator substrate；
- 不写“虽然 adaptive routing 没必要，但我们的框架仍然通用”。

这是实验设计失败，不是 claim 缩小的理由。

## CASE C — `CREDIT_NOT_LEARNABLE`

如果 Gate H 有真实 oracle gap，但 critic 学不会：

- 先读错分 cases；
- 做一次 representation redesign；
- 加 trajectory embedding / recurrent state；
- frozen split 上重跑 3 seed。

第二次仍失败：

```text
DISCARD_CONTROLLER_FAMILY
```

不要继续调 hidden size / lr / UCB coefficient。

## CASE D — `CONTROLLER_FAIL`

如果 critic 能预测 utility，但 online ExPolicy 仍输 best fixed：

- 说明 exploration / optimization 方法失败；
- 当前 nonlinear-bandit controller 直接 discard；
- 可以另开一个明确不同的 policy-learning formulation，但不能把结果写成“ExPolicy 对 passive 有小幅收益”。

## CASE E — `ONLY_ONE_ASSIMILATOR_WORKS`

如果最终只有 SkillOpt 或只有 MemRL 有用：

- adaptive assimilation 的核心实验条件不成立；
- 不把项目改写成 SkillOpt scheduler / memory admission paper；
- 换 testbed，或者停止 joint ExPolicy 方向。

## CASE F — `NOVELTY_COLLISION`

若 MAGE / SESA / 新出现工作已经实现 1–4 的完整交叉：

- 停止；
- 不通过术语区分强行续命；
- 重新选择核心科学问题。

---

# 17. 明确禁止的“救结果”方式

本轮尤其禁止：

- 在 R7 上只把 Linear TS 换 MLP，然后把 AULC 提升写成主贡献；
- 调 C2 sampling prior；
- 直接给 controller capability gold label；
- 人工限制“一次只能激活一个 skill”来制造 assimilation trade-off；
- 给 wrong skill 额外人工 penalty；
- 通过加大 skill cost 让 all-skills 变差，然后声称 learned policy 更优；
- 只报最好 seed；
- future_test 用于 SkillOpt validation；
- future_test 用于 controller early stopping；
- Gate M 失败后改成“只追求 sample efficiency”；
- 只在 ALFWorld 某一个 task family 成功后改成 capability-specific claim；
- 把 MemRL / SkillOpt / counterfactual credit 本身写成 ExPolicy novelty。

---

# 18. 我们希望 R8 最终形成的真正论文级证据

如果本路线成功，论文的核心证据链应当是：

```text
1. Existing self-evolving agents optimize one persistence substrate
   or couple curriculum with a fixed learning mechanism.

2. In a heterogeneous persistent-state setting,
   the value of an experience depends on what the current agent can do with it.

3. No fixed acquisition policy or fixed assimilator is globally optimal.

4. ExPolicy learns:
      what experience to acquire next
      ×
      how to turn that experience into persistent capability.

5. Learned joint control beats:
      passive acquisition
      best fixed assimilator
      acquisition-only learning
      assimilation-only learning
      random joint control
      closest dual-controller baseline
   on untouched future capability.

6. NoFeedback / RewardShuffle collapse the gain,
   demonstrating that the improvement comes from feedback-driven learning
   rather than lucky activation or a prewired curriculum.

7. The effect survives task-order shuffle and a second public environment.
```

只有这条证据链成立，才把 ExPolicy 写成：

> **Existing agents learn from experience; ExPolicy learns what to experience and how to learn from it.**

而不是把一个失败的完整目标缩成较弱的 memory / skill / sampling claim。

---

# 19. R8 最重要的执行原则

> **先证明 decision space 有真实、非退化、可辨识的 utility heterogeneity，再训练 controller。**

R7 已经证明 pipeline 可以运行；R8 不再需要为 pipeline engineering 付主要实验预算。

本轮真正要避免的是：

```text
在一个 fixed policy 已经达到 substrate ceiling 的环境里，
继续优化 learned policy，最后只能把“更快碰到 C2”包装成方法贡献。
```

如果当前 substrate 不支持完整 ExPolicy，就换 substrate；如果 controller 在可辨识环境中仍学不会，就 discard controller；如果 novelty 已被完整抢占，就换研究问题。

**禁止通过 narrow claim 保住方法。**
