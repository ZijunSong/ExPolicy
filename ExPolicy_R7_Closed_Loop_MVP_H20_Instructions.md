# ExPolicy R7：Closed-Loop Experience Policy MVP — H20 执行指令

> **本轮唯一首要目标：把 ExPolicy 的核心方法 pipeline 做成真正完整的闭环。**
>
> 不再把“experience 是否从零诱导出 unique skill”作为前置门槛。本轮允许直接使用已经验证可工作的 frozen skill bank。  
> ExPolicy 的研究对象是 **experience policy 本身**：  
>
> 1. 当前状态下，下一步应该获得什么 experience；
> 2. 获得 experience 后，应该如何利用它；
> 3. 利用结果如何改变 persistent capability state；
> 4. future capability 的真实变化如何反过来更新下一轮 Experience Policy。
>
> 本轮目标不是立刻拿到最终论文 SOTA，而是第一次把以下闭环在真实 Search 环境中完整跑通：
>
> ```text
> Current Agent State
>         ↓
> Capability Diagnosis / State Encoder
>         ↓
> Acquisition Policy
>         ↓
> Real Environment Experience
>         ↓
> Experience Features / Batch Summary
>         ↓
> Assimilation Policy
>         ↓
> Persistent Capability Intervention
>         ↓
> Shadow Future Evaluation
>         ↓
> Realized Utility / Reward
>         ↓
> Experience Policy Update
>         ↓
> New Agent State
>         └──────────────→ next round
> ```
>
> **只有上面整个循环真实执行并且 policy 参数/后验确实随 realized utility 更新，才算本轮 pipeline 打通。**

---

## 0. 本轮结论边界

### 0.1 本轮要证明什么

本轮只验证下面三个层次的问题。

### P1. Engineering closure

ExPolicy 是否已经成为一个真正运行的 stateful closed-loop controller，而不是一串离线脚本：

- agent 有显式 persistent state；
- policy 根据当前 state 选择 acquisition；
- acquisition 经过真实 SearchEnv；
- policy 在看到真实 experience 后选择 assimilation action；
- assimilation action 真正改变 agent；
- shadow probe 测到 action 后的 future capability；
- 这个 realized gain 被写回 policy；
- 下一轮 acquisition / assimilation 分布因此发生变化。

### P2. Policy responsiveness

Experience Policy 是否真的“在学”：

- 相同初始 prior 下，不同 realized utility 序列会导致不同 posterior / action probability；
- 删除 reward feedback 后，policy 不再适应；
- 打乱 reward 与 action 的对应关系后，policy 的行为/性能应退化；
- 不允许把预定义 frontier / q×p / count threshold 伪装成 learned policy。

### P3. Initial method signal

在同样 interaction budget 下，完整闭环 ExPolicy 是否已经开始优于：

- passive acquisition + fixed assimilation；
- random acquisition + random assimilation；
- passive + always-episodic；
- passive + fixed skill gating；
- strongest non-oracle baseline。

P3 是本轮的**方法信号**，不是 P1/P2 的工程通过条件。

---

## 0.2 本轮不再要求什么

以下问题本轮全部降级为后续研究，不得阻塞 R7：

1. experience 是否能从零生成一个 unique executable skill；
2. real packet 是否一定优于 empty / shuffled packet；
3. 是否需要 skill induction / skill editing；
4. 是否需要训练 actor 权重；
5. 是否需要重新证明 RASA / frontier / AVA；
6. 是否要做 10 seed / Bamboogle / 新 benchmark；
7. 是否要把 skill 本身包装成 ExPolicy 的 novelty。

**Skill 在本轮只是一个可用 assimilation action / capability intervention。**

---

# 1. 硬性执行约束

## 1.1 机器与 GPU

服务器：当前 8×H20-3e 节点。

默认只使用：

- GPU 4：retriever encoder / shared retrieval service；
- GPU 5：R7 seed 0；
- GPU 6：R7 seed 1；
- GPU 7：R7 seed 2。

GPU 0–3 默认视为他人占用，**禁止 kill、抢卡或修改他人进程**。

开跑前必须：

```bash
nvidia-smi
ps -ef | grep -E "retrieval|vllm|ray|python" | grep -v grep
```

如果 4–7 中有新增的非本项目任务，优先调整本项目映射，不允许清理别人进程。

---

## 1.2 Repo

主 repo 继续使用：

```text
/data/ppnm/expolicy
```

新代码单独放在：

```text
expolicy/closed_loop/
scripts/r7/
tests/r7/
```

新产物只能写：

```text
outputs/r7_closed_loop/
```

不得覆盖：

```text
outputs/r5_*
outputs/r6_*
/data/ppnm/skillrl_esi/outputs/esi_mvp/
```

建议创建分支：

```bash
cd /data/ppnm/expolicy
git checkout -b r7-closed-loop
```

如果分支已存在则直接切换，不重复创建。

---

## 1.3 模型与 Search 环境

Actor：

```text
/data/ppnm/models/Qwen2.5-3B-Instruct
```

要求：

- actor 权重冻结；
- 正式 eval `temperature=0`；
- acquisition rollout 必须经过真实 Search environment；
- 禁止 mock LLM；
- 禁止 gold-as-hit；
- 禁止直接把 gold answer / gold document 塞给 runtime。

Search 优先复用已经通过 R5.3–R6 审计的真实环境：

```text
Wikipedia 2018
E5-base-v2
21,015,324 indexed docs
```

如果现有 `:18080` 服务仍可用，先做健康检查；若不可用，由 R7 wrapper 启动同一份 index，不改变 corpus。

必须记录：

```text
corpus path
index path
retriever model
port
PID
startup command
SHA256 / version information
```

---

# 2. 本轮最重要的设计原则：Skill bank 冻结，Experience Policy 才是变量

## 2.1 不再重新做 skill induction

R7 不使用 ESI real/empty/shuffle 作为方法主体。

直接建立：

```text
FrozenCapabilityBank
```

第一版可以复用已经在公开 Search 环境中验证过的三个 executable procedure：

- C1 `bridge_decomposition`
- C2 `comparison_alignment`
- C3 `long_chain_search`

这些 skill 在 R5.3 中已经分别完成 validation，因此 R7 把它们当作：

> **pre-existing capability interventions**

而不是：

> experience-induced skills

运行时不要直接动态 import 旧 R5.3 Python 对象。

需要将 skill 定义复制/迁移到：

```text
expolicy/closed_loop/frozen_skill_bank/
```

并生成：

```text
FROZEN_SKILL_BANK_MANIFEST.json
```

至少记录：

```json
{
  "skill_id": "...",
  "source_path": "...",
  "source_git_head": "...",
  "sha256": "...",
  "capability": "...",
  "validated_before_r7": true,
  "r7_mutable": false
}
```

R7 开始后 skill byte/hash 不允许变化。

---

## 2.2 Skill runtime 必须是真执行

激活 skill 后，skill 必须真正改变工具调用流程，例如：

```text
Search
Open
Search
Open
Answer
```

所有工具动作仍然通过真实 Search environment。

禁止：

- skill 直接返回 gold answer；
- skill 根据 task id 查答案；
- skill 根据 dataset filename 选择答案；
- skill 绕过 SearchEnv 直接访问 retriever；
- skill 运行时读取 final future_test label。

---

# 3. R7 的核心方法：两阶段 Experience Policy

第一版不要上复杂 PPO / GRPO。

本轮优先目标是：

> **闭环正确 > 算法复杂。**

因此实现一个可在线更新、可审计、状态依赖的两阶段 policy：

```text
π_acq(c | S_t)
π_assim(a | S_t, E_t)
```

其中：

- `π_acq` 决定下一批 experience 从哪个 capability 获取；
- `π_assim` 在真实 batch 采完以后，决定怎么利用这批 experience。

第一版 canonical controller 使用：

> **Bayesian Linear Thompson Sampling / Linear Contextual Bandit**

原因：

- 少量在线 transition 就能更新；
- 不需要先构造大量 policy training episode；
- 每次 posterior 更新可审计；
- 能直接验证“future utility 是否改变下一轮 decision”；
- 后续可无缝替换成 MLP actor-critic，而不改闭环协议。

不要把本轮写成 Thompson Sampling 方法论文。它只是第一版 ExPolicy controller。

---

# 4. Persistent Agent State

新增：

```python
ExPolicyState
```

必须可 JSON 序列化，并在每一轮完整落盘。

建议字段：

```python
@dataclass
class ExPolicyState:
    round_id: int
    budget_used: int

    # capability state
    capability_seen_count: Dict[str, int]
    capability_success_count: Dict[str, int]
    capability_failure_count: Dict[str, int]

    # current measured capability
    shadow_em_by_capability: Dict[str, float]
    shadow_macro: float
    retention_score: float

    # persistent intervention state
    active_skills: Dict[str, bool]
    episodic_memory_size: Dict[str, int]
    semantic_memory_size: Dict[str, int]

    # recent experience statistics
    last_batch_capability: str | None
    last_batch_success_rate: float | None
    last_batch_mean_searches: float | None
    last_batch_repeat_rate: float | None
    last_batch_premature_answer_rate: float | None

    # utility history
    last_realized_utility: float | None
    utility_ema_by_capability: Dict[str, float]
    utility_ema_by_assim_action: Dict[str, float]

    # policy state
    acquisition_posterior: ...
    assimilation_posterior: ...

    # cost
    total_tokens: int
    total_tool_calls: int
    total_wall_seconds: float
```

每轮保存：

```text
outputs/r7_closed_loop/<condition>/seedX/round_00/state_before.json
outputs/r7_closed_loop/<condition>/seedX/round_00/acquisition_decision.json
outputs/r7_closed_loop/<condition>/seedX/round_00/experience_batch.jsonl
outputs/r7_closed_loop/<condition>/seedX/round_00/assimilation_decision.json
outputs/r7_closed_loop/<condition>/seedX/round_00/state_after.json
outputs/r7_closed_loop/<condition>/seedX/round_00/reward.json
outputs/r7_closed_loop/<condition>/seedX/round_00/policy_update.json
```

---

# 5. 数据划分

## 5.1 不允许继续使用 final future_test 训练 policy

R7 必须重新切一个**独立 policy-learning protocol**。

数据至少分为：

```text
calibration
acquisition_pool
shadow_reward_probe
retention_probe
future_test
```

要求：

```text
task-id overlap = 0
exact-question overlap = 0
```

如果复用 HotpotQA / 2Wiki / MuSiQue：

- capability taxonomy 沿用：
  - C1 bridge_decomposition
  - C2 comparison_alignment
  - C3 long_chain_search
- 不能用 dataset name 作为 policy feature；
- policy 不允许读取 gold capability label of future_test。

建议规模：

```text
calibration:          60 / capability
acquisition_pool:    240 / capability
shadow_reward_probe:  60 / capability
retention_probe:      60
future_test:         150 / dataset
```

若现有本地数据不足，保持比例并 fail-closed 记录实际规模，不复制题目补数。

冻结：

```text
split_seed = 20260819
```

输出：

```text
outputs/r7_closed_loop/audit/SPLIT_R7.json
outputs/r7_closed_loop/audit/SPLIT_AUDIT.md
```

---

# 6. Capability Diagnosis

本轮不再把 hand-crafted frontier score 当 acquisition policy。

但 ExPolicy 仍然需要显式 state observation。

在正式 round 0 前，只允许用 `calibration` 得到：

```text
base EM by capability
base failure rate
mean searches
repeat-query rate
premature-answer rate
router confidence
```

这些只是 `S_0` 的 observation，不直接决定 acquisition。

禁止重新构造：

```text
4s(1-s)
q*p
manual transfer_potential
frontier coefficient
hand-written weighted score
```

---

# 7. Acquisition Policy

## 7.1 High-level action space

第一版只让 acquisition policy 选择 capability：

```text
A_acq = {C1, C2, C3}
```

在选定 capability 内，实例选择采用**冻结的无偏规则**：

```text
uniform without replacement
```

可选增加一个简单 diversity tie-break，但必须在所有方法共享，且不能使用 reward / future utility。

这样 R7 真正学习的是：

> 把有限 experience budget 分配到哪个 capability。

而不是重新发明 instance-level heuristic。

---

## 7.2 Acquisition state feature

每个 capability action `c` 的 contextual feature 建议由以下组成：

```text
[1,
 current_shadow_em(c),
 failure_rate_so_far(c),
 seen_fraction(c),
 recent_utility_ema(c),
 skill_active(c),
 episodic_count(c),
 semantic_count(c),
 remaining_budget_fraction,
 last_batch_was_c,
 round_fraction]
```

禁止包含：

```text
future_test result
gold answer
dataset id shortcut
task id
R5.3/R5.4 hand-written frontier score
```

---

## 7.3 Acquisition controller

实现接口：

```python
class AcquisitionPolicy:
    def select(self, state, available_capabilities, rng) -> AcquisitionDecision:
        ...

    def update(self, state_before, capability, realized_utility, state_after):
        ...
```

canonical 实现：

```text
Linear Thompson Sampling
```

每个 capability 一个 Bayesian linear head，prior 固定：

```text
mean = 0
covariance = α I
noise variance = fixed
```

超参写死到 config，不根据 final test 调。

建议：

```yaml
acquisition:
  type: linear_thompson
  prior_precision: 1.0
  reward_noise: 0.05
```

如数值不稳定只允许修 implementation bug，不做 performance sweep。

---

# 8. Experience Collection

每一轮：

```text
batch_size = 8
```

总预算：

```text
6 rounds × 8 = 48 unique experiences
```

流程：

```text
π_acq chooses capability c_t
        ↓
uniform sample 8 unseen tasks from c_t
        ↓
run CURRENT agent state on all 8
        ↓
record full real trajectories
```

这里的 **CURRENT agent** 很重要：

如果上一轮已经激活 C2 skill，那么下一轮遇到 C2 acquisition task 时应该运行带当前 persistent state 的 agent。

因此：

> 禁止把所有 acquisition experience 预先用 base actor 缓存后直接复用。

允许 cache 的只有：

- retriever responses；
- deterministic environment I/O；
- 完全相同 `(task, persistent_state_hash)` 的重复恢复。

正常 R7 acquisition task 本身无 replacement，因此不应出现依赖静态 base rollout 的伪闭环。

---

# 9. Experience Batch Summary

采完 8 条以后构造：

```python
ExperienceBatchSummary
```

至少包含：

```text
capability
n
success_rate
failure_rate
mean_reward
mean_searches
mean_turns
repeat_query_rate
premature_answer_rate
retrieval_miss_rate
answer_synthesis_failure_rate
unique_query_ratio
evidence_completion_rate
token_cost
tool_cost
```

不得包含：

```text
future_test labels
future_test task ids
gold future answers
```

该 summary 是 assimilation policy 的 observation。

---

# 10. Assimilation Policy

这是 R7 相对之前 acquisition-only 路线最关键的新闭环。

## 10.1 Action space

第一版使用四个真实、可执行 action：

```text
A0 = DISCARD
A1 = EPISODIC_MEMORY
A2 = SEMANTIC_MEMORY
A3 = ACTIVATE_EXISTING_SKILL
```

含义：

### DISCARD

不修改 persistent capability state。

### EPISODIC_MEMORY

把该 batch 的真实轨迹写入 episodic store。

后续推理时：

- 按 query / capability 检索；
- 只注入训练侧 trajectory summary；
- 禁止注入 gold future answer；
- memory retrieval 必须真正影响 prompt。

### SEMANTIC_MEMORY

把该 batch 压缩成不包含原题答案的 capability-level semantic note。

允许 LLM summarizer，但必须：

- 只看 acquired batch；
- 不看 future_test；
- 记录完整输入输出；
- summarizer model 固定。

如果现有 semantic memory 实现已经稳定，可复用；否则本轮只要求功能正确，不优化它。

### ACTIVATE_EXISTING_SKILL

激活对应 capability 的 frozen executable skill。

注意：

- skill 不是从这 8 条经验“学出来”的；
- experience 是触发 policy decision 的证据；
- policy 决定当前是否值得把这个 capability intervention 加入 persistent state。

若该 skill 已经 active，则该 action masked，不允许重复选择伪造 transition。

---

## 10.2 Assimilation feature

建议输入：

```text
state features
+
batch summary
+
candidate action indicator
```

至少包括：

```text
current_shadow_em(cap)
current_skill_active(cap)
batch_success_rate
batch_failure_rate
batch_repeat_query_rate
batch_premature_answer_rate
batch_mean_searches
batch_evidence_completion
current episodic count
current semantic count
remaining budget
recent utility of this action
```

---

## 10.3 Assimilation controller

接口：

```python
class AssimilationPolicy:
    def select(self, state, experience_summary, valid_actions, rng):
        ...

    def update(
        self,
        state_before,
        experience_summary,
        action,
        realized_utility,
        state_after,
    ):
        ...
```

canonical 实现同样采用：

```text
Linear Thompson Sampling
```

不要复用旧 `HeuristicRouter`。

不要写：

```text
failure -> episodic
success -> skill
count >= 8 -> activate
```

这类规则可以做 baseline，但不能进入 R7 主方法。

---

# 11. 每一轮真实闭环

实现统一入口：

```python
run_r7_episode(...)
```

每个 round 必须严格按下面顺序：

```text
1. Load current ExPolicyState S_t
2. Encode state
3. π_acq selects capability c_t
4. Sample 8 unseen tasks from c_t
5. Run current agent in real SearchEnv
6. Build ExperienceBatchSummary E_t
7. π_assim selects action a_t
8. Apply a_t to persistent agent
9. Evaluate SHADOW probe under updated state
10. Evaluate retention probe
11. Compute realized utility r_t
12. Update π_acq with r_t
13. Update π_assim with r_t
14. Serialize policy posterior + new state S_{t+1}
15. Continue to next round
```

任何一步失败：

```text
fail closed
```

禁止跳过并继续。

---

# 12. Realized Utility

本轮统一 reward：

\[
r_t
=
\Delta J_{\text{shadow}}
-
0.5 \cdot \max(0,-\Delta J_{\text{retention}})
-
\lambda_{\text{tok}} C_{\text{tok}}
-
\lambda_{\text{tool}} C_{\text{tool}}
\]

其中：

```text
ΔJ_shadow
= shadow_macro(S_{t+1}) - shadow_macro(S_t)
```

建议第一版：

```yaml
reward:
  retention_weight: 0.5
  token_cost_per_million: 0.01
  tool_cost_per_100: 0.005
```

成本只做轻微 tie-break，不能再出现 E1 那种 8-task probe 一题 gain 与 token cost 同量级、把所有 utility 压负的问题。

Shadow probe 至少：

```text
60 / capability
```

保证单题粒度足够细。

---

# 13. Policy update 必须真实影响下一轮

每轮输出：

```text
policy_before.json
policy_after.json
```

必须至少记录：

```text
posterior mean
posterior covariance
action scores
sampled Thompson values
chosen action
reward
parameter delta norm
```

审计必须验证：

```text
parameter_delta_norm > 0
```

且 round `t+1` 的 action score 与 round `t` 相比确实使用更新后的 posterior。

禁止：

- 记录了 reward 但 policy 不更新；
- policy 更新了但下一轮仍读取旧 checkpoint；
- 每轮重新初始化 prior；
- seed 间共享 online posterior。

---

# 14. Baselines

本轮至少跑以下条件。

## B0 Passive + No Assimilation

```text
acquisition = uniform random
assimilation = DISCARD
```

## B1 Passive + Episodic

```text
acquisition = uniform random
assimilation = always EPISODIC
```

## B2 Passive + Fixed Skill Gate

为公平对照旧路线：

```text
acquisition = uniform random
assimilation = CountOnly K=8 -> activate frozen skill
```

仅作 baseline。

## B3 Random Joint Policy

```text
acquisition = random capability
assimilation = random valid action
```

## B4 Oracle All-Skills Upper Bound

在 episode 开始前打开所有 skill：

```text
C1+C2+C3
```

它不是公平 baseline，只报 upper bound。

## M0 R7 ExPolicy

```text
learned π_acq
+
learned π_assim
+
realized-utility feedback
```

---

# 15. 关键因果消融

本轮闭环是否真实，主要靠下面三个消融验证。

## A. No-Feedback

完整执行 acquisition / assimilation，但：

```text
policy.update(...)
```

关闭。

posterior 永远保持 prior。

目标：

> 证明 M0 的行为变化不是仅由 budget / state mask 自动产生。

---

## B. Reward-Shuffle

完整执行真实环境和真实 assimilation。

但 policy update 时：

```text
r_t
```

在 round 内随机错配给另一个 action transition。

最终 future_test 仍正常评。

目标：

> 证明 action 与 realized utility 的对应关系确实被 policy 使用。

---

## C. Assimilation-Fixed

保留 learned acquisition：

```text
π_acq
```

但 assimilation 固定为：

```text
CountOnly K=8
```

目标：

> 判断完整 ExPolicy 的增益是否只来自 acquisition。

---

# 16. Seed 与运行规模

Stage-1：

```text
seed = 0, 1, 2
budget = 48
round = 6
batch = 8
```

正式 future_test：

- episode 开始前评一次；
- budget 16 / 32 / 48 评；
- final 作为主结果。

Shadow probe：

- 每一轮都评；
- 只用于 policy reward；
- 不作为最终论文结果。

Final future_test：

- **绝不进入 policy update。**

---

# 17. GPU 调度

推荐：

## GPU 4

只跑 retriever encoder / shared service。

例如：

```text
CUDA_VISIBLE_DEVICES=4
retrieval_server
```

不要同时在 GPU4 起大 actor。

## GPU 5 / 6 / 7

主方法三 seed：

```text
GPU5 -> M0 seed0
GPU6 -> M0 seed1
GPU7 -> M0 seed2
```

M0 完成后，同一张卡顺序跑该 seed 的：

```text
B0
B1
B2
B3
NoFeedback
RewardShuffle
AssimilationFixed
```

B4 AllSkills 可以共用一次 deterministic evaluation，不必重复做无意义 rollout。

所有脚本必须支持：

```bash
--resume
```

但 resume 只能在：

```text
config hash
split hash
policy checkpoint hash
state hash
```

全部匹配时继续。

---

# 18. 需要新增的代码

建议目录：

```text
expolicy/
  closed_loop/
    __init__.py
    state.py
    experience.py
    reward.py
    agent_state.py

    policies/
      base.py
      acquisition_linear_ts.py
      assimilation_linear_ts.py
      random_policy.py
      fixed_policy.py

    assimilation/
      discard.py
      episodic.py
      semantic.py
      activate_skill.py

    frozen_skill_bank/
      manifest.json
      bridge_decomposition.py
      comparison_alignment.py
      long_chain_search.py

    runner/
      episode.py
      evaluator.py
      checkpoint.py

scripts/r7/
  build_r7_split.py
  audit_r7_split.py
  validate_frozen_skill_bank.py
  smoke_r7_closed_loop.py
  run_r7_condition.py
  run_r7_suite.py
  watchdog_r7.py
  analyze_r7.py
  audit_r7_causal_path.py

tests/r7/
  test_state_roundtrip.py
  test_policy_update_changes_posterior.py
  test_reward_not_future_test.py
  test_future_test_never_used_for_update.py
  test_current_state_affects_rollout.py
  test_skill_hash_frozen.py
  test_invalid_action_mask.py
  test_no_feedback_keeps_prior.py
  test_reward_shuffle_changes_credit.py
  test_resume_hash_guard.py
  test_no_gold_leakage.py
  test_round_transition.py
```

---

# 19. Pipeline audit

正式开 GPU 前必须生成：

```text
outputs/r7_closed_loop/audit/R7_PIPELINE_AUDIT.md
```

必须逐项 PASS：

1. real retriever reachable；
2. real SearchEnv step；
3. actor frozen；
4. split overlap=0；
5. final future_test 不进入 policy；
6. skill bank hash frozen；
7. acquisition policy 可更新；
8. assimilation policy 可更新；
9. state 可序列化；
10. memory action 真写入 persistent store；
11. active skill 真改变 runtime；
12. current agent state 影响下一轮 rollout；
13. reward 来自 shadow delta；
14. policy update 使用该 reward；
15. updated posterior 被下一轮读取；
16. no-feedback posterior 不变；
17. reward-shuffle 不读真实 pair；
18. task id / gold answer 不进入 policy feature；
19. dataset name 不进入 policy feature；
20. resume 不跨 config/split/state；
21. 所有 condition 共用同一 final test；
22. 所有 baseline budget 相同；
23. no mock / no gold-as-hit；
24. GPU 不抢占他人进程。

要求：

```text
24 / 24 PASS
```

才允许正式 Stage-1。

---

# 20. Smoke test

先只跑：

```text
1 seed
2 rounds
batch_size = 2
```

目标不是性能，只检查闭环。

必须人工/脚本确认：

```text
S0
→ acquisition decision
→ 2 real trajectories
→ assimilation decision
→ persistent state change
→ shadow eval
→ reward
→ posterior update
→ S1
→ next acquisition decision
```

如果第二轮 policy checkpoint 与第一轮完全相同：

```text
SMOKE FAIL
```

不得正式开跑。

输出：

```text
outputs/r7_closed_loop/smoke/SMOKE_TRACE.md
```

---

# 21. Stage-1 主结果表

`analyze_r7.py` 至少生成：

```text
R7_FINAL_METRICS.csv
R7_LEARNING_CURVES.csv
R7_ACTION_DISTRIBUTION.csv
R7_POLICY_POSTERIOR_TRACE.csv
R7_STATE_TRANSITIONS.csv
R7_COST.csv
R7_GATE_DECISION.json
R7_MASTER_SUMMARY.md
```

主表：

| condition | final macro | ΔFuture | AULC | #skills active | epi size | sem size | tokens | tools |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

还必须报告：

### Acquisition behavior

```text
每轮选了哪个 capability
最终 C1/C2/C3 experience 数量
action entropy
posterior evolution
```

### Assimilation behavior

```text
DISCARD 次数
EPISODIC 次数
SEMANTIC 次数
ACTIVATE_SKILL 次数
每个 capability 下动作分布
```

### Closed-loop trace

至少选一个 seed 做完整可视化：

```text
round
state before
acquisition
experience summary
assimilation
shadow delta
reward
policy update
state after
```

---

# 22. 本轮两个独立 Gate

本轮必须把“pipeline 是否闭环”和“方法是否已经强”分开判定。

---

## Gate P：PIPELINE_PASS

以下必须全部成立。

### P-A. Statefulness

至少一个 persistent intervention 在某轮改变，且后续 rollout 读取新状态。

### P-B. Acquisition adaptivity

`π_acq` posterior 在每次非零 update 后变化，并至少一个 seed 的 acquisition action distribution 相对 round 0 prior 发生可观变化。

建议判据：

```text
max TV(action_dist_t, action_dist_0) >= 0.10
```

### P-C. Assimilation adaptivity

`π_assim` posterior 真实更新，且 3 seed 合计至少出现两种不同 assimilation action。

不能 18/18 全选同一 action 后仍声称动态 routing 已建立。

### P-D. Reward closure

shadow Δ → reward → policy update → next-round action 的完整因果路径由日志和测试确认。

### P-E. Feedback ablation

NoFeedback 与 M0 的 posterior trace 不同；NoFeedback 保持 prior。

### P-F. Leakage audit

final future_test 完全不进入 policy。

如果：

```text
P-A ∧ P-B ∧ P-C ∧ P-D ∧ P-E ∧ P-F
```

则：

```text
PIPELINE_PASS
```

即使最终 performance 暂时不超过 strongest baseline，也算 ExPolicy 核心闭环第一次真正打通。

---

## Gate M：METHOD_SIGNAL_PASS

建议跑前冻结：

1. `mean(final_M0) >= max(B1, B2, B3) + 0.02`
2. 至少 `2/3 seed` M0 严格优于 strongest non-oracle baseline；
3. M0 AULC 至少高 `+0.01`；
4. M0 > NoFeedback；
5. M0 > RewardShuffle；
6. M0 > AssimilationFixed 或至少显示 learned assimilation 有独立正贡献。

全部通过：

```text
METHOD_SIGNAL_PASS
```

否则：

```text
PIPELINE_PASS_METHOD_AMBIGUOUS
```

或：

```text
PIPELINE_PASS_METHOD_FAIL
```

**不要因为 Method Gate 没过，就回头修改 pipeline 定义。**

---

# 23. 决策树

## CASE_A：PIPELINE_PASS + METHOD_SIGNAL_PASS

这是最理想结果。

下一步才允许：

- 将 linear TS 换成真正 learned neural Experience Policy；
- 增加 cross-benchmark；
- 接入外部 SkillRL / SESA skill bank；
- 做 10 seed；
- 做更丰富 assimilation action；
- 重新进入 ExPolicy 主论文实验。

---

## CASE_B：PIPELINE_PASS，但 Method Signal 未过

这是完全可接受的 R7 结果。

说明：

> ExPolicy closed-loop 方法已经真正成立为可运行系统，但当前 controller 还不够好。

下一步只优化：

```text
policy representation / credit assignment / exploration
```

不要回去：

- 重做 skill induction；
- 调 RASA；
- 调 frontier；
- 调 CountOnly K；
- 修改 final test；
- 改 skill 内容。

---

## CASE_C：PIPELINE_FAIL

只有以下情况才判 pipeline 未打通：

- policy 没有真正 update；
- update 后下一轮仍读旧 policy；
- assimilation action 没有改变 persistent state；
- acquisition 使用静态 cached base trajectory；
- final test 泄漏进 reward；
- skill runtime 绕过 SearchEnv；
- state 无法跨 round 持久化；
- feedback ablation 与主方法完全同构且检查到代码没用 reward。

CASE_C 时只修工程闭环，不讨论 paper performance。

---

# 24. 本轮必须避免的旧问题

## 24.1 不再把 count gate 当 ExPolicy

`count >= 8 -> activate skill`

只能作为 B2 baseline。

主方法里 skill 是否激活必须由：

```text
π_assim
```

决定。

---

## 24.2 不再把 hand-crafted score 当 policy

禁止把：

```text
frontier
q*p
RASA
AVA shadow utility
manual transfer potential
```

直接作为 M0 acquisition score。

它们可以作为历史 baseline，但不是 R7 Experience Policy。

---

## 24.3 不再让 final future_test 参与 online decision

所有 online learning reward 来自：

```text
shadow_reward_probe
```

Final future_test 只用于 checkpoint evaluation 与 final report。

---

## 24.4 不再把 skill provenance 变成主问题

R7 frozen skill bank 只要：

```text
真实
可执行
固定
不泄漏
```

即可。

如果后续需要更强 novelty，再替换为外部公开 skill bank。

---

# 25. 推荐实现顺序

Cursor / Coding Agent 严格按下面顺序做，不要并行发散。

## Step 1

实现：

```text
ExPolicyState
checkpoint
split
audit
```

跑纯 CPU tests。

---

## Step 2

迁移 frozen skill bank，跑：

```text
validate_frozen_skill_bank.py
```

确认 SearchEnv 下 skill 真有行为差异。

---

## Step 3

实现 assimilation 四 action，并分别单独 smoke：

```text
discard
episodic
semantic
activate_skill
```

---

## Step 4

实现：

```text
AcquisitionPolicy
AssimilationPolicy
LinearThompsonPolicy
```

用 synthetic numeric reward 做**单元测试**可以，但正式 experiment 禁止 mock environment。

---

## Step 5

实现：

```text
run_r7_episode()
```

只跑 2×2 smoke。

---

## Step 6

完成 24/24 pipeline audit。

---

## Step 7

正式跑：

```text
M0 seed 0/1/2
```

先不要同时跑全部 baseline。

确认：

```text
posterior changes
state changes
next-round actions read new posterior
```

---

## Step 8

再跑：

```text
B0/B1/B2/B3
NoFeedback
RewardShuffle
AssimilationFixed
```

---

## Step 9

统一 analyze。

禁止在看完结果后：

- 调 reward weight；
- 调 prior；
- 改 split；
- 改 skill；
- 改 batch size；
- 改 Gate。

如果结果差，按 Case B/C 记录。

---

# 26. 建议命令入口

最终应支持类似：

```bash
cd /data/ppnm/expolicy

python scripts/r7/build_r7_split.py \
  --seed 20260819

python scripts/r7/audit_r7_split.py

python scripts/r7/validate_frozen_skill_bank.py

pytest -q tests/r7

python scripts/r7/smoke_r7_closed_loop.py

python scripts/r7/audit_r7_causal_path.py

python scripts/r7/run_r7_suite.py \
  --conditions expolicy \
               passive_noop \
               passive_episodic \
               passive_count_skill \
               random_joint \
               no_feedback \
               reward_shuffle \
               assimilation_fixed \
  --seeds 0 1 2 \
  --budget 48 \
  --batch-size 8 \
  --gpus 5 6 7 \
  --resume

python scripts/r7/analyze_r7.py
```

具体 CLI 可按 repo 风格调整，但必须保证等价能力。

---

# 27. Watchdog

新增：

```text
scripts/r7/watchdog_r7.py
```

监控：

```text
last completed round
last state checkpoint
last policy checkpoint
GPU utilization
retriever health
log mtime
```

如果 15 min 无任何：

```text
new rollout
new evaluation
new checkpoint
```

才标记：

```text
STALL
```

允许 kill **本项目自己的 worker** 后 `--resume`。

禁止 kill shared retriever 前不记录 PID 和 ownership。

---

# 28. 最终必须写回 expolicy-result-record.md

R7 完成后，在 canonical record 增加：

```text
## R7 Closed-Loop ExPolicy
```

至少记录：

### Setting

- repo / git head
- split hash
- model
- retriever
- skill-bank hash
- policy type
- reward
- budget
- GPU
- wall time

### Pipeline result

```text
PIPELINE_PASS / FAIL
```

逐项写 Gate P。

### Method result

```text
METHOD_SIGNAL_PASS / AMBIGUOUS / FAIL
```

逐项写 Gate M。

### 主表

最终 macro / ΔFuture / AULC / cost。

### 行为分析

- acquisition 分布；
- assimilation 分布；
- posterior trace；
- state transition；
- feedback ablation。

### 结论

严格区分：

> “闭环已经打通”

和：

> “当前 policy 已经优于 baseline”。

---

# 29. 本轮真正的停止条件

R7 不应该再无限追加实验。

本轮只回答两个问题：

### 问题 1

> **ExPolicy 的核心闭环是否终于成为一个真实、stateful、feedback-driven 的运行系统？**

### 问题 2

> **第一版 learned Experience Policy 是否已经出现优于 fixed/passive policy 的信号？**

拿到这两个答案后立即停止。

不在本轮：

- 追 SOTA；
- 扩 10 seed；
- 调几十组超参；
- 重新做 skill induction；
- 重新做 frontier/RASA/AVA；
- 加第四个 benchmark；
- 做正式 RL actor training。

---

# 30. 给 Coding Agent 的最终一句话

> **不要再围绕“experience 是否生成了 skill”优化局部模块。请把 ExPolicy 当作一个真正作用于 agent 学习过程的上层 policy：它必须在每一轮基于当前 persistent capability state 主动选择下一批真实 experience，观察真实 trajectory 后动态选择 assimilation action，应用该 action 改变 agent state，再用独立 shadow future probe 得到 realized utility，并用该 utility 更新下一轮 acquisition 与 assimilation policy。R7 的首要交付物不是更高分数，而是这个闭环第一次完整、真实、可审计地运行起来。**
