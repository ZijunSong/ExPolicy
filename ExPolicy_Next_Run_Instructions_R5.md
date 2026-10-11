# ExPolicy 下一轮执行指令：R5 Active Acquisition under Validated Executable Assimilation

> Repo：`/data/ppnm/expolicy`  
> 模型：`/data/ppnm/models/Qwen2.5-3B-Instruct`  
> World：Helion，`data/world/helion.json`，world seed=13  
> 本轮目标：**只修复并重新检验 Gate B：在已经验证有效的 executable-skill assimilation 后面，Active Acquisition 是否优于 Passive Acquisition。**  
> 本轮禁止进入 R6 learned router / E5–E8。

---

## 0. 当前结论与下一步唯一科学问题

上一轮已经得到：

- R4：12 个 batch 中 11/12 存在正 `U_capability`；
- executable skill 的 mean `U_capability = +0.237`，11/12 为 unique best；
- episodic / semantic / LoRA 大多数仍为负；
- hindsight oracle 仅比 always-skill 高约 `0.003`，因此当前 **没有训练 learned carrier router 的空间**；
- 旧 E2 的 Active Acquisition 是在 ineffective episodic assimilation 后面测的，因此 E2 阴性结果不能直接视为 Active Acquisition 假设被证伪。

所以本轮只回答一个问题：

> **当 experience 能通过 executable procedure 真正改变 agent capability 后，在相同 experience budget 下，主动选择“学什么”是否比被动采样带来更高的未来能力收益？**

不要扩展成“完整 ExPolicy 已成立”，也不要同时引入 learned routing、weight internalization 或新的 carrier。

---

# 1. R5-0：先做 Acquisition → Assimilation 因果通路审计

这是本轮最重要的前置条件。

## 1.1 禁止直接这样跑

禁止：

```text
不同 acquisition policy
    ↓
采不同 experience
    ↓
所有方法最终直接挂载完全相同、预先存在的 3 个 executable skills
    ↓
评测
```

如果 skill 与 acquired experience 无关，那么 acquisition 对最终 agent 没有真实因果影响，R5 即使跑完也不能回答 Gate B。

## 1.2 必须满足的 causal wiring

每个方法必须从同一个 clean agent 开始：

```text
clean agent
  ↓
acquire experiences under policy π_acq
  ↓
experience batch
  ↓
SkillAssimilator.fit(experience_batch)
  ↓
产生 / 更新 / 激活 executable skill artifact
  ↓
future evaluation
```

实现并检查统一接口，例如：

```python
skill = SkillAssimilator.fit(
    experiences=batch,
    capability=capability,
    frozen_primitives=...
)
```

要求：

1. 可以复用 R2 已验证的 executable runtime primitives 和 schema；
2. **不能**让 `fit()` 对任意 batch 都直接返回完全相同的常量 skill；
3. acquired experience 至少必须决定：
   - 该 capability 的 skill 是否被激活；
   - skill version / evidence support；
   - 至少一个非平凡 runtime configuration、procedure branch 或 primitive composition；
4. 不得使用 gold answer、gold document id、future probe、validation probe 来构造 skill；
5. 可以从真实 trajectory 中使用：
   - tool sequence；
   - success / failure；
   - query decomposition pattern；
   - evidence completion；
   - bridge / temporal / cross-document trace structure；
   - 已执行后才可观测的 verifier 信号；
6. 所有 skill artifact 必须保存 provenance：
   - source experience ids；
   - capability；
   - compiler version；
   - config；
   - activation reason；
   - validation result。

### 最低可接受实现

如果当前代码还没有真正的 skill induction，允许本轮采用“**冻结 procedure primitive library + experience-conditioned activation/configuration**”：

- R2 的三个 procedure primitive 可以提前存在于代码中，但默认全部 disabled；
- 每个 target capability 累积至少 **8 条真实 acquired experiences** 后，才能调用 `fit()`；
- `fit()` 从这 8 条或更多 trajectories 中统计/选择 procedure configuration，并生成独立 artifact；
- 没有足够 experience 的 capability 不得直接启用对应 skill。

`batch_size=8` 与 R4 的 carrier batch 定义保持一致，不要在看到结果后修改阈值。

## 1.3 必须新增 causal audit

实现：

```text
scripts/audit_r5_causal_path.py
```

至少自动检查：

- clean agent 初始无 active skill；
- 不同 acquisition batch 的 skill provenance 不同；
- 删除某 capability 的 acquired experiences 后，该 capability skill 不会被激活；
- shuffled / empty batch 不允许偷偷加载 R2 全局 skill；
- probe task id / gold answer / gold doc id 未进入 skill artifact；
- 相同 batch + 相同 seed 能复现完全相同 artifact；
- 不同 batch 至少在 activation/provenance/config 中有真实差异。

若 causal audit FAIL，**停止，不运行 R5 主实验。**

---

# 2. 冻结实验变量

R5 主实验开始后冻结以下变量，不允许根据中途结果调整：

```text
model = Qwen2.5-3B-Instruct
world_seed = 13
retrieval = local Helion BM25
experience_budget = 48
skill_activation_batch = 8
acquisition methods =
    passive
    reward_seeking
    uncertainty
    frontier
    failure_replay
seeds = 0, 1, 2
```

禁止：

- mock LLM；
- mock retrieval；
- 伪造成功 trajectory；
- 把失败改写为成功后再计为真实 acquired experience；
- 根据 R5 probe 结果改 skill；
- 为不同 acquisition 方法使用不同 assimilation 实现；
- 为某个方法单独调 prompt / budget / threshold；
- 读取 held-out probe 的 gold 信息进行 acquisition；
- 运行 learned carrier router。

---

# 3. 先构建严格隔离的 R5 数据划分

实现：

```text
scripts/build_r5_splits.py
```

每个 seed 必须产生三个互斥集合：

```text
acquisition_pool
future_probe
retention_probe
```

目标 capability：

```text
S1 multi-hop / query decomposition
S2 temporal / disambiguation
S4 cross-document synthesis
```

要求：

1. `future_probe` 与 `acquisition_pool` 的 task id 严格不重叠；
2. 尽可能避免：
   - 相同 gold answer；
   - 相同 gold document pair；
   - 仅实体名替换的近重复模板；
3. `retention_probe` 使用 S0；
4. 三个 seed 使用不同采样，但所有 acquisition 方法在同一 seed 下共享：
   - 同一个 acquisition candidate pool；
   - 同一个 future probe；
   - 同一个 retention probe；
5. acquisition 过程中不得有 replacement；
6. 如果 S4 数据不足以满足严格隔离，脚本必须 **fail closed 并明确报告缺口**，禁止静默复用 R4/R1 中已暴露的 probe。

建议目标：

```text
future_probe:
    S1 = 16
    S2 = 16
    S4 = 16
    total = 48

retention_probe:
    S0 = 24
```

若现有 Helion 无法给出 16 个完全独立 S4，则允许降低到统一的 `12 × 3 capability = 36`，但必须在运行前固定，不能看结果后改。

输出：

```text
outputs/r5_acquisition_skill/splits/seed_{0,1,2}.json
outputs/r5_acquisition_skill/SPLIT_AUDIT.md
```

---

# 4. Acquisition policy：沿用 E2 定义，不重新发明算法

本轮目的不是靠重新设计 policy 获胜，而是修复 assimilation 后重测 E2。

保留五种方法：

### passive

从当前可选 candidate 中按冻结随机策略采样。

### reward_seeking

优先选择当前估计 immediate reward 高的任务。

它是重要 negative/control baseline：E2 中它 immediate reward 最高，但严重偏向 S0/S3 易题。

### frontier

优先选择当前 capability frontier / low capability score 区域。

### uncertainty

使用当前已有 uncertainty 定义，不在本轮修改 scoring 公式。

### failure_replay

使用 E2 已修复版本：基于 recent failure capability + frontier + low capability score，禁止重新出现“failure type 当 capability”的旧 bug。

---

# 5. R5 主循环：统一 Skill Assimilation

实现或整理为单一 runner：

```text
scripts/run_r5_acquisition_skill.py
```

逻辑必须是：

```python
agent = clean_agent()

for t in range(48):
    x = acquisition_policy.select(common_candidate_state)
    traj = agent.run_real_task(x)
    experience_store.add(traj)

    if should_refit(t, capability):
        skill = SkillAssimilator.fit(real_experiences_for_capability)
        validate_skill_without_using_future_probe(skill)
        if accepted:
            agent.skill_library.install(skill)

final_evaluate(agent)
```

## 5.1 Skill refit 时机

固定在 acquisition budget checkpoint：

```text
8 / 16 / 24 / 32 / 40 / 48
```

但某 capability 只有在累计支持数 `>=8` 时才允许首次激活。

后续可以用新增 experience 更新该 skill；必须产生新 version，不得覆盖 provenance。

## 5.2 Skill acceptance

只允许使用：

- acquired experience 内部训练/构造信息；
- 从 acquisition pool 内预先划出的 skill-validation subset，或 trajectory-level verifier；

不得使用最终 future probe。

保存：

```text
skill_v001.json
skill_v002.json
...
```

并记录：

```text
n_support
support_success_rate
support_failure_rate
capability
procedure/config
tool compliance
evidence completion
source experience ids
```

---

# 6. 为了只测 Acquisition，主结果不要再混入 Routing failure

R4 中 executable skill 的 retention 大跌，主要来自“把 capability-bound procedure 强制套到 S0”。

R5 的科学问题是 Acquisition，而不是“模型能否自己选择何时调用 skill”。

因此主实验采用 **evaluation-controlled capability dispatch**：

- S1 probe 仅允许对应 S1 skill；
- S2 probe 仅允许对应 S2 skill；
- S4 probe 仅允许对应 S4 skill；
- S0 retention 不应用这些 skill；
- 若该 acquisition 方法没有通过真实 acquired experiences 激活对应 skill，则该 capability 使用 base agent。

这相当于一个 **oracle capability dispatcher**，仅用于隔离 Acquisition 变量。

必须在结果中明确：

> R5 证明/否定的是 acquisition under validated assimilation，不证明 deployable learned routing。

可以额外跑一个 secondary stress test：blind/global skill application，但它 **不进入 Gate B 主判定**。

---

# 7. 评测指标

每个 method × seed 至少记录：

## 7.1 Acquisition 行为

```text
immediate_reward
selected task distribution
selected capability distribution
success/failure distribution
tokens
tool_calls
# experiences per target capability
time/budget when each skill is first activated
# active skills at budget 48
```

## 7.2 Capability outcome

对最终 agent 评：

```text
S1 future EM
S2 future EM
S4 future EM
macro future EM
retention EM
```

同时保存 clean base agent 在同一 probe 上的结果。

主增益：

```text
ΔFuture = Future_EM_after - Future_EM_base
```

并继续保留与 R4 一致的 capability utility 作为辅助指标：

```text
U_capability
```

但在 oracle capability dispatch 下，S0 不应错误触发 hard-capability skill，因此 retention penalty 预期应显著小于 R4。

## 7.3 Learning curve

至少在 budget：

```text
0 / 16 / 32 / 48
```

做 checkpoint evaluation，画：

```text
future EM vs acquisition budget
U vs acquisition budget
# activated hard-capability skills vs budget
```

这样不要只比较最终一个点。

---

# 8. Gate B：运行前固定判据

不要使用“挑最好看的一个 active method”作为成功标准。

Primary active methods：

```text
frontier
uncertainty
failure_replay
```

Control：

```text
passive
reward_seeking
```

定义 Gate B PASS 需要同时满足：

1. 三个 primary active methods 中，至少 **2/3 的 3-seed mean ΔFuture > passive**；
2. 至少一个 primary active method：
   - `mean(ΔFuture_active - ΔFuture_passive) >= 0.05`；
   - 且在 **至少 2/3 seeds** 上优于 passive；
3. 该方法自身相对 clean base 的 mean `ΔFuture > 0`；
4. 提升不能只来自单一 capability：
   - 至少 **2/3 target capabilities** 的 mean gain 非负；
5. 结果幅度不能只等于单题翻转：
   - 报告 bootstrap CI 或 seed variance；
   - 若差异完全落在单题粒度内，判为 **AMBIGUOUS**，不判 PASS。

### Gate B FAIL

若：

- active 与 passive 基本持平；
- reward_seeking 反而最好且机制只是易题 exploitation；
- active 只改变采样分布却不能更早/更多激活有用 skill；
- 三 seed 方向反复；
- gain 仅来自 probe 单题翻转；

则 Gate B 判 FAIL。

---

# 9. 必做消融：证明提升确实来自“更会采经验”

只做以下必要消融，不扩实验树。

## A. Skill disabled

保持 acquisition 结果不变，但 final eval 禁用 acquired skills：

```text
active acquisition + no assimilation
```

预期如果 R5 机制成立：

```text
Active + Skill > Active + NoSkill
```

否则 acquisition gain 可能只是采样本身改变了短时 agent state。

## B. Shuffled experience → skill

保持 acquisition 数量不变，将 capability 对应 experience 打乱后再 fit skill。

目的：证明不是“budget 到 8 自动送 skill”。

```text
correct experience-conditioned skill
vs
shuffled experience-conditioned skill
```

## C. Fixed pre-installed skill（diagnostic only）

所有方法直接预装三个 R2 skill。

如果此时五种 acquisition method 几乎无差别，反而证明本轮 causal design 是必要的。

该实验只作 sanity，不进入主表。

---

# 10. 推荐 GPU 调度

当前有多卡时优先按 method × seed 并行，不让不同方法共享可变状态。

例如先跑 seed=0：

```text
GPU 2: passive
GPU 3: reward_seeking
GPU 4: uncertainty
GPU 5: frontier
GPU 6: failure_replay
```

然后 seed=1 / seed=2。

如果部分卡不可用，自动重新分配，但每个独立 run：

- 单独 output directory；
- 单独 agent state；
- 单独 skill library；
- 单独 random seed；
- 不共享 mutable cache / memory / adapter。

---

# 11. 输出目录

统一：

```text
outputs/r5_acquisition_skill/
├── splits/
├── passive/
│   ├── seed_0/
│   ├── seed_1/
│   └── seed_2/
├── reward_seeking/
├── uncertainty/
├── frontier/
├── failure_replay/
├── ablations/
├── R5_SUMMARY.json
├── R5_PER_SEED.csv
├── R5_LEARNING_CURVE.csv
├── R5_SKILL_ACTIVATION.csv
├── R5_GATE_B.md
└── R5_REPORT.md
```

每个 run 必须保存：

```text
config.json
selected_experiences.jsonl
trajectories.jsonl
skill_versions/
checkpoint_metrics.jsonl
final_metrics.json
token_stats.json
```

---

# 12. 建议执行顺序

先进入仓库：

```bash
cd /data/ppnm/expolicy
```

先做现有代码回归：

```bash
pytest -q
```

然后按顺序完成：

```text
[1] 实现 / 检查 SkillAssimilator.fit
[2] 实现 audit_r5_causal_path.py
[3] causal audit PASS
[4] build_r5_splits.py
[5] split audit PASS
[6] seed=0 跑五种方法
[7] 自动检查输出完整性、skill provenance、无 probe leakage
[8] seed=1、2
[9] 跑三项必要 ablation
[10] analyze_r5_gate.py
[11] 写 R5_GATE_B.md 与 R5_REPORT.md
[12] 更新 expolicy-result-record.md
[13] 停止，等待人工查看结果
```

建议最终补一个统一入口：

```bash
python scripts/run_r5_suite.py \
  --model /data/ppnm/models/Qwen2.5-3B-Instruct \
  --world data/world/helion.json \
  --world-seed 13 \
  --budget 48 \
  --seeds 0 1 2 \
  --methods passive reward_seeking uncertainty frontier failure_replay \
  --output outputs/r5_acquisition_skill
```

如果仓库现有 CLI 结构不同，保持项目现有风格实现等价入口，不要为了匹配这条命令破坏已有 runner。

---

# 13. 本轮停止条件

无论结果好坏，R5 完成后都先停。

## 若 Gate B PASS

只允许得出：

> 在可用的 experience-conditioned executable assimilation 下，active acquisition 能以相同 experience budget 获得更高 future capability gain。

下一轮再考虑：

- 更大 world / 更严格 held-out；
- acquisition × assimilation joint test；
- 是否需要新的 carrier diversity。

**仍然不要直接训练 R6 learned carrier router**，因为 R4 中 always-skill 与 oracle 只差约 0.003。

## 若 Gate B FAIL

说明在修复 assimilation 后，Active Acquisition 仍没有可靠优势。

此时不要继续堆 acquisition heuristic。应重新判断 ExPolicy 的核心是否应从：

```text
learn what experience to collect
```

转向：

```text
learn what executable capability/procedure to externalize,
compose, activate and retire
```

即把研究中心收缩到 capability-conditioned self-harness / skill lifecycle，而不是强行维护“active acquisition 一定有效”的假设。

## 若 AMBIGUOUS

只允许做一次预注册的统计加强：

- 扩 probe；
- 或增加 seeds；

禁止一边看结果一边修改 acquisition score / skill threshold。

---

# 14. 写入实验记录时必须明确的措辞

如果 R5 成功，也不要写：

```text
ExPolicy is validated.
Adaptive assimilation is validated.
Learned routing is validated.
```

只写：

```text
R5 repairs the acquisition test behind a validated executable assimilation path.
Gate B is supported / not supported under this controlled setting.
```

当前 learned carrier routing 和 weight internalization 仍然没有被支持。

---

# 15. 最终交付要求

Cursor Agent 完成本轮后必须给出：

1. 改动文件列表；
2. causal wiring 说明；
3. split leakage audit；
4. 五种 acquisition × 3 seeds 完整结果；
5. per-capability learning curve；
6. skill activation timeline；
7. 三项必要 ablation；
8. Gate B PASS / FAIL / AMBIGUOUS；
9. 对旧 E2 结论是否需要修订；
10. 更新后的 `expolicy-result-record.md`。

**不要在 R5 后自动继续 R6、E5、E6、E7、E8。**
