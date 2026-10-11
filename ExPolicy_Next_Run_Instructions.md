# ExPolicy 下一步运行指令：先修 Assimilation，再回测 Acquisition

> 仓库：`/data/ppnm/expolicy`
> 当前模型：`Qwen2.5-3B-Instruct`
> 当前结论：E0 证明 benchmark 有理想 frontier，但 E1/E2/E3/E4 表明当前主要瓶颈不是 acquisition，而是 **experience 尚未被 carrier 稳定吸收成 capability gain**。
> 因此本轮禁止直接进入 E5–E8，也不要扩大到 5 stage × 200 task × 3 seed。先完成下面四个步骤。

---

# 0. 本轮目标

本轮只回答一个核心问题：

> **在同 capability、同实体关系/桥接模式的 near-transfer 任务上，一批真实经验经过正确 carrier 吸收后，能否稳定产生正的 realized utility？**

如果答案仍然是否定的，则 ExPolicy 当前问题在 assimilation implementation / experience representation，而不是 acquisition policy。

本轮执行顺序：

```text
R1 Near-Transfer Utility Diagnostic
    ↓
R2 Executable Skill Repair
    ↓
R3 Verified-Success Batch Weight Internalization
    ↓
R4 Rebuild Carrier Utility Matrix
    ↓
仅当出现稳定正 U：
    R5 Re-run Active Acquisition
    ↓
    R6 Re-run Routing / Learned Assimilation
```

---

# 1. R1：Near-Transfer Utility Diagnostic

## 1.1 不改 acquisition，不训练 policy

固定从 E0/E2 已有真实轨迹中抽取 experience。

优先选择：

```text
S1 multi-hop / query decomposition
S2 temporal / entity disambiguation
S4 cross-document synthesis
```

不要用 S0/S3 作为主要学习目标，因为它们已经接近或达到 ceiling。

## 1.2 构造 capability-matched near-transfer probe

当前 E1/E3 最大的问题之一是 probe 太远或粒度太粗。

为每条 / 每组 experience 构造：

```text
experience capability
    ↓
same capability
same reasoning pattern
same bridge structure
different held-out entity / document / surface form
```

例如：

```text
multi-hop CEO → company → city
```

对应 near-transfer probe 应仍是：

```text
entity A → intermediate relation → entity B
```

但绝对不能复用相同答案、相同 document pair 或原题。

## 1.3 每组不要只放 1 条 experience

改为 batch intervention。

建议第一轮：

```text
batch size = 8 experiences
3 capability buckets
3 independent batches / bucket
```

即至少：

```text
S1 multi-hop: 3 × 8
S2 temporal: 3 × 8
S4 cross-doc: 3 × 8
```

总计 72 条 experience，可复用已有真实轨迹。

每个 batch 使用：

```text
near probe = 16 tasks
far probe = 16 tasks
retention probe = 16 tasks
```

避免 1/6、1/8 粒度直接决定结论。

---

# 2. 先修 Episodic Memory 的写入和检索

当前 episodic memory 不能只依赖通用 embedding 相似度。

请检查当前 memory write / retrieve path，并改成 capability-aware retrieval。

每条 episodic experience 至少保存：

```text
question
normalized task pattern
capability
failure_type
intermediate entities
bridge relation
search queries
successful evidence path
failed action
correction
final outcome
```

检索 query 不再只使用：

```text
question embedding
```

而至少组合：

```text
question
+ capability
+ failure signature
+ inferred task pattern
```

实现 hybrid retrieval score：

```text
score =
a * semantic_similarity
+ b * capability_match
+ c * failure_match
+ d * structural_pattern_match
+ e * recency_or_success_weight
```

第一版可以使用真实统计特征加权，不要求 learned retriever。

## 2.1 Memory 注入必须结构化

不要直接把整条历史 trajectory 塞入 prompt。

retrieved memory 统一压缩成：

```text
Situation
Relevant failure
Useful evidence pattern
Corrected procedure
Applicability
Warning / when not to use
```

限制每条 memory token budget。

## 2.2 R1-Episodic 验收

比较：

```text
No memory
Old episodic retrieval
New capability-aware episodic retrieval
```

要求输出：

```text
near EM
far EM
retention EM
tokens
retrieval hit rate
memory actually used rate
```

最关键指标：

```text
near_transfer_gain
```

Go 条件建议：

```text
至少 2/3 capability buckets 的平均 near gain > 0
且跨 batch 方向一致
```

若仍无正 gain，先不要进入 learned policy。

---

# 3. R2：把 Skill 修成真正可执行的 Procedure

当前 skill 主要是 search/open 列表式 prompt 注入，E3b 中平均最差。

不要继续把这种 skill 用于 Carrier Matrix。

## 3.1 新 Skill Schema

至少：

```python
Skill(
    name,
    capability,
    preconditions,
    query_template,
    decomposition_steps,
    tool_sequence,
    state_variables,
    evidence_requirements,
    stopping_condition,
    answer_synthesis_rule,
    verifier,
    failure_recovery,
)
```

Skill 必须真正控制 runtime，而不仅是附加自然语言建议。

## 3.2 先实现三个真实 skill

### Skill A：Multi-hop Bridge Search

```text
identify target relation
→ decompose into hop-1 / hop-2
→ search hop-1 entity
→ verify bridge entity
→ search hop-2 relation
→ collect both supporting docs
→ synthesize only after both hops verified
```

### Skill B：Temporal Resolution

```text
extract temporal constraint
→ identify candidate facts
→ attach timestamps
→ discard temporally invalid evidence
→ rank valid evidence
→ answer
```

### Skill C：Cross-Document Synthesis

```text
decompose requested fields
→ retrieve evidence independently
→ bind evidence to entity
→ verify consistency
→ synthesize structured answer
```

## 3.3 Skill 必须从真实 experience 中生成/修订

不要 hard-code benchmark answer。

可以用 experience abstraction 生成 skill proposal，但必须：

```text
proposal
→ schema validation
→ execute on validation tasks
→ compare against no-skill
→ accept / reject
```

只有 validation gain >= 0 的 skill 才进入 persistent skill store。

## 3.4 R2 验收

对每个 skill：

```text
16 near-transfer tasks
16 far-transfer tasks
16 retention tasks
```

比较：

```text
No skill
Old skill
Executable skill
```

必须记录：

```text
tool sequence compliance
evidence completion rate
near EM gain
far EM gain
retention delta
token overhead
```

如果 executable skill 仍然没有正 near gain，则 skill carrier 暂时从主实验移除。

---

# 4. R3：Verified-Success Batch Weight Internalization

不要对单条 experience 做 LoRA。

不要用未验证失败 experience 直接 SFT。

## 4.1 数据只来自 verified success / corrected success

优先：

```text
S1 multi-hop
S2 temporal
S4 cross-doc
```

每个 capability 至少构造：

```text
32–64 verified trajectories
```

如果真实 success 不足，可使用：

```text
failed trajectory
→ branch / correction
→ environment verified successful trajectory
```

但最终用于 weight training 的必须是 **真实执行后验证成功** 的 corrected trajectory。

## 4.2 Training example 不只保留 final answer

至少包含：

```text
task
reasoning-relevant state
search / decomposition decisions
verified evidence
correct tool-use sequence
final synthesis
```

同时避免把 benchmark-specific document ID、答案 shortcut 作为可泛化监督。

## 4.3 LoRA 安全修复

在运行 weight carrier 前先修 runner：

```text
base_model
→ load fresh adapter
→ evaluate
→ unload adapter
→ destroy model instance if needed
→ clear CUDA cache
→ next intervention
```

禁止多个 adapter 叠加载污染后续 evaluation。

每个 intervention 保存：

```text
base model hash
adapter path
training data ids
seed
steps
learning rate
before metrics
after metrics
```

## 4.4 第一轮 LoRA 配置

先用小规模稳定配置，不追求最大提升：

```text
LoRA rank: 8 or 16
epochs: 1–2
small LR
batch training
early stop by validation
```

不要使用 5–20 step 单条 SFT 作为 weight carrier 代表。

## 4.5 R3 验收

每个 capability batch 都测：

```text
near 16+
far 16+
retention 16+
```

至少比较：

```text
No update
Episodic batch
Semantic batch
Executable skill
LoRA batch
```

---

# 5. R4：重建 Carrier Utility Matrix

只有 R1–R3 的 carrier 已经各自进入“合理实现”后，重新做 Carrier Matrix。

## 5.1 Matrix 单位从 single experience 改成 experience batch

推荐：

```text
12 batches
每 batch = 8 experiences
```

覆盖：

```text
multi-hop
temporal
cross-doc
```

每个 capability 至少 4 batches。

Carriers：

```text
episodic
semantic
skill
weight
```

如果 skill 某 capability 没有通过 R2 validation，则该格标记：

```text
N/A
```

不要用坏 skill 强行填矩阵。

## 5.2 Utility 先去掉 token cost 做机制判断

当前评测成本本身会压过 1–2 题的 gain。

先同时报告两个指标：

```text
U_capability =
Δnear
+ 0.5 * Δfar
- 0.5 * max(0, -Δretention)

U_efficiency =
U_capability
- normalized_train_and_inference_cost
```

论文机制判断优先看：

```text
U_capability
```

系统效率再看：

```text
U_efficiency
```

不要把“评测自身 token 成本”混进 experience intrinsic utility。

## 5.3 R4 Go 条件

至少满足：

```text
A. >= 50% batches 存在某 carrier U_capability > 0

B. best carrier 随 capability / batch 发生变化，
   且不是一个 fixed carrier 全局统治

C. oracle routing 明显优于 best fixed carrier

D. 差异幅度 > probe 单题粒度，不只是 ±1 task
```

若仍然失败：

```text
暂停 Adaptive Assimilation 主线
```

不要训练 learned router。

---

# 6. R5：只有 R4 通过后，重做 Active Acquisition

此前 E2 的 acquisition 没有机会转化成 capability gain，因为固定 assimilation 无效。

这次使用 R4 中验证最有效的 assimilation 机制。

## 6.1 Acquisition 比较

```text
passive
reward_seeking
uncertainty
frontier
failure_replay
```

先不训练 learned ranker。

预算：

```text
48 experiences
```

比原 24 略大，但保持所有方法一致。

## 6.2 关键改动：capability-conditioned assimilation

例如：

```text
S1 multi-hop → skill / weight（以 R4 结果为准）
S2 temporal → semantic / weight
S4 unstable → episodic / active replay
```

不要再所有 acquisition 共用一个明显无效的 episodic-only assimilation。

## 6.3 Gate B 新标准

必须满足：

```text
Active method future gain > 0
且
Active > Passive
```

最好至少两个 independent seeds 方向一致，再进入 learned acquisition。

---

# 7. R6：只有 R4/R5 通过后再训练 Learned Router

训练 supervision 使用新的：

```text
Batch × Carrier Utility Matrix
```

不要使用旧 E3b 的 12×3 弱负值矩阵。

比较：

```text
best fixed carrier
heuristic
random
learned router
hindsight oracle
```

核心指标：

```text
routing regret
realized capability utility
retention
```

只有：

```text
learned router < best-fixed regret
```

才恢复 E5 2×2。

---

# 8. 本轮不要做

明确禁止优先执行：

```text
E5 2×2 factorial
E6 separate vs joint
E7 full EvolveLoop
E8 5×200×3 seeds
复杂 GRPO Experience Policy
第二个 benchmark
更大模型
harness self-modification
```

这些都等 Assimilation 能产生稳定正 utility 后再做。

---

# 9. 建议具体执行顺序

```bash
cd /data/ppnm/expolicy
source .venv/bin/activate
```

然后由 Cursor Agent 依次完成并运行：

```text
Step 1
实现 capability-matched near-transfer probe builder

Step 2
修 episodic write/retrieve：
question + capability + failure + structural pattern

Step 3
运行 R1 episodic batch utility

Step 4
实现 executable skill runtime

Step 5
运行 R2 skill validation

Step 6
修 LoRA adapter load/unload/rebuild 安全路径

Step 7
构造 verified-success batch dataset

Step 8
运行 R3 weight batch internalization

Step 9
运行 R4 batch-level Carrier Utility Matrix

Step 10
根据 R4 自动生成 GO_NO_GO_R4.md
```

不要在 R4 前运行 R5/R6。

---

# 10. 必须新增的脚本

请实现并实际运行：

```text
scripts/build_near_transfer_probes.py
scripts/run_r1_near_transfer.py
scripts/run_r2_executable_skill.py
scripts/build_verified_success_batches.py
scripts/run_r3_weight_batch.py
scripts/run_r4_batch_carrier_matrix.py
scripts/analyze_r4_go_nogo.py
```

每个脚本都必须支持：

```text
--seed
--model
--gpu / device
--output_dir
--resume
```

---

# 11. 输出目录

统一：

```text
outputs/r1_near_transfer/
outputs/r2_executable_skill/
outputs/r3_weight_batch/
outputs/r4_batch_carrier_matrix/
```

每个目录保存：

```text
config.yaml
manifest.json
trajectories.jsonl
interventions.jsonl
before_metrics.json
after_metrics.json
summary.json
```

R4 额外保存：

```text
carrier_matrix.csv
carrier_matrix.json
GO_NO_GO_R4.md
```

---

# 12. 本轮最终停止点

本轮做到 R4 即停止并汇报。

只有当 R4 明确显示：

```text
experience can be absorbed
+
carrier choice materially changes realized capability gain
```

才继续 R5。

否则不要通过增加任务量、seed 或更复杂 policy 掩盖问题。

---

# 13. 最终需要回答的科学问题

本轮不是要证明完整 ExPolicy。

只需要回答：

> **之前的 Gate A/B/C 失败，究竟是因为“Experience Policy 假设不成立”，还是因为当前 experience assimilation 实现根本没有把经验转化为可迁移能力？**

如果 near-transfer、executable skill、verified-success batch LoRA 能产生稳定正 utility，那么此前 E2 失败应解释为：

```text
Acquisition was evaluated behind an ineffective assimilation bottleneck.
```

这时再回到 Active Acquisition 才有科学意义。

如果这些修复后仍没有稳定正 utility，则应该重新审视：

```text
experience representation
benchmark learning protocol
以及 ExPolicy 的核心假设
```

而不是继续扩大实验规模。
