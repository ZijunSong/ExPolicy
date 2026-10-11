# ExPolicy 下一阶段实验路线图

> 当前状态：Phase 1 已完成真实 LLM 验收；Qwen2.5-3B-Instruct 在 Helion Archive 50-task 测试上达到 86% EM，真实 BM25 检索，100 条真实轨迹已可形成 capability profile，episodic memory 已验证可检索同类经验。  
> 下一阶段目标：**不要立刻扩成论文规模，而是先用最小成本验证 ExPolicy 的四个核心论文命题是否成立。**

---

# 0. 总体判断：下一步先做“机制验伪”，不是直接做 full-scale

目前最危险的问题不是代码不完整，而是：

> **当前静态 Agent 已经达到 86% EM，Helion Archive 对 Qwen2.5-3B-Instruct 是否仍然具有足够的学习空间？**

如果 benchmark 大部分任务已经被 static agent 掌握，那么：

- active acquisition 很难比 passive 显著更好；
- carrier routing 的差异会被 ceiling effect 吞掉；
- weight internalization 可能只带来 1–2 个点的噪声；
- experience utility 的分布会大量集中在 0；
- joint Experience Policy 很难学到稳定 signal。

因此实验顺序应该是：

```text
Benchmark / Capability Audit
        ↓
Realized Utility Sanity Check
        ↓
Acquisition-only Pilot
        ↓
Assimilation-only Pilot
        ↓
Joint Experience Policy Pilot
        ↓
Online Evolution Loop
        ↓
Full 5-stage × 200-task × 3-seed
```

---

# 1. Experiment 0：Benchmark Capability-Gap Audit

## 目的

首先回答：

> Helion Archive 是否真的存在足够多的“当前模型不会，但通过 experience 可以学会”的 frontier tasks？

这是后面所有实验成立的前提。

## 1.1 不要只看 overall EM

对完整 638 道题至少按以下维度统计：

```text
stage
capability type
question length
retrieval hops
number of relevant docs
distractor density
entity ambiguity
temporal dependency
evidence conflict
tool-call depth
```

输出：

```text
accuracy
average tool calls
search recall
open recall
evidence recall
answer EM
failure type distribution
```

## 1.2 对每个 capability stage 做 static evaluation

建议先跑：

```text
Qwen2.5-3B-Instruct
5 stages
每 stage 80–120 tasks
seed = 0
```

不要训练。

目标是得到：

| Stage | EM | Search Fail | Evidence Fail | Reasoning Fail | Avg Tool Calls |
|---|---:|---:|---:|---:|---:|
| S0 | | | | | |
| S1 | | | | | |
| S2 | | | | | |
| S3 | | | | | |
| S4 | | | | | |

## 1.3 必须额外测 pass@k / self-consistency ceiling

对一部分错误任务重复 rollout：

```text
k = 4
temperature = 0.7
```

如果：

```text
single-run wrong
但 pass@4 经常能对
```

说明这些任务是：

```text
unstable capability
```

非常适合 active experience acquisition。

如果：

```text
pass@4 仍然错
```

说明是：

```text
real capability gap
```

更适合 weight / skill / harness update。

## 1.4 Gate

只有满足以下条件之一才继续：

### 理想状态

至少存在：

```text
25%–45%
```

的 frontier/error tasks。

### 如果 overall EM > 90%

不要直接进入主实验。

应主动构造 harder split，例如：

```text
更深 multi-hop
更多 distractor
cross-document synthesis
temporal update
conflicting evidence
entity ambiguity
longer trajectories
```

### 如果 overall EM < 50%

也不理想。

说明任务可能超出 learnable frontier。

Experience Policy 最有价值的区域通常不是：

```text
already solved
```

也不是：

```text
completely impossible
```

而是：

```text
current frontier
```

---

# 2. Experiment 1：Realized Utility 是否真实存在且可区分？

这是 ExPolicy 最重要的机制 sanity check。

在训练任何 learned Experience Policy 之前，先证明：

> 不同 experience 的未来学习价值真的不同。

## 2.1 抽取 40–60 个真实 experience

建议分成：

```text
15 success
15 ordinary failure
15 frontier failure
15 repeated-pattern experience
```

如果成本较高，第一轮至少 30 个。

## 2.2 对每个 experience 做 before / after intervention

固定：

```text
agent_v0
future probe set
retention probe set
```

对每个 experience：

```text
A. no update
B. apply experience
```

测：

```text
future performance delta
retention delta
token cost
training cost
interaction cost
```

计算：

\[
U(e)
=
\Delta C_{\text{future}}
-
\lambda \Delta C_{\text{retention-loss}}
-
\alpha Cost
\]

第一轮不要过度复杂化公式。

推荐先定义：

```text
U =
future_EM_gain
- 0.5 * retention_EM_drop
- normalized_cost
```

然后做 sensitivity analysis。

## 2.3 需要画的第一个关键图

### Figure A：Experience Utility Distribution

横轴：

```text
realized utility
```

分组：

```text
successful
ordinary failure
frontier failure
repeated experience
```

你希望看到：

```text
utility 并非均匀
且 frontier failure 的正 utility 比例明显更高
```

## 2.4 关键统计

统计：

```text
Positive Utility Rate
Mean Utility
Median Utility
Utility Variance
Top-20% Utility Contribution
```

尤其看：

```text
top 20% experiences
贡献了多少 cumulative capability gain？
```

如果呈明显 long-tail：

> 这就是 Active Experience Acquisition 最强的实验动机。

---

# 3. Experiment 2：Active Acquisition Pilot

这是第一个真正验证“Learning What to Experience”的实验。

暂时固定 assimilation，只改变 acquisition。

## 3.1 所有方法使用同一个 assimilation

第一轮建议：

```text
episodic + semantic
```

或者：

```text
固定 weight update
```

不要同时改变 acquisition 和 assimilation。

## 3.2 比较

至少：

```text
Random / Passive
Uncertainty
Failure Replay
Frontier Heuristic
Reward-Seeking
Learned Utility Ranker
```

如果 learned policy 还没有足够数据：

先比较前五个。

## 3.3 严格同预算

例如：

```text
50 acquired experiences
```

或者：

```text
500 environment steps
```

所有方法完全一致。

核心不能比较：

```text
谁获得了更多 experience
```

而应该比较：

```text
相同 experience budget 下谁学得更多
```

## 3.4 指标

主指标：

```text
Future EM Gain
```

辅助：

```text
Capability Gain / Experience
Capability Gain / 1K Tool Calls
Capability Gain / 1M Tokens
Retention
Positive Utility Hit Rate
Mean Realized Utility
```

## 3.5 最关键的对照

一定加入：

```text
Reward-Seeking
vs
Learning-Utility-Seeking
```

这是非常有论文辨识度的实验。

你想证明：

> 当前任务 reward 最大的 experience，并不一定是最值得学习的 experience。

如果出现：

```text
Reward-Seeking:
    immediate reward 高
    future gain 低

Utility-Seeking:
    immediate reward 未必最高
    future gain 明显更高
```

这是非常强的结果。

---

# 4. Experiment 3：Carrier Utility Matrix

这是验证 Adaptive Assimilation 的核心实验。

不要一上来训练 router。

先回答：

> 同一个 experience 放到不同 carrier，真的会产生不同效果吗？

## 4.1 第一轮 carrier 不要六个全开

建议先做四个：

```text
episodic
semantic
skill
weight
```

暂时关闭：

```text
harness
```

原因不是退化，而是先控制工程变量。

Harness evolution 可以作为后续增强实验。

## 4.2 选取 30–50 个 experience

要求覆盖：

```text
specific factual case
repeated semantic pattern
procedural failure
reasoning pattern
retrieval pattern
long-horizon failure
```

## 4.3 对每个 experience 做四路 intervention

```text
agent_v0
├── episodic
├── semantic
├── skill
└── weight
```

统一评测：

```text
near-transfer probes
far-transfer probes
retention probes
```

得到：

```text
Experience × Carrier Utility Matrix
```

示例：

| Experience | Episodic | Semantic | Skill | Weight |
|---|---:|---:|---:|---:|
| e1 | +8 | +2 | 0 | +1 |
| e2 | +1 | +7 | +4 | +5 |
| e3 | 0 | +2 | +10 | +3 |

## 4.4 这一步必须回答三个问题

### Q1

是否存在明显的：

```text
best carrier depends on experience
```

### Q2

是否存在：

```text
某些 carrier 短期好但长期 retention 差
```

例如 weight update。

### Q3

是否存在 carrier promotion：

```text
episodic
→ semantic
→ skill
→ weight
```

对应 experience lifecycle。

---

# 5. Experiment 4：Adaptive Assimilation Pilot

当 Carrier Utility Matrix 有了之后，再训练 learned router。

## 5.1 输入

至少：

```text
experience embedding
capability tags
failure type
success/failure
uncertainty
repetition frequency
transfer history
verification score
cost
```

## 5.2 输出

```text
episodic
semantic
skill
weight
discard
```

第一版 harness 可以继续关闭。

## 5.3 Baseline

```text
Random Routing
Fixed Memory
Fixed Weight
Heuristic Routing
Learned Routing
Hindsight Oracle
```

## 5.4 指标

不要只看 classification accuracy。

真正重要的是：

```text
Routing Regret
```

定义：

```text
best possible carrier utility
-
chosen carrier utility
```

以及：

```text
final future capability gain
```

即使 router classification accuracy 不高，只要 utility regret 低，仍然说明它学对了。

---

# 6. Experiment 5：2×2 Factorial——第一张真正的论文主表

等 Active Acquisition 和 Adaptive Assimilation 都单独有效之后，做：

| | Fixed Assimilation | Adaptive Assimilation |
|---|---:|---:|
| Passive Acquisition | Passive-OEL | Adaptive-only |
| Active Acquisition | Active-only | **Full ExPolicy** |

这张表极其重要。

它直接回答：

```text
Active acquisition 是否有效？
Adaptive assimilation 是否有效？
两者是否互补？
联合是否产生额外增益？
```

## 6.1 建议第一轮规模

不要直接 3 seed × full benchmark。

先：

```text
100–150 acquisition tasks
1 seed
3 capability stages
```

如果结果稳定，再扩展。

## 6.2 预期论文结果模式

理想情况：

```text
Passive-OEL      +X
Active-only      +X+a
Adaptive-only    +X+b
Full ExPolicy    +X+a+b+c
```

其中：

```text
c > 0
```

说明 acquisition 与 assimilation 存在协同。

如果：

```text
c ≈ 0
```

说明 joint policy 的必要性需要重新检查。

---

# 7. Experiment 6：Separate Policy vs Joint Experience Policy

只有在 2×2 实验确认二者都有效后，才值得训练真正的 joint policy。

比较：

```text
Separate:
    AcquisitionPolicy
    AssimilationPolicy

Joint:
    shared encoder
    acquisition head
    assimilation head
    utility head
```

## 7.1 为什么这个实验不能太早做

如果 Active-only 或 Adaptive-only 本身都没有 gain，

那么 joint policy 再复杂也没有研究意义。

所以顺序必须是：

```text
mechanism works
→ modules work
→ joint works better
```

## 7.2 关键指标

```text
Future EM
Learning Efficiency
Utility Prediction Error
Acquisition Regret
Routing Regret
Retention
```

还应测试：

```text
joint policy 是否更快学会 experience-carrier interaction
```

---

# 8. Experiment 7：Online EvolveLoop

直到前面的静态 intervention 实验跑通后，再启用完整在线闭环。

推荐：

```text
Stage 0
→ acquire
→ assimilate
→ evaluate
→ update Experience Policy
→ Stage 1
→ ...
```

## 8.1 第一轮只跑 3 stages

建议：

```text
3 stages
100 tasks / stage
seed 0
```

观察：

```text
agent capability trajectory
carrier distribution
experience utility trajectory
retention
policy drift
```

## 8.2 重点看有没有 collapse

检查：

```text
weight update 后旧能力下降
memory 过量导致 context pollution
skill 数量持续膨胀
semantic memory 冲突
acquisition policy 越来越 exploit
utility predictor calibration 恶化
```

这些现象本身都可能成为论文的重要分析。

---

# 9. Experiment 8：Full Paper Scale

只有满足下面的 Gate 才进入：

```text
Active > Passive
Adaptive > Fixed
Full > Active-only
Full > Adaptive-only
Joint >= Separate
```

至少多数 seed / pilot 上成立。

然后再跑：

```text
5 capability stages
≥ 200 tasks / stage
3 seeds
```

---

# 10. 建议最终主实验矩阵

最终主表至少保留：

```text
Static
Passive Memory
Passive OEL
Uncertainty Acquisition
Frontier Acquisition
Active-only
Adaptive-only
Separate Active+Adaptive
Full Joint ExPolicy
Oracle Acquisition
Oracle Routing
```

Oracle 只做 upper bound。

---

# 11. 建议论文 Figure / Table 顺序

## Figure 1

ExPolicy Framework

```text
Diagnose
→ Acquire
→ Experience
→ Verify
→ Assimilate
→ Update
→ Evaluate Utility
→ Improve Experience Policy
```

## Figure 2

Realized Experience Utility Distribution

证明：

```text
different experiences have dramatically different future value
```

## Figure 3

Reward vs Learning Utility

证明：

```text
reward-seeking != learning-seeking
```

## Figure 4

Experience × Carrier Utility Matrix

证明：

```text
one-size-fits-all assimilation is suboptimal
```

## Table 1

2×2 factorial：

```text
Passive/Active
×
Fixed/Adaptive
```

## Table 2

Full baselines。

## Figure 5

Capability evolution over stages。

## Figure 6

Experience Policy 的 carrier usage / acquisition distribution 随时间变化。

---

# 12. 下一步最应该立即跑的三个实验

如果现在只选三个，我建议严格按照：

## Priority 1：Capability-Gap Audit

先判断 86% 是否已经过高。

输出：

```text
per-stage EM
failure distribution
pass@4
frontier task ratio
```

## Priority 2：Realized Utility Sanity Check

30–60 experiences。

验证：

```text
experience utility 是否有显著方差
frontier failure 是否真的高 utility
```

这是 ExPolicy 最根本的 empirical premise。

## Priority 3：Carrier Utility Matrix

先做：

```text
episodic / semantic / skill / weight
```

哪怕只做 20–30 experiences，也足够告诉你：

> Adaptive Assimilation 到底是不是一个真实问题，而不是系统设计想象。

---

# 13. 暂时不要优先做的事情

当前阶段不建议优先：

```text
1. 立刻跑 5 × 200 × 3 seed
2. 立刻训练复杂 GRPO Experience Policy
3. 立刻实现强 harness self-modification
4. 立刻加入 multi-agent
5. 立刻扩第二个 benchmark
6. 立刻追求更大模型
```

原因很简单：

> 当前最大的不确定性是论文假设，而不是工程规模。

先把：

```text
experience utility
active acquisition
adaptive assimilation
joint interaction
```

这四件事证实。

---

# 14. 推荐的近期执行顺序

```text
E0  Benchmark Capability-Gap Audit
    ↓
E1  Experience Utility Distribution
    ↓
E2  Active Acquisition Pilot
    ↓
E3  Carrier Utility Matrix
    ↓
E4  Learned Assimilation
    ↓
E5  2×2 Factorial
    ↓
E6  Separate vs Joint Experience Policy
    ↓
E7  Online EvolveLoop
    ↓
E8  Full Paper Scale
```

---

# 15. 论文是否值得继续投入的 Go / No-Go 标准

在 full-scale 之前建议设四个硬门槛。

## Gate A：Experience Utility Heterogeneity

不同 experience 的 realized utility 有明显方差：

```text
Top-20% experiences
贡献显著高于平均水平
```

## Gate B：Active Acquisition Works

相同 budget 下：

```text
Active Acquisition
>
Passive Acquisition
```

至少有稳定的 future capability gain。

## Gate C：Carrier Choice Matters

至少有相当比例 experience 满足：

```text
best carrier != globally best fixed carrier
```

且 adaptive routing 能降低 routing regret。

## Gate D：Joint Optimization Matters

```text
Full ExPolicy
>
Active-only
and
Full ExPolicy
>
Adaptive-only
```

如果四个 Gate 都成立，再投入论文规模实验，成功率会高很多。

---

# 16. 最关键的近期目标

下一阶段不要把目标写成：

> “把 ExPolicy 全部跑起来。”

而应该写成：

> **验证 Experience Policy 这个研究对象是否真实存在。**

具体来说，要证明三件事：

```text
1. Experience 的未来学习价值高度非均匀；
2. Agent 可以主动选择更有价值的 Experience；
3. 不同 Experience 的最佳吸收方式不同。
```

如果这三件事成立，

那么：

```text
Joint Experience Policy
```

就不是一个工程拼装，

而是一个自然且必要的学习问题。

这将成为 ExPolicy 最核心的论文逻辑。
