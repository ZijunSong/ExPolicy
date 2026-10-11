# ExPolicy 下一步实验指令：R5.1 Robustness & Causal Validation

> 适用机器：H20，仅 GPU 4 / 5 / 6 / 7 可用  
> Repo：`/data/ppnm/expolicy`  
> Base model：`/data/ppnm/models/Qwen2.5-3B-Instruct`  
> World：Helion，world seed=13  
> 原则：**不进入 R6 learned carrier router，不启动 E5–E8。** 本轮只补强 R5 的统计稳健性、因果解释和可部署性。

---

## 0. 为什么下一轮先做 R5.1，而不是 R6

当前已经得到的最强结果是：

- R4：executable skill 是唯一稳定正的 carrier，`mean U_cap=+0.237`，11/12 batch 为正；
- 但 R4 中 oracle 只比 always-skill 高约 `0.003`，说明当前数据**没有足够 carrier-routing 空间**，此时训练 learned carrier router 科学意义很弱；
- R5：在 experience-conditioned executable skill 后，`frontier` 的 `mean ΔFuture=+0.519`，相对 passive 高 `+0.296`，3/3 seed 胜出；
- 但 R5 仍有三个明显薄弱点：
  1. 只有 3 个 seed；
  2. skill 激活阈值固定为 8，最终增益呈明显的“是否达到激活门槛”式跳变，需要排除纯 threshold artifact；
  3. 主评测使用 oracle capability dispatch，不代表部署时能够自动选择 skill；
  4. temporal skill 在所有 acquisition policy 下都没有激活，S2 gain 恒为 0；
  5. 当前 frozen executable primitive 很强，需要进一步确认提升是否真的依赖 acquired experience 的**内容**，而不只是“累计到 8 条后解锁一个预先写好的 skill”。

因此，下一轮目标不是继续扩方法，而是回答：

> **R5 的 Active > Passive 是否在更多随机种子、不同激活阈值、保持相同 skill activation mask 的内容消融，以及非 oracle dispatch 下仍然成立？**

如果这些问题能够站住，再决定是否值得把 ExPolicy 扩展成完整的 learned acquisition / routing framework。

---

# 1. 总体执行约束

## 1.1 先做只读检查

```bash
cd /data/ppnm/expolicy

git status --short
nvidia-smi

python scripts/audit_r5_causal_path.py
```

必须确认：

- 原 R5 outputs 完整；
- `R5_CAUSAL_AUDIT.json` 仍为 PASS；
- 不覆盖 `outputs/r5_acquisition_skill/`；
- 原 R5 frozen split 不重建、不改 task id；
- future / retention / acquisition pool 继续保持原来的零 task-id overlap；
- 不修改 R2 executable procedure primitive 的逻辑；
- 不允许把 future probe 或 retention 的任何 gold 信息写入 skill artifact。

新产物统一写入：

```text
outputs/r5_1_robustness/
```

建议新建统一 driver：

```text
scripts/run_r5_1_robustness.py
```

允许复用 R5 的 acquisition、SkillAssimilator、probe evaluator 和 split loader；**不要复制出第二套不一致实现。**

---

# 2. GPU 分工总览

| GPU | Track | 核心问题 | 优先级 |
|---|---|---|---|
| GPU 4 | A. Seed Extension | frontier > passive 是否能跨更多 seed 稳定复现？ | P0 |
| GPU 5 | B. Temporal / Coverage Diagnostic | 为什么 S2 永远不激活？coverage-aware frontier 能否补上？ | P1 |
| GPU 6 | C. Activation & Content Causality | 增益是经验内容产生的，还是仅仅“够 8 条就解锁 skill”？ | **P0，最关键** |
| GPU 7 | D. Deployable Capability Dispatch | 去掉 oracle capability label 后，R5 增益还能保留多少？ | **P0** |

四个 Track 可以并行。

---

# 3. GPU 4：Track A — Seed Extension

## 3.1 目的

原 R5 只有 seed `0,1,2`。下一步只保留最关键的：

- `frontier`
- `passive`

把 seed 扩展到至少：

```text
3,4,5,6,7,8,9
```

最终与原来的 0–2 合并成 **10 seeds**。

不要重新跑 reward_seeking / uncertainty / failure_replay；当前最需要确认的是主结论：

```text
frontier > passive
```

## 3.2 Setting

保持与 R5 完全一致：

```text
budget = 48
skill activation threshold = 8
skill refit checkpoint = 8 / 16 / 24 / 32 / 40 / 48
probe checkpoint = 0 / 16 / 32 / 48
future = 原 frozen 36 tasks
retention = 原 frozen 24 S0 tasks
oracle capability dispatch = ON
```

## 3.3 推荐接口

如果现有 `run_r5_suite.py` 不支持 methods/seeds 参数，不要硬改原脚本行为；新增 wrapper：

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_r5_1_robustness.py \
  --track seed_extension \
  --methods frontier passive \
  --seeds 3 4 5 6 7 8 9 \
  --output outputs/r5_1_robustness/seed_extension
```

如果接口尚不存在，由 Agent 按上述接口实现。

## 3.4 必须输出

```text
SEED_EXTENSION_PER_SEED.csv
SEED_EXTENSION_SUMMARY.json
SEED_EXTENSION_BOOTSTRAP.json
SEED_EXTENSION.md
```

至少包含：

- 每 seed 的 base future EM；
- frontier / passive final future EM；
- `ΔFuture`;
- `frontier - passive`;
- 每 capability gain；
- skill activation checkpoint；
- acquisition capability histogram；
- tokens / tool calls；
- 10-seed mean / std；
- paired bootstrap 95% CI；
- frontier 胜 passive 的 seed 比例。

## 3.5 判读

主结论优先看：

```text
mean(frontier - passive)
paired 95% CI
win-rate
```

建议把以下作为“稳健支持”标准，而不是中途调参目标：

```text
10-seed mean gap > 0
paired bootstrap 95% CI 不跨 0
至少 7/10 seed frontier > passive
```

无论是否通过，都完整记录，不筛 seed。

---

# 4. GPU 5：Track B — Temporal / Coverage Diagnostic

## 4.1 目的

R5 中所有方法的 temporal gain 都是 0，因为没有任何方法累计到 8 条 temporal experience。

这会留下一个明显问题：

> frontier 的优势到底是普适地发现 capability frontier，还是当前 scoring 恰好只会大量选择 S1/S4？

先做 pool audit，再决定是否运行 coverage-aware acquisition。

## 4.2 Phase B0：无 GPU / 轻量 audit

先统计 frozen acquisition pool：

- 每个 capability 的可选样本数；
- 每个 capability 的原始 frontier score 分布；
- temporal 样本在全池 score percentile；
- 原 frontier 在 budget=48 时 temporal 的最高 rank；
- 如果 budget 扩成 64/96，原 frontier 是否自然会选到 ≥8 条 temporal。

输出：

```text
outputs/r5_1_robustness/temporal/AUDIT_TEMPORAL_POOL.json
outputs/r5_1_robustness/temporal/AUDIT_TEMPORAL_POOL.md
```

**不要为了得到阳性结果而复制 temporal task 或放回 future probe。**

## 4.3 Phase B1：capability-normalized frontier

仅当 frozen acquisition pool 中本来就有足够 temporal 样本时，增加一个诊断 policy：

```text
frontier_capnorm
```

定义原则：

1. 仍使用原 frontier score；
2. 只在 acquisition ranking 层做 capability-wise normalization；
3. 不读取 future probe；
4. 不读取 gold answer；
5. 不改变 budget；
6. 不修改 SkillAssimilator；
7. 不人为保证每类一定采 8 条。

推荐 normalization：

```text
z = (score - capability_mean) / (capability_std + eps)
```

再按 z 排序/选择。

不要加入手工 temporal bonus。

## 4.4 执行

```bash
CUDA_VISIBLE_DEVICES=5 \
python scripts/run_r5_1_robustness.py \
  --track temporal_coverage \
  --methods frontier frontier_capnorm passive \
  --seeds 0 1 2 \
  --budget 48 \
  --output outputs/r5_1_robustness/temporal
```

## 4.5 必须输出

```text
TEMPORAL_ACQUISITION_COUNTS.csv
TEMPORAL_ACTIVATION.csv
TEMPORAL_CAPABILITY_OUTCOME.csv
TEMPORAL_SUMMARY.md
```

重点看：

- temporal acquired count；
- temporal 首次达到 8 条的 budget；
- S2 skill 是否激活；
- temporal future EM / gain；
- macro future EM；
- 是否牺牲 S1/S4；
- acquisition entropy / capability coverage。

## 4.6 判读

若 `frontier_capnorm` 能：

```text
temporal acquired >= 8
temporal ΔFuture > 0
macro future 不显著低于原 frontier
```

则说明原 frontier 的主要问题是跨 capability score calibration，而不是 Active Acquisition 假设失效。

若仍采不到 temporal，则记录为：

```text
current frontier signal does not expose the S2 capability gap
```

不要继续堆 heuristic。

---

# 5. GPU 6：Track C — Activation Threshold & Experience-Content Causality

> **这是下一轮最关键的实验。**

## 5.1 核心问题

R5 当前机制是：

```text
acquire real experiences
→ 某 capability count >= 8
→ SkillAssimilator.fit(...)
→ 激活 frozen executable primitive
→ future EM 大涨
```

因此必须拆开两个变量：

```text
A. experience 数量 / activation mask
B. experience 内容本身
```

否则 R5 很容易被解释为：

> “frontier 只是更早攒够了解锁一个预装 skill 的门槛。”

---

## 5.2 Phase C1：activation threshold sensitivity

只跑：

```text
frontier
passive
```

阈值：

```text
K = 4, 8, 12, 16
```

保持 budget=48、同 frozen split、同 scoring。

执行：

```bash
CUDA_VISIBLE_DEVICES=6 \
python scripts/run_r5_1_robustness.py \
  --track activation_threshold \
  --methods frontier passive \
  --seeds 0 1 2 \
  --thresholds 4 8 12 16 \
  --budget 48 \
  --output outputs/r5_1_robustness/activation_threshold
```

输出：

```text
THRESHOLD_PER_SEED.csv
THRESHOLD_LEARNING_CURVE.csv
THRESHOLD_ACTIVATION.csv
THRESHOLD_SUMMARY.md
```

重点画/记录：

```text
x = budget
y = ΔFuture
group = method × K
```

以及：

```text
time-to-first-skill
time-to-second-skill
AUC of ΔFuture over budget
final frontier-passive gap
```

---

## 5.3 Phase C2：same-mask content ablation

必须复用原 R5 已获得的 acquisition trajectory。

对于每个 `method × seed × capability`，**固定完全相同的 activation mask 和 activation checkpoint**，仅改变 `fit()` 看到的经验内容。

至少四种条件：

### C2-a RealFit

原始 R5：

```text
真实 acquired experiences
→ fit
→ skill
```

### C2-b CountOnly

只根据：

```text
capability count >= K
```

激活对应 frozen primitive。

`fit()` 不读取 experience 内容。

这个条件直接测：

> 原 R5 是否其实只是 count-gated skill unlock。

### C2-c WithinCapShuffle

保持：

- capability 不变；
- experience 数量不变；
- activation checkpoint 不变；

但在该 capability 的 acquisition pool 内随机替换为**其它训练侧 experience**。

禁止使用 future / retention task。

### C2-d CrossCapContent

仍强制保持与 RealFit 一样的 activation mask，但把 fit 内容替换为其它 capability 的 experience。

注意：

- **不能因为 cross-cap 内容而额外激活新的 skill**；
- activation mask 必须由 RealFit 冻结；
- 这样才能隔离“内容是否重要”，而不是重复 R5 shuffled ablation 中的 white-gift 问题。

执行建议：

```bash
CUDA_VISIBLE_DEVICES=6 \
python scripts/run_r5_1_robustness.py \
  --track content_causality \
  --methods frontier passive \
  --seeds 0 1 2 \
  --fit-modes real count_only within_cap_shuffle cross_cap_content \
  --freeze-activation-mask-from-r5 \
  --output outputs/r5_1_robustness/content_causality
```

输出：

```text
CONTENT_CAUSALITY_PER_SEED.csv
CONTENT_CAUSALITY_PER_CAP.csv
CONTENT_CAUSALITY_SUMMARY.md
```

## 5.4 最重要的判读规则

如果：

```text
RealFit ≈ CountOnly
RealFit ≈ WithinCapShuffle
```

则不要再把机制写成强意义的：

```text
experience assimilation
```

论文应收缩为：

```text
experience-conditioned / evidence-gated skill activation
```

如果：

```text
RealFit > CountOnly
RealFit > WithinCapShuffle
RealFit >> CrossCapContent
```

而且差异跨 seed 稳定，则可以更有底气地说：

```text
acquired experience content materially shapes the executable skill.
```

**这个实验无论阳性还是阴性都必须保留，禁止按结果删除。**

---

# 6. GPU 7：Track D — Deployable Capability Dispatch

## 6.1 目的

R5 主表使用 oracle capability dispatch：

```text
S1 probe → multi-hop skill
S2 probe → temporal skill
S4 probe → cross-doc skill
S0 retention → no skill
```

它很好地隔离了 acquisition 变量，但不是 deployable setting。

下一步不训练 R6 carrier router，而只做一个更窄的问题：

> **仅根据测试时可观察到的 question text，能否决定是否调用哪一个 executable skill？**

这是 skill dispatch，不是 carrier routing。

---

## 6.2 Router 约束

推荐先实现一个非常简单、独立的 text router：

```text
TF-IDF / bag-of-words
+ multinomial logistic regression
```

标签：

```text
none
multi_hop
temporal
cross_doc
```

训练数据只能来自：

- acquisition pool；
- 或明确与 future/retention 不重叠的 training/validation tasks。

禁止：

- future probe capability metadata；
- task id prefix；
- gold answer；
- gold document id；
- stage name；
- filename/path 泄漏。

必须对输入做 task-id / template-id 泄漏检查。

输出 router 本身的：

```text
accuracy
macro-F1
confusion matrix
per-class recall
```

---

## 6.3 下游条件

至少评测：

```text
oracle_dispatch          # 当前 R5 upper bound
text_router_dispatch     # deployable 主条件
no_skill                 # clean base
all_skills_fixed_order   # stress control，不作为主方法
```

对 acquisition method 只保留：

```text
frontier
passive
```

先用 seed 0/1/2。

执行：

```bash
CUDA_VISIBLE_DEVICES=7 \
python scripts/run_r5_1_robustness.py \
  --track deployable_dispatch \
  --methods frontier passive \
  --seeds 0 1 2 \
  --dispatch oracle text_router no_skill all_skills_fixed_order \
  --output outputs/r5_1_robustness/deployable_dispatch
```

## 6.4 必须输出

```text
DISPATCH_ROUTER_METRICS.json
DISPATCH_CONFUSION.csv
DISPATCH_DOWNSTREAM.csv
DISPATCH_PER_CAP.csv
DISPATCH_SUMMARY.md
```

重点报告：

```text
oracle future EM
text-router future EM
oracle - text-router gap
frontier - passive under text-router
retention delta
false-positive skill invocation rate on S0
```

## 6.5 判读

最理想结果：

```text
text-router dispatch 保留大部分 oracle gain
且 frontier > passive 仍成立
```

如果 text-router 明显掉点，则不要试图用 future label 修 router；结论应写成：

```text
Active Acquisition 已在 oracle-dispatch controlled setting 下成立，
但 deployable skill dispatch 仍是独立未解决问题。
```

---

# 7. 汇总分析

四卡完成后，统一运行一个无训练分析脚本：

```bash
cd /data/ppnm/expolicy

python scripts/analyze_r5_1_robustness.py \
  --root outputs/r5_1_robustness \
  --r5-root outputs/r5_acquisition_skill
```

必须生成：

```text
outputs/r5_1_robustness/R5_1_MASTER_SUMMARY.md
outputs/r5_1_robustness/R5_1_MASTER_TABLE.csv
outputs/r5_1_robustness/R5_1_DECISION.json
```

`R5_1_MASTER_SUMMARY.md` 至少包含四张主表：

1. 10-seed frontier vs passive；
2. threshold sensitivity；
3. RealFit / CountOnly / WithinCapShuffle / CrossCapContent；
4. oracle dispatch vs deployable text-router dispatch。

另附：

5. temporal coverage diagnostic。

---

# 8. 本轮之后的决策树

## Case A：结果最强

如果同时满足：

```text
1. 10-seed frontier > passive 稳定；
2. 不同 K 下方向基本一致；
3. RealFit 明显优于 CountOnly / shuffle；
4. text-router 下 frontier > passive 仍保留；
```

则下一轮才值得考虑：

```text
R6'：learned acquisition policy
```

而不是 learned carrier router。

原因：R4 已经表明 carrier 选择几乎被 skill 统治；真正还有学习空间的是：

```text
what experience to acquire
```

---

## Case B：Seed 稳健，但 RealFit ≈ CountOnly

则项目应明确收缩为：

```text
Experience-Gated Skill Activation
```

核心贡献变成：

```text
active acquisition determines which capability-specific executable procedures become eligible / activated.
```

不要再强调“经验内容被 skill assimilator 深度吸收”。

---

## Case C：Oracle 强，deployable dispatch 弱

则保留 R5 作为 mechanism experiment：

```text
acquisition matters under controlled capability dispatch
```

后续单独研究：

```text
observable-state skill dispatch
```

不要把它和 R6 carrier routing 混在一起。

---

## Case D：扩 seed 后 frontier gap 明显缩水

停止继续加复杂模块。

此时最正确的动作是：

```text
把 R5 定位为 pilot / mechanism evidence，
不继续做完整 ExPolicy。
```

---

# 9. 禁止事项

本轮严格禁止：

- 跑 `R6 learned carrier router`；
- 跑 E5–E8；
- 为了激活 temporal 而复制/重采 future task；
- 改 R5 frozen future/retention split；
- 把 capability ground-truth 偷渡给 deployable router；
- 通过 task id / filename / stage prefix 泄漏 capability；
- 根据中间结果删掉失败 seed；
- 根据结果修改 activation threshold 后只报告最好的一档；
- 让 shuffled/cross-cap content 改变 activation mask；
- 预安装全部 R2 skill 后再声称 acquisition 有贡献；
- 覆盖原 `outputs/r5_acquisition_skill/`。

---

# 10. 推荐启动顺序

四卡都空闲时直接并行：

```bash
# GPU 4 — seed extension
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_r5_1_robustness.py \
  --track seed_extension \
  --methods frontier passive \
  --seeds 3 4 5 6 7 8 9 \
  --output outputs/r5_1_robustness/seed_extension
```

```bash
# GPU 5 — temporal / coverage
CUDA_VISIBLE_DEVICES=5 \
python scripts/run_r5_1_robustness.py \
  --track temporal_coverage \
  --methods frontier frontier_capnorm passive \
  --seeds 0 1 2 \
  --budget 48 \
  --output outputs/r5_1_robustness/temporal
```

```bash
# GPU 6 — mechanism causality
CUDA_VISIBLE_DEVICES=6 \
python scripts/run_r5_1_robustness.py \
  --track content_causality \
  --methods frontier passive \
  --seeds 0 1 2 \
  --fit-modes real count_only within_cap_shuffle cross_cap_content \
  --freeze-activation-mask-from-r5 \
  --output outputs/r5_1_robustness/content_causality
```

GPU 6 完成 content causality 后继续：

```bash
CUDA_VISIBLE_DEVICES=6 \
python scripts/run_r5_1_robustness.py \
  --track activation_threshold \
  --methods frontier passive \
  --seeds 0 1 2 \
  --thresholds 4 8 12 16 \
  --budget 48 \
  --output outputs/r5_1_robustness/activation_threshold
```

```bash
# GPU 7 — deployable dispatch
CUDA_VISIBLE_DEVICES=7 \
python scripts/run_r5_1_robustness.py \
  --track deployable_dispatch \
  --methods frontier passive \
  --seeds 0 1 2 \
  --dispatch oracle text_router no_skill all_skills_fixed_order \
  --output outputs/r5_1_robustness/deployable_dispatch
```

---

# 11. 对执行 Agent 的实现要求

如果 `scripts/run_r5_1_robustness.py` 或 `scripts/analyze_r5_1_robustness.py` 尚不存在：

1. 先阅读并复用：
   - `scripts/run_r5_suite.py`
   - `scripts/audit_r5_causal_path.py`
   - R5 split loader
   - R5 acquisition policy implementation
   - `SkillAssimilator`
   - R5 probe evaluator
2. **不要重新实现一套 agent/runtime。**
3. 所有新增 CLI 参数必须写入 `--help`。
4. 每个 Track 先做 1 个 seed 的 smoke test。
5. smoke test 必须验证：
   - GPU 绑定正确；
   - output 不覆盖 R5；
   - split hash 与 R5 相同；
   - future / acquisition / retention task-id overlap=0；
   - 没有 gold/task-id leakage；
   - resume 能工作；
   - 单个失败 run 不会让其它结果被写成成功。
6. smoke test 通过后自动运行正式实验。
7. 正式实验结束后自动执行对应 analyzer。
8. 不使用 mock model、mock retrieval、伪造经验或 synthetic success 替代真实 rollout。

---

# 12. 本轮最终目标

这一轮不是为了“再多做几张表”，而是决定 R5 的论文表述到底应该落在哪一层：

```text
最强：
active acquisition
→ experience-content-dependent executable assimilation
→ deployable skill dispatch
→ robust future capability gain
```

还是更保守但仍成立的：

```text
active acquisition
→ experience-gated skill activation
→ oracle-dispatch mechanism gain
```

这两个故事的学术强度差很多。

因此，本轮优先级应当是：

```text
Content causality ≈ Seed robustness ≈ Deployable dispatch
> Temporal coverage
> 任何新的 learned router
```
