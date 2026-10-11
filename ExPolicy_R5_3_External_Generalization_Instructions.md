# ExPolicy 下一轮实验指令：R5.3 External Generalization on Public Search Benchmarks

> **本轮核心目标**
>
> R5.2 已经证明：在 Helion 上，`capability-frontier acquisition → experience-gated executable skill activation` 显著优于 Ace-Skill Prioritized Sampler 与 passive，并且具有更高 sample efficiency。
>
> 下一轮**不要继续调 Helion、不要改 frontier 小权重、不要改 K、不要训练 R6 learned carrier router**。
>
> 本轮只回答一个问题：
>
> > **这一机制能否从 Helion 泛化到公开、非模板化、社区常用的 Search-Agent benchmark？**
>
> 若不能通过外部 benchmark，则当前结果仍应视为 Helion-specific mechanism result；若能通过，才开始形成论文主结果。

---

## 0. 当前结论与本轮冻结项

先完整阅读：

```text
outputs/r5_acquisition_skill/
outputs/r5_1_robustness/
outputs/r5_2_baseline_collision/
expolicy-result-record*.md
```

重点读取：

```text
R5_2_MASTER_SUMMARY.md
R5_2_DECISION.json
R5_2_FINAL_METRICS.csv
R5_2_LEARNING_CURVES.csv
ACE_SKILL_IMPLEMENTATION_AUDIT.md
```

### 已经确认、不要重复验证的结论

1. Helion 上 `frontier > passive`，10/10 seed。
2. Helion 上 `frontier > Ace-Skill`，10/10 seed。
3. Ace-Skill 在该设定下与 passive 打平。
4. 当前有效机制是：
   ```text
   capability-frontier acquisition
       →
   sufficient distinct capability evidence
       →
   executable skill becomes eligible
       →
   future capability gain
   ```
5. `RealFit <= CountOnly`，因此**不要再宣称 experience content 被深度 assimilate**。
6. executable skill 本身不是 novelty；novelty 候选在：
   ```text
   what experience to acquire
   →
   which missing capability should become eligible
   →
   when to activate its executable procedure
   ```

---

# 1. 外部 benchmark：固定使用公开 Search-R1 / SESA 风格的多跳搜索协议

本轮主外部环境使用：

```text
HotpotQA
2WikiMultiHopQA
MuSiQue
```

附加 zero-shot transfer probe：

```text
Bamboogle
```

原因：

- 这四类任务都属于公开的 search / multi-hop QA 社区常用 evaluation。
- SESA 在统一 search-agent protocol 下评测 HotpotQA、2Wiki、MuSiQue、Bamboogle。
- SkillRL / Search-R1 系列代码也直接支持同一类 search benchmark。
- 它们比 Helion 更接近公开、多样、非人为模板化的搜索任务。
- HotpotQA / 2Wiki / MuSiQue 可以提供不同 multi-hop structure，适合验证 capability-frontier 是否仍然有意义。

**不要**为了让 ExPolicy 更容易获胜而自行合成第二套 Helion-like 数据。

---

# 2. 搜索后端：改成公开 Wikipedia local retriever

本轮不能继续用：

```text
data/world/helion.json
```

应切到 Search-R1 / SkillRL 兼容的本地 Wikipedia 检索后端。

目标配置：

```text
Wikipedia 2018 corpus
local retriever server
E5-base-v2 compatible dense retrieval
```

优先复用 Search-R1 / SkillRL 已公开的 search environment 与 corpus preparation 逻辑。

参考：

```text
https://github.com/PeterGriffinJin/Search-R1
https://github.com/aiming-lab/SkillRL
https://github.com/Zenghuang-Fu/SESA-Self-Evolving-Search-Agents
```

## 2.1 实现原则

不得使用：

```text
mock retrieval
fake passages
gold supporting facts 直接作为 search result
提前把 future supporting docs 塞给 agent
```

必须是真实：

```text
agent query
→ local retrieval server
→ top-k passages
→ agent open/read
→ answer
```

如果 repo 已存在可兼容 search API，则只新增 adapter，不要大改 agent 主体。

建议新增：

```text
expolicy/envs/public_search/
expolicy/envs/public_search/retriever.py
expolicy/envs/public_search/datasets.py
expolicy/envs/public_search/evaluator.py
```

---

# 3. 数据划分：严禁 future / acquisition 泄漏

对三个主数据集分别构造：

```text
calibration
acquisition_pool
skill_validation
future_probe
```

Bamboogle **只作为 final zero-shot probe**，不进入 acquisition，不参与调 skill，不参与 calibration。

## 3.1 建议规模

先做 Stage-1 pilot：

```text
HotpotQA:
    calibration       48
    acquisition_pool  400
    skill_validation   60
    future_probe      100

2Wiki:
    calibration       48
    acquisition_pool  400
    skill_validation   60
    future_probe      100

MuSiQue:
    calibration       48
    acquisition_pool  400
    skill_validation   60
    future_probe      100

Bamboogle:
    final probe       all available / maximum 125
```

主 future：

```text
n = 300
```

三个数据集 macro-average，不允许某个大数据集按样本数吞掉其它数据集。

如果本地资源不足，可以把 `future_probe` 暂时降到每集 60 做 smoke/pilot，但正式主结果必须恢复到 ≥100 / dataset。

## 3.2 Split 约束

必须满足：

```text
task-id overlap = 0
question exact overlap = 0
```

并审计：

```text
answer overlap
named-entity overlap
supporting-title overlap
```

后面三项只报告，不要求完全为零，因为真实 multi-hop QA 会自然共享实体。

输出：

```text
outputs/r5_3_external_generalization/SPLIT_AUDIT.md
outputs/r5_3_external_generalization/split_hashes.json
```

split 一旦冻结，不得根据结果重抽。

---

# 4. Capability taxonomy：必须可迁移，不能照搬 Helion 标签

Helion 的：

```text
S1 / S2 / S4
multi_hop / temporal / cross_doc
```

不能直接拿数据集名替代。

需要构造一个**公开 benchmark 内部共享**的 capability taxonomy。

本轮冻结为三个 executable-search capability：

```text
C1: bridge_decomposition
C2: comparison_alignment
C3: long_chain_search
```

## 4.1 C1 — bridge_decomposition

任务需要：

```text
Q
→ 找到中间实体 / bridge
→ 基于 bridge 发起下一次检索
→ 得到最终实体或属性
```

典型覆盖：

```text
HotpotQA bridge
2Wiki bridge / compositional chain
MuSiQue 2-hop chain
```

## 4.2 C2 — comparison_alignment

任务需要：

```text
分别解析 entity A / entity B
→ 检索同类型属性
→ 对齐时间 / 数值 / 类别
→ 比较后回答
```

典型覆盖：

```text
HotpotQA comparison
2Wiki comparison
```

## 4.3 C3 — long_chain_search

任务需要至少 3 个连续 evidence hops：

```text
e1
→ e2
→ e3
→ ...
→ final answer
```

优先来自：

```text
MuSiQue 3-hop / 4-hop
2Wiki 可验证的长链样本
```

---

# 5. Capability label 的使用边界

Primary controlled experiment 可以使用**数据集公开 metadata**构造 acquisition-side capability 标签。

但是：

1. 不得把 `dataset_name` 直接当 capability。
2. 不得用 future gold answer / supporting facts 推断 acquisition 任务 capability。
3. 不得人工逐题修正标签。
4. 所有 mapping rule 必须写成 deterministic code。
5. 在正式运行前冻结 mapping。

建议新增：

```text
scripts/build_public_capability_map.py
```

输出：

```text
outputs/r5_3_external_generalization/CAPABILITY_MAP_AUDIT.md
```

包含：

```text
每个 dataset × capability 数量
mapping rule
unmapped rate
ambiguous rate
随机抽 30 条人工只读检查结果
```

人工检查只判断 mapping 是否合理，**不得根据 future performance 修改单条标签**。

---

# 6. Secondary deployable capability router

Primary 仍允许 oracle capability label，以隔离 acquisition policy。

但本轮必须补一个比 Helion 更有意义的 deployable sanity：

```text
question text
→ capability router
→ eligible skill
```

推荐：

```text
TF-IDF + LogisticRegression
```

和一个轻量 embedding baseline：

```text
Qwen3-Embedding-0.6B / available local embedding model
→ nearest centroid / linear classifier
```

训练数据只能来自：

```text
calibration + acquisition-side labeled examples
```

future 标签只用于评测 router accuracy，不能回流训练。

输出：

```text
router accuracy
macro-F1
per-cap recall
downstream EM under router
oracle-router gap
false skill activation rate
```

**不要**为了把 router 做到 1.0 而调 prompt / 加 future pattern。

---

# 7. External executable skills：重新做最小化 capability-specific validation

不能把 Helion primitive 里任何：

```text
固定 document pattern
固定字段名
Helion entity schema
模板字符串
```

带到公开 benchmark。

只允许复用**控制流思想**。

## 7.1 Skill-C1: Bridge Search Procedure

控制流：

```text
1. 从问题中识别 primary entity / relation
2. 发起 first-hop search
3. 读取候选证据
4. 提取 bridge entity
5. 使用 bridge + unresolved relation 发起 next search
6. 收集 supporting evidence
7. 交给 answer synthesis
```

不能写死任何 gold entity。

## 7.2 Skill-C2: Comparison Alignment Procedure

控制流：

```text
1. 解析两个待比较实体
2. 独立搜索 entity A
3. 独立搜索 entity B
4. 抽取同一 comparison attribute
5. 检查 unit / date / qualifier
6. 执行比较
7. 输出答案
```

## 7.3 Skill-C3: Long-Chain Search Procedure

控制流：

```text
1. 初始化 unresolved subgoal
2. search/open
3. 提取 next-hop entity
4. 更新 evidence state
5. 检查是否满足 final target
6. 若未满足，继续下一 hop
7. 最多执行固定 hop/tool budget
8. synthesis
```

必须维护：

```text
visited entities
visited documents
evidence chain
remaining subgoal
```

避免循环检索。

---

# 8. 先验证 skill，不验证 acquisition

在正式比较 acquisition policy 前，先做 external R2-style validation。

对每个 capability：

```text
no_skill
vs
executable_skill
```

在 `skill_validation` 上评估。

至少输出：

```text
Answer Accuracy / EM
F1
tool calls
search recall / supporting evidence recall（若可计算）
procedure compliance
invalid tool-call rate
token cost
```

### Skill accept 条件

每个 capability：

```text
ΔAccuracy > 0
AND
提升超过 >= 2 个 validation examples
AND
没有明显 catastrophic tool-loop / parsing failure
```

至少：

```text
2 / 3 capabilities accepted
```

才能进入 acquisition 主实验。

如果只有 0–1 个 skill accepted：

> **停止。**

这说明 external environment 中 executable assimilation backend 尚未成立，不能拿 acquisition policy 做结论。

此时只允许修：

```text
search adapter bug
tool schema bug
procedure genericity bug
answer parser bug
```

**不允许调 frontier score。**

---

# 9. Calibration：公平初始化 capability frontier

Helion 有 E0 capability audit；外部 benchmark 也必须有对应步骤。

对每个 capability 使用冻结的 calibration set，运行 static base agent。

输出：

```text
accuracy
pass@4 on failures（最多 16 failed examples / cap）
avg tool calls
search miss
reasoning / synthesis failure proxy
```

得到 capability profile：

```text
hard gap
unstable frontier
already mastered
```

所有 acquisition 方法共享同一 calibration 结果。

记录固定 calibration 成本：

```text
tokens
tool calls
rollouts
```

在最终 efficiency 表中同时报告：

```text
excluding shared calibration
including shared calibration
```

不要隐藏 frontier 需要 calibration 的成本。

---

# 10. Acquisition policy：只比较已经冻结的方法

主方法：

```text
frontier
```

strong baseline：

```text
ace_skill_prioritized
```

control：

```text
passive
```

辅助负控：

```text
reward_seeking
```

如果已有代码直接兼容，可加：

```text
uncertainty
failure_replay
```

但它们不是本轮阻塞项。

---

# 11. Frontier 方法：禁止针对新 benchmark 调权重

必须从当前 R5.2 代码中抽取并冻结：

```text
frontier scoring rule
priority composition
tie-breaking
update rule
```

新 benchmark 只允许替换：

```text
capability IDs
capability calibration statistics
task pool
retrieval backend
```

不允许：

```text
看到 Hotpot/2Wiki/MuSiQue 结果后调 frontier coefficient
为 MuSiQue 单独加 bonus
人为提高某个 capability sampling probability
对 Ace-Skill 不做同等预算处理
```

必须输出：

```text
FRONTIER_PORT_AUDIT.md
```

明确哪些代码完全复用、哪些是环境 adapter。

---

# 12. Gating：继续固定 CountOnly unique-task

Primary：

```text
K = 8
```

skill eligibility：

```text
某 capability 获得 >= 8 个 distinct task IDs
→ 对应 executable primitive eligible
```

重复 rollout 不计入 unique count。

禁止：

```text
RealFit
CrossCap fit
content-conditioned config learning
dynamic K tuning
```

本轮仍然验证的是：

```text
what to acquire for gating
```

不是：

```text
how to learn a procedure from content
```

---

# 13. Acquisition budget

Primary：

```text
budget = 48
```

checkpoint：

```text
0 / 8 / 16 / 24 / 32 / 40 / 48
```

full future eval 至少：

```text
0 / 16 / 32 / 48
```

记录每个 checkpoint：

```text
unique evidence per capability
active skills
future macro accuracy
per-dataset accuracy
per-capability accuracy
tokens
tool calls
```

---

# 14. 先做 3-seed pilot，不要直接烧 10 seed

Stage-1：

```text
seed = 0,1,2
```

methods：

```text
frontier
ace_skill_prioritized
passive
```

### Stage-1 继续条件

只有满足：

```text
mean(frontier ΔFuture) > mean(Ace ΔFuture)
AND
mean(frontier ΔFuture) > mean(passive ΔFuture)
AND
frontier 至少在 2/3 seed > Ace
AND
frontier 至少多激活一个 target capability，或 AULC 明显更高
```

才扩展到 10 seed。

否则先停止分析，不要自动调方法。

---

# 15. Stage-2：10-seed confirmation

若 Stage-1 PASS：

```text
seed = 0..9
```

不得筛 seed。

建议 GPU 调度前先：

```bash
nvidia-smi
```

如果 H20 4–7 当前仍空闲，可分配：

```text
GPU4: seeds 0,4,8
GPU5: seeds 1,5,9
GPU6: seeds 2,6
GPU7: seeds 3,7
```

如果某卡被占用：

- 自动使用剩余空闲卡；
- 不抢占已有进程；
- runner 必须支持 resume。

---

# 16. Primary outcome

主指标：

```text
Macro Future Accuracy
```

定义：

```text
macro(
    HotpotQA accuracy,
    2Wiki accuracy,
    MuSiQue accuracy
)
```

同时保留：

```text
normalized EM
token-level F1
```

不要把多个指标人为加权成新的 composite 主分数。

---

# 17. 必须输出的外部泛化指标

## 17.1 Performance

```text
Base Accuracy
Final Accuracy
ΔFuture
per-dataset Accuracy
per-capability Accuracy
Bamboogle zero-shot Accuracy
```

## 17.2 Sample efficiency

```text
AULC
budget_to_first_skill
budget_to_second_skill
budget_to_third_skill
ΔFuture / rollout
ΔFuture / unique task
ΔFuture / 100k tokens
ΔFuture / 100 tool calls
```

## 17.3 Acquisition behavior

```text
unique tasks / 48
duplicates
capability histogram
dataset histogram
capability entropy
# active skills
```

重点检查：

> frontier 的优势究竟来自真正选择 capability gap，还是仅仅偏向某个 dataset。

因此一定要区分：

```text
dataset allocation
vs
capability allocation
```

## 17.4 Search-agent diagnostics

建议输出：

```text
avg search calls
avg open/read calls
supporting evidence recall（可计算时）
answer synthesis failure rate
invalid tool-call rate
loop rate
average trajectory length
```

这些作为诊断，不替代主 Accuracy。

---

# 18. Bamboogle：只做 final zero-shot transfer

Bamboogle 不参与：

```text
calibration
skill validation
acquisition
threshold tuning
router training
```

在主实验结束后，用最终 activated skill set 做：

```text
no_skill
frontier-activated skills
Ace-activated skills
passive-activated skills
```

比较。

目的：

> 检查 acquisition 决定的 skill eligibility 是否能迁移到一个没有参与 acquisition 的困难 decomposition benchmark。

如果 Bamboogle 下降，也必须原样报告。

---

# 19. Strong baseline 与现有工作如何处理

## 19.1 Ace-Skill

必须保留为 acquisition strongest baseline。

继续使用 R5.2 已审计实现，不重新“简化”。

## 19.2 SESA

SESA 使用 self-play task generation + frontier shaping + evolving skill memory + continual RL。

它不是 acquisition-only baseline。

本轮不要把 SESA 拆成一个 heuristic 然后称为完整 baseline。

但是因为本轮 benchmark 与 SESA 的公开 evaluation suite 对齐，最终表中可以加入：

```text
SESA paper/released checkpoint reference result
```

前提是明确标记：

```text
different training budget / backbone / protocol
reference only, not apples-to-apples
```

若后续决定做 system-level comparison，再单独按其官方代码复现。

## 19.3 SkillRL

同理。

SkillRL 改变：

```text
skill distillation
skill retrieval
skill evolution
RL training
```

不是单独的 acquisition policy。

本轮不要做假的 “SkillRL acquisition baseline”。

可作为后续 system-level baseline。

---

# 20. 本轮禁止事项

```text
- 不调 Helion
- 不调 frontier coefficient
- 不调 K
- 不训 R6 carrier router
- 不重启 RealFit
- 不做 LoRA/SFT assimilation sweep
- 不人工挑选容易展示优势的 public QA subset
- 不把 dataset id 当 capability
- 不用 future labels 训练 acquisition/router
- 不因为某 seed 失败就删 seed
- 不因为 external result 较弱就重新抽 split
- 不使用 mock retrieval
```

---

# 21. 预注册的 Go / No-Go

## Case A — External Generalization PASS

10-seed 正式结果满足：

```text
mean(frontier - Ace) >= 0.03
AND
paired bootstrap 95% CI lower bound > 0
AND
frontier wins >= 7/10 seeds
AND
frontier > passive
AND
至少 2/3 dataset 的 mean gain 不为负
```

并满足至少一个 efficiency 条件：

```text
AULC(frontier) > AULC(Ace)
OR
frontier 更早激活第二个 target capability
```

### Case A 后

停止 method tuning。

下一步进入：

```text
paper-scale baseline expansion
```

优先：

1. 在同一 public search suite 上补更多模型 backbone；
2. 做 SESA / SkillRL system-level comparison；
3. 再决定是否加入第三 benchmark；
4. 开始组织主表、机制图和论文 claim。

## Case B — Frontier > passive，但与 Ace-Skill 持平

例如：

```text
frontier > passive
but
frontier - Ace CI crosses 0
```

说明：

> capability-frontier 在 Helion 上的优势没有稳健迁移到公开 search distribution。

此时**不要继续调 public benchmark 小权重**。

下一步应回到方法层，考虑真正的：

```text
Evidence-Sufficiency Gating
```

而不是 count-only：

```text
count >= K
```

重点研究：

```text
quantity
+ diversity
+ repeated verified failure
+ evidence consistency
+ marginal expected utility
```

## Case C — Frontier 不优于 passive

说明：

> Helion 的 positive result 很可能依赖 synthetic capability structure。

此时暂停论文主线扩张。

先做 failure analysis：

```text
capability taxonomy 是否失效
frontier calibration 是否不可迁移
skill primitives 是否覆盖错误
公开任务 capability 是否高度连续而非离散
```

不要直接继续堆更多 benchmark。

---

# 22. 输出目录

统一：

```text
outputs/r5_3_external_generalization/
```

至少包含：

```text
R5_3_MASTER_SUMMARY.md
R5_3_DECISION.json
R5_3_FINAL_METRICS.csv
R5_3_PER_SEED.csv
R5_3_PER_DATASET.csv
R5_3_PER_CAPABILITY.csv
R5_3_LEARNING_CURVES.csv
R5_3_ACTIVATION_TIMES.csv
R5_3_ACQUISITION_DISTRIBUTION.csv
R5_3_EFFICIENCY.csv
R5_3_SEARCH_DIAGNOSTICS.csv
R5_3_PAIRED_BOOTSTRAP.json
R5_3_BAMBOOGLE_TRANSFER.csv

SPLIT_AUDIT.md
CAPABILITY_MAP_AUDIT.md
FRONTIER_PORT_AUDIT.md
SKILL_VALIDATION.md
ROUTER_AUDIT.md
RETRIEVER_AUDIT.md

per_seed/
```

---

# 23. 推荐实现脚本

新增：

```text
scripts/setup_public_search_backend.py
scripts/build_r5_3_public_splits.py
scripts/build_public_capability_map.py
scripts/validate_r5_3_skills.py
scripts/run_r5_3_external.py
scripts/analyze_r5_3_external.py
scripts/eval_r5_3_bamboogle.py
```

runner 必须支持：

```text
--methods
--seeds
--budget
--gpu
--resume
--datasets
--retriever-url
--output-dir
```

---

# 24. 实现顺序

严格按以下顺序，不要直接一口气跑正式实验。

## Step 1 — 环境 smoke test

完成：

```text
dataset loader
retrieval server
tool call
answer parser
metric
```

随机 10 题，只检查系统可运行。

不计正式结果。

## Step 2 — Freeze split

执行：

```bash
python scripts/build_r5_3_public_splits.py \
  --datasets hotpotqa 2wiki musique bamboogle \
  --output-dir outputs/r5_3_external_generalization
```

写 hash。

之后不改。

## Step 3 — Capability mapping audit

```bash
python scripts/build_public_capability_map.py \
  --input-dir outputs/r5_3_external_generalization \
  --output-dir outputs/r5_3_external_generalization
```

检查每类样本足够支持：

```text
K=8
```

若某 capability acquisition pool 总量 < 24：

不要复制样本。

记录 coverage limitation。

## Step 4 — Static capability audit

跑 calibration。

生成 external E0-like report。

## Step 5 — Executable skill validation

```bash
python scripts/validate_r5_3_skills.py \
  --datasets hotpotqa 2wiki musique \
  --output-dir outputs/r5_3_external_generalization
```

至少 2/3 skill accepted 才继续。

## Step 6 — 3-seed acquisition pilot

示例统一入口：

```bash
python scripts/run_r5_3_external.py \
  --methods frontier ace_skill_prioritized passive \
  --seeds 0 1 2 \
  --budget 48 \
  --datasets hotpotqa 2wiki musique \
  --resume \
  --output-dir outputs/r5_3_external_generalization
```

分析：

```bash
python scripts/analyze_r5_3_external.py \
  --input-dir outputs/r5_3_external_generalization \
  --stage pilot
```

不满足 Stage-1 继续条件则停止。

## Step 7 — 扩到 10 seed

仅在 pilot PASS 后执行。

先：

```bash
nvidia-smi
```

再按当前空闲 GPU 分组。

如果 4–7 空闲，可参考：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_r5_3_external.py \
  --methods frontier ace_skill_prioritized passive \
  --seeds 0 4 8 \
  --budget 48 \
  --datasets hotpotqa 2wiki musique \
  --resume \
  --output-dir outputs/r5_3_external_generalization &

CUDA_VISIBLE_DEVICES=5 python scripts/run_r5_3_external.py \
  --methods frontier ace_skill_prioritized passive \
  --seeds 1 5 9 \
  --budget 48 \
  --datasets hotpotqa 2wiki musique \
  --resume \
  --output-dir outputs/r5_3_external_generalization &

CUDA_VISIBLE_DEVICES=6 python scripts/run_r5_3_external.py \
  --methods frontier ace_skill_prioritized passive \
  --seeds 2 6 \
  --budget 48 \
  --datasets hotpotqa 2wiki musique \
  --resume \
  --output-dir outputs/r5_3_external_generalization &

CUDA_VISIBLE_DEVICES=7 python scripts/run_r5_3_external.py \
  --methods frontier ace_skill_prioritized passive \
  --seeds 3 7 \
  --budget 48 \
  --datasets hotpotqa 2wiki musique \
  --resume \
  --output-dir outputs/r5_3_external_generalization &

wait
```

如果卡不空闲，不要抢卡；重新分配。

## Step 8 — Bamboogle final transfer

正式 10-seed 主结果完成后再跑：

```bash
python scripts/eval_r5_3_bamboogle.py \
  --input-dir outputs/r5_3_external_generalization \
  --output-dir outputs/r5_3_external_generalization
```

禁止看到 Bamboogle 后再改 skill。

## Step 9 — Final analysis

```bash
python scripts/analyze_r5_3_external.py \
  --input-dir outputs/r5_3_external_generalization \
  --stage final
```

生成 Case A/B/C。

---

# 25. `R5_3_MASTER_SUMMARY.md` 必须回答的问题

最终总结不能只贴表。

必须逐条回答：

1. Helion 的 `frontier > Ace-Skill` 是否在公开 Search-Agent benchmark 上复现？
2. 10 seed 是否稳定？
3. 哪些 dataset 复现、哪些没有？
4. 哪些 capability 被 frontier 优先覆盖？
5. 优势来自 capability allocation，还是 dataset bias？
6. Ace-Skill 在公开环境中是否仍退化为 passive？
7. 第二个 skill 是否仍由 frontier 更早激活？
8. external executable skill 是否真的产生正 transfer？
9. oracle capability dispatch 换成 text router 后，优势还剩多少？
10. Bamboogle zero-shot transfer 是正、零还是负？
11. 该结果支持：
    ```text
    capability-frontier acquisition
    ```
    还是只支持：
    ```text
    Helion-specific heuristic
    ```
12. 下一步属于 Case A / B / C 哪一个？

---

# 26. 本轮最重要的一句话

> **R5.2 已经证明“不是 Ace-Skill 的换壳”；R5.3 不再问方法能否继续变强，而是问这个优势能否离开 Helion 后仍然存在。**

如果外部泛化通过，再扩大 baseline 和模型规模；如果不过，不要用更多 Helion 内部实验掩盖问题。
