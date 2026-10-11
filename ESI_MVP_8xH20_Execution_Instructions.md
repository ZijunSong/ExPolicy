# ESI-MVP：Experience-to-Executable-Skill Induction 最小闭环执行指令

> 交给八卡 H20 机器上的 Coding/Research Agent 直接执行。  
> **首要目标不是追求论文完整性，而是把核心 pipeline 真正打通：从真实 experience 内容中诱导/修复 executable skill，并在一个公开数据集上稳定优于更新前的 base model。**
>
> Repo：`/data/ppnm/expolicy`  
> Base model：`/data/ppnm/models/Qwen2.5-3B-Instruct`  
> 机器：8×H20-3e  
> 已有公开检索环境：Wikipedia 2018 + E5-base-v2 HNSW，历史服务端口 `18080`  
> 禁止 mock LLM、mock retrieval、gold-as-hit、测试集泄漏、手工把 test case 修进 skill。

---

## 0. 本轮只回答一个问题

当前已有实验表明：

1. episodic / semantic / LoRA 并不能稳定把经验转化为可迁移能力；
2. executable skill 能显著提高 capability performance；
3. 但此前正增益主要来自 **CountOnly / evidence-count gating 解锁 frozen primitive**；
4. 因而 acquisition 再聪明，也容易退化为“更快打开预先存在的 skill mask”。

本轮不继续优化 acquisition。

本轮只验证：

> **能否从真实训练侧 experience 的内容本身，生成/修改一个此前不存在的 executable skill，使它在未见验证/测试问题上提高 Qwen2.5-3B-Instruct 的效果？**

只要这个闭环成立，就说明后续研究有资格重新讨论 experience value / acquisition policy。

---

# 1. MVP 名称与核心对象

工作名：

**ESI-MVP — Experience-to-Executable-Skill Induction**

核心映射：

```text
raw experience trajectories
        ↓
ExperiencePacket
        ↓
Skill Inducer
        ↓
candidate SkillSpec(s)
        ↓
safe compiler
        ↓
ExecutableSkill
        ↓
held-out validation
        ↓
accepted skill
        ↓
future test
```

必须满足：

```text
B1 != B2
    ↓
induced SkillSpec may differ
    ↓
runtime behavior may differ
    ↓
future performance may differ
```

也就是说，experience 不允许只充当：

```text
count(capability) >= K → unlock pre-installed skill
```

---

# 2. 数据集：只做 2WikiMultiHopQA

## 2.1 主数据集

本轮只做：

**2WikiMultiHopQA**

优先聚焦已有 taxonomy 中：

```text
comparison_alignment
```

原因是历史实验中该 capability 的 executable procedure 转移最明显，最适合用于首先验证“experience → executable capability”闭环。

但是：

> **严禁直接导入、复制、调用或轻微改写 R5.3/R5.4 的 frozen C2 executable primitive 作为最终方法。**

允许复用：

- 通用 `search`
- 通用 `open`
- retriever client
- 通用 answer evaluator
- dataset loader
- text-router 基础设施
- generic runtime operators

不允许复用：

- 已写好的 `comparison_alignment` 完整 procedure
- 任何包含目标任务专用 decision logic 的 frozen primitive
- R2/R5.3 中“打开就能工作”的完整 skill 实现

---

# 3. 成功标准

## 3.1 最小成功标准（本轮首要目标）

在一个**从未参与 skill induction / candidate selection 的 2Wiki future test split**上：

```text
EM(ESI-induced skill) > EM(base model)
```

并至少满足：

```text
absolute EM gain >= +0.03
```

如果 test=150 题，则至少约 +5 题。

这是本轮 MVP 的最低 SHIP 条件。

---

## 3.2 推荐成功标准

若计算资源允许，跑 seeds = 0,1,2：

```text
mean ΔEM >= +0.03
至少 2/3 seeds induced > base
```

Stretch target：

```text
mean ΔEM >= +0.05
3/3 seeds induced > base
```

---

## 3.3 不要求本轮完成的事项

本轮**不要求**：

- Active Acquisition > Passive
- learned acquisition policy
- learned carrier routing
- LoRA / SFT
- 多 benchmark 泛化
- Bamboogle
- 10 seed
- 完整论文主表
- 新的 AVA/RASA/frontier score
- 自动 skill library 生命周期
- skill composition
- weight internalization

先把最小闭环打通。

---

# 4. 数据切分：必须重新建立 MVP split

不要直接拿已经反复看过的 R5.3/R5.4/R6 future 作为本轮最终 test。

实现：

```text
scripts/build_esi_mvp_split.py
```

优先复用 repo 中 R5.3/R5.4 的 2Wiki loader，然后构造：

```text
train_experience_pool = 240
validation            = 80
future_test           = 150
negative_control      = 80
```

其中：

- `train_experience_pool`：允许 rollout、允许看 gold answer 计算 train-side correctness；
- `validation`：允许用于 candidate selection / one-round repair；
- `future_test`：严格 final only；
- `negative_control`：优先选 2Wiki 中非 comparison 的题，用于检测 skill 是否被错误泛化。

如果可用样本不足，按比例缩小，但必须满足：

```text
train >= 120
validation >= 40
future_test >= 100
negative_control >= 40
```

split seed 固定：

```text
20260818
```

输出：

```text
outputs/esi_mvp_2wiki/split.json
outputs/esi_mvp_2wiki/SPLIT_AUDIT.md
```

审计必须包含：

```text
task-id overlap = 0
exact-question overlap = 0
train/val/test hashes
dataset distribution
capability distribution
```

如果已有 R5.3/R5.4/R6 split 文件可读：

**额外排除这些实验用过的 future IDs。**

如果由于 loader / dataset metadata 原因无法完全排除，必须在 `SPLIT_AUDIT.md` 明确写出，而不能静默继续。

---

# 5. 先跑 clean baseline

实现：

```text
scripts/run_esi_base_eval.py
```

条件：

```text
model = Qwen2.5-3B-Instruct
temperature = 0
retrieval = Wikipedia2018 + E5
真实 query → retrieve → open → answer
```

分别评：

```text
validation
future_test
negative_control
```

注意：

**future_test 的 base 分数可以提前算，因为它不会影响方法；但是 candidate induction / selection 代码不允许读取 future-test answer / metric。**

保存：

```text
outputs/esi_mvp_2wiki/base/
    validation.jsonl
    future_test.jsonl
    negative_control.jsonl
    metrics.json
```

---

# 6. Experience Collection

实现：

```text
scripts/run_esi_collect.py
```

在 `train_experience_pool` 上运行 base agent，收集真实 trajectory。

每条经验至少保存：

```json
{
  "task_id": "...",
  "question": "...",
  "gold_answer_train_only": "...",
  "predicted_answer": "...",
  "correct": false,
  "tool_calls": [],
  "queries": [],
  "retrieved_docs": [],
  "opened_docs": [],
  "evidence_snippets": [],
  "termination_reason": "...",
  "token_count": 0,
  "predicted_capability": "comparison_alignment"
}
```

禁止：

- 把 gold document 当 retrieve hit；
- 在 runtime 中直接传 gold supporting facts；
- 为了产生 success 直接读取答案后构造 action trace。

训练侧 gold answer 只能用于：

```text
correctness / failure diagnosis / train-side repair supervision
```

---

# 7. 八卡 GPU：第一阶段并行采集

建议第一波直接 8 卡切 shard：

```text
GPU0 → shard 0/8
GPU1 → shard 1/8
GPU2 → shard 2/8
GPU3 → shard 3/8
GPU4 → shard 4/8
GPU5 → shard 5/8
GPU6 → shard 6/8
GPU7 → shard 7/8
```

统一设置：

```bash
CUDA_VISIBLE_DEVICES=<gpu>
```

每个 shard 独立输出：

```text
outputs/esi_mvp_2wiki/collect/shard_{i}.jsonl
```

最后 merge：

```text
outputs/esi_mvp_2wiki/collect/train_experiences.jsonl
```

禁止 8 个进程同时写一个 jsonl。

---

# 8. ExperiencePacket：真正给 inducer 的内容

新建：

```text
expolicy/skill_induction/experience_packet.py
```

不要直接把 240 条原始 trajectory 全塞进 prompt。

构造紧凑的 `ExperiencePacket`。

建议最多：

```text
16 high-confidence failures
8 successes
8 diverse failures
总计 <= 32 experiences
```

如果 base success 太少：

```text
successes < 4
```

允许从 train pool 上增加 **真实 search budget / stochastic rollout** 寻找成功轨迹：

```text
temperature = 0.6
最多 4 rollouts / task
```

仍然必须是真实 retrieval。

不允许用 test。

---

## 8.1 Failure 摘要

每条 failure 转成：

```text
Question structure
Observed tool sequence
Retrieved evidence summary
What information was found
What information was missing
Final wrong answer
Gold answer (TRAIN ONLY)
Failure category
```

failure category 先用规则 + LLM 二级判断：

```text
entity_parse
decomposition
query_construction
retrieval_coverage
attribute_extraction
entity_alignment
comparison_operator
answer_synthesis
premature_stop
other
```

保存：

```text
outputs/esi_mvp_2wiki/packets/packet_round0.json
```

---

# 9. SkillSpec：使用受限 DSL，而不是让 LLM 任意生成 Python

新建：

```text
expolicy/skill_induction/schema.py
```

使用 Pydantic 定义：

```python
class SkillSpec(BaseModel):
    name: str
    version: str
    target_description: str

    trigger_cues: list[str]

    entity_strategy: dict
    decomposition_strategy: dict

    query_templates: list[str]
    max_search_rounds: int
    retrieve_topk: int

    evidence_slots: list[dict]
    extraction_rules: list[dict]
    alignment_rules: list[dict]

    comparison_rule: dict
    termination_rule: dict
    answer_rule: dict

    provenance_experience_ids: list[str]
    rationale_summary: str
```

核心约束：

- `max_search_rounds <= 6`
- `retrieve_topk <= 10`
- 不允许 file IO
- 不允许 shell
- 不允许 arbitrary Python expression
- 不允许 hardcode test task id
- 不允许 gold answer
- 不允许 gold doc id
- 不允许 literal answer lookup table

---

# 10. Generic Runtime Operators

新建：

```text
expolicy/skill_induction/operators.py
```

只实现通用算子：

```text
detect_entities
classify_relation
instantiate_query
search
open
extract_attribute
normalize_value
align_entities
compare_values
synthesize_answer
check_evidence_complete
```

这些 operator 可以使用本地 Qwen2.5-3B-Instruct 做局部结构化推理。

例如：

```text
extract_attribute(question, entity, evidence)
```

让 3B 只返回：

```json
{
  "attribute": "...",
  "value": "...",
  "evidence_quote": "...",
  "confidence": 0.0
}
```

不要让 skill compiler 直接拼一个巨大 prompt 然后退化回普通 CoT。

---

# 11. Skill Compiler

新建：

```text
expolicy/skill_induction/compiler.py
```

接口：

```python
compiled = compile_skill(skill_spec)

result = compiled.run(
    question=question,
    tools=tool_env,
    model=base_model
)
```

编译后的 runtime 必须：

1. 根据 SkillSpec 决定搜索顺序；
2. 显式维护 evidence slots；
3. 只有满足 termination condition 才停止；
4. 最终回答必须由 evidence state 生成；
5. 记录完整 provenance。

输出 trace：

```json
{
  "skill_name": "...",
  "skill_version": "...",
  "steps": [],
  "evidence_state": {},
  "answer": "...",
  "tool_calls": 0,
  "tokens": 0
}
```

---

# 12. Skill Inducer

新建：

```text
expolicy/skill_induction/inducer.py
```

输入：

```text
ExperiencePacket
SkillSpec schema
allowed operators
```

输出：

```text
N = 8 candidate SkillSpecs
```

使用：

```text
Qwen2.5-3B-Instruct
temperature = 0.4 ~ 0.7
structured JSON generation
```

Prompt 的任务不是“回答问题”，而是：

```text
Infer a reusable executable procedure that would systematically repair
the recurring failures in the supplied experiences.

You may only use the listed runtime operators.
Do not memorize answers, document IDs, or task IDs.
Return a valid SkillSpec.
```

每个候选必须注明：

```text
provenance_experience_ids
```

从而可以验证 SkillSpec 真正受 experience 内容影响。

---

# 13. Candidate 生成与 8 卡并行

第一轮生成：

```text
candidate_00.yaml
...
candidate_07.yaml
```

随后每张 GPU 独立评一个 candidate：

```text
GPU0 → candidate 0
GPU1 → candidate 1
...
GPU7 → candidate 7
```

每个 candidate 先：

```text
schema validation
compiler validation
8-task smoke
```

任一步失败直接标：

```text
INVALID
```

不要为了“让实验跑完”偷偷 fallback 到 frozen primitive。

---

# 14. Validation protocol

实现：

```text
scripts/run_esi_candidate_validation.py
```

在 80 道 validation 上比较：

```text
base
candidate_i
```

同时在 `negative_control` 的一个 40-task dev subset 上测误触发。

候选评分：

```text
primary = validation EM
secondary = validation gain
safety = negative-control regression
cost = tool calls / tokens
```

接受候选的最低条件：

```text
ΔEM_validation >= +0.03
negative_control ΔEM >= -0.03
compiler_valid = true
no_leakage = true
```

若多个候选满足：

优先：

```text
highest validation EM
```

若同分：

```text
更少 tool calls
```

保存：

```text
outputs/esi_mvp_2wiki/candidates/
outputs/esi_mvp_2wiki/validation/CANDIDATE_TABLE.csv
```

---

# 15. 允许最多一次 Repair Round

如果 8 个 candidate 都未达到：

```text
ΔEM_validation >= +0.03
```

允许做 **一次** repair。

不是人工 hand-tune。

流程：

```text
best-but-failed candidate
        +
validation failure summaries
        ↓
inducer
        ↓
8 revised SkillSpecs
```

注意：

- repair 只能看 validation；
- 不得看 future_test；
- 仍然使用相同 DSL；
- 仍然每卡一个 candidate；
- 最多 Round 1；
- Round 1 后还不过，就报告 MVP FAIL，不无限调。

输出：

```text
packet_round1.json
candidate_r1_00.yaml ... candidate_r1_07.yaml
```

---

# 16. 最重要的因果检查：Experience Content 必须真的影响 Skill

实现三个低成本对照。

## A. Empty Experience

输入：

```text
ExperiencePacket = empty
```

让 inducer 生成 candidate。

如果 empty candidate 与 real candidate 同样强：

```text
不能声称 experience-induced
```

---

## B. Shuffled Experience

保持问题数量相同，但把：

```text
failure diagnosis / evidence / gold-train outcome
```

跨 task 随机打乱。

生成：

```text
shuffled candidate
```

---

## C. Provenance Difference

比较：

```text
real SkillSpec
empty SkillSpec
shuffled SkillSpec
```

至少检查：

```text
query templates
evidence slots
alignment rules
termination rules
comparison rule
```

输出：

```text
outputs/esi_mvp_2wiki/causal/CONTENT_CAUSALITY.md
```

本轮**不要求** real 一定显著优于 shuffled 才算 pipeline 打通。

但如果：

```text
real ≈ empty
```

必须明确写：

```text
pipeline performance gain exists,
but experience-content causality is not yet established.
```

---

# 17. Freeze before final test

一旦 validation 选出 accepted skill：

写：

```text
outputs/esi_mvp_2wiki/FREEZE.json
```

包含：

```json
{
  "skill_sha256": "...",
  "candidate_id": "...",
  "validation_metrics": {},
  "code_git_head": "...",
  "split_hash": "...",
  "timestamp": "..."
}
```

从这一刻开始：

**禁止修改：**

- SkillSpec
- compiler semantics
- trigger
- operator behavior
- validation split
- test split

只允许修：

```text
crash / serialization / OOM / deterministic infrastructure bug
```

任何会改变结果的逻辑修改，必须：

```text
invalidate old freeze
重新做 validation
```

---

# 18. Final Test

实现：

```text
scripts/run_esi_final_test.py
```

对：

```text
base
accepted_esi_skill
```

在 `future_test` 上评测。

首轮先 seed=0。

若：

```text
ΔEM >= +0.03
```

立即进入 seed 1/2 复现。

如果 seed0：

```text
0 < ΔEM < +0.03
```

记录为 weak positive，可继续 seeds 1/2，但不改方法。

如果：

```text
ΔEM <= 0
```

本轮 MVP 判 FAIL。

**测试之后禁止再修 SkillSpec。**

---

# 19. Dispatch：本轮不要把 router 做复杂

为了隔离核心问题，本轮允许两种模式。

## Primary

仅在已知的 2Wiki comparison-alignment evaluation subset 上启用 accepted skill。

这里的 capability subset 必须由数据预先标注 / frozen taxonomy 得到，不能利用答案。

该模式主要验证：

```text
experience → skill induction → executable behavior → future gain
```

## Secondary

如果已有 text-router 容易复用，再做：

```text
question text → skill/no-skill
```

但 secondary 不阻塞 MVP。

---

# 20. Negative Control

accepted skill 还必须在：

```text
negative_control
```

上评一次。

目标不是要求完全不降，而是检测 procedure 是否粗暴接管所有任务。

记录：

```text
base EM
skill EM
false activation rate
tool overhead
```

若 primary 是 oracle subset dispatch，则同时报告：

```text
skill forced on negative control
```

作为 stress test。

---

# 21. 必须新增的代码文件

建议目录：

```text
expolicy/
  skill_induction/
    __init__.py
    schema.py
    experience_packet.py
    failure_diagnosis.py
    operators.py
    compiler.py
    inducer.py
    validator.py
    registry.py
```

脚本：

```text
scripts/
  build_esi_mvp_split.py
  run_esi_base_eval.py
  run_esi_collect.py
  run_esi_synthesize.py
  run_esi_candidate_validation.py
  run_esi_causal_controls.py
  run_esi_final_test.py
  orchestrate_esi_mvp.py
  watchdog_esi_mvp.py
  analyze_esi_mvp.py
```

测试：

```text
tests/
  test_esi_schema.py
  test_esi_compiler.py
  test_esi_no_gold_leakage.py
  test_esi_split_isolation.py
  test_esi_clone_isolation.py
  test_esi_determinism.py
  test_esi_no_frozen_primitive_import.py
```

---

# 22. 严格禁止的捷径

以下情况直接视为实验无效：

### 22.1 复用 frozen C2 skill

禁止：

```text
import old_comparison_alignment_skill
copy old procedure
load R2/R5.3 C2 primitive as accepted skill
```

允许读其接口以兼容 runtime，但最终行为逻辑必须来自本轮 induced SkillSpec。

---

### 22.2 CountOnly activation

本轮不存在：

```text
count >= 8 → open skill
```

Skill 是否存在，必须由：

```text
ExperiencePacket → Inducer → SkillSpec
```

产生。

---

### 22.3 Test-aware repair

禁止：

```text
看 future-test 错题
→ 改 prompt
→ 改 SkillSpec
→ 重跑
```

---

### 22.4 Gold-as-hit

gold answer 可用于 train/validation metric。

禁止：

```text
gold document → open
gold supporting fact → evidence slot
gold answer → runtime hint
```

---

### 22.5 手工写死答案模式

例如：

```python
if "which was founded earlier" in q:
    return ...
```

不允许按具体问题字符串写规则。

DSL 中允许通用 relation/operator：

```text
earlier/later
larger/smaller
same/different
```

但必须由 inducer 基于 experiences 产生配置。

---

# 23. GPU 调度建议

## Phase A：Base + Split Audit

```text
GPU0 validation base
GPU1 future base
GPU2 negative control base
GPU3-7 idle / prepare collection
```

如果 future base 太快，可合并到 GPU0。

---

## Phase B：Experience Collection

八卡全开：

```text
GPU0-7 → 8 shards
```

---

## Phase C：Candidate Validation

八卡全开：

```text
GPU0-7 → candidate 0-7
```

---

## Phase D：Repair Round（如需要）

仍然：

```text
GPU0-7 → revised candidate 0-7
```

---

## Phase E：Final Seeds

若 skill 通过：

```text
GPU0 → seed0 base/skill
GPU1 → seed1 base/skill
GPU2 → seed2 base/skill
GPU3 → causal empty
GPU4 → causal shuffled
GPU5 → negative control
GPU6 → cost/profiling
GPU7 → spare / failed-job retry
```

不要为了“8 卡必须打满”同时做无关实验。

---

# 24. Orchestrator

实现：

```text
scripts/orchestrate_esi_mvp.py
```

推荐命令：

```bash
cd /data/ppnm/expolicy

python scripts/orchestrate_esi_mvp.py \
  --model /data/ppnm/models/Qwen2.5-3B-Instruct \
  --retriever-url http://127.0.0.1:18080 \
  --dataset 2wiki \
  --target-capability comparison_alignment \
  --gpus 0,1,2,3,4,5,6,7 \
  --split-seed 20260818 \
  --candidate-count 8 \
  --max-repair-rounds 1 \
  --output outputs/esi_mvp_2wiki
```

如果当前 repo CLI 不兼容，可按项目现有 pattern 调整参数，但最后必须提供一个一键 orchestrator。

---

# 25. Resume / Watchdog

参考已有 R5/R6 风格，支持：

```text
--resume
per-stage DONE.json
per-GPU lock
atomic write
```

每个阶段写：

```text
STATUS.json
DONE.json
```

watchdog：

```text
scripts/watchdog_esi_mvp.py
```

只处理：

```text
GPU process died
no output progress
OOM
retriever unavailable
corrupt partial jsonl
```

不要自动修改科学参数。

---

# 26. Retriever 预检

开跑前：

```bash
curl http://127.0.0.1:18080/health
```

如果接口名称不同，从 R5.3/R5.4 脚本中查实际 health/search endpoint。

必须确认：

```text
corpus = Wikipedia 2018
index docs ≈ 21M
E5 retriever
非 mock
```

写：

```text
outputs/esi_mvp_2wiki/RETRIEVER_AUDIT.md
```

---

# 27. Pipeline Audit

在 GPU 大规模运行前，先实现：

```text
scripts/audit_esi_pipeline.py
```

至少检查：

```text
[ ] split zero overlap
[ ] future-test not loaded by inducer
[ ] future-test not loaded by validator
[ ] SkillSpec has provenance IDs
[ ] provenance IDs all belong to train pool
[ ] no gold doc IDs in SkillSpec
[ ] no gold answers in SkillSpec
[ ] compiler forbids arbitrary code
[ ] old frozen C2 primitive not imported
[ ] empty packet cannot silently load old skill
[ ] candidate validation uses validation only
[ ] final test skill hash equals FREEZE hash
[ ] tool environment is real
[ ] retriever is real
```

所有核心项 PASS 后再跑。

---

# 28. Unit Tests

至少做到：

```bash
pytest -q \
  tests/test_esi_schema.py \
  tests/test_esi_compiler.py \
  tests/test_esi_no_gold_leakage.py \
  tests/test_esi_split_isolation.py \
  tests/test_esi_no_frozen_primitive_import.py
```

必须 100% PASS。

---

# 29. 结果汇总

实现：

```text
scripts/analyze_esi_mvp.py
```

生成：

```text
outputs/esi_mvp_2wiki/ESI_MVP_SUMMARY.md
outputs/esi_mvp_2wiki/ESI_MVP_METRICS.csv
outputs/esi_mvp_2wiki/ESI_MVP_DECISION.json
```

主表至少：

| condition | val EM | test EM | Δtest | neg-control EM | tools | tokens |
|---|---:|---:|---:|---:|---:|---:|
| base | | | 0 | | | |
| ESI induced | | | | | | |
| empty-experience control | | | | | | |
| shuffled-experience control | | | | | | |

若跑 3 seed，另表：

| seed | base | ESI | Δ |
|---:|---:|---:|---:|
| 0 | | | |
| 1 | | | |
| 2 | | | |

---

# 30. 自动决策树

## CASE_A_PIPELINE_PASS

满足：

```text
test ΔEM >= +0.03
```

若 3 seed：

```text
mean ΔEM >= +0.03
>=2/3 seeds win
```

结论：

```text
The experience-to-executable-skill pipeline is operational.
Training-side experience content can be converted into an executable
procedure that improves held-out performance over the frozen base model.
```

然后停止扩 acquisition。

下一阶段才研究：

```text
content causality
multiple skill candidates
skill editing
skill composition
acquisition conditioned on inducibility
```

---

## CASE_B_WEAK_POSITIVE

```text
0 < test ΔEM < +0.03
```

结论：

```text
pipeline mechanically works but effect size is too small.
```

保留所有 artifact，不看 test 调参。

下一轮重新开新 validation/test split 再改 inducer。

---

## CASE_C_NO_GAIN

```text
test ΔEM <= 0
```

立即判：

```text
MVP FAIL
```

不要继续 acquisition。

先检查：

```text
candidate skill quality
operator expressivity
compiler/runtime correctness
validation overfit
trigger/dispatch
```

---

## CASE_D_VALIDATION_FAIL

没有 candidate 在 validation 上达到：

```text
+0.03
```

即使 one repair round 后仍失败：

```text
STOP BEFORE FINAL TEST
```

这是最干净的失败。

---

# 31. Agent 行为约束

你是负责实现和运行实验的 Agent。

请直接执行，不要停留在方案讨论。

工作原则：

1. 先检查现有 repo，最大化复用 dataset/retriever/evaluator/runtime 基础设施；
2. 不要重写已有稳定模块；
3. 新模块必须有 unit test；
4. 不使用 mock；
5. 不用 test 调方法；
6. 遇到工程 bug 自行修；
7. 遇到科学指标不过，不偷偷放宽门槛；
8. 不为了跑出正结果修改 split seed；
9. 不因为单个 candidate 失败就手工写一个 task-specific rule；
10. 每阶段完成后更新 `ESI_RUN_LOG.md`；
11. 全部结果必须从实际输出自动生成 summary；
12. 不把未运行的实验写成已完成。

---

# 32. 执行顺序

严格按下面顺序：

```text
Step 1  inspect repo / locate R5.3-R6 reusable infrastructure
Step 2  implement split builder
Step 3  implement audit + unit tests
Step 4  build frozen MVP split
Step 5  run base
Step 6  collect train experiences
Step 7  build ExperiencePacket
Step 8  implement SkillSpec DSL
Step 9  implement generic operators + compiler
Step 10 implement inducer
Step 11 synthesize 8 candidates
Step 12 validate 8 candidates in parallel
Step 13 if needed, one repair round
Step 14 freeze best accepted SkillSpec
Step 15 run final test
Step 16 if positive, run seeds 1/2
Step 17 run empty/shuffled causal controls
Step 18 produce summary + decision
```

不要跳过 split/audit/freeze。

---

# 33. 最低交付物

任务结束时必须至少存在：

```text
outputs/esi_mvp_2wiki/
  SPLIT_AUDIT.md
  RETRIEVER_AUDIT.md
  PIPELINE_AUDIT.md
  split.json
  base/
  collect/
  packets/
  candidates/
  validation/
  causal/
  FREEZE.json                 # 仅 candidate 通过 validation 时存在
  ESI_MVP_METRICS.csv
  ESI_MVP_DECISION.json
  ESI_MVP_SUMMARY.md
  ESI_RUN_LOG.md
```

以及可运行代码：

```text
expolicy/skill_induction/*
scripts/orchestrate_esi_mvp.py
scripts/analyze_esi_mvp.py
tests/test_esi_*.py
```

---

# 34. 最终需要回答的 6 个问题

`ESI_MVP_SUMMARY.md` 最后必须明确回答：

1. **Base model 在 2Wiki future test 上是多少？**
2. **Induced executable skill 是否显著优于 base？**
3. **提升来自哪个 runtime behavior 改变？**
4. **SkillSpec 是否确实由 training experiences 生成，而不是加载 frozen primitive？**
5. **Empty / shuffled experience control 表现如何？**
6. **下一步值得继续研究的是 acquisition，还是 skill induction / skill editing？**

---

# 35. 本轮最重要的判定原则

不要把目标误解成：

> “做一个很复杂的新 ExPolicy。”

本轮目标只有：

> **先证明真实 experience 可以被转换成一个新的 executable capability，并在未见任务上让 frozen base model 变强。**

如果这一步都没有成立：

```text
不要继续研究 experience acquisition policy。
```

如果这一步成立：

下一阶段才有资格定义：

```text
V(B | S, A)
```

因为此时不同 experience batch 才可能通过不同 induced / edited skills，真正产生不同 future utility。

---

# 36. 立即开始

从：

```bash
cd /data/ppnm/expolicy
git status
git rev-parse HEAD
rg -n "R5_3|R5_4|R6|comparison_alignment|2Wiki|retriever|18080" .
```

开始定位可复用代码。

先实现：

```text
build_esi_mvp_split.py
audit_esi_pipeline.py
schema.py
compiler.py
```

通过最小 unit tests 后，再启动八卡 rollout。

**不要先做大规模实验，再补审计。**
