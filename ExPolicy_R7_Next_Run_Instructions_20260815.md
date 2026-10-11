# ExPolicy R7 下一阶段运行指令：Cross-Model Generalization、Strong Assimilator Swap 与 Experience Policy Pivot

> 日期：2026-08-15  
> repo：`/data/ppnm/expolicy`  
> 默认模型目录：`/data/ppnm/models`  
> 默认 GPU：`4,5,6,7`（若实际空闲卡不同，只改 GPU 映射，不改实验协议）  
> 本轮目标不是“继续给 R6 调参”，而是回答三个更重要的问题：
>
> 1. 当前在 Helion 上最强的 `frontier acquisition → experience-gated executable skill activation` 是否能跨模型、跨标准 Search-Agent benchmark 成立；
> 2. R5.3–R6 的失败是否主要由 **CountOnly + frozen replacement-skill** 造成的退化上界，而不是 “experience policy” 这个方向本身没有价值；
> 3. 在换成真正 **content-sensitive** 的 assimilation backend 后，是否存在足够大的 experience-value / carrier-value 差异，使一个 learned Experience Policy 成为必要且可发表的对象。

---

## 0. 本轮最高原则

### 0.1 不允许继续做的事情

禁止直接继续以下路线：

- 不继续调 R5.3 的 `frontier_score` 小权重；
- 不继续调 RASA 的 `delta / confidence / PARK threshold`；
- 不继续调 R6 的 `M=8 / batch=8 / shadow composition / λ` 来追 +0.02；
- 不在旧 `CountOnly K=8` backend 上训练 AVA surrogate；
- 不因为某个子结果弱，就把论文 claim 缩成“某一 capability / 某一个 benchmark / 某一种 heuristic 有效”；
- 不把 Helion 上的 10-seed 阳性继续当作主要泛化证据；
- 不用 mock LLM、mock retriever、gold-as-hit、oracle capability label 作为最终 deployable 主结果；
- 不为了让大模型跑起来而静默改成 4-bit / 8-bit；显存不足时应显式改 tensor parallel 或减少并发；
- 不把 `all-MiniLM-L6-v2`、`cross-encoder-ms-marco-MiniLM-L-6-v2`、`e5-base-v2` 当作 policy backbone，它们只属于 embedding / reranking / retrieval 组件；
- `harness-1` 只有在 preflight 证明它是可加载的 causal LM checkpoint 后才能进入模型表。

### 0.2 如果结果不好，必须按以下优先级处理

1. **先排实现问题 / benchmark artifact；**
2. **再 redesign experiment 或更换 assimilation backend；**
3. **再 change framing 到更强、更本质的问题；**
4. 如果上述都没有 headroom，**discard method**；
5. 禁止通过 narrowing claim 保住旧方法。

---

# 1. 最新实验记录给出的真实结论

当前结果不能再概括成“ExPolicy 不行”，更准确的机制结论是：

1. R2 / R4 证明 executable skill 是目前唯一稳定产生正 near-transfer gain 的载体；
2. episodic / semantic / 当前 LoRA 并没有稳定把 experience 变成 capability；
3. R5 / R5.1 / R5.2 证明，在 Helion + experience-gated executable skill 下，frontier acquisition 明显优于 passive / Ace-Skill；
4. R5.3 在 HotpotQA / 2Wiki / MuSiQue 上完全不泛化；
5. R5.4 的 RASA redesign 没有超过 Ace / passive；
6. R6 即使直接用 exact post-assimilation shadow utility，也只有 `0.1200 vs 0.1193`，因为 CountOnly 下最终性能几乎退化成 activation mask lookup；
7. 因而当前最大的实验瓶颈已经不是“价值估计器不够好”，而是：
   - assimilation backend 的表达能力太弱；
   - acquired experience 的**内容**并没有决定学到什么；
   - frozen skill 只要被 count gate 打开就接近达到固定上界；
   - 这导致 acquisition policy 没有足够大的可优化空间。

因此，R7 的首要任务不是继续给 acquisition heuristic 加 feature，而是**恢复一个真正 content-dependent 的 experience → capability 通道**，再重新检验 policy 是否必要。

---

# 2. Novelty Collision Audit：任何新实现之前必须先完成

截至 2026-08-15，以下 public prior art 已经非常接近“experience policy”。因此不能再把“policy 决定经验怎么用”本身当 novelty。

| 工作 | 已占据的核心空间 | 对 ExPolicy 的限制 |
|---|---|---|
| Ace-Skill, arXiv:2605.08887 | prioritized experience sampling + skill organization | “主动采更有价值的数据”本身不新 |
| SkillOS, arXiv:2605.06614 | learned skill curator；insert/update/delete SkillRepo | “学习 skill curation policy”本身不新 |
| COVE, arXiv:2608.01234 | harness-side vs parametric-side routing + stage-aware trigger | “memory/skill/weight carrier routing”不能单独作为 novelty |
| ExpWeaver, arXiv:2605.07164 | agent 决定何时在 runtime 使用 experience | “when to use experience”已被系统研究 |
| MemCon, arXiv:2607.13591 | learned policy 控制 retrieve / inject / consolidate / forget | memory control policy 已被占据 |
| OPD-Evolver, arXiv:2606.17628 | select/use/write/maintain experience，并通过 OPD 内化 evolver | holistic memory-management policy 已经很接近 |
| SESA, arXiv:2607.29468 | task distribution 与 skill memory 双向 co-evolution | “经验改变后续训练分布”本身也已有直接工作 |
| SkillOpt, arXiv:2605.23904 | rollout-driven、held-out gated skill optimization | skill 作为 trainable external state 已经有强方法 |
| SkillMentor, arXiv:2607.27360 | learned blind-spot diagnosis → 生成诊断任务并形成 corrective skills | “学会发现该练什么”已有非常接近工作 |
| HASP, arXiv:2605.17734 | experience → executable intervention program | executable skill 本身不是 novelty |
| OEL, arXiv:2603.16856 | deployment experience → experiential knowledge → OPCD internalization | experience → parameter 已有完整范式 |
| OPCD, arXiv:2602.12275 | on-policy context distillation | 参数内化本身不新 |
| PATH-Bench, arXiv:2608.01149 | path-dependent lifelong experience + selective experience use | path-dependent experience value 已被明确提出 |

## 2.1 R7 允许继续追求的 stronger novelty

下一阶段只有下面这种级别的贡献才值得继续：

> **Learned Experience Lifecycle Policy**：同一个可迁移 policy 根据当前 executor capability、历史 experience state、budget 与 downstream assimilation response，联合决定  
> **(A) acquire what → (B) transform / assimilate into which substrate → (C) validate / promote / retire when**，  
> 并直接优化 delayed future capability gain、forgetting 与 compute cost，而不是分别用三个 hand-crafted heuristic。

关键区别必须是：

- Ace-Skill：只管 acquisition；
- SkillOS：主要管 SkillRepo curation；
- COVE：主要做 harness/parameter routing 与 rule/judge trigger；
- ExpWeaver / MemCon：主要管 runtime usage；
- OPD-Evolver：主要管 memory evolver；
- SESA：task generation 与 skill bank co-evolve；
- **ExPolicy-v2 必须证明：这些阶段的决策是耦合的，而且 joint policy 相比 strongest stage-wise composition 有额外收益。**

如果做不到这一点，禁止把“Experience Policy”作为论文主贡献。

---

# 3. R7-A：Repository / Skill Ownership / Case Audit

这是所有 GPU 实验之前的强制阶段。

## 3.1 先冻结当前状态

```bash
cd /data/ppnm/expolicy

mkdir -p outputs/r7_policy_generalization/audit
git rev-parse HEAD | tee outputs/r7_policy_generalization/audit/GIT_HEAD.txt
git status --porcelain=v1 | tee outputs/r7_policy_generalization/audit/GIT_STATUS.txt

python - <<'PY'
from pathlib import Path
p = Path("outputs/r7_policy_generalization/audit/RUN_START.txt")
import datetime
p.write_text(datetime.datetime.now().isoformat() + "\n")
PY
```

**不要自动 reset / checkout / clean。**  
如果工作区非 clean，只记录，不覆盖当前实验资产。

---

## 3.2 Cursor Agent 必须先实现并运行以下审计脚本

若脚本不存在，先实现完整版本：

```text
scripts/r7/
├── audit_repo_and_skill_ownership.py
├── audit_r5_r6_cases.py
├── audit_model_compatibility.py
├── audit_novelty_collision.py
├── build_r7_search_suite.py
├── validate_r7_splits.py
├── evaluate_skill_backend_matrix.py
├── orchestrate_r7.py
└── analyze_r7.py

tests/r7/
├── test_no_gold_leakage.py
├── test_skill_does_not_replace_answer_by_default.py
├── test_model_axis_is_real.py
├── test_split_isolation.py
├── test_acquisition_policy_only_changes_acquisition.py
├── test_assimilator_content_sensitivity.py
└── test_shadow_never_reads_future.py
```

要求：

- 不允许 mock；
- 所有测试必须基于真实 repo classes / real serialized artifacts；
- 若发现旧实现设计本身不适合新实验，应新增 `r7/` implementation，不覆盖 R5/R6；
- 所有旧结果保持可复现；
- 任何 protocol 变更必须写进 audit。

运行：

```bash
python scripts/r7/audit_repo_and_skill_ownership.py \
  --repo /data/ppnm/expolicy \
  --r2 outputs/r2_executable_skill \
  --r5 outputs/r5_acquisition_skill \
  --r5-3 outputs/r5_3_external_generalization \
  --r5-4 outputs/r5_4_rasa \
  --r6 outputs/r6_ava \
  --out outputs/r7_policy_generalization/audit

python scripts/r7/audit_r5_r6_cases.py \
  --r5-3 outputs/r5_3_external_generalization \
  --r5-4 outputs/r5_4_rasa \
  --r6 outputs/r6_ava \
  --per-condition-cases 30 \
  --out outputs/r7_policy_generalization/audit/cases

pytest -q tests/r7
```

---

## 3.3 Skill Ownership Audit 必须回答的 8 个问题

输出 `SKILL_OWNERSHIP_AUDIT.md`：

1. skill 激活后，最终 answer 是谁生成的？
   - base model；
   - model + procedural intervention；
   - 还是 skill executor 直接输出？

2. R2 / R5.3 skill 中有多少比例的 episode：
   - `model_tokens_after_skill == 0`；
   - final answer 完全来自 deterministic span extraction；
   - skill 实际上替代了 policy model。

3. 相同 skill + 相同检索结果换不同 backbone 时，final answer 是否仍完全相同？

4. skill 是否看到了：
   - gold answer；
   - gold document id；
   - task id；
   - dataset name；
   - capability oracle label；
   - future split metadata。

5. text-router 的输入是否只有 deployable observable text？

6. CountOnly 激活以后，experience **内容**是否还影响任何 downstream state？

7. R6 的 8-mask lookup 是否可以完全预测最终 score？

8. 若答案是“skill 基本替代 model”，必须写：

```text
OLD_SKILL_BACKEND = REPLACEMENT_EXECUTOR
CROSS_MODEL_GENERALIZATION_WITH_OLD_BACKEND = INVALID_AS_MODEL_EVIDENCE
```

此时旧 skill backend 只能作为 diagnostic baseline，不允许拿它跑一大堆模型后声称 cross-model generalization。

---

## 3.4 Case Audit 必须人工可读地抽样

至少输出：

```text
R7_CASEBOOK.md
```

每类至少 20–30 个：

- R5.3 frontier 错误 C1 集中案例；
- R5.4 C3 UNRESOLVED / 偶然 repair 案例；
- R6 AVA seed0 长时间停在 C1 的案例；
- C2 skill 成功案例；
- skill 激活后仍失败的案例；
- passive / Ace 成功激活而 frontier 未激活的案例。

每个 case 记录：

```yaml
question:
dataset:
predicted_capability:
base_answer:
base_correct:
acquisition_reason:
retrieved_docs:
skill_before:
skill_after:
skill_interventions:
model_answer_after_skill:
correct_after:
what_changed:
failure_origin:
```

重点判断失败到底属于：

- acquisition allocation 错；
- router 错；
- retrieval 错；
- skill 本身不可修；
- skill 修好了搜索，但 final synthesis 错；
- skill executor 直接替代 model；
- task 本来就超出当前 skill bank。

---

# 4. R7-B：所有本地 policy model 的兼容性预检

当前模型目录：

```text
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen3-1.7B
Qwen3-4B-Instruct-2507
Qwen3-4B-Thinking-2507
Qwen3-30B-A3B-Thinking-2507
Phi-4-reasoning
gemma-4-E4B-it
gpt-oss-20b
harness-1
```

`all-MiniLM-L6-v2 / cross-encoder-ms-marco-MiniLM-L-6-v2 / e5-base-v2` 不进入 policy backbone 表。

先运行：

```bash
python scripts/r7/audit_model_compatibility.py \
  --model-root /data/ppnm/models \
  --models \
    Qwen2.5-3B-Instruct \
    Qwen2.5-7B-Instruct \
    Qwen3-1.7B \
    Qwen3-4B-Instruct-2507 \
    Qwen3-4B-Thinking-2507 \
    Qwen3-30B-A3B-Thinking-2507 \
    Phi-4-reasoning \
    gemma-4-E4B-it \
    gpt-oss-20b \
    harness-1 \
  --real-generation-smoke 8 \
  --out outputs/r7_policy_generalization/audit/model_compatibility
```

必须记录：

- HF architecture；
- tokenizer；
- chat template；
- generation config；
- reasoning/thinking mode；
- vLLM 是否可直接加载；
- dtype；
- 单卡峰值显存；
- 是否需要 TP；
- tool-call formatting 是否兼容当前 agent；
- 8 条真实 ReAct/search smoke 是否能正常闭环。

### 禁止

如果某模型单卡 OOM：

- 优先 tensor parallel；
- 或让该模型独占 2/4 卡；
- **不得静默量化后与其它模型放进同一主表。**

如果必须量化，单独形成 `quantized diagnostic`，不进 primary cross-model table。

---

# 5. R7-C：扩展到标准 7-Search-QA Benchmark Suite

不要只继续 HotpotQA / 2Wiki / MuSiQue。

本轮统一采用 Search-R1 系列工作常用的 7 个 QA benchmark：

### Single-hop

1. Natural Questions (NQ)
2. TriviaQA
3. PopQA

### Multi-hop

4. HotpotQA
5. 2WikiMultiHopQA
6. MuSiQue
7. Bamboogle

检索继续优先复用已经建好的：

```text
Wikipedia 2018
E5-base-v2
21,015,324 docs
```

除非 audit 证明 corpus / preprocessing 与某 benchmark 严重不匹配。

---

## 5.1 构建冻结 split

```bash
python scripts/r7/build_r7_search_suite.py \
  --datasets nq triviaqa popqa hotpotqa 2wiki musique bamboogle \
  --retriever e5 \
  --wiki-index existing \
  --split-seed 20260815 \
  --calibration-per-dataset 64 \
  --acquisition-per-dataset 300 \
  --skill-validation-per-dataset 80 \
  --future-per-dataset 150 \
  --bamboogle-use-full-if-insufficient \
  --out data/r7_search_suite

python scripts/r7/validate_r7_splits.py \
  --root data/r7_search_suite \
  --require-task-id-disjoint \
  --require-question-disjoint \
  --require-answer-overlap-report \
  --write-sha256 \
  --out outputs/r7_policy_generalization/audit/split
```

如果某 dataset 数量不足：

- 不复制；
- 不 replacement sampling；
- 用全部合法样本；
- 明确记录实际 n；
- 不为了对齐 150 而重抽。

---

# 6. R7-D：先换掉退化的 assimilation backend，再重新谈 Experience Policy

这是本轮最关键的 redesign。

## 6.1 Backend 0：旧 CountOnly + replacement skill

保留，但只作为：

```text
diagnostic / historical reproduction baseline
```

用途：

- 复现 R5.3–R6；
- 证明 mask lookup pathology；
- 不能再作为新方法的唯一 backend。

---

## 6.2 Backend 1：Interventional Executable Skill（强制实现）

借鉴 HASP 的思想，但**明确不把 executable skill 本身作为 novelty**。

新的 skill 不允许直接给 final answer。它只能：

- 修改 query；
- 强制追加一次 bridge search；
- 调整 open 顺序；
- 保存 evidence slots；
- 检查 comparison operands 是否齐全；
- 检查 long-chain unresolved entities；
- 在失败状态插入 corrective context；
- 最后必须把控制权交还给 policy model，由 policy model 生成最终答案。

三个初始 skill：

```text
S_bridge:
  detect partial bridge evidence
  -> extract bridge entity
  -> search second-hop evidence
  -> return evidence to model

S_compare:
  identify comparison operands
  -> retrieve evidence for both
  -> align attribute/time/unit
  -> return structured evidence to model

S_long_chain:
  maintain unresolved evidence slots
  -> iteratively search next missing relation
  -> stop on evidence sufficiency / budget
  -> return evidence graph to model
```

禁止：

- 写死 dataset pattern；
- 写死 gold span；
- 写死具体 entity；
- 直接从 document 抽 final answer 作为 agent output；
- dataset name 参与 dispatch。

### 验收

每个 skill 至少在 3 个模型上分别验证：

```text
base
old replacement skill
new interventional skill
```

要求：

1. `model_tokens_after_skill > 0`；
2. skill 只改变 action/evidence state；
3. final output 来自当前 backbone；
4. 至少 2/3 capability 有 `ΔEM > 0`；
5. retention 不允许因为 skill 被全局强制调用而崩；
6. 每个 skill 的 trigger 必须 deployable。

---

## 6.3 Backend 2：Content-Sensitive Skill Assimilator

不要再自己发明第三个 count heuristic。

优先借鉴 / 复现 **SkillOpt-style** 的强 content-dependent skill optimization：

```text
acquired trajectories
→ analyze success/failure
→ bounded add/delete/replace edits to skill artifact
→ held-out validation
→ accept only improving update
```

这里 SkillOpt 只是 strong assimilation backend / baseline，不是本研究 novelty。

关键公平性：

- `passive / Ace / frontier / future learned policy` 必须共用**完全相同**的 skill optimizer；
- 唯一变化只能是被送进 optimizer 的 experience；
- validation split 固定；
- optimizer model 固定；
- edit budget 固定；
- skill initial state 固定；
- 不允许为某 acquisition method 单独调 prompt。

若官方实现能直接接入，优先 pin 官方代码 commit；若不能直接接入，则按论文算法实现并写 `SKILLOPT_PORT_AUDIT.md`，逐项核对：

```text
optimizer input
edit operators
acceptance gate
validation protocol
edit budget
rejected-edit buffer
epoch/meta update
deployment artifact
```

### Content-Sensitivity Gate

先不跑 acquisition policy，先回答：

> 不同 experience batch 是否真的会产生不同 downstream capability？

对同一个 model / capability：

- 构造 12 个不同 experience batches；
- 用相同初始化 skill；
- 分别 optimize；
- 在同一 held-out future 上测。

至少满足一个：

```text
std(future_gain across batches) >= 0.03
```

或：

```text
best_batch - worst_batch >= 0.08
```

并且至少 2 个 capability 满足。

否则：

```text
CONTENT_SENSITIVE_SKILL_BACKEND = FAIL
```

此时不要继续 acquisition-policy；直接换 Backend 3。

---

## 6.4 Backend 3：Parametric Experience Assimilation

如果 SkillOpt-style backend 仍然没有足够 content sensitivity，下一步不是调 acquisition，而是直接借鉴：

- OEL；
- OPCD；
- OPID / skill-conditioned OPD；
- 现有项目里已经安全实现过的 LoRA/OPD infrastructure。

目的不是宣称新的 distillation，而是建立第二种真正不同的 experience carrier：

```text
external executable skill
vs
parametric internalization
```

然后构造：

```text
same experience batch
→ skill optimization
→ OPCD/OEL-style internalization
→ held-out near/far/retention
```

只有出现真实 carrier heterogeneity 后，才允许重新开启 `carrier policy`。

---

# 7. R7-E：Cross-Model Skill / Assimilation Transfer Matrix

不要一上来对 9 个模型跑完整 acquisition。

先做低成本上界矩阵。

## 7.1 第一阶段模型

Primary：

```text
Qwen2.5-3B-Instruct          # anchor
Qwen2.5-7B-Instruct          # same family scaling
Qwen3-4B-Instruct-2507       # family shift
Qwen3-4B-Thinking-2507       # reasoning-mode shift
Phi-4-reasoning              # architecture/family shift
gpt-oss-20b                  # larger open model
```

Secondary：

```text
Qwen3-1.7B
gemma-4-E4B-it
Qwen3-30B-A3B-Thinking-2507
```

Secondary 只有 preflight 通过且资源允许后进入。

---

## 7.2 先跑 Static Headroom Matrix

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python scripts/r7/evaluate_skill_backend_matrix.py \
  --models-config configs/r7_models.yaml \
  --suite data/r7_search_suite \
  --conditions \
    base \
    old_replacement_skill \
    interventional_skill \
    content_optimized_skill \
  --per-dataset 80 \
  --seeds 0 1 2 \
  --scheduler auto \
  --out outputs/r7_policy_generalization/model_skill_matrix
```

输出至少：

```text
MODEL_SKILL_MATRIX.csv
MODEL_CAPABILITY_MATRIX.csv
MODEL_DATASET_MATRIX.csv
MODEL_COST_MATRIX.csv
MODEL_SKILL_HEADROOM.md
```

每个 `(model, dataset, capability)` 记录：

- base EM/F1；
- retrieval recall；
- old skill gain；
- interventional skill gain；
- content-optimized skill gain；
- retention；
- tool calls；
- tokens；
- wall time。

---

## 7.3 Policy Headroom 指标

对每个 model：

```text
H_skill = best_validated_skill - base
H_policy = oracle_experience_batch - passive_experience_batch
H_carrier = oracle_carrier - best_fixed_carrier
```

R7 后续只在有真实 headroom 的模型上做 full experiment。

注意：这不是“筛结果”。

必须：

- 先报告所有模型的 headroom；
- 再根据跑前固定阈值决定是否值得投入更多算力。

建议 gate：

```text
H_skill >= 0.05
```

且至少存在一种：

```text
H_policy >= 0.03
or
H_carrier >= 0.03
```

---

# 8. R7-F：在 Strong Assimilator 后重新比较 Acquisition

只有 Backend 1/2 至少一个通过验证后才运行。

比较：

```text
clean
passive
Ace-Skill prioritized
legacy frontier
coverage-balanced
exact post-assimilation oracle
```

其中：

- `legacy frontier`：只用于判断旧 Helion idea 是否能跨模型复活；
- `coverage-balanced`：简单强 baseline，防止再次出现 “均匀覆盖就赢了”；
- `exact post-assimilation oracle`：只用于测 policy headroom，不作为 deployable 方法。

固定：

```text
budget = 48
checkpoint = 0/8/16/24/32/40/48
same candidate pool
same assimilator
same skill initialization
same validation split
same final future split
```

运行：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python scripts/r7/orchestrate_r7.py \
  --stage acquisition \
  --models \
    Qwen2.5-3B-Instruct \
    Qwen2.5-7B-Instruct \
    Qwen3-4B-Instruct-2507 \
    Qwen3-4B-Thinking-2507 \
  --suite data/r7_search_suite \
  --assimilator content_skill \
  --methods passive ace_skill legacy_frontier coverage exact_utility \
  --budget 48 \
  --checkpoints 0 8 16 24 32 40 48 \
  --seeds 0 1 2 \
  --scheduler auto \
  --out outputs/r7_policy_generalization/acquisition
```

如果资源足够且上述 4 模型至少 2 个出现 policy headroom，再扩：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python scripts/r7/orchestrate_r7.py \
  --stage acquisition \
  --models Phi-4-reasoning gpt-oss-20b \
  --suite data/r7_search_suite \
  --assimilator content_skill \
  --methods passive ace_skill legacy_frontier coverage exact_utility \
  --budget 48 \
  --checkpoints 0 8 16 24 32 40 48 \
  --seeds 0 1 2 \
  --scheduler auto \
  --resume \
  --out outputs/r7_policy_generalization/acquisition
```

---

# 9. R7-F 的核心 Go / Redesign / Discard 判据

## Case A — 旧 frontier 真正跨模型成立

满足：

```text
legacy_frontier > max(passive, Ace, coverage)
```

并且：

- 至少 3/4 primary models；
- 7-benchmark macro 至少 +0.03；
- 每个模型 ≥2/3 seeds 赢；
- 不是单一 dataset 贡献全部增益；
- content-sensitive assimilator 下仍成立。

则：

```text
OLD_FRONTIER_GENERALIZES = TRUE
```

可进入 learned policy，但论文不能只写 “frontier acquisition”。

---

## Case B — 旧 frontier 失败，但 exact experience utility 有大 headroom

例如：

```text
legacy_frontier ≈ passive
exact_utility - best_deployable >= 0.05
```

且至少 3 个模型成立。

这是最理想的 pivot：

> heuristic 失败，但“experience policy problem”真实存在。

此时进入 R8 learned Experience Policy。

---

## Case C — experience batch 没有 headroom，但 carrier 有 headroom

满足：

```text
H_policy < 0.03
H_carrier >= 0.05
```

则：

- discard acquisition-centric ExPolicy；
- 转为 **Experience Lifecycle Policy / Cross-Substrate Assimilation Policy**；
- primary action 变成：
  - external skill；
  - memory；
  - parameter internalization；
  - discard / defer；
- 必须直接比较 COVE、SkillOS、OPD-Evolver 类 baseline。

---

## Case D — strong assimilator 后仍没有 policy/carrier headroom

满足：

```text
H_policy < 0.03
H_carrier < 0.03
```

且多个模型 / benchmark 都如此。

则：

```text
DISCARD_EXPOLICY_METHOD = TRUE
```

不要再缩 claim。

下一篇直接 pivot 到：

> **Experience → New Capability Construction / Internalization**

也就是研究：

- 如何从 failure trajectory 构造 genuinely new executable skill；
- 如何验证 skill 是否真正增加 capability boundary；
- 何时从 external skill 内化到 weights；
- 如何避免 progressive capability collapse；
- 如何让 skill creation / internalization 跨模型迁移。

此时 SkillOpt / SkillOS / HASP / OPCD / OEL / OPID 是主要 baseline，而不是 acquisition paper。

---

# 10. R7-G：只有 Oracle Headroom 存在时才训练 Learned Experience Policy

**严禁在没有 oracle headroom 时训练 policy。**

如果 Case B 成立，下一步定义真正的 Experience Policy。

## 10.1 State

至少包括：

```yaml
executor_profile:
  model_family
  base_capability_vector
  tool_success_rate
  retrieval_recall
  reasoning_failure_rate

experience_state:
  cap_counts
  recent_failures
  novelty
  diversity
  redundancy
  observed_transfer
  observed_interference

assimilator_state:
  skill_validation_gain
  skill_content_sensitivity
  parametric_gain
  retention_delta
  previous_update_success

budget_state:
  remaining_experience_budget
  remaining_compute_budget
  current_round
```

禁止把：

- dataset name；
- task id；
- gold label；
- future answer

作为 policy feature。

---

## 10.2 Action

不要再把 action 限制成“选哪个 capability”。

R8 的最小 action 应是：

```text
(acquire_target, assimilation_action)
```

其中：

```text
acquire_target:
  candidate batch / task family / stream slice

assimilation_action:
  external_skill
  parametric_internalization
  memory
  defer
  discard
```

如果后续证据支持，再加：

```text
promote
revise
retire
```

---

## 10.3 Reward

不要使用 immediate reward。

主 reward：

```text
R_t =
  Δ held-out future capability
  - λ_forget * forgetting
  - λ_token * token cost
  - λ_train * training cost
```

必须是 delayed、post-assimilation 的 capability reward。

---

## 10.4 Strongest Baseline Composition

新 learned policy 必须至少对比：

```text
Ace-Skill acquisition
+ SkillOpt-style assimilation
+ ExpWeaver/SEU-style selective use
```

以及：

```text
SkillOS
COVE
OPD-Evolver
```

能直接复现的尽量复现；不能直接接入时做 component-matched baseline。

如果 learned policy 只赢 passive，而打不过 strongest composition，不构成论文。

---

# 11. R7-H：增加 PATH-Bench，而不是只继续 QA

这是我建议并行准备的第二条 benchmark 线。

原因：

- PATH-Bench 专门研究 path-dependent experience；
- 有 positive / interfering history；
- 有 AP / FWT / BWT / forgetting；
- 本身就带 Selective Experience Use (SEU) baseline；
- 它能直接检验“policy 是否真的学会管理 experience 的长期价值”，比固定 QA pool 更适合 Experience Policy。

但它不是 Search-Agent benchmark，因此作为 **Experience Policy generality test**，不要混进 Search-R1 主表。

实施：

```text
R7-Search:
  NQ / TriviaQA / PopQA / HotpotQA / 2Wiki / MuSiQue / Bamboogle

R7-Lifelong:
  PATH-Bench BigCodeBench track
  PATH-Bench WildToolBench track
```

必须先使用 PATH-Bench 官方 release / official protocol；没有 release 时不得自行 mock。

重点比较：

```text
No Experience
Vanilla Memory/Skill
PATH-Bench SEU
Ace-Skill-style acquisition
SkillOS-style curation
ExPolicy-v2
```

最重要的是测试 cross-model，因为 PATH-Bench 原论文明确留下了 diverse backbone 扩展空间。

优先模型：

```text
Qwen2.5-7B-Instruct
Qwen3-4B-Instruct-2507
gpt-oss-20b
```

如果 Experience Policy 在 QA 有效、在 PATH-Bench 也能同时提升：

```text
AP ↑
FWT ↑
FGT ↓
```

且不是只靠更大 context，这会比继续做 Helion 强得多。

---

# 12. R7-I：可选 Long-Horizon Search Stress Test

如果 7-QA suite 出现明确阳性，再接 **LoHoSearch**。

不要在 R7 前半段就跑。

目标不是刷分，而是问：

> experience policy 是否在真正长 horizon / 大 search space 下更有价值？

重点指标：

```text
accuracy
search turns
evidence coverage
stuck/recovery rate
context growth
experience intervention count
```

如果只有短 QA 有增益，LoHoSearch 完全不成立，应视为 method generality failure，而不是把 claim 缩成 short multi-hop QA。

---

# 13. GPU 调度建议

先运行 compatibility probe，根据真实显存重新分配。

默认四卡：

### GPU 4

```text
Qwen2.5-3B-Instruct
Qwen3-1.7B
Qwen3-4B-Instruct-2507
```

### GPU 5

```text
Qwen2.5-7B-Instruct
Qwen3-4B-Thinking-2507
```

### GPU 6

```text
Phi-4-reasoning
gemma-4-E4B-it
```

### GPU 7

```text
gpt-oss-20b
```

`Qwen3-30B-A3B-Thinking-2507`：

- preflight 后决定；
- 若单卡安全，独占一张；
- 若需要 TP=2，则在小模型阶段结束后占两卡；
- 不与其它任务硬抢显存。

推荐运行：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python scripts/r7/orchestrate_r7.py \
  --stage all-prepolicy \
  --config configs/r7_policy_generalization.yaml \
  --scheduler auto \
  --resume \
  --watchdog \
  --out outputs/r7_policy_generalization
```

watchdog 必须：

- 检查 process 存活；
- 检查 GPU utilization；
- 检查最近 artifact mtime；
- stall 时只重启当前 `(model, condition, seed)`；
- 不删除完整结果；
- 不覆盖已通过 hash 校验的 run。

---

# 14. 结果文件要求

最终必须生成：

```text
outputs/r7_policy_generalization/
├── audit/
│   ├── GIT_HEAD.txt
│   ├── GIT_STATUS.txt
│   ├── NOVELTY_COLLISION.md
│   ├── SKILL_OWNERSHIP_AUDIT.md
│   ├── MODEL_COMPATIBILITY.md
│   ├── SPLIT_AUDIT.md
│   ├── R7_CASEBOOK.md
│   └── PROTOCOL_AUDIT.md
├── model_skill_matrix/
│   ├── MODEL_SKILL_MATRIX.csv
│   ├── MODEL_CAPABILITY_MATRIX.csv
│   ├── MODEL_DATASET_MATRIX.csv
│   └── MODEL_SKILL_HEADROOM.md
├── assimilation/
│   ├── CONTENT_SENSITIVITY.csv
│   ├── SKILLOPT_PORT_AUDIT.md
│   └── ASSIMILATOR_DECISION.json
├── acquisition/
│   ├── R7_FINAL_METRICS.csv
│   ├── R7_LEARNING_CURVES.csv
│   ├── R7_PER_MODEL.csv
│   ├── R7_PER_DATASET.csv
│   ├── R7_PER_CAPABILITY.csv
│   ├── R7_POLICY_HEADROOM.csv
│   └── R7_COST.csv
├── cases/
├── logs/
├── R7_MASTER_SUMMARY.md
└── R7_DECISION.json
```

---

# 15. `R7_DECISION.json` 必须自动输出的字段

```json
{
  "old_skill_backend_is_replacement_executor": null,
  "interventional_skill_valid": null,
  "content_sensitive_assimilator_valid": null,
  "old_frontier_cross_model_generalizes": null,
  "oracle_experience_policy_headroom": null,
  "carrier_headroom": null,
  "pathbench_ready": null,
  "recommended_case": "A|B|C|D",
  "next_action": ""
}
```

---

# 16. 本轮真正希望看到的结果

## 最理想结果

不是“frontier 在另一个模型上也多赢 2 题”。

而是：

```text
1. 旧 frontier 不一定稳定；
2. 但在 content-sensitive assimilation 下，不同 experience batch 的 downstream value 出现稳定异质性；
3. exact oracle 明显超过 passive / Ace / coverage；
4. 该 oracle headroom 在多个 backbone 上同时存在；
5. carrier utility 随 model / task / experience 改变；
6. 因而存在一个真实的 sequential decision problem；
7. learned Experience Policy 再从这个 headroom 中回收大部分 gain。
```

这时论文的中心问题就能变成：

> **Experience learning 已经从“如何存 memory / 如何写 skill / 如何 internalize”发展到下一阶段：agent 必须学习一个跨阶段 policy，决定自己应该经历什么、把经历变成什么、以及什么时候让它进入长期能力。**

这比继续讲某个 `frontier_score` 强得多。

---

# 17. 最坏结果也要有明确科研价值

如果：

- 强 content-sensitive assimilator 仍然让不同 experience batch 几乎等价；
- exact post-assimilation oracle 在多个模型 / benchmark 上都没有 >0.03 的 policy headroom；
- carrier 也始终存在 global winner；
- PATH-Bench 上 learned policy 也打不过 SEU / SkillOS / strongest composition；

那么必须正式写：

```text
Experience-policy headroom is empirically absent under current agent-learning substrates.
```

并 **discard ExPolicy method**。

下一阶段不要缩 claim，而是直接换成更本质的：

```text
Experience → Capability Construction
```

即研究“什么样的 experience transformation 才能真的创造新能力”，而不是“在没有可区分学习价值的 experience 上继续学习 selector”。

---

# 18. 给 Cursor Agent 的最终执行要求

请按以下顺序执行，不要跳步：

```text
Phase 0  Repository + novelty + skill ownership audit
Phase 1  Model compatibility probe
Phase 2  7-benchmark frozen split + retrieval audit
Phase 3  Interventional skill backend implementation + validation
Phase 4  Content-sensitive assimilator integration
Phase 5  Cross-model static headroom matrix
Phase 6  Acquisition comparison under the validated assimilator
Phase 7  Exact oracle headroom analysis
Phase 8  Automatic Case A/B/C/D decision
Phase 9  Only if Case B/C supports it: implement learned Experience Policy
Phase 10 Prepare PATH-Bench cross-model track
```

每个 Phase 必须：

- 有完整 artifact；
- 有 deterministic resume；
- 有 protocol test；
- 有 failure log；
- 不允许用 dummy / mock 代替未完成模块；
- 不允许把 failed run 当 completed；
- 不允许为了过 gate 修改已经冻结的阈值；
- 不允许删除 negative results；
- 所有结果追加到新的 R7 record，不覆盖 R5/R6。

最重要的是：

> **这轮不是为了证明 ExPolicy 一定对，而是为了给“Experience Policy 是 experience-learning 下一阶段”这个更大的命题一次真正公平、强 baseline、跨模型、跨 benchmark 的检验。**
>
> 如果 old method 不行，就 redesign；如果 framing 被 prior art 占据，就换更强的 framing；如果真实 optimization headroom 不存在，就 discard method。不要用缩小 claim 的方式保项目。
