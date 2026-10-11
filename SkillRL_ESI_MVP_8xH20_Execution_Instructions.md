# SkillRL-ESI MVP：八卡 H20 端到端执行指令

> 本文件交给 8×H20 机器上的 Coding/Research Agent 直接执行。  
> 不要停留在方案讨论，不要先重写一套新的 Agent/Experience infrastructure。
>
> **本轮总原则：先把官方 SkillRL Search backbone 原样搭通并验收，再只在“experience → skill → execution”这一层实现我们的 ESI 改造。**
>
> 最终首要目标只有一个：
>
> **在 SkillRL 已验证可工作的 Search 基础设施上，把真实训练侧 experience 转换成一个新的 executable skill，并让冻结的 base actor 在一个未见的公开 Search 数据集 split 上优于 learning 前。**
>
> 本轮不追求完整论文，不追求 Active Acquisition，不追求 RL 最终性能，不追求多 benchmark。

---

# 0. 为什么这次必须先搭 SkillRL，而不是继续扩自研 ExPolicy infra

此前 ExPolicy MVP 已经把核心问题收缩为：

```text
experience
    ↓
assimilation operator
    ↓
future capability
```

旧实验中：

- episodic / semantic / LoRA 没有稳定产生 transferable capability；
- executable procedure 能显著改善目标能力；
- 但此前的正结果主要来自 count/evidence gating 解锁 frozen primitive；
- 因而 acquisition policy 最终容易退化为“决定先打开哪个已有 skill”。

所以本轮不再同时创新：

```text
environment
rollout
trajectory logging
distributed execution
skill memory
evaluation
training
experience assimilation
```

而是固定一个成熟开源框架，减少 infra ambiguity。

本轮采用：

**SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning**

官方仓库：

```text
https://github.com/aiming-lab/SkillRL
```

论文：

```text
https://arxiv.org/abs/2602.08234
```

截至 2026-08-18，官方仓库明确包含：

```text
agent_system/
memory_data/
skill_generation/
examples/
tests/
verl/
```

并原生提供 Search 环境与：

```text
examples/grpo_trainer/run_search_skills.sh
skill_generation/search.py
```

Search skill generation 代码中明确覆盖：

```text
nq
popqa
triviaqa
hotpotqa
2wikimultihopqa
musique
bamboogle
```

SkillRL 原生 skill bank 是 textual skill，典型字段为：

```text
skill_id
title
principle
when_to_apply
```

这正是本轮要改造的位置：

```text
SkillRL:
experience → textual skill → prompt injection

ESI:
experience → typed executable SkillSpec → compiler → runtime execution
```

---

# 1. 本轮科学问题

本轮只回答：

> **在成熟 SkillRL Search framework 上，真实 experience 内容能否被转换成新的 executable behavior，从而使冻结 base actor 在未见任务上变强？**

核心链路：

```text
SkillRL Search environment
        ↓
base actor rollout
        ↓
native trajectory / memory records
        ↓
ExperiencePacket
        ↓
ESI Skill Inducer
        ↓
Typed SkillSpec
        ↓
safe compiler
        ↓
Executable Skill Runtime
        ↓
SkillRL SearchEnv
        ↓
held-out validation
        ↓
freeze
        ↓
held-out future test
```

必须避免重新退化为：

```text
count(capability) >= K
        ↓
unlock pre-installed procedure
```

本轮 skill 必须由本轮 experience 内容生成。

---

# 2. 两级 Gate：严格执行

整个项目分成两个大 Gate。

---

## Gate 0 — SkillRL Foundation Acceptance

在写 ESI 代码之前，必须证明 upstream SkillRL Search infra 可工作。

至少验收：

```text
G0.1 repo clone / import PASS
G0.2 Search environment dependency PASS
G0.3 SearchEnv reset/step PASS
G0.4 local actor inference PASS
G0.5 native no-skill Search rollout PASS
G0.6 native textual SkillBank load/injection PASS
G0.7 evaluation/result logging PASS
```

推荐再做：

```text
G0.8 tiny GRPO/verl training smoke PASS or WARN
```

如果 G0.1–G0.7 中任何核心项未过：

```text
禁止开始 ESI method implementation
```

先修 infra。

---

## Gate 1 — ESI Pipeline Acceptance

Gate 0 PASS 后才执行：

```text
experience collection
→ ExperiencePacket
→ SkillSpec induction
→ executable compiler
→ candidate validation
→ freeze
→ final test
```

最低成功：

```text
held-out test:
ESI executable skill EM > frozen base EM

且 absolute gain >= +0.03
```

推荐：

```text
3 seeds
mean ΔEM >= +0.03
>= 2/3 seeds win
```

---

# 3. 不要修改现有 ExPolicy repo 作为主代码库

旧 repo：

```text
/data/ppnm/expolicy
```

只作为：

```text
historical protocol reference
old split ID reference
old result reference
```

不要继续把 ESI 主实现塞入 `/data/ppnm/expolicy`。

新 repo 路径：

```text
/data/ppnm/skillrl_esi
```

如果路径已存在：

```text
先检查 git remote / git status
不得直接删除
```

---

# 4. Clone、上游锁定与 branch

执行：

```bash
cd /data/ppnm

if [ ! -d skillrl_esi/.git ]; then
  git clone https://github.com/aiming-lab/SkillRL.git skillrl_esi
fi

cd /data/ppnm/skillrl_esi

git remote -v
git status
git rev-parse HEAD
git log -1 --oneline
```

确保有 upstream：

```bash
git remote get-url upstream >/dev/null 2>&1 || \
git remote add upstream https://github.com/aiming-lab/SkillRL.git
```

建立工作分支：

```bash
git checkout -b esi-mvp 2>/dev/null || git checkout esi-mvp
```

不要修改 `main`。

立即记录：

```text
outputs/esi_mvp/UPSTREAM_LOCK.json
outputs/esi_mvp/ENVIRONMENT_AUDIT.md
```

`UPSTREAM_LOCK.json` 至少：

```json
{
  "repo": "https://github.com/aiming-lab/SkillRL",
  "upstream_commit": "...",
  "working_branch": "esi-mvp",
  "date": "2026-08-18"
}
```

---

# 5. 环境必须隔离

不要污染已有 ExPolicy / QEMU / 其它任务环境。

优先创建：

```bash
conda create -n skillrl-esi python=3.10 -y
conda activate skillrl-esi
```

如果服务器统一使用其他环境管理器，可改用等价隔离环境，但必须记录。

官方 README 当前安装路径为：

```bash
pip install -r requirements.txt
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
pip install openai
```

Search 额外安装：

```bash
cd agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
cd /data/ppnm/skillrl_esi
```

注意：

- 不要盲目覆盖服务器现有 CUDA driver；
- 不要安装另一套系统 CUDA；
- 若官方固定的 `vllm/flash-attn/torch` 与本机 CUDA 不兼容，优先解决 Python package compatibility；
- 可以调整 infra version 以适配 H20，但必须记录 diff；
- 不允许因为依赖问题改科学逻辑。

记录：

```bash
python -V
pip -V
nvidia-smi
nvcc --version || true
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
pip freeze > outputs/esi_mvp/pip_freeze.txt
```

---

# 6. Gate 0.1：Repo / Import Acceptance

运行：

```bash
python - <<'PY'
import agent_system
import verl
print("agent_system OK")
print("verl OK")
PY
```

检查关键路径真实存在：

```bash
test -f agent_system/environments/env_package/search/envs.py
test -f skill_generation/search.py
test -f examples/grpo_trainer/run_search_skills.sh
```

检查：

```bash
git diff --stat upstream/main...HEAD
```

此时原则上应该只有我们新增的 audit/output glue，没有 ESI method。

写：

```text
outputs/esi_mvp/gate0/G0_1_REPO_IMPORT.md
```

---

# 7. Gate 0.2：理解 SkillRL Search action protocol，不准猜

先读：

```text
agent_system/environments/env_package/search/envs.py
agent_system/environments/env_package/search/third_party/
examples/grpo_trainer/run_search_skills.sh
examples/search/
```

使用：

```bash
rg -n "SearchEnv|search_url|ground_truth|max_turns|postprocessed_action|reward" \
  agent_system examples
```

需要在：

```text
outputs/esi_mvp/gate0/SEARCH_PROTOCOL.md
```

记录：

1. question 如何进入 `SearchEnv.reset`;
2. ground truth 在哪里使用；
3. agent action 的合法文本格式；
4. search action 格式；
5. answer action 格式；
6. reward 如何计算；
7. termination 如何触发；
8. search backend URL 如何配置；
9. dataset records 的字段；
10. trajectory 在哪里记录。

**不要根据旧 ExPolicy 的 search/open API 猜 SkillRL 的 action grammar。**

后面的 executable skill 必须通过 SkillRL SearchEnv 的合法 action interface 执行。

---

# 8. Gate 0.3：Search backend / SearchEnv smoke

SkillRL 当前 `SearchMultiProcessEnv` 会把：

```text
env_config.search.search_url
```

分配给底层 SearchEnv。

因此先定位 upstream Search server 的启动方式：

```bash
rg -n "search_url|retriever|server|serve|search server" \
  examples agent_system
```

按 upstream 自己的方式启动。

不要优先接旧 `/data/ppnm/expolicy` 的 Wikipedia+E5 server。

第一阶段必须：

```text
SkillRL SearchEnv + SkillRL-compatible search backend
```

先形成闭环。

只有 upstream search backend 明确缺失/不可运行时，才允许写一个**薄 adapter**把现有 Wikipedia+E5 服务暴露成 SkillRL 所要求的同一 HTTP schema。

若写 adapter：

```text
adapter 只做 request/response schema conversion
不得改变 ranking
不得插入 gold docs
不得调用 old frozen skill
```

创建：

```text
outputs/esi_mvp/gate0/SEARCH_BACKEND_AUDIT.md
```

Smoke 必须实际完成：

```text
reset one real Search task
→ send one legal search action
→ receive real observation
→ send more actions if needed
→ final answer
→ receive reward
```

保存完整 JSON trace。

---

# 9. Gate 0.4：Actor model

本轮正式 actor 固定为：

```text
/data/ppnm/models/Qwen2.5-3B-Instruct
```

不要用 SkillRL 官方 RL checkpoint 作为 ESI final actor。

原因：

```text
我们的目标是比较 learning 前后同一个 frozen actor，
把提升归因于 external executable assimilation。
```

检查：

```bash
test -d /data/ppnm/models/Qwen2.5-3B-Instruct
```

用 vLLM 启动/加载最小 inference smoke。

需要验证：

```text
prompt in
model output
GPU memory stable
batch inference works
```

写：

```text
outputs/esi_mvp/gate0/ACTOR_AUDIT.md
```

---

# 10. Gate 0.5：Native no-skill rollout

不要立即用 ESI。

先基于 SkillRL 的：

```text
agent_system
multi_turn_rollout
SearchEnv
reward/eval
```

建立一个 evaluation-only runner：

```text
examples/esi/run_search_eval.py
```

这个 runner 不是重新造 Agent framework。

它只能：

```text
load SkillRL Search dataset
instantiate SkillRL SearchEnv
call existing actor/rollout utility
record official reward
record trajectory
```

条件：

```text
skills disabled
no old ExPolicy memory
no old frozen primitive
temperature = 0
```

至少跑：

```text
16 real Search tasks
```

结果必须包含：

```text
task_id
dataset/data_source
question
actions
observations
final answer
reward
won
tool/search count
tokens
```

---

# 11. Gate 0.6：Native textual SkillRL SkillBank smoke

SkillRL 的 native SkillBank 是我们正式 baseline。

使用 upstream 自带的 skillbank / memory file。

不要先生成新 ESI skill。

先确认：

```text
native textual skill
→ SkillRL retrieval
→ prompt injection
→ Search rollout
```

能够正常运行。

至少同一批 16 tasks 上跑：

```text
native_no_skill
native_textual_skill
```

这里不要求 textual skill 一定提升。

Gate 0 只要求：

```text
skillbank load 正常
retrieval 正常
prompt injection 正常
episode 正常结束
metrics 正常生成
```

保存：

```text
outputs/esi_mvp/gate0/native_textual_skill/
```

---

# 12. Gate 0.7：Trajectory / Memory Data smoke

SkillRL 的核心价值之一是从 rollout 得到 experience/memory。

检查官方：

```text
memory_data/
memory_data/prompt/prompt.txt
skill_generation/search.py
```

确认现有 memory schema。

从 Gate 0 rollout 生成/转换至少：

```text
8 successful experiences
8 failed experiences
```

如果 16 tasks 不足：

```text
继续 rollout，直到得到至少可用的 success/failure 样本；
不要伪造 outcome。
```

保存为：

```text
outputs/esi_mvp/gate0/native_memory_data.json
```

再验证：

```text
skill_generation/search.py
```

能够读取该 schema。

注意：

官方 `skill_generation/search.py` 当前依赖 Azure OpenAI/o3 client。

Gate 0 不要求实际调用外部 API。

需要验证：

```text
input schema compatible
categorization compatible
pattern extraction compatible
```

如果服务器已有合法 Azure/OpenAI credential，可额外做 native textual skill generation smoke。

如果没有 credential：

```text
记为 EXTERNAL_API_NOT_CONFIGURED
不作为 Gate 0 核心失败
```

本轮 ESI inducer 将使用本地模型，不依赖外部 API。

---

# 13. Gate 0.8：Tiny GRPO / verl training smoke（推荐）

目的：

```text
验证未来要接回 SkillRL recursive RL 时，
verl / Ray / multi-GPU / checkpoint pipeline 不是坏的。
```

不要跑正式 RL。

复制：

```text
examples/grpo_trainer/run_search_skills.sh
```

新建：

```text
examples/esi/run_search_skills_smoke.sh
```

只改：

```text
数据规模
batch
rollout 数
training steps
output path
GPU 数
```

不改 algorithm。

目标：

```text
1–2 optimizer/update steps 能走完
能写 checkpoint
无 hang
```

若本地 3B actor 与脚本接口兼容，使用本地 actor。

若 upstream 明确要求其 SFT checkpoint，可：

1. 先尝试 local actor；
2. 若不兼容且官方 checkpoint 可获得，允许用官方 Search SFT checkpoint仅做 infra smoke；
3. 记录 model 与原因。

G0.8 失败不阻塞 ESI MVP，只要 G0.1–G0.7 已通过。

状态：

```text
PASS / WARN
```

---

# 14. Gate 0 总判

生成：

```text
outputs/esi_mvp/gate0/SKILLRL_FOUNDATION_ACCEPTANCE.md
outputs/esi_mvp/gate0/GATE0_DECISION.json
```

核心项：

| Gate | Requirement |
|---|---|
| G0.1 | repo/import |
| G0.2 | protocol understood |
| G0.3 | real SearchEnv step |
| G0.4 | frozen actor inference |
| G0.5 | no-skill rollout |
| G0.6 | native textual SkillBank rollout |
| G0.7 | trajectory/memory schema |
| G0.8 | tiny GRPO smoke, optional |

只有：

```text
G0.1–G0.7 PASS
```

才可进入 ESI。

---

# 15. Gate 1 数据集：优先 2WikiMultiHopQA

ESI MVP 首先只做：

```text
2wikimultihopqa
```

目标优先：

```text
comparison-style multi-hop
```

因为 SkillRL `skill_generation/search.py` 已原生把 multi-hop QA 分为包括：

```text
comparison
multi_hop_reasoning
```

两类。

不要再自己发明一套与 SkillRL 不兼容的 Search taxonomy。

优先复用：

```text
SkillRL query-type classification
```

必要时只增加更细的 metadata，不改变 upstream data source。

---

# 16. 新 MVP split

建立：

```text
examples/esi/build_esi_split.py
```

从 SkillRL 实际使用的 2Wiki data source 中切：

```text
train_experience_pool = 240
validation = 80
future_test = 150
negative_control = 80
```

若 comparison 子集不足：

最低：

```text
train >= 120
validation >= 40
future_test >= 100
negative_control >= 40
```

split seed：

```text
20260818
```

`negative_control`：

优先 2Wiki 中非 comparison 的 multi-hop 题。

split 必须：

```text
task-id overlap = 0
exact-question overlap = 0
```

此外，若 `/data/ppnm/expolicy` 中 R5.3/R5.4/R6 的公开 benchmark IDs 能无歧义映射到 SkillRL task IDs：

```text
额外排除旧 future IDs
```

如果无法可靠映射：

```text
不要猜映射
在 audit 中声明无法自动排除
```

输出：

```text
outputs/esi_mvp/split.json
outputs/esi_mvp/SPLIT_AUDIT.md
```

---

# 17. Base conditions

在正式 induction 之前冻结三个 baseline：

```text
B0 = frozen actor, no skill
B1 = frozen actor + native SkillRL textual skillbank
B2 = frozen actor + raw trajectory/memory retrieval（如果 upstream已有直接模式）
```

B2 若 upstream 没有稳定直接 memory retrieval：

```text
可不做
```

本轮真正必须的 baseline：

```text
B0
B1
```

这样最终可以回答：

```text
ESI 是否只是“任何 skill prompt 都会提升”
```

---

# 18. Base eval

使用 Gate 0 已验证的 runner，不写第二套 evaluator。

执行：

```text
validation
future_test
negative_control
```

分别跑：

```text
B0
B1
```

保存：

```text
outputs/esi_mvp/base/no_skill/
outputs/esi_mvp/base/native_textual_skill/
```

`future_test` base 可以预计算，但：

```text
candidate selection
inducer
repair
```

绝对不能读取 future_test metric / answer / error list。

---

# 19. Experience Collection：必须走 SkillRL rollout

新建：

```text
examples/esi/collect_experiences.py
```

但其 rollout 必须调用 Gate 0 已验证的 SkillRL Search pipeline。

禁止：

```text
重新接旧 ExPolicy agent
自己写 requests.get 搜索
直接从 gold supporting docs 构 trajectory
```

在 `train_experience_pool` 上收集。

每条保留 upstream 原始 trajectory，并另外构造 ESI view：

```json
{
  "experience_id": "...",
  "task_id": "...",
  "data_source": "2wikimultihopqa",
  "query_type": "comparison",
  "question": "...",
  "reward": 0.0,
  "won": false,
  "actions": [],
  "observations": [],
  "postprocessed_actions": [],
  "final_answer": "...",
  "gold_answer_train_only": "...",
  "search_count": 0,
  "token_count": 0,
  "termination_reason": "..."
}
```

Gold 只允许：

```text
train-side correctness
failure diagnosis
```

不能进入 runtime hint。

---

# 20. 8 卡 Experience Collection

使用真实 8 卡并行 actor rollout。

建议：

```text
GPU0 shard 0/8
GPU1 shard 1/8
GPU2 shard 2/8
GPU3 shard 3/8
GPU4 shard 4/8
GPU5 shard 5/8
GPU6 shard 6/8
GPU7 shard 7/8
```

每卡独立写：

```text
outputs/esi_mvp/experiences/shard_0.jsonl
...
shard_7.jsonl
```

最后 deterministic merge。

禁止：

```text
多个进程同时 append 同一文件
```

---

# 21. ESI 代码必须作为 SkillRL extension，而不是 fork 内大面积侵入

新增：

```text
agent_system/esi/
    __init__.py
    schema.py
    experience_packet.py
    failure_diagnosis.py
    inducer.py
    compiler.py
    runtime.py
    search_adapter.py
    validator.py
    registry.py
```

新增脚本：

```text
examples/esi/
    build_esi_split.py
    run_search_eval.py
    collect_experiences.py
    synthesize_skills.py
    validate_candidates.py
    run_causal_controls.py
    run_final_test.py
    orchestrate_esi_mvp.py
    analyze_esi_mvp.py
    watchdog_esi_mvp.py
```

尽量不要改：

```text
verl/
SearchEnv core
reward manager core
multi-turn rollout core
```

若必须改 upstream：

```text
每一个改动都要有注释：
# ESI EXTENSION
```

并记录到：

```text
outputs/esi_mvp/UPSTREAM_DIFF.md
```

---

# 22. ExperiencePacket

不要把全部 raw trajectory 塞进 inducer prompt。

构造：

```python
ExperiencePacket
```

建议内容：

```text
16 high-confidence failures
8 diverse successes
8 diverse additional failures
<= 32 total
```

失败样本优先包含：

```text
question
action sequence
search queries
observed evidence
wrong final answer
train-only gold
failure diagnosis
```

成功样本优先用于：

```text
发现可复用 search/decomposition/alignment pattern
```

---

# 23. Failure Diagnosis

先实现轻量 diagnosis：

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

诊断输入只来自：

```text
train experience
```

允许 train gold answer 参与判断最终错在哪里。

禁止：

```text
gold document 强行注入 trajectory
```

---

# 24. SkillSpec：Typed DSL

不要让 inducer 生成任意 Python。

使用 Pydantic。

推荐：

```python
class SkillSpec(BaseModel):
    name: str
    version: str
    target_query_type: str
    target_description: str

    trigger_cues: list[str]

    decomposition_strategy: dict
    query_templates: list[str]

    evidence_slots: list[dict]
    extraction_rules: list[dict]
    alignment_rules: list[dict]

    decision_rules: list[dict]
    termination_rule: dict
    answer_rule: dict

    max_turns: int
    max_searches: int

    provenance_experience_ids: list[str]
    rationale_summary: str
```

硬约束：

```text
max_turns <= SkillRL SearchEnv allowed max
max_searches <= configured budget
no shell
no file IO
no arbitrary eval/exec
no network except via SearchEnv
no hard-coded test IDs
no gold answers
no gold doc IDs
no answer lookup table
```

---

# 25. 最关键设计：Executable Skill 不允许绕过 SkillRL SearchEnv

建立：

```text
agent_system/esi/search_adapter.py
```

它必须把 ESI generic operator 翻译成 **SkillRL 官方 Search action**。

例如内部 operator 可以是：

```text
DECOMPOSE
FORM_QUERY
SEARCH
EXTRACT
ALIGN
COMPARE
VERIFY
ANSWER
```

但：

```text
SEARCH / ANSWER
```

最终必须经过：

```python
SearchEnv.step(action)
```

或 upstream 等价 interface。

禁止：

```python
requests.get(retriever_url)
```

绕过 environment。

这样：

```text
reward
observation
search cost
termination
```

仍由 SkillRL 定义。

---

# 26. Generic Operators

实现：

```text
decompose_question
identify_entities
form_atomic_query
consume_search_observation
extract_claim
align_entities
compare_attributes
check_evidence_complete
compose_final_answer
```

operator 可以调用 frozen actor 做局部结构化判断。

但 operator 不应：

```text
把整个问题重新丢给一个巨大 CoT prompt
```

否则 executable skill 会退化为 prompt skill。

每个 operator 必须输出结构化 state。

---

# 27. Runtime State

Executable runtime 维护：

```python
class ESIState:
    question
    subgoals
    entities
    pending_queries
    evidence_slots
    executed_queries
    search_observations
    hypotheses
    final_answer
    step_count
```

每步 trace：

```json
{
  "operator": "...",
  "input_state_hash": "...",
  "action": "...",
  "observation": "...",
  "state_delta": {},
  "skill_id": "...",
  "provenance": []
}
```

最终可以分析：

```text
行为到底改变在哪里
```

---

# 28. Skill Inducer

默认 inducer 先用：

```text
/data/ppnm/models/Qwen2.5-3B-Instruct
```

单独运行，不更新 actor weights。

生成：

```text
8 candidate SkillSpecs
```

temperature：

```text
0.6
```

必须 constrained/structured parse。

Inducer prompt：

```text
You are given real Search-agent experiences collected from the training split.
Infer a reusable executable procedure that repairs recurrent failures.

The output is NOT an answer and NOT a textual advice card.
Return one valid SkillSpec using only the allowed typed operators.

Do not memorize:
- answers
- task IDs
- gold document IDs
- exact training question strings

Every nontrivial rule must cite provenance_experience_ids.
```

---

# 29. Inducer model fallback

MVP 首轮优先同一个 3B model。

如果出现：

```text
8/8 candidates schema-invalid
```

或：

```text
structured output failure > 50%
```

允许 Agent 查找本机已经存在的更强 local instruct model：

```bash
find /data/ppnm/models -maxdepth 2 -type f -name config.json -print
```

可选一个已经存在的 7B–32B instruct model作为 **inducer only**。

约束：

1. 不因为 test performance 切模型；
2. 只能因为结构化 generation 无法工作；
3. 只能切一次；
4. 在 validation 前决定；
5. 记录 inducer path；
6. actor 始终冻结 Qwen2.5-3B-Instruct。

不要为了 inducer 随意下载几十 GB 新模型。

---

# 30. Candidate Synthesis

生成：

```text
outputs/esi_mvp/candidates/r0/candidate_00.yaml
...
candidate_07.yaml
```

每个先静态验证：

```text
schema
provenance
no leakage
compiler
operator legality
```

再做：

```text
8-task validation smoke
```

失败标：

```text
INVALID
```

严禁 fallback 到 native textual skill 或旧 ExPolicy primitive。

---

# 31. 8 卡 Candidate Validation

每张 GPU 一个 candidate：

```text
GPU0 candidate 0
...
GPU7 candidate 7
```

validation split：

```text
80 tasks 或可用全量
```

比较：

```text
B0 no skill
B1 SkillRL textual skill
E0 candidate_i executable
```

主指标：

```text
EM / official Search reward success rate
```

同时记录：

```text
searches
turns
tokens
runtime failures
```

---

# 32. Candidate Acceptance

最低：

```text
ΔEM_vs_B0 >= +0.03
negative_control ΔEM_vs_B0 >= -0.03
compiler_valid = true
no_leakage = true
```

推荐同时：

```text
candidate >= native textual SkillRL baseline
```

但这不是 MVP 硬门槛。

优先选择：

```text
highest validation EM
```

tie-break：

```text
fewer SearchEnv turns
```

---

# 33. 允许一次 validation-only Repair Round

如果没有 candidate 达到 +0.03：

允许一次：

```text
best candidate
+
validation failure traces
        ↓
inducer
        ↓
8 revised candidates
```

禁止：

```text
人工查看 future test
```

Round 1 仍不通过：

```text
STOP
```

不要无限调。

---

# 34. Content Causality Controls

至少：

## C0 Empty Experience

```text
ExperiencePacket = empty
→ inducer
→ executable candidate
```

## C1 Shuffled Experience

保持样本数量，打乱：

```text
question
failure diagnosis
evidence
outcome
```

之间对应关系。

## C2 Native SkillRL textual skill

使用原生 textual skillbank。

主比较：

```text
B0 no skill
B1 native textual SkillRL
E0 real-experience executable
E-empty
E-shuffle
```

如果：

```text
E0 > B0
但 E0 ≈ E-empty
```

只能写：

```text
Executable runtime helps,
but experience-content causality is not established.
```

---

# 35. Freeze

validation 选完后写：

```text
outputs/esi_mvp/FREEZE.json
```

包含：

```text
upstream commit
ESI git commit
actor model path/hash
inducer model path
split hash
candidate SkillSpec hash
compiler/runtime hash
validation result
```

Freeze 后禁止修改：

```text
SkillSpec
operator semantics
compiler
dispatch
split
```

只允许修 crash/OOM/serialization。

任何逻辑改动：

```text
invalidate freeze
重新 validation
```

---

# 36. Final Test

正式条件：

```text
B0 no skill
B1 native textual SkillRL
E0 accepted executable skill
E-empty
E-shuffle
```

在：

```text
future_test
```

上一次性评估。

第一轮：

```text
seed = 0
```

判定：

```text
ΔEM(E0 - B0) >= +0.03
```

通过后再扩：

```text
seed 1
seed 2
```

不要 test 后再改 skill。

---

# 37. Primary Dispatch

本轮不要做复杂 router。

Primary：

```text
只在 frozen comparison query-type subset 上启用 ESI skill
```

query type 必须来自：

```text
SkillRL train-side/frozen classification
```

不能利用 test answer。

Secondary 可选：

```text
question text → native SkillRL query-type classifier → skill
```

不阻塞 MVP。

---

# 38. 八卡最终阶段调度

建议：

```text
GPU0 ESI seed0
GPU1 ESI seed1
GPU2 ESI seed2
GPU3 textual SkillRL baseline
GPU4 empty control
GPU5 shuffled control
GPU6 negative control
GPU7 spare / retry / profiling
```

如果只跑 seed0：

优先并行各 condition。

---

# 39. 不做正式 RL 训练

本轮虽然以 SkillRL 为 backbone，但 ESI MVP 首要证明：

```text
experience → external executable capability
```

所以：

```text
不要因为 SkillRL 支持 GRPO，
就先做长时间 RL。
```

Gate 0 的 tiny GRPO 只是 infra smoke。

ESI MVP 正式 actor 权重保持冻结。

这样结果最容易解释：

```text
before actor == after actor
唯一变化 = experience-induced executable skill
```

---

# 40. Pipeline Audit

实现：

```text
examples/esi/audit_esi_pipeline.py
```

至少：

```text
[ ] upstream commit locked
[ ] G0.1–G0.7 PASS
[ ] real SearchEnv used
[ ] official Search reward/evaluator used
[ ] actor weights unchanged
[ ] train/val/test zero overlap
[ ] future test inaccessible to inducer
[ ] future test inaccessible to repair
[ ] ESI Search goes through SearchEnv.step
[ ] no direct retriever bypass
[ ] no old ExPolicy frozen primitive import
[ ] no CountOnly activation
[ ] every SkillSpec has train provenance
[ ] no gold answer in SkillSpec
[ ] no gold doc ID in SkillSpec
[ ] no exact test string in SkillSpec
[ ] final SkillSpec hash matches FREEZE
```

全部核心项 PASS 后才 final test。

---

# 41. Unit Tests

新增：

```text
tests/esi/
    test_schema.py
    test_compiler.py
    test_search_adapter.py
    test_no_retriever_bypass.py
    test_no_gold_leakage.py
    test_split_isolation.py
    test_no_old_primitive_import.py
    test_freeze_hash.py
```

至少：

```bash
pytest -q tests/esi
```

100% PASS。

不要删除 upstream tests。

若可行，额外：

```bash
pytest -q tests
```

若 upstream 原本有环境依赖导致部分无法运行，记录，不偷偷 skip 我们新增 tests。

---

# 42. Orchestrator

实现：

```text
examples/esi/orchestrate_esi_mvp.py
```

推荐：

```bash
cd /data/ppnm/skillrl_esi
conda activate skillrl-esi

python examples/esi/orchestrate_esi_mvp.py \
  --actor-model /data/ppnm/models/Qwen2.5-3B-Instruct \
  --dataset 2wikimultihopqa \
  --query-type comparison \
  --gpus 0,1,2,3,4,5,6,7 \
  --split-seed 20260818 \
  --candidate-count 8 \
  --max-repair-rounds 1 \
  --output outputs/esi_mvp
```

search backend 参数：

```text
从 Gate 0 已验证 config 自动读取
```

不要在 ESI script 中另设一套 search system。

---

# 43. Resume / Atomic Output

支持：

```text
--resume
per-stage DONE.json
file locks
atomic rename
```

每阶段：

```text
STATUS.json
DONE.json
```

失败：

```text
FAILED.json
```

内容：

```text
exception
command
git commit
GPU
last completed item
```

---

# 44. Watchdog

`watchdog_esi_mvp.py` 只处理：

```text
process dead
OOM
Ray worker dead
vLLM server dead
Search server unreachable
corrupt partial output
```

不自动：

```text
调 threshold
换 split
换 candidate
提高 test budget
```

---

# 45. 结果汇总

生成：

```text
outputs/esi_mvp/ESI_MVP_SUMMARY.md
outputs/esi_mvp/ESI_MVP_METRICS.csv
outputs/esi_mvp/ESI_MVP_DECISION.json
outputs/esi_mvp/ESI_RUN_LOG.md
```

主表：

| condition | val EM | test EM | Δ vs B0 | neg EM | searches | turns | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 no skill | | | 0 | | | | |
| B1 SkillRL textual | | | | | | | |
| ESI real | | | | | | | |
| ESI empty | | | | | | | |
| ESI shuffled | | | | | | | |

若 3 seed：

| seed | B0 | B1 textual | ESI | ESI-B0 |
|---:|---:|---:|---:|---:|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |

---

# 46. Runtime Behavior Analysis

不要只报分数。

自动分析 accepted ESI 相比 B0：

```text
first-query quality
number of searches
repeated query rate
multi-hop decomposition rate
evidence-slot completion
premature answer rate
comparison/alignment errors
```

从实际 traces 统计。

最终必须回答：

> **ESI 到底改变了什么行为，才让 performance 发生变化？**

---

# 47. 决策树

## CASE_INFRA_BLOCKED

如果 Gate 0 核心项没过：

```text
STOP ESI
```

结论：

```text
SkillRL Search foundation is not yet operational.
Do not attribute any failure to ESI.
```

只继续修 infra。

---

## CASE_UPSTREAM_PASS_ESI_VALIDATION_FAIL

Gate 0 PASS，但一次 repair 后 validation 无 candidate +0.03：

```text
STOP BEFORE FUTURE TEST
```

优先诊断：

```text
SkillSpec expressivity
compiler semantics
operator granularity
inducer quality
runtime adapter
```

不要回去做 acquisition。

---

## CASE_ESI_WEAK

future：

```text
0 < ΔEM < +0.03
```

保留结果：

```text
pipeline mechanically works,
effect size insufficient
```

不看 test 调参。

下一轮重新 split。

---

## CASE_ESI_PASS

seed0：

```text
ΔEM >= +0.03
```

若 3 seed：

```text
mean ΔEM >= +0.03
>=2/3 win
```

结论：

```text
The SkillRL-based experience-to-executable-skill pipeline is operational.
A frozen actor improves on held-out Search tasks after real training-side
experience is converted into an executable SkillSpec and executed through
the unchanged SkillRL Search environment.
```

下一步才考虑：

```text
skill editing
skill composition
inducibility-aware acquisition
multiple assimilators
OPCD/OEL weight internalization
```

---

# 48. 本轮不允许做的事情

严禁：

```text
直接从旧 ExPolicy 导入 R2/R5 C2 frozen procedure
CountOnly count>=K 解锁 procedure
test-aware repair
gold-as-hit
gold supporting docs 注入 runtime
手工按具体 test question 写 rule
直接访问 retriever 绕过 SearchEnv
更换 split seed 求正结果
无限 candidate/repair sweep
正式长时间 RL
frontier/RASA/AVA acquisition
```

---

# 49. SkillRL native baseline 不允许被偷偷弱化

B1 native textual SkillRL 必须：

```text
按照 upstream SkillBank interface 正常 retrieve/inject
```

不能为了让 ESI 好看：

```text
减少 B1 top-k
破坏 prompt
关闭它需要的 query type
```

如果 upstream default 不适合我们的 2Wiki subset：

```text
使用官方推荐 Search config
记录 config
```

---

# 50. ExPolicy 旧代码允许复用什么

允许：

```text
split audit 思路
freeze-before-test protocol
paired evaluation analysis
logging format
old public task IDs for exclusion
统计代码思想
```

不允许：

```text
old search runtime
old C1/C2/C3 executable primitives
old frontier acquisition
old CountOnly gate
old RASA
old AVA selector
```

本轮要明确证明：

```text
新 positive 不依赖旧 ExPolicy capability procedure
```

---

# 51. Git discipline

每个阶段至少一个清晰 commit：

```text
chore: establish SkillRL Search foundation
test: add foundation acceptance checks
feat: add ESI typed skill schema
feat: add SearchEnv executable runtime
feat: add experience-driven inducer
exp: add validation and freeze protocol
exp: run final ESI MVP
```

不要把：

```text
generated outputs
large checkpoints
dataset caches
```

误 commit。

记录 final：

```bash
git status
git log --oneline --decorate -10
git diff upstream/main...HEAD --stat
```

---

# 52. 最低交付物

代码：

```text
agent_system/esi/*
examples/esi/*
tests/esi/*
```

结果：

```text
outputs/esi_mvp/
    UPSTREAM_LOCK.json
    ENVIRONMENT_AUDIT.md
    UPSTREAM_DIFF.md

    gate0/
        G0_1_REPO_IMPORT.md
        SEARCH_PROTOCOL.md
        SEARCH_BACKEND_AUDIT.md
        ACTOR_AUDIT.md
        SKILLRL_FOUNDATION_ACCEPTANCE.md
        GATE0_DECISION.json

    SPLIT_AUDIT.md
    split.json

    base/
        no_skill/
        native_textual_skill/

    experiences/
    packets/
    candidates/
    validation/
    causal/

    FREEZE.json
    PIPELINE_AUDIT.md

    ESI_MVP_METRICS.csv
    ESI_MVP_DECISION.json
    ESI_MVP_SUMMARY.md
    ESI_RUN_LOG.md
```

如果 Gate 0 失败：

`FREEZE.json` 等后续文件可以不存在，但必须完整报告失败。

---

# 53. 最终 Summary 必须回答 10 个问题

`ESI_MVP_SUMMARY.md` 最后逐条回答：

1. SkillRL upstream commit 是什么？
2. Gate 0 哪些项目 PASS/WARN/FAIL？
3. SkillRL native SearchEnv 是否真实运行？
4. Frozen Qwen2.5-3B actor 的 B0 score 是多少？
5. Native SkillRL textual skill B1 score 是多少？
6. Real-experience induced executable skill score 是多少？
7. ESI 是否达到 +0.03？
8. Empty / shuffled controls 表现如何？
9. ESI 改变了哪些真实 Search runtime behavior？
10. 下一步应优先做 skill induction/editing，还是已经有资格重新做 experience acquisition？

---

# 54. Agent 的执行原则

你是负责这台 8×H20 机器的 Research/Coding Agent。

不要询问“是否继续”。

按 Gate 自动执行。

原则：

1. 先读 upstream code，再写 adapter；
2. 先 Gate 0，再 ESI；
3. upstream 能复用就不重写；
4. 所有 Search 必须经过真实 SearchEnv；
5. 不使用 mock；
6. 工程 bug 自行修；
7. 科学 Gate 不过就停，不改标准；
8. 不用 test 调参；
9. 不换 split seed；
10. 每个结论必须来自实际 artifact；
11. 不把未运行实验写成完成；
12. 不因 8 张卡空闲而启动无关实验；
13. 优先让 pipeline 正确，其次才追求 GPU 利用率。

---

# 55. 严格执行顺序

```text
Step 01  clone/pin SkillRL
Step 02  create isolated env
Step 03  install upstream + Search dependency
Step 04  read and document Search action protocol
Step 05  bring up real Search backend
Step 06  SearchEnv one-task smoke
Step 07  frozen local actor smoke
Step 08  native no-skill rollout
Step 09  native textual SkillBank rollout
Step 10  native trajectory/memory schema smoke
Step 11  optional tiny GRPO smoke
Step 12  write Gate 0 acceptance
Step 13  if Gate 0 FAIL: stop ESI
Step 14  build frozen 2Wiki split
Step 15  evaluate B0/B1
Step 16  collect real training experiences
Step 17  build ExperiencePacket
Step 18  implement typed SkillSpec
Step 19  implement SearchEnv adapter/compiler/runtime
Step 20  synthesize 8 skills
Step 21  validate 8 candidates on 8 GPUs
Step 22  at most one repair round
Step 23  run empty/shuffled controls
Step 24  freeze accepted candidate
Step 25  pipeline audit
Step 26  final future test seed0
Step 27  if +0.03, run seeds1/2
Step 28  behavior analysis
Step 29  final summary + decision
```

---

# 56. 立即开始执行

首先：

```bash
cd /data/ppnm
nvidia-smi

# clone if absent
git clone https://github.com/aiming-lab/SkillRL.git skillrl_esi || true

cd /data/ppnm/skillrl_esi

git status
git remote -v
git rev-parse HEAD

ls agent_system
ls skill_generation
ls examples/grpo_trainer

rg -n "SearchEnv|search_url|2wikimultihopqa|run_search_skills|skills_only_memory" \
  agent_system skill_generation examples README.md
```

然后：

```text
先完成 SkillRL Search foundation audit。
```

**不要先写 ESI candidate generator。**

只有 `GATE0_DECISION.json` 显示：

```text
G0.1–G0.7 = PASS
```

之后才进入 ESI。

---

# 57. 本轮最重要的一句话

不要把本轮做成：

```text
“又一个自研 Search Agent framework + ESI”
```

而要做成：

```text
SkillRL = 成熟 experience-learning / rollout substrate
ExPolicy = 严格实验 protocol 的来源
ESI = 唯一需要创新的 experience-to-executable-capability operator
```

本轮先证明：

> **在一个别人已经能跑通、能学习 experience、能维护 SkillBank 的成熟框架里，我们只替换 assimilation/execution 这一层，就能把真实 experience 变成新的 executable capability，并让 frozen actor 在未见 Search tasks 上变强。**

只有这一步成立，下一阶段才重新打开：

```text
Which experience should we acquire?
Which assimilator should absorb it?
When should a skill be edited, composed, promoted, or internalized?
```
