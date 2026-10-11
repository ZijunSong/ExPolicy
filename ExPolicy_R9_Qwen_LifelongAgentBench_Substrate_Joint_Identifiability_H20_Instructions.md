# ExPolicy R9：LifelongAgentBench Substrate & Joint Identifiability —— H20 执行指令

> **目标机器**：8×H20-3e  
> **默认只使用 GPU 4 / 5 / 6 / 7；单卡约 144GB；严禁杀掉或抢占其他用户进程。**  
> **主 repo**：`/data/ppnm/expolicy`  
> **建议新分支**：`r9-lab-substrate`  
> **本轮性质**：**method substrate / identifiability experiment，不是为了“救”R7 controller。**
>
> 本指令依据最新 R8 结论制定：R7 SearchEnv 已被证明结构退化；ALFWorld + frozen 3B foundation 又低于可解释区间，因此下一步必须先找到真正允许 heterogeneous persistent assimilation 的 benchmark，再谈 learned ExPolicy。
>
> **2026-08-21 修订：取消 gated Llama actor，改为公开 Qwen Model Ladder。利用单卡 144GB H20，在正式主实验前预注册筛选 Qwen3-30B-A3B / QwQ-32B / Qwen3-14B / Qwen3-32B；选定后全程冻结。**

---

## 0. 本轮唯一科学目标

当前不要继续问：

> “Linear TS 是否还能换成更强的 bandit / MLP / RL？”

R8 已经回答：在旧 SearchEnv 上不值得问。

本轮只回答两个更基础的问题：

### Q1. 是否存在一个公开、可复现、foundation 合理的 lifelong-agent substrate？

要求 frozen actor 在 no-learning 条件下：

- 不能接近 0；
- 不能接近 ceiling；
- 必须同时存在大量成功与失败轨迹；
- 后续 intervention 有真实上升/下降空间。

### Q2. 在同一条真实 experience 上，是否真的存在“该怎么学”的非平凡选择？

即对于同一个 experience batch \(B\) 和当前 persistent state \(S\)，不同 assimilation operator：

\[
A \in
\{
\text{DISCARD},
\text{EPISODIC},
\text{SEMANTIC},
\text{SKILL}
\}
\]

是否产生显著不同的 future capability gain：

\[
U(B,A\mid S)
=
J(A(S,B);D_{\mathrm{shadow}})
-
J(S;D_{\mathrm{shadow}})
-
0.5\max(0,-\Delta J_{\mathrm{retention}})
\]

并且不存在一个固定 operator 在所有 batch / task family 上统治。

**只有 Q1、Q2 都成立，才允许进入下一轮 learned joint ExPolicy。**

---

# 1. 为什么这次仍优先 LifelongAgentBench，以及如何选择 Qwen actor

本轮第一选择仍固定为：

**LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners**

官方代码：

```text
https://github.com/caixd-220529/LifelongAgentBench
```

数据：

```text
https://huggingface.co/datasets/csyq/LifelongAgentBench
```

选择它不是因为“更容易刷分”，而是因为它同时满足：

1. DB / OS / KG 三个异构 interactive environment；
2. 任务天然 skill-grounded、跨任务共享 atomic skills；
3. 自动 verifier，不依赖 LLM judge；
4. 原论文明确测 lifelong experience replay；
5. 公开代码、Docker 环境和固定 task sequence；
6. 已有 backbone ablation 能直接帮助我们判断 foundation 是否会 ceiling / collapse。

原 LifelongAgentBench 报告的 no-replay 结果提供了一个重要提醒：

```text
Llama-3.1-8B-Instruct:
DB ≈ 0.19 / OS ≈ 0.43 / KG ≈ 0.28

Qwen2.5-7B-Instruct:
DB ≈ 0.74

Qwen2.5-32B-Instruct:
DB ≈ 0.82

QwQ-32B:
DB ≈ 0.29
```

因此：

- **不能直接把“更大的 Qwen”当成更好的 substrate。**
- Qwen2.5-7B / 32B 在 DB 已经有明显 ceiling 风险；
- QwQ-32B 在论文的 DB no-replay 上反而处于理想中间区间；
- 新的 Qwen3 系列在 LifelongAgentBench 上没有足够公开结果可直接判断，因此必须先做严格、预注册的 model-screen。

本机 H20 单卡显存约 **144GB**，因此 30B–32B BF16 模型可以单卡部署。模型选择的瓶颈不是显存，而是：

> **哪个 Qwen actor 能让 DB / OS / KG 同时处于非退化、可学习、可被 persistent intervention 改善的区间。**

---

## 1.1 Qwen Model Ladder：正式实验前一次性筛选，不允许事后换模型

候选按下面四个模型冻结：

| ID | Hugging Face | 类型 | 角色 |
|---|---|---|---|
| Q0 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 30.5B MoE / 3.3B active，non-thinking only | **首选现代 agent actor** |
| Q1 | `Qwen/QwQ-32B` | 32.5B dense reasoning model | **LAB 已有 DB≈0.29 的 foundation anchor** |
| Q2 | `Qwen/Qwen3-14B` | 14B dense | 若 30B/32B 在部分 env ceiling，用较弱 dense Qwen 检查 |
| Q3 | `Qwen/Qwen3-32B` | 32.8B dense | 检查 MoE active-capacity 不足是否导致 floor |

四个模型均使用 Qwen 官方 Hugging Face repo；当前模型卡均为公开 `apache-2.0`，不依赖 Llama 类 gated authorization。

### 非 thinking / thinking 约束

为了避免“思考模式”成为隐藏实验变量：

- `Qwen3-30B-A3B-Instruct-2507`：官方只支持 non-thinking，直接使用；
- `Qwen3-14B` / `Qwen3-32B`：**固定 non-thinking mode**；
- `QwQ-32B`：其 reasoning behavior 是模型本身定义的一部分，不人为去掉 reasoning；必须单独记录 reasoning token / parser valid rate。

对于 Qwen3-14B / 32B，client 必须显式传：

```python
extra_body={
    "chat_template_kwargs": {
        "enable_thinking": False
    }
}
```

如果当前 vLLM / tokenizer 版本无法正确关闭 thinking：

```text
MODEL_MODE_UNSUPPORTED
```

该 candidate 退出 screening；不得用 prompt hack 模拟关闭 thinking。

---

## 1.2 Model screen 与正式实验必须物理隔离

在 `split.json` 中额外建立：

```text
model_screen
```

要求每个 environment：

```text
24 tasks / model / env
```

即每个完整 candidate：

```text
72 no-learning episodes
```

screen task：

- 不进入 calibration；
- 不进入 acquisition_pool；
- 不进入 assim_validation；
- 不进入 shadow_reward_probe；
- 不进入 retention_probe；
- 不进入 future_test；
- 不用于 operator prompt / skill / memory 设计。

模型 screening 只允许决定：

> **哪一个 frozen Qwen actor 用于后续 R9。**

不能决定：

- 哪些 environment 被保留；
- 哪种 assimilation operator 被删除；
- 哪个 Gate threshold 被修改。

---

## 1.3 预注册 Model Selection Rule

对 model \(m\)、environment \(e\) 的 no-learning success 记为：

\[
s_{m,e}
\]

### Full foundation candidate

候选必须先满足：

```text
DB, OS, KG 三个环境全部：
0.10 <= success <= 0.70
```

同时：

```text
parser_valid_rate >= 0.90
infra_failure_rate <= 0.05
```

否则不能进入正式 R9。

### 若多个模型都 Full-PASS

预先固定目标中心：

```text
target_success = 0.40
```

计算：

\[
Score(m)
=
\frac{1}{3}\sum_e |s_{m,e}-0.40|
+
0.5\max_e |s_{m,e}-0.40|
+
0.5(1-\text{parser-valid-rate}_m)
\]

**Score 最低者获选。**

Tie-break 固定顺序：

```text
Q0 Qwen3-30B-A3B-Instruct-2507
>
Q1 QwQ-32B
>
Q2 Qwen3-14B
>
Q3 Qwen3-32B
```

这个 tie-break 不是“选性能最高”，而是优先：

1. 最新、non-thinking、tool-use 友好的 Qwen3-30B-A3B；
2. 已有 LAB foundation anchor 的 QwQ-32B；
3. 再考虑 capacity adjustment。

### 如果没有任何 Full-PASS model

判定：

```text
MODEL_LADDER_NO_FULL_PASS
```

**本轮停止。**

禁止：

- 只删掉 ceiling 的 DB；
- 只保留 OS/KG；
- 看到结果后新加第五个模型；
- 改 prompt；
- 改 max_steps；
- 把 foundation band 从 10%–70% 放宽。

这时正确动作是：

> 重新选择 benchmark × Qwen actor substrate，或重新设计环境，而不是 narrow ExPolicy claim。

---

## 1.4 模型下载与本地资产

先检查：

```bash
find /data/ppnm/models -maxdepth 1 -type d \
  \( -name '*Qwen3*' -o -name '*QwQ*' \) -print
df -h /data
```

缺失时优先使用：

```bash
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir /data/ppnm/models/Qwen3-30B-A3B-Instruct-2507

hf download Qwen/QwQ-32B \
  --local-dir /data/ppnm/models/QwQ-32B

hf download Qwen/Qwen3-14B \
  --local-dir /data/ppnm/models/Qwen3-14B

hf download Qwen/Qwen3-32B \
  --local-dir /data/ppnm/models/Qwen3-32B
```

若环境只有旧 `huggingface-cli`，可以等价使用：

```bash
huggingface-cli download ...
```

这几个模型是公开模型，不应要求 gated-model authorization。

若下载因网络 / 磁盘失败，写：

```text
outputs/r9_lab/audit/MODEL_ASSET_BLOCKED.md
```

这是 infra block，不得静默改模型列表。

# 2. 开跑前必须先做 Novelty Collision Audit

## 2.1 当前 ExPolicy 要保护的贡献边界

论文主贡献仍必须是下面这个 conjunction，而不是其中某个单轴：

> **Agent 学习一个作用于自身学习过程的 Experience Policy：既决定下一步获得什么真实环境经验，也决定得到该经验后使用哪一种 heterogeneous persistent assimilation operator，以最大化 held-out future capability。**

需要保护的四个必要成分：

| 记号 | 必要成分 |
|---|---|
| A | **active environment experience acquisition**：决定下一批真实环境 experience 去哪里获得 |
| B | **heterogeneous persistent assimilation choice**：同一 experience 可以选择 memory / skill / weight / harness 等不同持久更新方式 |
| C | **stateful closed-loop learning**：前一次 persistent update 改变之后的 acquisition / assimilation 决策 |
| D | **future-capability objective**：优化 held-out future task capability，而不是当前 task reward / retrieval accuracy |

### Hard collision 条件

若发现任意已公开工作已经明确满足：

```text
A ∧ B ∧ C ∧ D
```

则：

```text
NOVELTY_COLLISION_HARD_STOP
```

**停止实现 learned ExPolicy。**

必须 redesign scientific question；禁止通过“我们多了一个小模块 / 换了 benchmark / 换了 reward”来 narrow claim。

---

## 2.2 必读 / 必审计近邻

至少全文检查以下工作；不要只看 abstract。

### 高风险

1. **MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs**  
   arXiv:2605.10064  
   风险：task-level search bandit + skill-level routing bandit + shared reward + persistent evolution。  
   **必须重点确认：**
   - task bandit 到底是在“选择真实新 experience / task”，还是只做 retrieval/search strategy；
   - skill bandit 是 inference-time routing，还是对 acquired experience 选择 persistent assimilation operator；
   - 是否用 held-out future capability 给 experience 定价。

2. **Ask Only When Needed: Proactive Retrieval from Memory and Skills for Experience-Driven Lifelong Agents / ProactAgent**  
   arXiv:2604.20572  
   风险：typed factual / episodic / behavioral-skill repositories + RL retrieval。  
   重点确认它优化的是：
   - “什么时候/检索什么已有经验”，还是
   - “下一步去环境中获得什么新经验”。

3. **Evolving Agents (EvA)**，ACL 2026  
   DOI: `10.18653/v1/2026.acl-long.754`  
   风险：Controller + memory + quasi-symbolic abstraction + online adaptation。  
   必须确认其 Controller meta-action 是否已经覆盖 acquisition × heterogeneous persistent assimilation。

### 中高风险

4. **MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents**  
   arXiv:2602.02474

5. **Learning How to Remember / MCMA**，ACL Findings 2026  
   arXiv:2601.07470

6. **From Player to Master / MemoPilot**，ICML 2026  
   arXiv:2606.08656

7. **SkillOpt: Executive Strategy for Self-Evolving Agent Skills**  
   arXiv:2605.23904  
   官方代码：
   `https://github.com/microsoft/SkillOpt`

8. **MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory**  
   arXiv:2601.03192  
   官方代码：
   `https://github.com/MemTensor/MemRL`

9. **How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior**，ACL 2026

10. **Rethinking Self-Evolving Agent Skills: Feedback Dynamics over Multiple Rounds**  
    arXiv:2608.02636

---

## 2.3 必须生成

```text
outputs/r9_lab/audit/NOVELTY_COLLISION_R9.md
outputs/r9_lab/audit/NOVELTY_MATRIX.csv
```

`NOVELTY_MATRIX.csv` 至少包含：

```text
paper
venue_or_status
new_env_experience_acquisition
heterogeneous_assimilation_choice
stateful_closed_loop
heldout_future_capability_objective
same_experience_operator_choice
joint_policy
code_available
collision_risk
evidence
```

结尾只能给两个状态之一：

```text
NOVELTY_STATUS = PROCEED
```

或

```text
NOVELTY_STATUS = HARD_STOP
```

如果全文 / code 访问失败、关键列无法确认：

```text
NOVELTY_STATUS = INCOMPLETE
```

**INCOMPLETE 也禁止开正式 GPU 主实验。**

---

# 3. Repo 与第三方代码审计

先不要写新 controller。

## 3.1 主 repo

```bash
cd /data/ppnm/expolicy

git status --short
git rev-parse HEAD
git log -1 --oneline

git switch -c r9-lab-substrate
```

如果工作区不是 clean：

- 不允许 `git reset --hard`；
- 不允许覆盖 R7/R8 产物；
- 先把 diff 写入：

```text
outputs/r9_lab/audit/PREEXISTING_WORKTREE_DIFF.patch
```

记录：

```text
outputs/r9_lab/audit/BASE_GIT_HEAD.txt
```

---

## 3.2 第三方 repo

统一放：

```text
/data/ppnm/third_party/
```

至少：

```bash
cd /data/ppnm/third_party

git clone https://github.com/caixd-220529/LifelongAgentBench.git
git clone https://github.com/microsoft/SkillOpt.git
```

MemRL 仅作强 baseline / implementation reference；本轮 Gate H 不依赖它跑通：

```bash
git clone https://github.com/MemTensor/MemRL.git
```

每个 repo 立即记录：

```bash
git rev-parse HEAD
git status --short
```

写入：

```text
outputs/r9_lab/audit/THIRD_PARTY_COMMITS.md
```

### 严禁

- 修改第三方源码后仍声称“官方 MemRL / SkillOpt”；
- cherry-pick 我们旧 R5/R7 frozen C1/C2/C3 skill；
- 把旧 CountOnly K=8 后端带进本轮；
- 用 ALFWorld workaround；
- 用 future_test 调 prompt、skill、LoRA、threshold。

若必须适配 benchmark，所有 glue code 放在：

```text
expolicy/r9_lab/
```

第三方 checkout 保持 clean。

生成：

```text
outputs/r9_lab/audit/THIRD_PARTY_PATCH_AUDIT.md
```

要求：

```text
LifelongAgentBench working tree clean = true
SkillOpt working tree clean = true
MemRL working tree clean = true  # 若 clone
```

---

# 4. 机器 / Docker / GPU preflight

执行并落盘：

```bash
mkdir -p /data/ppnm/expolicy/outputs/r9_lab/audit

date
hostname
nvidia-smi
df -h
free -h
docker --version
docker info
java -version || true
```

保存到：

```text
outputs/r9_lab/audit/SYSTEM_SNAPSHOT.txt
```

### GPU 使用规则

默认：

| GPU | 用途 |
|---|---|
| 4 | frozen actor vLLM / DB worker |
| 5 | frozen actor / OS worker |
| 6 | frozen actor / KG worker |
| 7 | SkillOpt optimizer / counterfactual eval overflow |

但真正开进程前必须再次 `nvidia-smi`。

若 4–7 某卡已有他人高显存任务：

- 不 kill；
- 自动重映射到当前真正空闲的允许 GPU；
- 如果不足 3 张空闲 H20，暂停正式 experiment；
- smoke 可以单卡执行。

---

# 5. LifelongAgentBench 先做“原生复现”，禁止直接接 ExPolicy

## 5.1 环境

单独建环境，避免污染原 ExPolicy：

```bash
conda create -n expolicy-r9 python=3.10 -y
conda activate expolicy-r9
```

按官方 repo 安装。

Docker 至少检查：

```bash
docker pull mysql
docker pull ubuntu
```

并按照 LifelongAgentBench 官方 Dockerfile 构建 DB / OS 环境。

KG 必须使用官方 SPARQL / KG implementation。

---

## 5.2 Qwen model-screen server

不要一开始就启动正式 actor。

先逐个 candidate 启动 model-screen vLLM server。

建议统一：

```text
port = 18100
dtype = bfloat16
max_model_len = 32768
temperature = 0
top_p = 1
```

### Q0：Qwen3-30B-A3B-Instruct-2507

```bash
export MODEL=/data/ppnm/models/Qwen3-30B-A3B-Instruct-2507

CUDA_VISIBLE_DEVICES=4 \
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen3-30b-a3b-r9 \
  --host 127.0.0.1 \
  --port 18100 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.72
```

该模型是 non-thinking only，client 无需设置 `enable_thinking=False`。

### Q1：QwQ-32B

```bash
export MODEL=/data/ppnm/models/QwQ-32B

CUDA_VISIBLE_DEVICES=4 \
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwq-32b-r9 \
  --host 127.0.0.1 \
  --port 18100 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.72
```

QwQ 可能产生较长 reasoning；必须记录：

```text
reasoning_tokens
total_output_tokens
parser_valid_rate
timeout_rate
```

不要因为它慢而单独给更少 max_steps。

### Q2 / Q3：Qwen3-14B / Qwen3-32B

同样使用单卡 BF16 server。

client 固定：

```python
temperature = 0
top_p = 1

extra_body={
    "chat_template_kwargs": {
        "enable_thinking": False
    }
}
```

---

## 5.3 执行 Model Ladder

实现：

```text
scripts/r9/02_model_screen.py
scripts/r9/03_select_model.py
```

每个 candidate：

```text
DB model_screen 24
OS model_screen 24
KG model_screen 24
```

推荐 H20 并行：

```text
GPU4 -> Q0
GPU5 -> Q1
GPU6 -> Q2
GPU7 -> Q3
```

如果四模型不能同时驻留，则两两串行；**screening 结果不能因为调度方式改变。**

输出：

```text
outputs/r9_lab/R9_MODEL_SCREEN.csv
outputs/r9_lab/audit/MODEL_SELECTION.md
outputs/r9_lab/model_manifest.json
```

`model_manifest.json` 至少包含：

```json
{
  "selected_model_id": "Q0/Q1/Q2/Q3",
  "hf_repo": "...",
  "local_path": "...",
  "revision_or_snapshot": "...",
  "dtype": "bfloat16",
  "thinking_mode": "...",
  "selection_score": 0.0,
  "screen_split_sha256": "..."
}
```

### Model screen 结束后立即冻结

获选模型写入：

```bash
export R9_ACTOR_MODEL=/data/ppnm/models/<SELECTED_QWEN>
export R9_ACTOR_NAME=<selected-served-model-name>
```

之后 Phase 1–4：

- 不再比较 model size；
- 不再换 actor；
- 不再改 thinking mode；
- 不再改 system prompt；
- 不因 Gate H / J 失败切换模型。

---

## 5.4 正式 actor server

关闭 screening server 后，只启动获选模型。

示例：

```bash
CUDA_VISIBLE_DEVICES=4 \
python -m vllm.entrypoints.openai.api_server \
  --model "$R9_ACTOR_MODEL" \
  --served-model-name "$R9_ACTOR_NAME" \
  --host 127.0.0.1 \
  --port 18100 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.72
```

client 固定：

```text
temperature = 0
top_p = 1
```

如果获选 Q2/Q3：

```text
enable_thinking = false
```

如果获选 Q1 QwQ：

- 保留其原生 reasoning；
- parser 只消费 benchmark 规定的最终 action；
- reasoning 不得进入 verifier gold path。

不得因为正式 foundation success 比 screen 略低就重新选模型。

# 6. Phase 1：Foundation Gate

## 6.1 先审计官方任务顺序与 split 语义

实现任何 split 前，必须阅读：

- LifelongAgentBench dataloader；
- `sample_index` 的含义；
- 官方 train / valid / test 是否存在；
- previous-experience callback 如何工作；
- DB / OS / KG 是否有真实跨任务 environment state dependency，还是只有 experience/skill dependency。

生成：

```text
outputs/r9_lab/audit/LAB_SPLIT_SEMANTICS.md
```

**禁止凭经验猜 split。**

---

## 6.2 构造 frozen R9 split

若官方没有满足本实验需求的现成互斥 split，则使用：

```text
split_seed = 20260822
```

要求每个 environment 都具有：

```text
calibration
acquisition_pool
assim_validation
shadow_reward_probe
retention_probe
future_test
```

原则：

1. task_id 零重叠；
2. exact instruction 零重叠；
3. future_test 永远不进入 operator fitting / policy update；
4. split 可以用 benchmark skill metadata 做**分层**；
5. 后续 controller 不允许看到 gold skill label；
6. 如果原 benchmark 存在必须保持的 serial dependency，则优先保持官方 order，不要为了随机 split 破坏 benchmark 语义。

建议目标规模（数据允许时，每 env）：

```text
calibration        80
acquisition_pool  200
assim_validation   60
shadow_probe       60
retention_probe    40
future_test        80
```

如果某 env 总量不够，按比例缩放，但：

```text
future_test >= 60 / env
shadow_probe >= 40 / env
```

生成：

```text
outputs/r9_lab/split.json
outputs/r9_lab/audit/SPLIT_AUDIT.md
```

同时记录 SHA256。

---

## 6.3 Selected-Qwen no-learning foundation

只允许使用 `model_manifest.json` 中已经冻结的 selected Qwen actor：

```text
frozen selected Qwen actor
no memory
no skill
no weight update
no previous trajectory replay
temperature = 0
formal calibration 与 model_screen 完全隔离
```

如果 selected actor 是 Qwen3-14B / Qwen3-32B：

```text
enable_thinking = false
```

如果 selected actor 是 QwQ-32B：

```text
保留原生 reasoning behavior
不得用 prompt 去关闭 reasoning
```

每个 environment 至少 `n=80 calibration`。

主指标：

```text
task success rate
mean action steps
invalid action rate
tool/parser failure rate
token usage
wall time
```

### Gate F

三个环境分别判断。

**PASS 区间：**

```text
0.10 <= success <= 0.70
```

并要求：

- 每个 env 至少 8 success；
- 每个 env 至少 8 failure；
- 非零 parser-valid action rate；
- failure 不是主要由 infra / parser / Docker crash 导致。

### 判定

#### CASE_F_PASS

3/3 env 全过。

→ 进入 Phase 2。

#### CASE_F_MODEL_MISMATCH

selected Qwen 在独立 formal calibration 上不再处于可解释带，但 infra 正常。

→ **本轮停止。**

这说明 model-screen 估计不稳定或 benchmark×actor substrate 不可靠。

禁止：

- 回到 Model Ladder 选第二名；
- 看 formal calibration 后改 selection score；
- 调 prompt / max_steps / thinking；
- 只删掉失败 environment。

下一轮必须重新预注册 benchmark×Qwen actor substrate。

#### CASE_F_INFRA_FAIL

Docker / parser / API / benchmark integration 导致错误。

→ 只修工程 bug，然后从 clean foundation 重新跑。

工程修复不算方法调参。

---

# 7. Phase 2：先验证每个 Assimilation Operator 是“真的能工作”

本轮不发明新的 assimilation 方法。

我们要用**已经存在的、概念上不同的 persistent learning carrier**构造 action space。

正式 action：

```text
A0 DISCARD
A1 EPISODIC_REPLAY
A2 SEMANTIC_MEMORY
A3 SKILLOPT_EDIT
```

可选 `A4 LoRA` **本轮先不进入主 Gate**；只有 H 过后下一轮再加入 weight-space carrier。

原因：现在最重要的是证明“不同 persistent operator 有不可替代的条件优势”，不是再次用大量工程把 action space 做大。

---

## 7.1 A0 DISCARD

语义严格为：

```text
experience 被观察到，但不改变任何 persistent capability state
```

这是必要 baseline。

---

## 7.2 A1 EPISODIC_REPLAY

尽量忠实复现 LifelongAgentBench 自带 experience replay：

- 只写 verified successful trajectories；
- 保留 task / observation / action sequence / final verifier outcome；
- retrieval 规则固定；
- `top_k=4` 作为主设置；
- 不使用 future / shadow label。

禁止：

- 手工给不同 environment 写不同 prompt；
- 根据 future performance 删除 bad memory；
- 用 gold action plan 替换真实 rollout。

输出：

```text
memory_state.jsonl
retrieval_trace.jsonl
```

---

## 7.3 A2 SEMANTIC_MEMORY

这是一个普通、非 novelty 的 structured memory carrier。

每个 acquired batch 结束后，只从**该 batch 的真实 trajectory + verifier result**生成结构化条目：

```json
{
  "scope": "...",
  "trigger": "...",
  "principle": "...",
  "failure_pattern": "...",
  "recommended_action_pattern": "...",
  "evidence_task_ids": ["..."]
}
```

要求：

- 不写 gold answer；
- 不读取 future / shadow；
- 条目必须带 evidence provenance；
- 去重规则固定；
- inference retrieval `top_k=4`；
- 生成器默认使用同一个 **frozen selected Qwen actor**，不允许换强 teacher。

如果失败轨迹无法产生可信 correction：

```text
failure_pattern 可以写；
recommended_action_pattern 不允许伪造。
```

---

## 7.4 A3 SKILLOPT_EDIT

直接使用 Microsoft SkillOpt 的**原始算法语义**：

- persistent state = `best_skill.md`；
- acquired experience = scored rollout batch；
- optimizer 将 rollout 转成 bounded add / delete / replace edit；
- edit 必须经过 `assim_validation`；
- validation 不提升则 reject；
- future_test 不进入 selection。

Benchmark adapter 写在：

```text
expolicy/r9_lab/skillopt_adapter/
```

不要 patch `third_party/SkillOpt`。

主设置：

```text
target actor = selected Qwen actor
optimizer = 同一个本地 selected Qwen actor
max optimizer update = 1 bounded edit / experience batch
```

这样避免“强 teacher 蒸馏”成为隐藏变量。

若 SkillOpt 的官方 contract 无法支持 local OpenAI-compatible backend：

- 先做 adapter；
- 保持其 edit / validation / reject 语义；
- 在 `SKILLOPT_ADAPTER_AUDIT.md` 写清楚；
- 不得把自定义 prompt optimizer 称为官方 SkillOpt。

---

# 8. Operator Sanity Gate

在正式 counterfactual matrix 前，每个 env 随机取：

```text
2 个 batch × 8 experiences
```

分别应用 A1/A2/A3。

至少证明：

1. persistent state 文件发生真实变化；
2. 新状态会被后续 rollout 实际读取；
3. disable operator 后结果回到 base 路径；
4. clone A 的 update 不污染 clone B；
5. future_test id 没进入任何 update；
6. SkillOpt rejected edit 不得偷偷写入 best skill；
7. episodic / semantic retrieval trace 能证明命中的是 training-side experience。

生成：

```text
outputs/r9_lab/audit/OPERATOR_SANITY.md
```

任一 operator 工程上无效：

```text
OPERATOR_SANITY_FAIL
```

只修实现，再重跑 sanity。

不得带着假的 action 进入 Gate H。

---

# 9. Phase 3：Heterogeneous Assimilation Counterfactual Matrix

这是本轮最关键实验。

## 9.1 Experience batch

从 `acquisition_pool` 预先冻结：

```text
12 batches / env
8 experiences / batch
```

总计：

```text
36 batches
288 real environment experiences
```

要求：

- 无 replacement；
- DB / OS / KG 各 12；
- 在 env 内按 skill concurrency / difficulty 分层；
- 不按 downstream utility 选 batch；
- batch IDs 在运行前写死。

输出：

```text
outputs/r9_lab/audit/FROZEN_BATCHES.json
```

---

## 9.2 Static counterfactual matrix

对于每个 batch \(B_i\)：

```text
clean persistent state S0
    ├── DISCARD
    ├── EPISODIC
    ├── SEMANTIC
    └── SKILLOPT
```

四个 branch 完全独立 clone。

每个 branch 更新后只评：

```text
shadow_reward_probe
retention_probe
```

禁止看 future_test。

记录：

```text
U = Δshadow_success
    - 0.5 * max(0, -Δretention_success)
```

同时单独记录：

```text
tokens
tool calls
optimizer calls
wall time
GPU seconds
state size
```

**Gate H 主判据先不用 compute cost。**

因为当前问题是：

> 是否存在真实的 operator choice？

不是：

> 哪个 operator 最便宜？

成本作为 secondary efficiency table。

输出矩阵：

```text
outputs/r9_lab/R9_ASSIM_MATRIX.csv
```

至少列：

```text
batch_id
env
difficulty
skill_count
base_success_rate
batch_success_rate
failure_type_stats
U_discard
U_episodic
U_semantic
U_skillopt
best_action
second_best_action
best_margin
```

---

# 10. Gate H：必须排除“又一个 R7 退化 action space”

Gate H 必须同时满足 H1–H6。

## H1：至少两个非 trivial operator 真正赢过

全局 36 batch 中，至少两个 action 的：

```text
unique-best fraction >= 0.15
```

并且不能只靠 ±1 个 probe task 的翻转。

---

## H2：Hindsight oracle 必须明显超过 global fixed

定义：

```text
oracle = 每个 batch 选 hindsight best action
best_fixed = 全 36 batch 固定使用同一个 action
```

要求：

```text
mean_U(oracle) >= mean_U(best_fixed) + 0.03
```

如果 probe 粒度使 0.03 小于两题，则用：

```text
>= 2 probe-task equivalents
```

取更严格者。

---

## H3：不能只是“DB 用 memory，OS 用 skill”的 environment-ID router

计算：

```text
best_per_env_fixed
```

即 DB / OS / KG 各允许一个固定 operator。

要求：

```text
oracle >= best_per_env_fixed + 0.02
```

并且至少 **2/3 environment** 内部满足：

```text
>= 2 个 operator 各自 unique-best >= 0.20
```

如果 oracle 优势只来自不同 environment 固定对应不同 carrier：

```text
H3 FAIL
```

这不够支撑 ExPolicy。

---

## H4：best action 必须具有可预测结构，而不是纯噪声

用**只允许 training-side / current-state feature**：

```text
env
task difficulty
skill concurrency
trajectory length
success/failure
invalid-action rate
repetition
context length
current memory size
current skill size
previous operator statistics
```

做 leave-one-batch-out 或 leave-one-skill-cluster-out：

- majority baseline
- env-only logistic regression
- full-feature logistic regression
- small MLP

要求：

```text
full-feature accuracy >= majority + 10 percentage points
```

且：

```text
full-feature >= env-only + 5 percentage points
```

否则 action labels 可能不可学。

---

## H5：至少两个 operator 有真实正 utility 区域

不能出现：

```text
SkillOpt 所有地方正
其它全部负
```

要求至少两个非-discard operator：

```text
在 >= 4 个 batch 上 U > 0
```

并且其中至少一个 operator 在另一些 batch 上明显差于另一个 operator。

---

## H6：非泄漏 / 非 artifact

必须：

- future_test 从未参与 update / selection；
- shadow 与 retention task 无 overlap；
- exact instruction 无 overlap；
- SkillOpt validation 与 shadow 分离；
- agent 输入中没有 gold skill IDs / gold action plans；
- benchmark verifier 只返回 success / execution feedback，不注入 reference action。

---

# 11. Gate H 判定树

## CASE_H_PASS

H1–H6 全过。

结论：

> **LifelongAgentBench 首次提供了可辨识的 heterogeneous assimilation decision space。**

继续 Phase 4：Joint Identifiability。

---

## CASE_H_FIXED_DOMINATES

best fixed ≈ oracle，或只有一个 operator 真正赢。

结论：

> substrate 仍不支持 adaptive assimilation。

**停止。**

不得把论文改成：

- “SkillOpt scheduler”
- “better memory”
- “在某一个 task family 上选择 carrier”

下一步应该换 substrate / operator set，而不是 narrow claim。

优先 fallback：

```text
ScienceWorld
WebShop
```

但**不要在本轮结果出来后立刻边看边调并开跑 fallback**；先输出 case report，再单独预注册下一轮。

---

## CASE_H_ENV_ROUTER_ONLY

global oracle > fixed，但 per-env fixed ≈ oracle。

说明只学到了：

```text
DB -> A
OS -> B
KG -> C
```

这属于环境标签 routing，不够形成强 ExPolicy。

**停止 / redesign。**

---

## CASE_H_UNLEARNABLE

oracle gap 明显，但 full-feature classifier 不优于 majority / env-only。

说明 decision space 存在，但目前 state/experience representation 不够。

这时允许下一轮 redesign **representation**，因为 oracle gap 已证明方法空间存在。

禁止把 claim 缩成 hindsight oracle。

---

# 12. Phase 4：Joint Acquisition × Assimilation Identifiability

**仅 CASE_H_PASS 才运行。**

本阶段仍然**不训练 learned controller**。

目的：

> 在一个已经有 heterogeneous assimilation 的 substrate 上，再问 acquisition choice 是否也具有独立价值，以及二者是否必须联合决策。

---

## 12.1 Joint candidate construction

3 个 seed：

```text
seed = 0,1,2
```

每个 seed 4 rounds。

每轮从尚未见过的 acquisition pool 中，冻结生成：

```text
M = 4 candidate micro-batches
batch size = 4
```

candidate 必须：

- 同一轮 environment 分布平衡；
- 不使用 future / shadow performance 构造；
- 只通过随机 + skill/diversity coverage 产生候选；
- candidate generation 不能本身成为 acquisition heuristic。

对于当前 live state \(S_t\)，对：

```text
4 candidate batches × 4 assimilation actions
```

做 temporary clone + shadow evaluation。

得到：

\[
Q(B,A\mid S_t)
\]

只用 shadow / retention。

---

## 12.2 四个 oracle 诊断

每轮计算：

### O0 passive-fixed

随机 batch + 全局 best fixed operator。

### O1 passive-oracle-assim

随机 batch + hindsight best assimilation。

回答：

> “只学 how to use experience” 的上界。

### O2 oracle-acq-fixed

在 4 个 candidate 里选最好 batch，但 assimilation 固定为 global best fixed。

回答：

> “只学 what to acquire” 的上界。

### O3 oracle-joint

```text
argmax_(batch, operator) Q(B,A|S)
```

回答：

> joint decision 是否真的必要。

只允许 O3 的选择影响 live state。

final `future_test` 仅在：

```text
budget 0 / 8 / 16 experiences
```

做评测，绝不回传。

---

# 13. Gate J：Joint Necessity

要求同时满足：

## J1

```text
O3 final future
>= max(O1, O2, O0) + 0.03
```

3-seed mean。

## J2

```text
>= 2/3 seed O3 > strongest axis-only oracle
```

## J3

O3 committed trajectory 中：

```text
>= 2 种 assimilation operator 被真正使用
```

且：

```text
>= 2 种 acquisition task/skill region 被选择
```

## J4

O3 优势不能由单一 env id 解释。

报告：

```text
within-env joint advantage
```

## J5

shadow-selected joint action 与 actual downstream delta 至少有一个：

```text
Spearman >= 0.35
```

或

```text
top-1 action mean downstream delta
>= random action + 0.03
```

## J6

无 future leakage。

---

# 14. Gate J 判定

## CASE_J_PASS

这才意味着：

> **ExPolicy 的两个决策轴在真实公开 benchmark 上同时具有非平凡、可学习的 decision space。**

此时**停止本轮**。

不要立刻在同一实验里调 learned controller。

下一轮 R10 才训练：

```text
state/experience/operator value model
or joint contextual policy
```

并比较：

```text
Full Joint ExPolicy
vs passive + adaptive assimilation
vs active acquisition + fixed assimilation
vs strongest fixed
vs official LifelongAgentBench replay
vs MemRL
vs SkillOpt
```

如果 R9 同时过 H + J，论文已经从“探索阶段”进入“主实验阶段”。

---

## CASE_J_ASSIM_ONLY

O1 ≈ O3，acquisition 没有独立价值。

**不要把论文缩成 adaptive assimilation。**

这会破坏 ExPolicy 原始强贡献。

选择：

```text
redesign acquisition candidate space
or change benchmark
or discard joint ExPolicy formulation
```

---

## CASE_J_ACQ_ONLY

O2 ≈ O3，assimilation 没有独立价值。

同样不能 narrow 成 active acquisition。

换 substrate / operator set。

---

## CASE_J_NO_JOINT_GAP

两个轴各自有信号，但 joint 不比 axis-only 强。

说明“联合 Experience Policy”缺少必要性。

停止；重新思考 formulation。

---

# 15. 必须做的 Case-level 分析

无论 PASS / FAIL，都生成：

```text
outputs/r9_lab/cases/
```

至少包含：

## 15.1 Operator flip cases

每对主要 operator：

```text
episodic > skillopt
skillopt > episodic
semantic > skillopt
skillopt > semantic
```

各取 utility margin 最大的 5 个 case。

保存：

- 原 task；
- trajectory；
- verifier feedback；
- persistent state before / after；
- retrieved memory / semantic entry / skill diff；
- shadow tasks 被翻对/翻错的具体 IDs；
- retention regressions；
- 为什么 operator A 比 B 更好。

---

## 15.2 Same-environment flip

必须专门找：

```text
同为 DB，但 best operator 不同
同为 OS，但 best operator 不同
同为 KG，但 best operator 不同
```

这是排除 environment-ID shortcut 的核心证据。

---

## 15.3 Negative-transfer cases

每个 non-discard operator 至少找 5 个：

```text
U < 0
```

的 case。

分析它为什么伤害 future capability：

- stale / irrelevant replay
- abstraction over-generalization
- skill edit overfit
- context interference
- action-pattern mismatch
- retention damage

---

# 16. 代码结构要求

不要把 R9 写成一次性 notebook。

建议：

```text
expolicy/
  r9_lab/
    benchmark/
      adapter.py
      split.py
      verifier.py
    state/
      persistent_state.py
      clone.py
    assimilators/
      base.py
      discard.py
      episodic.py
      semantic.py
      skillopt.py
    evaluation/
      shadow.py
      retention.py
      future.py
      utility.py
    joint_oracle/
      candidates.py
      counterfactual.py
      oracle.py
    audit/
      leakage.py
      state_isolation.py
      third_party.py
```

scripts：

```text
scripts/r9/
  00_preflight.py
  01_audit_split.py
  02_model_screen.py
  03_select_model.py
  04_foundation.py
  05_operator_sanity.py
  06_build_frozen_batches.py
  07_run_assim_matrix.py
  08_analyze_gate_h.py
  09_run_joint_oracle.py
  10_analyze_gate_j.py
  11_case_analysis.py
  12_write_master_summary.py
```

---

# 17. 单元测试 / 审计要求

正式 GPU 主跑前至少覆盖：

```text
tests/r9/
```

必须测试：

1. model_screen 与所有 formal split task-id overlap = 0；
2. model-screen selection score 与 tie-break 完全按预注册规则；
3. selected model hash / mode 写入 manifest 后不可变；
4. split task-id overlap = 0；
5. exact-question overlap = 0；
3. future_test 不进入 update；
4. shadow / retention 不写 persistent state；
5. clone 后 memory 独立；
6. clone 后 skill file 独立；
7. DISCARD 真 no-op；
8. rejected SkillOpt edit 不改变 live state；
9. semantic provenance 只来自 acquired batch；
10. episodic retrieval 只读取已提交 experience；
11. temporary counterfactual branch 不污染 live state；
12. candidate batch 只有 selected batch 被 commit；
13. final future metric 不进入 oracle selection；
14. resume 不会重复提交 batch；
15. hash guard 检测 split / model / config 变化；
16. random seed 可复现；
17. verifier 的 gold/reference action 不进入 LLM prompt；
18. environment reset 与官方 benchmark 一致。

要求：

```bash
pytest -q tests/r9
```

全部 PASS 才允许正式实验。

生成：

```text
outputs/r9_lab/audit/R9_PIPELINE_AUDIT.md
```

---

# 18. H20 并行运行建议

## Qwen Model Screen

四个 candidate 可直接一模型一卡：

```text
GPU4 -> Q0 Qwen3-30B-A3B-Instruct-2507
GPU5 -> Q1 QwQ-32B
GPU6 -> Q2 Qwen3-14B
GPU7 -> Q3 Qwen3-32B
```

每张 H20 144G 足够放置对应 BF16 权重和受控 KV cache；仍须以实际 `nvidia-smi` 为准。

screen 结束后必须关闭未选模型 server，正式实验只保留 selected Qwen actor。

## Foundation

三 env 分卡：

```text
GPU4 -> DB
GPU5 -> OS
GPU6 -> KG
GPU7 -> spare / infrastructure / optimizer smoke
```

## Assimilation matrix

按 batch shard：

```text
GPU4 -> shard 0
GPU5 -> shard 1
GPU6 -> shard 2
GPU7 -> shard 3
```

每个 shard 内 branch 必须串行或使用独立 state directory。

**禁止多个 worker 写同一个 persistent state 文件。**

目录：

```text
outputs/r9_lab/matrix/{batch_id}/{action}/
```

用：

```text
flock
atomic rename
DONE.json
config hash
```

实现 resume。

---

# 19. Watchdog

正式任务必须有：

```text
scripts/r9/watchdog.py
```

每个 job 定期写：

```text
STATUS_LIVE.json
```

包含：

```text
pid
gpu
stage
batch_id
action
last_progress_time
completed_tasks
tokens
```

stall 判定建议：

```text
20 min 无 task / rollout / optimizer progress
```

只能 kill **本轮记录的 PID**。

严禁：

```bash
pkill python
killall python
```

---

# 20. 本轮禁止事项

无论结果多差，都禁止：

1. 回 R7 SearchEnv 换 controller；
2. 调 CountOnly K；
3. 重新使用 C1/C2/C3 frozen skill bank；
4. 因 LifelongAgentBench 某 env 难就删掉该 env；
5. 看 future_test 后改 operator prompt；
6. 看 Gate H 后重抽 batch；
7. 改 utility 权重把结果“调过线”；
8. SkillOpt 不赢就改其 validation split；
9. 把只在 DB 成立写成“ExPolicy works on lifelong agents”；
10. joint 不成立后改论文成 active-only / assimilation-only 弱 claim；
11. 只报 AULC / 单 environment 来掩盖 final failure；
12. 失败后静默换 actor；
13. formal foundation / Gate H / Gate J 失败后回到 Model Ladder 选择第二名；
14. 为了让 Qwen3 过 Gate 临时开启/关闭 thinking；
15. 用 mock Docker / mock verifier / gold-as-observation；
16. 把第三方 operator 的贡献写成 ExPolicy novelty。

---

# 21. 如果结果不好，正确动作是什么

本轮必须使用下面的 fail-closed 逻辑。

```text
Novelty collision
    -> redesign problem / discard formulation

Model Ladder no full pass
    -> discard current LifelongAgentBench×Qwen candidate set
    -> do NOT delete an environment or widen the band

Formal Foundation fail
    -> discard selected benchmark×actor substrate
    -> do NOT fall back to second-ranked Qwen inside the same round

Foundation pass
but operator fixed dominates
    -> discard substrate/operator set
    -> do NOT train policy

Oracle > fixed
but env-only router explains it
    -> redesign substrate
    -> do NOT claim adaptive assimilation

Oracle gap exists
but features unlearnable
    -> redesign state / experience representation

Heterogeneous assimilation pass
but joint acquisition adds nothing
    -> joint ExPolicy formulation unsupported
    -> change benchmark / acquisition action space or discard

H + J both pass
    -> only then train learned ExPolicy in R10
```

**严禁把失败路线通过 narrowing claim 继续包装成论文。**

---

# 22. 最终必须输出的文件

至少：

```text
outputs/r9_lab/
  R9_MASTER_SUMMARY.md
  R9_DECISION.json
  R9_MODEL_SCREEN.csv
  R9_FOUNDATION.csv
  R9_ASSIM_MATRIX.csv
  R9_GATE_H.json
  R9_JOINT_ORACLE.csv            # 仅 H PASS
  R9_GATE_J.json                 # 仅 H PASS
  R9_COST.csv
  split.json
  model_manifest.json

  audit/
    NOVELTY_COLLISION_R9.md
    NOVELTY_MATRIX.csv
    BASE_GIT_HEAD.txt
    THIRD_PARTY_COMMITS.md
    THIRD_PARTY_PATCH_AUDIT.md
    SYSTEM_SNAPSHOT.txt
    MODEL_SELECTION.md
    LAB_SPLIT_SEMANTICS.md
    SPLIT_AUDIT.md
    OPERATOR_SANITY.md
    R9_PIPELINE_AUDIT.md

  cases/
    CASE_INDEX.md
    operator_flip/
    same_env_flip/
    negative_transfer/
```

---

# 23. `R9_MASTER_SUMMARY.md` 必须按这个顺序写

```text
1. Novelty status
2. Benchmark / Qwen Model Ladder / commits
3. Model-screen result and frozen actor
4. Formal foundation result
5. Operator sanity
6. 36×4 counterfactual matrix
7. Global fixed vs hindsight oracle
8. Per-env fixed vs hindsight oracle
9. Within-env operator diversity
10. Feature separability
11. Gate H
12. Joint oracle experiment（若运行）
13. Gate J（若运行）
14. Cost
15. Case analysis
16. Final decision
17. What is NOT supported
18. Exact next action
```

---

# 24. 最终 `R9_DECISION.json` 只允许以下状态

```text
NOVELTY_HARD_STOP
MODEL_LADDER_NO_FULL_PASS
FOUNDATION_FAIL
OPERATOR_SANITY_FAIL
SUBSTRATE_FIXED_DOMINATES
SUBSTRATE_ENV_ROUTER_ONLY
SUBSTRATE_UNLEARNABLE
HETEROGENEOUS_ASSIMILATION_PASS
JOINT_IDENTIFIABILITY_FAIL
JOINT_IDENTIFIABILITY_PASS
```

若：

```text
JOINT_IDENTIFIABILITY_PASS
```

则下一步非常明确：

> **R10：训练真正的 learned joint Experience Policy，并做 full vs acquisition-only vs assimilation-only vs strongest fixed / MemRL / SkillOpt。**

若不是这个状态，**禁止直接训练 R10**。

---

# 25. 本轮推荐的论文级判断标准

我们不是为了“再拿到一个 positive number”。

真正值得继续写 ExPolicy 的最小证据链应当变成：

```text
R7:
closed-loop runtime exists
        ↓
R8:
old SearchEnv is structurally degenerate
        ↓
R9-F:
new public substrate has non-degenerate frozen foundation
        ↓
R9-H:
same experience genuinely admits different useful persistent assimilation choices
        ↓
R9-J:
choosing what to experience AND how to assimilate it jointly beats either axis alone
        ↓
R10:
a learned deployable policy approximates that joint oracle and beats strong baselines
```

只有形成这条链，论文才应该继续保持：

> **Existing agents learn from experience; ExPolicy learns how to experience and how to learn from it.**

而不是退化成：

- memory router；
- skill selector；
- curriculum sampler；
- task bandit；
- SkillOpt wrapper；
- experience replay heuristic。

---

# 26. 预计时间

在 LifelongAgentBench / Qwen Model Ladder 顺利的情况下：

```text
Novelty + repo / split / infra audit      2–4 h
Qwen model screen                         3–8 h
Selected-Qwen formal Foundation           2–5 h
Operator integration + sanity             4–8 h
36×4 counterfactual matrix                8–20 h
Gate H analysis + cases                   1–3 h
Joint oracle（仅 H PASS）                 6–15 h
```

因此：

- **若模型资产和 Docker 已就绪，Gate H 结果通常仍可在约 1–2 天内得到；**
- **若 H PASS，完整 R9-J：约 1–2 天；**
- 若 `JOINT_IDENTIFIABILITY_PASS`，再剩一轮 R10 learned controller 主实验，之后即可进入论文主写作阶段。

这里的时间不是 deadline，只是机器排程参考。

---

# 27. 开跑前最后检查

正式 GPU run 前，执行 agent 必须明确回答并写入 `R9_PIPELINE_AUDIT.md`：

```text
[ ] Novelty PROCEED?
[ ] Third-party repos clean?
[ ] model_screen 与正式 split 物理隔离?
[ ] Qwen Model Ladder screen completed?
[ ] Model selection rule applied without post-hoc override?
[ ] selected Qwen actor frozen in model_manifest.json?
[ ] thinking/non-thinking mode frozen?
[ ] Official LAB environment really executed?
[ ] Foundation split frozen?
[ ] Future_test physically isolated from update path?
[ ] All operators really mutate persistent state?
[ ] Counterfactual clone isolation tested?
[ ] Batch IDs frozen before utility evaluation?
[ ] Gate H thresholds written before main matrix?
[ ] Failure will STOP rather than tune threshold?
```

任意一项为 NO：

```text
DO_NOT_START_FORMAL_RUN
```

---

## 最终原则

**这轮的成功不是“ExPolicy 分数变高”，而是证明 ExPolicy 值得被学习。**

R8 已经告诉我们：如果 action space 没有真实异质性，换再强的 controller 都没有意义。

所以 R9 必须先用公开、可复现的 LifelongAgentBench 严格证明：

\[
\exists B_1,B_2,A_1,A_2:
A_1 \neq A_2,\quad
A_1=\arg\max_A U(B_1,A),\quad
A_2=\arg\max_A U(B_2,A)
\]

而且这种差异不能由 environment ID 简单解释。

在此基础上，再进一步证明：

\[
\max_{B,A}U(B,A)
>
\max_B U(B,A_{\mathrm{fixed}})
\]

以及：

\[
\max_{B,A}U(B,A)
>
U(B_{\mathrm{passive}},A_{\mathrm{oracle}})
\]

**只有这两个不等式都在真实实验中出现，下一轮 learned ExPolicy 才有科学必要性。**
