# ExPolicy experiment record

## ExPolicy 核心目标（主线定义）

> **ExPolicy 的核心不是“从经验中生成 skill”，而是学习一个作用于经验本身的动态 policy：决定下一步应该获得什么经验，以及获得经验之后应该如何利用它，从而最大化未来能力增长。**

我们真正要解决的是两个相互耦合的问题：

1. **Experience Acquisition Policy — 学什么 / 去哪里获得经验**
   - 在有限交互预算下，主动决定下一步应该采集哪类 experience。
   - 目标不是最大化当前 rollout reward，也不是简单优先 failure / uncertainty / difficulty，而是优先获得**对未来能力提升最有价值**的经验。
   - 核心问题是：  
     **Given the current capability state, what should the agent experience next?**

2. **Experience Assimilation Policy — 得到经验后怎么处理**
   - 一条 experience 不必默认写入 memory，也不必一定转成新 skill。
   - ExPolicy 应根据当前 agent 状态和 experience 特征，动态决定对它执行什么 capability intervention，例如：
     - discard / ignore
     - episodic memory
     - semantic memory / knowledge
     - retrieve / activate existing skill
     - edit / compose / promote skill
     - update harness / procedure
     - weight update
     - 其它可持久能力更新
   - 核心问题是：  
     **Given an acquired experience, what should the agent do with it?**

因此，ExPolicy 的总体定义是：

\[
\boxed{
\text{Experience Policy}
=
\text{Acquisition Policy}
+
\text{Assimilation Policy}
}
\]

更一般地，可写成：

\[
\pi_{\mathrm{ExPolicy}}
(a_t^{\mathrm{acq}}, a_t^{\mathrm{assim}}
\mid S_t, H_t, e_{1:t})
\]

其中 \(S_t\) 表示当前 capability / persistent state，\(H_t\) 表示交互历史，\(e_{1:t}\) 表示已经获得的 experience；policy 的目标是最大化**未来任务分布上的能力增长**，而不是当前一步的 reward。

主线 pipeline 应始终理解为：

```text
Current Agent State
        ↓
Capability Diagnosis
        ↓
ExPolicy: choose what experience to acquire
        ↓
Real Environment Interaction
        ↓
Observed Experience / Trajectory / Evidence
        ↓
ExPolicy: choose how to use the experience
        ↓
discard / memory / existing skill / skill edit / harness / weight / ...
        ↓
Updated Persistent Capability State
        ↓
Held-out Future Capability Evaluation
        ↓
repeat
```

### 研究边界

- **Skill 只是 assimilation action / carrier 之一，不是 ExPolicy 本身。**
- ExPolicy **不要求**证明每个 executable skill 都必须由当前 experience 内容从零诱导出来；必要时可以直接使用现有工作提供的、人工构造的或预训练好的 skill bank。
- `experience → unique SkillSpec` 的内容因果性是一个可研究的子问题，但**不是 ExPolicy 主线成立的前置条件**。
- 当前最重要的问题不是继续证明 skill 是否“真由经验生成”，而是构造一个真正**可学习、动态、跨环境泛化**的 Experience Policy：
  1. 能判断**下一步值得获得什么经验**；
  2. 能根据 agent 当前状态和新经验，决定**如何利用该经验**；
  3. 最终在相同交互/计算预算下，比 passive acquisition、固定 assimilation 或简单 heuristic 获得更高的 future capability gain。

### 当前实验结果应如何服务主线

现有 R1–R6 与 SkillRL-ESI 的作用，是逐步确认 ExPolicy 所需的底层条件，而不是把项目收缩成 skill induction：

- R1–R4：确认普通 episodic / semantic / LoRA assimilation 很弱，而 executable procedure 是当前可工作的 capability intervention。
- R5：在有效 executable-skill backend 下，证明“采什么 experience”确实可能改变 future capability。
- R5.1：发现 Helion 上的主要机制更接近 experience-gated skill activation，而不是深度 content assimilation。
- R5.2：证明 Helion frontier acquisition 并非 Ace-Skill generic prioritized sampling 的简单换壳。
- R5.3–R6：说明现有 frontier / RASA / AVA acquisition heuristic 缺乏公开环境上的稳定泛化，因此**真正未解决的是 Experience Policy 本身**。
- SkillRL-ESI：进一步确认 executable skill 可以作为有效 assimilation action；empty/shuffled 对照未证明 experience-content causality，但这不应阻塞 ExPolicy 主线。
- R7：把 Experience Policy 做成真实 closed-loop controller（acquisition + assimilation + shadow reward + posterior update）。闭环打通（PIPELINE_PASS）；第一版 Linear TS **未**超过 CountOnly / all-skills 上界（METHOD_AMBIGUOUS）。
- R8：先证明 decision space 可辨识，再训练 controller。R7 frozen-mask SearchEnv **结构退化**（`R7_ENV_STRUCTURALLY_DEGENERATE`）；ALFWorld 上官方 MemRL/SkillOpt 在冻结 3B 未过 foundation（smoke success=0）。**未训练 ExPolicy。**

因此，后续项目的核心目标重新统一为：

> **Existing agents learn from experience; ExPolicy learns how to experience and how to learn from it.**

也即：**让 agent 学会管理自己的学习过程，而不是固定地采经验、固定地存经验、固定地调用某一种学习机制。**



> Canonical log for the mechanism-first roadmap (`ExPolicy_Next_Experiment_Roadmap.md`)、assimilation-repair round (`ExPolicy_Next_Run_Instructions.md`)、R5 (`ExPolicy_Next_Run_Instructions_R5.md`)、R5.1 (`ExPolicy_Next_Run_Instructions_R5_1_H20_GPU4-7.md`)、R5.2 (`ExPolicy_R5_2_Strong_Baseline_Collision_Instructions.md`)、R5.3 (`ExPolicy_R5_3_External_Generalization_Instructions.md`)、R5.4 (`ExPolicy_R5_4_RASA_Redesign_Run_Instructions.md`)、R6 (`ExPolicy_R6_AVA_Next_Run_Instructions.md`)、SkillRL-ESI MVP (`SkillRL_ESI_MVP_8xH20_Execution_Instructions.md`)、**R7 Closed-Loop MVP** (`ExPolicy_R7_Closed_Loop_MVP_H20_Instructions.md`) 以及 **R8 Structural Redesign** (`ExPolicy_R8_Next_Run_Instructions.md`)。
> **状态以本节「本轮总览」为准。** 每个实验写清 Setting / 结果 / 结论；未跑完的不写成已验证。

---

## 本轮总览（更新于 2026-08-21）

| 实验                                    | 状态                                    | 结论                                                         |
| --------------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| E0 Capability-Gap Audit                 | **完成**                                | overall EM=57.3%，error=42.7% → **IDEAL_FRONTIER**；S1 为硬缺口（pass@4=0），S4 为不稳定（pass@4=0.75） |
| E1 Realized Utility                     | **完成（机制弱阳性）**                  | 单条 episodic 干预无法在 8-task 远迁移 probe 上得到正 U；ordinary failure 的 future_gain 高于 success。**Gate A 未干净通过。** |
| E2 Active Acquisition Pilot             | **完成（当时 Gate B 未通过）**          | 同 budget=24、固定 episodic+semantic 下，5 种 acquisition 的 large-probe future_gain 全 ≤0。**应读成无效 assimilation 后面的评测，已被 R5 修订。** |
| E3 Carrier Utility Matrix               | **E3a+E3b 完成（Gate C 弱/未过）**      | E3b：12×3，3/12 有 U>0；unique best = episodic 4 / semantic 3。mean U 仍全负。 |
| E4 Learned Assimilation                 | **诊断完成（不训练）**                  | heuristic regret=0.083 = random，差于 always-episodic（0.036）。不训练 learned router。 |
| E5–E8 / R6 learned carrier              | **仍不做**                              | R6 Exact AVA 未过 Gate A/B/C；按 CASE_C **不进入 surrogate / 10-seed / 第二 assimilator**。 |
| **R1 Near-Transfer Utility**            | **完成（Go 未过）**                     | 仅 S4 new-episodic 平均 near gain>0（+0.104，2/3 batch）；S1/S2 为负。hybrid 未优于 old episodic。 |
| **R2 Executable Skill**                 | **完成（三 skill 均 accept）**          | executable near gain：S1 +0.562 / S2 +0.438 / S4 +1.000；old prompt skill 无正 near。retention 大跌（procedure 绑死 capability） |
| **R3 Weight Batch**                     | **完成（LoRA 弱/未过）**                | LoRA U：S1 −0.062 / S2 −0.062 / S4 +0.094。正 U 仍主要来自 executable skill。 |
| **R4 Batch Carrier Matrix**             | **完成（字母 GO，机制有保留）**         | 11/12 batch 有 U>0；skill mean U=+0.237，几乎全局最优（11/12）。oracle 仅比 always-skill 高 0.003。episodic/weight 仍多为负。 |
| **R5 Active Acquisition × Skill**       | **完成（Gate B PASS）**                 | 在 experience-conditioned executable skill 后，frontier mean ΔFuture=+0.519（3 seed），比 passive 高 +0.296。uncertainty / failure_replay 仅略高于 passive（幅度 < 1 题）。reward_seeking Δ=0。不证明 learned routing。 |
| **R5.1 Robustness & Causal Validation** | **完成（Case B）**                      | 10-seed frontier−passive = +0.292，CI 不跨 0，10/10 胜出。RealFit ≉ 强于 CountOnly（0.519 vs 0.546）→ **experience-gated skill activation**，不是内容内化。Helion 上 text-router = oracle。temporal / capnorm 仍采不到 S2。当时不进入路线图 R6 learned carrier（后来的 Exact AVA 见下表 R6）。 |
| **R5.2 Strong-Baseline Collision**      | **完成（Case A）**                      | 同 CountOnly unique-task 后端、10 seed：frontier−Ace-Skill = **+0.306**（CI 点质量、**10/10**）。Ace-Skill = passive（gap=0）。AULC 与第二 skill 激活均 frontier 更快。不是 Ace-Skill 类 generic prioritized sampling。不继续堆 Helion；进入第二 benchmark。 |
| **R5.3 External Generalization**        | **完成（Case C，3-seed stop）**         | 公开 HotpotQA / 2Wiki / MuSiQue + Wikipedia 2018 E5。Skill backend 3/3 accept（C2 ΔEM +0.417），但冻结 frontier **未泛化**：ΔFuture=−0.0067 vs Ace +0.081 / passive +0.082；0/3 seed 赢；48/48 全采 HotpotQA C1。Stage-1 未过，**不扩 10 seed、不跑 Bamboogle、不调系数**。Helion 结果视为 Helion-specific heuristic。 |
| **R5.4 RASA Redesign**                  | **完成（Case E，3-seed stop）**         | 丢弃 Helion frontier，在**新冻结公开 holdout**上测 RASA（`q×p` acquisition + Sequential Readiness Test）。`rasa_full` mean macro=**0.111**，低于 Ace/passive **0.120**（0/3 胜，差 −0.009）。SRT 正确 ACTIVE C2、PARK C1，但 C3 `P_good=0.105` 未过 PARK 线，剩余 ~35/48 预算砸在几乎不可修的 long-chain。Gate A–E 未过，仅 F 过。**不扩 10 seed、不跑 Bamboogle、不改阈值。RASA 主线证伪。** |
| **R6 Exact AVA Stage-1**                | **完成（CASE_C，3-seed stop）**         | Novelty `PROCEED`；pipeline audit PASS；task-level `q×p` Spearman=0.29。冻结 CountOnly 上 exact shadow utility **不能**超过 Ace/passive：`exact_ava` mean=**0.120** vs best baseline **0.119**（+0.001，远小于 Gate A 的 +0.02）；seed 胜 **1/3**。最终分数几乎完全由激活 mask 决定（C1+C2+C3=0.120，C2-only=0.111）。**不训 surrogate、不改 M/K/shadow、不缩 claim。停止 ExPolicy acquisition 主线。** |
| **SkillRL-ESI MVP**                     | **完成（CASE_ESI_PASS，内容因果未过）** | 在冻结 SkillRL Search + Qwen2.5-3B 上，train experience → typed executable SkillSpec → SearchEnv 执行。future_test **ESI 0.3933 vs B0 0.2867（Δ=+0.107，3/3 seed）**，过 +0.03。但 **empty 0.46 ≥ ESI**，shuffled 0.40；增益来自 comparison 执行程序，不是这份 train packet 的独特内容。B1 native textual 在 test 上 **更差**（0.1933）。下一步做 skill induction/editing，**不重开 acquisition**。 |
| **R7 Closed-Loop ExPolicy**             | **完成（PIPELINE_PASS，METHOD_AMBIGUOUS）** | 第一次把 acquisition → 真实 SearchEnv → assimilation → persistent state → shadow reward → policy update 整圈跑通。Gate P 六项 PASS。第一版 Linear Thompson **未**超过 strongest non-oracle：M0 mean **0.047** vs CountOnly / all-skills **0.091**。增益仍由是否激活 C2 frozen skill 决定。Case B：闭环成立，controller 不够好。不回头改 skill / split / CountOnly K。 |
| **R8 Structural Redesign**              | **完成（R7 环境退化 + ALFWorld foundation FAIL；未训 controller）** | Novelty `PROCEED`。R8-A 全量 83/83：**`R7_ENV_STRUCTURALLY_DEGENERATE`**（oracle_joint 0.064 < CountOnly 0.091；S2 天花板 gap=0；unique-best≥20% 的动作只有 ACTIVATE）。R8-B split 冻结 PASS；官方 MemRL/SkillOpt 无源码 patch。3B no-learning smoke **success=0**（none 22 局 / SkillOpt 10 局）；MemRL smoke 卡在 MOS 初始化；120 局 foundation 16/120 后挂死。Gate H / Stage-1 **未跑**。**不在 R7 SearchEnv 换更强 bandit，不把项目改写成 SkillOpt 调度器。** |

### Setting（共用）

| 项                            | 值                                                           |
| ----------------------------- | ------------------------------------------------------------ |
| 机器                          | 本机 8×H20-3e。R1–R4 用空闲卡 **2 / 3 / 6**；R5 / R5.1 / R5.2 / R5.3 / R5.4 / **R6** 用 **4 / 5 / 6 / 7**（不抢占卡上已有 openpi-comet / curobo）。R6：`scripts/orchestrate_r6_ava.py` + 四卡 wrapper + `scripts/watchdog_r6_ava.py`。**SkillRL-ESI** 同样只用 **4 / 5 / 6 / 7**（0–3 被 casbot/openpi 占满）；retriever encoder 在 GPU 4。**R7** 默认 GPU **5/6/7** 三 seed，GPU **4** 仅在空余显存足够时并行 B4 all-skills；0–3 未占用。**R8-A** 四卡 worker GPU **4/5/6/7**；**R8-B** vLLM Qwen2.5-3B 占 GPU **4**（`gpu_memory_utilization=0.18`，端口 **18090**），embedder CPU 端口 **18091**；0–3 未占用 |
| repo                          | `/data/ppnm/expolicy`（R1–R6 / **R7** branch `r7-closed-loop`，base git `5c9dbf22`）。**SkillRL-ESI 主代码** `/data/ppnm/skillrl_esi`（SkillRL `8e66726`，branch `esi-mvp`）；旧 ExPolicy 只作 protocol / split 参考，不导入 frozen primitive |
| model                         | `Qwen2.5-3B-Instruct` @ `/data/ppnm/models/Qwen2.5-3B-Instruct`（ESI / **R7** / **R8** 全程冻结，无正式 RL；eval `temperature=0`） |
| retrieval（R1–R5.2）          | 本地 Helion BM25（`data/world/helion.json`，world seed=13）  |
| retrieval（R5.3 / R5.4 / R6 / **R7**） | Wikipedia 2018 + E5-base-v2 本地 dense retriever（21,015,324 docs；端口 **18080** CPU；无 mock / 无 gold-as-hit） |
| retrieval（SkillRL-ESI）      | SkillRL `retrieval_server.py` 端口 **8030**（同一 wiki-18 E5 HNSW 资产，FAISS CPU；schema 为 SkillRL `query`，不用旧 18080） |
| retrieval / env（**R8-B**）   | ALFWorld TextWorld `/data/ppnm/data/alfworld`；MemRL 本地 padded MiniLM embedder 端口 **18091**（384→3072 零填充，避免改官方 `vector_dimension`）；不用 Wikipedia retriever |
| 禁止                          | mock LLM / 伪造检索 / 把失败改写成成功；R5.1–R5.4 不覆盖 `outputs/r5_acquisition_skill/`；R5.3 不调 frontier 系数 / K / 不重抽 split；R5.4 不改 SRT 阈值 / 不重抽 holdout / Case E 后不扩 Stage-2；**R6 Exact AVA 未过则不训 surrogate、不改 M/K/shadow、不缩 claim**；**ESI 不绕过 SearchEnv、不用 CountOnly、不用 test 调参、不换 split seed**；**R7 不把 CountOnly/frontier 当主方法、final future_test 不进 policy update、不因 Method Gate 失败回头改 pipeline 定义**；**R8 不在 R7 frozen-mask SearchEnv 上只换更强 bandit、不魔改 MemRL/SkillOpt 源码后称为原方法、foundation <10% 不静默换 prompt/换大模型、Gate H 未过不训 ExPolicy、不把论文缩成 SkillOpt 调度器** |

---

## 2026-08-13 E0 Capability-Gap Audit

### Setting

- 目的：判断 Helion 对 3B static agent 是否仍有 25–45% 的 frontier/error 空间
- 不训练；greedy T=0
- sample seed=0，按 split 分层；S2/S3/S4 题量不足 80，用该 stage 全量
- world seed=13；Qwen2.5-3B-Instruct；GPU 5/6/7 并行
- 产物：`outputs/e0_audit/`

### 结果（static）

| Stage                     |       n |        EM | Search miss | Open miss | Reason fail | Synth fail | Search Recall | Avg tools |
| ------------------------- | ------: | --------: | ----------: | --------: | ----------: | ---------: | ------------: | --------: |
| S0 single-hop             |     100 |     0.930 |       0.000 |     0.010 |       0.050 |      0.020 |          1.00 |      2.83 |
| S1 multi-hop / decompose  |     100 |     0.160 |       0.040 |     0.170 |       0.510 |      0.330 |          0.50 |      5.92 |
| S2 temporal / disambig    |      50 |     0.400 |       0.020 |     0.020 |       0.360 |      0.240 |          0.98 |      4.98 |
| S3 conflict / reliability |      48 |     1.000 |       0.000 |     0.000 |       0.000 |      0.000 |          1.00 |      3.04 |
| S4 recovery / synthesis   |      49 |     0.449 |       0.000 |     0.000 |       0.041 |      0.245 |          0.69 |      3.41 |
| **Overall**               | **347** | **0.573** |           — |         — |          76 |         59 |      **0.81** |  **4.14** |

Capability 细目：

- `query_decomposition` EM=0.00（n=27）；`multi_hop` EM=0.22（n=73）
- `cross_document_synthesis` EM=0.04（n=24）；`failure_recovery` EM=0.83（n=24）
- `conflicting_evidence` / `source_reliability` 全对（prompt 已写“prefer official”）
- S4 `success`=0.71 > `EM`=0.45：contains-match 放宽了 `X; Y` 合成题

pass@4（T=0.7，每 stage 最多 8 道 greedy 错题，n=31；S3 无错题）：

| Stage   | n_fail sampled |   pass@4 | hard-gap (pass@4=0) | 解读                                    |
| ------- | -------------: | -------: | ------------------: | --------------------------------------- |
| S0      |              7 |     0.43 |                0.57 | 少量不稳定                              |
| S1      |              8 | **0.00** |            **1.00** | **真实能力缺口**（4 次采样仍全错）      |
| S2      |              8 |     0.12 |                0.88 | 以真实缺口为主                          |
| S4      |              8 | **0.75** |                0.25 | **不稳定能力**，适合 active acquisition |
| Overall |             31 |     0.32 |                0.68 | 约 1/3 不稳定、2/3 硬缺口               |

### 结论

- **Gate（roadmap §1.4）判定：`IDEAL_FRONTIER`。** overall error=42.7%，落在 25–45% 目标带，**不需要立刻整库加难，也不是不可学。**
- 学习空间不在 S0/S3（已掌握 / 被 system prompt 解掉），而在 **S1 multi-hop、S2 temporal、S4 cross-doc**。
- pass@4 把两类缺口切开了：
  - **S1（及大部分 S2）= real capability gap** → 更适合 skill / weight / 结构化 procedure，而不是再采样同一分布。
  - **S4 = unstable capability** → 最适合 active experience acquisition。
- S1 的主要瓶颈不是“完全搜不到”（search miss 仅 4%），而是 **部分 gold 召回 + 桥接推理**。
- S2/S3 题量偏小（48–50），后续主实验若要 stage 均衡需要扩 world，但不阻塞 E1–E3。
- **继续 E1。**

---

## 2026-08-13 E1 Realized Utility Sanity Check

### Setting

- 从 E0 轨迹抽 32 条：success / ordinary_failure / frontier_failure / repeated_pattern 各 8
- 固定 apply：每条写入 **episodic**；成功经验额外写 **semantic**。不做 LoRA / harness
- 固定 probe：future=8 道 S1 `future`（multi-hop/decomp）；retention=8 道 S0 `retention`
- 公式：`U = future_EM_gain - 0.5 * retention_EM_drop - tokens/1e6`
- agent_v0 先评 probe 一次，再逐条 clone+apply+复评
- 产物：`outputs/e1_utility/E1_SUMMARY.json`、`utilities.jsonl`、`E1_SENSITIVITY.json`

### 结果

| 量                            |           值 |
| ----------------------------- | -----------: |
| n experiences                 |           32 |
| before future EM              |        0.125 |
| before retention EM           |        0.875 |
| **Positive U rate**           |     **0.00** |
| Mean U                        |       −0.160 |
| Median U                      |       −0.133 |
| U variance                    |       0.0049 |
| Mean future_gain              |       −0.031 |
| Positive future_gain rate     | 4/32 = 0.125 |
| U without cost, positive rate | 4/32 = 0.125 |

| Bucket               |     mean U | mean U (no cost) | mean future_gain | #gain>0 |
| -------------------- | ---------: | ---------------: | ---------------: | ------: |
| success              |     −0.174 |           −0.055 |           −0.047 |     0/8 |
| **ordinary_failure** | **−0.093** |       **+0.039** |       **+0.047** | **3/8** |
| frontier_failure     |     −0.180 |           −0.062 |           −0.062 |     0/8 |
| repeated_pattern     |     −0.193 |           −0.070 |           −0.062 |     1/8 |

单条干预的评测成本约 1e5 tokens → cost 项 ≈0.12，与 8-task probe 的 1/8=0.125 EM 同量级，所以 **U 被成本项整体压成负数**。

### 结论

- **Gate A（utility heterogeneity）未干净通过。** 在「单条经验 + episodic/semantic + 8 题远迁移 probe」下，realized utility 没有可用的正值分布，也谈不上 top-20% 贡献累计能力增益。
- 仍有两个弱信号，值得带到 E2/E3，但不能当成论文主结果：
  1. **success 的 future_gain 全 ≤0**；ordinary failure 是唯一 mean future_gain>0 的桶（失败经验 ≥ 成功经验，符合 Hypothesis E 的弱形式）。
  2. **frontier_failure 并未高于 ordinary_failure**——与“frontier 一定更值”的预期不符；S1 硬缺口（pass@4=0）靠一条记忆无法解开。
- 工程含义：后续必须改成 **batch experience（E2）** 和 **carrier 对照（E3）**，不能再把“一条 episodic”当成 assimilation 的代表。
- **不宣称 Active Acquisition 已被证明。** 进入 E2 时固定同一套 batch memory assimilation，只换 acquisition 策略。

---

## 2026-08-13 E2 Active Acquisition Pilot

### Setting

- 目的：同 experience budget 下，active 是否比 passive 带来更高 future capability gain（Gate B）
- 固定 assimilation：每条经验写 **episodic**；成功经验额外写 **semantic**。不做 LoRA / harness / skill
- 固定 budget：**24** 条 acquired experience（5 种方法同一 pool、同一 seed=0）
- 方法：`passive` / `frontier` / `reward_seeking` / `uncertainty` / `failure_replay`（learned ranker 因尚无 U 监督而未跑）
- 模型：Qwen2.5-3B-Instruct；GPU 5/6/7
- 小 probe（采集脚本内）：future=8、retention=8，仅作 sanity，**不作为 Gate B 判据**
- **Gate B 复评**：`scripts/eval_e2_large_probe.py`，加载各方法已保存的 `agent_v001` memory，在 **24 future + 16 retention** 上只评测、不再采集
  - future：S1/S2/S4 的 future+validation，seed=99
  - retention：S0 retention+validation，seed=99
- 产物：`outputs/e2_acquisition/{method}/`、`outputs/e2_acquisition/E2_LARGE_PROBE.json`
- 已知修复：第一轮 `failure_replay` 把 failure **type** 误当成 capability，24 题全变成 `single_hop`；已改按 recent 失败 capability + frontier + 低 capability_score 选择，并在 GPU 5 重跑。下文只用修复后结果。

### 结果 1：选择行为（budget=24）

| method                   | immediate reward | 选中 capability 分布                                         | tokens | tool calls |
| ------------------------ | ---------------: | ------------------------------------------------------------ | -----: | ---------: |
| reward_seeking           |        **0.917** | 15 single_hop + 5 source_reliability + 4 conflicting_evidence | 135120 |         80 |
| passive                  |            0.708 | 12 single_hop + 5 multi_hop + 3 conflicting_evidence + 2 temporal + 1 recovery + 1 reliability | 134357 |         81 |
| frontier                 |            0.250 | **24 全是 multi_hop**                                        | 231762 |        134 |
| failure_replay（修复后） |            0.250 | **24 全是 multi_hop**                                        | 240123 |        139 |
| uncertainty              |            0.125 | 12 multi_hop + 11 query_decomposition + 1 entity_disambiguation | 262357 |        167 |

小 probe（n=8）上 future_gain 为 −0.125/−0.250，与 1/8 粒度同量级，只反映噪声，不写入 Gate B。

### 结果 2：Large probe 复评（Gate B）

base：future EM=0.208（n=24），retention EM=1.000（n=16）

| method         | future EM | **future_gain** | retention EM | retention Δ |
| -------------- | --------: | --------------: | -----------: | ----------: |
| reward_seeking |     0.208 |       **0.000** |        0.875 |      −0.125 |
| passive        |     0.167 |          −0.042 |        0.875 |      −0.125 |
| frontier       |     0.167 |          −0.042 |        0.875 |      −0.125 |
| failure_replay |     0.167 |          −0.042 |        0.938 |      −0.062 |
| uncertainty    |     0.125 |      **−0.083** |        0.938 |      −0.062 |

24 题上 1 题翻转 = 0.042，上表差异全部落在 ±1–2 题。所有方法 retention 均相对 ceiling 下降（memory 干扰 S0）。

### 结论

- **Gate B（Active > Passive，且有稳定 future gain）未通过。** 在固定 episodic+semantic、budget=24 下，没有任何 active 方法相对 base 得到正的 future_gain；frontier / failure_replay / uncertainty 也不优于 passive。
- 这与 E1 一致：当前 assimilation 载体（episodic 近邻记忆）对 S1 硬缺口几乎没有远迁移。**acquisition 的差异还没机会变成 capability 差异。**
- 仍有一个定性、可写进论文动机但不能当主表的结果：
  - **Reward-seeking 的 immediate reward 最高（0.92），却几乎只采 S0/S3 易题；**
  - **Frontier / failure_replay 专采 multi_hop，immediate reward 仅 0.25；**
  - 即 **当前任务 reward ≠ 学习价值的选择**，但「学得更多」尚未被 future EM 证实。
- **不进入 E5 2×2。** 下一步仍按路线图做 E3（换 carrier，而不是继续加大 acquisition budget）。若 E3 表明 skill/weight 才能吸收 S1 失败，再回头用该 carrier 重做一次 acquisition-only，而不是现在宣称 Gate B 成立。
- **后记（2026-08-14）：** R4 确认 executable skill 才能吸收经验后，R5 用该 carrier 重测 Gate B 为 **PASS**。E2 本节数字仍有效，但不再作为最终 Gate B；见「R5 Active Acquisition under Validated Executable Assimilation」。

---

## 2026-08-13 E3a Carrier Utility Matrix（pilot，skip-weight）

### Setting

- 目的：同一条经验写入不同 carrier，realized utility 是否不同（Gate C）
- n=8 experiences，从 E0 轨迹抽样，seed=0
- carriers：episodic / semantic / skill；**本轮 skip LoRA weight**（单条 SFT 成本高，且 E3a 先看 memory/skill）
- harness 关闭
- probe 各 6 题：near=S1 validation，**far=S3 future**，retention=S0 retention
- U = Δnear + 0.5 Δfar − 0.5 max(0, −Δretention)
- 产物：`outputs/e3_carrier_matrix/`（`E3_SUMMARY.json`、`matrix.jsonl`）
- 墙钟：1055s，GPU 6

### 抽样问题（事后发现）

E0 经验的 failure_type **只有** `reasoning_failure` / `answer_synthesis_failure`，没有 retrieval/planning。原抽样按空桶填充，最终 8 条里 **6 成功 / 4 条 single_hop**，覆盖不足。

### 结果

before：near EM=0.167，**far EM=1.000（S3 触顶）**，retention EM=1.000。probe 粒度 = 1/6 ≈ 0.167。

| experience    | cap              | success | episodic U | semantic U | skill U | reported best                |
| ------------- | ---------------- | ------- | ---------: | ---------: | ------: | ---------------------------- |
| t_sh_spec_034 | single_hop       | Y       |      0.000 |     −0.167 |   0.000 | episodic（与 skill 并列）    |
| t_mh_ceo_030  | multi_hop        | N       |     −0.167 |     −0.167 |  −0.333 | episodic（与 semantic 并列） |
| t_mh_city_021 | multi_hop        | Y       |     −0.167 |     −0.167 |  −0.083 | **skill**                    |
| t_xd_evt_008  | cross_doc        | N       |     −0.250 |     −0.167 |  −0.167 | semantic（与 skill 并列）    |
| t_fr_evt_003  | failure_recovery | Y       |     −0.417 |     −0.167 |  −0.167 | semantic（与 skill 并列）    |
| t_sh_spec_068 | single_hop       | Y       |     −0.167 |     −0.250 |   0.000 | **skill**                    |
| t_sh_hq_020   | single_hop       | Y       |     −0.167 |     −0.250 |  −0.083 | **skill**                    |
| t_sh_hq_011   | single_hop       | Y       |      0.000 |     −0.167 |   0.000 | episodic（与 skill 并列）    |

| carrier  |                    mean U |
| -------- | ------------------------: |
| skill    | **−0.104**（global best） |
| episodic |                    −0.167 |
| semantic |                    −0.188 |

脚本报告 `fraction_best_differs_from_global=0.625`，但 5/8 的 “best” 是 **U 并列时取 dict 插入顺序**，不是唯一最优。唯一非并列最优只有 3 条，且全是 skill。**没有任何 carrier 对任何经验给出 U>0。**

### 结论

- **Gate C 未干净通过。** 不同 carrier 确实会产生不同（多为负）效果，skill 平均伤害最小，但这更像「skill 经常接近 no-op / 少干扰」，而不是「选对 carrier 能学会」。
- 两个设定缺陷使本轮不能当作 Carrier Matrix 主结果：
  1. **far=S3 已 100% EM**，远迁移项恒为 0 或只会下降；
  2. 经验覆盖偏成功/single-hop，且 n=6 probe 噪声 = 整档 U。
- **不训练 E4 learned router。** 用负 U + 并列标签训练 router 没有科学意义。
- **E3b**：修正分层抽样（失败/难 capability），far 改为 S2（E0 EM=0.40），probe=8，n=12，仍 skip-weight。产物目录 `outputs/e3b_carrier_matrix/`，不覆盖 E3a。

---

## 2026-08-13 E3b Carrier Utility Matrix（修正 probe / 抽样）

### Setting

- 相对 E3a 的修正：far **S2**（E0 EM=0.40，有升降空间）；分层抽样覆盖 success_easy / success_hard / reasoning_hard / synthesis；记录并列 best
- n=12，probe=8，seed=1，skip-weight，harness off
- near=S1 validation+future；far=S2 future+validation；retention=S0
- U = Δnear + 0.5 Δfar − 0.5 max(0, −Δretention)
- GPU 5，墙钟 2518s
- 产物：`outputs/e3b_carrier_matrix/`

抽样：6 成功 / 6 失败；capability = temporal×4, multi_hop×4, single_hop×2, source_reliability×1, query_decomposition×1。

### 结果

before：near=0.25，far=0.25，retention=0.875。粒度 1/8=0.125。

| task          | cap          | ok   |      epi U |      sem U |    skill U | unique / tied best |
| ------------- | ------------ | ---- | ---------: | ---------: | ---------: | ------------------ |
| t_sh_hq_022   | single_hop   | Y    | **+0.062** |     −0.062 | **+0.062** | tie epi/skill      |
| t_tm_new_021  | temporal     | Y    |     −0.125 |     −0.188 |     −0.125 | tie epi/skill      |
| t_mh_ceo_018  | multi_hop    | N    |      0.000 |     −0.125 |     −0.250 | **episodic**       |
| t_tm_old_001  | temporal     | N    |     −0.375 | **−0.188** |     −0.250 | **semantic**       |
| t_tm_new_003  | temporal     | N    |     −0.188 | **−0.062** |     −0.312 | **semantic**       |
| t_tm_new_017  | temporal     | Y    |     −0.188 |     −0.188 |     −0.312 | tie epi/sem        |
| t_mh_ceo_040  | multi_hop    | N    |      0.000 |     −0.125 |     −0.188 | **episodic**       |
| t_mh_ceo_057  | multi_hop    | Y    | **+0.125** |     −0.125 |     −0.125 | **episodic**       |
| t_sr_ceo_003  | reliability  | Y    |     −0.062 | **+0.062** |     −0.250 | **semantic**       |
| t_mh_ceo_020  | multi_hop    | N    |     −0.125 |     −0.125 |     −0.312 | tie epi/sem        |
| t_qd_adv_056  | query_decomp | N    |      0.000 |     −0.062 |     −0.125 | **episodic**       |
| t_sh_prod_022 | single_hop   | Y    |      0.000 |     −0.062 |      0.000 | tie epi/skill      |

| 量                               | 值                              |
| -------------------------------- | ------------------------------- |
| mean U episodic                  | **−0.073**（global best fixed） |
| mean U semantic                  | −0.104                          |
| mean U skill                     | −0.182                          |
| n with any U>0                   | **3 / 12**                      |
| unique best                      | episodic 4，semantic 3，skill 0 |
| tied best                        | 5 / 12                          |
| best-set 不含 global（episodic） | **0.25**                        |

定性模式（样本小，只作假设）：

- **temporal 失败 → semantic 相对最好**（仍多为负，但是伤害更小）
- **multi-hop → episodic 相对最好**（常为 0，其它更负）
- skill 平均最差，常把 near 打低 1–2 题

### 结论

- 相对 E3a，本轮 **probe 可动、抽样覆盖失败/难任务、出现 3 条正 U**，Carrier Matrix 第一次有可用信号。
- **Gate C 仍未干净通过。** Q1（best 随 experience 变）只有弱证据：7 条 unique 里 3 条 semantic ≠ global episodic（25%）。Q2（短期好长期差）无稳定 weight 对照。Q3（carrier promotion）无证据。
- mean U 全负：当前三条 memory/skill 载体对 3B+Helion **仍以干扰为主**，不是“选对就能学会”。
- **本轮不做 LoRA weight 列。** 单条 success 的 5–20 step SFT 不是 weight carrier 的设定语义；且 `PeftModel.from_pretrained` 叠加载、评测后不卸载，现有 E3 runner 对 weight 不安全。要做 E3c 需先修 unload/rebuild。
- **不把 25% 差异写成论文主结果。** 进入 E4 只做无 GPU 的 routing-regret 诊断，不训练 router。

---

## 2026-08-13 E4 Routing Regret 诊断（无训练、无新 rollout）

### Setting

- 目的：在已采集的 E3b 矩阵上，比较 Oracle / Heuristic / Fixed-carrier / Random 的 routing regret
- **不训练** Experience Policy MLP；不做新的 LLM 评测
- Heuristic = 代码里的 `HeuristicRouter`（失败→episodic，procedural 成功→skill，…）
- 若 heuristic 选出 weight/harness/discard（本矩阵没有），将 U 记为可用 carrier 的 min（本轮未发生，unavailable_rate=0）
- 产物：`outputs/e4_routing_regret/E4_SUMMARY.json`

### 结果

| 策略                 | mean realized U | mean regret vs oracle |
| -------------------- | --------------: | --------------------: |
| Hindsight oracle     |          −0.036 |                 0.000 |
| **Fixed episodic**   |      **−0.073** |             **0.036** |
| Fixed semantic       |          −0.104 |                 0.068 |
| Random（三载体均匀） |          −0.120 |                 0.083 |
| Heuristic router     |          −0.120 |                 0.083 |
| Fixed skill          |          −0.182 |                 0.146 |

Heuristic 选出：episodic 7、skill 5。它把 5 条成功经验送进 skill，而 skill 是平均最差载体，所以 heuristic = random，**明显差于 always-episodic**。

### 结论

- 路线图要求 adaptive routing **降低 regret**。在当前矩阵上 **heuristic 没有降低 regret**。
- 因此 **不训练 learned router（E4 正式实验）**：监督信号弱（mean U<0，正样本 3/12，幅度=1 道 probe），现成 heuristic 已经比不过固定 episodic。
- **Gate C 第二句不成立。** 停止进入 E5–E8。

---

## 2026-08-13 R1 Near-Transfer Utility Diagnostic（完成）

### Setting

- 目的：在 **同 capability、同桥接模式、不同实体/文档** 的 near-transfer probe 上，检验 batch episodic 能否产生正的 realized utility。不改 acquisition，不训练 policy。
- 经验：从 E0 真实轨迹抽 **3 capability × 3 batch × 8 = 72** 条
  - `s1_multi_hop`（`multi_hop` / `mh_ceo`+`mh_city`）
  - `s2_temporal`（`temporal_reasoning` / `tm_old`+`tm_new`）
  - `s4_cross_doc`（`cross_document_synthesis` / `xd_evt`）
- 每个 batch 的 probe：**near=16 / far=16 / retention=16**。near 禁止复用原题、相同 gold 答案或相同 gold document pair。
- 对照：`no_memory` / `old_episodic`（question embedding + Q→A 注入）/ `new_episodic`（hybrid score：semantic + capability + failure + pattern + recency；结构化 procedure card，不含答案/doc id）
- 模型：Qwen2.5-3B-Instruct；GPU 2/3/6 分 bucket 并行；seed=0
- 产物：`outputs/r1_near_transfer/{s1_multi_hop,s2_temporal,s4_cross_doc}/`
- Go 条件：至少 2/3 bucket 的平均 near gain > 0，且跨 batch 方向一致
- 墙钟：S1 1550s / S2 1702s / S4 1500s

### 实现要点

- `scripts/build_near_transfer_probes.py` 写出 9 个满额 probe set（每组 16/16/16）。
- Episodic write 保存 question / pattern / capability / failure / bridge / queries / evidence path / correction；retrieve 为 hybrid。
- 三个 executable skill 的 **无 LLM 烟测**（控制 runtime、从打开文档抽 span，不写死 gold）：multi-hop n=144 EM=0.986；temporal n=48 EM=1.000；cross-doc n=24 EM=1.000。
- R3 数据：`outputs/r3_weight_batch/verified_index.json`（S1=64，S2=48，S4=24）。

### 结果 1：逐 batch near_transfer_gain（相对 no_memory；粒度 1/16=0.0625）

| batch | no_mem near |  old Δnear |  new Δnear | old Δfar | new Δfar | old Δret | new Δret |
| ----- | ----------: | ---------: | ---------: | -------: | -------: | -------: | -------: |
| s1_b0 |       0.188 | **+0.062** |     −0.188 |   −0.062 |   −0.188 |   +0.062 |   −0.188 |
| s1_b1 |       0.125 | **+0.125** |     −0.062 |   −0.062 |   −0.062 |   −0.062 |   −0.125 |
| s1_b2 |       0.062 |     −0.062 | **+0.062** |   −0.188 |   −0.250 |   +0.062 |   −0.125 |
| s2_b0 |       0.500 |     −0.125 |     −0.375 |   −0.062 |   −0.062 |   +0.125 |   +0.125 |
| s2_b1 |       0.375 |      0.000 |     −0.188 |   −0.062 |   −0.062 |   +0.062 |   −0.125 |
| s2_b2 |       0.500 |     −0.125 |     −0.188 |   −0.062 |   −0.062 |    0.000 |   −0.188 |
| s4_b0 |       0.000 |      0.000 | **+0.188** |    0.000 |    0.000 |    0.000 |   −0.125 |
| s4_b1 |       0.062 |     −0.062 |     −0.062 |   −0.062 |   −0.188 |   −0.062 |   −0.250 |
| s4_b2 |       0.062 |     −0.062 | **+0.188** |   −0.062 |   −0.188 |   −0.125 |   −0.062 |

### 结果 2：bucket 均值（3 batch 平均 EM）

| bucket | cond         | near EM |      Δnear | far EM |   Δfar | ret EM |   Δret |  hit | used(near) | used(far) |
| ------ | ------------ | ------: | ---------: | -----: | -----: | -----: | -----: | ---: | ---------: | --------: |
| S1     | no_memory    |   0.125 |          0 |  0.271 |      0 |  0.854 |      0 |    0 |          0 |         0 |
| S1     | old_episodic |   0.167 | **+0.042** |  0.167 | −0.104 |  0.875 | +0.021 | 1.00 |       0.93 |      0.00 |
| S1     | new_episodic |   0.062 | **−0.062** |  0.104 | −0.167 |  0.708 | −0.146 | 1.00 |       0.97 |      0.00 |
| S2     | no_memory    |   0.458 |          0 |  0.083 |      0 |  0.875 |      0 |    0 |          0 |         0 |
| S2     | old_episodic |   0.375 | **−0.083** |  0.021 | −0.062 |  0.938 | +0.062 | 1.00 |       1.00 |      0.00 |
| S2     | new_episodic |   0.208 | **−0.250** |  0.021 | −0.062 |  0.813 | −0.062 | 1.00 |       1.00 |      0.00 |
| S4     | no_memory    |   0.042 |          0 |  0.167 |      0 |  0.875 |      0 |    0 |          0 |         0 |
| S4     | old_episodic |   0.000 |     −0.042 |  0.125 | −0.042 |  0.813 | −0.062 | 1.00 |       1.00 |      0.00 |
| S4     | new_episodic |   0.146 | **+0.104** |  0.042 | −0.125 |  0.729 | −0.146 | 1.00 |       1.00 |      0.00 |

`retrieval_hit_rate=1.0`：三条记忆都被检索并注入。`memory_used_rate` 是卡片是否提到当前 task 的 capability/pattern，不是模型是否真的用了记忆；far=0 是因为注入的是源 capability 卡片。

new 的 token 开销明显高于 old（结构化 procedure card 更长）。例如 S1_b0 near tokens：no_mem 147k / old 160k / new 195k。

### Go 判定

- S1 new 平均 Δnear = −0.0625，方向一致率 1/3
- S2 new 平均 Δnear = −0.250，方向一致率 0/3
- S4 new 平均 Δnear = **+0.104**，方向一致率 2/3
- **2/3 bucket 平均 near gain > 0？否（仅 1/3）。Go 未过。**
- old_episodic 同样未过：仅 S1 均值为正，且 b2 已翻负。

### 结论

- **Capability-aware episodic 没有把真实经验稳定转成 near-transfer 能力。** 检索是通的（hit=1.0），问题在吸收：结构化卡片对 S1/S2 的干扰大于帮助，且常伤 retention。
- 唯一阳性：S4 cross-doc 的 new-episodic，2/3 batch Δnear=+0.188（+3/16 题）。这与 E0「S4=不稳定能力」一致，但单桶阳性不够过 Go。
- old Q→A episodic 在 S1 有弱正 near（+0.042），但远迁移和跨 batch 不稳定；new 比 old 更差，说明本轮 hybrid+procedure card **不是** 正确的吸收形式。
- **不因此停 R2/R3。** 指令要求 episodic / executable skill / verified LoRA 各自修到合理实现后再做 R4。R1 只说明 episodic 载体仍不够；下一步看 skill 是否能真正控制 runtime。
- **不进入 learned policy / R5。** 等 R4 的 U_capability 矩阵。

---

## 2026-08-13 R2 Executable Skill Validation（完成）

### Setting

- 目的：把 skill 从 prompt 建议改成真正控制 runtime 的 procedure，看 near-transfer 是否出现正 gain。
- 对照：No skill / Old prompt skill（experience 抽象出的非 executable 列表）/ Executable skill（Skill A multi-hop bridge、Skill B temporal resolution、Skill C cross-doc synthesis）
- 每个 skill：near=16 / far=16 / retention=16（复用 R1 各 bucket 第一组 probe；与 R1 b0 同一批题）
- 验收：tool sequence compliance、evidence completion、near/far EM gain、retention delta、token overhead。**validation near gain ≥ 0 才 accept**。
- Executable skill **不走 LLM**：runtime 按 schema 调 search/open，从打开文档抽 span，不写死 gold / 不复用原题答案。
- 模型：Qwen2.5-3B-Instruct；seed=0；GPU 6/2/3 分 capability 并行
- 墙钟：~250–262s / capability
- 产物：`outputs/r2_executable_skill/`（`summary.json` + 各 cap 子目录）

### 结果

| skill     | cond           |   near EM |      Δnear | far EM |   Δfar | ret EM |   Δret | compliance | evidence | tokens near |
| --------- | -------------- | --------: | ---------: | -----: | -----: | -----: | -----: | ---------: | -------: | ----------: |
| multi_hop | no_skill       |     0.188 |          0 |  0.250 |      0 |  0.812 |      0 |          — |        — |        147k |
| multi_hop | old_skill      |     0.000 |     −0.188 |  0.250 |      0 |  0.938 | +0.125 |          — |        — |         78k |
| multi_hop | **executable** | **0.750** | **+0.562** |  0.188 | −0.062 |  0.125 | −0.688 |       0.80 |     1.00 |           0 |
| temporal  | no_skill       |     0.562 |          0 |  0.062 |      0 |  0.750 |      0 |          — |        — |        119k |
| temporal  | old_skill      |     0.562 |          0 |  0.062 |      0 |  1.000 | +0.250 |          — |        — |         59k |
| temporal  | **executable** | **1.000** | **+0.438** |  0.062 |      0 |  0.000 | −0.750 |       0.75 |     1.00 |           0 |
| cross_doc | no_skill       |     0.000 |          0 |  0.125 |      0 |  0.750 |      0 |          — |        — |         84k |
| cross_doc | old_skill      |     0.000 |          0 |  0.000 | −0.125 |  1.000 | +0.250 |          — |        — |         60k |
| cross_doc | **executable** | **1.000** | **+1.000** |  0.000 | −0.125 |  0.000 | −0.750 |       0.80 |     1.00 |           0 |

三 skill schema 校验通过，全部 **accepted=true**（near gain ≥ 0）。

U_capability = Δnear + 0.5 Δfar − 0.5 max(0, −Δret)：

- multi_hop：0.562 − 0.031 − 0.344 = **+0.188**
- temporal：0.438 − 0 − 0.375 = **+0.062**
- cross_doc：1.000 − 0.062 − 0.375 = **+0.562**

### 结论

- **Executable skill 第一次在 near-transfer 上给出稳定、大幅度正 gain。** 这与 E3b「old skill 平均最差」不矛盾：旧 skill 是 search/open 列表式 prompt，本轮 old_skill 在 S1 仍伤 3/16 题。
- 增益来自 **runtime 被 procedure 接管**（tokens=0，不经过 3B 生成），不是权重变好。evidence completion=1.0，tool compliance 0.75–0.80。远迁移接近 0：skill 绑死单一 capability，不会把 multi-hop 程序搬到 temporal/S0。
- **Retention 大跌是机制结果，不是评测事故：** 把 multi-hop/temporal/cross-doc procedure 强制套到 S0 single-hop 上会做错工具序列。U_capability 已按公式扣 retention；扣完后三格仍为正，因此 **不从 Carrier Matrix 移除 skill**。
- R4 的 skill 列应对 **匹配 capability 的 batch** 使用对应 executable skill；跨 capability 的格子按设计会差。
- **继续 R3**（verified-success batch LoRA）。不进入 R5。

---

## 2026-08-13 R3 Verified-Success Batch Weight Internalization（完成）

### Setting

- 目的：用 **verified success batch** 做 LoRA（禁止单条 5–20 step SFT、禁止未验证失败轨迹），对照 no-update / episodic / semantic / executable skill / LoRA。
- 数据：S1 multi_hop=64（全 E0 success），S2 temporal=48，S4 cross_doc=24（任务池上限）。0 条 corrected。document ID 已清洗为 `[DOC]`。
- LoRA：rank=8，epochs=2，lr=5e-5；S1 实际 115 step（early stop），S2 96 step，S4 25 step。评测后 `unload_adapter` + `rebuild`。
- Probe：与 R2 相同，每 cap near/far/retention 各 16。
- 模型：Qwen2.5-3B-Instruct；seed=0；GPU 6/2/3
- 墙钟：627–730s / capability
- 产物：`outputs/r3_weight_batch/`（含 `adapter_meta.json`、adapter 路径、train ids）

### 结果（Δ 相对 no_update；U_capability = Δnear + 0.5 Δfar − 0.5 max(0,−Δret)）

| cap  |    n | carrier              |      near |      Δnear |   far |       Δfar |   ret |   Δret |      U_cap |
| ---- | ---: | -------------------- | --------: | ---------: | ----: | ---------: | ----: | -----: | ---------: |
| S1   |   64 | no_update            |     0.188 |          0 | 0.250 |          0 | 0.812 |      0 |          0 |
| S1   |   64 | episodic             |     0.062 |     −0.125 | 0.125 |     −0.125 | 0.812 |      0 |     −0.188 |
| S1   |   64 | semantic             |     0.125 |     −0.062 | 0.188 |     −0.062 | 0.875 | +0.062 |     −0.094 |
| S1   |   64 | **executable skill** | **0.750** | **+0.562** | 0.188 |     −0.062 | 0.125 | −0.688 | **+0.188** |
| S1   |   64 | LoRA                 |     0.188 |          0 | 0.188 |     −0.062 | 0.750 | −0.062 |     −0.062 |
| S2   |   48 | no_update            |     0.500 |          0 | 0.062 |          0 | 0.750 |      0 |          0 |
| S2   |   48 | episodic             |     0.188 |     −0.312 | 0.062 |          0 | 0.688 | −0.062 |     −0.344 |
| S2   |   48 | semantic             |     0.312 |     −0.188 | 0.000 |     −0.062 | 0.875 | +0.125 |     −0.219 |
| S2   |   48 | **executable skill** | **1.000** | **+0.500** | 0.062 |          0 | 0.000 | −0.750 | **+0.125** |
| S2   |   48 | LoRA                 |     0.375 |     −0.125 | 0.188 |     +0.125 | 1.000 | +0.250 |     −0.062 |
| S4   |   24 | no_update            |     0.000 |          0 | 0.125 |          0 | 0.812 |      0 |          0 |
| S4   |   24 | episodic             |     0.000 |          0 | 0.062 |     −0.062 | 0.688 | −0.125 |     −0.094 |
| S4   |   24 | semantic             |     0.000 |          0 | 0.125 |          0 | 0.812 |      0 |          0 |
| S4   |   24 | **executable skill** | **1.000** | **+1.000** | 0.000 |     −0.125 | 0.000 | −0.812 | **+0.531** |
| S4   |   24 | LoRA                 |     0.062 | **+0.062** | 0.188 | **+0.062** | 0.875 | +0.062 | **+0.094** |

LoRA train：S1 mean_loss 1.46 → best 0.21；S2 1.64 → 0.25；S4 3.45 → 2.47（n=24，拟合不足）。

### 结论

- **Verified-success batch LoRA 没有把 3B 权重内化成稳定 near-transfer 能力。** S1/S2 的 U 为负；S4 仅 +1/16 题，幅度等于 probe 粒度。
- 同 probe 上 **executable skill 仍是唯一稳定正 U 的载体**，与 R2 一致。episodic / semantic batch 继续为负，与 R1 一致。
- LoRA 路径工程上是安全的（unload+rebuild，adapter_meta 完整），失败是 **监督信号/表示** 问题：SFT 目标仍是 `<tool>answer</tool>` + 最终答案，而 S1 缺口在桥接搜索，不在把答案写出来。
- Weight 列仍进入 R4（实现已合理），但预期多数格子 ≤0。Skill 三 cap 均已 accept，R4 不标 N/A。
- **继续 R4。** 不进入 R5。

---

## 2026-08-13 R4 Batch-level Carrier Utility Matrix（完成）

### Setting

- 目的：在 **experience batch** 粒度重建 Carrier Matrix，判断经验能否被吸收、以及 carrier 选择是否改变 realized capability gain。
- 12 batches，每 batch=8 experiences；每 capability 4 batches（S1/S2/S4）。
- Carriers：episodic（hybrid）/ semantic / skill（R2 三 skill 均 accept，无 N/A）/ weight（verified success LoRA rank=8, epochs=1）。
- Probe：near/far/retention 各 16。Go 只用 `U_capability = Δnear + 0.5 Δfar − 0.5 max(0, −Δretention)`；token 成本另报 `U_efficiency`。
- 模型：Qwen2.5-3B-Instruct；seed=0；GPU 6/2/3；墙钟约 45–50 min / bucket
- 产物：`outputs/r4_batch_carrier_matrix/`（`carrier_matrix.csv/json`、`GO_NO_GO_R4.md`）
- **S4 覆盖限制：** 全库只有 24 道 `t_xd_evt`，b0–b2 互不重叠；b3 从 verified pool 回填，与 b0 重叠 4 题、b1 重叠 3 题、b2 重叠 1 题。S1/S2 的 4×8 无重叠。

### 结果 1：12 × 4 U_capability（粒度 1/16=0.0625）

| batch |  epi U |      sem U |    skill U |   wt U | unique best                  |
| ----- | -----: | ---------: | ---------: | -----: | ---------------------------- |
| s1_b0 | −0.531 |     −0.344 | **−0.094** | −0.188 | skill（唯一 U 最大，仍为负） |
| s1_b1 | −0.281 |     −0.094 | **+0.281** | −0.125 | **skill**                    |
| s1_b2 | −0.188 |     −0.062 | **+0.219** | −0.156 | **skill**                    |
| s1_b3 | −0.125 |     −0.125 | **+0.156** | −0.125 | **skill**                    |
| s2_b0 | −0.156 |     +0.094 | **+0.250** | −0.094 | **skill**                    |
| s2_b1 | −0.094 |     −0.062 | **+0.094** | +0.062 | **skill**                    |
| s2_b2 | −0.562 |     −0.125 | **+0.125** | −0.094 | **skill**                    |
| s2_b3 | −0.250 | **+0.094** |     +0.062 | −0.031 | **semantic**                 |
| s4_b0 | +0.031 |     −0.062 | **+0.562** |  0.000 | **skill**                    |
| s4_b1 |  0.000 |     −0.125 | **+0.375** | +0.062 | **skill**                    |
| s4_b2 | −0.062 |      0.000 | **+0.438** | −0.094 | **skill**                    |
| s4_b3 | −0.062 |     −0.188 | **+0.375** | −0.125 | **skill**                    |

Skill 的 Δnear 普遍 +0.38 到 +1.00，但 Δretention 约 −0.69 到 −1.00（procedure 绑死 capability，套到 S0 全错）。U 仍多为正，是因为 near 涨幅大于 0.5×retention 跌幅。

### 结果 2：carrier 均值

| carrier   | mean U_cap |      #U>0 | unique best | 解读                                   |
| --------- | ---------: | --------: | ----------: | -------------------------------------- |
| **skill** | **+0.237** | **11/12** |          11 | 唯一稳定正载体；S4 最强（mean +0.438） |
| semantic  |     −0.083 |      2/12 |           1 | 仅 S2 两次小正（+0.094）；S1 全负      |
| weight    |     −0.076 |      2/12 |           0 | 两格 +1/16 题，其余 ≤0                 |
| episodic  |     −0.190 |      1/12 |           0 | 最差；S1/S2 持续伤害                   |

bucket × carrier mean U：S1 skill +0.141（3/4 正）；S2 skill +0.133（4/4 正）、semantic 0.0；S4 skill +0.438（4/4 正）。

### Go 判定（脚本 `analyze_r4_go_nogo.py` → `GO_NO_GO_R4.md`）

| Gate     | 标准                         | 数字                                | 判定          |
| -------- | ---------------------------- | ----------------------------------- | ------------- |
| A        | ≥50% batch 有某 carrier U>0  | **11/12 = 92%**                     | **PASS**      |
| B        | best 随 batch 变，非全局统治 | unique best = skill 11 / semantic 1 | **字母 PASS** |
| C        | oracle > best fixed          | 0.2396 vs 0.2370（差 0.003）        | **字母 PASS** |
| D        | 幅度 > 1/16                  | skill mean 0.237 >> 0.0625          | **PASS**      |
| **总判** | A∧B∧C∧D                      |                                     | **GO**        |

**机制保留（必须写进结论，不能只报 GO）：**

- B 很弱：skill 11/12 接近全局统治；唯一翻盘是 S2_b3 semantic +0.094 vs skill +0.062（半题）。
- C 更弱：oracle 相对 always-skill 只高 **0.003**，远小于 1 题。不支持「值得训 learned router」。
- 正 U 几乎全部来自 **executable skill 的 near-transfer**，不是 memory/weight 把经验写进了模型。

### 结论

- **经验可以被吸收。** 修复后的 executable skill 在 11/12 batch 上给出正 U_capability，Δnear 幅度是数题而不是 ±1。此前 E1/E2/E3 的失败应解释为：**Acquisition / routing 是在无效的 episodic/prompt-skill assimilation 后面评的。**
- **Carrier 选择会改变 realized gain**——但是以「skill vs 其它」的粗粒度，不是四载体之间的精细 routing。always-skill ≈ oracle。
- Episodic / semantic / LoRA **本轮仍不能**作为 S1 硬缺口的吸收机制。R5 若继续，必须 **capability-conditioned：默认 executable skill**，禁止再共用 episodic-only。
- **当时按 R1–R4 指令停在 R4。** 字母 GO 允许随后只比较 acquisition × skill assimilation；learned router（R6）没有监督多样性。R5 已于 2026-08-14 按 `ExPolicy_Next_Run_Instructions_R5.md` 跑完，见下节。

---

## 2026-08-14 R5 Active Acquisition under Validated Executable Assimilation（完成）

### Setting

- 目的：在 **已经验证有效的 executable-skill assimilation** 后面重测 Gate B：同 experience budget 下，active 是否优于 passive。
- **不**证明 deployable learned routing / adaptive assimilation / 完整 ExPolicy。主评测使用 **oracle capability dispatch**（S1/S2/S4 probe 只允许对应 skill；S0 retention 不用这些 skill）。
- Causal path（`scripts/audit_r5_causal_path.py`，14/14 PASS 后才开跑）：
  `clean agent → acquire under π → SkillAssimilator.fit(real experiences) → experience-conditioned executable skill → future eval`
  - 冻结 R2 三个 procedure primitive，默认全部 disabled
  - 某 capability 累计 **≥8** 条真实 acquired experience 才允许 `fit()` / 首次激活
  - empty / shuffled / 删掉该 cap 经验 → 不得加载未修改的 R2 全局 skill
  - artifact 不含 probe task id / gold answer / gold doc id
- 数据划分（`scripts/build_r5_splits.py`，运行前冻结）：
  - future probe = **12 × 3 cap = 36**（Helion S4 只有 24 题，无法严格隔离 16+8，fail-closed 降到 12 并写入 `SPLIT_AUDIT.md`）
  - retention = 24 道 S0
  - acquisition pool = 554，无 replacement；同一 seed 下五种方法共享三套集合
  - task-id overlap future∩acq = future∩ret = acq∩ret = **0**
  - 已知缺口：S4 单一 pattern `event_org_and_lead`；target-cap gold-answer overlap future∩acq=18（报告，不静默修补）
- 方法（E2 定义，不改 scoring）：passive / reward_seeking / uncertainty / frontier / failure_replay
- Primary active = frontier, uncertainty, failure_replay；control = passive, reward_seeking
- budget=48；skill refit checkpoint = 8/16/24/32/40/48；probe eval checkpoint = 0/16/32/48
- 模型：Qwen2.5-3B-Instruct；world seed=13；seeds=0,1,2；GPU **4/5/6/7** 并行（`scripts/run_r5_suite.py`）
- 墙钟：约 7–9 min / (method × seed)；suite 14:51–15:31
- 产物：`outputs/r5_acquisition_skill/`（`R5_SUMMARY.json`、`R5_GATE_B.md`、`R5_PER_SEED.csv`、`R5_LEARNING_CURVE.csv`、`R5_SKILL_ACTIVATION.csv`、`SPLIT_AUDIT.md`、`R5_CAUSAL_AUDIT.json`、ablations）

### 冻结 Gate B 判据（跑前固定）

1. 3 个 primary 中至少 **2/3** 的 3-seed mean ΔFuture > passive
2. 至少一个 primary：`mean(Δ_active − Δ_passive) ≥ 0.05` 且 ≥2/3 seed 优于 passive
3. 该方法相对 clean base 的 mean ΔFuture > 0
4. ≥2/3 target capability 的 mean gain 非负
5. 差异不能只等于单题翻转（grain = 1/36 ≈ 0.028）

### 结果 1：Acquisition 行为（budget=48，3-seed 均值）

| method         | immediate reward | 选中分布（典型）                                 | 首次激活                                    | #active skills@48 | tokens | tools |
| -------------- | ---------------: | ------------------------------------------------ | ------------------------------------------- | ----------------: | -----: | ----: |
| reward_seeking |        **1.000** | 24 conflicting + 23 reliability + 1 long_context | 无                                          |              0.00 |   262k |   144 |
| passive        |            0.611 | ~24 single_hop + ~12 multi_hop + 少量其它        | MH @ 24–40                                  |              1.00 |   316k |   197 |
| frontier       |            0.236 | ~37 multi_hop + ~11 cross-doc                    | **S4 @ 8**，MH @ 16–24                      |          **2.00** |   385k |   241 |
| failure_replay |            0.118 | **48 全是 multi_hop**                            | MH @ 8                                      |              1.00 |   423k |   267 |
| uncertainty    |            0.000 | 46 query_decomp + 2 entity_disambig              | MH @ 16（query_decomp 映射到 S1 primitive） |              1.00 |   546k |   374 |

没有方法攒够 8 条 temporal，**S2 skill 从未激活**。S4 只有 frontier 采到 ≥8 条（pool 里该 cap 仅 12 题）。

### 结果 2：Capability outcome（ΔFuture = Future_EM − base；n_future=36）

clean base 3-seed mean future EM = 0.241（seed 0/1/2 = 0.278 / 0.194 / 0.250）。retention 因 oracle dispatch 基本不动。

| method         | mean ΔFuture |   std | CI95           | per-seed ΔFuture      | mean ret Δ | per-cap mean gain (MH / temporal / S4) |
| -------------- | -----------: | ----: | -------------- | --------------------- | ---------: | -------------------------------------- |
| **frontier**   |   **+0.519** | 0.026 | [0.500, 0.556] | 0.500 / 0.556 / 0.500 |      0.000 | **+0.722 / 0.000 / +0.833**            |
| uncertainty    |       +0.241 | 0.026 | [0.222, 0.278] | 0.222 / 0.278 / 0.222 |     +0.014 | +0.722 / 0.000 / 0.000                 |
| failure_replay |       +0.231 | 0.035 | [0.194, 0.278] | 0.222 / 0.278 / 0.194 |     +0.028 | +0.694 / 0.000 / 0.000                 |
| passive        |       +0.222 | 0.023 | [0.194, 0.250] | 0.222 / 0.250 / 0.194 |      0.000 | +0.667 / 0.000 / 0.000                 |
| reward_seeking |    **0.000** | 0.000 | [0.000, 0.000] | 0 / 0 / 0             |     +0.014 | 0 / 0 / 0                              |

frontier vs passive：mean gap = **+0.296**（约 10.7 题），**3/3 seed 都赢**。uncertainty / failure_replay 相对 passive 的 gap 只有 +0.019 / +0.009，落在 1 题粒度内。

最终 per-cap future EM（3-seed mean）：

| method         |        MH | temporal |        S4 |     macro |
| -------------- | --------: | -------: | --------: | --------: |
| base           |     0.278 |    0.361 |     0.083 |     0.241 |
| reward_seeking |     0.278 |    0.361 |     0.083 |     0.241 |
| passive        |     0.944 |    0.361 |     0.083 |     0.463 |
| failure_replay |     0.972 |    0.361 |     0.083 |     0.472 |
| uncertainty    |     1.000 |    0.361 |     0.083 |     0.481 |
| **frontier**   | **1.000** |    0.361 | **0.917** | **0.759** |

### 结果 3：Learning curve（3-seed mean ΔFuture）

| budget | passive | reward_seeking | uncertainty |  frontier | failure_replay |
| -----: | ------: | -------------: | ----------: | --------: | -------------: |
|      0 |   0.000 |          0.000 |       0.000 |     0.000 |          0.000 |
|     16 |   0.000 |          0.000 |       0.222 | **0.426** |          0.241 |
|     32 |   0.130 |          0.000 |       0.222 | **0.519** |          0.241 |
|     48 |   0.222 |          0.000 |       0.241 | **0.519** |          0.231 |

frontier 在 budget=8 就激活 S4，budget=16 再激活 S1，之后曲线持平。passive 要到 24–40 才攒够 8 条 MH。reward_seeking 全程不激活任何 hard-capability skill。

### 结果 4：三项必要消融

**A. Skill disabled**（同一批 acquired experience，final eval 关掉 skill）  
`Active+Skill − Active+NoSkill` 与主表 ΔFuture **逐 seed 相等**。NoSkill = clean base。R5 没有写其它 carrier，增益 100% 来自 skill 激活，不是采样改变了短时 agent state。

**B. Shuffled / unfiltered fit**（打乱 capability 后再 fit；诊断「budget 到 8 是否自动送 skill」）

| method         | correct future EM | shuffled future EM | correct − shuffled |
| -------------- | ----------------: | -----------------: | -----------------: |
| frontier       |         **0.759** |              0.676 |         **+0.083** |
| uncertainty    |             0.481 |              0.676 |             −0.194 |
| failure_replay |             0.472 |              0.694 |             −0.222 |
| passive        |             0.463 |              0.685 |             −0.222 |
| reward_seeking |             0.241 |              0.704 |             −0.463 |

shuffled 会把 **未采到的 cap** 也装上 skill（S1/S4 的 shuffled skill EM≈0.92–1.0，temporal shuffled skill 反而差，EM≈0.08–0.17）。所以没采 S4 的方法在 shuffled 下会被「白送」S4 skill，看起来更好。主路径禁止这种 unfiltered fit（causal audit PASS）。**只有真正采到 S1+S4 的 frontier，correct > shuffled。**

**C. Fixed pre-installed R2 skills**（诊断 only，不进 Gate B）  
三 seed future EM 全是 **1.000**，mean ΔFuture=+0.759。五种 acquisition 会变得无差别。说明：若不把 skill 的激活绑在 acquired experience 上，R5 回答不了 Gate B。

### Gate B 判定

| 条件                                   | 数字                                       | 判定     |
| -------------------------------------- | ------------------------------------------ | -------- |
| 1. ≥2/3 primary mean ΔFuture > passive | 3/3（0.519 / 0.241 / 0.231 > 0.222）       | **PASS** |
| 2. 某 primary gap≥0.05 且 ≥2/3 seed 赢 | frontier gap=+0.296，**3/3 seed**          | **PASS** |
| 3. 该方法 ΔFuture > 0 vs base          | frontier +0.519                            | **PASS** |
| 4. ≥2/3 cap mean gain ≥ 0              | frontier：MH +0.722，temporal 0，S4 +0.833 | **PASS** |
| 5. 不是单题翻转                        | frontier−passive=0.296 ≫ 0.028             | **PASS** |
| **总判**                               |                                            | **PASS** |

**机制保留（必须写进结论）：**

- PASS **主要由 frontier 贡献**。uncertainty / failure_replay 相对 passive 的优势 < 1 题，单独看是 AMBIGUOUS。
- temporal 全方法 gain=0：谁都没采够 8 条 S2。条件 4 把 0 算作非负；正增益来自 S1（所有能激活 MH skill 的方法）和 S4（只有 frontier）。
- 主评测是 **oracle capability dispatch**，隔离的是 acquisition 变量，**不证明模型能自己决定何时调用 skill**。
- 增益来自 **experience-conditioned 地打开已验证的 executable procedure**，不是 3B 权重或记忆内化。

### 结论

- **在可用的 experience-conditioned executable assimilation 下，active acquisition 能以相同 budget 获得更高 future capability gain。** Gate B 在该受控设定下 **PASS**。
- 旧 E2 的阴性结果应修订为：*Acquisition was evaluated behind an ineffective episodic assimilation bottleneck.* 换到 R2/R4 验证过的 skill 路径后，**选什么经验会改变最终能力**——因为不同 policy 决定了哪些 capability skill 被激活、何时被激活。
- 机制与 E2 的选择行为一致，但这次转化成了 capability：
  - **reward_seeking**：immediate reward=1.0，只采 S3 易题，0 个 hard skill，ΔFuture=0（负控成立）。
  - **passive**：随机碰到足够 MH 后才激活 1 个 skill，ΔFuture=+0.222。
  - **failure_replay / uncertainty**：专采 S1 家族，更早激活 MH skill；相对 passive 几乎不再涨，因为 passive 最终也会激活 MH。
  - **frontier**：同时压 S1 硬缺口和 S4 不稳定区，**唯一激活 2 个 skill**，也是唯一拿到 S4 增益的方法。
- Ablation A/C 支持因果：关掉 skill → 回到 base；预装全部 R2 skill → 五种方法无差别。
- **不宣称** ExPolicy / Adaptive Assimilation / learned routing 已被证明。R4 的 always-skill ≈ oracle 没有改变。当前能吸收经验的仍然几乎只有 capability-bound executable procedure。

**按 R5 指令当时停止。R5.1 已于 2026-08-14 按 `ExPolicy_Next_Run_Instructions_R5_1_H20_GPU4-7.md` 跑完，见下节。不跑 R6，不上 E5–E8。**

---

## 2026-08-14 R5.1 Robustness & Causal Validation（完成，Case B）

### Setting

- 目的：回答 R5 的 Active > Passive 是否在 **更多 seed、不同激活阈值、相同 activation mask 的内容消融、非 oracle dispatch** 下仍然成立。**不进入 R6 learned carrier router，不启动 E5–E8。**
- 原则：不覆盖 `outputs/r5_acquisition_skill/`；不重建 seed 0–2 frozen split；不改 R2 primitive；future / retention / acquisition 零 task-id overlap。
- 四条并行 track（`scripts/run_r5_1_robustness.py`）：

| Track                 | GPU                | 问题                                   | 方法 / 条件                                                  |
| --------------------- | ------------------ | -------------------------------------- | ------------------------------------------------------------ |
| A Seed extension      | 4，完成后 6/7 并入 | frontier > passive 能否跨 10 seed 复现 | frontier / passive；seed **3–9** 新跑，与 R5 的 0–2 合并     |
| B Temporal / coverage | 5                  | 为何 S2 永不激活；cap-norm 能否补上    | pool audit + `frontier_capnorm` × seed 0–2；frontier/passive 复用 R5 |
| C1 Threshold          | 6                  | 增益是否只是 K=8 门槛跳变              | frontier / passive；K = 4 / 8 / 12 / 16；复用 R5 trajectory  |
| C2 Content causality  | 6                  | 内容 vs 只按 count 解锁                | RealFit / CountOnly / WithinCapShuffle / CrossCapContent；**冻结 R5 activation mask** |
| D Deployable dispatch | 7                  | 去掉 oracle cap label 后增益还剩多少   | oracle / text_router（TF-IDF+LR） / no_skill / all_skills_fixed_order |

- 共用：budget=48；world seed=13；Qwen2.5-3B-Instruct；Helion BM25；oracle dispatch 仍用于 A/B/C。
- Track A 新 split 用同一 `build_seed_split`（seed 3–9）；seed 0–2 只复制 R5 hash，不改写。预检：`R5_CAUSAL_AUDIT.json` 仍 PASS 14/14。
- 冻结 split SHA256（与原 R5 一致）：seed 0 `61753eee02a792b27d21b5d05ced9c68e976e5b3645bc48477dfaa021ab645db`；seed 1 `710a123cc92fa1f0b7f92a8ed80eebcd2aa6c1e93fe4b0fb1cd70e14f2a10e78`；seed 2 `94c9e74b5aa26b1cefa136ebe1304ed83e93be96ae78b9b05c64be12c41d2962`。
- Track C2 / D / C1 复用 R5 已采集轨迹 + skill 执行 / base LLM fallback，不重新采集。C2 冻结原 R5 activation mask。text-router 只用 acquisition-pool 题面，不用 future 标签。
- 墙钟：约 20:32–21:22（Track C/D 数分钟结束；A 为长杆，三卡并行约 50 min）。
- 产物：`outputs/r5_1_robustness/`（`R5_1_MASTER_SUMMARY.md`、`R5_1_DECISION.json`、`R5_1_MASTER_TABLE.csv` 及各 track 子目录）。driver：`scripts/run_r5_1_robustness.py`、`scripts/analyze_r5_1_robustness.py`。
- 稳健支持门槛（A，跑前固定，不筛 seed）：10-seed mean gap>0；paired bootstrap 95% CI 不跨 0；≥7/10 seed frontier>passive。

### 结果 1：Track A — 10-seed frontier vs passive

合并 R5 seed 0–2 与本轮 seed 3–9。grain = 1/36 ≈ 0.028。

| seed | base future EM | frontier Δ | passive Δ | frontier−passive | frontier #active | passive #active |
| ---: | -------------: | ---------: | --------: | ---------------: | ---------------: | --------------: |
|    0 |          0.278 |     +0.500 |    +0.222 |       **+0.278** |                2 |               1 |
|    1 |          0.194 |     +0.556 |    +0.250 |       **+0.306** |                2 |               1 |
|    2 |          0.250 |     +0.500 |    +0.194 |       **+0.306** |                2 |               1 |
|    3 |          0.278 |     +0.500 |    +0.222 |       **+0.278** |                2 |               1 |
|    4 |          0.250 |     +0.500 |    +0.139 |       **+0.361** |                2 |               1 |
|    5 |          0.250 |     +0.528 |    +0.250 |       **+0.278** |                2 |               1 |
|    6 |          0.278 |     +0.472 |    +0.194 |       **+0.278** |                2 |               1 |
|    7 |          0.222 |     +0.556 |    +0.278 |       **+0.278** |                2 |               1 |
|    8 |          0.194 |     +0.556 |    +0.278 |       **+0.278** |                2 |               1 |
|    9 |          0.250 |     +0.528 |    +0.250 |       **+0.278** |                2 |               1 |

| 量                         | 值                           |
| -------------------------- | ---------------------------- |
| n                          | **10**（无 seed 被丢弃）     |
| mean(frontier Δ)           | **+0.519**                   |
| mean(passive Δ)            | +0.228                       |
| **mean(frontier−passive)** | **+0.292**                   |
| std                        | 0.027                        |
| paired bootstrap 95% CI    | **[0.278, 0.308]**（不跨 0） |
| win-rate                   | **10/10**                    |

10 seed 上 frontier 均激活 2 个 skill；多数为 S4@8 + MH@16–24（seed 7 为 MH@16 后 S4@24）。passive 均只激活 MH。temporal gain 仍全为 0。retention 基本不动。

稳健门槛：mean gap>0 **PASS**；CI 不跨 0 **PASS**；≥7/10 胜 **PASS**。

### 结果 2：Track C1 — 激活阈值敏感性（复用 R5 轨迹，seed 0–2）

refit checkpoint 仍为 8/16/24/32/40/48（指令要求与 R5 相同），因此 K=4 最早也只能在 budget=8 激活。

| method   |    K | mean ΔFuture | 典型首次激活                      | #active@48（3-seed） |
| -------- | ---: | -----------: | --------------------------------- | -------------------- |
| frontier |    4 |   **+0.519** | S4@8，MH@16                       | 2 / 2 / 2            |
| frontier |    8 |   **+0.519** | 同 K=4                            | 2 / 2 / 2            |
| frontier |   12 |       +0.333 | 常失去 S4（pool 仅 8–12 条）      | 1 / 2 / 1            |
| frontier |   16 |       +0.241 | 多只剩 MH                         | 1 / 1 / 1            |
| passive  |    4 |       +0.222 | MH@8–24                           | 1                    |
| passive  |    8 |       +0.222 | MH@24–40                          | 1                    |
| passive  |   12 |       +0.222 | MH@32–40                          | 1                    |
| passive  |   16 |       +0.065 | seed 0/1 **永不激活**；seed 2 @40 | 0 / 0 / 1            |

各 K 下方向均为 frontier > passive。增益随 K 变大而下降，符合「够数才解锁」的门槛形状，而不是某个 K 被事后挑出来报最好一档。

### 结果 3：Track C2 — same-mask 内容因果（本轮最关键）

冻结原 R5 的 activation mask 与 checkpoint，只改 `fit()` 看到的内容。禁止 shuffled/cross-cap 额外激活新 skill。

Frontier 3-seed mean ΔFuture：

| fit mode                                                    | mean ΔFuture | mean future EM | 相对 RealFit                 |
| ----------------------------------------------------------- | -----------: | -------------: | ---------------------------- |
| RealFit（真实 acquired）                                    |   **+0.519** |          0.759 | —                            |
| **CountOnly**（只按 count 解锁 frozen primitive，不读内容） |   **+0.546** |          0.787 | **+0.028**（不差于 RealFit） |
| CrossCapContent（换其它 cap 经验，mask 不变）               |   **+0.546** |          0.787 | 与 CountOnly 相同            |
| WithinCapShuffle（同 cap 的其它训练侧经验）                 |       +0.407 |          0.648 | **−0.111**                   |

Passive（只激活 MH）上四种条件几乎重合（Δ ≈ 0.19–0.24）。

逐 seed（frontier）：

| seed | RealFit | CountOnly |                                                   Shuffle | CrossCap |
| ---: | ------: | --------: | --------------------------------------------------------: | -------: |
|    0 |   0.500 |     0.528 |                                                     0.500 |    0.528 |
|    1 |   0.556 |     0.583 | **0.250**（n_active=1；S4 `n_skill=0`，S4 EM 掉回 0.083） |    0.583 |
|    2 |   0.500 |     0.528 |                                                     0.472 |    0.528 |

CountOnly / CrossCap 把 S4 EM 从 RealFit 的 0.917 推到 1.000（默认 primitive 略优于内容条件化 config）。Shuffle 在 seed 1 未能装上 S4 skill（`n_skill=0`），是 Δ 下降的主因。

**判读（按指令 §5.4）：RealFit ≉ 明显优于 CountOnly / 甚至略差 → 不得再写成强意义的 experience assimilation。** 主机制是 **experience-conditioned / evidence-gated skill activation**。Shuffle 掉点作为次要 caveat 保留，不能支撑「内容被深度吸收」。

### 结果 4：Track D — deployable skill dispatch

Router：仅用 acquisition-pool 的 **question text** + 训练侧 capability 标签；TF-IDF (1–2gram) + multinomial LR（class_weight=balanced）。禁止 future 标签、task id、stage、filename、gold。泄漏检查：question / vocab 均无 task-id，`leakage_pass=true`。

|      seed |   val acc |  macro-F1 | per-class recall                 |
| --------: | --------: | --------: | -------------------------------- |
| 0 / 1 / 2 | **1.000** | **1.000** | none / MH / temporal / S4 全 1.0 |

Helion 题面模板可分（「headquartered」「chief executive in YEAR」「organized … and who led」），acc=1.0 **不能外推到开放域**。

下游 3-seed mean future EM：

| method   |    oracle | text_router | no_skill | all_skills_fixed_order | S0 FP（oracle/router） |
| -------- | --------: | ----------: | -------: | ---------------------: | ---------------------- |
| frontier | **0.759** |   **0.759** |    0.241 |                  0.787 | **0 / 0**              |
| passive  |     0.463 |       0.463 |    0.241 |                  0.491 | **0 / 0**              |

- oracle − text-router gap = **0**
- text-router 下 frontier − passive = **+0.296**，与 oracle 相同
- no_skill = clean base
- all_skills_fixed_order 是 stress control：frontier S0 FP rate ≈ 0.29，retention 下降约 0.08–0.13；不作为主方法

### 结果 5：Track B — temporal / coverage

Frozen pool（n=554）静态 frontier score：temporal 最优 rank=**133**，mean percentile≈0.74；**top-48/64/96 的 temporal 条数 = 0/0/0**。扩 budget 到 96 也不会自然采到 ≥8 条 S2。pool 内 temporal 只有 24 条，未复制、未放回 future。

| method               | temporal acquired (s0/s1/s2) | S2 激活 | mean ΔFuture | 行为                                                         |
| -------------------- | ---------------------------- | ------- | -----------: | ------------------------------------------------------------ |
| frontier             | 0 / 0 / 0                    | 否      |   **+0.519** | 集中 S1+S4（entropy ≈ 0.54）                                 |
| passive              | 2 / 2 / 3                    | 否      |       +0.222 | 分散，仍 <8 条 S2                                            |
| **frontier_capnorm** | 0 / 0 / 0                    | 否      |    **0.000** | z-norm 打散到 S3 易题（conflicting 24 + reliability 17），**牺牲 S1/S4**，0 个 target skill |

按指令停止继续堆 heuristic。记录：

> current frontier signal does not expose the S2 capability gap

### 决策树（`R5_1_DECISION.json`）

| Case   | 条件                                                         | 本轮                              |
| ------ | ------------------------------------------------------------ | --------------------------------- |
| A 最强 | 10-seed 稳 + 各 K 同向 + RealFit ≫ CountOnly/shuffle + text-router 仍 frontier>passive | **未过**（内容因果阴性）          |
| **B**  | seed 稳健，但 RealFit ≈ CountOnly                            | **命中**                          |
| C      | oracle 强、deployable 弱                                     | 未过（Helion 上 router = oracle） |
| D      | 扩 seed 后 gap 缩水                                          | 未过（10/10 仍赢，gap 未缩）      |

**Case B 含义：** 项目应明确收缩为 **Experience-Gated Skill Activation**。核心贡献是 *active acquisition 决定哪些 capability-specific executable procedures 变得 eligible / activated*。不要再强调「经验内容被 skill assimilator 深度吸收」。

### 结论

- **R5 的 frontier > passive 在 10 seed 上统计稳健**（mean gap +0.292，CI 不跨 0，10/10）。这加固的是 acquisition 测试，不是完整 ExPolicy。
- **机制应收缩。** CountOnly ≥ RealFit，CrossCap ≈ CountOnly：增益主要来自「采够该 cap 的条数 → 解锁对应 frozen primitive」，不是 `fit()` 从轨迹里学到了关键 config。Shuffle 掉点说明同 cap 内容并非完全无关，但幅度和稳定性都不够写成 assimilation 主故事。
- 阈值扫描支持门槛叙事：K 升高 → 更晚/更少激活 → Δ 下降；K=4 与 K=8 因 refit 网格从 8 起而重合。
- Helion 上 **text-router 保住了全部 oracle gain**，且 frontier>passive 仍成立。这是模板可分世界的结果，**单独作为可部署 dispatch 的开放域证据不足**；但本轮不必写成 Case C。
- Temporal 仍是 frontier 信号的盲区；cap-norm 只会毁掉 S1/S4 优势，补不上 S2。
- **不进入 R6 learned carrier router，不上 E5–E8。** Case A 未过，不考虑 learned acquisition policy。若后续再做，也只应研究 *what experience to acquire for gating*，而不是 carrier routing 或内容内化。

**不宣称** experience assimilation / Adaptive Assimilation / 开放域 skill dispatch 已被证明。

**论文表述（R5.1 Case B，按指令 §12）：** 写成

`active acquisition → experience-gated skill activation → oracle（及 Helion text-router）future gain`

不要写成 content-dependent assimilation / 完整 ExPolicy。Helion 上 text-router = oracle 是模板可分世界的结果，不升级为开放域 deployable dispatch 证据。

R5.1 当时按指令停在 Case B。**R5.2 已于 2026-08-15 按 `ExPolicy_R5_2_Strong_Baseline_Collision_Instructions.md` 跑完**，用 strongest acquisition baseline（Ace-Skill Prioritized Sampler）做 collision test，见下节。

---

## 2026-08-15 R5.2 Strong-Baseline Collision Test（完成，Case A）

### Setting

- 目的：在继续改方法之前，回答 **当前 `frontier acquisition → experience-gated executable skill activation` 是否显著优于最新、最接近的主动采样 baseline（Ace-Skill Prioritized Sampler）**。比较的是 **acquisition policy 本身**，不是 assimilation。
- **不**搬 Ace-Skill 的 clustered organizer / skill memory。后端统一冻结为 R5.1 已确认的 **CountOnly / evidence-count gating**。
- **不**训 R6 learned carrier router、不做 LoRA/SFT sweep、不重跑 RealFit、不把 RealFit 当主贡献。
- Ace-Skill 实现（`expolicy/acquisition/ace_skill.py`，审计见 `ACE_SKILL_IMPLEMENTATION_AUDIT.md`）：
  - 只复现 Prioritized Sampler（官方 `SqrtBiasSampler` + `configs/prioritize.yaml`）
  - 超参按论文/yaml：**ρ=0.95, γ=0.4, ε=0.1**（不用 class 默认 0.9/0.2）
  - **α/β 初值跟官方代码为 0**，不跟论文正文 (0.5, 0.5)；两者初始 v 都是 0.5，后续更新会分叉
  - 允许跨 round 重复采样；同一 batch 内 `replace=False`
  - 6 rounds × batch 8；reward = binary EM ∈ {0,1}
- Skill 激活 **K=8 按 distinct task ID**，不是 rollout 次数。重复刷同一题不能加速解锁。
- 公平协议：
  - 方法：`frontier` vs `ace_skill_prioritized` vs `passive`（uncertainty / failure_replay / reward_seeking 只作已有负控，本轮不新跑）
  - 后端：acquisition → unique-cap evidence count → K=8 → frozen R2 primitive → future eval
  - Primary dispatch = oracle capability；secondary = Helion text-router（确认排序不反转）
- 冻结项：Qwen2.5-3B-Instruct；Helion world seed=13；R5/R5.1 seed 0–9 split；future=36；retention=24；budget=48；R2 三 primitive
- 复用：frontier / passive **采集轨迹**复用 R5（seed 0–2）与 R5.1 seed_extension（seed 3–9），在 R5.2 目录用 CountOnly unique-task 重评，不覆盖旧产物。Ace-Skill 10 seed **新采集**。
- 因此 R5.2 的 frontier / passive Δ 是 **CountOnly unique-task 重评**，与 R5.1 RealFit 不可逐格对比（例如 seed 0：R5.1 RealFit +0.500 vs R5.2 CountOnly +0.528）。方向与 R5.1 一致：CountOnly ≥ RealFit。
- GPU **4/5/6/7** 并行（seed 0,4,8 / 1,5,9 / 2,6 / 3,7）；`scripts/run_r5_2_baselines.py --resume`；墙钟约 7 min，watchdog 无 stall，exit 0
- Full future eval checkpoint = 0/16/32/48；中间 8/24/40 只记 activation / coverage
- 产物：`outputs/r5_2_baseline_collision/`（`R5_2_MASTER_SUMMARY.md`、`R5_2_DECISION.json`、`R5_2_FINAL_METRICS.csv`、`R5_2_LEARNING_CURVES.csv`、`R5_2_ACTIVATION_TIMES.csv`、`R5_2_ACQUISITION_DISTRIBUTION.csv`、`R5_2_EFFICIENCY.csv`、`R5_2_PAIRED_BOOTSTRAP.json`、`ACE_SKILL_IMPLEMENTATION_AUDIT.md`、`per_seed/`）
- 跑前冻结的 Case A 判据：`mean(frontier−Ace-Skill) ≥ 0.05` **且** CI 下界 > 0 **且** ≥7/10 胜出，并至少一项 sample-efficiency 优势（AULC 或更快激活第二个 target skill）

### 结果 1：10-seed 主比较（CountOnly unique-task；grain = 1/36 ≈ 0.028）

| seed | frontier Δ | Ace-Skill Δ | passive Δ | frontier−Ace | Ace−passive |
| ---: | ---------: | ----------: | --------: | -----------: | ----------: |
|    0 |     +0.528 |      +0.222 |    +0.222 |   **+0.306** |           0 |
|    1 |     +0.583 |      +0.278 |    +0.278 |   **+0.306** |           0 |
|    2 |     +0.528 |      +0.222 |    +0.222 |   **+0.306** |           0 |
|    3 |     +0.528 |      +0.222 |    +0.222 |   **+0.306** |           0 |
|    4 |     +0.528 |      +0.222 |    +0.222 |   **+0.306** |           0 |
|    5 |     +0.556 |      +0.250 |    +0.250 |   **+0.306** |           0 |
|    6 |     +0.528 |      +0.222 |    +0.222 |   **+0.306** |           0 |
|    7 |     +0.583 |      +0.278 |    +0.278 |   **+0.306** |           0 |
|    8 |     +0.583 |      +0.278 |    +0.278 |   **+0.306** |           0 |
|    9 |     +0.556 |      +0.250 |    +0.250 |   **+0.306** |           0 |

| 量                                |                  frontier |             Ace-Skill |               passive |
| --------------------------------- | ------------------------: | --------------------: | --------------------: |
| mean ΔFuture                      |                **+0.550** |                +0.244 |                +0.244 |
| mean #active target skills        |                   **2.0** |                   1.0 |                   1.0 |
| mean unique tasks / 48            |                      48.0 |                  46.1 |                  48.0 |
| mean duplicate rollouts           |                         0 |                   1.9 |                     0 |
| mean unique MH / temporal / S4    |     **37.7 / 0.0 / 10.3** |      14.1 / 2.7 / 1.2 |      16.3 / 2.0 / 0.6 |
| mean future EM MH / temporal / S4 | **1.000 / 0.383 / 1.000** | 1.000 / 0.383 / 0.083 | 1.000 / 0.383 / 0.083 |
| mean ΔRetention                   |                         0 |                     0 |                     0 |

Paired bootstrap（10 seed）：

| 对比                     |   mean gap | 95% CI                                            | win-rate       |
| ------------------------ | ---------: | ------------------------------------------------- | -------------- |
| **frontier − Ace-Skill** | **+0.306** | **[0.306, 0.306]**（点质量：每 seed 都是 +11/36） | **10/10**      |
| frontier − passive       |     +0.306 | [0.306, 0.306]                                    | 10/10          |
| Ace-Skill − passive      |  **0.000** | [0.000, 0.000]                                    | 0/10（10 tie） |

Ace-Skill 与 passive 的 ΔFuture **逐 seed 完全相同**。gap ≫ 一题粒度。

### 结果 2：Sample efficiency

| 指标                               | frontier                | Ace-Skill                     | passive       |
| ---------------------------------- | ----------------------- | ----------------------------- | ------------- |
| mean AULC（0/16/32/48 Future EM）  | **0.651**               | 0.343                         | 0.358         |
| mean budget to first target skill  | **8.7**（9/10 为 S4@8） | 28.2（仅 MH）                 | 24.5（仅 MH） |
| mean budget to second target skill | **17.8**                | **>48**（10/10 未激活第二个） | **>48**       |
| n_activated@48                     | 2                       | 1                             | 1             |

Frontier 更快解锁 **两个** target skill（S4 + MH）。Ace-Skill / passive 都只攒够 MH，从未激活 S4（unique S4 均值 1.2 / 0.6，< K=8）。

### 结果 3：Ace-Skill 是否因重复采样丢掉 unique evidence？

- 平均 duplicate_rollouts = **1.9**（最多 5），unique tasks = 46.1/48。
- 重复幅度小，**不是**输掉的原因。输在 **capability 分配**：Ace-Skill 的 Beta/sqrt-bias 按 instance proficiency 采，散到 S0/S2/少量 S4，MH 只约 14 条、S4 远不够 8；frontier 按 capability frontier 压 S1+S4（~38 MH + ~10 S4）。
- Unique-task 门控生效：Ace-Skill 的重复 rollout 没有被算进 K。

### 结果 4：Temporal 仍是盲区

三方法 temporal future EM 均为 **0.383**（= base，gain=0）。Ace-Skill 平均采到 2.7 条 unique S2，passive 2.0，frontier 0——都 <8，无人激活 temporal skill。与 R5.1 Track B 一致。

### 结果 5：Text-router sanity

Helion text-router 下 10-seed ΔFuture 与 oracle CountOnly **逐 seed 相同**（router val acc=1.0，无 question/vocab leakage）；frontier > Ace-Skill / passive **不反转**。这仍是模板可分世界，不升级为开放域 dispatch 证据。

### Case 判定（`R5_2_DECISION.json`）

| 条件                                            | 数字                     | 判定       |
| ----------------------------------------------- | ------------------------ | ---------- |
| mean(frontier−Ace-Skill) ≥ 0.05                 | +0.306                   | **PASS**   |
| CI 下界 > 0                                     | 0.306                    | **PASS**   |
| ≥7/10 胜                                        | **10/10**                | **PASS**   |
| AULC frontier > Ace-Skill **或** 更快第二 skill | 0.651>0.343；17.8 vs >48 | **PASS**   |
| **总判**                                        |                          | **Case A** |

Case B（持平 / CI 跨 0）与 Case C（Ace-Skill 明确更强）均未命中。

### 结论

- **Collision test 通过。** 在完全相同的 rollout budget、frozen CountOnly unique-task 后端、同一 10 个 frozen split 下，frontier 不是 Ace-Skill 类 generic prioritized sampling 的换壳。
- Ace-Skill 在本设定下 **退化成 passive**：instance-level proficiency 没有把预算压到「尚未 eligible 的 capability」，因此只激活 MH，ΔFuture 与随机流相同。
- 当前可写的独立 acquisition 贡献是 **capability-frontier allocation that gates which executable primitives become eligible**，不是「对难样本做 Beta 加权再采一遍」。
- Temporal 对所有 acquisition policy 仍是盲区；本轮按指令不为此堆 heuristic。
- **按 Case A：不要继续优化 Helion 内部**（不调 frontier 小权重、不调 K、不上 R6 / LoRA / RealFit）。下一步进入 **第二 benchmark / Search-Agent 外部泛化**，并尽量对接 SESA / SkillRL / SLIM 等公开协议。HASP 已证明 textual→executable，executable 本身不是 novelty。

**不宣称** 完整 ExPolicy、内容依赖 assimilation、或开放域 skill dispatch。

R5.2 当时按 Case A 进入第二 benchmark。**R5.3 已于 2026-08-15 按 `ExPolicy_R5_3_External_Generalization_Instructions.md` 跑完**（3-seed pilot stop / Case C），见下节。

---

## 2026-08-15 R5.3 External Generalization on Public Search Benchmarks（完成，Case C，3-seed stop）

### Setting

- 目的：R5.2 已在 Helion 上证明 `capability-frontier acquisition → experience-gated executable skill activation` 显著优于 Ace-Skill 与 passive。本轮**只问**该机制能否离开 Helion，泛化到公开、非模板化、社区常用的 Search-Agent benchmark。
- **不**调 Helion、不调 frontier 系数、不调 K、不训 R6、不重启 RealFit、不重抽 split、不用 mock retrieval。
- 环境：
  - 数据集：HotpotQA / 2WikiMultiHopQA / MuSiQue 主评；Bamboogle 仅作最终 zero-shot probe（Stage-1 失败后**未跑**）
  - 检索：Wikipedia 2018 100-word chunks + E5-base-v2 HNSW；corpus = index = **21,015,324**；真实 `query → retrieve → open → answer`
  - HF dump `wiki-18.jsonl.gz` 解压后是 tar，已抽出 `wiki-18.unpacked.jsonl`
  - 端口 **18080**（8000 被无关 FastAPI 占用，未抢）
- Capability taxonomy（公开任务内部共享，不用数据集名当 cap）：
  - C1 `bridge_decomposition`（HotpotQA bridge / 2Wiki compositional / MuSiQue 2-hop）
  - C2 `comparison_alignment`（HotpotQA / 2Wiki comparison）
  - C3 `long_chain_search`（MuSiQue 3–4 hop）
- Split（seed=13，冻结后未改）：每主集 calibration 48 / acquisition 400 / skill_validation 60 / future_probe 100；Bamboogle 125。task-id overlap = 0，question exact overlap = 0。hash(`split.json`) = `cf0cb8619317e0dd25e344fe8c0d883a96885a4b3f8dcaba8fed4804a3bee0f5`
- 后端：CountOnly unique-task **K=8**；frontier / Ace-Skill / gating 规则从 R5.2 **原样移植**（`FRONTIER_PORT_AUDIT.md`）。公开 cap 的 `transfer_potential` 全部是冻结的 **0.45**（Helion 名字才是 0.8）。
- 方法：`frontier` vs `ace_skill_prioritized` vs `passive`；budget=48；full eval 0/16/32/48
- 主指标：macro(HotpotQA, 2Wiki, MuSiQue) future accuracy；grain = 1/100 per dataset，macro 约 0.003
- 模型：Qwen2.5-3B-Instruct，T=0；GPU **4/5/6** 跑 seeds 0/1/2（GPU7 在 3-seed 映射下空闲）；未抢占卡上已有进程
- Stage-1 继续条件（跑前固定）：mean(frontier Δ) > Ace **且** > passive **且** ≥2/3 seed 赢 Ace **且** 更多 skill 或更高 AULC。不满足则 **停止，不扩 10 seed**
- 产物：`outputs/r5_3_external_generalization/`（`R5_3_MASTER_SUMMARY.md`、`R5_3_DECISION.json`、各 CSV、`SPLIT_AUDIT.md`、`CAPABILITY_MAP_AUDIT.md`、`FRONTIER_PORT_AUDIT.md`、`SKILL_VALIDATION.md`、`ROUTER_AUDIT.md`、`RETRIEVER_AUDIT.md`）
- 墙钟：约 10:30–11:09（含语料解包、retriever 加载、base_eval 749 题、skill val、3-seed acquire）

### 结果 1：环境 / skill / calibration（acquisition 之前）

Calibration（n=144，静态 base agent）：macro EM=0.028；C1=0.056 / C2=0.021 / C3=0.000。共享成本 1,897,105 tokens、971 tool calls。

Skill validation（`no_skill` vs executable procedure，至少 2/3 accept 才允许主实验）：

| capability              |    n | no_skill EM | skill EM |        ΔEM | improved | accept            |
| ----------------------- | ---: | ----------: | -------: | ---------: | -------: | ----------------- |
| C1 bridge_decomposition |   90 |       0.022 |    0.044 |     +0.022 |        3 | **yes**（刚过线） |
| C2 comparison_alignment |   60 |       0.000 |    0.417 | **+0.417** |       25 | **yes**           |
| C3 long_chain_search    |   30 |       0.000 |    0.067 |     +0.067 |        2 | **yes**（刚过线） |

**3/3 accepted。** C2 是唯一大正 transfer。Router（仅 calib+acq 标签）：TF-IDF+LR future acc=0.897；MiniLM=0.777。未进入 acquisition 主比较。

Base future（budget=0，三数据集各 100）：macro=**0.0233**（HotpotQA 0.04 / 2Wiki 0.02 / MuSiQue 0.01）。Bamboogle 静态 EM=0.008（仅环境记录）。

### 结果 2：3-seed acquisition pilot（主表）

| method                | final macro |     ΔFuture |   AULC | 第 2 个 skill | #active skills |
| --------------------- | ----------: | ----------: | -----: | ------------- | -------------- |
| **frontier**          |      0.0167 | **−0.0067** | 0.0178 | >48           | 1              |
| ace_skill_prioritized |      0.1044 |     +0.0811 | 0.0620 | 28.7          | 2–3            |
| passive               |      0.1056 |     +0.0822 | 0.0726 | 23.0          | 2–3            |

逐 seed ΔFuture：

| seed | frontier | Ace-Skill | passive | frontier−Ace | frontier−passive |
| ---: | -------: | --------: | ------: | -----------: | ---------------: |
|    0 |  −0.0067 |    +0.080 |  +0.083 |       −0.087 |           −0.090 |
|    1 |  −0.0067 |    +0.083 |  +0.080 |       −0.090 |           −0.087 |
|    2 |  −0.0067 |    +0.080 |  +0.083 |       −0.087 |           −0.090 |

Paired bootstrap（n=3）：

| 对比               |   mean gap | 95% CI                   | wins    |
| ------------------ | ---------: | ------------------------ | ------- |
| frontier − Ace     | **−0.088** | [−0.090, −0.087]         | **0/3** |
| frontier − passive | **−0.089** | [−0.090, −0.087]         | **0/3** |
| Ace − passive      |     −0.001 | [−0.003, +0.003]（过 0） | 1/3     |

Per-dataset final accuracy（3-seed mean）：

| dataset  | frontier |   Ace | passive |
| -------- | -------: | ----: | ------: |
| HotpotQA |     0.05 |  0.22 |    0.22 |
| 2Wiki    |     0.00 |  0.09 |    0.09 |
| MuSiQue  |     0.00 | 0.003 |   0.007 |

Per-capability future accuracy：frontier 三 cap 均为 0.02/0.02/0.00；Ace / passive 的 C2 达到 **0.28**（与 skill-val 正迁移一致）。

Learning curve（3-seed mean macro）：

| budget | frontier |   Ace | passive |
| -----: | -------: | ----: | ------: |
|      0 |    0.023 | 0.023 |   0.023 |
|     16 |    0.017 | 0.048 |   0.050 |
|     32 |    0.017 | 0.074 |   0.103 |
|     48 |    0.017 | 0.104 |   0.106 |

### 结果 3：Acquisition 行为（失败分析核心）

Frontier **三颗种子完全相同**：48/48 rollouts = HotpotQA `bridge_decomposition`；capability entropy ≈ 0；budget=8 激活 C1 后从未激活 C2/C3。

Ace / passive 跨 C1/C2/C3 和三数据集采样，因而能在 budget 内激活 C2（部分 seed 还激活 C3）。

冻结 Helion diagnoser 在公开 taxonomy 上的 degenerate mode：

- `learnability = 4s(1−s)`：calib C1=0.056、C2=0.021、C3=0.0 → **C3 learnability=0**，C1 最高
- `transfer_potential` 对所有公开 cap 都是 **0.45**
- novelty 下降不足以在 budget=48 内把采样从 C1 推开

谁采到 C2，future macro 就从 ~0.02 跳到 ~0.10。Frontier 没采到 C2，激活弱 C1 还略伤 base（Δ=−0.0067）。

### Stage-1 门控 / Case 判定

| 条件                       | 数字                              | 判定               |
| -------------------------- | --------------------------------- | ------------------ |
| mean(frontier Δ) > Ace     | −0.0067 vs +0.081                 | **FAIL**           |
| mean(frontier Δ) > passive | −0.0067 vs +0.082                 | **FAIL**           |
| frontier ≥2/3 seed > Ace   | **0/3**                           | **FAIL**           |
| 更多 skill 或更高 AULC     | 1 skill / AULC 0.018 vs Ace 0.062 | **FAIL**           |
| **Stage-1**                |                                   | **STOP**           |
| 10-seed / Bamboogle        |                                   | **未跑**（按指令） |

`R5_3_DECISION.json`：**Case C**。Frontier 不优于 passive。这是 3-seed 上的预注册 stop，不是筛 seed 后的 10-seed 正式 Case C；方向已无歧义（三 seed 点质量相同），扩 10 seed 不会翻盘。

### 结论

- **Helion 的 `frontier > Ace-Skill` 没有在公开 Search-Agent benchmark 上复现。** Skill backend 成立（3/3 accept），acquisition policy 失败。
- 失败不是 mock retrieval、不是 split 泄漏、不是 skill 写错：检索是 Wikipedia 2018 + E5；split 已冻结；C2 procedure 有 +0.417 EM。失败是 **冻结 Helion frontier score 把预算全部压进 C1/HotpotQA**，系统性错过唯一有用的 C2。
- Ace-Skill 在公开环境中仍 ≈ passive（CI 过 0），但方向与 Helion 相反：均匀覆盖更能激活 C2，passive 略强或持平。
- 该结果支持 **Helion-specific heuristic**，不支持把 capability-frontier acquisition 写成论文主结果。
- **按 Case C 停止：** 不调 public 小权重 / K / frontier coefficient；不重抽 split；不扩 10 seed；不跑 Bamboogle；不堆更多 benchmark。暂停论文主线扩张。当时写下的下一步（evidence-sufficiency gating，而不是 count-only K=8 + Helion learnability）已由 **R5.4 RASA** 执行，并在 Case E 被证伪，见下节。

**不宣称** 外部泛化通过、不宣称 10-seed 正式 Case C（未跑满 10 seed）、不宣称 Bamboogle transfer。

R5.3 当时按 Case C 停止扩 frontier。**R5.4 已于 2026-08-15 按 `ExPolicy_R5_4_RASA_Redesign_Run_Instructions.md` 跑完**（3-seed Stage-1 / Case E），见下节。

---

## 2026-08-15 R5.4 RASA Redesign（完成，Case E，3-seed stop）

### Setting

- 目的：R5.3 已关闭 Helion-derived `frontier_score` 主线。本轮**不再调系数 / K / 重抽 R5.3 split**，而是做方法级 redesign：**RASA — Remediability-Aware Sequential Acquisition**。核心问题从「哪些题难」改成「哪些能力缺口既存在又可被当前 frozen skill 修复」。
- 允许验证的唯一 claim：给定固定、已验证的 executable skill bank 与有限 stream budget，主动采新经验以估计 `q=P(base fails|c)` 与 `p=P(skill repairs|base failed,c)`，并顺序决定 **ACTIVATE / CONTINUE / PARK**。
- **禁止 claim**：skill 创建/改写、evidence-gated promotion、executable skill 本身、self-play、frontier curriculum、generic failure sampling。
- Novelty audit（`outputs/r5_4_rasa/NOVELTY_AUDIT.md`）：截至 2026-08-15，无工作同时满足 (1) 固定 skill bank (2) 主动采新经验 (3) 显式 `q` 与 `p` (4) 用该估计做 ACTIVATE/CONTINUE/PARK。**PROCEED**。最近邻风险：Bayesian-Agent / RADEG / SLIM / MSCE，均缺四件套中的若干项。
- 环境：复用 R5.3 的 Wikipedia 2018 + E5-base-v2（corpus=index=21,015,324；端口 18080）；Qwen2.5-3B-Instruct；T=0；C1/C2/C3 公开 taxonomy；R5.3 frozen executable primitives（byte/hash 冻结）。
- **新 holdout，禁止复用 R5.3 future 当主结果。** `split_seed=20260815`；每数据集 acquisition_pool=300、future_test=150。与 R5.3 全部 split 零 task-id / 零 exact-question overlap；R5.4 acq∩future=0。hash(`split_r5_4.json`) = `015250d6083bffb411cf852bac7f7f211e1593d93b547f1ba2c683904dd27a9f`。R5.3 产物未覆盖。
- Dispatch：deployable TF-IDF + multinomial LR，**只**用 R5.3 calibration+acquisition 的 question text + capability label 训练（n=1344）。R5.3 heldout acc=0.903 / skill-val acc=0.90；无 task-id leakage。R5.4 future 标签不进拟合。
- RASA 算法（阈值跑前冻结，失败不改）：
  - 每 cap：`q~Beta(1,1)`，`p~Beta(1,1)`；clean base 更新 `q`；仅失败时用 frozen skill replay 更新 `p`；`z=q·p`，4096 次 MC
  - Evidence diversity：acquisition 题面 E5 embedding 上 per-cap KMeans（k=4，`random_state=13`）；ACTIVATE 要求 `cluster_coverage≥2`；within-cap 优先最少覆盖 cluster + 最大 cosine 距离，无 replacement
  - Sequential Readiness Test：`delta=0.10`，`confidence=0.90`。ACTIVATE：`P(z≥0.10)≥0.90` 且失败≥4 且 cluster≥2。PARK：`P_good≤0.10` 且失败≥4，**或**失败≥8 且 `repair_count=0`。其余 UNRESOLVED
  - Acquisition：每 cap 最多 2 题 warm start（上限 6）；之后对 UNRESOLVED cap 做 Thompson `argmax z_sample`；ACTIVE/PARKED 不再采；全 terminal 则 early stop
- Budget：`B_acq=48` unique tasks；skill remediation replay **不计 unique experience**，但计入 tokens / tools / wall / `diagnostic_replay_count`。Checkpoint：0/16/32/48
- Stage-1 因果矩阵（8 条件 × seeds 0,1,2，同一 frozen split）：

| condition            | acquisition | activation             | 目的                              |
| -------------------- | ----------- | ---------------------- | --------------------------------- |
| `passive_k8`         | passive     | CountOnly K=8          | 随机 baseline                     |
| `ace_k8`             | Ace-Skill   | CountOnly K=8          | 最强 generic prioritized baseline |
| `legacy_frontier_k8` | 旧 frontier | CountOnly K=8          | R5.3 失败方法保留                 |
| `rasa_k8`            | RASA `q×p`  | CountOnly K=8          | 只测新 acquisition                |
| `passive_srt`        | passive     | SRT                    | 只测新 gate                       |
| `rasa_full`          | RASA `q×p`  | SRT                    | 完整方法                          |
| `rasa_qonly`         | `z=q` 采样  | SRT（激活仍用 `q·p`）  | 去掉 remediability                |
| `rasa_nodiversity`   | RASA `q×p`  | SRT，within-cap random | 去掉 diversity                    |

- GPU **4/5/6/7** 顺序 wrapper（每卡两个条件串行，四卡并行）：GPU4 `passive_k8→ace_k8`；GPU5 `legacy_frontier_k8→rasa_k8`；GPU6 `passive_srt→rasa_full`；GPU7 `rasa_qonly→rasa_nodiversity`。`--resume` 只跳过 hash/metrics 完整的 `(condition, seed)`。
- Protocol audit `scripts/audit_r5_4_protocol.py`：**26/26 PASS** 后才开 GPU。
- 主指标：macro(HotpotQA, 2Wiki, MuSiQue) future accuracy，各 150 题，grain ≈ 1/450 ≈ 0.0022；Gate A 的 +0.02 ≈ 9 题。
- 墙钟：prep 14:30；base_eval 14:36–14:49（四卡切 450 future）；Stage-1 14:49–15:32；analyze 15:32。约 43 min GPU。
- 产物：`outputs/r5_4_rasa/`（`NOVELTY_AUDIT.md`、`SPLIT_AUDIT.md`、`ROUTER_AUDIT.md`、`PROTOCOL_AUDIT.md`、`R5_4_STAGE1_MASTER.csv`、`R5_4_LEARNING_CURVES.csv`、`R5_4_POSTERIORS.csv`、`R5_4_DECISIONS.csv`、`R5_4_COST.csv`、`R5_4_GATE_DECISION.json`、`R5_4_MASTER_SUMMARY.md`、`per_seed/`、`logs/`）
- Stage-1 Go（跑前固定，须同时满足）：A `mean(rasa_full) ≥ max(passive,ace)+0.02` 且 3/3 胜；B mean ΔFuture>0；C full−passive_srt≥0.01；D full−rasa_k8≥0.01；E full−qonly≥0.01 或 AULC/terminal-budget 优势；F ≥2/3 cap 到达 ACTIVE/PARKED，禁止 48/48 砸在单一 UNRESOLVED cap。**只有 CASE_A_PASS 才允许 Stage-2（seed 3–9 + Bamboogle）。**

### 结果 1：Stage-1 主表（3-seed mean；clean base macro = 0.0133）

| condition          | final macro | ΔFuture | n_active | n_terminal | HotpotQA | 2Wiki |     MuSiQue |
| ------------------ | ----------: | ------: | -------: | ---------: | -------: | ----: | ----------: |
| **passive_k8**     |  **0.1200** | +0.1067 |        3 |          0 |    0.193 | 0.133 |   **0.033** |
| **ace_k8**         |  **0.1200** | +0.1067 |        3 |          0 |    0.193 | 0.133 |   **0.033** |
| legacy_frontier_k8 |      0.0156 | +0.0022 |        1 |          0 |    0.020 | 0.013 |       0.013 |
| rasa_k8            |      0.1111 | +0.0978 |        1 |          0 |    0.187 | 0.140 |       0.007 |
| passive_srt        |      0.1119 | +0.0985 |      1–2 |        1–3 |    0.187 | 0.140 | 0.007–0.013 |
| **rasa_full**      |  **0.1111** | +0.0978 |        1 |          2 |    0.187 | 0.140 |       0.007 |
| rasa_qonly         |      0.1111 | +0.0978 |        1 |          2 |    0.187 | 0.140 |       0.007 |
| rasa_nodiversity   |      0.1111 | +0.0978 |        1 |          3 |    0.187 | 0.140 |       0.007 |

逐 seed：passive / Ace 三 seed 全是 0.120；`rasa_full` / `rasa_k8` / `rasa_qonly` 三 seed 全是 0.111；`legacy_frontier` 三 seed 全是 0.016。`passive_srt` seed 2 为 0.113（多激活了 C3）。`rasa_nodiversity` seed 1/2 在 unique 25/22 时三 cap 全 terminal，early stop。

### 结果 2：Learning curve（3-seed mean macro）

| budget |  passive_k8 |      ace_k8 | legacy |   rasa_k8 | passive_srt | rasa_full | rasa_qonly |
| -----: | ----------: | ----------: | -----: | --------: | ----------: | --------: | ---------: |
|      0 |       0.013 |       0.013 |  0.013 |     0.013 |       0.013 |     0.013 |      0.013 |
|     16 | 0.013–0.020 | 0.020–0.111 |  0.016 | **0.111** | 0.013–0.111 | **0.111** |  **0.111** |
|     32 | 0.020–0.120 | 0.118–0.120 |  0.016 |     0.111 | 0.111–0.113 |     0.111 |      0.111 |
|     48 |   **0.120** |   **0.120** |  0.016 |     0.111 | 0.111–0.113 |     0.111 |      0.111 |

Ace 在 budget=16 已能激活 C2（macro 跳到 0.111），budget=32–48 再激活 C1/C3 涨到 0.120。RASA 家族在 C2 ACTIVE 后曲线持平，从未拿到 C3 的 MuSiQue 增益。

### 结果 3：RASA 后验与决策（`rasa_full`，三 seed 同构）

| cap                       | state          | budget_to_terminal | n_fail | repair | mean q | mean p | mean z |    P_good | 解读                                                     |
| ------------------------- | -------------- | -----------------: | -----: | -----: | -----: | -----: | -----: | --------: | -------------------------------------------------------- |
| C2 `comparison_alignment` | **ACTIVE**     |        9 / 11 / 10 |      5 |  **2** |   0.86 |   0.43 |   0.37 | **0.974** | 可修复缺口，正确激活                                     |
| C1 `bridge_decomposition` | **PARKED**     |       17 / 19 / 17 |      8 |  **0** |   0.90 |   0.10 |   0.09 |     0.349 | futility：8 次零修复                                     |
| C3 `long_chain_search`    | **UNRESOLVED** |                  — |     34 |  **1** |   0.95 |  0.056 |  0.053 | **0.105** | `P_good` 刚压过 0.10；1 次偶然修复挡住「8 次零修复」规则 |

C2 ACTIVE + C1 PARKED 之后，只剩 C3 为 UNRESOLVED，Thompson 把剩余约 **35/48** unique tasks 全部砸进 long-chain。Gate F 仍过（2/3 terminal），但 acquisition 效率被 C3 吃掉。

`rasa_k8` 没有 SRT PARK：同一 `q×p` 采样持续压 C2（约 39/48），CountOnly 只激活 C2（C1/C3 unique <8），final 与 `rasa_full` 相同。`rasa_qonly` 的终态与 `rasa_full` 逐 cap 相同（full−qonly=0）。

### 结果 4：成本（3-seed；replay 不计 unique experience）

| condition        | unique | diagnostic replays | executor episodes |   tokens |   tools |  wall s |
| ---------------- | -----: | -----------------: | ----------------: | -------: | ------: | ------: |
| passive_k8       |     48 |                  0 |                48 |    ~628k |    ~319 |    ~423 |
| ace_k8           |     48 |                  0 |                48 |    ~636k |    ~327 |    ~430 |
| rasa_full        |     48 |             **47** |            **95** |    ~684k |    ~659 |    ~350 |
| rasa_qonly       |     48 |                 47 |                95 |    ~681k |    ~657 |    ~356 |
| rasa_nodiversity |  22–48 |              21–45 |             43–93 | 288–615k | 260–515 | 164–388 |

RASA 的 remediation replay 使 executor episodes 约翻倍。论文若继续，必须同时报 new-experience efficiency 与 compute-normalized efficiency；本轮主结果已失败，不再展开。

### Stage-1 闸门 / Case 判定（`R5_4_GATE_DECISION.json`）

| Gate                          | 标准                                            | 数字                                       | 判定     |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------ | -------- |
| A 最终能力                    | mean(full) ≥ max(passive,ace)+0.02 且 3/3 胜    | 0.111 vs 0.120，margin **−0.009**，**0/3** | **FAIL** |
| B 正增益                      | mean ΔFuture > 0                                | **+0.098**                                 | **PASS** |
| C 不只靠 gate                 | full − passive_srt ≥ 0.01                       | **−0.0007**                                | **FAIL** |
| D 不只靠新 acquisition + 旧 K | full − rasa_k8 ≥ 0.01                           | **0.000**                                  | **FAIL** |
| E remediability 因果          | full − qonly ≥ 0.01 或 AULC/terminal 优势       | **0.000**；无 AULC/term edge               | **FAIL** |
| F 不塌缩                      | ≥2/3 cap terminal；禁止 48/48 砸单一 UNRESOLVED | 2/3 terminal（C1 PARK + C2 ACTIVE）        | **PASS** |
| **总判**                      | A∧B∧C∧D∧E∧F                                     |                                            | **FAIL** |

脚本自动分型：**`CASE_E_METHOD_FAIL`**（full 不优于最强 baseline）。同时满足 Case C 味道（full ≈ `passive_srt`）与 Case D 味道（full = qonly），但决策树按指令优先 Case E。Stage-2 **未跑**。

### 结论

- **RASA 在新冻结公开 holdout 上被证伪。** 完整方法最终能力低于 Ace-Skill / passive（0.111 vs 0.120），0/3 seed 赢；+0.02 门槛未过。按 §14 Case E：**停止这条论文线。**
- 机制并非「什么都没学到」。SRT **正确**把可修的 C2 激活、把 8 次零修复的 C1 停掉；相对 R5.3 的 48/48 C1 塌缩，这是真实的诊断进步。失败在第三步：C3 的 `P_good=0.105` 略高于 PARK 线 0.10，且 1 次偶然修复挡住 futility，于是把大部分预算继续砸在几乎不可修的 long-chain 上。Ace/passive 靠 CountOnly 覆盖把三个 cap 都打开，MuSiQue 从 0.007 涨到 0.033，最终 macro 更高。
- Ablation 否定独立贡献：`rasa_full = rasa_k8 = rasa_qonly`；`rasa_full ≈ passive_srt`。Remediability 没有把 acquisition 从 failure-aware / gate-only 里分开。按指令：**不要改写成 “failure-aware acquisition” 或 “evidence-gated promotion” 论文。**
- `legacy_frontier_k8` 再次几乎无效（Δ=+0.002），与 R5.3 Case C 一致，作为失败 baseline 保留即可。
- **按 Case E 停止 RASA 主线：** 不改 `delta` / `confidence` / 4-failure / 8-zero-repair / cluster=2；不重抽 split；不扩 seed 3–9；不跑 Bamboogle；不把 AULC、单 dataset 或 Helion 旧结果改写成主论文。R5.3 的 Case C 与 R5.4 的 Case E 一并写入历史负结果。后续按 `ExPolicy_R6_AVA_Next_Run_Instructions.md` 另开 Exact AVA（见下节），**不是**对 RASA 的阈值修补。

**不宣称** RASA 成立、不宣称 remediability-aware acquisition 已被证明、不宣称 SRT 相对 CountOnly 有独立增益。Stage-2 / 10-seed / Bamboogle **未跑**（按预注册 stop）。

---

## 2026-08-15 R6 Exact AVA Stage-1（完成，CASE_C，3-seed stop）

### Setting

- 目的：R5.3 证伪 Helion frontier 泛化，R5.4 证伪 `q×p`/RASA。本轮**不再发明第三个 hand-crafted proxy**，只验证核心 principle：经验价值是否应由 **当前 assimilator 真正 assimilate 之后的 held-out capability gain** 定义。工作名 **AVA — Assimilation-Validated Acquisition**。
- 正式量：\(V(B\mid S,A)=J(A(S,B);D_{\text{shadow}})-J(S;D_{\text{shadow}})\)。Stage-1 只测昂贵 exact oracle，不加 \(\lambda\) cost，不训 selector。
- **允许验证的唯一问句**：在冻结的公开 skill bank + text-router + CountOnly unique-evidence activation 上，exact post-assimilation shadow utility 能否指导 acquisition，使其在未触碰的 R5.4 future_test 上超过 Ace/passive。
- **禁止**：改 frontier 系数、新的 `q×p×novelty`、调 RASA PARK、用 dataset/oracle cap 做主方法、用 future-test 训练 selector、exact AVA 未过就训 RL、重抽 R5.4 future holdout、shadow 看 final future。
- Novelty collision audit（截至 2026-08-15，`outputs/r6_ava/audit/NOVELTY_COLLISION.md`）：硬停止条件为同时满足 (1) 选择对象是未来 env experience batch (2) temporary/shadow assimilation (3) 值是 post-assimilation held-out future gain (4) 用该值决定下一批 acquisition。列出的 Ace-Skill / Bayesian-Agent / SLIM / EvoLib / SkillRL / SESA / SkillHEX / Harness-R1 / SkillSV / PROGRESS / SkillOpt 等均缺至少一项。最接近为 SkillOpt（shadow 的是 skill *edit*，不是 env batch）。**`NOVELTY_STATUS = PROCEED`。**
- Pipeline audit `scripts/audit_r6_pipeline.py`：**27/27 PASS**。R5.3 48/48 C1 确为分数排序；`rasa_full` 每次 acquisition 后更新 q/p；`repair_count` 只在 base failure 上更新；future 未进 router/RASA；seed 传入 RASA，三 seed 同构来自冻结 skill + 同一激活 mask。
- Case analysis `scripts/analyze_r5_4_cases_for_r6.py`：136 cases。Q1=**B+C**（C3 稀疏可修 + replay `p` ≠ 部署边际效用）。Q2：CountOnly 激活 C3 让 MuSiQue 0.007→0.033，RASA 因 cap-level `p` 不 PARK/不 ACTIVE C3。Q3：task-level `q×p` vs `skill_repaired` Spearman=**0.291**；cap-level 0.753 是 3 个 cap 的离散点，不是经验排序。停车 C3 仍是 C2-only≈0.111<0.120，故**不是单一阈值 bug**。**ENTER_R6=true**。
- 环境：复用 R5.3/R5.4 Wikipedia 2018 + E5（corpus=index=21,015,324；`:18080`）；Qwen2.5-3B-Instruct；T=0；冻结 R5.3 primitives + R5.4 text-router pickle（不重训）。
- **Split（写一次，禁止重抽）**：从 R5.4 每数据集 300 acquisition 内切 `candidate=240` / `shadow=60`；**R5.4 future_probe 450 原样拷贝**。seed=`20260816`。sha256(`split_r6_ava.json`)=`bebc932be6a743be8b55372b6de567a77a76d5c68b45631ff9cf8afc9b13bb6a`。candidate/shadow/future 两两 task-id 与 exact-question overlap = 0。
- Exact AVA 候选构造：剩余 pool 上 E5 k-means，每轮 `M=8` 个 micro-batch，每批 8 unique tasks；覆盖用、不赋价值分。Selector 只取 \(\Delta\)ShadowFuture 最大的 batch 提交。CountOnly \(A\) 只依赖 unique (task_id, predicted_cap) 计数，shadow 评分因此是激活 mask 的确定性函数，8 个 mask 预计算后跨 seed 复用。提交 batch 仍真实 LLM acquire（可复用 R5.4 同 task 轨迹缓存）。
- Clone isolation：shadow 只碰 `EvidenceState.clone()`；单测覆盖 live counts / skill mask / RNG / B1↛B2 / future id 不进 selector / router 不重训。`pytest` 14/14 PASS 后才开 Stage-1。
- Budget：`B_acq=48` unique；6 rounds × 8。Checkpoint：0/8/16/24/32/40/48。主指标：macro(HotpotQA, 2Wiki, MuSiQue) on **untouched R5.4 future_test**（grain ≈ 0.0022；Gate A 的 +0.02 ≈ 9 题）。
- Stage-1 条件（同一 split、同一 48 unique、同一冻结 backend）：

| condition               | acquisition                  | activation    | 角色                  |
| ----------------------- | ---------------------------- | ------------- | --------------------- |
| `exact_ava`             | shadow \(\Delta J\) 选 batch | CountOnly K=8 | 主方法                |
| `passive`               | 随机                         | CountOnly K=8 | 最强对照之一          |
| `ace_skill_prioritized` | Ace-Skill                    | CountOnly K=8 | 最强 generic baseline |
| `legacy_frontier`       | 旧 frontier                  | CountOnly K=8 | R5.3 失败对照         |
| `rasa_full`             | RASA `q×p`                   | SRT           | R5.4 失败对照         |

- GPU **4/5/6/7**：先四卡切 shadow base_eval（180）与 7 个非空 mask 的 skill eval；再 GPU4/5/6 并行 `exact_ava` seed 0/1/2，GPU7 串行 baselines。AVA 三 seed 先完成后把 Ace / frontier / RASA 拆到空闲的 4/5/6，四卡打满。`--resume` + per-seed flock。Watchdog 15 min `STATUS_LIVE` stall → kill/resume。
- git HEAD（跑时）：`5c9dbf22eab3b43c6c65433037def2fb927d3b2f`。
- 墙钟：22:30 orchestrate → 22:31–22:37 shadow base（180 行）→ 22:37–22:40 mask cache → 22:40–22:55 Stage-1 + analyze。约 25 min GPU（轨迹缓存命中加速 acquire）。可选 `random_cluster` **未跑**（指令标 optional）。
- 产物：`outputs/r6_ava/`（`audit/NOVELTY_COLLISION.md`、`R6_PIPELINE_AUDIT.md`、`case_analysis/CASE_REPORT.md`、`SPLIT_AUDIT.md`、`R6_STAGE1_MASTER.csv`、`R6_LEARNING_CURVES.csv`、`R6_GATE_DECISION.json`、`R6_STAGE1_MASTER.md`、`R6_SHADOW_RANK_CORRELATION.csv`、`R6_DATASET_CAP_DISTRIBUTION.csv`、`R6_COST.csv`、`per_seed/`、`mask_eval/`、`logs/`）
- Stage-1 Go（跑前冻结，须 A∧B∧C∧D∧E∧F）：A `mean(exact_ava) ≥ max(passive,ace)+0.02`；B 3/3 seed 胜；C AULC+0.02 或少 8 条 unique 达到 baseline final；D Spearman(shadow Δ, downstream)≥0.40 或 top-1 显著高于 random；E 不退化成 always-2Wiki / always-C2；F 无 final-test leakage。**只有 CASE_A_EXACT_AVA_PASS 才允许 surrogate。**

### 结果 0：冻结 CountOnly 下 future 是激活 mask 的确定性函数

预计算 8 个 mask 在 R5.4 future_test（n=450）上的 macro（与 R5.4 已发表数字一致）：

| 激活 mask    | future macro | HotpotQA | 2Wiki |   MuSiQue |
| ------------ | -----------: | -------: | ----: | --------: |
| none（base） |       0.0133 |    0.020 | 0.013 |     0.007 |
| C1 only      |       0.0200 |    0.027 | 0.007 |     0.027 |
| C3 only      |       0.0156 |    0.020 | 0.013 |     0.013 |
| **C2 only**  |   **0.1111** |    0.187 | 0.140 |     0.007 |
| C1+C2        |       0.1178 |    0.193 | 0.133 |     0.027 |
| C2+C3        |       0.1133 |    0.187 | 0.140 |     0.013 |
| **C1+C2+C3** |   **0.1200** |    0.193 | 0.133 | **0.033** |

Shadow probe（n=180）上同样是 C2 主导：none 0.028 → C2 0.183 → 全开 0.194。因此 **exact AVA 最多把终局推到 0.120**；Gate A 要求的 +0.02 在这个冻结 \(A\) 上不可达，除非 mask 之外还有内容依赖——而 CountOnly 没有。

### 结果 1：Stage-1 主表（3-seed；clean base macro = 0.0133）

| condition                 | final macro | ΔFuture |  AULC | 终局 mask             | HotpotQA | 2Wiki |               MuSiQue |
| ------------------------- | ----------: | ------: | ----: | --------------------- | -------: | ----: | --------------------: |
| **exact_ava**             |  **0.1200** | +0.1067 | 0.088 | C1+C2+C3（3/3）       |    0.193 | 0.133 |                 0.033 |
| **passive**               |  **0.1193** | +0.1060 | 0.069 | 2/3 全开；seed2 缺 C3 |    0.193 | 0.133 | 0.033 / 0.033 / 0.027 |
| **ace_skill_prioritized** |  **0.1193** | +0.1060 | 0.064 | 2/3 全开；seed2 缺 C3 |    0.193 | 0.133 | 0.033 / 0.033 / 0.027 |
| legacy_frontier           |      0.1133 | +0.1000 | 0.088 | C2+C3（3/3）          |    0.187 | 0.140 |                 0.013 |
| rasa_full                 |      0.1111 | +0.0978 | 0.043 | 仅 C2（3/3）          |    0.187 | 0.140 |                 0.007 |

逐 seed final：AVA 三 seed 全 0.120；passive/Ace seed 0/1 = 0.120、seed 2 = 0.1178（未激活 C3）；frontier 三 seed 全 0.1133；rasa_full 三 seed 全 0.1111。相对 per-seed 最强 baseline 的 paired gap：0 / 0 / +0.0022 → **1/3 胜**。

AVA 采集分布（3 seed 合计 144 unique）：2Wiki 72 / MuSiQue 59 / HotpotQA 13；router cap C1 57 / C3 55 / C2 32。不是 always-2Wiki 或 always-C2。

### 结果 2：Learning curve（per-seed macro）

| budget |    AVA s0 |    AVA s1 |    AVA s2 | passive s0 |    Ace s0 |   rasa s0 |
| -----: | --------: | --------: | --------: | ---------: | --------: | --------: |
|      0 |     0.013 |     0.013 |     0.013 |      0.013 |     0.013 |     0.013 |
|      8 |     0.020 | **0.111** | **0.111** |      0.013 |     0.013 |     0.013 |
|     16 |     0.022 |     0.118 |     0.118 |  **0.111** |     0.020 |     0.013 |
|     24 |     0.022 | **0.120** | **0.120** |      0.111 |     0.020 |     0.013 |
|     32 |     0.022 |     0.120 |     0.120 |  **0.120** |     0.020 |     0.013 |
|     40 | **0.120** |     0.120 |     0.120 |      0.120 | **0.120** |     0.013 |
|     48 |     0.120 |     0.120 |     0.120 |      0.120 |     0.120 | **0.111** |

AVA seed 1/2 在 budget=8 就激活 C2（shadow 选对了高 \(\Delta\) batch），budget=24 全开。Seed 0 聚类长时间停在近乎无用的 C1，直到 40 才跳到 0.120，把 mean AULC 拉到 0.088。相对 best-baseline AULC 0.069，差值 **+0.019 < 0.02**，Gate C 未过。

### 结果 3：Shadow 有效性与泄漏

- 每轮候选里 shadow \(\Delta\) 最大的 batch **总是被提交**（top-1 hit = 18/18 = 1.0）。这只说明 selector 按定义执行，不说明 shadow 与 *committed-batch 的最终 downstream 增量* 有 ≥0.40 的排序相关（记录的 Spearman proxy=0.334；Gate D 因 top-1≫random 记 PASS）。
- Future ids / questions 未进入 candidate、shadow、router。Split 审计与 `tests/test_r6_no_future_leakage.py` PASS。Gate F PASS。

### 结果 4：成本（3-seed mean）

| condition             | unique | tokens | wall s |
| --------------------- | -----: | -----: | -----: |
| exact_ava             |     48 |  ~643k |   ~202 |
| passive               |     48 |  ~621k |   ~154 |
| ace_skill_prioritized |     48 |  ~646k |   ~170 |
| legacy_frontier       |     48 |  ~642k |    ~77 |
| rasa_full             |     48 |  ~643k |    ~72 |

AVA 的额外成本在 mask 预计算（一次、四卡共享），不在 per-seed 48 条 acquire。RASA 墙钟短是因为 SRT 轨迹缓存 + 终局只装 C2。

### Stage-1 闸门 / Case 判定（`R6_GATE_DECISION.json`）

| Gate          | 标准                                      | 数字                                    | 判定     |
| ------------- | ----------------------------------------- | --------------------------------------- | -------- |
| A 最终能力    | mean(AVA) ≥ max(passive,ace)+0.02         | 0.1200 vs 0.1193，margin **+0.0007**    | **FAIL** |
| B seed 一致性 | 3/3 严格胜过该 seed 的 max(passive,ace)   | **1/3**（仅 seed 2，因 baseline 缺 C3） | **FAIL** |
| C 样本效率    | AULC+0.02 或少 8 unique 达 baseline final | AULC +0.019；seed 0 拖到 budget 40      | **FAIL** |
| D shadow 有效 | Spearman≥0.40 或 top-1≫random             | Spearman 0.33；**top-1=1.0**            | **PASS** |
| E 无 shortcut | 不 always-2Wiki / always-C2               | 2Wiki 50%；C2 仅 32/144                 | **PASS** |
| F 无 leakage  | future 不进 selector/shadow               | split+tests PASS                        | **PASS** |
| **总判**      | A∧B∧C∧D∧E∧F                               |                                         | **FAIL** |

脚本自动分型：**`CASE_C_EXACT_AVA_FAIL`**。Surrogate / 10-seed / Bamboogle / 第二 assimilator / matched-vs-mismatched **未跑**。

### 结论

- **Exact AVA 在冻结 CountOnly 公开 holdout 上没有可部署增益。** 终局 0.120 只是「把三个 frozen skill 都打开」，Ace/passive 同样能做到；要求的 +0.02 与 3/3 seed 均未过。按指令 CASE_C：**立即停止 ExPolicy acquisition-policy 主线。**
- 这不是实现事故。Mask 表证明：在当前 \(A\) 下 \(J\) 几乎是激活集合的查表。Shadow oracle 最多加速「何时打开 C2」（seed 1/2 在 budget=8 做到了），不能创造 CountOnly 没有的能力。Seed 0 还说明无监督聚类并不保证早开 C2。
- 与 R5.3/R5.4 一致的机制图：真正有用的是 **C2 primitive**（Δ≈+0.098），C1/C3 只再加约 +0.009。Frontier 采 C1、RASA 卡在 C3 而不装 skill、AVA 用 shadow 去选「打开哪些 cap」——三者都没有超出「count-gated 解锁冻结 procedure」。
- **按 CASE_C 停止：** 不改 shadow 组成 / M=8 / batch=8 / λ / 聚类以求 C2；不重定义 macro；不换 seed；不写成「在部分能力上有效」；**不训练 AVA-Surrogate**。可保留 exact-utility / mask 表作为内部 insight，不单独弱化成论文。

**不宣称** assimilation-conditioned experience value 已被证明、不宣称 exact AVA 可部署、不宣称需要 learned selector。P5–P7 **未跑**（按预注册 stop）。

---

## 2026-08-19 SkillRL-ESI MVP（完成，CASE_ESI_PASS，内容因果未过）

按 `SkillRL_ESI_MVP_8xH20_Execution_Instructions.md`。R6 已停止 ExPolicy acquisition 主线；本轮只回答：在**已验证可工作的 SkillRL Search backbone** 上，真实 train-side experience 能否被转换成新的 executable skill，使冻结 base actor 在未见 Search split 上变强。

不问：Active Acquisition、正式 RL 性能、多 benchmark、完整论文。

### Setting

- **科学问句：** `experience → typed SkillSpec → compiler → SearchEnv execution` 能否让冻结 3B actor 在 held-out comparison 2Wiki 上优于 learning 前（ΔEM ≥ +0.03）。
- **backbone：** SkillRL `https://github.com/aiming-lab/SkillRL` commit `8e66726ed866a4e0a7f053586a41022798192e6c`，工作分支 `esi-mvp`。主代码 `/data/ppnm/skillrl_esi`；**不**把 ESI 塞回 `/data/ppnm/expolicy`，也**不**导入 R2/R5 C2 frozen primitive / CountOnly。
- **环境：** conda `skillrl-esi`（Python 3.10.20，torch 2.8.0+cu128，vllm 0.11.0，flash-attn 2.7.4.post1）。
- **Actor：** `/data/ppnm/models/Qwen2.5-3B-Instruct`，权重冻结；eval `temperature=0`，`max_model_len=8192`。
- **Search：** SkillRL `examples/search/retriever/retrieval_server.py` `:8030`；wiki-18 E5 HNSW + `e5-base-v2`；FAISS CPU，encoder 在 GPU 4。旧 ExPolicy `:18080` 未用。所有 SEARCH/ANSWER 经 `SearchMultiProcessEnv.step`，禁止直连 retriever。
- **GPU：** 只用 4–7（0–3 为他人 casbot/openpi，未杀）。G0.8 在 GPU 5，Ray 隔离到 `/tmp/esi_g08`。
- **数据集：** SkillRL `2wikimultihopqa` comparison 子集。split seed=`20260818`：

| split                 |    n | query type               |
| --------------------- | ---: | ------------------------ |
| train_experience_pool |  240 | comparison               |
| validation            |   80 | comparison               |
| future_test           |  150 | comparison               |
| negative_control      |   80 | non-comparison multi-hop |

task-id / exact-question overlap=0。旧 R5.3/R5.4/R6 公开 ID 无法无歧义映射到 SkillRL task_id，**未猜映射、未自动排除**。

- **条件：**
  - B0：frozen actor，无 skill
  - B1：frozen actor + SkillRL native textual SkillBank（官方 `claude_style_skills_search.json`，未削弱 top-k / prompt）
  - E0：accepted ESI SkillSpec `esi_cmp_03`（`candidate_03`，comparison-only dispatch）
  - Eempty / Eshuffle：empty / shuffled experience 诱导的因果对照
- **诱导：** 240 条真实 SkillRL rollout → ExperiencePacket → 8 candidates（validation 上选 Δ≥+0.03）；至多 1 轮 repair。Freeze 后再 future_test。
- **成功标准：** held-out test ESI EM > B0 且 Δ≥+0.03；推荐 3 seed、mean Δ≥+0.03、≥2/3 胜。
- **产物：** `/data/ppnm/skillrl_esi/outputs/esi_mvp/`（`GATE0_DECISION.json`、`split.json`、`FREEZE.json`、`ESI_MVP_SUMMARY.md`、`ESI_MVP_METRICS.csv`、`ESI_MVP_DECISION.json`、`PIPELINE_AUDIT.md`、`final/seed{0,1,2}/`）

### Gate 0（SkillRL foundation）

| Gate | 要求                          | 判定                                                         |
| ---- | ----------------------------- | ------------------------------------------------------------ |
| G0.1 | repo/import                   | PASS                                                         |
| G0.2 | Search action protocol 文档化 | PASS                                                         |
| G0.3 | 真实 SearchEnv step           | PASS（smoke reward=1.0）                                     |
| G0.4 | frozen actor inference        | PASS                                                         |
| G0.5 | native no-skill rollout       | PASS（16 comparison：EM=0.1875）                             |
| G0.6 | native textual SkillBank      | PASS（同 16 题：EM=0.375；load/retrieve/inject/finish）      |
| G0.7 | trajectory/memory schema      | PASS（9 success / 23 fail；Azure 未配，不记核心失败）        |
| G0.8 | tiny GRPO/verl smoke          | PASS（GPU5，1 step，ckpt `gate0/grpo_smoke/global_step_1`；算法未改） |

G0.1–G0.7 通过后才写 ESI method。G0.8 是 infra smoke，**不是** ESI 正式训练。

### 结果

**Validation（n=80，freeze 前）**

| condition                      |        EM |    Δ vs B0 |
| ------------------------------ | --------: | ---------: |
| B0 no skill                    |     0.325 |          0 |
| B1 SkillRL textual             |    0.3375 |    +0.0125 |
| ESI `candidate_03`（accepted） | **0.450** | **+0.125** |
| ESI empty                      |     0.400 |     +0.075 |
| ESI shuffled                   |     0.400 |     +0.075 |

8 candidates：`c03`/`c07`=0.45；`c00`/`c01`/`c04`/`c05`=0.40；`c02`/`c06`=0.3375。Accepted=`candidate_03`（sha256 `4ac71936…f0c9`）。第一波 val 全 0.00：4 次 search 耗尽 `max_turns` 未发 `<answer>`；修 runtime 后重验再 freeze。

**Held-out future_test（n=150，freeze 后；seed 0/1/2）**

官方 SearchEnv EM = reward≥1。`temperature=0` + ESI runtime 确定，三套 seed **逐格相同**（已实跑，不是口头跳过）。

| condition               | val EM |    test EM | Δ vs B0 (test) | neg EM | mean searches | mean turns |
| ----------------------- | -----: | ---------: | -------------: | -----: | ------------: | ---------: |
| B0 no skill             |  0.325 |     0.2867 |              0 |  0.050 |          2.13 |       2.85 |
| B1 SkillRL textual      | 0.3375 |     0.1933 |         −0.093 |  0.025 |          2.13 |       2.87 |
| ESI real (`esi_cmp_03`) |  0.450 | **0.3933** |     **+0.107** |  0.050 |          2.85 |       3.85 |
| ESI empty               |  0.400 |  **0.460** |         +0.173 |      — |          2.85 |       3.85 |
| ESI shuffled            |  0.400 |      0.400 |         +0.113 |      — |          2.85 |       3.85 |

| seed |     B0 |     B1 |    ESI |                ESI−B0 |
| ---: | -----: | -----: | -----: | --------------------: |
|    0 | 0.2867 | 0.1933 | 0.3933 |                +0.107 |
|    1 | 0.2867 | 0.1933 | 0.3933 |                +0.107 |
|    2 | 0.2867 | 0.1933 | 0.3933 |                +0.107 |
| mean | 0.2867 | 0.1933 | 0.3933 | **+0.107**（3/3 win） |

Negative control：comparison-only dispatch，ESI ≡ B0（0.050，Δ=0 ≥ −0.03）。Train rollout EM=0.3125（75/240）。

**Runtime vs B0（future_test traces）**

|                              |    B0 |   ESI |
| ---------------------------- | ----: | ----: |
| premature answer（先答后搜） | 0.167 | 0.000 |
| repeated query rate          | 0.280 | 0.000 |
| multi-search rate            | 0.600 | 0.927 |
| mean searches                |  2.13 |  2.85 |

ESI 行为：把 comparison 拆成两个实体 → 每实体一次 SearchEnv query → 抽 year/attribute → 一次 `<answer>`。不是把整题塞进新 CoT。

Pipeline audit 核心项全部 PASS；`tests/esi` 10 passed。

### 结论

- **相对冻结 no-skill actor：CASE_ESI_PASS。** test ΔEM=+0.107 ≥ +0.03，3/3 seed。SkillRL SearchEnv 真实运行；actor 权重未改；skill 经 SearchEnv.step 执行。
- **相对 empty/shuffled：内容因果未建立。** val 上 E0 0.45 > empty/shuffle 0.40；test 上 **empty 0.46 ≥ E0 0.393**。可迁移增益来自 **typed comparison 执行程序 + SearchEnv**，不是这份 240 条 train packet 的独特映射。不能写成「experience 内容被内化」。
- **B1 native textual SkillRL 不是弱化对照。** 官方 SkillBank 正常 retrieve/inject；test 上 B1 **低于** B0（0.193 vs 0.287）。ESI 的正 Δ 不是「任何 skill prompt 都会涨」。
- 与 R2–R6 的关系：再次确认 executable procedure 能抬 frozen 3B；但本轮把 CountOnly / 旧 C2 primitive 拿掉之后，**空 packet 也能诱导出同等或更强 runtime**。这把机制从「经验内容 → 新能力」进一步收缩为「诱导出一个可执行 comparison template」。
- **下一步：** 优先 skill induction / editing（必须在 val 和 test 上都打过 empty/shuffle）。**在那之前不重开 experience acquisition**（frontier / RASA / AVA）。不把 G0.8 的 1-step GRPO 写成 ESI 学习结果。

**不宣称** experience-content causality、不宣称 SkillRL textual skill 已优于 no-skill、不宣称可以重新做 acquisition。

---

## 2026-08-21 R7 Closed-Loop ExPolicy（完成，PIPELINE_PASS，METHOD_AMBIGUOUS）

按 `ExPolicy_R7_Closed_Loop_MVP_H20_Instructions.md`。R6 已停止离线 acquisition heuristic 主线；ESI 确认 executable skill 可作为 assimilation action，但**不**把 skill induction 当 R7 前置门槛。本轮只回答两件事：

1. ExPolicy 是否第一次成为 **stateful、feedback-driven 的真实闭环 controller**（Gate P）；
2. 第一版 Linear Thompson Experience Policy 是否已经优于 fixed / passive / CountOnly（Gate M）。

不问：从零诱导 unique skill、empty/shuffled packet、正式 RL actor、10 seed、Bamboogle、重开 frontier / RASA / AVA。

### Setting

| 项 | 值 |
| --- | --- |
| repo / 分支 | `/data/ppnm/expolicy`，`r7-closed-loop` |
| git head（skill 冻结基线） | `5c9dbf22eab3b43c6c65433037def2fb927d3b2f` |
| 新代码 | `expolicy/closed_loop/`、`scripts/r7/`、`tests/r7/` |
| Actor | `Qwen2.5-3B-Instruct`，权重冻结；eval `temperature=0` |
| Retriever | Wikipedia 2018 E5 HNSW，`:18080` CPU；21,015,324 docs；index `data/wiki18/e5_HNSW64.index` |
| split seed / hash | `20260819` / `08a86552642143c1…e253790b`；task_id 与 exact-question overlap=0 |
| Skill bank | 冻结 R5.3 C1/C2/C3；`r7_mutable=false`。execute `5ef88446…`；C1 `03673d6c…`；C2 `d93c9f6d…`；C3 `6835befa…` |
| Policy | Linear Thompson Sampling（acq 3 cap × dim=11；assim 4 action）；**不是** frontier / RASA / AVA / CountOnly |
| Reward | `Δshadow − 0.5·max(0,−Δretention) − 0.01·tokens/1e6 − 0.005·tools/100`；**不用** final `future_test` |
| Budget | 6 round × batch 8 = **48** unique acquire；full eval @ 0/16/32/48 |
| GPU | 默认 5/6/7 三 seed；GPU 4 并行 B4；0–3 未占用。卡上他人进程未杀 |
| 日历墙钟 | 2026-08-19 10:43 → 2026-08-21 00:30；条件 GPU-time 合计 ~107 h |
| 产物 | `outputs/r7_closed_loop/`（`R7_GATE_DECISION.json`、`R7_MASTER_SUMMARY.md`、各 CSV、`audit/`） |

**Split 规模**（`audit/SPLIT_AUDIT.md` OVERALL PASS）

| split | n | 构成 |
| --- | ---: | --- |
| calibration | 180 | 60 / cap |
| acquisition_pool | 720 | 240 / cap |
| shadow_reward_probe | 180 | 60 / cap |
| retention_probe | 60 | 跨 cap |
| future_test | 450 | HotpotQA / 2Wiki / MuSiQue 各 150 |

**条件**

| ID | 名称 | 作用 |
| --- | --- | --- |
| M0 | `expolicy` | 完整闭环 Linear TS |
| B0 | `passive_noop` | 被动采 + DISCARD |
| B1 | `passive_episodic` | 被动采 + 一律 episodic |
| B2 | `passive_count_skill` | 被动采 + 冻结 CountOnly 开 skill（strongest non-oracle） |
| B3 | `random_joint` | 随机 cap × 随机 assim |
| B4 | `all_skills` | oracle：三 frozen skill 从 budget 0 全开（仅 seed 0） |
| A1 | `no_feedback` | 同 M0 但不写 posterior |
| A2 | `reward_shuffle` | 打乱 reward↔action 对应 |
| A3 | `assimilation_fixed` | 学 acquisition，assim 固定为 CountOnly |

Smoke（2 round × batch 2）：`smoke/SMOKE_PASS.json` PASS。Pipeline audit **24/24**。`tests/r7` 14 passed。

### Pipeline result：`PIPELINE_PASS`

| Gate | 标准 | 本轮数字 | 判定 |
| --- | --- | --- | --- |
| P-A Statefulness | persistent intervention 改变后续 rollout | M0 三 seed 均激活 1 个 frozen skill；后续 dispatch 读到新 mask | **PASS** |
| P-B Acquisition adaptivity | posterior 更新，且 action 分布相对 round 0 可观变化 | 每轮非零 update 后 `acq_delta>0`；**max TV=0.024 < 建议 0.10**。分析脚本因 posterior Δ>0 记 PASS；分布漂移偏弱 | **PASS（弱）** |
| P-C Assimilation adaptivity | posterior 更新，且 ≥2 种 action | 四类均出现：ACTIVATE 3 / DISCARD 5 / EPISODIC 4 / SEMANTIC 6（M0 18 round） | **PASS** |
| P-D Reward closure | shadow Δ → reward → update → 下一轮读新 posterior | reward.source=`shadow_reward_probe`；`uses_future_test=false`；smoke round1 读 round0 后验 | **PASS** |
| P-E Feedback ablation | NoFeedback 保持 prior，且与 M0 trace 不同 | A1 三 seed 全部 round `acq_delta=assim_delta=0`；M0 每轮 Δ≪0 | **PASS** |
| P-F Leakage | final future_test 不进 policy | split 隔离 + resume hash guard + 测试 | **PASS** |
| **总判** | P-A∧…∧P-F | | **PIPELINE_PASS** |

闭环路径已由日志与测试钉死，**不是**离线脚本串。P-B 的 TV 未达 0.10 建议线，记为「后验在更新，但第一版 TS 的 acquisition 分布几乎没被 reward 拉开」。

### Method result：`PIPELINE_PASS_METHOD_AMBIGUOUS`

| Gate M | 冻结标准 | 数字 | 判定 |
| --- | --- | --- | --- |
| 1 | mean(M0) ≥ strongest non-oracle + 0.02 | 0.0474 vs B2/B4 **0.0911**（差 **−0.044**） | **FAIL** |
| 2 | ≥2/3 seed 严格优于 strongest | **0/3**（最好 seed1=0.084 < 0.091） | **FAIL** |
| 3 | M0 AULC 至少 +0.01 | M0 AULC **0.030** vs B2 **0.043** / B4 **0.091** | **FAIL** |
| 4 | M0 > NoFeedback | 0.0474 > 0.0437 | **PASS**（幅度极小） |
| 5 | M0 > RewardShuffle | 0.0474 > 0.0437 | **PASS**（幅度极小） |
| 6 | M0 > AssimilationFixed 或 learned assim 有独立贡献 | 0.0474 < AF **0.0681** | **FAIL** |
| **总判** | | | **PIPELINE_PASS_METHOD_AMBIGUOUS** |

脚本分型：**CASE_B**。闭环成立，controller 不够好。

### 主表（3-seed mean；B4 仅 seed 0）

| condition | final macro | ΔFuture | AULC | skills_active | mean tokens | mean wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **M0 expolicy** | **0.0474** | +0.030 | 0.030 | 1 / seed | 485k | 5.0 h |
| B0 passive_noop | 0.0163 | +0.003 | 0.013 | 0 | 669k | 6.1 h |
| B1 passive_episodic | 0.0326 | +0.020 | 0.026 | 0 | 727k | 6.1 h |
| **B2 CountOnly** | **0.0911** | +0.078 | 0.043 | 3 | 335k | 2.1 h |
| B3 random_joint | 0.0496 | +0.037 | 0.027 | 0–2 | 675k | 5.0 h |
| **B4 all_skills** | **0.0911** | 0（已在预算 0） | 0.091 | 3 | eval-only | ~100 s |
| A1 no_feedback | 0.0437 | +0.028 | 0.029 | 1 | 526k | 4.6 h |
| A2 reward_shuffle | 0.0437 | +0.030 | 0.030 | 1–2 | 510k | 4.5 h |
| A3 assimilation_fixed | 0.0681 | +0.055 | 0.054 | 2–3 | 306k | 2.2 h |

**Per-seed final macro**

| seed | M0 | B0 | B1 | B2 | B3 | A1 | A2 | A3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.033 | 0.013 | 0.038 | **0.091** | **0.091** | 0.011 | 0.022 | 0.022 |
| 1 | **0.084** | 0.020 | 0.027 | **0.091** | 0.027 | **0.089** | 0.076 | **0.091** |
| 2 | 0.024 | 0.016 | 0.033 | **0.091** | 0.031 | 0.031 | 0.033 | **0.091** |

**Learning curve（macro @ budget）**

| budget | M0 s0 | M0 s1 | M0 s2 | B2 s0 | B2 s1 | B2 s2 | B4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.016 | 0.018 | 0.018 | 0.013 | 0.016 | 0.011 | **0.091** |
| 16 | 0.016 | 0.009 | 0.022 | 0.022 | 0.022 | 0.022 | 0.091 |
| 32 | 0.011 | **0.080** | 0.024 | 0.022 | 0.022 | **0.091** | 0.091 |
| 48 | 0.033 | **0.084** | 0.024 | **0.091** | **0.091** | **0.091** | 0.091 |

### 行为分析

**终局能力几乎是「有没有打开 C2」的查表。** future_test n=450，C2 EM≈0.282 时 macro≈0.084–0.091；C3 EM=0；C1 EM≈0.03–0.04。与 R5.3–R6 同一机制。

| run | 激活 skill | C1 EM | C2 EM | C3 EM | macro |
| --- | --- | ---: | ---: | ---: | ---: |
| M0 seed0 | 仅 C3 | 0.044 | 0.027 | 0.000 | 0.033 |
| M0 seed1 | **仅 C2** | 0.026 | **0.282** | 0.000 | **0.084** |
| M0 seed2 | 仅 C1 | 0.037 | 0.009 | 0.000 | 0.024 |
| B2 / B4 | C1+C2+C3 | 0.037 | **0.282** | 0.000 | **0.091** |

M0 seed1 在 round_02 对 `comparison_alignment` 选 `ACTIVATE_EXISTING_SKILL`，shadow Δ≈**+0.105**（reward +0.105），之后 future 从 0.009 跳到 0.080。Seed0 把预算砸在 C3；seed2 三次采到 C2 但都 **DISCARD**，从未激活唯一有用的 skill。

**Acquisition（M0 18 round）：** C1×8 / C2×4 / C3×6。Linear TS **没有**稳定路由到 C2。max TV vs round-0 prior 仅 0.024。

**Assimilation（M0）：** 四类都用过，但 ACTIVATE 只有 3/18，且 2/3 激活了错误 cap。CountOnly 三 seed 都在 48 条内开齐三 skill，终局打到 all-skills 上界 0.091；seed2 更早（budget 32）打开 C2。Random seed0 碰巧在 round_05 激活 C2，也到 0.091。AssimilationFixed 两 seed 第 1 round 就激活 C2 → 0.091，把 mean 拉到 0.068，**高于** 同时学 assim 的 M0。

**Feedback：** NoFeedback 后验冻结，但 seed1 仍在 round_02 按**初始 prior 的随机抽签**激活了 C2（0.089），说明「碰巧开 C2」不需要学习。M0 相对 A1/A2 的 mean 优势只有 +0.004，不能支持「credit assignment 已经在工作」。

**Reward 形态：** 未激活 C2 的 round，shadow Δ 接近 0 或为负，再减去 token/tool 后 reward 常为负。单 batch EM 变化粗，线性 TS 难以从 −0.02 量级噪声里学会「去采 C2 并 ACTIVATE」。

### 结论

- **问题 1：闭环已经打通。** `PIPELINE_PASS`。persistent state、真实 SearchEnv、四类 assim、shadow reward、posterior 写回、NoFeedback 对照、future_test 隔离全部成立。这是 R1–R6 从未交付过的 **Experience Policy runtime**。
- **问题 2：当前 policy 没有优于 baseline。** `METHOD_AMBIGUOUS`。M0 mean 0.047 低于 CountOnly / all-skills 0.091，也低于只学 acquisition、assim 钉死 CountOnly 的 A3（0.068）。增益仍由是否激活 C2 frozen skill 决定；Linear TS 经常 DISCARD C2 或 ACTIVATE C1/C3。
- **按 CASE_B 停止本轮（当时）：** 下一步只优化 `policy representation / credit assignment / exploration`。**不**重做 skill induction、不调 RASA / frontier / CountOnly K、不改 final test、不改 skill 内容、不扩 10 seed。**R8 随后证明：在当前 frozen-mask SearchEnv 上换更强 controller 没有可辨识决策空间，这条路关闭。**
- 与 ESI 的关系：skill 在本轮只是可用的 assimilation action。R7 **没有**把「experience 是否诱导 unique skill」重新打开。

**不宣称** Linear TS 已是可部署 Experience Policy、不宣称 learned routing 已超过 CountOnly、不宣称 P-B 的 action 分布已被 reward 显著拉开。宣称的只有：**闭环系统第一次真实跑通；第一版 controller 方法信号未过。**

---

## 2026-08-21 R8 Structural Redesign（完成：R7 环境退化 + ALFWorld foundation FAIL；未训 controller）

按 `ExPolicy_R8_Next_Run_Instructions.md`。R7 已证明闭环 runtime 成立，但 Linear TS 未超过 CountOnly / all-skills。本轮**禁止**把失败解释成「换一个更强 bandit」；必须先回答：当前 substrate 上 assimilation 是否还有可辨识的决策空间。严格按 gate 顺序：Novelty → R8-A 结构审计 → 仅当退化时才做 R8-B 异构 backend → 仅当 Gate H 过才训 Stage-1。不问：在 R7 SearchEnv 上重训 controller、人工制造 skill 互斥、把 MemRL/SkillOpt 本身写成 novelty。

### Setting

| 项 | 值 |
| --- | --- |
| 协议 | `ExPolicy_R8_Next_Run_Instructions.md`；λ 冻结与 R7 相同：`retention_weight=0.5`，`token_cost_per_million=0.01`，`tool_cost_per_100=0.005` |
| 机器 | 本机 8×H20-3e；只用 GPU **4/5/6/7**（0–3 被 casbot/openpi 占满；4–7 上其他用户进程未杀） |
| 模型 | 冻结 `Qwen2.5-3B-Instruct`；R8-B 由 vLLM 0.25.1 提供 OpenAI 兼容接口（`127.0.0.1:18090`，`max-model-len=4096`，`gpu_memory_utilization=0.18`，`max-num-seqs=16`） |
| R8-A 环境 | 复用 R7 公开 Wikipedia+E5 SearchEnv + 冻结 C1/C2/C3 skill bank（`outputs/r7_closed_loop`）；retriever `127.0.0.1:18080`（未动 SkillRL 8030） |
| R8-B 环境 | ALFWorld TextWorld（`/data/ppnm/data/alfworld`，六族 AlfredTWEnv） |
| 第三方 backend | MemRL `c1b322ca`（MIT）；SkillOpt `3c8873f0`（MIT）。官方 checkout，**无源码 patch**（`THIRD_PARTY_PATCH_AUDIT.md`） |
| Split | `scripts/r8/build_alfworld_splits.py --seed 20260821`；从 **train 可玩 `game.tw-pddl`** 切（官方 valid_unseen 配额不够，fail-closed 不用）。sha256 `8413f7fc91dddf40ee294e4dbff6a6b528ac1e3b6dcf5f1c4a94ab781d1ccadb` |
| SkillOpt manifest | train=`acquisition_pool`，val=`assim_validation`，test=`shadow_reward_probe`（**不是** `future_test`） |
| smoke split | `skillopt_smoke` 12/6/6 |
| Python | R8-A：`expolicy/.venv`（vLLM）；R8-B smoke：`miniconda3/envs/cto`（ALFWorld 0.4.2 + SkillOpt editable + MemOS） |
| 监督 | `supervise_r8_preflight.py` / `supervise_r8_backend.py`；stall 只杀本轮 PID |
| 产物 | `outputs/r8_preflight/`，`outputs/r8_backend_validation/`，`R8_NOVELTY_AUDIT.md` |

冻结 bucket（隔离审计 PASS：gamefile / turk instruction 跨 bucket overlap=0）：

| bucket | n | 每族 |
| --- | ---: | ---: |
| calibration | 120 | 20 |
| acquisition_pool | 360 | 60 |
| assim_validation | 120 | 20 |
| shadow_reward_probe | 120 | 20 |
| retention_probe | 20 | 较易族（simple 18 + look 2） |
| future_test | 180 | 30 |

### Phase 0 Novelty

12 篇必读全文 + 2026 近邻。Hard-stop 仅当一篇同时满足 ExPolicy 四条交集（真实环境 acquisition × 异构持久 assimilator × held-out future capability 作经验价值 × 联合 stateful loop）。

判定：`PROCEED`（`R8_NOVELTY_AUDIT.md`）。**未**触发 `NOVELTY_COLLISION_HARD_STOP`。最危险近邻是 MAGE（task bandit + skill bandit 同环），但 MAGE 选的是检索/任务图节点而非真实 env batch，且不是「同一条 experience 上的异构持久 operator」。MemRL / SkillOpt 只抢占单轴（memory Q / skill edit），**不是**本轮 novelty。

### Phase 1 / R8-A 结果（83/83 DONE，2026-08-21 17:34）

`scripts/r8/audit_r7_structural.py`：counterfactual 72 + sequential oracle 3 + mask ceiling 8。**不训练。**

**S1 sequential hindsight oracle**（沿 M0 真实 acquisition 流逐步提交最优 assim）

| 量 | 值 |
| --- | ---: |
| oracle_joint final | **0.0644** |
| strongest fixed / CountOnly / all_skills | **0.0911** |
| M0 final | 0.0474 |
| mean per-step oracle U | 0.0244 |
| mean per-step best-fixed U（always ACTIVATE） | 0.0260 |
| per-step utility gap（oracle − best fixed） | **−0.0016** |
| S1 gate（oracle_joint ≥ fixed + 0.03，即 ≥0.121） | **False**（substrate 上界就是 0.0911，门控在数学上不可能过） |

| seed | oracle final | 相对 M0 是否偏离 |
| ---: | ---: | --- |
| 0 | 0.0178 | True（激活 C3，未开 C2） |
| 1 | 0.0889 | True（开 C2） |
| 2 | 0.0867 | True（开 C2） |

**S2 mask ceiling**（冻结 C1/C2/C3 开关的 8 个 mask，future_test 实评）

| mask | macro |
| --- | ---: |
| none | 0.0156 |
| C1 | 0.0222 |
| C2 | 0.0822 |
| C3 | 0.0178 |
| C1+C2 | **0.0911** |
| C1+C3 | 0.0222 |
| C2+C3 | 0.0800 |
| C1+C2+C3 | **0.0911** |

max_mask − strongest_fixed = **0.0000**（阈值 0.02）。CountOnly = all_skills = C1+C2。增益几乎全是打开 C2。

**18×4 counterfactual assimilation matrix**（缺格 0）

| action | unique-best 次数 | 占比 |
| --- | ---: | ---: |
| DISCARD | 2 | 11.1% |
| EPISODIC_MEMORY | 3 | 16.7% |
| SEMANTIC_MEMORY | 1 | 5.6% |
| ACTIVATE_EXISTING_SKILL | 6 | **33.3%** |

unique-best ≥20% 的动作数 = **1**（需要 ≥2）。C2 上 ACTIVATE 相对次优的 mean margin = 0.108。

**S3 feature separability**（offline，无 online update；n=18，LOSO）

| 指标 | 值 |
| --- | ---: |
| majority（ACTIVATE） | 33.3% |
| logreg LOSO | 11.1% |
| MLP LOSO | **27.8%**（比 majority **低 5.6 pp**） |
| ridge mean Pearson | 0.217 |
| MLP-reg mean Pearson | 0.039 |

**R8-A 五门全 False** → **`R7_ENV_STRUCTURALLY_DEGENERATE`**。禁止在当前 frozen-mask SearchEnv 上训练新 controller。

### Phase 2 / R8-B 结果（backend validation 未完成正式 3-seed）

先独立复现两个官方 assimilator，不写 ExPolicy。

**Smoke（seed 0）**

| job | 状态 | success | 细节 |
| --- | --- | ---: | --- |
| `none_s0_smoke` | DONE | **0.0** | n=22，wins=0 |
| `skillopt_s0_smoke` | DONE | **0.0** | train 3 step 全 `skip_no_patches`，`best_score=0`；eval n=10，wins=0 |
| `memrl_s0_smoke` | FAIL | — | `MOS_MemoryService_hang`：卡在 `OpenAI LLM instance initialized`，约每 2 分钟 crash/relaunch，从未进入 ALFWorld |
| `foundation_no_learning` | 未完成 | — | 目标 120 局 calibration；写出 16 个 `conversation.json` 后于 13:27 停在 AlfredTWEnv 重初始化（8810/8810 进度条循环） |

3B 官方 SkillOpt actor 几乎不产出 `<action>` 标签，环境反复 `look`。这不是「训练不够」，no-learning 就已经是 0。

**Foundation gate（协议 10%–70%）**

已完成的 no-learning 数字是 **0%**，低于 10% 地板。120 局正式 calibration 未跑完，但不构成剩余过 gate 的路径（同一冻结 3B actor）。落盘：`outputs/r8_backend_validation/MODEL_FOUNDATION_GATE_FAIL.md`。

**因此未启动：** 全量 3-seed `no_update / memrl / skillopt`、`analyze_backend_validation.py`、Gate H、R8 Stage-1 joint policy。

### 结论

- **Novelty 仍在四条交集上。** 单轴（memory policy / skill edit / dual bandit slogan）已被近邻抢占，但完整 conjunction 仍空。这只允许继续做 ExPolicy，**不**允许把 MemRL 或 SkillOpt 写成贡献。
- **R7 SearchEnv 不再适合当 method benchmark。** 终局 utility 是 C2 激活 mask 的查表；hindsight oracle 的终局（0.064）甚至低于 CountOnly（0.091）；逐步最优 assim 的平均 U 也不如 always-ACTIVATE。换更强 bandit / MLP 不能创造不存在的决策空间（S3 MLP 比 majority 还差）。
- **ALFWorld + 冻结 3B 也还不是可用的异构 testbed。** 官方 SkillOpt no-learning smoke = 0，低于 foundation 地板；MemRL 官方 MOS 初始化在本机适配下未跑通。Gate H 没有可解释的 assimilation gain 可测。
- **按协议停止本轮。** 不训 ExPolicy，不静默换 prompt / 换大模型，不把项目改写成 SkillOpt 调度器，不在 R7 上继续堆 controller。下一步如果还做，应换一个 **no-learning 落在 10%–70%、且天然存在异构持久 intervention** 的公开环境（协议点名 WebShop / ScienceWorld / LifelongAgentBench），而不是给 ALFWorld 人工加惩罚或给 R7 加 skill 互斥。

**不宣称** R7 换 controller 就能赢、不宣称 ALFWorld 3B 上 MemRL/SkillOpt 已被复现为有效 assimilator、不宣称 Gate H / Stage-1 已测。宣称的只有：**R7 环境结构退化已用全量 counterfactual/oracle/ceiling 证明；当前 3B ALFWorld foundation 未过；本轮未训练 Experience Policy。**

---

## Go / No-Go（对照路线图 §15）

| Gate                          | 标准                                                         | 本轮判定                                                     |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| A Utility heterogeneity       | top-20% 贡献显著高于平均                                     | **E1 未过；R4 后部分成立**：正 U 集中在 skill 载体（11/12 batch），不是经验之间的效用异质性 |
| B Active > Passive            | 同 budget 稳定 future gain                                   | **R5 PASS；R5.1 10-seed 加固**（Helion mean gap +0.292，10/10）。**R5.2 Helion collision PASS**（frontier−Ace +0.306，10/10）。**R5.3 公开环境 FAIL**（frontier Δ=−0.007，输给 Ace/passive，0/3）。**R5.4 RASA FAIL**（full 0.111 < Ace/passive 0.120，0/3）。**R6 Exact AVA FAIL**（0.120 vs 0.119，+0.001，1/3 胜）。机制在 Helion 上是 gated activation；离开 Helion 后冻结 frontier / RASA / exact AVA 均未超过 Ace/passive。 |
| C Carrier choice matters      | best ≠ global，且 adaptive 降 regret                         | **弱通过**：skill vs memory/weight 差异大；always-skill ≈ oracle（差 0.003），learned router 无空间 |
| D Joint > 单独                | Full > active-only 且 > adaptive-only                        | **未测**                                                     |
| **R4 assimilation repair**    | 经验可吸收 + carrier 改变 gain                               | **字母 GO / 机制保留**（正 U 几乎全来自 executable skill）   |
| **R5 Gate B**                 | 有效 assimilation 后 Active > Passive                        | **PASS**（3-seed；不证明 learned routing）                   |
| **R5.1 Case B**               | 稳健性 + 内容因果 + 可部署 dispatch                          | **命中 Experience-Gated Skill Activation**。CountOnly ≥ RealFit；Helion text-router = oracle。不进入 R6。 |
| **R5.2 Case A**               | frontier 显著优于 strongest acquisition baseline             | **Helion 命中**。frontier−Ace-Skill = +0.306（10/10）；Ace-Skill=passive。停止堆 Helion，进入第二 benchmark。 |
| **R5.3 Case C**               | Helion 优势能否离开合成环境                                  | **未通过。** 公开 Search-Agent 上 frontier < passive / Ace（3/3 seed）。Skill backend 成立，冻结 frontier allocation 塌缩到 C1。3-seed stop，不扩 10 seed。 |
| **R5.4 Case E**               | RASA（repairable-deficit acquisition + SRT）能否在新公开 holdout 上赢 Ace/passive | **未通过。** `rasa_full` 0.111 < Ace/passive 0.120（0/3）。Gate A–E 失败，仅 F 过。Remediability 无独立贡献（full=qonly）。3-seed stop，不扩 Stage-2。 |
| **R6 CASE_C**                 | Exact AVA（post-assimilation shadow utility）能否在冻结 CountOnly 上赢 Ace/passive | **未通过。** `exact_ava` 0.120 vs best baseline 0.119（margin +0.001，Gate A 要 +0.02）；seed 胜 1/3。Gate A/B/C 失败，D/E/F 过。终局由激活 mask 查表决定。3-seed stop，**不训 surrogate**。 |
| **SkillRL-ESI CASE_ESI_PASS** | 在 SkillRL Search 上，真实 experience → executable SkillSpec 能否让冻结 actor 在未见 split 上 +0.03 | **相对 B0 通过**（test Δ=+0.107，3/3）。**相对 empty/shuffled 未通过**（test empty 0.46 ≥ ESI 0.393）。Gate 0.1–0.8 PASS。不重开 acquisition。 |
| **R7 CASE_B**                 | 闭环 Experience Policy 是否第一次真实运行；第一版 Linear TS 是否优于 CountOnly / passive | **Pipeline PASS**（P-A–P-F）。**Method AMBIGUOUS**：M0 mean **0.047** < CountOnly / all-skills **0.091**（0/3 seed 过 +0.02）。下一步只改 policy/credit/exploration，不回头改 skill / split / CountOnly。 |
| **R8-A `R7_ENV_REUSABLE`**    | oracle_joint ≥ fixed+0.03；≥2 个 assim unique-best≥20%；step U gap≥0.02；MLP LOSO ≥ majority+15pp | **未通过。** 判定 **`R7_ENV_STRUCTURALLY_DEGENERATE`**。oracle_joint **0.064 < 0.091**；unique-best≥20% 动作只有 ACTIVATE；step gap **−0.0016**；MLP 27.8% < majority 33.3%；S2 ceiling gap=0。禁止在冻结 C1/C2/C3 SearchEnv 上换 controller。 |
| **R8-B foundation 10%–70%**   | 冻结 3B + 官方 SkillOpt actor，ALFWorld no-learning success 落在可解释带 | **FAIL。** smoke none/SkillOpt **0%**；120 局 calibration 16/120 后挂死。落盘 `MODEL_FOUNDATION_GATE_FAIL.md`。未换 prompt / 未换大模型。 |
| **R8 Gate H**                 | MemRL 与 SkillOpt 相对 no-update 至少各族 +0.05，且 oracle-per-family > best fixed +0.03 | **未测。** foundation 已 FAIL；MemRL smoke 未进入 ALFWorld。按协议 **不训 ExPolicy**。 |

### 本轮停止点（按 `ExPolicy_R8_Next_Run_Instructions.md`：R7 环境退化 + foundation FAIL）

R6 停止的是**离线 acquisition heuristic**。R7 把主线做成闭环 runtime：Gate P 已过，Gate M 未过。R8 接着问「这个 runtime 所在的 substrate 是否还支持学习 assimilation」。答案是否定的：

- **停止本轮实验。** 不在 R7 SearchEnv 上换 Linear TS / MLP / 更强 bandit；不扩 10 seed；不调 CountOnly K；不改 skill 内容；不重抽 ALFWorld split；不静默换 3B prompt 或更大模型来过 foundation；不把 MemRL/SkillOpt 写成 novelty。
- **不进入** Gate H 全量 3-seed、R8 Stage-1 joint policy、Stage-2。
- **下一步如果还做 ExPolicy：** 换一个 no-learning 落在 10%–70%、且天然存在异构持久 intervention 的公开环境（协议候选：WebShop / ScienceWorld / LifelongAgentBench），先过 foundation 与 Gate H，再谈 controller。不要给 ALFWorld 人工加惩罚，也不要把项目收缩成 SkillOpt 调度器。

对 R1–R4 核心问题的回答仍然成立：

> 之前 Gate A/B/C 失败，主要是因为 **experience assimilation 实现没有把经验变成可迁移能力**，而不是已经证伪了 Experience Policy 假设。

R5 追加的回答：

> 在可用的 experience-conditioned executable assimilation 下，**主动选择学什么**（尤其是 frontier）能比被动采样带来更高的未来能力收益。这修复的是 acquisition 测试，不是完整 ExPolicy。

R5.1 把机制再收缩一层：

> 增益来自 **experience-gated skill activation**（采够该 cap 的条数 → 解锁对应 frozen primitive），不是 `fit()` 从轨迹内容里学到了关键 config。Helion 上 text-router 保住了 oracle gain，但这是模板可分世界，不能外推开放域。

R5.2 再追加一层：

> 该 acquisition 机制 **不是** Ace-Skill 类 generic prioritized sampling 的换壳。在相同 CountOnly unique-task 后端下，frontier 10/10 优于 Ace-Skill（gap +0.306）；Ace-Skill 与 passive 打平。下一步应做外部 benchmark，而不是继续优化 Helion frontier score。

R5.3 把外部泛化问句关闭：

> Helion 上的 frontier 优势 **没有**迁到公开 HotpotQA / 2Wiki / MuSiQue。Executable skill backend 在公开环境成立（C2 ΔEM +0.417），但冻结 Helion frontier score 把 48 次预算全部压进 C1/HotpotQA，错过唯一有用的 C2。结果支持 Helion-specific heuristic，不支持论文主结果。Case C：停止扩实验、停止调权重。

R5.4 把 redesign 问句关闭：

> 把 acquisition target 从 generic difficulty/frontier 改成 **repairable capability deficit（`q×p`）**，并配 Sequential Readiness Test，**没有**在新冻结公开 holdout 上超过 Ace / passive。SRT 能正确 ACTIVE C2、PARK C1，但 C3 未 PARK，剩余预算被几乎不可修的 long-chain 吃掉；`rasa_full = rasa_k8 = rasa_qonly`。Case E：RASA 主线证伪，停止。

R6 把 principle 问句关闭：

> 用 **当前 assimilator 的 post-assimilation shadow future gain** 当 exact oracle 选下一批经验，**没有**在冻结 CountOnly 公开 holdout 上超过 Ace / passive。终局 0.120 只是打开三个 frozen skill 的 mask 上界；Gate A/B/C 失败。CASE_C：不训 surrogate，停止 acquisition 主线。

SkillRL-ESI 把 assimilation 问句收缩到成熟 Search backbone：

> 在 SkillRL SearchEnv + 冻结 Qwen2.5-3B 上，把 train-side experience 编成 executable SkillSpec **能**让 held-out comparison 2Wiki 相对 no-skill 提升 +0.107（3/3）。但 empty-experience 对照在 test 上 **不低于** real ESI，所以增益是 comparison 执行程序，不是这份 packet 的内容内化。CASE_ESI_PASS 只对 B0 成立；下一步是 induction/editing，不是 acquisition。

R7 把 Experience Policy 本身做成 runtime：

> **闭环已经打通，第一版 policy 还没有赢。** acquisition → 真实 SearchEnv → 动态 assim → persistent state → shadow reward → posterior update → 下一轮，六项 Gate P 全过。Linear Thompson 的 final macro 0.047 低于 CountOnly / all-skills 0.091；几乎全部 future 增益仍来自是否激活 C2 frozen skill。CASE_B：只改 controller，不回头改 skill backend。

R8 把「只改 controller」这条路关掉：

> **当前 R7 SearchEnv 的决策空间结构退化，ALFWorld 3B 异构 backend 也未过 foundation。** hindsight oracle 终局 0.064 低于 CountOnly 0.091；mask 天花板 gap=0；MLP 分不开最优 assim。官方 SkillOpt no-learning smoke=0。因此失败不是「Linear TS 不够强」，而是 **method benchmark 不成立**。本轮未训练 ExPolicy。

证据：

1. 修好 executable skill 后，near-transfer 出现稳定、大幅度正 gain（R2/R4）。
2. 同期 episodic / semantic / LoRA 仍然失败（R1/R3/R4）——旧 E2 用的正是这条无效路径。
3. R5 把 assimilation 换成 skill 后，frontier 3-seed ΔFuture=+0.519，相对 passive +0.296；Ablation A 显示增益全部来自 skill 激活。
4. R5.1：10-seed gap=+0.292（CI 不跨 0，10/10）；各 K 下 frontier>passive；**RealFit 0.519 ≲ CountOnly 0.546**；Helion text-router EM = oracle EM。
5. R5.2：同 CountOnly unique-task 后端，**frontier−Ace-Skill = +0.306（10/10）**；Ace-Skill ΔFuture 与 passive 逐 seed 相同；frontier 更快激活第二个 target skill（17.8 vs >48）。重复采样不是 Ace-Skill 失败原因（mean dup=1.9）。
6. R5.3：公开 Wikipedia+E5 上 skill 3/3 accept，但 **frontier ΔFuture=−0.0067，Ace/passive ≈ +0.08，0/3 seed 赢**；48/48 全是 HotpotQA C1。Stage-1 未过，未扩 10 seed。
7. R5.4：新 holdout 上 8 条件 × 3 seed。**rasa_full mean macro=0.111 < Ace/passive 0.120（0/3，Gate A–E 失败）**。C2 ACTIVE / C1 PARKED / C3 UNRESOLVED 并吞 ~35/48 budget；full=qonly=rasa_k8。Stage-2 未跑。
8. R6：同一 R5.4 future_test、冻结 CountOnly。**exact_ava mean=0.120 vs Ace/passive 0.119（+0.001，1/3 胜，Gate A/B/C 失败）**。future macro 由 8 个激活 mask 查表（全开=0.120，C2-only=0.111）。Novelty PROCEED、pipeline 27/27、pytest 14/14。Surrogate 未训。
9. SkillRL-ESI（2026-08-19）：SkillRL `8e66726` + 冻结 3B + 2Wiki comparison split seed 20260818。**ESI test EM=0.3933 vs B0 0.2867（Δ=+0.107，3/3）**；B1 textual=0.1933；empty=0.46；shuffled=0.40。Gate 0.1–0.8 PASS；pipeline audit PASS；`tests/esi` 10/10。未用旧 CountOnly / C2 primitive。
10. 当前能吸收经验并抬 frozen 3B 的几乎只有 **executable procedure**。R1–R6 里主要是 count-gated 解锁；ESI 拿掉 CountOnly 之后，**空 packet 仍能诱导出同等或更强 runtime**。learned routing / weight internalization / Helion 外 frontier 泛化 / RASA / Exact AVA / **experience-content → unique SkillSpec** 均未被支持。
11. R7（2026-08-19→21）：公开 Wikipedia+E5 + 冻结 C1/C2/C3 skill bank。**Gate P PIPELINE_PASS**（24/24 audit，NoFeedback 后验冻结，四类 assim 均出现）。**Gate M AMBIGUOUS**：M0 mean **0.0474** vs CountOnly/all-skills **0.0911**（0/3 过 +0.02）；seed1 仅因激活 C2 到 0.084。Linear TS max TV=0.024。产物 `outputs/r7_closed_loop/`。
12. R8（2026-08-21）：Novelty `PROCEED`。R8-A 83/83 → **`R7_ENV_STRUCTURALLY_DEGENERATE`**（oracle_joint 0.0644，CountOnly=all_skills=0.0911，S2 gap=0，MLP LOSO −5.6 pp）。R8-B split sha256 `8413f7fc…` 隔离 PASS；MemRL `c1b322ca` / SkillOpt `3c8873f0` 无源码 patch。3B smoke success=**0**；MemRL MOS hang；foundation 16/120 hung。产物 `outputs/r8_preflight/`、`outputs/r8_backend_validation/MODEL_FOUNDATION_GATE_FAIL.md`。未训 Stage-1。

**不宣称 Active Acquisition 在任意 assimilation 下成立，也不宣称 Adaptive Assimilation、内容依赖的 skill fit、开放域 skill dispatch、RASA、Exact AVA、ESI experience-content causality、R7 Linear TS 已优于 CountOnly、或 R8 已在 ALFWorld 上复现有效异构 assimilator。R7 只证明闭环 runtime 成立；R8 证明该 runtime 的当前 SearchEnv substrate 结构退化，且 3B ALFWorld foundation 未过。**

