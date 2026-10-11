# ExPolicy 进展汇报

> 整理日期：2026-10-09  
> 正式实验记录：`expolicy/expolicy-result-record.md`（总览更新于 **2026-08-21**，写到 R8）  
> 本文件补上记录里还没写回去的 **R9**（2026-08-21 启动，**2026-08-22 09:37 整轮停掉**）  
> 主 repo：`/data/ppnm/expolicy`  
> 状态以产物和记录为准。没跑完的实验不写成已验证。

---

## 0. 一句话现状

ExPolicy 停在 **R9 的同化对照矩阵做到一半、还没有出判定**。

经验策略本身——决定下一步采什么经验、采到之后怎么用——**还没有训练出来**。R7 把闭环 runtime 跑通了，但第一版控制器没有赢过固定基线；R8 证明旧搜索环境的决策空间是退化的，ALFWorld + 冻结 3B 又过不了 foundation。R9 改到 LifelongAgentBench 上重找 substrate：模型已经选定（Qwen3-14B），foundation 落在可学习区间，算子健全性通过，**36 个 batch 的四分支矩阵只完成了 21 个完整 batch，Gate H / Gate J 都没做。**

---

## 1. 项目要解决的问题

ExPolicy 的核心不是“从经验里生成 skill”，而是学习一条作用在经验本身上的动态策略：在有限交互预算下，决定下一步应该获得什么经验，以及拿到经验之后应该如何利用它，从而最大化未来能力增长。

两个耦合问题：

1. **Experience Acquisition Policy — 学什么 / 去哪里获得经验**  
   给定当前能力状态，下一步该经历什么。目标不是当前 rollout reward，也不是简单优先失败、不确定或困难样本，而是优先获得对未来能力提升最有价值的经验。

2. **Experience Assimilation Policy — 得到经验后怎么处理**  
   一条经验不必默认写入 memory，也不必转成新 skill。策略根据当前状态和经验特征，决定执行哪一种持久能力干预，例如：丢弃、episodic memory、semantic memory、检索或激活已有 skill、编辑 / 组合 / 提升 skill、更新 harness、更新权重。

\[
\text{Experience Policy}
=
\text{Acquisition Policy}
+
\text{Assimilation Policy}
\]

主线 pipeline：

```text
Current Agent State
        ↓
Capability Diagnosis
        ↓
ExPolicy: choose what experience to acquire
        ↓
Real Environment Interaction
        ↓
Observed Experience
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

研究边界（记录里写死的）：

- Skill 只是 assimilation action 之一，不是 ExPolicy 本身。
- 不要求先证明每个 executable skill 都由当前经验内容从零诱导出来。
- `experience → unique SkillSpec` 的内容因果是子问题，不是主线成立的前置条件。
- 主线要证明的是：在相同交互 / 计算预算下，一条可学习、动态、跨环境的 Experience Policy，能比被动采集、固定同化或简单启发式，得到更高的 future capability gain。

---

## 2. 总览

| 实验 | 状态 | 结论 |
| --- | --- | --- |
| E0 Capability-Gap Audit | 完成 | Helion overall EM=57.3%，error=42.7%，判定 `IDEAL_FRONTIER`。S1 是硬缺口（pass@4=0），S4 不稳定（pass@4=0.75）。 |
| E1 Realized Utility | 完成（机制弱阳性） | 单条 episodic 干预在 8-task 远迁移 probe 上得不到正 U。Gate A 未干净通过。 |
| E2 Active Acquisition Pilot | 完成（当时 Gate B 未过） | 固定 episodic+semantic 下，5 种 acquisition 的 future_gain 全 ≤0。后被 R5 修订为“无效 assimilation 后面的评测”。 |
| E3 Carrier Utility Matrix | 完成（Gate C 弱 / 未过） | 12×3 里 3/12 有 U>0。mean U 仍全负。 |
| E4 Learned Assimilation | 诊断完成，不训练 | heuristic regret=0.083，等于 random，差于 always-episodic（0.036）。 |
| R1 Near-Transfer Utility | 完成（Go 未过） | 仅 S4 new-episodic 平均 near gain>0（+0.104）。S1/S2 为负。 |
| R2 Executable Skill | 完成（三 skill 均 accept） | executable near gain：S1 +0.562 / S2 +0.438 / S4 +1.000。旧 prompt skill 无正 near。 |
| R3 Weight Batch | 完成（LoRA 弱 / 未过） | LoRA U：S1 −0.062 / S2 −0.062 / S4 +0.094。正 U 仍主要来自 executable skill。 |
| R4 Batch Carrier Matrix | 完成（字母 GO，机制有保留） | 11/12 batch 有 U>0；skill mean U=+0.237，几乎全局最优。oracle 只比 always-skill 高 0.003。 |
| R5 Active Acquisition × Skill | 完成（Gate B PASS） | frontier mean ΔFuture=+0.519（3 seed），比 passive 高 +0.296。不证明 learned routing。 |
| R5.1 Robustness | 完成（Case B） | 10-seed frontier−passive=+0.292，10/10。机制是 experience-gated skill activation，不是内容内化。 |
| R5.2 Strong-Baseline Collision | 完成（Case A） | frontier−Ace-Skill=+0.306，10/10。Ace-Skill=passive。不是 generic prioritized sampling 换壳。 |
| R5.3 External Generalization | 完成（Case C，3-seed stop） | 公开 HotpotQA / 2Wiki / MuSiQue 上冻结 frontier 不泛化。ΔFuture=−0.0067，0/3 seed 赢。 |
| R5.4 RASA Redesign | 完成（Case E，3-seed stop） | `rasa_full` macro=0.111 < Ace/passive 0.120，0/3。RASA 主线证伪。 |
| R6 Exact AVA | 完成（CASE_C，3-seed stop） | exact shadow utility 不能超过 Ace/passive：0.120 vs 0.119（+0.001，1/3）。停止 acquisition 主线，不训 surrogate。 |
| SkillRL-ESI MVP | 完成（相对 B0 通过，内容因果未过） | test ESI 0.3933 vs B0 0.2867（Δ=+0.107，3/3）。empty 0.46 ≥ ESI。增益来自 comparison 执行程序。 |
| R7 Closed-Loop | 完成（PIPELINE_PASS，METHOD_AMBIGUOUS） | 闭环第一次跑通。Linear TS mean 0.047 < CountOnly / all-skills 0.091。 |
| R8 Structural Redesign | 完成（环境退化 + foundation FAIL；未训 controller） | R7 SearchEnv 判定 `R7_ENV_STRUCTURALLY_DEGENERATE`。ALFWorld 冻结 3B smoke success=0。Gate H / Stage-1 未跑。 |
| **R9 Substrate & Joint Identifiability** | **未完成（2026-08-22 停机）** | Qwen3-14B 已选定；foundation 在带内；算子健全性 PASS。同化矩阵 21/36 个完整 batch。**Gate H、Gate J、learned ExPolicy 都没做。** 正式记录尚未回写。 |

E5–E8 以及 R6 的 learned carrier **按规则没有做**。

---

## 3. 共用设置

| 项 | 值 |
| --- | --- |
| 机器 | 本机 8×H20-3e。实验默认只用 GPU 4/5/6/7，不抢占 0–3 上已有进程。 |
| 早期环境 | Helion，本地 BM25，`data/world/helion.json`，world seed=13。R1–R5.2。 |
| 公开搜索环境 | Wikipedia 2018 + E5-base-v2，21,015,324 docs。R5.3–R7 用端口 18080。SkillRL-ESI 用 SkillRL retrieval server 端口 8030。 |
| R8-B 环境 | ALFWorld TextWorld，`/data/ppnm/data/alfworld`。 |
| R9 环境 | LifelongAgentBench：DB / OS / KG。 |
| 模型 | R1–R8 冻结 `Qwen2.5-3B-Instruct`。R9 模型阶梯筛选后冻结 `Qwen3-14B`。全程无正式 RL，eval `temperature=0`。 |
| 禁止项 | mock LLM、伪造检索、把失败改写成成功。未过 gate 不扩 seed、不调系数、不缩 claim、不把项目改写成某个单轴调度器。 |

---

## 4. 已完成阶段：机制是怎么被逐步收紧的

### 4.1 E0–E4：先确认 Helion 上有学习空间，再发现普通记忆吸收不了经验

E0（2026-08-13）在冻结 3B、greedy T=0 上审计 Helion：

| Stage | n | EM | 解读 |
| --- | ---: | ---: | --- |
| S0 single-hop | 100 | 0.930 | 已掌握 |
| S1 multi-hop / decompose | 100 | 0.160 | 主缺口 |
| S2 temporal / disambig | 50 | 0.400 | 以真实缺口为主 |
| S3 conflict / reliability | 48 | 1.000 | 被 system prompt 解掉 |
| S4 recovery / synthesis | 49 | 0.449 | 不稳定能力 |
| Overall | 347 | 0.573 | error=42.7%，落在 25–45% 目标带 |

pass@4：S1=0.00（真实能力缺口），S4=0.75（不稳定，更适合 active acquisition）。判定 `IDEAL_FRONTIER`，继续 E1。

随后在固定 episodic / semantic / 当时的 skill prompt 上：

- E1：单条 episodic 的 positive U rate=0.00，mean future_gain=−0.031。ordinary failure 是唯一 mean future_gain>0 的桶。
- E2：budget=24 下五种 acquisition 的 large-probe future_gain 全 ≤0。当时 Gate B 未过。
- E3b：mean U 全负（episodic −0.073 仍是 global best fixed）。
- E4：hindsight oracle mean U=−0.036；fixed episodic regret=0.036；heuristic router regret=0.083，与 random 相同。**不训练 learned router。**

这一阶段的正确读法：不是已经证伪 Experience Policy，而是 assimilation 还没把经验变成可迁移能力。

### 4.2 R1–R4：唯一稳定正效用的载体是可执行 procedure

| 轮次 | 结果 |
| --- | --- |
| R1 | 近迁移上，只有 S4 new-episodic 平均增益为正（+0.104，2/3 batch）。S1/S2 为负。Go 未过。 |
| R2 | 三个 executable skill 全部 accept。near gain：S1 +0.562、S2 +0.438、S4 +1.000。旧 prompt skill 没有正 near。retention 大跌，procedure 绑死了 capability。 |
| R3 | verified-success batch LoRA：S1 −0.062、S2 −0.062、S4 +0.094。权重更新不是稳定吸收机制。 |
| R4 | 12 batch × 4 carrier。11/12 batch 有 U>0；skill mean U=+0.237，11/12 几乎全局最优。oracle 只比 always-skill 高 0.003。episodic / weight 仍多为负。字母 GO，机制有保留。 |

R4 允许的下一步只是比较 acquisition × skill assimilation。learned router 没有监督多样性。

### 4.3 R5–R5.2：在 Helion 上，“采什么”会改变未来能力，但机制是解锁 skill

R5 把 assimilation 换成 experience-conditioned executable skill（默认全部 disabled，采够才激活）。Gate B **PASS**：

- frontier mean ΔFuture = **+0.519**（3 seed）
- 相对 passive **+0.296**
- uncertainty / failure_replay 只略高于 passive（幅度小于 1 题）
- reward_seeking Δ=0
- 关掉 skill 回到 base；预装全部 R2 skill 后五种 acquisition 无差别

这证明的是：在有效 skill 后端上，选什么经验会改变最终能力，因为不同 policy 决定了哪些 capability skill 被激活、何时被激活。它不证明 learned routing，也不证明内容内化。

R5.1（Case B）把机制再收一层：

- 10-seed frontier−passive = **+0.292**，置信区间不跨 0，10/10 胜出
- RealFit 0.519 并不强于 CountOnly 0.546
- Helion 上 text-router = oracle
- 结论：**experience-gated skill activation**（采够该 capability 的条数 → 解锁对应 frozen primitive），不是 `fit()` 从轨迹内容里学到关键配置

R5.2（Case A）排除“只是 Ace-Skill 式优先采样换壳”：

- 同一 CountOnly unique-task 后端、10 seed
- frontier−Ace-Skill = **+0.306**，10/10
- Ace-Skill 与 passive 打平（gap=0）
- frontier 更快激活第二个 skill
- 停止继续堆 Helion，进入第二个 benchmark

### 4.4 R5.3–R6：离开 Helion 之后，采集启发式都不泛化

公开环境是 HotpotQA / 2Wiki / MuSiQue + Wikipedia 2018 E5。

| 轮次 | 判定 | 数字 |
| --- | --- | --- |
| R5.3 | Case C，3-seed stop | Skill backend 3/3 accept（C2 ΔEM +0.417），但冻结 frontier ΔFuture=**−0.0067**，Ace +0.081 / passive +0.082，**0/3 seed 赢**。48/48 预算全采 HotpotQA C1。不扩 10 seed，不跑 Bamboogle，不调系数。 |
| R5.4 | Case E，3-seed stop | 丢掉 Helion frontier，改测 RASA（`q×p` + Sequential Readiness Test）。`rasa_full` mean macro=**0.111** < Ace/passive **0.120**（0/3，差 −0.009）。SRT 正确 ACTIVE C2、PARK C1，但 C3 `P_good=0.105` 未过 PARK 线，剩余约 35/48 预算砸在几乎不可修的 long-chain。Gate A–E 未过，仅 F 过。RASA 主线证伪。 |
| R6 | CASE_C，3-seed stop | Novelty `PROCEED`，pipeline audit PASS，task-level `q×p` Spearman=0.29。冻结 CountOnly 上 exact shadow utility：`exact_ava` mean=**0.120** vs best baseline **0.119**（+0.001，远小于 Gate A 要求的 +0.02），seed 胜 **1/3**。终局几乎完全由激活 mask 决定（C1+C2+C3=0.120，C2-only=0.111）。**不训 surrogate，不改 M/K/shadow，不缩 claim。停止 ExPolicy acquisition 主线。** |

### 4.5 SkillRL-ESI：可执行 skill 能涨分，但不是这份经验的独特内容

设置：SkillRL `8e66726`，工作分支 `esi-mvp`，主代码 `/data/ppnm/skillrl_esi`。冻结 Qwen2.5-3B + SkillRL SearchEnv。不导入旧 ExPolicy 的 CountOnly / frozen primitive。

Held-out future_test（n=150，三 seed 逐格相同）：

| condition | val EM | test EM | Δ vs B0 |
| --- | ---: | ---: | ---: |
| B0 no skill | 0.325 | 0.2867 | 0 |
| B1 SkillRL textual | 0.3375 | 0.1933 | −0.093 |
| ESI real | 0.450 | **0.3933** | **+0.107** |
| ESI empty | 0.400 | **0.460** | +0.173 |
| ESI shuffled | 0.400 | 0.400 | +0.113 |

- 相对 B0：`CASE_ESI_PASS`。Δ=+0.107 ≥ +0.03，3/3 seed。
- 相对 empty / shuffled：**内容因果未建立**。test 上 empty 0.46 ≥ ESI 0.393。增益来自 typed comparison 执行程序，不是这 240 条 train packet 的独特映射。
- B1 官方文本 skill 在 test 上更差（0.1933），所以正增益不是“任何 skill prompt 都会涨”。
- 当时写下的下一步是 skill induction / editing，并且在打过 empty/shuffle 之前不重开 acquisition。R7 没有把这件事当成前置门槛。

### 4.6 R7：闭环第一次跑通，第一版策略没有赢

按 `ExPolicy_R7_Closed_Loop_MVP_H20_Instructions.md`。分支 `r7-closed-loop`，基线 git `5c9dbf22`。Actor 仍是冻结 Qwen2.5-3B。Skill bank 是冻结的 R5.3 C1/C2/C3，`r7_mutable=false`。Policy 是 Linear Thompson Sampling，不是 frontier / RASA / AVA / CountOnly。Reward 用 shadow probe，**不用** final future_test。预算 6 round × batch 8 = 48。日历墙钟 2026-08-19 10:43 → 2026-08-21 00:30。

本轮只回答两件事：闭环是否真实跑通（Gate P）；第一版 Linear TS 是否已经优于 fixed / passive / CountOnly（Gate M）。

**Gate P：PIPELINE_PASS**

| Gate | 判定 | 要点 |
| --- | --- | --- |
| P-A Statefulness | PASS | 持久干预改变后续 rollout；每 seed 激活 1 个 frozen skill |
| P-B Acquisition adaptivity | PASS（弱） | 后验每轮有更新，但 max TV=0.024，低于建议的 0.10 |
| P-C Assimilation adaptivity | PASS | 四类都出现：ACTIVATE 3 / DISCARD 5 / EPISODIC 4 / SEMANTIC 6 |
| P-D Reward closure | PASS | reward 来自 shadow probe，`uses_future_test=false` |
| P-E Feedback ablation | PASS | NoFeedback 三 seed 全部 `acq_delta=assim_delta=0` |
| P-F Leakage | PASS | final future_test 不进 policy |
| Pipeline audit | 24/24 | `tests/r7` 14 passed |

**Gate M：PIPELINE_PASS_METHOD_AMBIGUOUS，脚本分型 CASE_B**

| 条件 | final macro | ΔFuture | AULC | skills |
| --- | ---: | ---: | ---: | --- |
| M0 expolicy（Linear TS） | **0.0474** | +0.030 | 0.030 | 1 / seed |
| B0 passive_noop | 0.0163 | +0.003 | 0.013 | 0 |
| B1 passive_episodic | 0.0326 | +0.020 | 0.026 | 0 |
| **B2 CountOnly** | **0.0911** | +0.078 | 0.043 | 3 |
| B3 random_joint | 0.0496 | +0.037 | 0.027 | 0–2 |
| **B4 all_skills**（仅 seed 0） | **0.0911** | 0 | 0.091 | 3 |
| A1 no_feedback | 0.0437 | +0.028 | 0.029 | 1 |
| A2 reward_shuffle | 0.0437 | +0.030 | 0.030 | 1–2 |
| A3 assimilation_fixed | 0.0681 | +0.055 | 0.054 | 2–3 |

Method gate：M0 没有超过 strongest non-oracle +0.02（0.0474 vs 0.0911，差 −0.044）；0/3 seed 严格更优；AULC 也更差。M0 只比 NoFeedback / RewardShuffle 高约 +0.004。只学 acquisition、同化钉死 CountOnly 的 A3（0.068）高于同时学两边的 M0。

行为上，终局几乎是“有没有打开 C2”的查表：C2 EM≈0.282 时 macro≈0.084–0.091；C3 EM=0。M0 seed1 激活了 C2，到 0.084；seed0 激活了无用的 C3；seed2 采到 C2 但全部 DISCARD。Linear TS 没有稳定路由到 C2。

当时 CASE_B 的写法是“闭环成立，下一步只改 policy / credit / exploration”。**R8 随后把这条路关掉了。**

### 4.7 R8：失败不是控制器不够强，是决策空间本身退化

按 `ExPolicy_R8_Next_Run_Instructions.md`。禁止把 R7 的失败解释成“换一个更强 bandit”。必须先证明 assimilation 决策空间可辨识，才允许训练 controller。

**Novelty：`PROCEED`。** 12 篇必读加 2026 近邻，没有一篇同时满足四条交集：真实环境 acquisition、异构持久 assimilator、held-out future capability 作为经验价值、联合 stateful loop。MAGE、MemRL、SkillOpt 都只占单轴。

**R8-A（83/83，不训练）判定 `R7_ENV_STRUCTURALLY_DEGENERATE`：**

| 检查 | 结果 |
| --- | --- |
| S1 hindsight oracle 终局 | **0.0644**，低于 CountOnly / all_skills **0.0911**。门控要求 oracle ≥ fixed+0.03，在这个 substrate 上数学上不可能过。 |
| 逐步 utility | oracle 0.0244 vs always-ACTIVATE 0.0260，gap **−0.0016** |
| S2 mask ceiling | C2=0.0822；C1+C2=C1+C2+C3=**0.0911**。max_mask − strongest_fixed = **0**。增益几乎全是打开 C2。 |
| 18×4 反事实矩阵 | unique-best ≥20% 的动作只有 **ACTIVATE_EXISTING_SKILL**（33.3%）。需要至少两个。 |
| S3 可分性 | majority 33.3%；MLP LOSO **27.8%**，比 majority 低 5.6 个百分点。 |

因此禁止在冻结 C1/C2/C3 的 SearchEnv 上换 Linear TS / MLP / 更强 bandit。

**R8-B：ALFWorld + 官方 MemRL / SkillOpt，无源码 patch。Foundation FAIL。**

| job | 结果 |
| --- | --- |
| no-learning smoke | success=**0**（n=22，wins=0） |
| SkillOpt smoke | success=**0**（n=10）。train 3 step 全 `skip_no_patches`。3B 几乎不产出 `<action>`，环境反复 `look`。 |
| MemRL smoke | 卡在 MOS 初始化，从未进入 ALFWorld |
| 120 局 foundation | 写出 16 个 conversation 后挂在 AlfredTWEnv 重初始化 |

协议要求 no-learning 落在 10%–70%。已有数字是 0%，低于地板。因此没有启动全量 3-seed、Gate H、Stage-1 joint policy。

R8 结论：下一步如果还做，应换一个 no-learning 落在 10%–70%、且天然存在异构持久 intervention 的公开环境。协议点名的候选是 WebShop、ScienceWorld、LifelongAgentBench。不给 ALFWorld 人工加惩罚，不把项目改写成 SkillOpt 调度器。

---

## 5. R9：换到 LifelongAgentBench，做到矩阵中途停机

指令：`ExPolicy_R9_Qwen_LifelongAgentBench_Substrate_Joint_Identifiability_H20_Instructions.md`  
产物目录：`outputs/r9_lab/`  
进程记录：`outputs/r9_lab/jobs/PIDS.txt` 末行写明 **2026-08-22T09:37:13+08:00 停止全部在跑的 R9 实验进程**（矩阵、vLLM、supervisor、下载、Virtuoso）。  
`expolicy-result-record.md` **没有 R9 一节**。下面全部来自 R9 产物，不是正式记录里的已签字结论。

### 5.1 这一轮要回答的两个问题

R9 不是为了挽救 R7 的 Linear TS。它只回答：

**Q1.** 是否存在一个公开、可复现、foundation 合理的 lifelong-agent substrate？  
冻结 actor 在 no-learning 下不能接近 0，也不能接近天花板，必须同时有大量成功和失败轨迹。

**Q2.** 在同一条真实经验上，是否真的存在“该怎么学”的非平凡选择？  
同一 batch、同一初始状态上，DISCARD / EPISODIC / SEMANTIC / SKILL 的 future capability gain 要显著不同，而且不能有一个固定算子在所有 batch、所有任务族上统治。

效用定义：

\[
U(B,A\mid S)
=
J(A(S,B); D_{\mathrm{shadow}})
-
J(S; D_{\mathrm{shadow}})
-
0.5\max(0, -\Delta J_{\mathrm{retention}})
\]

**只有 Q1 和 Q2 都成立，才允许进入下一轮 learned joint ExPolicy。** Q2 的正式判定是 Gate H（H1–H6）。Gate H 过后才做 Gate J（采集与同化是否必须联合）。

环境选 LifelongAgentBench（DB / OS / KG），原因是三个异构交互环境、任务共享 atomic skill、自动 verifier、官方测 lifelong experience replay、代码和任务序列公开。原论文里 Qwen2.5-7B/32B 在 DB 上有天花板风险，QwQ-32B 的 DB no-replay 约 0.29，所以必须先做预注册的 Qwen 模型筛选，不能默认“更大的 Qwen 更好”。

筛选带：**success ∈ [0.10, 0.70]**，parser 有效率等附加约束在跑前写死。规则是 Full-PASS 里选最低 Score，平局顺序 Q0 > Q1 > Q2 > Q3。

| ID | 模型 | 角色 |
| --- | --- | --- |
| Q0 | Qwen3-30B-A3B-Instruct-2507 | 大 MoE |
| Q1 | QwQ-32B | 论文里的 foundation anchor |
| Q2 | Qwen3-14B | 若大模型天花板，用较弱 dense 检查 |
| Q3 | Qwen3-32B | 大 dense |

### 5.2 模型筛选：选定 Qwen3-14B

每个环境 24 题，与正式 split 隔离。产物：`outputs/r9_lab/R9_MODEL_SCREEN.csv`、`audit/MODEL_SELECTION.md`、`select_model/DONE.json`、`model_manifest.json`。

| 模型 | DB | OS | KG | 判定 |
| --- | ---: | ---: | ---: | --- |
| Q0 Qwen3-30B-A3B | **0.917** | 0.542 | 0.625 | 失败。DB 超出 [0.10, 0.70] |
| Q1 QwQ-32B | 0.417 | 0.083 | 0.208 | 失败。DB parser_valid_rate=0.50 < 0.9；OS parser 仅 0.042 |
| **Q2 Qwen3-14B** | **0.417** | **0.583** | **0.417** | **Full-PASS，score=0.171，入选** |
| Q3 Qwen3-32B | 0.583 | 0.625 | 0.667 | Full-PASS，score=0.372，高于 Q2，未选 |

冻结清单：

- `selected_model_id`: Q2
- 本地路径：`/data/ppnm/models/Qwen3-14B`
- dtype：bfloat16
- thinking：关闭
- served name：`qwen3-14b-r9`
- `frozen: true`

### 5.3 正式 foundation：Q1 这一侧落在带内

Split seed `20260822`，sha256 `63ca95237468378f5eb3a74df0875b23c38f00a2c4bf58200d4473df350bf989`。model_screen 与正式 bucket 隔离。产物：`R9_FOUNDATION.csv`、`audit/SPLIT_AUDIT.md`。

| env | n | success | 成功 / 失败 | parser_valid | infra_fail | mean steps |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| DB | 73 | **0.616** | 45 / 28 | 1.000 | 0 | 2.85 |
| OS | 73 | **0.575** | 42 / 31 | 1.000 | 0 | 4.85 |
| KG | 57 | **0.404** | 23 / 34 | 0.877 | 0 | 8.42 |

三个环境都在 0.10–0.70 内，而且成功、失败都不少。相对 R8 的 ALFWorld 3B（success=0），这一侧的 substrate 条件已经满足。KG 的 parser 有效率 0.877，低于 DB/OS 的 1.0，写进记录时不要说成三个环境解析都完美。

正式 split 规模：

| bucket | DB | OS | KG |
| --- | ---: | ---: | ---: |
| model_screen | 24 | 24 | 24 |
| calibration | 73 | 73 | 57 |
| acquisition_pool | 183 | 183 | 140 |
| assim_validation | 55 | 55 | 43 |
| shadow_reward_probe | 55 | 55 | 43 |
| retention_probe | 37 | 37 | 29 |
| future_test | 73 | 73 | 60 |

### 5.4 算子健全性：通过，但 SkillOpt 在 sanity 里就没有被接受

产物：`outputs/r9_lab/audit/OPERATOR_SANITY.md`，总状态 **`OPERATOR_SANITY_PASS`**。DB / OS / KG 各自 `OPERATOR_SANITY_PASS`。

检查项：状态文件会变；后续 rollout 读到新状态；关掉后回到基线；clone A 不污染 clone B；future_test 不进更新；SkillOpt 拒绝时不写状态；检索只命中训练侧经验。

Sanity 用的是每环境两个 8 条经验的 batch，不是正式矩阵。观察：

- EPISODIC_REPLAY、SEMANTIC_MEMORY：`changed=true`，后续能读到。
- SKILLOPT_EDIT：`changed=false`，`accepted=false`，原因 `validation_did_not_improve`。拒绝是干净的（不写状态），所以健全性可以通过，但这已经说明 skill 编辑这条动作在 sanity batch 上没有产生持久更新。

### 5.5 同化矩阵：计划 36×4，停机时完整四分支只有 21 个 batch

计划（指令 §9）：每个环境 12 个 batch，每批 8 条真实经验，共 36 batch、288 条经验。每个 batch 从干净状态 S0 分出四个独立 clone：DISCARD、EPISODIC、SEMANTIC、SKILLOPT。只评 shadow 和 retention，禁止看 future_test。冻结文件：`outputs/r9_lab/FROZEN_BATCHES.json`（`n_batches=36`，`frozen_before_utility=true`）。

计划输出 `outputs/r9_lab/R9_ASSIM_MATRIX.csv`。**这个文件不存在。** 没有 Gate H 分析脚本的最终判定。

2026-10-09 清点 `outputs/r9_lab/matrix/` 的 `DONE.json`：

| env | 已出现的 batch | acquire 完成 | 四分支都完成 | 未完成 |
| --- | ---: | ---: | ---: | --- |
| DB | B00–B08（9/12） | 9 | **8**（B00–B07） | B08 只有 DISCARD；B09–B11 未开始 |
| OS | B00–B03（4/12） | 4 | **3**（B00–B02） | B03 只有 acquire；B04–B11 未开始 |
| KG | B00–B11（12/12） | 12 | **10**（B00–B09） | B10 缺 SEMANTIC 与 SKILLOPT；B11 只有 acquire |
| 合计 | 25 个 batch 目录 | 25 | **21 / 36** | 已完成格子 **87 / 144** |

另外 `_s0/{db,os,kg}` 有 DONE。停机前最后还在写的格子包括 `db-B08/EPISODIC_REPLAY`、`kg-B10/SEMANTIC_MEMORY`、`kg-B11/DISCARD`、`os-B03/DISCARD`。这些格子有 `STATUS_LIVE.json`，没有对应 `DONE.json`，不能当成完成结果。

### 5.6 已完成格子的描述性数字（不是 Gate H）

下面只统计带 `DONE.json` 的格子，用来说明停机时数据长什么样。**不能当成 Q2 已回答，也不能写成 Gate H 通过或失败。** 样本不完整：OS 只有 3 个完整 batch，DB 缺 4 个 batch，KG 缺 2 个完整 batch。

已完成格子上的平均 U：

| env | DISCARD | EPISODIC_REPLAY | SEMANTIC_MEMORY | SKILLOPT_EDIT |
| --- | ---: | ---: | ---: | ---: |
| DB | −0.050（0/9 为正，n=9） | −0.064（1/8，n=8） | −0.049（1/8，n=8） | −0.010（0/8，n=8） |
| OS | −0.012（0/3，n=3） | **+0.085（3/3，n=3）** | **+0.032（3/3，n=3）** | −0.061（0/3，n=3） |
| KG | −0.016（0/11，n=11） | **+0.099（11/11，n=11）** | −0.076（1/10，n=10） | 0.000（0/10，n=10） |

21 个完整 batch 里，hindsight 最优动作的计数：

| env | DISCARD | EPISODIC | SEMANTIC | SKILLOPT |
| --- | ---: | ---: | ---: | ---: |
| DB（8） | 1 | 2 | 2 | 3 |
| OS（3） | 0 | 3 | 0 | 0 |
| KG（10） | 0 | 9 | 1 | 0 |

读这些数字时必须带上三个实现事实，否则会把“谁最优”读错：

1. **SkillOpt 在 21 个已完成格子上全部拒绝。** `changed=0/21`，原因一律 `validation_did_not_improve`。它没有改写持久状态。DB 上它有时成为“最优”，是因为 U=0，而另外三个动作为负，不是因为 skill 编辑成功了。
2. **DB 上 DISCARD 的 U 九个 batch 完全相同（−0.049877）。** KG 的 SKILLOPT U 十个 batch 全是 0。这种格子间不变的数，更像评测偏移或 no-op，不是 batch 特异的学习信号。
3. **KG / OS 上 episodic replay 在已完成格子里稳定为正**（KG 11/11，OS 3/3），semantic 在 KG 上多为负、在 OS 的 3 个 batch 上为正，DB 上四种动作的平均 U 都 ≤ 0。如果剩余 batch 保持这个形状，很像“环境级的固定载体”，而这正是 Gate H3 要排除的情况。剩余 batch 没跑完之前，这只是风险，不是判定。

21 个完整 batch 的 best-vs-worst spread 均值约 0.145，说明格子内部动作之间的分数差并不小。差从哪来、是不是可学习的非平凡选择，要等矩阵补全后按 H1–H6 算，不能用 spread 代替。

### 5.7 Gate H / Gate J 的门槛（写在这里是为了说明还没跨过哪条线）

Gate H 必须 H1–H6 同时成立，才允许说 LifelongAgentBench 提供了可辨识的异构同化决策空间：

| 门 | 要求 | R9 现状 |
| --- | --- | --- |
| H1 | 全局 36 batch 里至少两个动作的 unique-best ≥ 0.15，且不能只靠 ±1 个 probe 翻转 | 未算。矩阵未齐。 |
| H2 | hindsight oracle 的 mean U ≥ 全局最优固定动作 + 0.03（若 0.03 小于两题，取更严的两题等价） | 未算。 |
| H3 | oracle ≥ 每环境一个固定动作 + 0.02，且至少 2/3 环境内部有 ≥2 个动作 unique-best ≥ 0.20。不允许退化成“DB 用 memory、OS 用 skill”的环境 ID 路由 | 未算。已完成格子在 KG/OS 上高度偏向 episodic，这是 H3 的风险信号。 |
| H4 | 只用训练侧特征做留一 batch 预测：full-feature ≥ majority + 10pp，且 ≥ env-only + 5pp | 未跑。 |
| H5 | 至少两个非 DISCARD 算子各自在 ≥4 个 batch 上 U>0，并且彼此有明显优劣交错 | 未正式判定。SkillOpt 目前没有正效用区域。 |
| H6 | future_test 不参与更新；shadow 与 retention 无重叠；SkillOpt 验证集与 shadow 分离；不注入 gold skill / gold plan | 健全性阶段的检查是过的；全矩阵的 H6 审计未作为 Gate H 结案。 |

判定树：

- `CASE_H_PASS`：进入 Phase 4 Joint Identifiability（Gate J）。
- `CASE_H_FIXED_DOMINATES`：固定动作约等于 oracle，或只有一个动作真的赢。停止。不得改写成 SkillOpt 调度器、更好的 memory、或某一个 task family 上选载体。

Gate J 问的是联合必要性：同时学习“采什么”和“怎么同化”是否比只学一边更好。**R9 没有进入这一阶段。**

### 5.8 R9 可以宣称的和不能宣称的

可以写进进展、但还不是论文结论的：

- 模型筛选按预注册规则选出了 Qwen3-14B，Qwen3-30B-A3B 在 DB 上天花板、QwQ-32B 解析失败。
- 正式 foundation 三个环境的 no-learning 成功率在 0.40–0.62，Q1 的“不要接近 0 / 不要贴天花板”这一侧成立。
- 四个同化分支的实现通过了算子健全性：状态隔离、回读、关闭还原、future_test 不泄漏。
- 矩阵跑过一大半 KG 和大部分已启动的 DB，然后在 2026-08-22 上午被整轮停掉。

不能写的：

- 不能写 Gate H 通过或失败。
- 不能写“episodic 已经是 LifelongAgentBench 上的正确同化策略”。
- 不能写 SkillOpt 已作为有效 assimilation action 被验证。21/21 已完成格子是拒绝。
- 不能写 ExPolicy 已经在 R9 上训练，或 joint acquisition×assimilation 已被证明必要。
- 不能把 `STATUS_LIVE.json` 里没写完的格子补成结果。

---

## 6. 主线现在停在哪，以及记录里规定的下一步

按时间把停止原因叠起来：

1. **R6** 停的是离线 acquisition heuristic。Exact AVA 没有超过 Ace/passive，不训 surrogate。
2. **ESI** 把内容因果停住。空 packet 也能诱导出同等或更强的 comparison runtime。
3. **R7** 把闭环 runtime 做成了，第一版 Linear TS 没有赢过 CountOnly。
4. **R8** 说明这不是“控制器太弱”：旧 SearchEnv 的 oracle 低于固定 CountOnly，ALFWorld 3B foundation 是 0。禁止在那个环境上换 bandit。
5. **R9** 按 R8 的下一步换到 LifelongAgentBench。substrate 和算子管线已经立起来，**辨识性实验没跑完，所以 learned ExPolicy 仍然不允许开始。**

如果按 R9 指令继续，下一件具体的事是：

- 用同一冻结 split、同一 Qwen3-14B、同一批 `FROZEN_BATCHES.json`，把剩下的格子跑完（DB B08 的三个未完成分支和 B09–B11；OS B03–B11；KG B10 的 SEMANTIC/SKILLOPT 与 B11 的四个分支）。
- 写出 `R9_ASSIM_MATRIX.csv`，按 H1–H6 做 Gate H。
- 只有 `CASE_H_PASS` 才进入 Gate J。
- Gate H 若是 `CASE_H_FIXED_DOMINATES`，按指令停止，不把论文改成 memory 或 SkillOpt 调度。
- 指令还写了：formal foundation / Gate H / Gate J 失败后，不要当场换模型；也不要回到 Model Ladder 把第二名 Qwen3-32B 当成补救，除非按原文的失败条款重新开一轮。当前 foundation 并没有失败，所以不存在“因为 foundation 去换 Q3”的理由。

---

## 7. 证据落点

| 内容 | 路径 |
| --- | --- |
| 正式记录（至 R8） | `expolicy/expolicy-result-record.md` |
| R7 指令 / 产物 | `ExPolicy_R7_Closed_Loop_MVP_H20_Instructions.md`，`outputs/r7_closed_loop/` |
| R8 指令 / 产物 | `ExPolicy_R8_Next_Run_Instructions.md`，`outputs/r8_preflight/`，`outputs/r8_backend_validation/` |
| R9 指令 | `ExPolicy_R9_Qwen_LifelongAgentBench_Substrate_Joint_Identifiability_H20_Instructions.md` |
| R9 模型筛选 | `outputs/r9_lab/R9_MODEL_SCREEN.csv`，`outputs/r9_lab/audit/MODEL_SELECTION.md`，`outputs/r9_lab/model_manifest.json` |
| R9 foundation / split | `outputs/r9_lab/R9_FOUNDATION.csv`，`outputs/r9_lab/audit/SPLIT_AUDIT.md`，`outputs/r9_lab/split.json` |
| R9 算子健全性 | `outputs/r9_lab/audit/OPERATOR_SANITY.md` |
| R9 冻结 batch 与未完成矩阵 | `outputs/r9_lab/FROZEN_BATCHES.json`，`outputs/r9_lab/matrix/` |
| R9 停机记录 | `outputs/r9_lab/jobs/PIDS.txt` |

---

## 8. 不宣称清单

截至 2026-10-09，ExPolicy **没有**证明下列命题：

- Active acquisition 在任意 assimilation 下成立。
- Adaptive assimilation / learned routing 已经优于固定载体。
- 经验内容被内化成 unique SkillSpec（ESI 的 empty/shuffled 对照否定了这一点）。
- Helion frontier、RASA、Exact AVA 可以迁到公开多跳搜索环境。
- R7 的 Linear Thompson 优于 CountOnly。
- 在 R7 SearchEnv 上换更强控制器就能制造出决策空间。
- ALFWorld + 冻结 3B 上 MemRL / SkillOpt 是有效 assimilator。
- LifelongAgentBench 上异构同化动作可辨识（R9 Gate H 未做）。
- 联合 Experience Policy 已经被训练或已经被证明必要（R9 Gate J 未做）。

已经可以站住的只有这些：

- 在 Helion 上，可执行 procedure 是当时唯一稳定的能力干预；“采哪类经验”能通过解锁这些 procedure 改变未来分数，但这是环境特异的门控激活，离开 Helion 后没有泛化。
- 在公开搜索环境上，冻结 executable skill 的激活 mask 几乎决定终局；第一版闭环控制器打不过“采够就打开全部 frozen skill”。
- 这个搜索环境的反事实上界低于该固定策略，所以问题在 substrate，不在还没试过的 bandit。
- LifelongAgentBench + Qwen3-14B 把 no-learning 放到了中间区间，同化管线的健全性通过了，辨识性矩阵没有做完。
