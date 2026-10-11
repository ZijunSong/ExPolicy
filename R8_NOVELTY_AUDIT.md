# ExPolicy R8 Novelty Collision Audit

- **Date:** 2026-08-21
- **Auditor:** R8 preflight novelty pass (full-text, not abstract-only)
- **Scope:** Must-read 12 papers + 2025–2026 web search for the joint claim *experience acquisition policy + heterogeneous assimilation operators + future capability reward*
- **Hard-stop rule:** A prior paper triggers `NOVELTY_COLLISION_HARD_STOP` **only if it simultaneously implements all four** of the ExPolicy unique intersection below. Single-axis prior art (memory policy, skill editing, curriculum, skill routing, self-evolution, experience distillation) is **preempted as a novelty claim**, but is **not** a hard stop.

---

## 0. ExPolicy unique intersection (frozen)

R8 novelty can only be claimed at this **intersection**. All four must be present in one prior paper to stop:

1. **Acquisition of real environment experience.** The agent *actively chooses the next batch of real environment interaction*, not a retrieval item, memory entry, or generated self-play problem.
2. **Heterogeneous persistent assimilators on the same experience.** For *that same acquired experience*, a policy chooses among multiple persistent update operators (e.g. discard / episodic-memory update / procedural-skill update / …), not a single substrate.
3. **Held-out future capability as experience value.** The value of an experience is the *held-out future capability change after that experience is actually consumed by currently available assimilators*, not immediate reward, difficulty, similarity, or a fixed curriculum score.
4. **Joint stateful loop.** Acquisition and assimilation policies update in the *same* loop, and the mutated persistent state changes *what experience should be acquired next*.

**Forbidden single-point novelty claims:** memory policy; skill editing; counterfactual memory credit; task curriculum; skill routing; self-evolution; experience distillation.

---

## 1. Fetch status

All 12 must-read papers were read in intro + method + related work + claims. One ACL PDF timed out; the arXiv equivalent was used.

| Paper | Primary full text | Fallback | Status |
|---|---|---|---|
| AgeMem | ACL PDF `2026.acl-long.981.pdf` | ACL HTML | OK |
| MemoPilot | arXiv PDF `2606.08656` | — | OK |
| HiMPO | arXiv PDF `2606.16285` | — | OK |
| MemRL | arXiv PDF `2601.03192` | GitHub README + `configs/rl_alf_config.yaml` | OK |
| SkillOpt | arXiv PDF `2605.23904` | GitHub README + docs + `configs/alfworld/default.yaml` | OK |
| Mem2Evolve | arXiv PDF `2604.10923` | ACL HTML `2026.acl-long.952` | ACL PDF **timeout**; arXiv full text OK |
| MAGE | arXiv PDF `2605.10064` | HTML v1 | OK |
| SESA | arXiv PDF `2607.29468` | — | OK |
| ProactAgent | arXiv PDF `2604.20572` | HTML | OK |
| MemEvolve | arXiv PDF `2512.18746` | HTML / Semantic Scholar authors | OK |
| MemSkill | arXiv PDF `2602.02474` | — | OK |
| Meta Context Engineering | arXiv PDF `2601.21557` | — | OK |

Raw notes: `outputs/r8_preflight/novelty_notes/`.

Additional 2026 papers actually fetched after search (not just titles): **Evolving-RL**, **JERP**, **SEARL**, **MSCE**, **MemCon**, **Memento-Skills**, **ERSkill**, **Rethinking Continual Experience Internalization**.

Cross-model Codex MCP review was **not available** in this environment (no Codex MCP server). Verdict below is from direct full-text reading.

---

## 2. Must-read papers

### 2.1 AgeMem — Agentic Memory

- **Title:** Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents
- **Authors:** Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, Libing Wu
- **Year / venue:** 2026, ACL (long), pages 21457–21483
- **URL:** https://aclanthology.org/2026.acl-long.981/ · PDF: https://aclanthology.org/2026.acl-long.981.pdf

**What they actually do (from method, not abstract).**  
AgeMem folds LTM and STM into one LLM policy whose action space includes language generation plus six memory tools: LTM `ADD` / `UPDATE` / `DELETE` and STM `RETRIEVE` / `SUMMARY` / `FILTER`. Training is three-stage progressive RL (store into LTM → STM control under distractors → coordinated retrieval/reasoning) with step-wise GRPO that back-propagates a *terminal* trajectory reward to earlier memory actions. The reward in Eq. (1) is a weighted mix of *task performance and memory quality* plus penalties for redundant storage and context bloat. Sessions / queries are given; the agent does not choose which real environment to interact with next.

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Trajectory is a given conversation/task. Stage 1 “casual interactions” and Stage 3 “formal query” are *provided sessions*, not a policy over next real env batches. |
| 2 | **PARTIAL** | Heterogeneous ops exist, but they are *memory-store ops*, not assimilators of a newly acquired env experience into episodic vs procedural vs discard. Quote: “a unified policy over heterogeneous memory actions (both persistent LTM operations and contextual STM operations) is trained end-to-end.” |
| 3 | **NO** | Credit is terminal task reward, not held-out future capability after an assimilator consumes an experience. Quote: “rt is typically zero for intermediate steps and assigned after trajectory completion.” |
| 4 | **NO** | No acquisition policy. Memory state affects later retrieval, but “what to experience next” is not learned. |

**Preempts:** learned policy over memory operations (store / retrieve / update / summarize / discard).  
**Remaining delta:** AgeMem never chooses the next *real env batch*, never routes one experience across skill vs episodic vs discard assimilators, and never scores that choice by held-out future capability.

---

### 2.2 MemoPilot

- **Title:** From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory
- **Authors:** Yishuo Cai, Xingyu Guo, Xuancheng Huang, Jinhua Du, Can Huang, Wenxuan Huang, Wenhan Ma, Yuyang Hu, Aohan Zeng, Jie Tang, Xu Sun
- **Year / venue:** 2026, arXiv 2606.08656 (ICML-style preprint)
- **URL:** https://arxiv.org/abs/2606.08656 · PDF: https://arxiv.org/pdf/2606.08656

**What they actually do.**  
A trainable memory copilot `G_θ` updates textual memory `m_t = G_θ(e_t, m_{t-1})` after each game; a *frozen* player `π` then plays the next game conditioned only on `m_t`. Memory updating is an MDP optimized with multi-turn GRPO. The practical credit signal is a **one-step next-game proxy** `R_{i,t} = r_{i,t+1}` (not a held-out capability probe after choosing among assimilators). The interaction stream is sequential games against constructed opponents (RPS, Limit Hold’em) plus a StreamBench transfer; the agent does not choose *which* environment experience to collect. Action space is the space of textual memories — a **single** update operator.

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Games arrive as a sequential TTL stream `{g_t}`. No policy over “which real env batch next.” |
| 2 | **NO** | Single operator: rewrite the memory artifact. Quote: “the action space A is the space of textual memories.” |
| 3 | **PARTIAL** | Memory quality is judged by *downstream next-game outcome*, which is future relative to the write — but it is the same task stream’s immediate next reward, not a held-out capability distribution after heterogeneous assimilators. Quote: “we estimate turn-level advantages using the one-step outcome R_{i,t}=r_{i,t+1}.” |
| 4 | **NO** | Only the memory-update policy is trained. Player is frozen. Updated memory changes *how the player acts*, not *what experience is acquired next*. |

**Preempts:** RL-trained *how to update memory* over sequential interactions.  
**Remaining delta:** no real-env acquisition policy; no operator choice; value is next-game score, not held-out assimilator-conditioned capability.

---

### 2.3 HiMPO

- **Title:** HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents
- **Authors:** Jiangze Yan, Yi Shen, Wenjing Zhang, Jieyun Huang, Zhaoxiang Liu, Ning Wang, Kai Wang, Shiguo Lian
- **Year / venue:** 2026, arXiv 2606.16285
- **URL:** https://arxiv.org/abs/2606.16285 · PDF: https://arxiv.org/pdf/2606.16285

**What they actually do.**  
HiMPO treats each compressed-memory write `m_{i,t} ~ π_θ(· | H_{i,t})` as a state-writing action. Credit is (i) **local counterfactual utility**: replace previous vs updated memory under the *same pre-write state* and score answerability of the oracle outcome; (ii) **hindsight relevance** as a gate. The memory-specific advantage is applied only to memory tokens; the rest of the trajectory uses ordinary outcome GRPO. Tasks are long-horizon QA / browsing; the agent does not choose what to experience in a real env, and there is only one persistent substrate (compressed memory).

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Queries `q_i` are given. Tool calls collect evidence for that query; there is no batch-level env-experience acquisition policy. |
| 2 | **NO** | Single write type: compressive memory summary. Quote: “the memory write at step t can therefore be viewed as a policy-controlled action.” |
| 3 | **PARTIAL** | Local utility is a counterfactual of *information recoverability for the current target outcome*, not held-out future tasks after assimilation. Quote: “Δ_{i,t} estimates the marginal utility of the current memory write.” |
| 4 | **NO** | No acquisition policy co-trained with assimilation. |

**Preempts:** counterfactual / hindsight credit assignment for memory writes.  
**Remaining delta:** credit-for-memory-write ≠ policy over (what to experience, which assimilator) under held-out future J.

---

### 2.4 MemRL

- **Title:** MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory
- **Authors:** Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Zhuo Li, Yujie Zheng, Weinan Zhang, Ying Wen, Zhiyu Li, Feiyu Xiong, Yutao Qi, Bo Tang, Muning Wen
- **Year / venue:** 2026, arXiv 2601.03192
- **URL:** https://arxiv.org/abs/2601.03192 · PDF: https://arxiv.org/pdf/2601.03192 · Code: https://github.com/MemTensor/MemRL

**What they actually do.**  
Frozen LLM + evolving episodic memory of Intent–Experience–Utility triplets. Retrieval is two-phase: cosine recall then Q-value re-rank. After the frozen model acts, environmental reward updates Q of retrieved items (Monte Carlo / one-step) and a summarized trajectory is written as a new triplet. Benchmarks include **ALFWorld** (official `run/run_alfworld.py`). The “action” in their M-MDP is *selecting a memory item*, not choosing the next real environment. There is one substrate: episodic memory (build strategy in config is even named `proceduralization`, but the learned object is still Q over retrieved episodes).

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Tasks come from HLE / BCB / ALFWorld / LLB splits. Quote: “the action space A_t becomes dynamic and discrete, corresponding to selecting a specific m from the memory bank.” That is retrieval, not env acquisition. |
| 2 | **NO** | Always write a new triplet + update Q of retrieved items. No discard vs skill vs episodic *choice*. |
| 3 | **NO** | Utility is expected *current-task* env reward of using memory m. Quote: “Q(s,m) as the expected utility of the subsequent action a augmented by memory m”; update `Q ← Q + α(r − Q)` with r = execution success. |
| 4 | **PARTIAL** | Q-updates change later retrieval, which changes later success, which writes more memory — a retrieval loop, not an acquisition+assimilation loop. |

**Preempts:** learn utility of retrieved episodic experience from env feedback; ALFWorld runtime memory RL.  
**Remaining delta:** MemRL scores *memories as retrieval actions*. ExPolicy scores *real env batches as experiences to assimilate through a chosen persistent operator*, using held-out future capability.

**ALFWorld reproduction (official).** See §6.1.

---

### 2.5 SkillOpt

- **Title:** SkillOpt: Executive Strategy for Self-Evolving Agent Skills
- **Authors:** Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, Chong Luo (Microsoft / SJTU / Tongji / Fudan)
- **Year / venue:** 2026, arXiv 2605.23904
- **URL:** https://arxiv.org/abs/2605.23904 · PDF: https://arxiv.org/pdf/2605.23904 · Code: https://github.com/microsoft/SkillOpt

**What they actually do.**  
Treats one Markdown skill document as the trainable external state of a *frozen* target. Loop: rollout a **training-split batch** → optimizer model proposes bounded add/delete/replace edits → **held-out selection split** accepts the candidate iff it strictly improves validation score → rejected edits go to a buffer; epoch-wise slow/meta update. This is the closest prior on *criterion 3’s held-out gate*, but: (i) batches are sequential/mined from a fixed `D_tr`, not a learned acquisition policy; (ii) there is exactly one assimilator (edit the skill doc). ALFWorld is a first-class benchmark under **direct chat** (`configs/alfworld/default.yaml`).

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Quote: “At each optimization step, the target model runs a rollout batch from D_tr with the current skill.” Acquisition is the dataset iterator, not π_acq. |
| 2 | **NO** | Quote: “a separate optimizer model turns scored rollouts into bounded add/delete/replace edits on a **single skill document**.” |
| 3 | **PARTIAL** | Strongest prior on held-out gating. Quote: “an edit is accepted only when it strictly improves a held-out validation score”; `s*_sel = argmax_{s ∈ C(D_tr)} mean_{D_sel} r(s)`. This is held-out capability of the *skill artifact*, not value of a specific experience under a choice among currently available assimilators. |
| 4 | **NO** | Skill state does not change a learned “what to experience next”; the next batch is the next training items. |

**Preempts:** trajectory-driven, validation-gated skill editing; ALFWorld skill optimization.  
**Remaining delta:** SkillOpt = *train one skill from a fixed split*. ExPolicy = *choose real experience and which persistent operator consumes it*, with future-J as the learning signal for both policies.

**ALFWorld reproduction (official).** See §6.2.

---

### 2.6 Mem2Evolve

- **Title:** Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation
- **Authors:** Zihao Cheng, Zeming Liu, Yingyu Shan, Xinyi Wang, Xiangrong Zhu, Yunpu Ma, Hongru Wang, Yuhang Guo, Wei Lin, Yunhong Wang
- **Year / venue:** 2026, ACL long (`2026.acl-long.952`); arXiv 2604.10923
- **URL:** https://aclanthology.org/2026.acl-long.952/ · arXiv PDF: https://arxiv.org/pdf/2604.10923  
- **Fetch note:** ACL PDF timed out; full method was read from arXiv 2604.10923 (same title/abstract as ACL).

**What they actually do.**  
Dual memory: **Asset Memory** (expert agents + MCP tools) and **Experience Memory** (distilled success/failure lessons). Forward pass is heuristic **reuse-first, create-on-demand**: if `sim(subtask, M_A) ≥ δ` reuse, else `Create(s_i | M_E, Web)`. After the task, an LLM-as-Judge scores the trajectory; new assets are validated/self-corrected into Asset Memory; Reflection writes experience items. Tasks are benchmark instances (including ALFWorld). There is **no learned policy** that, for one acquired env batch, chooses discard vs episodic vs skill. Asset vs experience updates are **both** executed by the pipeline, not alternative operators.

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Benchmark task `q_t` is given. The agent does not choose the next real env experience. |
| 2 | **PARTIAL** | Two persistent stores and create-vs-reuse, but the branch is similarity-thresholded recruitment, not a learned assimilator over the *same* experience. Quote: “Γ(s_i) = m* if sim ≥ δ else Create(s_i \| M_E, Web).” Backward pass *always* tries to keep assets and distill experience. |
| 3 | **NO** | Judge produces `r_t ∈ {0,1}` on the *current* trajectory (assuming no GT). Quote: “We assume ground-truth labels are inaccessible during backward evolution.” Not held-out future capability. |
| 4 | **PARTIAL** | Forward–backward loop: new assets/experience change later recruitment. Not a learned joint (acq, assim) policy. |

**Preempts:** experience-guided capability-asset co-evolution / distillation.  
**Remaining delta:** dual memory **architecture** ≠ learned policy over (what to experience, which operator to apply) under future-J.

---

### 2.7 MAGE — closest conceptual collision

- **Title:** MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs
- **Authors:** Ruiyi Yang, Zechen Li, Hao Xue, Imran Razzak, Flora D. Salim
- **Year / venue:** 2026, arXiv 2605.10064
- **URL:** https://arxiv.org/abs/2605.10064 · PDF: https://arxiv.org/pdf/2605.10064

**What they actually do.**  
Externalizes learning into EvoKG (capability / task / experience / environment subgraphs). Frozen execution backbone answers via retrieved success/failure memories. Across iterations, **three state variables co-evolve from the same per-question correctness reward:** the graph, a **task-level search bandit** (retrieval strategies), and a **skill-level routing bandit** (inference-time strategies). Navigator uses recency-weighted round-robin over *task types* (`s(t) = n_fail(t) + λ(k − k_last(t))`). EVOLVE **rotates** four graph-growth actions: principle extraction, dynamic-tool authoring, prompt refinement, skill splitting. Teacher (guidance tier) writes failures/principles/tools; learner harvests success traces. Evolution pool ≠ held-out test: after evolution, graph and bandits are **frozen** for reporting.

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **PARTIAL** | Intro literally says agents “must decide what to learn next and how to preserve what they have learned.” In the method, “what to learn next” is **curriculum over existing task types / retrieval recipes**, not choosing the next *real environment interaction batch*. Explorer is used “sequential environments only” to generate trajectories for graph writes (Crafter, WebShop), still not a learned π_acq over real-env batches. Quote: “a task-level search bandit learns retrieval strategies.” |
| 2 | **PARTIAL** | Multiple write modes exist (principles, failure memories, success memories, tools, prompt templates, skill split). They are **role-split and rotated**, not a policy that maps *one acquired experience* to {discard, episodic, skill, …}. Quote: “The EVOLVE step rotates through four action types… thus graph growth is not restricted to a single update mode.” |
| 3 | **NO** | Bandits and graph writes use **per-question correctness** on the evolution pool. Held-out eval is post-hoc frozen reporting, not the definition of an experience’s value after a chosen assimilator. Delta-guard rolls back mutable slots if accuracy drops, which is a safety check on graph edits, not future-J for (experience, operator). |
| 4 | **YES** | Quote: “the graph, a task-level search bandit, and a skill-level routing bandit are updated from the same per-question reward stream”; “memory growth, retrieval, and policy adaptation are coupled through one reward stream.” |

**Preempts:** the *broad slogan* “task-side policy + skill-side routing policy co-evolve from one feedback stream”; graph-structured self-evolution; frozen-backbone dual bandits.  
**Remaining delta (why not HARD_STOP):** MAGE’s “acquisition” is retrieval/curriculum over questions already in a pool; its “assimilation” is teacher/rotation graph writes, not a learned choice among heterogeneous persistent operators for the same real-env experience; its reward is current-question correctness, not held-out future capability after that experience is consumed.

This is the **highest-risk reviewer citation**. Positioning must explicitly contrast *retrieval/curriculum bandits* vs *real-env experience policy*, and *rotated graph writers* vs *assimilator choice on one experience*.

---

### 2.8 SESA

- **Title:** Self-Play Meets Skill Evolution: Self-Evolving Search Agents that Pose, Solve, and Remember
- **Authors:** Zenghuang Fu, Zhaoyang Li, Qiuyuan Ai, Haoyu Wu, Minghui Wu, Chenxu Zhao, Ante Wang, Guannan He, Changwei Wang
- **Year / venue:** 2026, arXiv 2607.29468
- **URL:** https://arxiv.org/abs/2607.29468 · PDF: https://arxiv.org/pdf/2607.29468

**What they actually do.**  
Zero-data search self-play: challenger poses a question with a verifiable target; solver retrieves from a skill bank and searches. Informative **frontier failures** are distilled into skills and written back. Updated memory changes solver success, which changes the challenger’s difficulty-shaped reward, which changes the future problem distribution. Skills are a **single** persistent substrate. Problems are **generated**, not real environment experience. Reward is answer match / bell-shaped frontier `r_p(x)`, not held-out future capability of consuming a specific experience.

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Explicitly generated self-play. Quote: “A challenger poses problems, while a separately parameterized solver alone retrieves skills.” This is the case R8 criterion 1 *excludes*. |
| 2 | **NO** | One assimilator: failure distillation into skill memory. Quote: “Informative failures are distilled into reusable skills and written back to memory.” |
| 3 | **NO** | Solver reward is `1{Judge(â, a*)=1}`; proposer reward is frontier-shaped success rate. Not held-out future J after assimilation. |
| 4 | **YES** | Quote: “The updated memory changes solver behavior and success, which changes the challenger’s reward and the distribution of future problems… task generation and skill memory co-evolve.” |

**Preempts:** task-distribution × skill-memory bidirectional loop (self-play).  
**Remaining delta:** generated problems ≠ real env experience; no heterogeneous operators; no assimilator-conditioned held-out future value.

---

### 2.9 ProactAgent

- **Title:** Ask Only When Needed: Proactive Retrieval from Memory and Skills for Experience-Driven Lifelong Agents
- **Authors:** Yuxuan Cai, Wei Li, Jie Zhou, Qin Chen, Xin Li, Bo Zhang, Liang He
- **Year / venue:** 2026, arXiv 2604.20572
- **URL:** https://arxiv.org/abs/2604.20572 · PDF: https://arxiv.org/pdf/2604.20572

**What they actually do.**  
Lifelong stream of interactive tasks (SciWorld, **ALFWorld**, StuLife). ExpOnEvo: act with retrieval → distill trajectories into typed stores (factual, episodic, success/failure/comparative skills) → GRPO on the policy. ProactRL: retrieval is an explicit action; paired-branch rollouts (retrieve vs no-retrieve from the same prefix) give a process reward only if retrieval improves outcome/efficiency. Distillation into the five stores is **pipeline-heuristic**, not a learned choice of which persistent operator consumes a given experience. Task instances arrive from the benchmark stream.

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Quote: “given a stream of tasks, the agent (1) interacts with the environment under the current policy.” The learned extra action is `Retrieve(q_t)`, not “which env batch to collect next.” |
| 2 | **PARTIAL** | Heterogeneous *stores* exist (`M^f, M^e, S+, S−, SΔ`), but writes are typed distillation after every episode, not a policy over operators for one experience. Quote: “After each episode, the evolution framework distills completed trajectories into new entries.” |
| 3 | **PARTIAL** | Paired-branch `Δ_i` is a counterfactual of *this trajectory’s* env return/length, not held-out future capability. Priority `p(r)` updates when retrieved entries are associated with improved outcomes — still current-task. |
| 4 | **PARTIAL** | Memory and policy co-evolve. Quote: “unifying experience accumulation and policy optimization into a single iterative cycle.” The missing piece is learned *acquisition of env experience* coupled to *assimilator choice*. |

**Preempts:** proactive when/what-to-retrieve over memory+skill experience; ALFWorld lifelong retrieval policy.  
**Remaining delta:** retrieval policy ≠ experience-acquisition policy; typed writes ≠ learned heterogeneous assimilators; process reward ≠ held-out future J.

---

### 2.10 MemEvolve

- **Title:** MemEvolve: Meta-Evolution of Agent Memory Systems
- **Authors:** Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, Shuicheng Yan
- **Year / venue:** 2025/2026, arXiv 2512.18746 (GitHub bib: 2025 preprint)
- **URL:** https://arxiv.org/abs/2512.18746 · PDF: https://arxiv.org/pdf/2512.18746 · Code: https://github.com/bingreeky/MemEvolve

**What they actually do.**  
Bilevel: **inner loop** populates experience under a *fixed* memory architecture; **outer loop** evolves the programmatic encode/store/retrieve/manage modules using inner-loop performance as fitness. EvolveLab reimplements 12 memory systems in that design space. The object of learning is *how the memory architecture works*, not which real env batch to collect or which assimilator to apply to it.

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Inner loop “adapts to a continuous stream of new tasks”; tasks are given. |
| 2 | **NO** | Outer loop searches architectures, not per-experience operator choice among currently available assimilators. |
| 3 | **PARTIAL** | Outer-loop fitness is downstream agent performance under the candidate architecture — related to future capability, but of a *memory program*, not of one consumed experience. |
| 4 | **PARTIAL** | Dual evolution of experience and architecture; not acq+assim policies in ExPolicy’s sense. |

**Preempts:** meta-evolution of memory architecture / “how to learn from experience.”  
**Remaining delta:** architecture search ≠ (π_acq, π_assim) over real env batches.

---

### 2.11 MemSkill

- **Title:** MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents
- **Authors:** Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, Wenya Wang
- **Year / venue:** 2026, arXiv 2602.02474
- **URL:** https://arxiv.org/abs/2602.02474 · PDF: https://arxiv.org/pdf/2602.02474

**What they actually do.**  
Memory construction is skill-conditioned: a controller selects Top-K skills from a bank (initialized as Insert/Update/Delete/Skip); an executor writes memories in one pass; a designer evolves the skill bank from hard cases. Controller RL uses **downstream memory-dependent training queries** as reward. Traces (dialogues / ALFWorld histories) are *given spans* to process. This is the closest prior on *heterogeneous memory ops as a learned selector*, but the “experience” is already-collected interaction text, not actively acquired real-env batches, and ops stay inside one memory-construction substrate.

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Quote: “Given an interaction trace, MemSkill processes it span by span.” No π over next env batch. |
| 2 | **PARTIAL** | Learned choice among Insert/Update/Delete/Skip (and evolved skills). These are memory-construction primitives, not episodic vs procedural vs discard assimilators of a new env experience that then changes the agent’s future *behavior substrate*. |
| 3 | **PARTIAL** | Quote: “the constructed memory is then evaluated on memory-dependent training queries to provide task reward.” Downstream of memory construction, but not held-out future capability after choosing among currently available *agent* assimilators. |
| 4 | **NO** | Skill-bank evolution does not change what environment to experience next. |

**Preempts:** evolving memory skills; RL skill selection for extract/consolidate/prune.  
**Remaining delta:** memory-skill controller on given traces ≠ joint env-acquisition and heterogeneous persistent-intervention policy.

---

### 2.12 Meta Context Engineering (MCE)

- **Title:** Meta Context Engineering via Agentic Skill Evolution
- **Authors:** Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, Guojie Song
- **Year / venue:** 2026, arXiv 2601.21557
- **URL:** https://arxiv.org/abs/2601.21557 · PDF: https://arxiv.org/pdf/2601.21557

**What they actually do.**  
Bi-level CE: meta-agent evolves **context-engineering skills** (how to represent/optimize context); base-agent executes those skills to write context as files/code. Outer objective is validation performance of the resulting context function. Domains are finance/chemistry/medicine/law/AI safety — not an embodied experience-acquisition loop.

**Coverage of 1–4**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **NO** | Optimizes context functions over a query set `X`, not next real-env batches. |
| 2 | **NO** | Skills specify CE procedures, not heterogeneous persistent assimilators of env experience. |
| 3 | **PARTIAL** | Quote: `s* = arg max_s J_val(c_s*)` subject to inner `c_s* = arg max J_train(c_s; s)`. Held-out val of context, not experience-after-assimilator. |
| 4 | **NO** | No env-acquisition policy. |

**Preempts:** meta-evolution of context-engineering / learning-to-learn skills.  
**Remaining delta:** CE skill evolution is orthogonal to ExPolicy’s env-experience × assimilator loop.

---

## 3. Additional 2025–2026 papers from search

Queries used (as specified):  
`"experience policy" acquisition assimilation persistent update agent 2026`; `"what to experience next" "how to learn from" agent skill memory`; `MAGE self-evolving agent dual bandit skill routing`; `heterogeneous memory skill update operator policy`; `ALFWorld MemRL SkillOpt joint policy`.

### 3.1 Evolving-RL (Fan et al., 2026) — closest on criterion 3 among extras

- **Title:** Evolving-RL: End-to-End Optimization of Experience-Driven Self-Evolving Capability within Agents
- **URL:** https://arxiv.org/abs/2605.10663 · https://github.com/Fanzy27/Evolving-RL
- **Method:** Shared policy as extractor + solver. Source task is solved, N candidate **textual skills** are extracted, then evaluated on K retrieved **downstream tasks**; aggregated downstream reward trains the extractor, skill-conditioned trajectories train the solver.
- **1 NO** — source `x^src ~ D`, not a learned env-acquisition policy.  
- **2 NO** — single assimilator: distill a textual skill.  
- **3 PARTIAL** — quote: “We evaluate the quality and generalization of a skill by its ability to improve solver performance on related tasks.” This is downstream transfer, but of a *skill string*, not of an experience consumed by a chosen member of a heterogeneous operator set, and the downstream set is similarity-retrieved from `D`, not a held-out capability probe after persistent state mutation.  
- **4 PARTIAL** — extractor and solver co-evolve; updated skills change later utilization, not what real env to acquire next.
- **Preempts:** jointly train extract+use; ALFWorld OOD skill transfer (paper: unseen ALFWorld 44.6% → 88.6% with skills).
- **Delta:** still one substrate, no π_acq over real env batches.

### 3.2 JERP (Ye & Yu, 2026)

- **Title:** Joint Learning of Experiential Rules and Policies for LLM Agents — https://arxiv.org/abs/2606.27136
- **Method:** Same ALFWorld/WebShop trajectories update **both** a natural-language rule pool and policy parameters (GRPO).
- **1 NO, 2 PARTIAL** (always both updates, not a choice), **3 NO, 4 PARTIAL**.
- **Preempts:** joint rule-pool + policy update from one trajectory. Not operator *selection*.

### 3.3 SEARL (ACL 2026 long 1125)

- **Title:** SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents
- **URL:** https://aclanthology.org/2026.acl-long.1125.pdf
- **Method:** Dynamic Tool Graph memory (create/reuse MCP tools) jointly trained with a tool-memory-aware policy; rewards = outcome + planning/creation/execution process terms.
- **1 NO, 2 NO** (tool-graph is the substrate), **3 NO, 4 PARTIAL**.
- **Preempts:** joint policy + tool-graph memory evolution.

### 3.4 MSCE (Tang et al., 2026)

- **Title:** From Memory to Skills: Evidence-Grounded Co-Evolution Governance for Long-Horizon LLM Agents — https://arxiv.org/abs/2607.16621
- **Method:** **Training-free** L1 traces → L2 policies → skills if positive estimated gain + evidence; reflection-weighted value backfill.
- **1 NO, 2 PARTIAL** (governed promotion, not a learned policy), **3 PARTIAL** (estimated gain), **4 NO**.
- **Preempts:** heuristic memory→skill co-evolution with estimated gain. Not a learned joint policy.

### 3.5 MemCon (Jiang et al., 2026)

- **Title:** Memory as a Controlled Process — https://arxiv.org/abs/2607.13591
- **Method:** Contextual bandit over Retrieve / PlanInject / Re-Retrieve / Consolidate / Forget / NoOp wrapping *any* memory backend; binary task success; **ALFWorld** included.
- **1 NO, 2 PARTIAL** (memory-access ops), **3 NO** (current-task binary), **4 NO**.
- **Preempts:** learned memory-operation controller (bandit). Still not env-acquisition × assimilator.

### 3.6 Memento-Skills (2026)

- **URL:** https://arxiv.org/abs/2603.18743
- **Method:** Read: skill router; Write: expand skill library from experience. Frozen LLM.
- **1 NO, 2 PARTIAL, 3 NO, 4 PARTIAL**.
- **Preempts:** skill-as-memory read/write loop.

### 3.7 ERSkill (2026)

- **URL:** https://arxiv.org/abs/2608.12720
- **Method:** Evolving *retrieval skills* (compose BM25/dense/rewrite) with router co-evolution.
- **1 NO, 2 NO, 3 NO, 4 PARTIAL**.
- **Preempts:** evolving retrieval programs, not env-experience assimilators.

### 3.8 Rethinking Continual Experience Internalization (2026)

- **URL:** https://arxiv.org/abs/2606.04703
- **Method:** Empirical recipe for multi-iteration internalization (principle-level experience, step-wise injection, off-policy distillation). Diagnoses collapse; does not learn π_acq or heterogeneous operators.

---

## 4. Intersection coverage matrix

| Paper | 1 Real-env acq | 2 Heterogeneous assimilators | 3 Held-out future-J | 4 Joint stateful loop | All four |
|---|---|---|---|---|---|
| AgeMem | NO | PARTIAL | NO | NO | no |
| MemoPilot | NO | NO | PARTIAL | NO | no |
| HiMPO | NO | NO | PARTIAL | NO | no |
| MemRL | NO | NO | NO | PARTIAL | no |
| SkillOpt | NO | NO | PARTIAL | NO | no |
| Mem2Evolve | NO | PARTIAL | NO | PARTIAL | no |
| **MAGE** | **PARTIAL** | **PARTIAL** | **NO** | **YES** | **no** |
| SESA | NO | NO | NO | YES | no |
| ProactAgent | NO | PARTIAL | PARTIAL | PARTIAL | no |
| MemEvolve | NO | NO | PARTIAL | PARTIAL | no |
| MemSkill | NO | PARTIAL | PARTIAL | NO | no |
| MCE | NO | NO | PARTIAL | NO | no |
| Evolving-RL | NO | NO | PARTIAL | PARTIAL | no |
| JERP | NO | PARTIAL | NO | PARTIAL | no |
| SEARL | NO | NO | NO | PARTIAL | no |
| MSCE | NO | PARTIAL | PARTIAL | NO | no |
| MemCon | NO | PARTIAL | NO | NO | no |
| Memento-Skills | NO | PARTIAL | NO | PARTIAL | no |
| ERSkill | NO | NO | NO | PARTIAL | no |

**No paper scores YES on all four.** Closest combinations:

- MAGE: 4 YES, 1–2 PARTIAL, 3 NO  
- SESA: 4 YES, 1–3 NO (self-play / single skill bank / current-task reward)  
- Evolving-RL / SkillOpt: strongest **downstream/held-out** flavor, but single skill substrate and no π_acq  
- AgeMem / MemSkill / MemCon: strongest **operator-set** flavor, but operators are memory ops on given traces  
- Mem2Evolve / MSCE / ProactAgent: dual memory+skill *stores*, heuristic or retrieval-centric, not learned assimilator choice

---

## 5. What is already preempted vs what remains

**Do not claim novelty for:**

| Claim | Already done by |
|---|---|
| Learned memory operations (store/update/discard/retrieve) | AgeMem, MemSkill, MemCon |
| RL over memory updates | MemoPilot, HiMPO, AgeMem |
| Utility of retrieved episodes from env feedback | MemRL |
| Validation-gated skill document editing | SkillOpt |
| Experience-guided tool/agent creation | Mem2Evolve, SEARL |
| Dual bandits (task/search + skill routing) from one reward | **MAGE** |
| Self-play curriculum co-evolving with skill memory | SESA |
| Proactive retrieval over memory+skills | ProactAgent |
| Meta-evolution of memory/CE architecture | MemEvolve, MCE |
| Extractor–solver co-training with downstream skill eval | Evolving-RL |
| Joint rules + policy from one trajectory | JERP |
| Training-free memory→skill promotion | MSCE |

**Still unoccupied (the only allowable R8 novelty sentence):**

> Learning a policy over both **what real environment experience to collect next** and **which heterogeneous persistent learning operator should consume that experience**, with value defined as **realized held-out future capability after the chosen assimilator actually mutates persistent state**, in one stateful loop where the new state changes the next acquisition.

---

## 6. ALFWorld backend notes (for later reproduction)

### 6.1 MemRL — official ALFWorld

- **Code:** https://github.com/MemTensor/MemRL  
- **Env:** ALFWorld `AlfredTWEnv`; must install data from https://github.com/alfworld/alfworld  
- **Provided files:** `configs/envs/alfworld.yaml`, `data/alfworld/alfworld_examples.json`  
- **Run:**

```bash
conda create -n memoryrl python=3.10 -y && conda activate memoryrl
pip install -r requirements.txt
# set llm.api_key and embedding.api_key in configs/rl_alf_config.yaml
python run/run_alfworld.py --config configs/rl_alf_config.yaml
```

- **Config highlights:** `environment.alfworld_env_type: AlfredTWEnv`; `experiment.num_sections: 10`, `batch_size: 32`; `rl_config.alpha: 0.3`, `gamma: 0.0`, `tau/sim_threshold: 0.62`, `weight_sim/weight_q: 0.5`.  
- **What it is:** frozen LLM + Q-valued episodic memory. **Not** an ExPolicy joint controller.

### 6.2 SkillOpt — official ALFWorld

- **Code:** https://github.com/microsoft/SkillOpt  
- **Docs:** https://microsoft.github.io/SkillOpt/docs/guideline.html  
- **Install extra:** `python -m pip install -e ".[alfworld]"`  
- **Config:** `configs/alfworld/default.yaml` (`env.name: alfworld`, `max_steps: 50`, splits under `data/alfworld_path_split`, seed skill `skillopt/envs/alfworld/skills/initial.md`)  
- **Run (source checkout, after loading `.env`):**

```bash
git clone https://github.com/microsoft/SkillOpt.git && cd SkillOpt
python -m pip install -e ".[alfworld]"
python scripts/train.py --config configs/alfworld/default.yaml
```

- **Harness limit:** paper Table 1 leaves ALFWorld blank for Codex / Claude Code because it “requires persistent embodied-environment interaction.” Direct-chat is the official path.  
- **What it is:** one skill document, held-out gate. **Not** heterogeneous assimilators or learned env acquisition.

ExPolicy R8, if it uses ALFWorld, should treat MemRL and SkillOpt as **frozen-substrate baselines** (episodic Q-memory vs gated skill editing), not as prior joint policies.

---

## 7. Reviewer-risk ranking (not hard stops)

1. **MAGE** — will be cited as “they already have task bandit + skill bandit in one loop.” Rebuttal must be method-level (retrieval vs real env; rotation vs per-experience operator; correctness vs held-out future-J).
2. **SESA** — “task distribution and skill memory co-evolve.” Rebuttal: generated self-play, single skill writer.
3. **Evolving-RL + SkillOpt** — “they already score updates by downstream/held-out performance.” Rebuttal: single skill substrate, no π_acq, SkillOpt’s `D_tr` iterator.
4. **Mem2Evolve / MSCE / ProactAgent** — dual memory+skill *systems*. Rebuttal: heuristic/retrieval, not learned assimilator choice on one experience.
5. **AgeMem / MemSkill / MemCon** — operator policies. Rebuttal: memory ops on given traces, not env-experience acquisition.

---

## 8. Verdict

```text
PROCEED
```

**Justification.** After full-text reading of all 12 must-read papers and the additional 2026 near-neighbors surfaced by the specified searches, **no prior work implements all four intersection conditions**. Several papers occupy one or two axes (MAGE occupies the joint-loop slogan most aggressively; SkillOpt/Evolving-RL occupy held-out/downstream scoring; AgeMem/MemSkill occupy operator sets; SESA occupies co-evolving task distribution and skills). The conjunction — *real-env experience acquisition × heterogeneous persistent assimilators on the same experience × value = held-out future capability after actual consumption × joint stateful update that changes next acquisition* — is still empty.

R8 may continue **only** if the method and writing stay inside that conjunction. Expanding the claim to “self-evolving agents,” “memory policy,” “skill routing,” or “dual bandits” would collide with MAGE/SESA/AgeMem and should be treated as a *positioning failure*, not as remaining novelty.

**Recommended claim sentence (use this, not a broader one):**

> We learn a joint policy that decides (i) which real environment experience to acquire next and (ii) which of several heterogeneous persistent update operators should consume it, using as reward the held-out future capability change realized after that operator mutates the agent’s persistent state.

---

## 9. Audit limitations

- ACL Mem2Evolve PDF timed out; arXiv 2604.10923 was read in full (matching ACL abstract/title).
- Codex MCP cross-model review was unavailable here.
- New arXiv uploads after 2026-08-21 are not covered.
- “Generated self-play problem” (SESA) and “retrieval item” (MemRL/MAGE/ProactAgent) were scored NO on criterion 1 by the frozen R8 definition, even when those systems *do* choose *something* next. If R8 later relaxes criterion 1 to include self-play or retrieval, **SESA and MAGE would need a re-audit**; under the current wording they still fail 2 and/or 3.
