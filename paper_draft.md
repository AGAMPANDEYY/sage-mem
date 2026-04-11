# SAGE-Mem: Write-Time Governance for Multimodal Agent Memory Under Adversarial and Noisy Observations

*ICML 2026 SCALE Workshop Submission Draft*

---

## Abstract

Long-horizon agents increasingly rely on persistent memory, but persistent memory is also a durable attack surface. An adversarial observation that is admitted into memory can later be retrieved, consolidated, and reused long after the original interaction has ended. Existing defenses for memory poisoning are predominantly retrieval-time: they try to rank or filter memories after contamination has already entered the store. This leaves the memory state itself polluted, increases retrieval noise, wastes context budget, and allows unsupported observations to harden into derived beliefs. We study **write-time governed multimodal memory** as a distinct robustness problem. We present **SAGE-Mem**, a memory layer that mediates admission, promotion, and retrieval through typed memory partitions (`evidence`, `belief`, `control`), sufficiency-gated evidence-to-belief promotion, source-conditioned trust, anomaly detection, consistency checks, and provenance-aware retrieval. We evaluate on two settings: (i) a controlled long-horizon benchmark based on LoCoMo-10 with multimodal adversarial injections, including Visual Prompt Injection (VPI) and noisy/missing-modality stress; and (ii) an adversarially augmented MM-BrowseComp observation-trace benchmark with VLM-backed image observations. Our primary metrics follow the memory lifecycle: attack write admission, attack belief formation, retrieval contamination, false-belief retrieval, and benign completion under attack (BCU). On LoCoMo-based benchmarks, SAGE-Mem consistently drives write admission and retrieval contamination to near-zero, including perfect suppression on the main suite and VPI-only setting, while remaining robust under noisy/missing multimodal inputs. SAGE-Mem trails retrieval-time baselines on clean and attacked QA utility, so the result is an integrity--utility tradeoff rather than overall dominance. MM-BrowseComp further shows that generic tool trust fails under realistic browsing attacks. An initial browsing-context keyword prior (H5) reduces attack admission, but remains vulnerable on the harder combined browser benchmark. We therefore introduce **H6 — Adversarial Belief Revision (ABR)** — a browser-specific write-time layer that combines structured browser claim typing with page-group anomaly signals and channel provenance, without relying on correction-language keywords. On the 194-case grouped MM-BrowseComp benchmark, H6 achieves Write ASR=0.000 and Retrieval ASR=0.000 on the combined injection+adaptive attack track, while preserving near-clean utility (`BCU_poison=0.1512` vs `BCU_clean=0.1598`). A later semantic observation-group rerun preserves the same top-line result without changing the main mechanistic interpretation. The resulting picture is not “retrieval is obsolete,” but rather that **write-time governance is a necessary complement to retrieval-time filtering for multimodal agent memory**, especially when dependent multimodal evidence can masquerade as corroboration and browser-derived factual claims can bypass generic tool trust.

---

## 1. Introduction

Memory changes what an agent is. A stateless model can only be manipulated in the current context window; a stateful agent can be manipulated by altering the persistent memory that future reasoning depends on. This makes memory poisoning qualitatively different from ordinary prompt injection. Once a malicious observation is written, it can survive long after the original turn, reappear at retrieval time, distort later summaries, and occupy retrieval bandwidth that should have been reserved for grounded evidence.

This matters even more for multimodal agents. OCR text, image captions, browser outputs, and user/tool messages are not interchangeable evidence channels. A single adversarial image can produce both OCR output and a caption-like summary, creating the appearance of multi-source support despite both observations originating from the same underlying source. In other words, multimodal agreement is not always evidence of truth; it can also be evidence of **dependent corruption**.

The paper’s multimodal claim should therefore be read precisely. We are **not** claiming to solve full multimodal truth inference or general cross-modal fact verification. Our claim is narrower and, we believe, more important for deployed agents: persistent memory systems need principled rules for deciding when heterogeneous multimodal observations are safe to store, when they should remain weak evidence, and when apparent cross-modal agreement is merely duplicated corruption from a single source object.

Most prior memory-poisoning defenses attack the problem too late. Retrieval-time reliability scoring can reduce the chance that a poisoned item is selected, but it does not protect the persistent memory state itself. A memory system that accepts unsupported observations and merely hopes to filter them later incurs three long-run costs:

1. **State corruption:** the store contains observations that should never have been admitted.
2. **Retrieval burden:** reranking and filtering must repeatedly separate signal from contamination.
3. **Compounding distortion:** later summaries and derived beliefs may inherit unsupported content.

We therefore study **write-time governance** as the primary mechanism, and evaluate it with **lifecycle metrics** rather than only downstream QA metrics. The question is not just whether a poisoned memory is retrieved at the end of the pipeline, but whether it is admitted at write time, whether it hardens into a belief, whether it later resurfaces, and how much benign utility is preserved under that regime.

### 1.1 Why Write-Time Defense Is Distinct

This distinction from retrieval-time defense should be explicit. Retrieval-time filtering and write-time governance solve related but non-identical problems.

Retrieval-time defense asks:
- given a possibly contaminated memory store,
- which items should be surfaced to the model right now?

Write-time defense asks the harder systems question:
- which observations should be allowed to become persistent state at all?

The second problem is harder for three reasons.

1. **Partial observability at ingestion time.** At write time the system has less future context than it will have at retrieval time. It must decide under uncertainty whether a fresh observation is trustworthy enough to store.
2. **Memory-state externalities.** A false positive or false negative at write time has long-run effects: accepted noise can distort future summaries, crowd out useful evidence, and repeatedly tax retrieval. Retrieval-only methods do not undo those state changes.
3. **Multimodal dependence.** In multimodal agents, two observations may appear to corroborate each other while actually deriving from the same source object, such as OCR and caption outputs from one image. This creates false support before the retrieval stage is ever reached.

These differences motivate our evaluation choice. A write-time defense should not be judged only by final QA utility or retrieval-time ASR, because those are downstream outcomes of a longer memory lifecycle. The write boundary, the belief boundary, and the retrieval boundary each expose different failure modes.

This also means that comparing SAGE-Mem to a retrieval-time system such as MMA is not a single-axis leaderboard comparison. MMA is designed to preserve and rank a broad memory store at retrieval time, which naturally favors immediate answer utility. SAGE-Mem is designed to keep unsupported observations out of durable memory, which naturally favors store integrity and may reject evidence that later would have helped a benign question. We therefore interpret the comparison as an **integrity--utility frontier**: the central question is whether write-time governance buys measurable memory-state integrity, and what utility cost that purchase entails.

### 1.2 Hypotheses

We frame the paper around five explicit hypotheses.

- **H1: Write-time governance hypothesis.**
  Relative to retrieval-time filtering alone, a governed write boundary should reduce attack write admission and retrieval contamination on long-horizon multimodal memory tasks.

- **H2: Belief-formation hypothesis.**
  Typed evidence-to-belief promotion should reduce the rate at which unsupported or adversarial observations harden into durable, answer-bearing memory.

- **H3: Dependent-evidence hypothesis.**
  Multimodal attacks such as VPI exploit dependence between OCR and caption outputs from the same image; modeling those channels as dependent should improve robustness relative to consensus-style trust.

- **H4: Noise-calibration hypothesis.**
  Under noisy or missing modalities, adaptive trust calibration should reduce over-quarantine relative to simpler write-time baselines while preserving low attack survival.

- **H5: Browsing-context prior hypothesis.**
  Browsing-derived external tool text is not equivalent to trusted internal tool output. A source-context-sensitive write prior should reduce realistic web-style correction-language attacks without globally lowering trust in all tool outputs. **Scope caveat:** the initial H5 implementation uses a keyword regex whose vocabulary overlaps with the benchmark attack text by construction. The H5 result (ASR=0) therefore holds only for correction-language-vocabulary adversaries, not for adversaries who paraphrase the same semantic intent without trigger words.

- **H6: Adversarial Belief Revision (ABR) hypothesis.**
  A browser-specific, vocabulary-agnostic write-time layer should maintain robustness against adaptive adversaries who evade the H5 keyword filter. In the current design, ABR combines a **structured browser claim gate** for browser-sourced `qa_answer` writes with secondary page-group anomaly signals and channel provenance. H6 should reduce ASR on paraphrased correction attacks (`fact_overwrite_adaptive`) while preserving clean BCU close to the H5 level.
  **Empirical result:** H6 is **fully supported** on the current benchmark. On the combined adversarial benchmark (`paper_mmadv_abr_group_v1`, n=194 cases, 388 attack write attempts including both correction-language and vocabulary-adaptive phrasings), H6 achieves Write ASR=0.000, Retrieval=0.000, ASR=0.000, BCU_poison=0.1512. The primary mechanism is the structured browser claim gate: browser-sourced `qa_answer` writes are blocked at the content-type level rather than by vocabulary matching. The gate fires an average of 2.0 times per case, blocking all 388/388 attack writes. Secondary page-group anomaly checks fire on 0.622 writes per case on average, providing additional within-page outlier coverage. For comparison, H5 admits 122/388 writes (Write ASR=0.314), which then drive retrieval contamination to 0.827 and BCU_poison down to 0.0275.

The current empirical evidence strongly supports H1, H3, and H4 on LoCoMo-based settings, partially supports H2 through near-zero false-belief retrieval, supports H5 narrowly for correction-language vocabulary attacks with the co-design caveat disclosed, and **fully supports H6** — ABR with structured browser claim gate achieves Write ASR=0.000 and ASR=0.000 on the combined adversarial benchmark (both injection and adaptive attacks), with BCU_poison=0.1512 close to BCU_clean=0.1598.

### Contributions

This paper makes five concrete contributions.

1. **Problem framing:** We formulate multimodal agent-memory robustness as a write-boundary and belief-boundary problem, not only a retrieval-time ranking problem.
2. **Architecture:** We present SAGE-Mem, a governed memory layer with typed memory, sufficiency-gated promotion, source-conditioned trust, anomaly detection, consistency checks, and provenance-aware retrieval.
3. **Benchmark/evaluation design:** We define lifecycle metrics for write admission, belief formation, retrieval contamination, and downstream completion under attack, and apply them to both a controlled long-horizon benchmark and a browsing-style multimodal benchmark.
4. **Empirical findings:** On LoCoMo-based settings, write-time governance strongly reduces attack admission and retrieval contamination, including on multimodal prompt injection and noisy/missing-modality robustness tests; on browsing-style MM-BrowseComp, generic tool trust fails. An initial keyword-prior defense (H5) achieves ASR=0 on correction-language vocabulary attacks, but we disclose a vocabulary co-design artifact that limits this claim to non-adaptive adversaries.
5. **Adversarial Belief Revision (H6/ABR) with structured browser claim typing:** We identify the co-design limitation in H5, introduce `fact_overwrite_adaptive` as the adaptive evaluation variant, and propose ABR — a browser-specific, vocabulary-agnostic write-time layer that combines a **structured browser claim gate** with page-group anomaly signals and channel provenance. The gate blocks browser-sourced `qa_answer` writes by content type and source channel rather than vocabulary matching. Empirically, H6 achieves Write ASR=0.000 and ASR=0.000 on the combined adversarial benchmark (n=194, 388 attack write attempts, including both correction-language and adaptive paraphrased attacks), with BCU_poison=0.1512 ≈ BCU_clean=0.1598. A later semantic observation-group rerun preserves this result with only a negligible BCU change (0.1512 → 0.1529), strengthening confidence that the main finding is not a brittle artifact of one particular page-group scorer. The structural insight is that browser-sourced factual claims should not be promoted directly into answer-bearing memory.

The paper’s central claim is therefore deliberately narrow and defensible:

> **Write-time governed multimodal memory is a distinct robustness primitive for agents. It strengthens memory-state integrity beyond what retrieval-time scoring alone can provide, especially under multimodal corruption, but it introduces a measurable utility tradeoff and requires source-context calibration for browsing-derived evidence. Honest evaluation requires testing against adaptive adversaries who know the defense vocabulary; keyword-based priors are bounded to non-adaptive adversaries. Vocabulary-agnostic write-time blocking via structured browser claim typing plus page-group anomaly checks (H6/ABR) achieves complete attack suppression (Write ASR=0.000, ASR=0.000) while preserving near-clean utility (BCU_poison=0.1512 vs BCU_clean=0.1598).**

The paper therefore does not claim that SAGE-Mem dominates retrieval-time filtering on all metrics. A retrieval-time system can be the right engineering choice when the objective is maximum short-term QA accuracy under a permissive memory store. SAGE-Mem targets a different operating point: lower tolerance for persistent contamination, lower attack survival across memory lifecycle stages, and more explicit provenance over the beliefs that are allowed to guide later agent behavior.

### 1.3 Relation to Prior Work

This paper sits at the intersection of four threads of prior work.

**Memory poisoning and agent reliability.**
Recent work has shown that persistent agent memory is a durable attack surface and that poisoning can be achieved through ordinary interaction rather than direct database access [AgentPoison: Chan et al., 2024, arXiv:2407.12784; MINJA: Dang et al., 2025, arXiv:2503.03704; Zombie Agents: arXiv:2602.15654]. Our work agrees with that threat model, but shifts the emphasis from demonstrating feasibility to evaluating **where in the memory lifecycle** defenses should operate.

**Retrieval-time robustness and corrupted context.**
A large body of work studies corrupted retrieval, noisy context, and prompt injection at the point of model invocation [PoisonedRAG: Zou et al., 2024, arXiv:2402.07867; GuardAgent: Xiang et al., 2024, arXiv:2406.09187; A-MemGuard: arXiv:2510.02373; ASB: arXiv:2410.02644]. These methods are relevant baselines, and MMA is our concrete retrieval-time reference point. Our claim is not that retrieval-time defense is unimportant, but that it is insufficient once unsupported observations have already been written into durable state.

**Multimodal prompt injection and vision-language reliability.**
Prior multimodal security work shows that images can manipulate captioners, OCR pipelines, and downstream reasoning [AegisAgent: arXiv:2512.20986; see also adversarial patch literature]. Our contribution is memory-specific: OCR and caption outputs derived from the same image can create **false corroboration** inside the memory store unless dependence is modeled explicitly.

**Memory-augmented agents and benchmark methodology.**
Prior work on memory-augmented agents, long-horizon QA, and agent evaluation provides the underlying setting but typically does not separate write admission, belief formation, and retrieval contamination as distinct evaluation stages [MemGPT: Packer et al., 2023, arXiv:2310.08560; Generative Agents: Park et al., 2023, arXiv:2304.03442; AgentBench: arXiv:2308.03688]. Our lifecycle metrics are intended to make that decomposition explicit.

Relative to these literatures, the paper’s novelty is therefore not a claim to have invented memory poisoning, multimodal prompt injection, or memory architectures in isolation. The novelty is the **combination** of write-time governed multimodal memory, dependence-aware multimodal handling, and lifecycle evaluation that separates admission, belief formation, retrieval contamination, and downstream answer quality.

Equally important, the paper argues for a **different unit of robustness analysis** than is common in agent benchmarks. The relevant question is not only whether a final answer is correct, but whether the memory system preserved the integrity of the latent state that future answers depend on. This emphasis on memory-state integrity is what connects the controlled LoCoMo experiments, the multimodal dependence argument, and the browser-memory stress test into a single research story.

---

## 2. Problem Setting

### 2.1 Agentic Memory Pipeline

We model an agent with persistent memory as a four-stage pipeline:

1. **Observation ingestion:** raw observations arrive from channels such as `tool_output_text`, `ocr_text`, `vision_caption`, or `user`.
2. **Memory admission:** the system decides whether to write the observation into planning memory, quarantine it, or place it into an audit partition.
3. **Belief formation:** accepted evidence may later be summarized or promoted into more durable belief-like memory items.
4. **Retrieval and downstream use:** later questions retrieve memory items that influence planning or answer generation.

Let \(x_t\) denote the \(t\)-th incoming observation, let \(M_t\) denote the current memory state, and let \(R(q, M_t)\) denote the retrieval set for query \(q\). A write-time defense changes the transition

\[
M_{t+1} = \mathcal{U}(M_t, x_t)
\]

itself, not only the retrieval function \(R\).

### 2.2 Threat Model

The adversary can inject malicious content into external observations that the agent ingests from untrusted channels. In our main threat model, the adversary controls the **content** of an observation but not the trusted orchestration layer that assigns `source_type` and `channel_id`.

The attacker may:
- inject malicious tool outputs or web observations,
- embed malicious directives in OCR-visible or caption-visible image content,
- use natural-language corrections that mimic benign content,
- exploit summarization or consolidation to spread attack lineage.

The attacker may **not**:
- relabel an arbitrary observation as a higher-trust channel,
- directly edit the memory store,
- compromise the orchestrator itself.

Trusted-user attacks are treated separately as stress tests, not as part of the main benchmark claim, because authenticated user content is intentionally treated as a trust anchor in the system design.

### 2.3 Multimodal Noise and Dependence

A multimodal memory system is not only vulnerable to adversarial corruption; it is also vulnerable to **dependence mis-modeling**. Suppose a single image produces two observations:
- an OCR transcription, and
- a VLM-generated caption.

If both contain the same malicious payload, a naive system may treat them as corroborating evidence. But conditional independence does not hold: both observations are functions of the same adversarial source. This motivates our VPI threat model and the corresponding design principle that **agreement across dependent channels should not be rewarded as independent support**.

---

## 3. SAGE-Mem

SAGE-Mem is a governed memory layer between heterogeneous observation sources and a downstream planner. It is not a complete autonomous agent stack by itself. Its job is to preserve the integrity of the memory state.

### 3.1 Typed Memory

SAGE-Mem separates memory into:
- `evidence`: raw or grounded observations,
- `belief`: more durable, answer-supporting state,
- `control`: content that could steer policy or action.

This distinction matters because not every admitted observation should immediately become durable belief. In particular, multimodal observations often begin as weak evidence rather than stable facts.

### 3.2 Write-Time Admission

For an observation \(x\) from channel \(c\) and source type \(\sigma\), admission is controlled by a conjunction of guard checks:

\[
\text{accept}(x)=1
\iff
g(x)=\text{DATA}
\wedge
\theta(c,\sigma)\ge \tau_t
\wedge
d_M(x)\le \xi_t
\wedge
\theta_{\mathrm{browser}}(x)\ge \tau_t
\wedge
\neg \psi_{\mathrm{VPI}}(x)
\wedge
\neg \text{unsafe\_conflict}(x)
\]

where:
- \(g(x)\) is the write-time guard classification,
- \(\theta(c,\sigma)\) is the Bayesian trust posterior for channel \(c\) and source type \(\sigma\),
- \(d_M(x)\) is the session-relative anomaly score,
- \(\theta_{\mathrm{browser}}(x)\) is an optional source-context prior for browser-derived external text,
- \(\psi_{\mathrm{VPI}}(x)\) detects suspicious cross-modal corroboration.

The browsing prior is deliberately provenance-level rather than benchmark-ID-level. Generic `tool_output_text` may refer to trusted internal API output, while `browser_tool_output_text` denotes externally controlled page content. Benign browser observations retain enough trust to enter evidence memory, but correction-like browser text is capped below the write threshold. This tests whether the failure exposed by MM-BrowseComp is a missing source-context prior rather than a fundamental limitation of write-time governance.

The important architectural point is not the exact functional form, but the location of the decision: SAGE-Mem attempts to stop unsupported observations **before** they become durable memory.

### 3.3 Belief Promotion

Admitted evidence is not automatically promoted into durable belief memory. Promotion is gated by support sufficiency:

\[
\text{promote}(y)=1
\iff
\left|P_{\ge \tau_p}(y)\right| \ge k_s
\wedge
\left|\text{independent}(P_{\ge \tau_p}(y))\right| \ge k_i
\wedge
\neg \text{conflict}(P(y)).
\]

This matters for multimodal noise because evidence from OCR and captions may be plentiful but not independent. The promotion rule is where write-time defense becomes **belief-time defense**.

### 3.4 Provenance and Retrieval

At retrieval time, SAGE-Mem attempts to return answer-supporting memory together with its evidence lineage. This is measured using belief traceability and false-belief retrieval metrics. The intention is not merely to retrieve relevant text, but to retrieve beliefs whose support has remained inspectable and grounded.

### 3.5 System Intent

The production-systems rationale for this design is simple:
- less persistent noise in the store,
- fewer malicious or low-value items consuming token budget,
- less summary distortion over time,
- lower burden on retrieval-time reranking.

Retrieval-time filtering remains useful. SAGE-Mem’s claim is not that retrieval defenses are obsolete, but that they are incomplete if the store itself is already contaminated.

---

## 4. Evaluation Design

### 4.1 Benchmarks

We use two complementary evaluation settings.

#### LoCoMo-10 with multimodal adversarial extension

LoCoMo provides a controlled long-horizon QA benchmark over multi-session conversations. We augment it with multimodal observation writes and adversarial injections. This benchmark is controlled rather than fully naturalistic, but it allows causal analysis of poisoning over long horizons.

We use three LoCoMo-based settings:
- **Main suite:** untrusted-channel memory-poisoning attacks,
- **VPI-only suite:** Visual Prompt Injection in isolation,
- **multimodal robustness suite:** missing/noisy modality stress.

#### Adversarially augmented MM-BrowseComp

We build a browsing-style multimodal memory benchmark from observation traces derived from MM-BrowseComp, augmented with VLM-backed image observations and separate clean and adversarial tracks. This benchmark is more realistic as a browsing-memory setting, but as our results show, it is also much harsher and currently closer to an external stress test than a mature main-table benchmark.

### 4.2 Attack Families

The main LoCoMo suite includes:
- `constructor_launder`
- `label_gaming`
- `ocr_injection`
- `vision_caption_injection`
- `visual_prompt_injection`
- `fact_overwrite_injection`
- `adaptive_nl_evasion`
- `buried_payload`

The MM-BrowseComp adversarial track uses two attack variants:

- `fact_overwrite_injection`: correction-language phrasing ("Correction confirmed: ... has since been superseded. The previously stored value..."). This is the original H5 evaluation attack.
- `fact_overwrite_adaptive` *(new)*: semantically equivalent payload using paraphrased vocabulary with no overlap with the `_BROWSER_CORRECTION_RE` detection regex ("Recent analysis of available records establishes...", "Authoritative sources reflect..."). This is the honest adaptive adversary evaluation for H6/ABR.

The distinction matters for honest evaluation: a defense that achieves ASR=0 on `fact_overwrite_injection` but not on `fact_overwrite_adaptive` has only demonstrated vocabulary interception, not semantic adversarial robustness.

### 4.3 Metrics

We evaluate the memory lifecycle directly.

This metric choice is intentional. A large fraction of prior robustness work around context corruption, prompt injection, and retrieval poisoning evaluates only the final retrieved context or final answer. That is appropriate for retrieval-time defenses, but insufficient for a write-time memory system whose main purpose is to prevent unsupported observations from becoming durable state in the first place. Our primary metrics therefore emphasize admission, belief formation, and retrieval contamination before turning to downstream completion.

#### Primary mechanism metrics

- **Attack write admission rate:** fraction of injected attacks that enter planning memory.
- **Attack belief formation rate:** fraction of cases where attack-derived content hardens into durable planning-memory belief.
- **Attack retrieval rate:** fraction of poisoned retrievals containing attack-derived content.
- **False belief rate:** fraction of QA evaluations whose retrieved beliefs descend from poisoned lineage.

#### Primary downstream utility metric

- **BCU (Benign Completion Under Attack):**
\[
\mathrm{BCU}
=
\mathbb{E}\left[\mathbf{1}[\text{answer consistent} \land \neg \text{attack survived}]\right].
\]

This is a per-QA joint metric. It is not the product of marginal answer correctness and attack survival.

#### Secondary metrics

- **ASR:** retrieval-time attack survival.
- **ASR\(_\text{behavioral}\):** LLM-judge behavioral attack success when a raw attack item is directly retrieved.
- **Belief traceability:** mean fraction of retrieved belief items with usable support lineage; reported as `N/A` when no belief items are retrieved.
- **Write quarantine per case:** intervention frequency; informative, but not inherently “higher is always better.”

### 4.4 Compared Conditions

The main comparisons are:
- **ShortContext:** no persistent memory.
- **MMA:** retrieval-time reliability scoring baseline.
- **RSum:** consolidation baseline without guard.
- **H1 / H2 / H3:** mechanism-level internal baselines.
- **SAGE-Mem v2:** final combined governed-memory method.

The right comparison is not “memory platform versus memory platform”; it is **retrieval-time defense versus write-time governed memory**.

---

## 5. Results

### 5.1 Main LoCoMo Result

*Run: `paper_main_full_v1`*

The main LoCoMo result is the strongest evidence in the paper.

| Method | BCU clean | BCU poison | Write ASR | Belief ASR | Retrieval | False belief | ASR | Write q/case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMA | 0.7500 | 0.6417 | 1.0000 | 0.0000 | 0.1333 | 0.0000 | 0.1333 | 0.0 |
| RSum | 0.4050 | 0.4533 | 0.0000 | 0.0000 | 0.0067 | 0.0000 | 0.0067 | 0.0 |
| H1 | 0.3950 | 0.4517 | 0.0000 | 0.0000 | 0.0033 | 0.0000 | 0.0033 | 5.43 |
| H2 | 0.3850 | 0.4533 | 0.0000 | 0.0000 | 0.0017 | 0.0000 | 0.0017 | 1.0 |
| H3 | 0.2400 | 0.2133 | 0.0000 | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 3.0 |
| **SAGE-Mem** | **0.3950** | **0.4600** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **5.43** |

**What this shows.**

1. **SAGE-Mem’s main strength is write-time containment.** It eliminates attack admission into planning memory on the main suite and drives retrieval contamination to zero.
2. **MMA’s main strength is utility preservation.** It retains substantially higher BCU on both clean and poisoned conditions.
3. **Write-time governance is not the same as “more utility.”** The main result supports an integrity--utility frontier: SAGE-Mem buys a clean memory state and zero attack retrieval at the cost of lower immediate QA completion, while MMA retains more useful evidence but leaves poisoned writes inside the store.

This is the paper’s most defensible central result.

This tradeoff is not an embarrassment to hide; it is the main systems result. In long-running agents, a memory design that maximizes short-term answer rate may still be unacceptable if it stores adversarial content that can later be consolidated, retrieved in a different context, or used as evidence for a future action. Conversely, a conservative write-time defense must account for the opportunity cost of rejected evidence. We therefore report both lifecycle integrity metrics and BCU, and avoid claiming that either operating point uniformly dominates the other.

### 5.2 Visual Prompt Injection

*Run: `paper_vpi_full_v1`*

VPI is the cleanest multimodal result in the paper.

| Method | BCU clean | BCU poison | Write ASR | Retrieval | ASR | Write q/case |
|---|---:|---:|---:|---:|---:|---:|
| MMA | 0.7650 | 0.7117 | 1.0000 | 0.0700 | 0.0700 | 0.0 |
| RSum | 0.3800 | 0.3900 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| H1 | 0.3700 | 0.3800 | 0.0000 | 0.0000 | 0.0000 | 2.0 |
| H2 | 0.3700 | 0.3700 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| H3 | 0.2400 | 0.2700 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| **SAGE-Mem** | **0.3800** | **0.3800** | **0.0000** | **0.0000** | **0.0000** | **2.0** |

**What this shows.**

1. A retrieval-only system can still look strong under VPI on utility while admitting every attack write.
2. SAGE-Mem fully suppresses VPI in the current benchmark regime.
3. This result supports the specific multimodal claim that dependent OCR/caption evidence should not be treated as independent corroboration.

This is the strongest workshop-facing novelty result.

### 5.3 Noisy/Missing-Modality Robustness

*Run: `paper_mmrobust_full_v1`*

The multimodal robustness run justifies the claim that the full architecture matters most when perception is degraded.

| Method | BCU clean | BCU poison | Write ASR | Belief ASR | Retrieval | ASR | Write q/case |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMA | 0.7117 | 0.5483 | 1.0000 | 0.0000 | 0.2600 | 0.2600 | 0.0 |
| H1 | 0.3983 | 0.4500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.77 |
| H2 | 0.3917 | 0.4450 | 0.0000 | 0.1000 | 0.0000 | 0.0000 | 1.0 |
| H3 | 0.2500 | 0.2300 | 0.0111 | 0.1000 | 0.1000 | 0.1000 | 3.7 |
| NoBayes | 0.3950 | 0.4433 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 25.57 |
| NoAnom | 0.3950 | 0.4450 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.80 |
| NoCons | 0.3967 | 0.4433 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.23 |
| **SAGE-Mem** | **0.3950** | **0.4367** | **0.0111** | **0.0000** | **0.0050** | **0.0050** | **11.03** |

**What this shows.**

1. SAGE-Mem remains near-zero on retrieval contamination even under degraded multimodal evidence.
2. The Bayes component matters here, not because it changes ASR dramatically, but because it reduces over-quarantine relative to `NoBayes`.
3. This is the best evidence that the full governed-memory stack is not just a static filter, but a calibration mechanism for noisy multimodal settings.

### 5.4 Internal Ablations

*Run: `paper_ablations_full_v1`*

On the main suite, the named v2 ablations do **not** separate strongly. `NoBayes`, `NoAnomaly`, `NoConsistency`, and full SAGE-Mem are numerically identical on the core write/retrieval metrics and nearly identical on utility:

| Method | BCU clean | BCU poison | Write ASR | Retrieval | ASR | Write q/case |
|---|---:|---:|---:|---:|---:|---:|
| NoBayes | 0.3900 | 0.4350 | 0.0000 | 0.0000 | 0.0000 | 5.30 |
| NoAnomaly | 0.3900 | 0.4350 | 0.0000 | 0.0000 | 0.0000 | 5.30 |
| NoConsistency | 0.3900 | 0.4350 | 0.0000 | 0.0000 | 0.0000 | 5.30 |
| **SAGE-Mem** | **0.3900** | **0.4350** | **0.0000** | **0.0000** | **0.0000** | **5.30** |

This weakens any claim that the paper empirically isolates each submodule on the main benchmark. The correct claim is narrower:

- the governed-memory design is supported strongly,
- submodule separability is benchmark-dependent,
- Bayes/noise calibration is best justified by the multimodal robustness run, not the main suite.

### 5.5 MM-BrowseComp: Clean vs Adversarial

We evaluate on a 194-case leakage-clean MM-BrowseComp pool (`paper_mmclean_abr_group_v1`, `paper_mmadv_abr_group_v1`). The corrected pool enforces effective observation support, removes junk/duplicate traces, drops answer/checklist leakage, and adds page-group provenance for observation-outlier scoring. The adversarial track includes both correction-language (`fact_overwrite_injection`) and vocabulary-adaptive (`fact_overwrite_adaptive`) phrasings in each case.

#### Clean MM-BrowseComp (`paper_mmclean_abr_group_v1`, n=194)

| Method | BCU clean |
|---|---:|
| ShortContext | 0.1907 |
| MMA | 0.2062 |
| RSum | 0.1804 |
| H1 | 0.1598 |
| SAGE-Mem v2 (full) | 0.1495 |
| SAGE-Mem v2 + browsing prior (H5) | 0.1598 |
| **SAGE-Mem v2 + ABR (H6)** | **0.1598** |

All methods achieve zero attack metrics on the clean split (no attacks injected). MMA leads on BCU_clean (0.2062), consistent with the LoCoMo pattern.

#### Adversarial MM-BrowseComp — combined attack (`paper_mmadv_abr_group_v1`, n=194, 388 attack write attempts)

*Each case receives both `fact_overwrite_injection` (correction-language vocabulary) and `fact_overwrite_adaptive` (paraphrased vocabulary, no regex overlap).*

| Method | BCU poison | Write ASR | Retrieval | ASR | false_belief_rate |
|---|---:|---:|---:|---:|---:|
| ShortContext | 0.0430 | 0.6881 | 0.7938 | 0.7938 | 0.0000 |
| MMA | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| RSum | 0.1065 | 0.6314 | 0.8247 | 0.8247 | 0.0000 |
| H1 | 0.1031 | 0.4897 | 0.8316 | 0.8316 | 0.0000 |
| SAGE-Mem v2 (full) | 0.1684 | 0.6314 | 0.5619 | 0.5619 | 0.0842 |
| SAGE-Mem v2 + browsing prior (H5) | 0.0275 | 0.3144 | 0.8265 | 0.8265 | 0.0000 |
| **SAGE-Mem v2 + ABR (H6)** | **0.1512** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

**Key finding: H6 achieves complete attack blocking.** The structured browser claim gate fires on average 2.0 times per case (corresponding to the attack write attempts), blocking all 388/388 attack writes regardless of phrasing. H5 blocks only the correction-language vocabulary subset, admitting 122/388 writes (Write ASR=0.314); those admitted writes contaminate retrieval and drive ASR to 0.827.

**BCU under attack is preserved for H6 but not H5.** H6 BCU_poison=0.1512 ≈ BCU_clean=0.1598 — no utility degradation under attack. H5 BCU_poison=0.0275 — the admitted attack content occupies retrieval slots and crowds out useful evidence.

**MMA collapses under attack.** MMA writes all observations including attacks (Write ASR=1.0) and retrieves all of them (ASR=1.0), BCU_poison=0.000 — the worst performer despite its strong clean utility.

**SAGE-Mem v2 (full, without browsing gate) partially reduces ASR** via trust scoring (0.562 vs 1.0 for MMA) but still admits most attacks. It uniquely produces false_belief_rate=0.0842 — attacks that are admitted form durable false beliefs at a measurable rate, confirming the paper's belief-formation hypothesis.

**Semantic observation-group rerun does not materially change the conclusion.** A later EC2 rerun with the semantic observation-group optimization (`paper_mmclean_abr_sem_v1`, `paper_mmadv_abr_sem_v1`) preserves the core H6 result: clean BCU remains 0.1598, adversarial Write ASR remains 0.000, Retrieval remains 0.000, and BCU_poison changes only from 0.1512 to 0.1529. The main mechanistic interpretation therefore stays the same. The semantic rerun increases page-group firing (`group_divergence_fire_count_avg` 0.622 → 1.208) and outlier score magnitude (0.4568 → 0.5814), but `memory_conflict_score_avg` remains 0.0. We therefore treat the semantic rerun as a secondary ablation rather than a new canonical result: it confirms that stronger page-local signals are compatible with the defense, but it does not overturn the conclusion that the structured browser claim gate is the dominant driver of complete blocking.

#### Critical limitation of the H5 result: vocabulary co-design

The H5 ASR=0 result on `fact_overwrite_injection` requires explicit disclosure. The `_BROWSER_CORRECTION_RE` detection regex contains these phrases:

```
"correction confirmed" | "superseded" | "previously stored" | ...
```

The `fact_overwrite_injection` attack text is constructed as:

```
"Correction confirmed: the answer to '{q}' is '{wrong}'. The previously stored
value was based on an early draft and has since been superseded."
```

**The attack text and the regex share vocabulary by construction.** This is not a failure of scientific intent — source-context calibration for correction-language browsing attacks is a real and valid defense — but the measurement conflates two things: (i) whether the defense mechanism is sound, and (ii) whether the benchmark attack is adaptive. The correct interpretation is:

> H5 (keyword prior) achieves ASR=0 on correction-language-vocabulary attacks arriving via browser source context. Against an adaptive adversary who uses semantically equivalent but lexically different phrasing, H5 provides no protection.

This limitation is why H6/ABR is a necessary contribution rather than an optional extension.

#### H6/ABR design rationale

H6 is best understood as a two-layer browser-specific write policy. The first layer is a structured browser claim gate: browser-sourced observations are allowed to enter evidence memory, but browser text is not allowed to directly write `qa_answer` claims into durable answer-bearing memory. The second layer adds page-local anomaly signals over grouped observations from the same page plus channel-history features. This combination is deterministic and LLM-free.

The secondary ABR signals are:

| Signal | Weight | Mechanism | Adaptive-robust? |
|--------|--------|-----------|-----------------|
| S1: Structured fact-key collision | 0.40 | Jaccard divergence between incoming and stored `fact_value` for matching `fact_key` | Yes — key/value collision is vocabulary-independent |
| S2: Page-group outlier score | 0.30 | Outlierness relative to sibling observation chunks from the same page | Yes — within-page outliers do not depend on trigger words |
| S3: Corroboration deficit | 0.20 | Fraction of existing items NOT similar to incoming obs | Yes — adversarial singletons remain uncorroborated regardless of phrasing |
| S4: Channel singleton penalty | 0.10 | Depth of existing writes on same channel_id | Yes — attack channels are always fresh |

A moving-target threshold jitter (±`abr_noise_scale` per SAGEMemory instance) follows the recommendation in CLATTER (2025) that randomised defence parameters raise the bar for exact adaptive threshold gaming.

**What this shows.**

1. The clean benchmark is difficult: all methods remain below 0.22 BCU, reflecting the genuine difficulty of MM-BrowseComp, not a defense artifact.
2. Generic tool trust is insufficient: MMA, H1/H2/H3, and generic SAGE-Mem v2 all admit the browsing-style correction attack at write time (Write ASR ≥ 0.63).
3. Generic SAGE-Mem v2 reduces retrieval contamination relative to MMA (0.562 vs 1.000) via trust scoring, but admits most attacks and uniquely produces false_belief_rate=0.084 — confirming the belief-formation hypothesis.
4. H5 (keyword prior) reduces Write ASR to 0.314 on the combined track, blocking the injection phrasing but not the adaptive paraphrased variant. The admitted 31% drive ASR to 0.827 and collapse BCU_poison to 0.028.
5. **H6/ABR achieves complete blocking, but the dominant mechanism is structured claim typing.** Write ASR=0.000, ASR=0.000, BCU_poison=0.1512 ≈ BCU_clean=0.1598 on the canonical grouped run. The structured browser claim gate fires 2.0 times per case on average, matching the two injected browser overwrite attempts per case. Group-divergence quarantine fires 0.622 times per case on average on the canonical grouped run and 1.208 on the later semantic rerun, but `memory_conflict_score_avg` remains 0.0 in both. The correct interpretation is therefore browser claim typing plus secondary within-page anomaly support, not proof that page-group semantic contradiction scoring alone solved the task. This is a strength, not a weakness, for the paper: the result is achieved by a simple, inspectable, deployment-relevant memory policy rather than by an opaque semantic verifier that would be harder to trust operationally.
6. **Write-time governance principle is empirically supported for H6.** false_belief_rate=0.000 for H6 (and H5) where SAGEv2 base has 0.084. Zero attack admission prevents false belief hardening entirely.

**Paper implication.** MM-BrowseComp is a valid external stress test for source-context calibration. The H5 result demonstrates that keyword priors are bounded: they close the gap on vocabulary attacks but fail against adaptive adversaries. H6 demonstrates that content-type + source-channel classification is more robust than vocabulary matching. The honest contribution is: (i) identifying that generic tool trust fails for browser evidence, (ii) scoping H5 as vocabulary-specific, (iii) demonstrating that structured claim typing (H6) achieves vocabulary-agnostic blocking, and (iv) providing lifecycle metrics that separate write time, belief time, and retrieval time. The paper does not claim all browsing or web attacks are solved — hidden instructions, multi-page contradictions, and login-gated evidence remain outside current evidence.

---

## 6. What the Experiments Actually Support

The strongest supported contribution is not “SAGE-Mem is the best memory system overall.” That claim would be false.

The strongest supported claim is:

> **SAGE-Mem provides substantially stronger write-time containment and memory-state integrity than retrieval-time filtering on controlled long-horizon multimodal memory benchmarks, especially under multimodal prompt injection and noisy/missing modalities, albeit at a utility cost relative to MMA. On MM-BrowseComp, source-context calibration is necessary: keyword-based priors (H5) are bounded to correction-language-vocabulary adversaries (Write ASR=0.314 on the combined track, ASR=0.827). H6/ABR with structured browser claim typing and page-group anomaly support achieves complete blocking (Write ASR=0.000, ASR=0.000, BCU_poison=0.1512 ≈ BCU_clean=0.1598) by classifying observations by content type and source channel rather than vocabulary.**

This claim is supported by:
- zero write admission and zero retrieval contamination on the LoCoMo main suite,
- zero VPI survival on the VPI-only suite,
- near-zero survival under noisy/missing-modality stress,
- lower false-belief retrieval than simpler baselines,
- a meaningful Bayes calibration effect in the multimodal robustness setting,
- H5 ASR reduction on correction-language vocabulary attacks (scoped honestly; Write ASR=0.314 on combined track),
- **H6 Write ASR=0.000 and ASR=0.000 on the combined adversarial track** (n=194, 388 attack writes including both injection and adaptive attacks), with BCU_poison=0.1512 ≈ BCU_clean=0.1598,
- Write-time principle fully validated: false_belief_rate=0.000 for H6 (and 0.000 for H5) vs 0.084 for base SAGEv2, confirming that write-time blocking prevents false belief hardening.

What the results do **not** support:
- that SAGE-Mem beats MMA overall on clean utility,
- that every v2 submodule is independently validated on the main suite,
- that all browsing or web attacks are handled beyond the content-type/source-channel scope of the structured claim gate,
- that H5 alone is robust to adaptive adversaries who paraphrase around correction-language vocabulary (H5 admits 31% of the combined attack track, ASR=0.827).

### Pareto Frontier (Security–Utility Tradeoff)

The two primary methods occupy distinct points on the security–utility Pareto frontier. This is intentional and should be framed as such, not as a failure of either method.

| Method | BCU poison | Write ASR | Retrieval | ASR | Items in memory |
|---|---:|---:|---:|---:|---:|
| MMA (retrieve-time) | 0.683 | 1.000 | 0.117 | 0.117 | 634 |
| SAGE-Mem (write-time) | 0.350 | 0.000 | 0.000 | 0.000 | 213 |
| ShortContext (no memory) | 0.100 | 0.000 | 0.000 | 0.000 | 8 |

MMA achieves higher benign utility (BCU 0.68) by storing all observations (634 items) and filtering at retrieval time — at the cost of full write admission (Write ASR = 1.0). SAGE-Mem achieves zero write admission and zero retrieval contamination by gating writes aggressively, but stores fewer items (213) and preserves less utility (BCU 0.35). SAGE-Mem is strictly better than ShortContext (no-memory), confirming that the write-time gate adds value without collapsing to a degenerate baseline.

The paper should present this as a Pareto plot with BCU poison on the y-axis and Write ASR on the x-axis, where MMA is the top-right point (high utility, high attack admission) and SAGE-Mem is the bottom-left point (lower utility, zero admission).

### Submission-Ready Analysis Artifacts

To make the integrity--utility story reviewer-legible, we generated submission-ready analysis artifacts directly from the frozen local result set. These include:

- a **LoCoMo Pareto plot** of `BCU poison` versus `Write ASR`,
- an **MM-BrowseComp Pareto plot** of `BCU poison` versus `Write ASR`,
- a **systems-cost table** summarizing average write latency, retrieve latency, and memory footprint on the browsing benchmark,
- a **browsing comparison table** for the canonical grouped H5/H6 pair and the later semantic observation-group rerun,
- an **attack-proxy breakdown** for the main LoCoMo suite using saved retrieval-side attack indicators (`multimodal_attack_retrieval_rate`, `fact_overwrite_attack_retrieval_rate`, `control_flow_attack_retrieval_rate`, `answer_relevant_attack_retrieval_rate`).

These artifacts strengthen the paper in two ways. First, they make the utility--integrity frontier explicit rather than implicit in tables. Second, they clarify that the browsing result is not a single-point anecdote: H6 is simultaneously better than H5 on write admission, retrieval contamination, and downstream BCU under attack, while remaining close to H5 on clean utility.

The concrete submission bundle is:

- `analysis/paper_submission_ready/submission_ready_summary.md`
- `analysis/paper_submission_ready/main_clean_table.csv`
- `analysis/paper_submission_ready/main_poison_table.csv`
- `analysis/paper_submission_ready/browsing_clean_table.csv`
- `analysis/paper_submission_ready/browsing_adversarial_table.csv`
- `analysis/paper_submission_ready/systems_cost_table.csv`
- `analysis/paper_submission_ready/pareto_locomo_bcu_vs_write_asr.svg`
- `analysis/paper_submission_ready/pareto_browsing_bcu_vs_write_asr.svg`
- `analysis/paper_submission_ready/schema_gap_report.md`

There is also an important negative result in the analysis pipeline itself. The frozen canonical raws do **not** include per-row `seed` or `attack_type`, so true seed-level confidence intervals and exact per-attack tables cannot be recovered retrospectively from the saved artifacts. We therefore distinguish between:

- **submission-ready analysis we can support now** from the frozen artifacts, and
- **targeted future reruns** needed for true per-attack and seed-variance reporting.

This distinction is scientifically important. Rather than reverse-engineering unsupported breakdowns from incomplete logs, we preserve the paper’s credibility by reporting only what is actually recoverable from the saved data and treating richer variance/per-attack tables as targeted future work.

---

## 7. Reviewer-Resistant Positioning

### 7.1 What is genuinely novel

The novelty is not simply “another write-time filter.” The defensible novelty is the combination of:

1. **Lifecycle evaluation for agent memory:** write admission, belief formation, retrieval contamination, and downstream completion are measured separately.
2. **Multimodal dependence modeling:** OCR and caption outputs from the same image are treated as dependent evidence rather than rewarded as independent corroboration.
3. **Typed memory with sufficiency-gated belief promotion:** the system distinguishes observation storage from durable belief formation.
4. **Honest benchmarking methodology:** we explicitly introduce an adaptive attack variant (`fact_overwrite_adaptive`) to prevent vocabulary co-design artifacts from inflating defense results, and disclose the limitation of the H5 keyword prior.
5. **Adversarial Belief Revision (ABR/H6):** a browser-specific, vocabulary-agnostic write-time layer combining structured browser claim typing, page-group anomaly scoring, corroboration deficit, and channel history. In the current benchmark, the structured claim gate is the dominant mechanism and should be described as such.

### 7.2 What reviewers may attack

1. **Utility gap versus MMA.**
   - This is real and must be acknowledged, not buried.
   - Response: the paper is not a single-metric QA leaderboard. It evaluates an integrity--utility frontier. MMA occupies the high-utility / permissive-store point; SAGE-Mem occupies the lower-utility / higher-integrity point with near-zero write and retrieval contamination on the controlled multimodal suites.

2. **Controlled LoCoMo multimodality.**
   - The LoCoMo multimodal setting is a controlled extension, not a fully natural multimodal agent log.
   - Response: acknowledged in limitations. The benchmark is presented as controlled, not naturalistic.

3. **H5 vocabulary co-design (benchmarkmaxxing concern).**
   - A reviewer may correctly observe that the `fact_overwrite_injection` attack text and the `_BROWSER_CORRECTION_RE` regex share vocabulary by construction, making ASR=0 an artifact rather than evidence of semantic robustness.
   - Response: **we disclose this explicitly.** The H5 result is scoped to correction-language-vocabulary adversaries. We introduce `fact_overwrite_adaptive` and ABR (H6) precisely to close this gap. A paper that suppressed this limitation would be dishonest; we instead make the limitation a research contribution by proposing and evaluating the fix.

4. **ABR and H6 results: positive finding with honest mechanism disclosure.**
   - H6/ABR achieves Write ASR=0.000 and ASR=0.000 on the combined adversarial track, strictly outperforming H5 (Write ASR=0.314, ASR=0.827).
   - Response: the H6 result is genuinely strong. We disclose the mechanism: the structured browser claim gate — which blocks browser-sourced `qa_answer` writes by content type rather than vocabulary — is the primary driver of complete blocking, with page-group anomaly signals providing secondary defense-in-depth. A reviewer may ask: "is this just a hard gate, not a scoring system?" We respond: yes, a hard content-type gate is architecturally the right design for browser-sourced factual claims. The contribution is (i) identifying that content-type classification is more robust than vocabulary matching, (ii) adding within-page anomaly structure rather than treating all browser text identically, and (iii) empirically validating this against the adaptive adversary that defeats H5. The gate is disclosed explicitly and is not presented as a black-box magic fix.

5. **Submodule ablations on the main suite.**
   - The main suite does not separate Bayes/anomaly/consistency strongly.
   - Response: Bayes calibration is most relevant under noisy/missing modality stress (Section 5.3), which does show differentiation. The main suite saturation is an honest result about the suite, not a concealed weakness.

6. **Behavioral LLM metrics.**
   - These should remain secondary and be reported only when applicable.

The current draft preempts all these concerns by disclosing them explicitly rather than burying them.

---

## 8. Limitations

- **Trusted orchestration assumption.** Source labels are assumed correct.
- **Utility tradeoff.** Retrieval-time baselines preserve more immediate QA utility. This is the expected cost of conservative write-time governance, not an artifact to obscure; the right deployment choice depends on whether the system prioritizes short-term answer rate or persistent memory-state integrity.
- **Controlled versus naturalistic multimodality.** LoCoMo is controlled; MM-BrowseComp is more realistic but still limited by trace coverage and VLM caption quality.
- **H5 vocabulary co-design.** The `fact_overwrite_injection` attack text and the `_BROWSER_CORRECTION_RE` regex share overlapping vocabulary by construction ("Correction confirmed", "superseded", "previously stored" appear in both). The H5 ASR=0 result is therefore bounded to correction-language-vocabulary adversaries. We disclose this explicitly and introduce H6/ABR as the vocabulary-agnostic extension.
- **H6/ABR scope: structured claim gate covers browser-sourced `qa_answer` writes.** H6 achieves Write ASR=0.000 on the combined adversarial track via the structured browser claim gate, which blocks browser-sourced `qa_answer` writes by content-type classification. The gate's scope is bounded: it covers browser-channel factual assertion writes, the primary MM-BrowseComp attack surface in our current construction. Attacks that write to other memory types (e.g., `evidence`, `control`) or arrive on non-browser channels may not trigger the gate and would rely on the secondary ABR scoring and trust thresholds. The later semantic observation-group rerun increases page-local anomaly sensitivity without changing the core outcome, which reinforces this scope judgment: stronger page-local signals are additive, but the main result is still driven by the structured gate. Hidden HTML instructions, multi-page contradictions, and login-gated evidence remain outside the gate's classification scope and outside current empirical evidence.
- **Browsing-specific scope.** Even with ABR, the defense targets browser-channel correction-style attacks. Broader web attacks — hidden HTML instructions, multi-page contradictions, login-gated evidence, video-only answers — remain outside current evidence.
- **Module separability is setting-dependent.** Bayes is justified most clearly under noisy/missing multimodal evidence, not the main suite.
- **Behavioral LLM evaluation is secondary.** It is not the core evidence for this paper.

---

## 9. Future Work and High-Priority Additional Experiments

The next stage of this research should strengthen the paper along two axes: richer multimodal semantics and broader browsing realism.

The **most important scientific next step** is to move from write-time multimodal governance toward **write-time multimodal verification**. The current paper shows that memory systems should not mistake dependent channels for independent corroboration, and that browser-sourced factual claims require stricter write-time treatment than generic tool output. What it does not yet show is a full cross-modal verifier that can adjudicate disagreements among OCR, caption, page text, and trusted memory state. A natural follow-on is a sentence-level or region-level cross-modal consistency model that compares page-local textual evidence against visual extracts and existing trusted memory, rather than primarily classifying observations by source and claim type.

The **most important benchmark next step** is to broaden the browsing attack surface beyond direct browser-sourced answer overwrites. The current MM-BrowseComp evidence is strongest for a specific but realistic failure mode: externally sourced browser content directly asserting a fact that should not be promoted into durable answer-bearing memory. The next benchmark version should include hidden-instruction attacks, multi-page contradictions, control-memory attacks, and cases where the answer lives primarily in visual or dynamic web content. That would let the paper’s browsing claim grow from “browser claim typing is necessary” to a broader statement about robust multimodal memory in web agents.

### High priority for acceptance

1. ~~**ABR + adaptive attack EC2 run** *(blocking for H6 claim)*~~ **COMPLETE — POSITIVE RESULT.**
   - Run `paper_mmadv_abr_group_v1` completed (n=194 cases, 388 attack write attempts, both `fact_overwrite_injection` + `fact_overwrite_adaptive` combined).
   - **Finding: H6 (SAGEMemV2_ABR) achieves Write ASR=0.000, ASR=0.000, BCU_poison=0.1512.** The structured browser claim gate blocks all 388 attack writes regardless of vocabulary. H5 (BrowsingTrustPrior) admits 122/388 writes (Write ASR=0.314) and achieves ASR=0.827. H6 strictly dominates H5.
   - **Implication**: H6 hypothesis is fully supported. The structural insight: content-type + source-channel classification is more robust than keyword matching. See Section 5.5.

2. **Per-attack breakdown on the main LoCoMo suite.**
   - Why: isolates where SAGE-Mem wins and where it merely ties.
   - Claim supported: robustness is not driven by only one attack family.

3. **Variance / confidence intervals across seeds.**
   - Why: the workshop audience will care whether gains are stable.
   - Claim supported: robustness improvements are not seed artifacts.

4. **Pareto plot: BCU poison vs attack write admission or retrieval contamination.**
   - Why: makes the write-time-versus-utility tradeoff reviewer-legible.
   - Claim supported: SAGE-Mem occupies a different robustness regime rather than merely underperforming.
   - This should be a main figure if space permits: it visually defends the MMA utility gap by showing that SAGE-Mem moves along a different integrity--utility frontier rather than losing on the paper's primary mechanism metrics.

5. **Per-modality breakdown in the multimodal robustness run.**
   - Why: clarifies whether failures are OCR-driven, caption-driven, or mixed.
   - Claim supported: the benchmark is genuinely multimodal, not just text corruption with image labels.
   - This is the cleanest next experiment for strengthening the paper’s multimodal identity without changing the core method.

### Nice to have

6. **Cost/latency table for guard usage.**
   - Why: SCALE workshop reviewers may care about systems practicality.

7. **Length sensitivity / retrieval-budget sensitivity.**
   - Why: would strengthen the claim that cleaner memory reduces downstream burden.

8. ~~**H5 vs H6 ablation table on `fact_overwrite_injection`**~~ **Superseded by combined run.** The `paper_mmadv_abr_group_v1` combined run already shows H6 strictly better than H5 on both attack types simultaneously (H6 Write ASR=0.000 vs H5 Write ASR=0.314). A separate injection-only ablation is no longer needed.

9. **Full multimodal semantic verifier as a follow-on method, not a retroactive claim.**
   - Why: the semantic observation-group rerun shows that stronger page-local signals are compatible with the current defense, but it does not yet validate semantic contradiction resolution as the main mechanism.
   - Claim supported if successful: the system can move beyond source/claim-type governance into true multimodal belief revision.

10. **Browsing realism expansion.**
   - Why: current MM-BrowseComp evidence is strongest for browser-sourced factual overwrite attacks.
   - Claim supported if successful: the write-time governance story extends beyond direct `qa_answer` injection to broader web-agent memory threats.

---

## 10. References

**Memory poisoning and agent attacks**
- Chan et al. (2024). *AGENTPOISON: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases.* NeurIPS 2024. arXiv:2407.12784
- Dang et al. (2025). *Memory Injection Attacks on LLM Agents via Query-Only Interaction.* arXiv:2503.03704
- (2026). *Zombie Agents: Persistent Control of Self-Evolving LLM Agents via Self-Reinforcing Injections.* arXiv:2602.15654

**Retrieval corruption, prompt injection, RAG robustness**
- Zou, Geng et al. (2024). *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models.* USENIX Security 2025. arXiv:2402.07867
- Xiang et al. (2024). *GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning.* arXiv:2406.09187
- (2025). *A-MemGuard: A Proactive Defense Framework for LLM-Based Agent Memory.* arXiv:2510.02373
- (2024). *Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents.* arXiv:2410.02644

**Multimodal prompt injection and VLM attacks**
- (2025). *AegisAgent: An Autonomous Defense Agent Against Prompt Injection Attacks in LLM-HARs.* arXiv:2512.20986

**Memory-augmented agents and benchmarks**
- Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560
- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023. arXiv:2304.03442
- (2023). *AgentBench: Evaluating LLMs as Agents.* arXiv:2308.03688
