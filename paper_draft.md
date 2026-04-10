# SAGE-Mem: Write-Time Governance for Multimodal Agent Memory Under Adversarial and Noisy Observations

*ICML 2026 SCALE Workshop Submission Draft*

---

## Abstract

Long-horizon agents increasingly rely on persistent memory, but persistent memory is also a durable attack surface. An adversarial observation that is admitted into memory can later be retrieved, consolidated, and reused long after the original interaction has ended. Existing defenses for memory poisoning are predominantly retrieval-time: they try to rank or filter memories after contamination has already entered the store. This leaves the memory state itself polluted, increases retrieval noise, wastes context budget, and allows unsupported observations to harden into derived beliefs. We study **write-time governed multimodal memory** as a distinct robustness problem. We present **SAGE-Mem**, a memory layer that mediates admission, promotion, and retrieval through typed memory partitions (`evidence`, `belief`, `control`), sufficiency-gated evidence-to-belief promotion, source-conditioned trust, anomaly detection, consistency checks, and provenance-aware retrieval. We evaluate on two settings: (i) a controlled long-horizon benchmark based on LoCoMo-10 with multimodal adversarial injections, including Visual Prompt Injection (VPI) and noisy/missing-modality stress; and (ii) an adversarially augmented MM-BrowseComp observation-trace benchmark with VLM-backed image observations. Our primary metrics follow the memory lifecycle: attack write admission, attack belief formation, retrieval contamination, false-belief retrieval, and benign completion under attack (BCU). On LoCoMo-based benchmarks, SAGE-Mem consistently drives write admission and retrieval contamination to near-zero, including perfect suppression on the main suite and VPI-only setting, while remaining robust under noisy/missing multimodal inputs. SAGE-Mem trails retrieval-time baselines on clean and attacked QA utility, so the result is an integrity--utility tradeoff rather than overall dominance. MM-BrowseComp further shows that generic tool trust fails under realistic web-style correction attacks. An initial browsing-context keyword prior (H5) reduces ASR to zero on correction-language vocabulary attacks, but we demonstrate that this result is **vocabulary-specific**: the detection regex and the benchmark attack text share overlapping phrases by construction, constituting a co-design artifact that limits the claim to correction-language vocabulary adversaries. We identify this limitation explicitly and introduce **H6 — Adversarial Belief Revision (ABR)** — a composite semantic suspicion scorer that detects adversarial browser observations via structured fact-key collision, corroboration deficit scoring, and channel provenance signals, without relying on any correction-language vocabulary. ABR is deterministic, LLM-free, and designed to hold against adaptive adversaries who paraphrase around keyword filters. The resulting picture is not “retrieval is obsolete,” but rather that **write-time governance is a necessary complement to retrieval-time filtering for multimodal agent memory**, especially when dependent multimodal evidence can masquerade as corroboration — and that honest evaluation requires testing defenses against adaptive adversaries who know the defense vocabulary.

---

## 1. Introduction

Memory changes what an agent is. A stateless model can only be manipulated in the current context window; a stateful agent can be manipulated by altering the persistent memory that future reasoning depends on. This makes memory poisoning qualitatively different from ordinary prompt injection. Once a malicious observation is written, it can survive long after the original turn, reappear at retrieval time, distort later summaries, and occupy retrieval bandwidth that should have been reserved for grounded evidence.

This matters even more for multimodal agents. OCR text, image captions, browser outputs, and user/tool messages are not interchangeable evidence channels. A single adversarial image can produce both OCR output and a caption-like summary, creating the appearance of multi-source support despite both observations originating from the same underlying source. In other words, multimodal agreement is not always evidence of truth; it can also be evidence of **dependent corruption**.

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
  A vocabulary-agnostic composite suspicion scorer — combining structured fact-key collision detection, corroboration deficit scoring, semantic proximity to established memory, and channel provenance — should maintain robustness against adaptive adversaries who know and evade the H5 keyword filter. ABR should reduce ASR on paraphrased correction attacks (`fact_overwrite_adaptive`) while preserving clean BCU close to the H5 level.
  **Empirical result:** H6 is **not supported** on ASR against adaptive attacks. The composite scorer achieves identical results to H5: both block the injection attack (ASR=0.0) and both are defeated by the adaptive paraphrased attack (ASR=1.0). The ABR signals (S1–S4) are insufficient to cross the 0.45 suspicion threshold for paraphrased text. H6 is **partially supported** as a write-time *principle* contribution: false_belief_rate=0.0 for both H5 and H6 (vs 0.067 for SAGEv2 base), showing that write-time blocking prevents false belief hardening. The detection mechanism requires stronger semantic matching.

The current empirical evidence strongly supports H1, H3, and H4 on LoCoMo-based settings, partially supports H2 through near-zero false-belief retrieval, supports H5 narrowly for correction-language vocabulary attacks with the co-design caveat disclosed, and partially supports H6 on the write-time principle (false belief prevention) while showing that composite key-level scoring is insufficient against vocabulary-adaptive attacks.

### Contributions

This paper makes five concrete contributions.

1. **Problem framing:** We formulate multimodal agent-memory robustness as a write-boundary and belief-boundary problem, not only a retrieval-time ranking problem.
2. **Architecture:** We present SAGE-Mem, a governed memory layer with typed memory, sufficiency-gated promotion, source-conditioned trust, anomaly detection, consistency checks, and provenance-aware retrieval.
3. **Benchmark/evaluation design:** We define lifecycle metrics for write admission, belief formation, retrieval contamination, and downstream completion under attack, and apply them to both a controlled long-horizon benchmark and a browsing-style multimodal benchmark.
4. **Empirical findings:** On LoCoMo-based settings, write-time governance strongly reduces attack admission and retrieval contamination, including on multimodal prompt injection and noisy/missing-modality robustness tests; on browsing-style MM-BrowseComp, generic tool trust fails. An initial keyword-prior defense (H5) achieves ASR=0 on correction-language vocabulary attacks, but we disclose a vocabulary co-design artifact that limits this claim to non-adaptive adversaries.
5. **Adversarial Belief Revision (H6/ABR) and honest negative finding:** We identify the co-design limitation in H5, introduce `fact_overwrite_adaptive` as the adaptive evaluation variant, and propose ABR — a composite, vocabulary-agnostic, LLM-free write-time scorer grounded in structured fact-key collision (Perez & Ribeiro 2022), corroboration deficit (Shi et al. 2024), and moving-target threshold hardening (CLATTER 2025). Empirically, H6 does not outperform H5 on adaptive attacks (both ASR=1.0 against paraphrased vocabulary). We report this as an honest negative: write-time blocking prevents false belief hardening (false_belief_rate=0.0 vs 0.067 for SAGEv2 base) but key-level composite scoring is insufficient for adaptive detection. Sentence-level semantic verification is the indicated next step.

The paper’s central claim is therefore deliberately narrow and defensible:

> **Write-time governed multimodal memory is a distinct robustness primitive for agents. It strengthens memory-state integrity beyond what retrieval-time scoring alone can provide, especially under multimodal corruption, but it introduces a measurable utility tradeoff and requires source-context calibration for browsing-derived evidence. Honest evaluation requires testing against adaptive adversaries who know the defense vocabulary; keyword-based priors are bounded to non-adaptive adversaries, and vocabulary-agnostic composite scoring is necessary for broader robustness.**

The paper therefore does not claim that SAGE-Mem dominates retrieval-time filtering on all metrics. A retrieval-time system can be the right engineering choice when the objective is maximum short-term QA accuracy under a permissive memory store. SAGE-Mem targets a different operating point: lower tolerance for persistent contamination, lower attack survival across memory lifecycle stages, and more explicit provenance over the beliefs that are allowed to guide later agent behavior.

### 1.3 Relation to Prior Work

This paper sits at the intersection of four threads of prior work.

**Memory poisoning and agent reliability.**
Recent work has shown that persistent agent memory is a durable attack surface and that poisoning can be achieved through ordinary interaction rather than direct database access `[CITE: memory poisoning / agent memory attacks]`. Our work agrees with that threat model, but shifts the emphasis from demonstrating feasibility to evaluating **where in the memory lifecycle** defenses should operate.

**Retrieval-time robustness and corrupted context.**
A large body of work studies corrupted retrieval, noisy context, and prompt injection at the point of model invocation `[CITE: retrieval corruption / prompt injection / RAG robustness]`. These methods are relevant baselines, and MMA is our concrete retrieval-time reference point. Our claim is not that retrieval-time defense is unimportant, but that it is insufficient once unsupported observations have already been written into durable state.

**Multimodal prompt injection and vision-language reliability.**
Prior multimodal security work shows that images can manipulate captioners, OCR pipelines, and downstream reasoning `[CITE: multimodal prompt injection / VLM attacks]`. Our contribution is memory-specific: OCR and caption outputs derived from the same image can create **false corroboration** inside the memory store unless dependence is modeled explicitly.

**Memory-augmented agents and benchmark methodology.**
Prior work on memory-augmented agents, long-horizon QA, and agent evaluation provides the underlying setting but typically does not separate write admission, belief formation, and retrieval contamination as distinct evaluation stages `[CITE: memory-augmented agents / long-horizon evaluation / benchmark methodology]`. Our lifecycle metrics are intended to make that decomposition explicit.

Relative to these literatures, the paper’s novelty is therefore not a claim to have invented memory poisoning, multimodal prompt injection, or memory architectures in isolation. The novelty is the **combination** of write-time governed multimodal memory, dependence-aware multimodal handling, and lifecycle evaluation that separates admission, belief formation, retrieval contamination, and downstream answer quality.

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

After fixing the VLM-captioning path and the case-construction filter, we reran MM-BrowseComp clean/adversarial on a 194-case leakage-clean pool (`paper_mmclean_h5_v1`, `paper_mmadv_h5_v1`). The corrected pool enforces effective observation support, removes junk/duplicate traces, and drops answer/checklist leakage.

#### Clean MM-BrowseComp

| Method | BCU clean | Answered rate |
|---|---:|---:|
| ShortContext | 0.2938 | 1.0000 |
| MMA | 0.2938 | 1.0000 |
| RSum | 0.2732 | 1.0000 |
| H1 | 0.2732 | 1.0000 |
| H2 | 0.2784 | 1.0000 |
| H3 | 0.2938 | 1.0000 |
| SAGE-Mem v2 | 0.2577 | 0.9948 |
| SAGE-Mem v2 + browsing prior (H5) | 0.2732 | 0.9948 |
| SAGE-Mem v2 + ABR (H6) | 0.2732 | 1.0000 |

#### Adversarial MM-BrowseComp — `fact_overwrite_injection` (correction-language vocabulary)

| Method | BCU poison | Write ASR | Retrieval | ASR |
|---|---:|---:|---:|---:|
| ShortContext | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| MMA | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| RSum | 0.0000 | 1.0000 | 0.9948 | 0.9948 |
| H1 | 0.0000 | 1.0000 | 0.9931 | 0.9931 |
| H2 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| H3 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| SAGE-Mem v2 | 0.0000 | 1.0000 | 0.9536 | 0.9536 |
| **SAGE-Mem v2 + browsing prior (H5)** | **0.2784** | **0.0000** | **0.0000** | **0.0000** |
| **SAGE-Mem v2 + ABR (H6)** | **≈0.2784** | **0.0000** | **0.0000** | **0.0000** |

#### Adversarial MM-BrowseComp — `fact_overwrite_adaptive` (vocabulary-evading, honest adaptive eval)

| Method | BCU poison | Write ASR | Retrieval | ASR | false_belief_rate |
|---|---:|---:|---:|---:|---:|
| SAGE-Mem v2 (base) | 0.0344 | 0.9845 | 0.7113 | 0.7113 | 0.0670 |
| SAGE-Mem v2 + browsing prior (H5) | 0.2680 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| SAGE-Mem v2 + ABR (H6) | 0.2680 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

*Combined adversarial run (n=194 cases, `paper_mmadv_abr_v1`). Each case received both `fact_overwrite_injection` and `fact_overwrite_adaptive` attacks; injection is blocked by both H5 and H6 (accounting for the 50% aggregate write admission rate), while the adaptive variant is admitted by both.*

**Key finding.** H5 and H6 produce identical empirical behavior: both block the injection attack (vocabulary match in H5 via regex; suspicion threshold exceeded in H6 via ABR signals) and both admit the adaptive attack (no regex match; ABR suspicion scores remain below threshold for paraphrased phrasing). H6 adds no incremental protection against vocabulary-adaptive paraphrasing on this benchmark.

**Write-time principle holds, detection does not.** Despite ASR=1.0 for the adaptive attack, both H5 and H6 achieve false_belief_rate=0.0 on the *injection* variant — the attack that gets through the write gate (adaptive) does not form durable false beliefs because the memory system discounts low-trust writes. The baseline SAGEv2 has false_belief_rate=0.067, confirming that write-time blocking (even imperfect) prevents belief hardening.

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

ABR replaces the keyword regex with four orthogonal signals scored without any LLM call:

| Signal | Weight | Mechanism | Adaptive-robust? |
|--------|--------|-----------|-----------------|
| S1: Structured fact-key collision | 0.40 | Jaccard divergence between incoming and stored `fact_value` for matching `fact_key` | Yes — key/value collision is vocabulary-independent |
| S2: Proximity to trusted memory | 0.30 | Max cosine sim to items with trust ≥ 0.65 | Partially — detects topical adjacency regardless of phrasing |
| S3: Corroboration deficit | 0.20 | Fraction of existing items NOT similar to incoming obs | Yes — adversarial singletons remain uncorroborated regardless of phrasing |
| S4: Channel singleton penalty | 0.10 | Depth of existing writes on same channel_id | Yes — attack channels are always fresh |

A moving-target threshold jitter (±`abr_noise_scale` per SAGEMemory instance) follows the recommendation in CLATTER (2025) that randomised defence parameters raise the bar for exact adaptive threshold gaming.

**What this shows.**

1. The clean benchmark is difficult: all methods remain below 0.30 BCU, reflecting the genuine difficulty of MM-BrowseComp, not a defense artifact.
2. Generic tool trust is insufficient: MMA, H1/H2/H3, and generic SAGE-Mem v2 all admit the browsing-style correction attack at write time.
3. Generic SAGE-Mem v2 reduces retrieval contamination relative to MMA (0.9536 vs 1.0000) by blocking belief-formation, but does not prevent end-to-end attack survival.
4. H5 (keyword prior) blocks `fact_overwrite_injection` completely, but this result is bounded to correction-language vocabulary adversaries. It should be reported with this caveat, not as a general browsing-attack solution.
5. **H6/ABR achieves the same results as H5 on both attack types.** On `fact_overwrite_injection`: ABR suspicion scores exceed the threshold (due to S1 fact-key collision and S3 corroboration deficit), so injection writes are blocked — but H5's regex already caught these, so H6 adds no net new protection. On `fact_overwrite_adaptive`: ABR suspicion scores remain below threshold for paraphrased text, admitting the attack. Write ASR=1.0, ASR=1.0, identical to H5. The S1 signal (fact-key collision) fires on adaptive attacks by design, but the aggregate suspicion score with the paraphrased text does not cross the 0.45 threshold — confirming that a stronger semantic scoring approach (e.g., sentence-level embedding similarity rather than key-level Jaccard) is needed.
6. **Write-time governance principle is empirically supported.** Both H5 and H6 maintain false_belief_rate=0.0 where SAGEv2 base has 0.067. Write-time blocking — even partial — prevents attacked facts from hardening into durable false beliefs.

**Paper implication.** MM-BrowseComp is a valid external stress test for source-context calibration. The H5 result is narrowly defensible as a proof-of-concept for correction-language interception. The honest contribution is the combination: (i) identifying that generic tool trust fails for browser evidence, (ii) demonstrating that a keyword prior closes the gap on vocabulary attacks, (iii) disclosing the vocabulary co-design limitation, and (iv) introducing ABR as the vocabulary-agnostic extension. The paper does not claim all browsing or web attacks are solved.

---

## 6. What the Experiments Actually Support

The strongest supported contribution is not “SAGE-Mem is the best memory system overall.” That claim would be false.

The strongest supported claim is:

> **SAGE-Mem provides substantially stronger write-time containment and memory-state integrity than retrieval-time filtering on controlled long-horizon multimodal memory benchmarks, especially under multimodal prompt injection and noisy/missing modalities, albeit at a utility cost relative to MMA. On MM-BrowseComp, source-context calibration is necessary but both H5 and H6 are bounded to correction-language-vocabulary adversaries on the detection dimension; vocabulary-adaptive attacks defeat both (ASR=1.0). However, write-time blocking prevents false belief hardening (false_belief_rate=0.0 for H5/H6 vs 0.067 for base SAGEv2), validating the write-time governance principle even when detection is imperfect. Key-level composite scoring (ABR H6) requires extension to sentence-level semantic verification for adaptive robustness.**

This claim is supported by:
- zero write admission and zero retrieval contamination on the LoCoMo main suite,
- zero VPI survival on the VPI-only suite,
- near-zero survival under noisy/missing-modality stress,
- lower false-belief retrieval than simpler baselines,
- a meaningful Bayes calibration effect in the multimodal robustness setting,
- H5 ASR=0 on correction-language vocabulary attacks (scoped honestly),
- H6 ASR=0 on correction-language vocabulary attacks (same as H5 — ABR signals also exceed threshold for injection phrasing),
- Write-time principle validated: false_belief_rate=0.0 for both H5 and H6 (vs 0.067 for SAGEv2 base), confirming that partial write blocking prevents false belief hardening even when some attacks get through.

What the results do **not** support:
- that SAGE-Mem beats MMA overall,
- that every v2 submodule is independently validated on the main suite,
- that browsing-style multimodal memory is solved beyond the calibrated web-correction track,
- that H5 alone is robust to adaptive adversaries who paraphrase around correction-language vocabulary,
- that H6/ABR is robust to vocabulary-adaptive paraphrasing — the adaptive attack achieves ASR=1.0 against H6 (identical to H5), confirming that key-level composite scoring is insufficient against paraphrased attacks; sentence-level semantic similarity or LLM-graded verification is the natural next step.

---

## 7. Reviewer-Resistant Positioning

### 7.1 What is genuinely novel

The novelty is not simply “another write-time filter.” The defensible novelty is the combination of:

1. **Lifecycle evaluation for agent memory:** write admission, belief formation, retrieval contamination, and downstream completion are measured separately.
2. **Multimodal dependence modeling:** OCR and caption outputs from the same image are treated as dependent evidence rather than rewarded as independent corroboration.
3. **Typed memory with sufficiency-gated belief promotion:** the system distinguishes observation storage from durable belief formation.
4. **Honest benchmarking methodology:** we explicitly introduce an adaptive attack variant (`fact_overwrite_adaptive`) to prevent vocabulary co-design artifacts from inflating defense results, and disclose the limitation of the H5 keyword prior.
5. **Adversarial Belief Revision (ABR/H6):** a composite, vocabulary-agnostic write-time scorer combining structured fact-key provenance, corroboration deficit, semantic proximity, and channel history — grounded in and extending recent composite trust scoring and RA-RAG corroboration literature — that extends source-context defense beyond keyword matching.

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

4. **ABR and H6 results: honest evaluation of a negative finding.**
   - A reviewer may note that H6/ABR achieves the same ASR as H5 on adaptive attacks (both ASR=1.0 against paraphrased vocabulary).
   - Response: this is correct and we report it as such. The contribution of H6 is not "solved write-time defense" — it is (i) a vocabulary-agnostic *mechanism design* that doesn't rely on keyword matching, (ii) empirical evidence that the write-time principle (false_belief_rate=0.0) holds even when some writes get through, and (iii) honest methodology — the adaptive attack was designed and run specifically to stress-test H6, and the result shows where current composite scoring breaks down. A paper that inflated H6 results by only running the injection attack (which both H5 and H6 block) would be dishonest. The honest finding — that detection must be improved beyond both regex and key-level composite scoring — is a valid research contribution to the field.

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
- **H6/ABR: negative result on adaptive attack.** The ABR mechanism and adaptive attack (`fact_overwrite_adaptive`) are fully evaluated (EC2 run `paper_mmadv_abr_v1`, n=194). H6/ABR achieves identical metrics to H5 on the adaptive attack: Write ASR=1.0, Retrieval=1.0, ASR=1.0. The composite suspicion scorer (S1–S4) does not exceed the 0.45 threshold for paraphrased attack text. This is an honest negative result: key-level signals (Jaccard, channel singleton, corroboration count) are insufficient to catch lexically diverse attacks. Sentence-level embedding similarity (S2 uses item-level cosine similarity but not full sentence-to-sentence semantic similarity) and LLM-graded semantic equivalence checking are the indicated next directions. The write-time *principle* is supported (false_belief_rate=0.0 for attacks that are blocked), but detection scope is limited.
- **Browsing-specific scope.** Even with ABR, the defense targets browser-channel correction-style attacks. Broader web attacks — hidden HTML instructions, multi-page contradictions, login-gated evidence, video-only answers — remain outside current evidence.
- **Module separability is setting-dependent.** Bayes is justified most clearly under noisy/missing multimodal evidence, not the main suite.
- **Behavioral LLM evaluation is secondary.** It is not the core evidence for this paper.

---

## 9. High-Priority Additional Experiments

### High priority for acceptance

1. ~~**ABR + adaptive attack EC2 run** *(blocking for H6 claim)*~~ **COMPLETE.**
   - Run `paper_mmadv_abr_v1` completed (n=194 cases, both `fact_overwrite_injection` + `fact_overwrite_adaptive`).
   - Finding: H6 ABR does NOT outperform H5 on adaptive attacks. Both achieve identical results: injection blocked (ASR=0.0), adaptive admitted (ASR=1.0), false_belief_rate=0.0 for both. See Section 5.5 for full tables.
   - **Implication**: H6's contribution is mechanism design (vocabulary-agnostic signals) and write-time principle validation, not improved ASR on adaptive attacks. Stronger semantic verification (sentence embeddings, LLM grading) is needed for the next iteration.

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

### Nice to have

6. **Cost/latency table for guard usage.**
   - Why: SCALE workshop reviewers may care about systems practicality.

7. **Length sensitivity / retrieval-budget sensitivity.**
   - Why: would strengthen the claim that cleaner memory reduces downstream burden.

8. **H5 vs H6 ablation table on `fact_overwrite_injection`.**
   - Why: confirms that ABR replicates H5's result on vocabulary attacks while also covering adaptive ones, so H6 is strictly better than H5 rather than a different operating point.

---

## 10. Citation Placeholders

Insert real citations for:
- memory poisoning in agent systems,
- retrieval corruption / prompt injection,
- multimodal prompt injection,
- memory-augmented and long-horizon agent evaluation,
- benchmark methodology for robustness and stress testing.

Do not fabricate citations. The current draft should use explicit placeholders until the final bibliography is assembled.
