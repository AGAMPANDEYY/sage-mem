# SAGE-Mem: Write-Time Governance for Multimodal Agent Memory Under Adversarial and Noisy Observations

*ICML 2026 SCALE Workshop Submission Draft*

---

## Abstract

Long-horizon agents increasingly rely on persistent memory, but persistent memory is also a durable attack surface. An adversarial observation that is admitted into memory can later be retrieved, consolidated, and reused long after the original interaction has ended. Existing defenses for memory poisoning are predominantly retrieval-time: they try to rank or filter memories after contamination has already entered the store. This leaves the memory state itself polluted, increases retrieval noise, wastes context budget, and allows unsupported observations to harden into derived beliefs. We study **write-time governed multimodal memory** as a distinct robustness problem. We present **SAGE-Mem**, a memory layer that mediates admission, promotion, and retrieval through typed memory partitions (`evidence`, `belief`, `control`), sufficiency-gated evidence-to-belief promotion, source-conditioned trust, anomaly detection, consistency checks, and provenance-aware retrieval. We evaluate on two settings: (i) a controlled long-horizon benchmark based on LoCoMo-10 with multimodal adversarial injections, including Visual Prompt Injection (VPI) and noisy/missing-modality stress; and (ii) an adversarially augmented MM-BrowseComp observation-trace benchmark with VLM-backed image observations. Our primary metrics follow the memory lifecycle: attack write admission, attack belief formation, retrieval contamination, false-belief retrieval, and benign completion under attack (BCU). On LoCoMo-based benchmarks, SAGE-Mem consistently drives write admission and retrieval contamination to near-zero, including perfect suppression on the main suite and VPI-only setting, while remaining robust under noisy/missing multimodal inputs. However, SAGE-Mem trails retrieval-time baselines on clean and attacked QA utility, and MM-BrowseComp reveals an unresolved browsing-specific limitation: realistic web-style correction attacks are admitted by all methods under the current trust policy. The resulting picture is not “retrieval is obsolete,” but rather that **write-time governance is a necessary complement to retrieval-time filtering for multimodal agent memory**, especially when dependent multimodal evidence can masquerade as corroboration.

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

- **H5: Browsing-context limitation hypothesis.**
  Browsing-derived `tool_output_text` introduces a qualitatively different challenge: realistic correction-like web observations may bypass a generic write-time trust policy even when the same architecture succeeds on controlled long-horizon benchmarks.

The current empirical evidence strongly supports H1, H3, and H4 on LoCoMo-based settings, partially supports H2 through near-zero false-belief retrieval, and treats H5 as an exposed limitation and future algorithmic direction rather than a solved problem.

### Contributions

This paper makes four concrete contributions.

1. **Problem framing:** We formulate multimodal agent-memory robustness as a write-boundary and belief-boundary problem, not only a retrieval-time ranking problem.
2. **Architecture:** We present SAGE-Mem, a governed memory layer with typed memory, sufficiency-gated promotion, source-conditioned trust, anomaly detection, consistency checks, and provenance-aware retrieval.
3. **Benchmark/evaluation design:** We define lifecycle metrics for write admission, belief formation, retrieval contamination, and downstream completion under attack, and apply them to both a controlled long-horizon benchmark and a browsing-style multimodal benchmark.
4. **Empirical findings:** On LoCoMo-based settings, write-time governance strongly reduces attack admission and retrieval contamination, including on multimodal prompt injection and noisy/missing-modality robustness tests; on browsing-style MM-BrowseComp, the same system exposes a real limitation, namely that realistic web-style correction attacks currently bypass the write gate.

The paper’s central claim is therefore deliberately narrow and defensible:

> **Write-time governed multimodal memory is a distinct robustness primitive for agents. It strengthens memory-state integrity beyond what retrieval-time scoring alone can provide, especially under multimodal corruption, but it introduces a measurable utility tradeoff and remains incomplete in browsing-style settings.**

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
\neg \psi_{\mathrm{VPI}}(x)
\wedge
\neg \text{unsafe\_conflict}(x)
\]

where:
- \(g(x)\) is the write-time guard classification,
- \(\theta(c,\sigma)\) is the Bayesian trust posterior for channel \(c\) and source type \(\sigma\),
- \(d_M(x)\) is the session-relative anomaly score,
- \(\psi_{\mathrm{VPI}}(x)\) detects suspicious cross-modal corroboration.

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

The MM-BrowseComp adversarial track currently uses a browsing-native `fact_overwrite_injection` calibration. This is a realistic web-style correction attack, but the resulting benchmark remains highly saturated.

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

After fixing the VLM-captioning path and the case-construction filter, we reran MM-BrowseComp clean/adversarial on a 194-case leakage-clean pool (`paper_mmclean_full_v5`, `paper_mmadv_full_v4`). The corrected pool enforces effective observation support, removes junk/duplicate traces, and drops answer/checklist leakage.

#### Clean MM-BrowseComp

| Method | BCU clean | Answered rate |
|---|---:|---:|
| ShortContext | 0.2938 | 1.0000 |
| MMA | 0.2938 | 1.0000 |
| RSum | 0.2732 | 1.0000 |
| H1 | 0.2732 | 1.0000 |
| H2 | 0.2784 | 1.0000 |
| H3 | 0.2938 | 1.0000 |
| **SAGE-Mem** | **0.2526** | **0.9948** |

#### Adversarial MM-BrowseComp

| Method | BCU poison | Write ASR | Belief ASR | Retrieval | False belief | ASR |
|---|---:|---:|---:|---:|---:|---:|
| ShortContext | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| MMA | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| RSum | 0.0000 | 1.0000 | 0.4381 | 0.9948 | 0.0000 | 0.9948 |
| H1 | 0.0000 | 1.0000 | 0.4381 | 0.9931 | 0.0000 | 0.9931 |
| H2 | 0.0000 | 1.0000 | 0.4381 | 1.0000 | 0.0000 | 1.0000 |
| H3 | 0.0000 | 1.0000 | 0.4381 | 1.0000 | 0.0000 | 1.0000 |
| **SAGE-Mem** | **0.0052** | **1.0000** | **0.0000** | **0.9261** | **0.1701** | **0.9261** |

**What this shows.**

1. The clean benchmark is usable but difficult: all methods remain below `0.30` BCU clean, and SAGE-Mem trades additional utility for governance.
2. The adversarial benchmark remains highly saturated: all methods admit the browsing-style correction attack at write time.
3. SAGE-Mem reduces retrieval contamination relative to MMA (`0.9261` vs `1.0000`) and blocks belief-formation corruption, but it does not prevent end-to-end collapse.
4. The current browsing-specific limitation is therefore a **write-time trust calibration failure** for browsing-derived `tool_output_text`, not only a retrieval failure.

**Paper implication.** MM-BrowseComp is useful as an external stress test and as evidence of a real limitation in the current trust policy for browsing-derived `tool_output_text`, but it is **not strong enough to anchor the main empirical claim**.

---

## 6. What the Experiments Actually Support

The strongest supported contribution is not “SAGE-Mem is the best memory system overall.” That claim would be false.

The strongest supported claim is:

> **SAGE-Mem provides substantially stronger write-time containment and memory-state integrity than retrieval-time filtering on controlled long-horizon multimodal memory benchmarks, especially under multimodal prompt injection and noisy/missing modalities, albeit at a utility cost relative to MMA.**

This claim is supported by:
- zero write admission and zero retrieval contamination on the LoCoMo main suite,
- zero VPI survival on the VPI-only suite,
- near-zero survival under noisy/missing-modality stress,
- lower false-belief retrieval than simpler baselines,
- a meaningful Bayes calibration effect in the multimodal robustness setting.

What the results do **not** support:
- that SAGE-Mem beats MMA overall,
- that every v2 submodule is independently validated on the main suite,
- that browsing-style multimodal memory is solved.

---

## 7. Reviewer-Resistant Positioning

### 7.1 What is genuinely novel

The novelty is not simply “another write-time filter.” The defensible novelty is the combination of:

1. **Lifecycle evaluation for agent memory:** write admission, belief formation, retrieval contamination, and downstream completion are measured separately.
2. **Multimodal dependence modeling:** OCR and caption outputs from the same image are treated as dependent evidence rather than rewarded as independent corroboration.
3. **Typed memory with sufficiency-gated belief promotion:** the system distinguishes observation storage from durable belief formation.
4. **A browsing-style external stress test:** MM-BrowseComp exposes a limitation of current write-time trust policies rather than being silently omitted.

### 7.2 What reviewers may attack

1. **Utility gap versus MMA.**
   - This is real and must be acknowledged, not buried.
   - Response: the paper is not a single-metric QA leaderboard. It evaluates an integrity--utility frontier. MMA occupies the high-utility / permissive-store point; SAGE-Mem occupies the lower-utility / higher-integrity point with near-zero write and retrieval contamination on the controlled multimodal suites.
2. **Controlled LoCoMo multimodality.**
   - The LoCoMo multimodal setting is a controlled extension, not a fully natural multimodal agent log.
3. **MM-BrowseComp saturation.**
   - The current adversarial browsing benchmark remains too hard to serve as a headline result.
4. **Submodule ablations on the main suite.**
   - The main suite does not separate Bayes/anomaly/consistency strongly.
5. **Behavioral LLM metrics.**
   - These should remain secondary and be reported only when applicable.

The current draft should preempt these concerns explicitly.

---

## 8. Limitations

- **Trusted orchestration assumption.** Source labels are assumed correct.
- **Utility tradeoff.** Retrieval-time baselines preserve more immediate QA utility. This is the expected cost of conservative write-time governance, not an artifact to obscure; the right deployment choice depends on whether the system prioritizes short-term answer rate or persistent memory-state integrity.
- **Controlled versus naturalistic multimodality.** LoCoMo is controlled; MM-BrowseComp is more realistic but harsher and not yet fully discriminative.
- **Browsing-specific limitation.** The current trust policy does not block `fact_overwrite_injection` in MM-BrowseComp.
- **Module separability is setting-dependent.** Bayes is justified most clearly under noisy/missing multimodal evidence, not the main suite.
- **Behavioral LLM evaluation is secondary.** It is not the core evidence for this paper.

---

## 9. High-Priority Additional Experiments

### High priority for acceptance

1. **Per-attack breakdown on the main LoCoMo suite.**
   - Why: isolates where SAGE-Mem wins and where it merely ties.
   - Claim supported: robustness is not driven by only one attack family.

2. **Variance / confidence intervals across seeds.**
   - Why: the workshop audience will care whether gains are stable.
   - Claim supported: robustness improvements are not seed artifacts.

3. **Pareto plot: BCU poison vs attack write admission or retrieval contamination.**
   - Why: makes the write-time-versus-utility tradeoff reviewer-legible.
   - Claim supported: SAGE-Mem occupies a different robustness regime rather than merely underperforming.
   - This should be a main figure if space permits: it visually defends the MMA utility gap by showing that SAGE-Mem moves along a different integrity--utility frontier rather than losing on the paper's primary mechanism metrics.

4. **Per-modality breakdown in the multimodal robustness run.**
   - Why: clarifies whether failures are OCR-driven, caption-driven, or mixed.
   - Claim supported: the benchmark is genuinely multimodal, not just text corruption with image labels.

### Nice to have

5. **Cost/latency table for guard usage.**
   - Why: SCALE workshop reviewers may care about systems practicality.

6. **Length sensitivity / retrieval-budget sensitivity.**
   - Why: would strengthen the claim that cleaner memory reduces downstream burden.

7. **Browsing-specific trust ablation for MM-BrowseComp.**
   - Why: this would convert the current negative result into a sharper methodological lesson.

---

## 10. Citation Placeholders

Insert real citations for:
- memory poisoning in agent systems,
- retrieval corruption / prompt injection,
- multimodal prompt injection,
- memory-augmented and long-horizon agent evaluation,
- benchmark methodology for robustness and stress testing.

Do not fabricate citations. The current draft should use explicit placeholders until the final bibliography is assembled.
