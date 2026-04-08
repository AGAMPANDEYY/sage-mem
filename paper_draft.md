# SAGE-Mem: Source-Attested Guarded Episodic Memory for Multimodal Long-Horizon Agents

*ICML 2026 SCALE Workshop Submission Draft*

---

## Abstract

Persistent memory is a prerequisite for long-horizon agents, but it is also a durable attack surface: a single adversarial observation written into memory can survive consolidation and be retrieved as trusted experience indefinitely. Existing defenses for memory poisoning largely operate at retrieval time and therefore cannot stop adversarial content from entering the memory store. We present **SAGE-Mem**, a write-time defense for multimodal long-horizon agents that (1) filters observations before they enter planning memory, (2) enforces monotone trust under consolidation and rewriting, and (3) detects false cross-modal corroboration in which OCR and VLM outputs from the same image jointly reinforce an adversarial directive. SAGE-Mem combines calibrated Bayesian channel trust, a session-relative Mahalanobis anomaly detector, a bounded multi-turn consistency graph, and a selective LLM guard at the write boundary. We evaluate on two complementary settings: a controlled long-horizon benchmark built on LoCoMo-10 with multimodal adversarial injections, and an adversarial extension of MM-BrowseComp constructed through observation-trace injection. The central finding is that multimodal agreement is not always evidence of truth: under Visual Prompt Injection, naive multi-source trust can amplify attack success, whereas SAGE-Mem treats directive agreement across visual channels from the same source as an attack signal. This positions write-time multimodal memory governance as a necessary defense primitive for agentic memory systems.

---

## 1. Introduction

Long-horizon agents that maintain persistent memory across sessions face a threat that retrieval-time defenses cannot address: adversarial observations can be written into memory during normal operation and survive there indefinitely. Once written, a poisoned item is indistinguishable from a legitimate memory at retrieval time if it carries a plausible trust score.

Prior work establishes that agent memory can be poisoned [AgentPoison 2024, MINJA 2025], that a dedicated guard model provides protection against direct instruction injection [GuardAgent 2024], and that multimodal inputs introduce additional attack surface [One Pic is All it Takes 2025]. These works treat memory poisoning as a retrieval-time or model-level problem. We focus on the write boundary: what is allowed to enter memory, and under what conditions.

We make three contributions:

1. **Write-time multimodal memory defense.** We formulate adversarial memory defense as a write-boundary problem and introduce a layered mechanism that combines source-conditioned trust, anomaly detection, consistency tracking, and guarded admission into persistent planning memory.

2. **Visual Prompt Injection (VPI).** We introduce a new multimodal attack in which both OCR and VLM caption channels derived from the same adversarially crafted image carry directive content, thereby creating false corroboration in systems that reward apparent source agreement.

3. **Evaluation for agentic multimodal memory robustness.** We evaluate SAGE-Mem on a controlled long-horizon multimodal setting and a browsing-style multimodal benchmark with adversarial observation traces, and we use a behavioral LLM judge to measure attack success beyond surface-form regex matches.

---

## 2. Threat Model

### 2.1 Agent and Memory Model

We consider a long-horizon agent that reads observations from heterogeneous channels (tool outputs, OCR-extracted image text, VLM-generated captions, web-fetched documents, user messages) and writes them to a persistent episodic memory store. The memory store is queried at each step; retrieved items form part of the agent's planning context.

Each memory write carries a `source_type` label (`user`, `tool_output_text`, `ocr_text`, `vision_caption`, `self_summary`, `tool_echo`) assigned by the orchestration layer, and a `channel_id` identifying the specific data source.

### 2.2 Trust Boundary Assumption

**The orchestration layer is trusted.** We assume `source_type` and `channel_id` labels are correctly assigned by the agent framework. An adversary who can compromise the orchestrator and relabel arbitrary text as `source_type="user"` is outside our threat model.

This is the standard trust boundary in agent memory security (MemTrust 2026, GuardAgent 2024) and corresponds to the realistic deployment model where the orchestrator is part of the trusted computing base. Source verification at the cryptographic layer is orthogonal to this work and noted as future work.

### 2.3 Attacker Capabilities

The attacker can:
- Control the *content* of any observation the agent reads from an external channel (web page, OCR'd document, tool response, crafted image).
- Inject observations at any point in the agent's session.
- Craft images whose OCR and VLM outputs both carry adversarial payloads (Visual Prompt Injection).
- Pose as a user submitting a correction (semantic mimicry).

The attacker **cannot**:
- Relabel their injection with an arbitrary `source_type`.
- Modify memory items already written.
- Access the memory store directly.

### 2.4 Attack Taxonomy

| Attack | Channel | Mechanism |
|---|---|---|
| Semantic mimicry | `user` | Poses as a user correction; source trust = 1.0 |
| Constructor launder | `self_summary` | Injects directive into a summary to survive consolidation |
| Label gaming | `tool_echo` | Claims elevated provenance via `claimed_source` field |
| OCR injection | `ocr_text` | Embeds directive in OCR-extracted text |
| Vision caption injection | `vision_caption` | Embeds directive in VLM caption |
| **Visual Prompt Injection** | `ocr_text` + `vision_caption` | Adversarial image produces directive from both channels (cross-modal corroboration) |
| Adaptive NL evasion | `tool_output_text` | Natural-language directive; no regex-catchable markers |
| Buried payload | inherited | Directive spliced inside legitimate document text |

**Semantic mimicry is unblockable by design.** Any system that accepts user messages must accept user corrections. We handle it via `CorrectionPlausibilityScorer` (plausibility = 0.50 × grounding + 0.35 × specificity − 0.15 × frequency penalty), which detects implausibly frequent or ungrounded corrections but cannot block a genuine user acting adversarially. This is an explicit scope limitation, not a bug.

---

## 3. SAGE-Mem Defense Architecture

### 3.1 Write-Time Gate

For a candidate observation $x$ with source type $\sigma$ and channel $c$, the write decision is:

$$
\text{accept}(x) = \begin{cases}
0 & \text{if } g(x) \in \{\text{DIRECTIVE, METADATA}\} \\
0 & \text{if } \theta(c,\sigma) < \tau_t \\
0 & \text{if } d_M(x) > \xi_t \quad \text{(anomaly)} \\
0 & \text{if } \psi_\text{VPI}(x, \mathcal{G}) = 1 \quad \text{(cross-modal)} \\
0 & \text{if } \text{conflict}(x, \hat{x}) \wedge t(x) \le t(\hat{x}) \\
1 & \text{otherwise}
\end{cases}$$

where:
- $g(x)$ is the multi-agent guard classification (Skeptic + Advocate ensemble)
- $\theta(c, \sigma) = \mathbb{E}[\text{Beta}(\alpha_{c,\sigma}, \beta_{c,\sigma})]$ is the Bayesian channel trust posterior
- $\tau_t$ is the reactive write threshold
- $\lambda$ is the reactive tightening rate; $n_q$ is the quarantine count observed in the current episode
- $d_M(x)$ is the Mahalanobis distance from the session's benign write distribution
- $\xi_t$ is the self-calibrating anomaly threshold (90th percentile of warmup scores)
- $\psi_\text{VPI}(x, \mathcal{G})$ is the cross-modal corroboration flag (Section 3.3)

### 3.2 Monotone Derived Trust

Any item $y$ derived from parents $P(y)$ satisfies:

$$t(y) \le \min\!\left(t_\text{base}(y),\; \gamma \cdot \min_{p \in P(y)} t(p)\right)$$

with chain decay $\gamma \in (0,1]$ and type-level caps: $t(\text{self\_summary}) \le c_s$, $t(\text{tool\_echo}) \le c_e$ (both from config, not hardcoded). This prevents trust laundering: a high-trust summary cannot be generated from low-trust parents.

### 3.3 Cross-Modal Consistency Defense (Visual Prompt Injection)

Visual Prompt Injection exploits the fact that both OCR and VLM pipelines process the same image. An adversarial image embeds directive text that both systems transcribe, creating two "independent" channels that appear to corroborate each other. A multi-source trust system (e.g., MonotoneProvenanceLedger's independence bonus) would *increase* trust in this case.

We detect VPI via two signals:

**Visual source plausibility:** OCR and VLM outputs should contain observable facts (quantities, names, descriptions), not meta-instructions about memory or trust. We apply a pattern classifier:

$$\psi_\text{plaus}(x, \sigma) = \begin{cases} 1 & \text{if } \sigma \in \{\text{ocr\_text}, \text{vision\_caption}\} \wedge x \text{ contains directive language} \\ 0 & \text{otherwise} \end{cases}$$

**False corroboration detection:** When two visual writes share the same `observation_group` (same image source) and both are flagged by $\psi_\text{plaus}$, this is cross-modal corroboration — a strong attack signal:

$$\psi_\text{VPI}(x, \mathcal{G}) = \psi_\text{plaus}(x, \sigma) \vee \left(\exists x' \in \mathcal{G} : \text{obs\_group}(x') = \text{obs\_group}(x) \wedge \psi_\text{plaus}(x', \sigma') \right)$$

Both channels are routed to audit partition. Benign multimodal observations (factual descriptions, numbers, labels) pass through unaffected.

### 3.4 Multi-Turn Consistency Graph

The consistency graph $G = (V, E)$ tracks CONFIRMS / CONTRADICTS / UPDATES edges across all writes in the current session. Nodes are bounded to a sliding window of 128 (LRU eviction), giving $O(128)$ comparisons per write rather than $O(N)$.

Channel consistency score for channel $c$:
$$\text{score}(c) = 1 - \frac{|\{(u,v) \in E : \text{rel}(u,v) = \text{CONTRADICTS} \wedge (u \in c \vee v \in c)\}|}{|V_c| \cdot (|V_c| - 1) / 2}$$

When $\text{score}(c) < 0.5$, the Bayesian posterior for channel $c$ is updated with a failure observation, tightening the reactive threshold for future writes from that channel.

### 3.5 Bayesian Channel Trust

Per-channel trust uses a Beta-Bernoulli conjugate model. For channel $c$ with source type $\sigma$:

$$\alpha_{c,\sigma} \leftarrow \alpha_{c,\sigma} + \mathbf{1}[\text{correct}], \quad \beta_{c,\sigma} \leftarrow \beta_{c,\sigma} + \mathbf{1}[\neg\text{correct}]$$

$$\theta(c, \sigma) = \frac{\alpha_{c,\sigma}}{\alpha_{c,\sigma} + \beta_{c,\sigma}}$$

The reactive write threshold tightens after attack detections:
$$\tau_t = \min\!\left(1,\; \tau_0 \cdot e^{\lambda \cdot n_q}\right)$$

All priors $(\alpha_0, \beta_0)$ are declared in `configs/default_trust_config.json`. No scalar is hardcoded in Python.

---

## 4. Experimental Setup

### 4.1 Benchmarks

**LoCoMo-10 + multimodal adversarial extension** (10 conversations, 5,882 turns, 630 QA pairs). We use LoCoMo as the controlled long-horizon memory benchmark and extend it with multimodal observation writes and adversarial injections. We evaluate on three splits: clean (no attacks), poisoned (attacks injected at random positions), and poisoned\_cross\_topic (attacks in an earlier topic, evaluated on a later topic). This setting is controlled rather than fully naturalistic, but it allows causal analysis of memory poisoning over long horizons.

**Adversarially augmented MM-BrowseComp** (73 cases with live-fetched observation traces). We construct an adversarial extension of MM-BrowseComp by injecting attacks into memory-ingestion traces rather than altering the underlying task labels or gold answers. This benchmark provides the higher-realism multimodal setting. Questions are fundamentally visual; vision augmentation (VLM-extracted facts from question images) is applied, and attacks are injected into the corresponding observation traces.

### 4.2 Metrics

- **BCU (Benign Completion Under Attack)** = answer\_consistent × (1 − attack\_survived). Primary metric: higher is better.
- **ASR (Attack Success Rate)** = fraction of poisoned retrievals containing adversarial content. Lower is better.
- **ASR\_behavioral** = LLM-judge ASR (BehavioralAttackJudge); more accurate than regex matching, adaptive-attack robust. Reported alongside string-based ASR when `--llm-eval` is set.
- **Conflict quarantine / case** = write-gate interventions per case.
- **DMC (Derived Memory Corruption)** = planning items that descend from attack lineage through summarization.

### 4.3 Attack Injection

All attack positions are sampled uniformly at random (`--position-mode random`) to prevent positional overfitting. Prior work with fixed injection positions (e.g., always at turn 2) may report artificially optimistic or pessimistic results depending on the defense's consolidation schedule. We report results across seeds {0, 1, 2}.

Stateful trust and anomaly modules are re-instantiated per evaluation case. This avoids leakage of posterior trust or anomaly baselines across conversations, which would otherwise invalidate long-horizon security evaluation.

### 4.4 Conditions

| Condition | Role in the study |
|---|---|
| ShortContext | No persistent memory |
| MMA retrieve-time | Retrieve-time reliability scoring only; no write defense |
| H1 ConstructorGuard | Write-time quarantine + guarded consolidation |
| H2 MonotoneLedger | Provenance-monotone trust propagation |
| H3 ActionFirewall | Action-time corroboration requirement |
| **SAGE-Mem** | Full write-time multimodal defense: Bayesian trust + anomaly detection + consistency graph + VPI defense |

We treat the proposed system as a single method: **SAGE-Mem** refers to the full v2 architecture throughout the paper. H1-H3 are not competing full systems; they are mechanism-level ablations that isolate the contribution of particular design choices.

**What we do not compare against externally:** General-purpose memory systems (mem0, Letta, Hindsight) do not provide adversarial write-time memory defense. Their inclusion would demonstrate attack feasibility, not relative security performance. The appropriate reference baseline is MMA, which represents retrieve-time defense without write-time governance.

---

## 5. Results (to be filled after full runs)

### 5.1 Main LoCoMo Table

*(Run: `python run_eval.py --sage-v2 --seeds 0 1 2 --out results/sagememv2_main.json`)*

| Condition | BCU clean | BCU poison | ASR poison | ASR\_behavioral | DMC rate | Write quarantine / case | Conflict quarantine / case | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ShortContext | | | | | | | | |
| MMA retrieve-time | | | | | | | | |
| H1 ConstructorGuard | | | | | | | | |
| H2 MonotoneLedger | | | | | | | | |
| H3 ActionFirewall | | | | | | | | |
| **SAGE-Mem** | | | | | | | | |

**Reading:** ASR\_behavioral is the primary attack-success metric. DMC measures whether an attack survives not only as a retrieved raw item, but as corruption of derived planning memory through summarization or rewriting.

### 5.2 Visual Prompt Injection

*(Run: `python run_eval.py --attacks visual_prompt_injection --sage-v2 --disable-cross-topic`)*

| Condition | BCU clean | ASR (VPI) | DMC rate | Cross-modal quarantine / case |
|---|---:|---:|---:|
| MMA retrieve-time | | | |
| H1 ConstructorGuard | | | |
| MonotoneLedger | | | |
| **SAGE-Mem** | | | |

**Expected finding:** multi-source trust can be exploited by VPI, because OCR and caption channels derived from the same image are not conditionally independent. SAGE-Mem inverts this failure mode: agreement on directive content across linked visual channels becomes a quarantine trigger rather than a trust bonus.

### 5.3 Per-Attack Breakdown

*(Run individual attacks with `--attacks <name>` to produce per-family rows)*

| Attack family | MMA ASR | H1 ASR | H2 ASR | H3 ASR | SAGE-Mem ASR |
|---|---:|---:|---:|---:|---:|
| Constructor launder | | | | | |
| Label gaming | | | | | |
| OCR injection | | | | | |
| Vision caption injection | | | | | |
| Visual Prompt Injection | | | | | |
| Adaptive NL evasion | | | | | |
| Buried payload | | | | | |

`Semantic mimicry` should be reported separately as an explicit threat-model limitation rather than folded into the main ASR claim, since authenticated user turns are intentionally treated as a trust anchor.

### 5.4 Adversarially Augmented MM-BrowseComp

*(Run with `--run-mm-browsecomp` using observation-trace-augmented cases)*

| Condition | BCU clean | BCU poison | ASR poison | ASR\_behavioral | Write quarantine / case | n |
|---|---:|---:|---:|---:|---:|---:|
| MMA retrieve-time | | | | | | |
| H1 ConstructorGuard | | | | | | |
| H2 MonotoneLedger | | | | | | |
| H3 ActionFirewall | | | | | | |
| **SAGE-Mem** | | | | | | |

This table supports the external-validity claim: the defense is not only effective in a controlled synthetic extension, but also in a browsing-style multimodal memory setting with realistic observation traces.
### 5.5 Overhead

*(Measure with `time` and token-count logging)*

| Tier | Components active | Latency vs. unguarded | Cost per 1k writes |
|---|---|---|---|
| Regex-only | Trust gate + anomaly + consistency graph | ~1.1x | $0 |
| + LLM guard (Tier 1) | + MultiAgentGuard (2 API calls / write) | ~Xx | ~$Y |
| + Tier 2 escalation | + Tier 2 on disagreement (~Z% of writes) | ~Xx | ~$Y |

**Claim to make:** The anomaly detector, consistency graph, and Bayesian trust update add negligible overhead (no API calls). The LLM guard is the dominant cost and is applied selectively. Report the per-tier breakdown honestly rather than a single "1.2x" claim.

---

## 6. Novelty Claim (Honest Version)

SAGE-Mem is **not** the first write-time defense. The defensible novelty is:

1. **Visual Prompt Injection** as a new attack class that no prior work benchmarks or defends against. VPI is qualitatively different from prior multimodal attacks because it exploits multi-source consensus rather than a single channel.

2. **Cross-modal consistency defense** that detects false corroboration — a defense mechanism with no prior instantiation in the agent memory literature.

3. **End-to-end calibration** with no hardcoded scalars: all thresholds are data-driven (Bayesian posterior, conformal prediction, percentile-based anomaly threshold).

4. **Behavioral evaluation** via LLM judge, replacing circular regex-based ASR detection.

5. **Bounded consistency graph** (O(128) sliding window) as a practical multi-turn coherence mechanism for long-horizon agents.

The correct workshop framing is: *we extend write-time memory defense to multimodal agentic memory, introduce VPI as a new cross-modal threat, and show that naive multi-source trust can amplify rather than mitigate multimodal poisoning.*

---

## 7. Limitations

- **Source attribution is assumed correct.** A compromised orchestrator that relabels arbitrary text as `source_type="user"` defeats all channel-based trust. Cryptographic source attestation is future work.
- **Semantic mimicry is unblockable.** Any system that must accept user corrections cannot fully block a malicious user.
- **VLM visual extraction is not evaluated with end-to-end adversarial image generation.** VPI is modeled at the memory-ingestion level; a full perceptual robustness study would require actual adversarial images passed through OCR and VLM pipelines.
- **Controlled versus naturalistic multimodality.** The LoCoMo extension is a controlled stress test rather than a native multimodal interaction log; MM-BrowseComp provides the realism anchor, but neither benchmark fully covers production multimodal agents.
- **Potential utility-robustness tradeoff.** Any write-time defense can reduce benign recall if thresholds are too conservative. This is why BCU-clean, BCU-poison, and per-case quarantine statistics must all be reported.
- **LLM guard cost.** The MultiAgentGuard makes 2 API calls per write (potentially 3 on escalation). This is not 1.2x overhead; it is a significant per-write cost that must be reported per-tier.

---

## References (partial)

- AgentPoison (2024): Long-term memory as a persistent attack surface
- MINJA (2025): Query-only memory injection without direct database access
- Zombie Agents (2026): Self-evolving memory reinforces single malicious write
- A-MemGuard (2025): Proactive defense parallel; closest prior work
- GuardAgent (2024): Dedicated guard model justification
- MemTrust (2026): Zero-trust write philosophy
- One Pic is All it Takes (2025): Single poisoned image corrupts visual retrieval
- Angelopoulos & Bates (2022): Conformal prediction coverage guarantees
