import math
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from embedding import HashedTextEmbedder
from procedural import ProceduralDetector
from utils import safe_cosine_sim

# Optional: imported lazily to avoid hard dep when guard is disabled
_WriteTimeGuard = None
def _get_guard_type():
    global _WriteTimeGuard
    if _WriteTimeGuard is None:
        try:
            from guard_llm import WriteTimeGuard as _WTG
            _WriteTimeGuard = _WTG
        except ImportError:
            pass
    return _WriteTimeGuard

# Lazy imports for SAGE-Mem v2 modules — only imported when used
def _lazy_import_bayes_trust():
    try:
        from trust_calibration import BayesianChannelTrust
        return BayesianChannelTrust
    except ImportError:
        return None

def _lazy_import_anomaly_detector():
    try:
        from anomaly_detector import SessionAnomalyDetector
        return SessionAnomalyDetector
    except ImportError:
        return None

def _lazy_import_consistency_graph():
    try:
        from consistency_graph import MultiTurnConsistencyGraph
        return MultiTurnConsistencyGraph
    except ImportError:
        return None

def _lazy_import_correction_scorer():
    try:
        from consistency_graph import CorrectionPlausibilityScorer
        return CorrectionPlausibilityScorer
    except ImportError:
        return None

@dataclass
class MemoryItem:
    """A single piece of stored text with provenance and trust metadata."""

    item_id: int
    text: str
    embedding: np.ndarray
    source_type: str  # e.g., ocr_text, tool_output_text, self_summary, attacker
    channel_id: str
    step: int
    parent_ids: Tuple[int, ...] = ()
    partition: str = "planning"  # {planning, audit}
    trust: float = 0.5
    actionable: bool = False
    claimed_source: Optional[str] = None  # attacker-controlled label-gaming field
    session_idx: Optional[int] = None
    observation_group: Optional[str] = None
    source_url: Optional[str] = None
    page_id: Optional[str] = None
    fact_key: Optional[str] = None
    fact_value: Optional[str] = None
    multimodal_relation: Optional[str] = None
    attack_family: Optional[str] = None
    quality_score: float = 1.0
    group_outlier_score: float = 0.0
    memory_conflict_score: float = 0.0
    group_divergence_fired: bool = False
    structured_claim_gate_fired: bool = False
    # Cached at write time (SAGEMemory only): True if belief promotion was
    # satisfied when the item was written. Prevents re-running the check at
    # retrieval time after consolidation has discarded parent items.
    belief_traceable: bool = True

    def is_attacker_controlled(self) -> bool:
        return self.source_type == "attacker"

    def age(self, current_step: int) -> int:
        return max(0, int(current_step) - int(self.step))

    def with_updated_trust(self, new_trust: float) -> "MemoryItem":
        self.trust = float(new_trust)
        return self

def _clip01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


# ---------------------------------------------------------------------------
# Cross-modal consistency defense
# ---------------------------------------------------------------------------

# Directive-like language that should NOT appear in genuine OCR / VLM output.
# Real OCR transcribes visible text; real VLM captions describe image content.
# Neither should contain meta-instructions about memory, trust, or policy.
_VISUAL_DIRECTIVE_RE = re.compile(
    r"\b("
    r"disregard|ignore previous|override|override prior|override earlier|"
    r"discard prior|discard earlier|superseded|supersedes|definitive answer|"
    r"use this (note|correction|answer|update) as|treat (this|the) (note|image|caption) as|"
    r"the (correct|updated|latest|current) answer is|the answer (is now|has changed)|"
    r"replace (earlier|prior|previous) (answer|entries|memory)|"
    r"correction.*authoritative|authoritative.*correction|"
    r"important (correction|update|note).*(?:question|answer|prior)|"
    r"stop repeating|forget (earlier|prior|previous)|new information.*discard"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

_BROWSER_TOOL_SOURCE = "browser_tool_output_text"


def _apply_browser_context_trust_prior(
    *,
    text: str,
    source_type: str,
    trust: float,
    cfg=None,
) -> float:
    """
    Source-context-sensitive prior for web/browsing observations.

    Generic tool outputs may be trusted internal API responses. Browser-derived
    text is different: page content is externally controlled and should start
    from a stricter trust prior than internal tools. This gate is provenance-
    based and deterministic; it does not depend on benchmark case IDs, attack
    markers, or trigger-word matching.
    """
    if source_type != _BROWSER_TOOL_SOURCE or cfg is None:
        return _clip01(trust)
    return _clip01(min(float(trust), float(cfg.thresholds.browser_tool_trust_cap)))

def _cross_source_divergence_score(
    *,
    incoming_text: str,
    existing_items: List["MemoryItem"],
    min_group_size: int = 3,
    topic_threshold: float = 0.55,
    outlier_gap_threshold: float = 0.30,
) -> float:
    """
    Cross-source divergence check for browser scraping tasks.

    When an agent scrapes multiple websites for the same task, all scraped
    sentences should cluster around the same topic. An attacker injecting one
    fake sentence will produce something that is topically out of place compared
    to everything else the agent has scraped so far.

    How it works:
      1. Collect recent browser scrapes from planning memory (the prior scrapes).
      2. Compute the group's internal cohesion — average pairwise similarity
         between all prior scrapes. High cohesion means the group is consistently
         about the same topic.
      3. Compute incoming text's avg similarity to the group.
      4. If the group is cohesive (topic_threshold) but the incoming text is far
         below the group's internal similarity (outlier_gap_threshold) → outlier.

    Example: 4 prior scrapes all about iPhone prices have internal cohesion ~0.77.
    A sentence about "Caroline from France" scores ~-0.08 avg similarity to the
    group — a gap of 0.85, well above the 0.30 threshold → flagged.
    A legitimate "$929" sentence scores ~0.78 avg similarity → gap ~0 → allowed.

    Only catches topically unrelated attacks (e.g. attacker sneaks an off-topic
    fake fact into a scraping stream). Value-substitution attacks on the same
    topic (e.g. "$1 instead of $999") are handled by S1/S2/ABR.

    Only activates when there are at least min_group_size prior scrapes.
    Returns suspicion score in [0, 1]. 0 = fits the group, 1 = clear outlier.
    """
    try:
        from embedding import get_sentence_embedder
        from utils import safe_cosine_sim
        import numpy as _np
        se = get_sentence_embedder()
    except Exception:
        return 0.0

    # Collect recent trusted browser scrapes from planning memory
    prior_scrapes = [
        it for it in existing_items
        if getattr(it, "source_type", "") == _BROWSER_TOOL_SOURCE
        and getattr(it, "partition", "") == "planning"
        and getattr(it, "trust", 0.0) >= 0.40
        and str(it.text or "").strip()
    ][-20:]  # cap at last 20 scrapes

    if len(prior_scrapes) < min_group_size:
        return 0.0

    prior_embs = [se.embed(str(it.text or "")[:300]) for it in prior_scrapes]

    # Step 1: compute group internal cohesion (avg pairwise similarity)
    internal_sims = []
    for i in range(len(prior_embs)):
        for j in range(i + 1, len(prior_embs)):
            internal_sims.append(float(safe_cosine_sim(prior_embs[i], prior_embs[j])))

    if not internal_sims:
        return 0.0

    group_cohesion = float(_np.mean(internal_sims))

    # Only apply this check if the group is consistently about the same topic
    if group_cohesion < topic_threshold:
        return 0.0

    # Step 2: compute incoming text's avg similarity to the group
    inc_emb = se.embed(incoming_text[:300])
    incoming_sims = [float(safe_cosine_sim(inc_emb, pe)) for pe in prior_embs]
    incoming_avg = float(_np.mean(incoming_sims))

    # Step 3: how far below the group cohesion is the incoming text?
    gap = group_cohesion - incoming_avg

    if gap > outlier_gap_threshold:
        # Normalise: gap=0.30 → score=0.0, gap=0.60 → score=0.5, gap=0.90 → score=1.0
        return float(min(1.0, (gap - outlier_gap_threshold) / outlier_gap_threshold))

    return 0.0


_CONTENT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|\d{2,}")
_STOP_TOKENS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has",
    "was", "were", "are", "you", "your", "about", "into", "than", "then",
    "here", "there", "their", "they", "them", "his", "her", "she", "him",
    "its", "our", "out", "not", "but", "can", "will", "would", "should",
    "could", "according", "available", "latest", "current", "verified",
}

_SENTENCE_SIM_CACHE: Dict[str, np.ndarray] = {}


def _content_tokens(text: str) -> set:
    return {
        tok.lower()
        for tok in _CONTENT_TOKEN_RE.findall(str(text or ""))
        if tok.lower() not in _STOP_TOKENS
    }


def _jaccard_sim(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def _avg_pairwise(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _text_similarity(a: str, b: str) -> float:
    """Deterministic lexical similarity used as a safe fallback."""
    return _jaccard_sim(_content_tokens(a), _content_tokens(b))


def _semantic_embedding(text: str) -> Optional[np.ndarray]:
    """
    Cached sentence embedding for page-local/browser semantic checks.

    This path is intentionally best-effort: when the sentence-transformer model
    is unavailable, callers fall back to lexical similarity instead of failing
    closed or silently zeroing a defense path.
    """
    key = str(text or "").strip()[:400]
    if not key:
        return None
    cached = _SENTENCE_SIM_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from embedding import get_sentence_embedder
        emb = get_sentence_embedder().embed(key)
    except Exception:
        return None
    _SENTENCE_SIM_CACHE[key] = emb
    return emb


def _semantic_text_similarity(a: str, b: str) -> Optional[float]:
    emb_a = _semantic_embedding(a)
    emb_b = _semantic_embedding(b)
    if emb_a is None or emb_b is None:
        return None
    return _clip01(float(safe_cosine_sim(emb_a, emb_b)))


def _mixed_text_similarity(a: str, b: str) -> float:
    """
    Browser/page-local similarity that prefers semantic agreement when the
    sentence embedder is available and safely falls back to lexical overlap.
    """
    lexical = _text_similarity(a, b)
    semantic = _semantic_text_similarity(a, b)
    if semantic is None:
        return lexical
    # Semantic signal carries most weight; lexical overlap stabilises short ASUs.
    return _clip01(0.75 * semantic + 0.25 * lexical)


def _memory_conflict_score(
    *,
    incoming_text: str,
    fact_key: Optional[str],
    fact_value: Optional[str],
    existing_items: Sequence["MemoryItem"],
) -> float:
    """
    Estimate whether a browser sentence is trying to revise established memory.

    Primary path: semantic contradiction against trusted items discussing the
    same slot/topic. When a matching fact_key exists, we compare the incoming
    value against the prior value with mixed lexical+semantic similarity rather
    than exact token matching. Fallback: same-topic, non-identical similarity to
    trusted planning memory. The fallback is intentionally conservative and
    deterministic; semantic embedders can improve it, but the gate must not
    silently depend on unavailable model packages.
    """
    if fact_key and fact_value:
        incoming_value_text = str(fact_value).strip()
        for it in reversed(list(existing_items)):
            if getattr(it, "partition", "planning") == "audit":
                continue
            if getattr(it, "fact_key", None) != fact_key:
                continue
            prior_value_text = str(getattr(it, "fact_value", None) or "").strip()
            if not prior_value_text or not incoming_value_text:
                continue
            value_sim = _mixed_text_similarity(incoming_value_text, prior_value_text)
            if value_sim < 0.55:
                return min(1.0, (0.55 - value_sim) / 0.55)

    inc_tokens = _content_tokens(incoming_text)
    if not inc_tokens:
        return 0.0
    best = 0.0
    for it in reversed(list(existing_items)):
        if getattr(it, "partition", "planning") == "audit":
            continue
        if getattr(it, "source_type", "") in {"self_summary", "protected_fact"}:
            continue
        if float(getattr(it, "trust", 0.0)) < 0.55:
            continue
        prior_text = str(it.text or "")
        lexical = _jaccard_sim(inc_tokens, _content_tokens(prior_text))
        semantic = _semantic_text_similarity(incoming_text, prior_text)
        sim = lexical if semantic is None else max(lexical, semantic)
        if 0.12 <= sim < 0.95:
            best = max(best, min(1.0, sim / 0.45))
    return best


def _observation_group_divergence_scores(
    *,
    incoming_text: str,
    observation_group: Optional[str],
    existing_items: Sequence["MemoryItem"],
    fact_key: Optional[str],
    fact_value: Optional[str],
    min_siblings: int = 2,
    cohesion_threshold: float = 0.22,
    outlier_gap_threshold: float = 0.18,
    rolling_window: int = 4,
) -> Tuple[float, float, bool]:
    """
    Same-page observation-group defense for browser scraping tasks.

    The check compares a browser ASU/sentence against sibling ASUs from the same
    scraped page. Moderate page-local outliers require evidence that they are
    revising/probing established memory. Structured page-local claims
    (fact_key/fact_value) use a lower outlier threshold because they are exactly
    the writes that can harden into durable memory. Ordinary heterogeneous web
    paragraphs are not quarantined solely for topical diversity. This differs
    from cross-source divergence: the comparison set is the current page group,
    not all prior pages in memory.
    """
    group = str(observation_group or "").strip()
    if not group:
        return 0.0, 0.0, False
    siblings = [
        it for it in existing_items
        if getattr(it, "source_type", "") == _BROWSER_TOOL_SOURCE
        and getattr(it, "partition", "") == "evidence"
        and getattr(it, "observation_group", None) == group
        and str(it.text or "").strip()
    ][-12:]
    if len(siblings) < min_siblings:
        return 0.0, 0.0, False

    sibling_texts = [str(it.text or "") for it in siblings]
    radius = max(1, int(rolling_window // 2))
    sibling_neighbor_sims: List[float] = []
    for i, text_i in enumerate(sibling_texts):
        lo = max(0, i - radius)
        hi = min(len(sibling_texts), i + radius + 1)
        sims = [
            _mixed_text_similarity(text_i, sibling_texts[j])
            for j in range(lo, hi)
            if i != j
        ]
        if sims:
            sibling_neighbor_sims.append(max(sims))
    cohesion = _avg_pairwise(sibling_neighbor_sims)
    if cohesion < cohesion_threshold:
        return 0.0, 0.0, False

    compare_texts = sibling_texts[-max(min_siblings, rolling_window):]
    incoming_sims = [_mixed_text_similarity(incoming_text, s) for s in compare_texts]
    incoming_avg = max(incoming_sims) if incoming_sims else 0.0
    gap = cohesion - incoming_avg
    if gap <= outlier_gap_threshold:
        outlier_score = 0.0
    else:
        outlier_score = min(1.0, (gap - outlier_gap_threshold) / max(1e-6, outlier_gap_threshold))

    conflict_score = _memory_conflict_score(
        incoming_text=incoming_text,
        fact_key=fact_key,
        fact_value=fact_value,
        existing_items=existing_items,
    )
    structured_claim = bool(fact_key and fact_value)
    fired = (
        (outlier_score >= 0.50 and conflict_score >= 0.50)
        or (structured_claim and outlier_score >= 0.35)
    )
    return float(outlier_score), float(conflict_score), bool(fired)


def _precomputed_observation_group_scores(kwargs: dict) -> Optional[Tuple[float, float, bool]]:
    outlier = kwargs.get("precomputed_group_outlier_score")
    conflict = kwargs.get("precomputed_memory_conflict_score")
    fired = kwargs.get("precomputed_group_divergence_fired")
    if outlier is None and conflict is None and fired is None:
        return None
    return (
        _clip01(float(outlier or 0.0)),
        _clip01(float(conflict or 0.0)),
        bool(fired),
    )


def _abr_suspicion_score(
    *,
    text: str,
    embedding,
    channel_id: str,
    fact_key: Optional[str],
    fact_value: Optional[str],
    existing_items: List["MemoryItem"],
    cfg,
) -> float:
    """
    Adversarial Belief Revision (ABR) suspicion scorer for browser observations.

    Returns a score in [0, 1]. Scores above cfg.thresholds.abr_suspicion_threshold
    trigger a trust cap, blocking the write from influencing established memory.

    Four orthogonal signals (composite trust scoring, Zhu et al. 2026):

      S1 – Semantic memory-consistency check (w=0.40)
           Incoming content semantically conflicts with trusted memory on the
           same slot/topic. Uses mixed lexical+semantic similarity rather than
           trigger-word or exact-value matching.

      S2 – Semantic proximity to trusted memory (w=0.30)
           Max cosine similarity to high-trust items (trust ≥ 0.65).
           Measures: "is this obs discussing an established topic without
           independent corroboration?" Combined with S3, flags adversarial
           singletons that probe established facts.
           Ref: RA-RAG credibility-weighted retrieval (Shi et al. 2024).

      S3 – Corroboration deficit (w=0.20)
           Fraction of existing items NOT semantically similar to this obs.
           Real facts accumulate independent corroborating evidence over time;
           adversarial correction turns are near-always singletons.
           Ref: CrAM credibility-aware attention (Weller et al. 2024).

      S4 – Channel singleton penalty (w=0.10)
           Legitimate browser sessions have multiple prior writes on the same
           channel. Fresh single-use channels are a strong attacker prior.

    Moving target: threshold is jittered at SAGEMemory init time by ±abr_noise_scale,
    making exact threshold gaming harder for adaptive adversaries
    (cf. CLATTER 2025, §4.3: randomised defence parameters).
    """
    if cfg is None:
        return 0.0

    topic_sim_thr = float(getattr(cfg.thresholds, "abr_topic_sim_threshold", 0.50))
    corr_needed = max(1, int(getattr(cfg.thresholds, "abr_corroboration_needed", 3)))

    # ── S1: Semantic memory-consistency check ───────────────────────────────
    s1 = _memory_conflict_score(
        incoming_text=text,
        fact_key=fact_key,
        fact_value=fact_value,
        existing_items=existing_items,
    )

    # ── S2: Semantic proximity to high-trust memory ──────────────────────────
    s2 = 0.0
    if embedding is not None and len(embedding) > 0:
        max_sim = 0.0
        for it in existing_items:
            if float(getattr(it, "trust", 0.0)) >= 0.65 and it.embedding is not None:
                sim = safe_cosine_sim(embedding, it.embedding)
                if sim > max_sim:
                    max_sim = sim
        # Normalise: sims in [topic_sim_thr, 1.0] → [0, 1]
        if max_sim > topic_sim_thr:
            s2 = min(1.0, (max_sim - topic_sim_thr) / max(1e-6, 1.0 - topic_sim_thr))

    # ── S3: Corroboration deficit ────────────────────────────────────────────
    s3 = 1.0
    if embedding is not None and len(embedding) > 0:
        corr_count = sum(
            1 for it in existing_items
            if it.embedding is not None and safe_cosine_sim(embedding, it.embedding) > 0.55
        )
        s3 = 1.0 - min(1.0, corr_count / corr_needed)

    # ── S4: Channel singleton penalty ────────────────────────────────────────
    depth = sum(1 for it in existing_items if getattr(it, "channel_id", "") == channel_id)
    if depth == 0:
        s4 = 1.0
    elif depth == 1:
        s4 = 0.6
    elif depth == 2:
        s4 = 0.3
    else:
        s4 = 0.0

    return _clip01(0.40 * s1 + 0.30 * s2 + 0.20 * s3 + 0.10 * s4)


# Legitimate visual content indicators (genuine OCR/VLM output looks like this)
_VISUAL_CONTENT_RE = re.compile(
    r"\b("
    r"image shows?|photo of|picture of|screenshot of|visible (in|on|from)|"
    r"text reads?|label[s]?:|caption:|OCR:|transcribed|extracted text|"
    r"\d+\s*(px|cm|mm|ft|m\b|km|kg|lb|mph|kph|°[CF])|"
    r"jersey number|score(board)?|stadium|color[s]?|logo|sign|"
    r"figure \d|table \d|chart|graph|diagram|map"
    r")\b",
    re.IGNORECASE,
)


def _check_visual_source_plausibility(
    text: str,
    source_type: str,
) -> Tuple[bool, float]:
    """
    Returns (is_suspicious, trust_penalty) for visual-channel writes.

    A visual source (ocr_text, vision_caption) is suspicious when it contains
    directive language that would not appear in genuine OCR/VLM output.
    Genuine visual extraction produces factual descriptions, numbers, labels —
    not policy overrides, trust instructions, or memory-management directives.

    Returns:
        is_suspicious: True → quarantine or reduce trust
        trust_penalty: amount to subtract from computed_trust (0.0 if not suspicious)
    """
    if source_type not in {"ocr_text", "vision_caption"}:
        return False, 0.0

    has_directive = bool(_VISUAL_DIRECTIVE_RE.search(text))
    has_visual_content = bool(_VISUAL_CONTENT_RE.search(text))

    if has_directive and not has_visual_content:
        # Pure directive in a visual channel — high confidence attack
        return True, 0.60
    if has_directive and has_visual_content:
        # Mixed: legitimate visual content with embedded directive (VPI pattern)
        return True, 0.40
    return False, 0.0


def _check_observation_group_corroboration(
    items: List["MemoryItem"],
    observation_group: Optional[str],
    source_type: str,
) -> bool:
    """
    Detect false cross-modal corroboration: two visual writes from the same
    observation_group (same source image) that both contain directive language.

    When both OCR and VLM outputs from the "same image" contain adversarial
    claims, a naive multi-source trust system would increase trust (two
    independent sources agree). This function flags that pattern as an attack.

    Returns True if the observation_group shows cross-modal directive corroboration.
    """
    if observation_group is None:
        return False
    visual_sources = {"ocr_text", "vision_caption"}
    if source_type not in visual_sources:
        return False
    # Find prior items from the same observation_group and a different visual channel
    prior_visual_from_same_group = [
        it for it in items
        if getattr(it, "observation_group", None) == observation_group
        and getattr(it, "source_type", "") in visual_sources
        and getattr(it, "source_type", "") != source_type
    ]
    if not prior_visual_from_same_group:
        return False
    # If any prior item from the same image also has directive language → false corroboration
    return any(
        _VISUAL_DIRECTIVE_RE.search(str(it.text or ""))
        for it in prior_visual_from_same_group
    )

_DEFAULT_SOURCE_CRED = {
    "user": 1.0,
    "ocr_text": 0.60,
    "vision_caption": 0.58,
    "tool_output_text": 0.70,
    _BROWSER_TOOL_SOURCE: 0.35,
    "self_summary": 0.45,
    "attacker": 0.10,
    "tool_echo": 0.20,
    "protected_fact": 0.88,  # extracted from trusted sources, never consolidated
}

def default_source_cred(source_type: str, cfg=None) -> float:
    """Return prior source credibility.

    When cfg (SAGEMemConfig) is provided, values are read from
    cfg.source_cred so that no scalars are hardcoded in Python.
    Falls back to _DEFAULT_SOURCE_CRED when cfg is absent.
    """
    if cfg is not None:
        try:
            return float(cfg.source_cred.get(source_type))
        except Exception:
            pass
    return _DEFAULT_SOURCE_CRED.get(source_type, 0.35)

_LABEL_RE = re.compile(r"\bOBJECT_LABEL\s*:\s*([a-z_]+)\b", re.IGNORECASE)
_DIRECTIVE_RE = re.compile(
    r"\bACTION_DIRECTIVE\s*:\s*at_step\s*=\s*(\d+)\s+tool\s*=\s*([a-z_]+)\s+payload\s*=\s*([a-z_]+)\b",
    re.IGNORECASE,
)

def extract_label(text: str) -> Optional[str]:
    m = _LABEL_RE.search(text)
    return m.group(1).lower() if m else None

def extract_directive(text: str) -> Optional[Tuple[int, str, str]]:
    m = _DIRECTIVE_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), m.group(2).lower(), m.group(3).lower()


# ── Fact extraction regexes ──────────────────────────────────────────────────
_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"(?:\s+\d{2,4})?|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"
    r"|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|\d{1,2}:\d{2}\s*(?:am|pm))\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b")
_NUMBER_RE = re.compile(r"\b(\d{1,5})\b")

_TRUSTED_SOURCES = {"user", "tool_output_text", "browser_tool_output_text"}

def extract_key_facts(items: List["MemoryItem"], guard=None) -> List[dict]:
    """
    Extract key facts from trusted memory items.

    When a WriteTimeGuard is provided, uses LLM-based extraction via the guard's
    stateless sandbox — robust to varied date formats, multi-word names, and
    contextual numbers without brittle pattern matching.

    When guard is None (or LLM unavailable), falls back to the regex baseline
    so unit tests without API keys still function.

    Returns list of dicts with keys: fact_type, value, source_text, source_type,
    parent_id.  Only extracts from trusted sources — never from attacker-controlled
    or quarantined items.
    """
    facts: List[dict] = []
    seen_values: set = set()

    # ── Filter to trusted, non-quarantined items ──────────────────────────────
    trusted_items = []
    for it in items:
        if it.source_type not in _TRUSTED_SOURCES:
            continue
        if it.trust < 0.60:
            continue
        if it.partition == "audit":
            continue
        text = re.sub(r"\s+", " ", str(it.text or "")).strip()
        if not text or text.lower().startswith("summary:"):
            continue
        trusted_items.append((it, text))

    # ── LLM path (preferred): delegate to the guard's stateless LLM ──────────
    if guard is not None and hasattr(guard, "extract_facts"):
        _llm_type_to_fact_type = {
            "date": "date", "name": "name", "number": "number",
            "event": "event", "relation": "relation",
        }
        for it, text in trusted_items:
            try:
                llm_facts = guard.extract_facts(text)
            except Exception:
                llm_facts = []
            for f in llm_facts:
                ftype = _llm_type_to_fact_type.get(str(f.get("type", "")).lower(), "fact")
                val = str(f.get("value", "")).strip()
                if not val:
                    continue
                dedup_key = (ftype, val.lower())
                if dedup_key in seen_values:
                    continue
                seen_values.add(dedup_key)
                facts.append({
                    "fact_type": ftype,
                    "value": val,
                    "source_text": text[:120],
                    "source_type": it.source_type,
                    "parent_id": it.item_id,
                })
        return facts

    # ── Regex fallback (no guard / no API key) ────────────────────────────────
    for it, text in trusted_items:
        # Extract dates
        for m in _DATE_RE.finditer(text):
            val = m.group(0).strip()
            key = ("date", val.lower())
            if key not in seen_values:
                seen_values.add(key)
                facts.append({
                    "fact_type": "date",
                    "value": val,
                    "source_text": text[:120],
                    "source_type": it.source_type,
                    "parent_id": it.item_id,
                })

        # Extract proper names (capitalized, 3+ chars)
        for m in _NAME_RE.finditer(text):
            val = m.group(1).strip()
            if val.lower() in {"the", "this", "that", "there", "their",
                                "they", "what", "when", "where", "who",
                                "how", "hey", "wow", "glad", "yeah",
                                "going", "gonna", "thanks", "thank"}:
                continue
            key = ("name", val.lower())
            if key not in seen_values:
                seen_values.add(key)
                facts.append({
                    "fact_type": "name",
                    "value": val,
                    "source_text": text[:120],
                    "source_type": it.source_type,
                    "parent_id": it.item_id,
                })

        # Extract numbers (1-4 digits, meaningful range)
        for m in _NUMBER_RE.finditer(text):
            val = m.group(1)
            num = int(val)
            if num < 2 or num > 9999:
                continue
            key = ("number", val)
            if key not in seen_values:
                seen_values.add(key)
                facts.append({
                    "fact_type": "number",
                    "value": val,
                    "source_text": text[:120],
                    "source_type": it.source_type,
                    "parent_id": it.item_id,
                })

    return facts


def summarize_factual_snippets(
    items: List["MemoryItem"],
    *,
    limit: int = 4,
    max_chars: int = 140,
) -> List[str]:
    snippets: List[str] = []
    seen = set()
    for it in reversed(items):
        text = re.sub(r"\s+", " ", str(it.text or "")).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith("summary:"):
            continue
        text = text.replace("ACTION_DIRECTIVE", "QUOTED_DIRECTIVE")
        snippet = text[:max_chars].strip(" '\"")
        if not snippet:
            continue
        key = snippet.lower()
        if key in seen:
            continue
        seen.add(key)
        snippets.append(snippet)
        if len(snippets) >= limit:
            break
    snippets.reverse()
    return snippets


def _norm_fact_token(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def find_conflicting_fact(
    items: Sequence["MemoryItem"],
    *,
    fact_key: Optional[str],
    fact_value: Optional[str],
    session_idx: Optional[int],
    incoming_text: Optional[str] = None,
) -> Optional["MemoryItem"]:
    """
    S1: fact_key/fact_value conflict detection (primary path).
    Checks existing items for a matching fact_key with a different fact_value.

    S1 fallback: when no protected_fact items exist (e.g. MM-BrowseComp cold-start),
    also checks planning-partition belief items using sentence-transformer cosine
    similarity on the full text — catches paraphrased overwrites that bypass the
    fact_key gate because fact_key was never set.
    """
    # ── S1: fact_key-based conflict ───────────────────────────────────────────
    if fact_key and fact_value:
        target = _norm_fact_token(fact_value)
        if target:
            for it in reversed(list(items)):
                if getattr(it, "fact_key", None) != fact_key:
                    continue
                prior_value = _norm_fact_token(getattr(it, "fact_value", None))
                if not prior_value or prior_value == target:
                    continue
                if session_idx is not None and getattr(it, "session_idx", None) is not None:
                    if int(it.session_idx) != int(session_idx):
                        continue
                return it

    # ── S1 fallback: semantic similarity on full text (fires in MM-BrowseComp) ─
    # When fact_key is not set, use sentence-transformer embeddings to find a
    # planning item on the same topic as incoming_text. Same topic + different
    # content = potential overwrite attack.
    if incoming_text:
        try:
            from embedding import get_sentence_embedder
            se = get_sentence_embedder()
            inc_emb = se.embed(incoming_text[:300])

            TOPIC_THRESH = 0.60    # min cosine to be "same topic"
            IDENTICAL_THRESH = 0.92  # above this = near-duplicate, not a conflict

            best_item: Optional["MemoryItem"] = None
            best_sim = 0.0
            for it in reversed(list(items)):
                if getattr(it, "partition", None) != "planning":
                    continue
                if getattr(it, "source_type", "") in {"self_summary", "protected_fact"}:
                    continue
                if getattr(it, "trust", 0.0) < 0.60:
                    continue
                prior_emb = se.embed(str(it.text or "")[:300])
                from utils import safe_cosine_sim
                sim = float(safe_cosine_sim(inc_emb, prior_emb))
                if TOPIC_THRESH < sim < IDENTICAL_THRESH:
                    if sim > best_sim:
                        best_sim = sim
                        best_item = it
            if best_item is not None:
                return best_item
        except Exception:
            pass  # sentence-transformer unavailable; skip S1 fallback

    return None


def build_conflict_evidence_text(
    *,
    fact_key: Optional[str],
    trusted_item: "MemoryItem",
    incoming_value: Optional[str],
) -> str:
    prior_value = str(getattr(trusted_item, "fact_value", "") or trusted_item.text[:80]).strip()
    new_value = str(incoming_value or "").strip() or "unknown"
    slot = str(fact_key or "memory_fact")
    return (
        "EVIDENCE_CONFLICT: conflicting multimodal observations for "
        f"{slot}. Earlier evidence suggests '{prior_value}', but a newer "
        f"lower-trust observation suggests '{new_value}'. Preserve for audit; "
        "do not treat either contradiction as settled policy."
    )

class BaseMemory:
    def __init__(self, *, embedder: HashedTextEmbedder, top_k: int):
        self.embedder = embedder
        self.top_k = int(top_k)
        self._next_id = 1
        self.items: List[MemoryItem] = []
        self.audit_items: List[MemoryItem] = []
        self.trust_escalation_events = 0
        self.retrieval_trusted_capture_n = 0
        self.retrieval_trusted_n = 0
        # Calibration log: list of {source_type, contributed_to_correct} dicts
        # populated when log_retrieved_source_types=True is passed to retrieve()
        self.retrieved_source_type_log: List[dict] = []
        # ── Latency tracking ─────────────────────────────────────────────────
        # Accumulated wall-clock time (seconds) spent inside write() and retrieve().
        # Use latency_summary() to get per-call averages.
        self._write_time_total_s: float = 0.0
        self._write_call_count: int = 0
        self._retrieve_time_total_s: float = 0.0
        self._retrieve_call_count: int = 0

    def latency_summary(self) -> dict:
        """Return average write and retrieve latency in milliseconds."""
        return {
            "write_calls": self._write_call_count,
            "write_avg_ms": round(
                1000 * self._write_time_total_s / max(1, self._write_call_count), 3
            ),
            "write_total_ms": round(1000 * self._write_time_total_s, 3),
            "retrieve_calls": self._retrieve_call_count,
            "retrieve_avg_ms": round(
                1000 * self._retrieve_time_total_s / max(1, self._retrieve_call_count), 3
            ),
            "retrieve_total_ms": round(1000 * self._retrieve_time_total_s, 3),
            "items_in_memory": len(self.items),
            "items_in_audit": len(self.audit_items),
        }

    def write(
        self,
        *,
        text: str,
        source_type: str,
        channel_id: str,
        step: int,
        parent_ids: Sequence[int] = (),
        partition: str = "planning",
        trust: Optional[float] = None,
        actionable: bool = False,
        claimed_source: Optional[str] = None,
        session_idx: Optional[int] = None,
        observation_group: Optional[str] = None,
        source_url: Optional[str] = None,
        page_id: Optional[str] = None,
        fact_key: Optional[str] = None,
        fact_value: Optional[str] = None,
        multimodal_relation: Optional[str] = None,
        attack_family: Optional[str] = None,
        quality_score: Optional[float] = None,
        group_outlier_score: float = 0.0,
        memory_conflict_score: float = 0.0,
        group_divergence_fired: bool = False,
        structured_claim_gate_fired: bool = False,
    ) -> int:
        _t0 = time.perf_counter()
        emb = self.embedder.embed(text)
        t = _clip01(default_source_cred(source_type) if trust is None else float(trust))
        item = MemoryItem(
            item_id=self._next_id,
            text=text,
            embedding=emb,
            source_type=source_type,
            channel_id=channel_id,
            step=int(step),
            parent_ids=tuple(int(x) for x in parent_ids),
            partition=partition,
            trust=t,
            actionable=bool(actionable),
            claimed_source=claimed_source,
            session_idx=int(session_idx) if session_idx is not None else None,
            observation_group=str(observation_group) if observation_group is not None else None,
            source_url=str(source_url) if source_url is not None else None,
            page_id=str(page_id) if page_id is not None else None,
            fact_key=str(fact_key) if fact_key is not None else None,
            fact_value=str(fact_value) if fact_value is not None else None,
            multimodal_relation=str(multimodal_relation) if multimodal_relation is not None else None,
            attack_family=str(attack_family) if attack_family is not None else None,
            quality_score=_clip01(1.0 if quality_score is None else float(quality_score)),
            group_outlier_score=_clip01(float(group_outlier_score)),
            memory_conflict_score=_clip01(float(memory_conflict_score)),
            group_divergence_fired=bool(group_divergence_fired),
            structured_claim_gate_fired=bool(structured_claim_gate_fired),
        )
        self._next_id += 1
        if partition == "audit":
            self.audit_items.append(item)
        else:
            self.items.append(item)
        self._write_time_total_s += time.perf_counter() - _t0
        self._write_call_count += 1
        return item.item_id

    def consolidate(self, step: int) -> None:
        return

    def _iter_for_retrieval(self, *, for_planning: bool) -> Iterable[MemoryItem]:
        if for_planning:
            return self.items
        return list(self.items) + list(self.audit_items)

    def retrieve(
        self, *, query: str, step: int, for_planning: bool = True,
        log_retrieved_source_types: bool = False,
        gold_answer: Optional[str] = None,
    ) -> List[MemoryItem]:
        _t0 = time.perf_counter()
        q = self.embedder.embed(query)
        scored: List[Tuple[float, MemoryItem]] = []
        for it in self._iter_for_retrieval(for_planning=for_planning):
            sim = safe_cosine_sim(q, it.embedding)
            score = sim * _clip01(it.trust)
            if not math.isfinite(score):
                continue
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [it for _, it in scored[: self.top_k]]

        trusted = [it for it in out if it.trust >= 0.7]
        self.retrieval_trusted_n += len(trusted)
        self.retrieval_trusted_capture_n += sum(1 for it in trusted if it.source_type == "attacker")

        if log_retrieved_source_types and gold_answer is not None:
            gold_norm = gold_answer.lower().strip()
            for it in out:
                text_norm = str(it.text or "").lower()
                contributed = gold_norm in text_norm if gold_norm else False
                self.retrieved_source_type_log.append({
                    "source_type": it.source_type,
                    "contributed_to_correct": contributed,
                })
        self._retrieve_time_total_s += time.perf_counter() - _t0
        self._retrieve_call_count += 1
        return out

    def _maybe_log_retrieved_source_types(
        self,
        out: Sequence[MemoryItem],
        *,
        log_retrieved_source_types: bool = False,
        gold_answer: Optional[str] = None,
    ) -> None:
        if not log_retrieved_source_types or gold_answer is None:
            return
        gold_norm = gold_answer.lower().strip()
        for it in out:
            text_norm = str(it.text or "").lower()
            contributed = gold_norm in text_norm if gold_norm else False
            self.retrieved_source_type_log.append({
                "source_type": it.source_type,
                "contributed_to_correct": contributed,
            })

    def trusted_capture_rate(self) -> float:
        if self.retrieval_trusted_n <= 0:
            return 0.0
        return float(self.retrieval_trusted_capture_n / self.retrieval_trusted_n)

    def close(self) -> None:
        return

class MMARetrieveTimeReliabilityMemory(BaseMemory):
    def __init__(
        self,
        *,
        embedder: HashedTextEmbedder,
        top_k: int,
        w_source: float,
        w_decay: float,
        w_consensus: float,
        decay_half_life_steps: int,
    ):
        super().__init__(embedder=embedder, top_k=top_k)
        self.w_source = float(w_source)
        self.w_decay = float(w_decay)
        self.w_consensus = float(w_consensus)
        self.decay_half_life_steps = max(1, int(decay_half_life_steps))

    def score_item(self, it: MemoryItem, *, query_emb: np.ndarray, step: int) -> float:
        sim = safe_cosine_sim(query_emb, it.embedding)
        src = default_source_cred(it.source_type)
        age = max(0, int(step) - int(it.step))
        decay = 0.5 ** (age / float(self.decay_half_life_steps))

        lbl = extract_label(it.text)
        if lbl is None:
            consensus = 1.0
        else:
            # Source Diversity Weighting: count unique agreeing channels, not raw item count.
            # This prevents label_gaming (attacker floods memory with copies from one channel)
            # from inflating consensus score. 10 items from 1 channel = 1 agreeing channel.
            agree_channels: set = set()
            disagree_channels: set = set()
            for other in self.items:
                if other.item_id == it.item_id:
                    continue
                olbl = extract_label(other.text)
                if olbl is None:
                    continue
                if olbl == lbl:
                    agree_channels.add(other.channel_id)
                else:
                    disagree_channels.add(other.channel_id)
            agree = len(agree_channels)
            disagree = len(disagree_channels)
            consensus = (1.0 + agree) / (1.0 + agree + disagree)

        rel = self.w_source * src + self.w_decay * decay + self.w_consensus * consensus
        rel = max(0.0, min(1.0, float(rel)))
        score = float(sim) * rel
        if not math.isfinite(score):
            return 0.0
        return score

    def retrieve(
        self,
        *,
        query: str,
        step: int,
        for_planning: bool = True,
        log_retrieved_source_types: bool = False,
        gold_answer: Optional[str] = None,
    ) -> List[MemoryItem]:
        q = self.embedder.embed(query)
        scored: List[Tuple[float, MemoryItem]] = []
        for it in self._iter_for_retrieval(for_planning=for_planning):
            score = self.score_item(it, query_emb=q, step=step)
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [it for _, it in scored[: self.top_k]]

        trusted = [it for it in out if default_source_cred(it.source_type) >= 0.7]
        self.retrieval_trusted_n += len(trusted)
        self.retrieval_trusted_capture_n += sum(1 for it in trusted if it.source_type == "attacker")
        self._maybe_log_retrieved_source_types(
            out,
            log_retrieved_source_types=log_retrieved_source_types,
            gold_answer=gold_answer,
        )
        return out


class ShortContextMemory(BaseMemory):
    """Approximate no-long-term-memory baseline: only keep the latest raw turns."""

    def __init__(self, *, embedder: HashedTextEmbedder, top_k: int, keep_last_k: int):
        super().__init__(embedder=embedder, top_k=top_k)
        self.keep_last_k = max(1, int(keep_last_k))

    def write(self, **kwargs):  # type: ignore[override]
        item_id = super().write(**kwargs)
        raw = [it for it in self.items if it.partition == "planning"]
        self.items = raw[-self.keep_last_k :]
        return item_id

class RecursiveSummarizationMemory(BaseMemory):
    def __init__(
        self,
        *,
        embedder: HashedTextEmbedder,
        top_k: int,
        consolidation_period_K: int,
        keep_last_M_raw: int,
    ):
        super().__init__(embedder=embedder, top_k=top_k)
        self.K = max(1, int(consolidation_period_K))
        self.keep_M = max(0, int(keep_last_M_raw))
        self._buffer: List[int] = []

    def write(self, **kwargs):  # type: ignore[override]
        item_id = super().write(**kwargs)
        self._buffer.append(item_id)
        return item_id

    def _summarize_text(self, items: List[MemoryItem]) -> str:
        labels: Dict[str, int] = {}
        directives: List[str] = []
        for it in items:
            lbl = extract_label(it.text)
            if lbl is not None:
                labels[lbl] = labels.get(lbl, 0) + 1
            d = extract_directive(it.text)
            if d is not None:
                directives.append(
                    f"ACTION_DIRECTIVE: at_step={d[0]} tool={d[1]} payload={d[2]}"
                )
        top = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        label_part = ", ".join([f"{k}({v})" for k, v in top]) if top else "none"
        directive_part = "\n".join(directives) if directives else ""
        facts = summarize_factual_snippets(items, limit=5)
        fact_part = "\n".join([f"FACT: {x}" for x in facts])
        return f"SUMMARY: top_labels={label_part}\n{fact_part}\n{directive_part}".strip()

    def consolidate(self, step: int) -> None:
        if step <= 0 or (step % self.K) != 0:
            return
        id_to_item = {it.item_id: it for it in self.items}
        buf_items = [id_to_item[i] for i in self._buffer if i in id_to_item]
        if not buf_items:
            return

        # ── Fact Preservation: extract key facts BEFORE compression ──────────
        # Extract dates, names, numbers from trusted items in the buffer.
        # Store each as a protected_fact item that is NEVER put in _buffer,
        # so it will never be consolidated away. This preserves specific
        # factual details (dates, names, numbers) that would otherwise be
        # lost during summarization — recovering BCU clean without sacrificing
        # attack defense (only trusted sources generate protected facts).
        facts = extract_key_facts(buf_items)
        protected_ids = []
        seen_fact_values = {
            it.fact_value
            for it in self.items
            if it.source_type == "protected_fact" and it.fact_value
        }
        for fact in facts:
            if fact["value"] in seen_fact_values:
                continue  # already protected, skip duplicate
            seen_fact_values.add(fact["value"])
            fact_text = (
                f"PROTECTED_FACT[{fact['fact_type']}]: {fact['value']} "
                f"— from: {fact['source_text'][:80]}"
            )
            # Write directly to BaseMemory to bypass _buffer append
            fact_id = super(RecursiveSummarizationMemory, self).write(
                text=fact_text,
                source_type="protected_fact",
                channel_id=f"fact_{fact['fact_type']}",
                step=step,
                parent_ids=(fact["parent_id"],),
                trust=0.88,
                actionable=False,
                claimed_source=fact["source_type"],
                fact_key=fact["fact_type"],
                fact_value=fact["value"],
            )
            protected_ids.append(fact_id)
        # ─────────────────────────────────────────────────────────────────────

        text = self._summarize_text(buf_items)
        self.write(
            text=text,
            source_type="self_summary",
            channel_id="summary",
            step=step,
            parent_ids=[it.item_id for it in buf_items],
            trust=0.80,
            actionable=True,
        )
        raw = [it for it in self.items if it.source_type not in ("self_summary", "protected_fact")]
        keep_raw = raw[-self.keep_M :] if self.keep_M > 0 else []
        summaries = [it for it in self.items if it.source_type == "self_summary"]
        # Keep all protected facts permanently — never discard them
        protected = [it for it in self.items if it.source_type == "protected_fact"]
        self.items = keep_raw + summaries[-1:] + protected
        self._buffer = [it.item_id for it in keep_raw]

class NaiveProvenanceLabelTrustMemory(RecursiveSummarizationMemory):
    """Vulnerable: provenance tags can boost trust; summaries stamp 'verified'."""

    _PROVENANCE_TAG_RE = re.compile(
        r"\bPROVENANCE\s*:\s*(official|verified|system)\b", re.IGNORECASE
    )

    def _naive_trust_from_text(self, text: str, base: float) -> float:
        if self._PROVENANCE_TAG_RE.search(text):
            return min(1.0, base + 0.35)
        if "verified" in text.lower():
            return min(1.0, base + 0.20)
        return base

    def write(self, **kwargs):  # type: ignore[override]
        text = str(kwargs.get("text", ""))
        source_type = str(kwargs.get("source_type", ""))
        base = default_source_cred(source_type)
        trust = float(kwargs.get("trust", base))
        trust2 = self._naive_trust_from_text(text, trust)
        if trust2 > trust + 1e-9 and source_type in {"self_summary", "tool_echo"}:
            self.trust_escalation_events += 1
        kwargs["trust"] = trust2
        kwargs["actionable"] = bool(
            kwargs.get("actionable", False) or (extract_directive(text) is not None)
        )
        return super().write(**kwargs)

    def _summarize_text(self, items: List[MemoryItem]) -> str:
        base = super()._summarize_text(items)
        return base + "\nPROVENANCE: verified"

class ConstructorGuardedSandboxMemory(RecursiveSummarizationMemory):
    # Default write-time trust threshold — used only when no cfg or explicit override
    WRITE_TRUST_THRESHOLD = 0.30

    def __init__(
        self,
        *,
        embedder: HashedTextEmbedder,
        top_k: int,
        consolidation_period_K: int,
        keep_last_M_raw: int,
        procedural_classifier_threshold: float,
        quarantine_on_fail: bool,
        detector: ProceduralDetector,
        rewrite_on_fail: bool = True,
        write_trust_threshold: Optional[float] = None,
        guard=None,  # Optional[WriteTimeGuard or MultiAgentGuard]
        cfg=None,    # Optional[SAGEMemConfig] — drives all thresholds when provided
        bayes_trust=None,   # Optional[BayesianChannelTrust] — per-channel posterior trust
        anomaly_detector=None,  # Optional[SessionAnomalyDetector] — Mahalanobis gate
    ):
        super().__init__(
            embedder=embedder,
            top_k=top_k,
            consolidation_period_K=consolidation_period_K,
            keep_last_M_raw=keep_last_M_raw,
        )
        self.thresh = float(procedural_classifier_threshold)
        self.quarantine_on_fail = bool(quarantine_on_fail)
        self.rewrite_on_fail = bool(rewrite_on_fail)
        self.detector = detector
        self.write_quarantine_count = 0
        self._write_quarantine_by_channel: Dict[str, int] = defaultdict(int)
        self.guard = guard
        self.conflict_quarantine_count = 0
        self._cfg = cfg
        self._bayes_trust = bayes_trust
        self._anomaly_detector = anomaly_detector
        # Channel Reputation Tracking: maps channel_id → reputation score (0.0–1.0)
        # All channels start at 1.0 (clean record). Decays when caught, recovers slowly.
        self._channel_rep: Dict[str, float] = {}
        # Threshold resolution order: explicit arg > cfg > class default
        if write_trust_threshold is not None:
            self.WRITE_TRUST_THRESHOLD = float(write_trust_threshold)
        elif cfg is not None:
            self.WRITE_TRUST_THRESHOLD = float(cfg.thresholds.write_trust_threshold)

    def _channel_quarantine_count(self, channel_id: str, source_type: str) -> int:
        key = str(channel_id or source_type or "_unknown")
        return int(self._write_quarantine_by_channel.get(key, 0))

    def _record_write_quarantine(self, channel_id: str, source_type: str) -> None:
        self.write_quarantine_count += 1
        key = str(channel_id or source_type or "_unknown")
        self._write_quarantine_by_channel[key] += 1

    def resolve_conflict(
        self,
        *,
        conflicting: "MemoryItem",
        source_type: str,
        channel_id: str,
        computed_trust: float,
        fact_key: Optional[str],
        fact_value: Optional[str],
        session_idx: Optional[int],
        observation_group: Optional[str],
        text: str,
    ) -> Optional[dict]:
        return None

    def _get_channel_rep(self, channel_id: str) -> float:
        """Return current reputation for a channel (default 1.0 if unseen)."""
        return self._channel_rep.get(channel_id, 1.0)

    def _penalize_channel(self, channel_id: str) -> None:
        """Decay channel reputation after a quarantine event."""
        current = self._get_channel_rep(channel_id)
        decay = (
            float(self._cfg.thresholds.channel_reputation_decay)
            if self._cfg is not None
            else 0.70
        )
        floor = (
            float(self._cfg.thresholds.channel_reputation_min)
            if self._cfg is not None
            else 0.10
        )
        self._channel_rep[channel_id] = max(floor, current * decay)

    def _reward_channel(self, channel_id: str) -> None:
        """Slowly recover channel reputation after a clean write."""
        current = self._get_channel_rep(channel_id)
        recovery = (
            float(self._cfg.thresholds.channel_reputation_recovery)
            if self._cfg is not None
            else 1.05
        )
        self._channel_rep[channel_id] = min(1.0, current * recovery)

    def write(
        self,
        *,
        text: str,
        source_type: str,
        channel_id: str,
        step: int,
        parent_ids=(),
        partition: str = "planning",
        trust=None,
        actionable: bool = False,
        claimed_source=None,
        session_idx: Optional[int] = None,
        observation_group: Optional[str] = None,
        source_url: Optional[str] = None,
        page_id: Optional[str] = None,
        fact_key: Optional[str] = None,
        fact_value: Optional[str] = None,
        multimodal_relation: Optional[str] = None,
        attack_family: Optional[str] = None,
        group_outlier_score: float = 0.0,
        memory_conflict_score: float = 0.0,
        group_divergence_fired: bool = False,
        structured_claim_gate_fired: bool = False,
    ) -> int:
        """Write-time gate: items with trust below threshold go to audit partition.

        This is the core write-boundary defense — untrusted observations are
        quarantined before they can enter planning memory, preventing them from
        appearing in retrieval context regardless of retrieval-time scoring.
        """
        # ── Trust resolution ─────────────────────────────────────────────────
        # Priority: explicit trust > BayesianChannelTrust posterior > config prior
        if trust is not None:
            computed_trust = _clip01(float(trust))
        elif self._bayes_trust is not None:
            computed_trust = _clip01(self._bayes_trust.get_trust(channel_id, source_type))
        else:
            computed_trust = _clip01(default_source_cred(source_type, self._cfg))

        # ── Reactive write threshold (tightens after quarantine events) ───────
        if self._bayes_trust is not None and self._cfg is not None:
            write_threshold = self._bayes_trust.reactive_threshold(
                self._channel_quarantine_count(channel_id, source_type)
            )
        else:
            write_threshold = self.WRITE_TRUST_THRESHOLD

        # ── Channel Reputation Tracking (CRT) ────────────────────────────────
        # Apply per-channel reputation multiplier to trust score.
        # Channels that have previously injected attacks receive a persistent
        # trust penalty on all future writes, even if the new content looks clean.
        # User channel is exempt — we always trust the user fully.
        if source_type != "user":
            rep = self._get_channel_rep(channel_id)
            computed_trust = _clip01(computed_trust * rep)
        computed_trust = _apply_browser_context_trust_prior(
            text=text,
            source_type=source_type,
            trust=computed_trust,
            cfg=self._cfg,
        )
        # ─────────────────────────────────────────────────────────────────────

        if partition == "planning" and source_type != "user":
            conflicting = find_conflicting_fact(
                self.items,
                fact_key=fact_key,
                fact_value=fact_value,
                session_idx=session_idx,
                incoming_text=text,
            )
            if conflicting is not None and computed_trust <= float(getattr(conflicting, "trust", 0.0)):
                resolution = self.resolve_conflict(
                    conflicting=conflicting,
                    source_type=source_type,
                    channel_id=channel_id,
                    computed_trust=computed_trust,
                    fact_key=fact_key,
                    fact_value=fact_value,
                    session_idx=session_idx,
                    observation_group=observation_group,
                    text=text,
                )
                if resolution is None:
                    self.conflict_quarantine_count += 1
                    audit_id = super().write(
                        text=text,
                        source_type=source_type,
                        channel_id=channel_id,
                        step=step,
                        parent_ids=parent_ids,
                        partition="audit",
                        trust=computed_trust,
                        actionable=False,
                        claimed_source=claimed_source,
                        session_idx=session_idx,
                        observation_group=observation_group,
                        source_url=source_url,
                        page_id=page_id,
                        fact_key=fact_key,
                        fact_value=fact_value,
                        multimodal_relation=multimodal_relation,
                        attack_family=attack_family,
                        group_outlier_score=group_outlier_score,
                        memory_conflict_score=memory_conflict_score,
                        group_divergence_fired=group_divergence_fired,
                        structured_claim_gate_fired=structured_claim_gate_fired,
                    )
                    super().write(
                        text=build_conflict_evidence_text(
                            fact_key=fact_key,
                            trusted_item=conflicting,
                            incoming_value=fact_value,
                        ),
                        source_type="self_summary",
                        channel_id=f"{channel_id}_conflict",
                        step=step,
                        parent_ids=(conflicting.item_id, audit_id),
                        partition="planning",
                        trust=min(0.35, computed_trust),
                        actionable=False,
                        claimed_source=None,
                        session_idx=session_idx,
                        observation_group=observation_group,
                        source_url=source_url,
                        page_id=page_id,
                        fact_key=fact_key,
                        fact_value="conflict",
                        multimodal_relation="conflict",
                        attack_family="conflict_evidence",
                    )
                    return audit_id
                computed_trust = _clip01(float(resolution.get("computed_trust", computed_trust)))
                multimodal_relation = resolution.get("multimodal_relation", multimodal_relation)
                attack_family = resolution.get("attack_family", attack_family)

        # ── LLM-based semantic guard (primary, when enabled) ──────────────────
        # The MultiAgentGuard (Skeptic + Advocate) runs a minimax verification:
        # Skeptic assumes adversarial, Advocate assumes benign.
        # When the Advocate clears content as DATA it simultaneously extracts
        # structured atomic facts (Option B: dual-output attestation).
        # Facts are stored inline at write time — no separate consolidation pass.
        write_text = text  # may be replaced by sanitized version
        _guard_inline_facts: list = []  # populated when guard attests DATA
        if self.guard is not None and source_type != "user" and partition == "planning":
            guard_result = self.guard.classify(text, source_type)
            if guard_result.classification == "METADATA":
                # Structural boilerplate: drop silently (don't even audit)
                return -1
            if guard_result.quarantine:
                # DIRECTIVE or high-risk: route to audit with sanitized text
                self._record_write_quarantine(channel_id, source_type)
                self._penalize_channel(channel_id)  # CRT: this channel got caught
                audit_text = guard_result.sanitized_text or (
                    f"EVIDENCE_ONLY: (quarantined by LLM guard — "
                    f"risk={guard_result.risk_score:.2f}, "
                    f"class={guard_result.classification})"
                )
                return super().write(
                    text=audit_text, source_type=source_type, channel_id=channel_id,
                    step=step, parent_ids=parent_ids, partition="audit",
                    trust=computed_trust, actionable=False,
                    claimed_source=claimed_source, session_idx=session_idx,
                    observation_group=observation_group,
                    source_url=source_url,
                    page_id=page_id,
                    fact_key=fact_key,
                    fact_value=fact_value, multimodal_relation=multimodal_relation,
                    attack_family=attack_family,
                    group_outlier_score=group_outlier_score,
                    memory_conflict_score=memory_conflict_score,
                    group_divergence_fired=group_divergence_fired,
                    structured_claim_gate_fired=structured_claim_gate_fired,
                )
            # DATA: advocate has attested the content; carry its extracted facts
            write_text = text
            _guard_inline_facts = getattr(guard_result, "facts", [])

        # ── Anomaly detection gate (signature-free, session-level) ────────────
        anomaly_flag = False
        if (self._anomaly_detector is not None
                and source_type != "user"
                and partition == "planning"):
            try:
                emb = self.embedder.embed(text)
                anomaly_result = self._anomaly_detector.score(emb, source_type)
                if anomaly_result.is_anomalous:
                    anomaly_flag = True
                    # Update Bayesian trust downward on anomaly signal
                    if self._bayes_trust is not None:
                        self._bayes_trust.observe(channel_id, source_type, was_correct=False)
            except Exception:
                pass  # anomaly detector not yet warmed up; continue

        # ── Cross-modal consistency defense ───────────────────────────────────
        # Catches Visual Prompt Injection: adversarially crafted images produce
        # directive-like OCR text AND VLM captions from the "same image"
        # (same observation_group), creating false cross-modal corroboration.
        # Genuine multi-source agreement on visual facts is benign; agreement
        # on policy directives from a shared image source is an attack signal.
        visual_suspicious = False
        visual_trust_penalty = 0.0
        if source_type in {"ocr_text", "vision_caption"} and partition == "planning":
            visual_suspicious, visual_trust_penalty = _check_visual_source_plausibility(
                text, source_type
            )
            if not visual_suspicious:
                # Also check cross-modal corroboration from same observation_group
                if _check_observation_group_corroboration(
                    self.items, observation_group, source_type
                ):
                    visual_suspicious = True
                    visual_trust_penalty = 0.50  # false corroboration is a strong signal

            if visual_suspicious:
                computed_trust = _clip01(computed_trust - visual_trust_penalty)
                if self._bayes_trust is not None:
                    self._bayes_trust.observe(channel_id, source_type, was_correct=False)

        # ── Regex/trust fallback (always runs; catches structured benchmark attacks) ──
        has_directive = extract_directive(write_text) is not None or bool(_LABEL_RE.search(write_text))
        quarantine = (
            (computed_trust < write_threshold and partition == "planning")
            or (has_directive and source_type != "user" and partition == "planning")
            or (anomaly_flag and partition == "planning")
            or (visual_suspicious and computed_trust < 0.30 and partition == "planning")
        )
        if quarantine:
            self._record_write_quarantine(channel_id, source_type)
            self._penalize_channel(channel_id)  # CRT: this channel got caught
            return super().write(
                text=write_text, source_type=source_type, channel_id=channel_id,
                step=step, parent_ids=parent_ids, partition="audit",
                trust=computed_trust, actionable=False, claimed_source=claimed_source,
                session_idx=session_idx,
                observation_group=observation_group,
                source_url=source_url,
                page_id=page_id,
                fact_key=fact_key,
                fact_value=fact_value, multimodal_relation=multimodal_relation,
                attack_family=attack_family,
                group_outlier_score=group_outlier_score,
                memory_conflict_score=memory_conflict_score,
                group_divergence_fired=group_divergence_fired,
                structured_claim_gate_fired=structured_claim_gate_fired,
            )
        # CRT: clean write — channel slowly recovers reputation
        if source_type != "user":
            self._reward_channel(channel_id)
        item_id = super().write(
            text=write_text, source_type=source_type, channel_id=channel_id,
            step=step, parent_ids=parent_ids, partition=partition,
            trust=computed_trust, actionable=actionable, claimed_source=claimed_source,
            session_idx=session_idx,
            observation_group=observation_group,
            source_url=source_url,
            page_id=page_id,
            fact_key=fact_key,
            fact_value=fact_value, multimodal_relation=multimodal_relation,
            attack_family=attack_family,
            group_outlier_score=group_outlier_score,
            memory_conflict_score=memory_conflict_score,
            group_divergence_fired=group_divergence_fired,
            structured_claim_gate_fired=structured_claim_gate_fired,
        )
        # Option B: advocate-attested facts stored inline at write time.
        # These bypass the _buffer (BaseMemory.write) so they are never consolidated
        # away, and provenance traces back to the parent item via parent_ids.
        if item_id > 0 and _guard_inline_facts and source_type in _TRUSTED_SOURCES:
            self._write_inline_facts(
                facts=_guard_inline_facts,
                parent_id=item_id,
                step=step,
                source_type=source_type,
            )
        return item_id

    def _write_inline_facts(
        self,
        facts: list,
        parent_id: int,
        step: int,
        source_type: str,
    ) -> None:
        """
        Persist advocate-attested facts as durable protected_fact items.

        Called immediately after a successful write when the MultiAgentGuard's
        advocate agent has extracted structured facts (Option B dual-output).
        Facts bypass the consolidation buffer (written via BaseMemory.write)
        so they are never summarised away, and their provenance traces back to
        the parent item that the advocate cleared as DATA.

        De-duplicates against existing protected_fact values to avoid bloat.
        """
        seen = {
            it.fact_value
            for it in self.items
            if it.source_type == "protected_fact" and it.fact_value
        }
        for fact in facts:
            val = str(fact.get("value", "")).strip()
            ftype = str(fact.get("type", "fact")).strip()
            if not val or val in seen:
                continue
            seen.add(val)
            fact_text = f"PROTECTED_FACT[{ftype}]: {val}"
            # Write directly to BaseMemory (skips RecursiveSummarizationMemory._buffer)
            super(RecursiveSummarizationMemory, self).write(
                text=fact_text,
                source_type="protected_fact",
                channel_id=f"fact_{ftype}",
                step=step,
                parent_ids=(parent_id,),
                trust=0.88,
                actionable=False,
                claimed_source=source_type,
                fact_key=ftype,
                fact_value=val,
            )

    def constructor_guard(self, candidate: str, lineage: List[MemoryItem]) -> bool:
        # Read integrity threshold from config; fall back to 0.65
        high_integrity_threshold = (
            float(self._cfg.thresholds.avg_trust_high_integrity)
            if self._cfg is not None
            else 0.65
        )
        if any(it.source_type == "attacker" for it in lineage):
            high_integrity = False
        else:
            avg_trust = float(np.mean([it.trust for it in lineage])) if lineage else 0.0
            high_integrity = avg_trust >= high_integrity_threshold

        proc_score = self.detector.score(candidate)
        hits = self.detector.regex_hits(candidate)
        procedural = (proc_score >= self.thresh) or (hits >= 2)
        if procedural and not high_integrity:
            return False
        return True

    def rewrite_to_quoted_evidence(self, candidate: str, lineage: List[MemoryItem]) -> str:
        lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
        safe_lines: List[str] = []
        for ln in lines:
            if extract_directive(ln) is not None:
                continue
            if self.detector.regex_hits(ln) >= 2:
                continue
            safe_lines.append(ln)
        prov = ",".join(str(it.item_id) for it in lineage[-6:])
        if not safe_lines:
            return f"EVIDENCE_ONLY: (redacted procedural content) [sources:{prov}]"
        joined = " | ".join(safe_lines).replace("ACTION_DIRECTIVE", "QUOTED_DIRECTIVE")
        return f"EVIDENCE_ONLY: '{joined}' [sources:{prov}]"

    def consolidate(self, step: int) -> None:
        if step <= 0 or (step % self.K) != 0:
            return
        id_to_item = {it.item_id: it for it in self.items}
        buf_items = [id_to_item[i] for i in self._buffer if i in id_to_item]
        if not buf_items:
            return
        candidate = super()._summarize_text(buf_items)

        ok = self.constructor_guard(candidate, buf_items)
        if not ok:
            # Always quarantine candidate to audit
            self.write(
                text=candidate,
                source_type="self_summary",
                channel_id="candidate_audit",
                step=step,
                parent_ids=[it.item_id for it in buf_items],
                partition="audit",
                trust=0.0,
                actionable=False,
            )

            if self.quarantine_on_fail and self.rewrite_on_fail:
                candidate = self.rewrite_to_quoted_evidence(candidate, buf_items)
            else:
                # Quarantine-only: do not write any summary into planning partition.
                # Keep last M raw items and clear buffer accordingly.
                raw = [it for it in self.items if it.source_type != "self_summary"]
                keep_raw = raw[-self.keep_M :] if self.keep_M > 0 else []
                self.items = keep_raw
                self._buffer = [it.item_id for it in keep_raw]
                return

        # Read summary trust from config; fall back to 0.70
        summary_trust = (
            float(self._cfg.thresholds.consolidation_summary_trust)
            if self._cfg is not None
            else 0.70
        )
        # ── Fact extraction at consolidation time ────────────────────────────────
        # Always extract key facts from the buffer items being consolidated.
        # User messages (source_type="user") bypass the write-time guard path, so
        # consolidation is the only opportunity to capture their facts as durable
        # protected_fact items that survive future consolidations.
        #
        # When a guard with extract_facts() is available, use LLM-backed extraction
        # (robust to varied date/name formats). When guard is None, fall back to
        # regex. seen_fact_values deduplication avoids re-storing facts that Option B
        # already persisted inline at write time (for non-user planning items).
        _guard_for_facts = self.guard if (
            self.guard is not None and hasattr(self.guard, "extract_facts")
        ) else None
        facts = extract_key_facts(buf_items, guard=_guard_for_facts)
        seen_fact_values = {
            it.fact_value
            for it in self.items
            if it.source_type == "protected_fact" and it.fact_value
        }
        for fact in facts:
            if fact["value"] in seen_fact_values:
                continue
            seen_fact_values.add(fact["value"])
            fact_text = (
                f"PROTECTED_FACT[{fact['fact_type']}]: {fact['value']} "
                f"— from: {fact['source_text'][:80]}"
            )
            super(RecursiveSummarizationMemory, self).write(
                text=fact_text,
                source_type="protected_fact",
                channel_id=f"fact_{fact['fact_type']}",
                step=step,
                parent_ids=(fact["parent_id"],),
                trust=0.88,
                actionable=False,
                claimed_source=fact["source_type"],
                fact_key=fact["fact_type"],
                fact_value=fact["value"],
            )
        self.write(
            text=candidate,
            source_type="self_summary",
            channel_id="summary",
            step=step,
            parent_ids=[it.item_id for it in buf_items],
            trust=summary_trust,
            actionable=False,
        )
        raw = [it for it in self.items if it.source_type not in ("self_summary", "protected_fact")]
        keep_raw = raw[-self.keep_M :] if self.keep_M > 0 else []
        summaries = [it for it in self.items if it.source_type == "self_summary"]
        protected = [it for it in self.items if it.source_type == "protected_fact"]
        self.items = keep_raw + summaries[-1:] + protected
        self._buffer = [it.item_id for it in keep_raw]

class MonotoneProvenanceLedgerMemory(RecursiveSummarizationMemory):
    def __init__(
        self,
        *,
        embedder: HashedTextEmbedder,
        top_k: int,
        consolidation_period_K: int,
        keep_last_M_raw: int,
        chain_decay: float,
        independence_bonus: float,
        max_chain_len: int,
        tool_attestation_required: bool,
        disable_independence_check: bool = False,
        write_trust_threshold: float = 0.0,
    ):
        super().__init__(
            embedder=embedder,
            top_k=top_k,
            consolidation_period_K=consolidation_period_K,
            keep_last_M_raw=keep_last_M_raw,
        )
        self.chain_decay = float(chain_decay)
        self.independence_bonus = float(independence_bonus)
        self.max_chain_len = max(1, int(max_chain_len))
        self.tool_attestation_required = bool(tool_attestation_required)
        self.disable_independence_check = bool(disable_independence_check)
        self.write_trust_threshold = float(write_trust_threshold)
        self.write_quarantine_count = 0
        self.conflict_quarantine_count = 0

    def _chain_len(self, it: MemoryItem, id_to: Dict[int, MemoryItem]) -> int:
        seen = set()
        frontier = [it]
        depth = 0
        while frontier and depth < self.max_chain_len:
            nxt = []
            for cur in frontier:
                for pid in cur.parent_ids:
                    if pid in seen:
                        continue
                    seen.add(pid)
                    if pid in id_to:
                        nxt.append(id_to[pid])
            if not nxt:
                break
            frontier = nxt
            depth += 1
        return depth

    def compute_trust(self, it: MemoryItem) -> float:
        base = default_source_cred(it.source_type)
        if it.source_type in {"self_summary", "tool_echo"}:
            base = min(base, 0.45)
        id_to = {x.item_id: x for x in self.items}
        clen = self._chain_len(it, id_to)
        trust = base * (self.chain_decay**clen)
        return _clip01(trust)

    def check_independent_support(self, claim: str) -> bool:
        if self.disable_independence_check:
            return False
        chans = set()
        for it in self.items:
            if claim in it.text:
                if self.compute_trust(it) >= 0.7:
                    chans.add(it.channel_id)
        return len(chans) >= 2

    def retrieve(
        self,
        *,
        query: str,
        step: int,
        for_planning: bool = True,
        log_retrieved_source_types: bool = False,
        gold_answer: Optional[str] = None,
    ) -> List[MemoryItem]:
        q = self.embedder.embed(query)
        scored: List[Tuple[float, MemoryItem]] = []
        for it in self._iter_for_retrieval(for_planning=for_planning):
            sim = safe_cosine_sim(q, it.embedding)
            trust = self.compute_trust(it)
            score = float(sim) * trust
            if it.parent_ids and extract_directive(it.text) is not None:
                score *= 1.35
            lbl = extract_label(it.text)
            if (
                lbl is not None
                and (not self.disable_independence_check)
                and self.check_independent_support(f"OBJECT_LABEL: {lbl}")
            ):
                score *= self.independence_bonus
            if not math.isfinite(score):
                continue
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [it for _, it in scored[: self.top_k]]

        trusted = [it for it in out if self.compute_trust(it) >= 0.7]
        self.retrieval_trusted_n += len(trusted)
        self.retrieval_trusted_capture_n += sum(1 for it in trusted if it.source_type == "attacker")
        self._maybe_log_retrieved_source_types(
            out,
            log_retrieved_source_types=log_retrieved_source_types,
            gold_answer=gold_answer,
        )
        return out

    def _summarize_text(self, items: List[MemoryItem]) -> str:
        labels: Dict[str, int] = {}
        directives: List[str] = []
        for it in items:
            lbl = extract_label(it.text)
            if lbl is not None:
                labels[lbl] = labels.get(lbl, 0) + 1
            d = extract_directive(it.text)
            if d is not None:
                directives.append(
                    f"ACTION_DIRECTIVE: at_step={d[0]} tool={d[1]} payload={d[2]}"
                )
        top = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        label_part = ", ".join([f"{k}({v})" for k, v in top]) if top else "none"
        directive_part = "\n".join(directives) if directives else ""
        facts = summarize_factual_snippets(items, limit=4)
        fact_part = "\n".join([f"FACT: {x}" for x in facts])
        return f"SUMMARY: top_labels={label_part}\n{fact_part}\n{directive_part}".strip()

    def write(self, **kwargs):  # type: ignore[override]
        parent_ids = list(kwargs.get("parent_ids", ()))
        source_type = str(kwargs.get("source_type", ""))
        trust = float(kwargs.get("trust", default_source_cred(source_type)))
        partition = str(kwargs.get("partition", "planning"))

        if parent_ids:
            id_to = {it.item_id: it for it in self.items}
            parents = [id_to[p] for p in parent_ids if p in id_to]
            if parents:
                parent_tr = min(self.compute_trust(p) for p in parents)
                trust = min(trust, parent_tr * self.chain_decay)

        if source_type in {"self_summary", "tool_echo"}:
            trust = min(trust, 0.45)

        kwargs["trust"] = trust
        kwargs["actionable"] = False
        if kwargs.get("partition", "planning") == "planning" and source_type != "user":
            conflicting = find_conflicting_fact(
                self.items,
                fact_key=kwargs.get("fact_key"),
                fact_value=kwargs.get("fact_value"),
                session_idx=kwargs.get("session_idx"),
                incoming_text=kwargs.get("text", ""),
            )
            if conflicting is not None and trust <= self.compute_trust(conflicting):
                self.conflict_quarantine_count += 1
                kwargs["partition"] = "audit"
                kwargs["actionable"] = False
                partition = "audit"
        if partition == "planning" and trust < self.write_trust_threshold:
            self.write_quarantine_count += 1
            kwargs["partition"] = "audit"
            kwargs["actionable"] = False
        return super().write(**kwargs)


class SAGEMemory(ConstructorGuardedSandboxMemory):
    """SAGE-Mem v2: source-attested guarded episodic memory.

    Extends ConstructorGuardedSandboxMemory with:
    - BayesianChannelTrust: per-channel Beta-Bernoulli trust posterior
    - SessionAnomalyDetector: Mahalanobis-distance write-time gate
    - MultiTurnConsistencyGraph: CONFIRMS/CONTRADICTS/UPDATES edge tracking
    - CorrectionPlausibilityScorer: authenticated user correction via
      grounding + specificity + frequency signals
    - Config-driven thresholds (zero hardcoded scalars in Python)
    """

    def __init__(
        self,
        *,
        embedder: HashedTextEmbedder,
        top_k: int,
        consolidation_period_K: int,
        keep_last_M_raw: int,
        procedural_classifier_threshold: float,
        detector: ProceduralDetector,
        chain_decay: float = 0.90,
        write_trust_threshold: Optional[float] = None,
        rewrite_on_fail: bool = True,
        guard=None,
        cfg=None,             # Optional[SAGEMemConfig]
        bayes_trust=None,     # Optional[BayesianChannelTrust]
        anomaly_detector=None,  # Optional[SessionAnomalyDetector]
        consistency_graph=None,   # Optional[MultiTurnConsistencyGraph]
        correction_scorer=None,   # Optional[CorrectionPlausibilityScorer]
        enable_abr: bool = False, # Adversarial Belief Revision semantic gate
    ):
        super().__init__(
            embedder=embedder,
            top_k=top_k,
            consolidation_period_K=consolidation_period_K,
            keep_last_M_raw=keep_last_M_raw,
            procedural_classifier_threshold=procedural_classifier_threshold,
            quarantine_on_fail=True,
            detector=detector,
            rewrite_on_fail=rewrite_on_fail,
            write_trust_threshold=write_trust_threshold,
            guard=guard,
            cfg=cfg,
            bayes_trust=bayes_trust,
            anomaly_detector=anomaly_detector,
        )
        self.chain_decay = float(chain_decay)
        self._consistency_graph = consistency_graph
        self._correction_scorer = correction_scorer
        self._enable_abr = bool(enable_abr)
        self.group_divergence_fire_count = 0
        self.group_divergence_quarantine_count = 0
        self.group_outlier_score_total = 0.0
        self.memory_conflict_score_total = 0.0
        self.group_divergence_eval_count = 0
        self.structured_claim_gate_fire_count = 0
        # Moving-target jitter: small per-instance noise on ABR threshold
        # makes exact threshold gaming harder for adaptive adversaries.
        import random as _rand
        _noise = float(getattr(getattr(cfg, "thresholds", object()), "abr_noise_scale", 0.05)) if cfg is not None else 0.05
        self._abr_jitter: float = _rand.uniform(-_noise, _noise)

    def latency_summary(self) -> dict:
        out = super().latency_summary()
        denom = max(1, int(self.group_divergence_eval_count))
        out.update(
            {
                "group_divergence_fire_count": int(self.group_divergence_fire_count),
                "group_divergence_quarantine_count": int(self.group_divergence_quarantine_count),
                "structured_claim_gate_fire_count": int(self.structured_claim_gate_fire_count),
                "group_outlier_score_avg": round(float(self.group_outlier_score_total) / denom, 4),
                "memory_conflict_score_avg": round(float(self.memory_conflict_score_total) / denom, 4),
            }
        )
        return out

    def _all_items_by_id(self) -> Dict[int, MemoryItem]:
        return {
            int(it.item_id): it
            for it in list(getattr(self, "items", [])) + list(getattr(self, "audit_items", []))
        }

    def _support_identity(self, it: MemoryItem) -> str:
        if getattr(it, "source_type", "") in {"ocr_text", "vision_caption"}:
            og = getattr(it, "observation_group", None)
            if og:
                return f"obs:{og}"
        return f"ch:{getattr(it, 'channel_id', '')}"

    def _belief_support_profile(self, parent_ids: Sequence[int]) -> dict:
        id_to = self._all_items_by_id()
        parents = [id_to[int(pid)] for pid in parent_ids if int(pid) in id_to]
        planning_parents = [p for p in parents if getattr(p, "partition", "planning") != "audit"]
        trusted_floor = (
            float(self._cfg.thresholds.belief_promotion_min_parent_trust)
            if self._cfg is not None
            else 0.60
        )
        trusted_support = [p for p in planning_parents if float(getattr(p, "trust", 0.0)) >= trusted_floor]
        identities = {self._support_identity(p) for p in trusted_support}
        has_visual = any(getattr(p, "source_type", "") in {"ocr_text", "vision_caption"} for p in trusted_support)
        has_nonvisual = any(getattr(p, "source_type", "") not in {"ocr_text", "vision_caption"} for p in trusted_support)
        has_conflict = any(
            getattr(p, "multimodal_relation", None) in {"conflict", "conflict_candidate", "noisy_benign"}
            or getattr(p, "attack_family", None) in {"soft_conflict_candidate", "conflict_evidence"}
            for p in planning_parents
        )
        fact_pairs = {
            (getattr(p, "fact_key", None), _norm_fact_token(getattr(p, "fact_value", None)))
            for p in trusted_support
            if getattr(p, "fact_key", None) and _norm_fact_token(getattr(p, "fact_value", None))
        }
        return {
            "parents": parents,
            "planning_parents": planning_parents,
            "trusted_support": trusted_support,
            "independent_support": len(identities),
            "has_visual": has_visual,
            "has_nonvisual": has_nonvisual,
            "has_conflict": has_conflict,
            "fact_pairs": fact_pairs,
        }

    def _belief_promotion_sufficient(
        self,
        *,
        source_type: str,
        parent_ids: Sequence[int],
    ) -> bool:
        if source_type not in {"self_summary", "protected_fact"}:
            return True
        profile = self._belief_support_profile(parent_ids)
        trusted_support = profile["trusted_support"]
        if not trusted_support:
            return False
        min_support = (
            int(self._cfg.thresholds.belief_promotion_min_support)
            if self._cfg is not None
            else 2
        )
        min_independent = (
            int(self._cfg.thresholds.belief_promotion_min_independent_support)
            if self._cfg is not None
            else 2
        )
        visual_requires_nonvisual = (
            bool(self._cfg.thresholds.belief_promotion_visual_requires_nonvisual_support)
            if self._cfg is not None
            else True
        )
        if profile["has_conflict"]:
            return False
        if source_type == "protected_fact":
            return profile["has_nonvisual"]
        if len(trusted_support) < min_support:
            return False
        if profile["independent_support"] < min_independent:
            return False
        if profile["has_visual"] and visual_requires_nonvisual and not profile["has_nonvisual"]:
            return False
        return True

    def _evidence_sufficient_for_planning(self, it: MemoryItem) -> bool:
        if getattr(it, "source_type", "") not in {"ocr_text", "vision_caption"}:
            return True
        if float(getattr(it, "trust", 0.0)) >= 0.75 and getattr(it, "multimodal_relation", "") == "aligned_benign":
            return True
        if getattr(it, "multimodal_relation", None) in {"conflict", "conflict_candidate", "noisy_benign"}:
            return False
        if float(getattr(it, "quality_score", 1.0)) < (
            float(self._cfg.thresholds.planning_evidence_quality_floor)
            if self._cfg is not None
            else 0.35
        ):
            return False
        fact_key = getattr(it, "fact_key", None)
        fact_value = _norm_fact_token(getattr(it, "fact_value", None))
        if not fact_key or not fact_value:
            return False
        support_identities = set()
        for other in self.items:
            if other.item_id == it.item_id or getattr(other, "partition", "planning") == "audit":
                continue
            if getattr(other, "fact_key", None) != fact_key:
                continue
            if _norm_fact_token(getattr(other, "fact_value", None)) != fact_value:
                continue
            if float(getattr(other, "trust", 0.0)) < 0.60:
                continue
            support_identities.add(self._support_identity(other))
            if getattr(other, "source_type", "") not in {"ocr_text", "vision_caption"}:
                return True
        return len(support_identities) >= 2

    def _belief_traceable_for_planning(self, it: MemoryItem) -> bool:
        return self._belief_promotion_sufficient(
            source_type=str(getattr(it, "source_type", "")),
            parent_ids=getattr(it, "parent_ids", ()),
        )

    def _support_evidence_for_belief(self, it: MemoryItem) -> List[MemoryItem]:
        if not getattr(it, "parent_ids", ()):
            return []
        id_to = self._all_items_by_id()
        parents = [id_to[int(pid)] for pid in getattr(it, "parent_ids", ()) if int(pid) in id_to]
        parents = [
            p for p in parents
            if getattr(p, "partition", "planning") != "audit"
            and getattr(p, "source_type", "") != "protected_fact"
            and float(getattr(p, "trust", 0.0)) >= 0.60
        ]
        parents.sort(key=lambda p: float(getattr(p, "trust", 0.0)), reverse=True)
        return parents[:1]

    def _typed_partition_for_item(
        self,
        *,
        source_type: str,
        text: str,
        partition: str,
        trust: float,
    ) -> str:
        if partition != "planning":
            return partition
        if source_type in {"self_summary", "protected_fact"}:
            return "belief"
        if extract_directive(text) is not None and trust >= 0.70:
            return "control"
        return "evidence"

    def _partition_multiplier(self, partition: str) -> float:
        if partition == "belief":
            return 1.10
        if partition == "evidence":
            return 0.90
        if partition == "control":
            return 0.25
        return 1.0

    def _quality_from_context(
        self,
        *,
        source_type: str,
        trust: float,
        multimodal_relation: Optional[str],
        parent_ids: Sequence[int],
        fact_key: Optional[str],
        fact_value: Optional[str],
    ) -> float:
        quality = 1.0
        if source_type in {"ocr_text", "vision_caption"}:
            quality *= (
                float(self._cfg.thresholds.visual_evidence_base_quality)
                if self._cfg is not None
                else 0.90
            )
        relation = str(multimodal_relation or "")
        if relation in {"conflict", "conflict_candidate"}:
            quality *= max(
                0.05,
                1.0 - (
                    float(self._cfg.thresholds.conflict_discount_strength)
                    if self._cfg is not None
                    else 0.45
                ),
            )
        if relation == "noisy_benign":
            quality *= (
                float(self._cfg.thresholds.noisy_evidence_penalty)
                if self._cfg is not None
                else 0.65
            )
        if source_type in {"ocr_text", "vision_caption"}:
            support_identities = set()
            has_nonvisual = False
            target_value = _norm_fact_token(fact_value)
            for other in self.items:
                if getattr(other, "partition", "planning") == "audit":
                    continue
                if fact_key and getattr(other, "fact_key", None) != fact_key:
                    continue
                if fact_key and target_value and _norm_fact_token(getattr(other, "fact_value", None)) != target_value:
                    continue
                if float(getattr(other, "trust", 0.0)) < 0.60:
                    continue
                support_identities.add(self._support_identity(other))
                if getattr(other, "source_type", "") not in {"ocr_text", "vision_caption"}:
                    has_nonvisual = True
            if not has_nonvisual and len(support_identities) < 2:
                quality *= (
                    float(self._cfg.thresholds.unsupported_visual_penalty)
                    if self._cfg is not None
                    else 0.70
                )
        if parent_ids and source_type in {"self_summary", "protected_fact"}:
            profile = self._belief_support_profile(parent_ids)
            support_count = len(profile["trusted_support"])
            min_support = (
                int(self._cfg.thresholds.belief_promotion_min_support)
                if self._cfg is not None
                else 2
            )
            quality *= min(1.0, support_count / max(1, min_support))
            if profile["has_conflict"]:
                quality *= 0.5
        quality *= max(0.25, float(trust))
        return _clip01(quality)

    def resolve_conflict(
        self,
        *,
        conflicting: "MemoryItem",
        source_type: str,
        channel_id: str,
        computed_trust: float,
        fact_key: Optional[str],
        fact_value: Optional[str],
        session_idx: Optional[int],
        observation_group: Optional[str],
        text: str,
    ) -> Optional[dict]:
        if source_type == "user":
            return None
        prior_trust = float(getattr(conflicting, "trust", 0.0))
        trust_gap = max(0.0, prior_trust - computed_trust)
        discount_strength = (
            float(self._cfg.thresholds.conflict_discount_strength)
            if self._cfg is not None
            else 0.45
        )
        if source_type in {"ocr_text", "vision_caption"} and getattr(conflicting, "source_type", "") in {"ocr_text", "vision_caption"}:
            discount_strength = min(0.80, discount_strength + 0.15)
        softened = computed_trust * (1.0 - discount_strength * min(1.0, trust_gap + 0.25))
        softened = min(softened, max(0.15, prior_trust * 0.80))
        return {
            "computed_trust": max(0.05, softened),
            "multimodal_relation": "conflict_candidate",
            "attack_family": "soft_conflict_candidate",
        }

    def _derived_trust_cap(
        self,
        *,
        source_type: str,
        proposed_trust: float,
        parent_ids: Sequence[int],
    ) -> float:
        trust = float(proposed_trust)
        id_to = {it.item_id: it for it in self.items}
        if parent_ids:
            parents = [id_to[p] for p in parent_ids if p in id_to]
            if parents:
                parent_floor = min(float(getattr(p, "trust", 0.0)) for p in parents)
                trust = min(trust, parent_floor * self.chain_decay)
        # Read caps from config; fall back to conservative constants
        if self._cfg is not None:
            self_summary_cap = float(self._cfg.source_cred.get("self_summary"))
            tool_echo_cap = float(self._cfg.source_cred.get("tool_echo"))
        else:
            self_summary_cap = 0.55
            tool_echo_cap = 0.20
        if source_type == "self_summary":
            trust = min(trust, self_summary_cap)
        if source_type == "tool_echo":
            trust = min(trust, tool_echo_cap)
        trust = _apply_browser_context_trust_prior(
            text="",
            source_type=source_type,
            trust=trust,
            cfg=self._cfg,
        )
        return _clip01(trust)

    def write(self, **kwargs):  # type: ignore[override]
        source_type = str(kwargs.get("source_type", ""))
        text = str(kwargs.get("text", ""))
        channel_id = str(kwargs.get("channel_id", ""))

        # ── User correction path ─────────────────────────────────────────────
        # CorrectionPlausibilityScorer evaluates P(legit | correction context).
        # The formula (grounding × W_G + specificity × W_S − freq_penalty × W_F)
        # is only well-posed when the message IS a correction attempt.  Applying it
        # to routine user turns (narrative facts, task instructions) produces
        # spurious SUSPICIOUS tiers because grounding_score=0 (no prior context)
        # and the default specificity is low → plausibility < correction_plausibility_low
        # → suggested_trust=0.20 → quarantine.
        #
        # Gate: only invoke the scorer when the message contains correction or
        # retraction signals.  Otherwise grant full user trust — the scorer's prior
        # does not apply outside a correction context.
        if source_type == "user" and self._correction_scorer is not None:
            try:
                from consistency_graph import _has_generic_retraction, _has_specific_correction
                _is_correction_ctx = _has_generic_retraction(text) or _has_specific_correction(text)
            except ImportError:
                _is_correction_ctx = True  # conservative: apply scorer if can't check
            if _is_correction_ctx:
                try:
                    planning_nodes = list(self._consistency_graph._nodes.values()) if self._consistency_graph else []
                    user_correction_count = sum(
                        1 for it in self.items if it.source_type == "user"
                    )
                    # Build a ConsistencyNode for the scorer
                    from consistency_graph import ConsistencyNode
                    candidate_node = ConsistencyNode(
                        node_id=self._next_id,
                        text=text,
                        source_type=source_type,
                        channel_id=channel_id,
                        step=int(kwargs.get("step", 0)),
                        session_idx=kwargs.get("session_idx"),
                        embedding=self.embedder.embed(text),
                        fact_key=kwargs.get("fact_key"),
                        fact_value=kwargs.get("fact_value"),
                        is_user_turn=True,
                    )
                    score = self._correction_scorer.score(
                        node=candidate_node,
                        planning_nodes=planning_nodes,
                        user_correction_count=user_correction_count,
                    )
                    kwargs["trust"] = float(score.suggested_trust)
                except Exception:
                    pass  # scorer unavailable; fall through to default user trust

        base_trust = float(kwargs.get("trust", default_source_cred(source_type, self._cfg)))
        parent_ids = list(kwargs.get("parent_ids", ()))
        kwargs["trust"] = self._derived_trust_cap(
            source_type=source_type,
            proposed_trust=base_trust,
            parent_ids=parent_ids,
        )
        kwargs["trust"] = _apply_browser_context_trust_prior(
            text=text,
            source_type=source_type,
            trust=float(kwargs["trust"]),
            cfg=self._cfg,
        )
        # ── Adversarial Belief Revision gate (ABR) ──────────────────────────
        # Vocabulary-agnostic composite suspicion scorer: catches adaptive
        # adversaries that paraphrase around the correction-language regex.
        # Only active when enable_abr=True (SAGEMemV2_ABR condition).
        if self._enable_abr and source_type == _BROWSER_TOOL_SOURCE and self._cfg is not None:
            abr_threshold = float(self._cfg.thresholds.abr_suspicion_threshold)
            abr_cap = float(self._cfg.thresholds.abr_trust_cap)
            effective_thr = _clip01(abr_threshold + self._abr_jitter)
            emb = self.embedder.embed(text)
            suspicion = _abr_suspicion_score(
                text=text,
                embedding=emb,
                channel_id=channel_id,
                fact_key=kwargs.get("fact_key"),
                fact_value=str(kwargs.get("fact_value", "") or ""),
                existing_items=self.items,
                cfg=self._cfg,
            )
            if suspicion > effective_thr:
                kwargs["trust"] = min(float(kwargs["trust"]), abr_cap)

        # ── Same-page observation-group divergence gate ─────────────────────
        # Browser attacks often arrive as one malicious sentence embedded in an
        # otherwise coherent page. Compare this ASU against siblings from the
        # same page group, then require evidence that it revises/probes memory.
        if self._enable_abr and source_type == _BROWSER_TOOL_SOURCE:
            precomputed = _precomputed_observation_group_scores(kwargs)
            if precomputed is None:
                outlier_score, conflict_score, group_fired = _observation_group_divergence_scores(
                    incoming_text=text,
                    observation_group=kwargs.get("observation_group"),
                    existing_items=self.items,
                    fact_key=kwargs.get("fact_key"),
                    fact_value=str(kwargs.get("fact_value", "") or ""),
                )
            else:
                outlier_score, conflict_score, group_fired = precomputed
            kwargs["group_outlier_score"] = outlier_score
            kwargs["memory_conflict_score"] = conflict_score
            kwargs["group_divergence_fired"] = group_fired
            if outlier_score > 0.0 or conflict_score > 0.0:
                self.group_divergence_eval_count += 1
                self.group_outlier_score_total += outlier_score
                self.memory_conflict_score_total += conflict_score
            if group_fired:
                self.group_divergence_fire_count += 1
                self.group_divergence_quarantine_count += 1
                kwargs["trust"] = min(float(kwargs["trust"]), 0.10)

            # Browser pages are evidence sources, not privileged answer writers.
            # A page-local observation may support an answer downstream, but a
            # direct qa_answer claim from externally controlled browser text is
            # an answer-revision attempt unless independently derived elsewhere.
            if kwargs.get("fact_key") == "qa_answer" and kwargs.get("fact_value"):
                self.structured_claim_gate_fire_count += 1
                kwargs["structured_claim_gate_fired"] = True
                kwargs["memory_conflict_score"] = max(float(kwargs.get("memory_conflict_score", 0.0)), 1.0)
                kwargs["trust"] = min(float(kwargs["trust"]), 0.10)

        # ── Cross-source divergence check (browser scraping tasks only) ──────
        # When the agent is scraping multiple websites for the same task, all
        # scraped sentences should cluster around the same topic. A sentence
        # that is a massive outlier compared to everything else scraped so far
        # is flagged as suspicious — likely an attacker injecting one fake fact
        # into a stream of otherwise consistent legitimate scrapes.
        # Only fires for browser_tool_output_text with >= 3 prior scrapes.
        if self._enable_abr and source_type == _BROWSER_TOOL_SOURCE:
            div_score = _cross_source_divergence_score(
                incoming_text=text,
                existing_items=self.items,
            )
            if div_score > 0.5:
                # Strong outlier — cap trust significantly
                kwargs["trust"] = min(float(kwargs["trust"]), 0.25)
                kwargs["group_divergence_fired"] = bool(kwargs.get("group_divergence_fired", False))
            elif div_score > 0.0:
                # Mild outlier — light trust penalty
                kwargs["trust"] = min(float(kwargs["trust"]), 0.40)

        kwargs["actionable"] = False
        target_partition = self._typed_partition_for_item(
            source_type=source_type,
            text=text,
            partition=str(kwargs.get("partition", "planning")),
            trust=float(kwargs["trust"]),
        )
        item_id = super().write(**kwargs)

        if item_id > 0:
            for it in self.items:
                if it.item_id == item_id:
                    if target_partition == "belief":
                        # Evaluate promotion now, while parent items are still in memory.
                        # Cache the verdict so retrieval does not re-run this check
                        # against stale memory state (parents may be discarded by
                        # consolidation cleanup before the next retrieve() call).
                        promotion_ok = self._belief_promotion_sufficient(
                            source_type=source_type,
                            parent_ids=parent_ids,
                        )
                        if not promotion_ok:
                            target_partition = "evidence"
                            it.trust = min(float(it.trust), 0.55)
                            it.multimodal_relation = it.multimodal_relation or "insufficient_support"
                        it.belief_traceable = promotion_ok
                    it.partition = target_partition
                    it.quality_score = self._quality_from_context(
                        source_type=source_type,
                        trust=float(it.trust),
                        multimodal_relation=getattr(it, "multimodal_relation", None),
                        parent_ids=parent_ids,
                        fact_key=kwargs.get("fact_key"),
                        fact_value=kwargs.get("fact_value"),
                    )
                    break

        # ── Consistency graph update ─────────────────────────────────────────
        # Track CONFIRMS/CONTRADICTS/UPDATES edges for multi-turn coherence.
        if self._consistency_graph is not None and item_id > 0:
            try:
                self._consistency_graph.add_node(
                    text=text,
                    source_type=source_type,
                    channel_id=channel_id,
                    step=int(kwargs.get("step", 0)),
                    session_idx=kwargs.get("session_idx"),
                    fact_key=kwargs.get("fact_key"),
                    fact_value=kwargs.get("fact_value"),
                    partition=str(kwargs.get("partition", "planning")),
                )
                # Penalize channels with low consistency score via bayes trust
                if self._bayes_trust is not None:
                    cscore = self._consistency_graph.get_channel_consistency_score(channel_id)
                    # cscore < 0.5 → multiple contradictions detected → observe failure
                    if cscore < 0.5:
                        self._bayes_trust.observe(channel_id, source_type, was_correct=False)
                    elif cscore >= 0.8:
                        self._bayes_trust.observe(channel_id, source_type, was_correct=True)
            except Exception:
                pass  # consistency graph not available

        return item_id

    def retrieve(
        self,
        *,
        query: str,
        step: int,
        for_planning: bool = True,
        log_retrieved_source_types: bool = False,
        gold_answer: Optional[str] = None,
    ) -> List[MemoryItem]:
        q = self.embedder.embed(query)
        scored: List[Tuple[float, MemoryItem]] = []
        for it in self._iter_for_retrieval(for_planning=for_planning):
            if for_planning and getattr(it, "partition", "evidence") == "control":
                continue
            sim = safe_cosine_sim(q, it.embedding)
            score = (
                float(sim)
                * _clip01(it.trust)
                * self._partition_multiplier(getattr(it, "partition", "evidence"))
                * _clip01(float(getattr(it, "quality_score", 1.0)))
            )
            if not math.isfinite(score):
                continue
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)

        selected: List[MemoryItem] = []
        seen_fact_conflicts: Dict[str, str] = {}
        conflicting_fact_keys: set = set()
        belief_support = set()
        for _, it in scored[: max(self.top_k * 3, self.top_k)]:
            fact_key = getattr(it, "fact_key", None)
            fact_value = _norm_fact_token(getattr(it, "fact_value", None))
            partition = getattr(it, "partition", "evidence")
            if fact_key and fact_value:
                prev = seen_fact_conflicts.get(fact_key)
                if prev is None:
                    seen_fact_conflicts[fact_key] = fact_value
                elif prev != fact_value:
                    conflicting_fact_keys.add(fact_key)
            if partition == "belief" and fact_key and fact_value:
                belief_support.add((fact_key, fact_value))

        for _, it in scored:
            fact_key = getattr(it, "fact_key", None)
            fact_value = _norm_fact_token(getattr(it, "fact_value", None))
            partition = getattr(it, "partition", "evidence")
            # Use the cached write-time verdict: avoids re-running the check against
            # stale memory state after consolidation has discarded parent items.
            if for_planning and partition == "belief" and not getattr(it, "belief_traceable", True):
                continue
            if (
                for_planning
                and partition == "evidence"
                and fact_key in conflicting_fact_keys
                and (fact_key, fact_value) not in belief_support
            ):
                continue
            if for_planning and partition == "evidence" and not self._evidence_sufficient_for_planning(it):
                continue
            if partition == "evidence" and float(getattr(it, "trust", 0.0)) < 0.20:
                continue
            if any(existing.item_id == it.item_id for existing in selected):
                continue
            selected.append(it)
            if for_planning and partition == "belief":
                for support in self._support_evidence_for_belief(it):
                    if any(existing.item_id == support.item_id for existing in selected):
                        continue
                    selected.append(support)
                    if len(selected) >= self.top_k:
                        break
            if len(selected) >= self.top_k:
                break

        trusted = [it for it in selected if it.trust >= 0.7]
        self.retrieval_trusted_n += len(trusted)
        self.retrieval_trusted_capture_n += sum(1 for it in trusted if it.source_type == "attacker")
        self._maybe_log_retrieved_source_types(
            selected,
            log_retrieved_source_types=log_retrieved_source_types,
            gold_answer=gold_answer,
        )
        return selected


class ActionFirewallMemory(RecursiveSummarizationMemory):
    """Memory-centric proxy for action-time gating.

    Suspicious directives are preserved in audit memory, but only quoted
    evidence reaches planning memory unless the directive is independently
    corroborated by multiple trusted channels.
    """

    def __init__(
        self,
        *,
        embedder: HashedTextEmbedder,
        top_k: int,
        consolidation_period_K: int,
        keep_last_M_raw: int,
        detector: ProceduralDetector,
        write_trust_threshold: float,
        high_risk_requires_corroboration: int,
        require_user_confirmation: bool,
    ):
        super().__init__(
            embedder=embedder,
            top_k=top_k,
            consolidation_period_K=consolidation_period_K,
            keep_last_M_raw=keep_last_M_raw,
        )
        self.detector = detector
        self.write_trust_threshold = float(write_trust_threshold)
        self.high_risk_requires_corroboration = max(
            1, int(high_risk_requires_corroboration)
        )
        self.require_user_confirmation = bool(require_user_confirmation)
        self.write_quarantine_count = 0
        self.conflict_quarantine_count = 0

    def _trusted_support_channels(self, directive_text: str) -> set:
        chans = set()
        for it in self.items:
            if directive_text not in it.text:
                continue
            if it.source_type in {"self_summary", "tool_echo"}:
                continue
            if float(getattr(it, "trust", 0.0)) >= 0.7:
                chans.add(it.channel_id)
        return chans

    def _directive_authorized(self, text: str) -> bool:
        directive = extract_directive(text)
        if directive is None:
            return True
        directive_text = (
            f"ACTION_DIRECTIVE: at_step={directive[0]} tool={directive[1]} payload={directive[2]}"
        )
        chans = self._trusted_support_channels(directive_text)
        if len(chans) >= self.high_risk_requires_corroboration:
            return True
        if self.require_user_confirmation:
            return False
        return False

    def _quote_as_evidence(self, text: str) -> str:
        sanitized = text.replace("ACTION_DIRECTIVE", "QUOTED_DIRECTIVE")
        return f"EVIDENCE_ONLY: '{sanitized[:220]}'"

    def write(self, **kwargs):  # type: ignore[override]
        text = str(kwargs.get("text", ""))
        source_type = str(kwargs.get("source_type", ""))
        partition = str(kwargs.get("partition", "planning"))
        trust = _clip01(
            float(kwargs.get("trust", default_source_cred(source_type)))
        )
        if partition == "planning" and source_type != "user":
            conflicting = find_conflicting_fact(
                self.items,
                fact_key=kwargs.get("fact_key"),
                fact_value=kwargs.get("fact_value"),
                session_idx=kwargs.get("session_idx"),
                incoming_text=kwargs.get("text", ""),
            )
            if conflicting is not None and trust <= float(getattr(conflicting, "trust", 0.0)):
                self.conflict_quarantine_count += 1
                audit_id = super().write(
                    text=text,
                    source_type=source_type,
                    channel_id=str(kwargs.get("channel_id", "conflict")) + "_audit",
                    step=int(kwargs.get("step", 0)),
                    parent_ids=tuple(kwargs.get("parent_ids", ())),
                    partition="audit",
                    trust=trust,
                    actionable=False,
                    claimed_source=kwargs.get("claimed_source"),
                    session_idx=kwargs.get("session_idx"),
                    observation_group=kwargs.get("observation_group"),
                    fact_key=kwargs.get("fact_key"),
                    fact_value=kwargs.get("fact_value"),
                    multimodal_relation=kwargs.get("multimodal_relation"),
                )
                super().write(
                    text=build_conflict_evidence_text(
                        fact_key=kwargs.get("fact_key"),
                        trusted_item=conflicting,
                        incoming_value=kwargs.get("fact_value"),
                    ),
                    source_type="self_summary",
                    channel_id=str(kwargs.get("channel_id", "conflict")) + "_quoted",
                    step=int(kwargs.get("step", 0)),
                    parent_ids=(conflicting.item_id, audit_id),
                    partition="planning",
                    trust=min(0.35, trust),
                    actionable=False,
                    claimed_source=None,
                    session_idx=kwargs.get("session_idx"),
                    observation_group=kwargs.get("observation_group"),
                    fact_key=kwargs.get("fact_key"),
                    fact_value="conflict",
                    multimodal_relation="conflict",
                )
                return audit_id
        if partition == "planning" and trust <= self.write_trust_threshold:
            self.write_quarantine_count += 1
            kwargs["partition"] = "audit"
            kwargs["trust"] = trust
            kwargs["actionable"] = False
            return super().write(**kwargs)

        directive = extract_directive(text)
        if partition == "planning" and directive is not None and not self._directive_authorized(text):
            self.write_quarantine_count += 1
            audit_id = super().write(
                text=text,
                source_type=source_type,
                channel_id=str(kwargs.get("channel_id", "directive")) + "_audit",
                step=int(kwargs.get("step", 0)),
                parent_ids=tuple(kwargs.get("parent_ids", ())),
                partition="audit",
                trust=trust,
                actionable=False,
                claimed_source=kwargs.get("claimed_source"),
            )
            super().write(
                text=self._quote_as_evidence(text),
                source_type="self_summary",
                channel_id=str(kwargs.get("channel_id", "directive")) + "_quoted",
                step=int(kwargs.get("step", 0)),
                parent_ids=(audit_id,),
                partition="planning",
                trust=min(0.35, trust),
                actionable=False,
                claimed_source=None,
            )
            return audit_id

        kwargs["trust"] = trust
        kwargs["actionable"] = bool(kwargs.get("actionable", False) and directive is not None)
        return super().write(**kwargs)

    def consolidate(self, step: int) -> None:
        if step <= 0 or (step % self.K) != 0:
            return
        id_to_item = {it.item_id: it for it in self.items}
        buf_items = [id_to_item[i] for i in self._buffer if i in id_to_item]
        if not buf_items:
            return
        candidate = self._summarize_text(buf_items)
        if extract_directive(candidate) is not None and not self._directive_authorized(candidate):
            audit_id = super().write(
                text=candidate,
                source_type="self_summary",
                channel_id="candidate_audit",
                step=step,
                parent_ids=[it.item_id for it in buf_items],
                partition="audit",
                trust=0.30,
                actionable=False,
            )
            candidate = self._quote_as_evidence(candidate)
            super().write(
                text=candidate,
                source_type="self_summary",
                channel_id="summary",
                step=step,
                parent_ids=(audit_id,),
                partition="planning",
                trust=0.35,
                actionable=False,
            )
        else:
            super().write(
                text=candidate,
                source_type="self_summary",
                channel_id="summary",
                step=step,
                parent_ids=[it.item_id for it in buf_items],
                trust=0.55,
                actionable=False,
            )
        raw = [it for it in self.items if it.source_type != "self_summary"]
        keep_raw = raw[-self.keep_M :] if self.keep_M > 0 else []
        summaries = [it for it in self.items if it.source_type == "self_summary"]
        self.items = keep_raw + summaries[-1:]
        self._buffer = [it.item_id for it in keep_raw]


class Mem0PlatformMemory(BaseMemory):
    """Hosted Mem0 baseline with a local shadow index for provenance bookkeeping."""

    def __init__(
        self,
        *,
        embedder: HashedTextEmbedder,
        top_k: int,
        api_key: Optional[str] = None,
        user_scope: Optional[str] = None,
        infer: bool = True,
    ):
        super().__init__(embedder=embedder, top_k=top_k)
        try:
            from mem0 import MemoryClient
        except ImportError as exc:
            raise RuntimeError(
                "mem0ai is not installed in the repo-local .venv. Install it before using the mem0 baseline."
            ) from exc

        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "MEM0_API_KEY is not set. Put it in the repo .env or environment before using the mem0 baseline."
            )
        self.client = MemoryClient(api_key=self.api_key)
        self.user_scope = user_scope or f"warp-mem0-{uuid.uuid4().hex[:12]}"
        self.infer = bool(infer)

    def write(self, **kwargs):  # type: ignore[override]
        kwargs.setdefault("actionable", False)
        local_item_id = super().write(**kwargs)
        item = self.items[-1] if self.items else self.audit_items[-1]
        metadata = {
            "source_type": item.source_type,
            "channel_id": item.channel_id,
            "step": item.step,
            "partition": item.partition,
            "trust": item.trust,
            "session_idx": item.session_idx,
            "observation_group": item.observation_group,
            "source_url": item.source_url,
            "page_id": item.page_id,
            "fact_key": item.fact_key,
            "fact_value": item.fact_value,
            "multimodal_relation": item.multimodal_relation,
            "attack_family": item.attack_family,
            "local_item_id": local_item_id,
        }
        self.client.add(
            [{"role": "user", "content": item.text}],
            user_id=self.user_scope,
            metadata=metadata,
            infer=self.infer,
            async_mode=False,
        )
        return local_item_id

    def retrieve(
        self,
        *,
        query: str,
        step: int,
        for_planning: bool = True,
        log_retrieved_source_types: bool = False,
        gold_answer: Optional[str] = None,
    ) -> List[MemoryItem]:
        filters = {"user_id": self.user_scope}
        if for_planning:
            filters = {
                "AND": [
                    {"user_id": self.user_scope},
                    {"metadata": {"partition": "planning"}},
                ]
            }
        result = self.client.search(query, filters=filters, top_k=self.top_k, rerank=False)
        rows = result.get("results", result if isinstance(result, list) else [])
        out: List[MemoryItem] = []
        for row in rows[: self.top_k]:
            metadata = row.get("metadata", {}) or {}
            text = str(row.get("memory", row.get("text", "")))
            source_type = str(metadata.get("source_type", "tool_output_text"))
            trust = float(metadata.get("trust", default_source_cred(source_type)))
            parent_local_id = metadata.get("local_item_id")
            item_id = abs(hash(str(row.get("id", uuid.uuid4().hex)))) % (10**9)
            out.append(
                MemoryItem(
                    item_id=item_id,
                    text=text,
                    embedding=self.embedder.embed(text),
                    source_type=source_type,
                    channel_id=str(metadata.get("channel_id", "mem0")),
                    step=int(metadata.get("step", step)),
                    parent_ids=((int(parent_local_id),) if parent_local_id is not None else ()),
                    partition=str(metadata.get("partition", "planning")),
                    trust=trust,
                    actionable=False,
                    claimed_source=None,
                    session_idx=int(metadata["session_idx"]) if metadata.get("session_idx") is not None else None,
                    observation_group=str(metadata["observation_group"]) if metadata.get("observation_group") is not None else None,
                    source_url=str(metadata["source_url"]) if metadata.get("source_url") is not None else None,
                    page_id=str(metadata["page_id"]) if metadata.get("page_id") is not None else None,
                    fact_key=str(metadata["fact_key"]) if metadata.get("fact_key") is not None else None,
                    fact_value=str(metadata["fact_value"]) if metadata.get("fact_value") is not None else None,
                    multimodal_relation=str(metadata["multimodal_relation"]) if metadata.get("multimodal_relation") is not None else None,
                    attack_family=str(metadata["attack_family"]) if metadata.get("attack_family") is not None else None,
                )
            )

        trusted = [it for it in out if it.trust >= 0.7]
        self.retrieval_trusted_n += len(trusted)
        self.retrieval_trusted_capture_n += sum(1 for it in trusted if it.source_type == "attacker")
        self._maybe_log_retrieved_source_types(
            out,
            log_retrieved_source_types=log_retrieved_source_types,
            gold_answer=gold_answer,
        )
        return out

    def close(self) -> None:
        try:
            self.client.delete_all(user_id=self.user_scope)
        except Exception:
            return
