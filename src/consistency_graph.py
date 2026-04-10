"""
consistency_graph.py — Multi-turn consistency graph for memory write authentication.

Architecture:
  ConsistencyGraph tracks every write to memory as a node.
  Edges represent semantic relationships: CONFIRMS, CONTRADICTS, UPDATES, UNRELATED.

  For user turns specifically:
    When a new user turn arrives that contains correction language,
    CorrectionPlausibilityScorer evaluates it against:
      1. Prior user turns (stylometric consistency)
      2. Planning partition items (evidence grounding)
      3. Correction frequency (behavioural anomaly)

  Output:
    CorrectionScore: plausibility ∈ [0, 1], tier ∈ {GROUNDED, UNGROUNDED, SUSPICIOUS},
    suggested write partition.

Theory:
  Semantic contradiction detection uses cosine similarity in hashed-embedding space.
  This is an approximation — for production, replace HashedTextEmbedder with
  sentence-transformers (paraphrase-MiniLM or similar).

  The key insight: a legitimate user correction specifies a concrete replacement value
  and is topically anchored to existing evidence. An adversarial mimicry attack
  tends to use generic retraction language ("disregard all prior...") without
  providing substantive evidence for the correction.

  Formal basis:
    Claim: P(adversarial | retraction + no_grounding) > P(adversarial | retraction + grounding)
    This follows from the asymmetry of the adversary's goal: the attacker wants to
    erase memory without triggering verification. Specific corrections with evidence
    are harder for an attacker to craft without domain knowledge.

References:
  Perez & Ribeiro (2022) "Ignore Previous Prompt: Attack Techniques For Language Models"
  Zhu et al. (2023) "PromptBench: Towards Evaluating the Robustness of LLMs"
  Greshake et al. (2023) "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications"
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import SAGEMemConfig
from embedding import HashedTextEmbedder, get_sentence_embedder
from utils import safe_cosine_sim


# ---------------------------------------------------------------------------
# Edge relation types
# ---------------------------------------------------------------------------

class EdgeRelation(str, Enum):
    CONFIRMS = "confirms"
    CONTRADICTS = "contradicts"
    UPDATES = "updates"
    UNRELATED = "unrelated"


# ---------------------------------------------------------------------------
# Correction tier (output of plausibility scoring)
# ---------------------------------------------------------------------------

class CorrectionTier(str, Enum):
    GROUNDED = "grounded"        # plausibility ≥ high threshold → full user trust
    UNGROUNDED = "ungrounded"    # low ≤ plausibility < high → write at reduced trust
    SUSPICIOUS = "suspicious"    # plausibility < low OR frequency exceeded → flag


@dataclass
class CorrectionScore:
    """Output of CorrectionPlausibilityScorer."""
    plausibility: float          # ∈ [0, 1]
    tier: CorrectionTier
    evidence_grounded: bool      # True if correction value appears in planning partition
    specificity_score: float     # how specific is the replacement value (0=generic, 1=specific)
    frequency_anomaly: bool      # True if user has submitted too many corrections
    retraction_detected: bool    # True if generic retraction language found
    suggested_trust: float       # trust to assign to this user write
    reasoning: str               # human-readable explanation


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyNode:
    """A single memory write represented as a graph node."""
    node_id: int
    text: str
    embedding: np.ndarray
    source_type: str
    channel_id: str
    step: int
    session_idx: Optional[int]
    fact_key: Optional[str]
    fact_value: Optional[str]
    is_user_turn: bool


# ---------------------------------------------------------------------------
# Graph edge
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyEdge:
    """Semantic relationship between two nodes."""
    from_id: int
    to_id: int
    relation: EdgeRelation
    confidence: float            # ∈ [0, 1], strength of evidence for this relation
    cosine_similarity: float     # raw cosine sim between embeddings
    detected_by: str             # "cosine" | "fact_key_conflict" | "retraction_pattern"


# ---------------------------------------------------------------------------
# Retraction language patterns (loaded from config or these defaults)
# ---------------------------------------------------------------------------

# These are detection signals, NOT defense decisions.
# The CorrectionPlausibilityScorer uses them as features, not rules.
_GENERIC_RETRACTION_PATTERNS = [
    re.compile(r"\bdisregard\s+(all|any|previous|prior|earlier)\b", re.IGNORECASE),
    re.compile(r"\bignore\s+(everything|all|prior|previous|earlier)\b", re.IGNORECASE),
    re.compile(r"\bthe\s+(previous|prior|earlier)\s+answer\s+was\s+wrong\b", re.IGNORECASE),
    re.compile(r"\bplease\s+(ignore|disregard|forget)\b", re.IGNORECASE),
    re.compile(r"\boverride\s+(all|any|previous|prior)\b", re.IGNORECASE),
    re.compile(r"\bstop\s+repeating\b", re.IGNORECASE),
    re.compile(r"\btreat\s+.{0,50}\s+as\s+the\s+(correct|right|true)\b", re.IGNORECASE),
]

_SPECIFIC_CORRECTION_PATTERNS = [
    re.compile(r"\bthe\s+(correct|right|actual|true)\s+(answer|value|result)\s+is\b", re.IGNORECASE),
    re.compile(r"\bshould\s+be\s+[\"']?[\w\s]{1,40}[\"']?\b", re.IGNORECASE),
    re.compile(r"\bit\s+is\s+actually\s+[\"']?[\w\s]{1,40}[\"']?\b", re.IGNORECASE),
]


def _has_generic_retraction(text: str) -> bool:
    return any(p.search(text) for p in _GENERIC_RETRACTION_PATTERNS)


def _has_specific_correction(text: str) -> bool:
    return any(p.search(text) for p in _SPECIFIC_CORRECTION_PATTERNS)


def _specificity_score(text: str) -> float:
    """
    Heuristic specificity: a correction is specific if it states a concrete
    replacement value rather than asking to erase prior content generically.

    Returns ∈ [0, 1]. 1.0 = very specific, 0.0 = purely generic retraction.
    """
    has_generic = _has_generic_retraction(text)
    has_specific = _has_specific_correction(text)

    if has_specific and not has_generic:
        return 0.9
    if has_specific and has_generic:
        return 0.5
    if has_generic and not has_specific:
        return 0.1
    # Neither pattern detected → assume benign correction with medium specificity
    return 0.6


# ---------------------------------------------------------------------------
# Multi-turn consistency graph
# ---------------------------------------------------------------------------

class MultiTurnConsistencyGraph:
    """
    Directed graph of memory writes and their semantic relationships.

    Nodes: every write to memory (planning or audit).
    Edges: detected semantic relationships between nodes.

    The graph is maintained incrementally — O(n) per write for edge computation
    where n is the number of nodes with the same fact_key or overlapping content.

    User-turn specific logic:
      When a user turn arrives with retraction language, the graph checks:
        1. Does any planning-partition node support the correction?
        2. Does this contradict prior user turns?
        3. Has the correction frequency exceeded the session limit?
    """

    # Maximum nodes retained in the sliding window for edge comparison.
    # Older nodes are evicted (LRU by insertion order) beyond this limit.
    # O(max_nodes) comparisons per write instead of O(N).
    MAX_NODES = 128
    # Maximum edges retained per node (prune lowest-confidence edges first).
    MAX_EDGES_PER_NODE = 16

    def __init__(self, cfg: SAGEMemConfig, embedder: HashedTextEmbedder):
        self._cfg = cfg
        self._embedder = embedder
        self._nodes: Dict[int, ConsistencyNode] = {}
        self._edges: List[ConsistencyEdge] = []
        self._node_counter: int = 0

        # Per-session user correction frequency
        self._user_correction_count: int = 0
        # User turn nodes only
        self._user_nodes: List[ConsistencyNode] = []
        # Planning partition nodes (for evidence grounding)
        self._planning_nodes: List[ConsistencyNode] = []
        # Insertion-order tracking for LRU eviction
        self._insertion_order: List[int] = []

    def add_node(
        self,
        *,
        text: str,
        source_type: str,
        channel_id: str,
        step: int,
        session_idx: Optional[int],
        fact_key: Optional[str],
        fact_value: Optional[str],
        partition: str = "planning",
    ) -> ConsistencyNode:
        """
        Add a write event to the graph and compute edges to existing nodes.
        Returns the new node (does not modify memory state — the graph is advisory).
        """
        # S2: use sentence-transformer embeddings for semantic contradiction detection.
        # Catches paraphrased overwrites (e.g. "Sweden" → "France") that hash-based
        # cosine misses because individual word hashes don't encode meaning.
        try:
            emb = get_sentence_embedder().embed(text)
        except Exception:
            emb = self._embedder.embed(text)  # fallback to hash embedder if ST unavailable
        nid = self._node_counter
        self._node_counter += 1

        is_user = (source_type == "user")
        node = ConsistencyNode(
            node_id=nid,
            text=text,
            embedding=emb,
            source_type=source_type,
            channel_id=channel_id,
            step=step,
            session_idx=session_idx,
            fact_key=fact_key,
            fact_value=fact_value,
            is_user_turn=is_user,
        )
        self._nodes[nid] = node
        self._insertion_order.append(nid)

        if is_user:
            self._user_nodes.append(node)
        if partition == "planning":
            self._planning_nodes.append(node)

        # Compute edges to recent relevant nodes (bounded window)
        self._compute_edges(node)

        # Evict oldest nodes if sliding window exceeded (LRU)
        while len(self._nodes) > self.MAX_NODES:
            evict_id = self._insertion_order.pop(0)
            self._nodes.pop(evict_id, None)
            # Prune edges referencing evicted node
            self._edges = [e for e in self._edges
                           if e.from_id != evict_id and e.to_id != evict_id]
            # Prune from list caches (user_nodes, planning_nodes)
            self._user_nodes = [n for n in self._user_nodes if n.node_id != evict_id]
            self._planning_nodes = [n for n in self._planning_nodes if n.node_id != evict_id]

        # Prune excess edges per node (keep highest-confidence ones)
        if len(self._edges) > self.MAX_NODES * self.MAX_EDGES_PER_NODE:
            self._edges.sort(key=lambda e: e.confidence, reverse=True)
            self._edges = self._edges[:self.MAX_NODES * self.MAX_EDGES_PER_NODE]
        return node

    def _compute_edges(self, new_node: ConsistencyNode) -> None:
        """
        Detect semantic relationships between new_node and existing nodes.
        Only considers nodes that are plausibly related (same fact_key or
        cosine similarity above a low threshold).
        """
        contra_thresh = self._cfg.thresholds.consistency_contradiction_cosine
        confirm_thresh = self._cfg.thresholds.consistency_confirm_cosine

        for existing in list(self._nodes.values()):
            if existing.node_id == new_node.node_id:
                continue

            # Skip pairs that are clearly unrelated (fast path)
            cosine = float(safe_cosine_sim(new_node.embedding, existing.embedding))

            # Fact-key conflict detection (highest confidence)
            if (
                new_node.fact_key is not None
                and existing.fact_key is not None
                and new_node.fact_key == existing.fact_key
                and new_node.fact_value != existing.fact_value
            ):
                edge = ConsistencyEdge(
                    from_id=new_node.node_id,
                    to_id=existing.node_id,
                    relation=EdgeRelation.UPDATES,
                    confidence=0.95,
                    cosine_similarity=cosine,
                    detected_by="fact_key_conflict",
                )
                self._edges.append(edge)
                continue

            # Semantic contradiction: similar topic, low cosine (opposing content)
            # A contradiction requires both items to be about the same topic (cosine > 0.1)
            # but have opposing content (cosine < contra_thresh).
            if 0.10 < cosine < contra_thresh:
                # Only flag as contradiction if one of them has retraction language
                has_retraction = _has_generic_retraction(new_node.text) or _has_generic_retraction(existing.text)
                if has_retraction:
                    edge = ConsistencyEdge(
                        from_id=new_node.node_id,
                        to_id=existing.node_id,
                        relation=EdgeRelation.CONTRADICTS,
                        confidence=0.7,
                        cosine_similarity=cosine,
                        detected_by="cosine+retraction",
                    )
                    self._edges.append(edge)
                    continue

            # Semantic confirmation: high cosine → same claim
            if cosine >= confirm_thresh:
                edge = ConsistencyEdge(
                    from_id=new_node.node_id,
                    to_id=existing.node_id,
                    relation=EdgeRelation.CONFIRMS,
                    confidence=min(1.0, cosine),
                    cosine_similarity=cosine,
                    detected_by="cosine",
                )
                self._edges.append(edge)

    def contradictions_for_node(self, node_id: int) -> List[ConsistencyEdge]:
        return [
            e for e in self._edges
            if (e.from_id == node_id or e.to_id == node_id)
            and e.relation == EdgeRelation.CONTRADICTS
        ]

    def confirmations_for_node(self, node_id: int) -> List[ConsistencyEdge]:
        return [
            e for e in self._edges
            if (e.from_id == node_id or e.to_id == node_id)
            and e.relation == EdgeRelation.CONFIRMS
        ]

    def get_channel_consistency_score(self, channel_id: str) -> float:
        """
        Returns a consistency score ∈ [0, 1] for all writes from a given channel.
        1.0 = entirely self-consistent, 0.0 = highly contradictory.

        Used to reduce trust for volatile channels dynamically.
        """
        channel_nodes: List[int] = [
            n.node_id for n in self._nodes.values() if n.channel_id == channel_id
        ]
        if len(channel_nodes) < 2:
            return 1.0  # not enough data

        total_pairs = 0
        contradiction_pairs = 0
        contradiction_node_ids: Set[int] = set(
            nid
            for e in self._edges
            if e.relation == EdgeRelation.CONTRADICTS
            for nid in (e.from_id, e.to_id)
        )
        for nid in channel_nodes:
            if nid in contradiction_node_ids:
                contradiction_pairs += 1
            total_pairs += 1

        return 1.0 - (contradiction_pairs / total_pairs)

    def reset_session(self) -> None:
        """Call between cases/sessions to clear session state."""
        self._nodes.clear()
        self._edges.clear()
        self._node_counter = 0
        self._user_correction_count = 0
        self._user_nodes.clear()
        self._planning_nodes.clear()
        self._insertion_order.clear()


# ---------------------------------------------------------------------------
# Correction plausibility scorer
# ---------------------------------------------------------------------------

class CorrectionPlausibilityScorer:
    """
    Evaluates whether a user correction is plausible or adversarial.

    Does NOT block corrections — outputs a trust modifier and tier.
    The calling code decides how to use it (write to "user_uncertain" partition,
    reduce trust, flag for audit, etc.).

    Three signals:
    1. Evidence grounding: does any planning-partition item support the correction?
    2. Specificity: does the correction state a concrete replacement, or just retract?
    3. Frequency anomaly: has this user submitted an unusual number of corrections?

    Formally:
        plausibility = w_grounding * grounding + w_specificity * specificity
                       - w_frequency * frequency_penalty

    where each component ∈ [0, 1] and weights sum to 1.
    """

    # Weights for the plausibility score components.
    # These should eventually be learned from labelled data.
    W_GROUNDING = 0.50
    W_SPECIFICITY = 0.35
    W_FREQUENCY = 0.15

    def __init__(self, cfg: SAGEMemConfig, graph: MultiTurnConsistencyGraph):
        self._cfg = cfg
        self._graph = graph

    def score(
        self,
        *,
        node: ConsistencyNode,
        planning_nodes: List[ConsistencyNode],
        user_correction_count: int,
    ) -> CorrectionScore:
        """
        Score a new user turn node for correction plausibility.

        node: the ConsistencyNode for the incoming user write (already added to graph).
        planning_nodes: current planning-partition nodes (for evidence grounding).
        user_correction_count: how many corrections this user has submitted so far.
        """
        retraction_detected = _has_generic_retraction(node.text)
        specificity = _specificity_score(node.text)

        # Signal 1: Evidence grounding
        # Is there any planning-partition item that semantically supports
        # the correction (i.e., the corrected value exists in evidence)?
        confirm_thresh = self._cfg.thresholds.consistency_confirm_cosine
        grounding_score = self._compute_grounding(node, planning_nodes, confirm_thresh)

        # Signal 2: Specificity (already computed above)
        specificity_score = specificity

        # Signal 3: Frequency anomaly
        max_freq = self._cfg.thresholds.correction_frequency_max
        frequency_anomaly = user_correction_count > max_freq
        # Frequency penalty: 0 until max_freq, then linearly increases
        frequency_penalty = min(1.0, max(0.0, (user_correction_count - max_freq) / max(1, max_freq)))

        # Composite plausibility
        plausibility = (
            self.W_GROUNDING * grounding_score
            + self.W_SPECIFICITY * specificity_score
            - self.W_FREQUENCY * frequency_penalty
        )
        plausibility = float(max(0.0, min(1.0, plausibility)))

        # Determine tier
        low_t = self._cfg.thresholds.correction_plausibility_low
        high_t = self._cfg.thresholds.correction_plausibility_high

        if plausibility >= high_t:
            tier = CorrectionTier.GROUNDED
        elif plausibility >= low_t:
            tier = CorrectionTier.UNGROUNDED
        else:
            tier = CorrectionTier.SUSPICIOUS

        # Suggested trust
        # GROUNDED → full user trust (1.0)
        # UNGROUNDED → reduced (e.g. 0.65 — above tool_output_text but not full user)
        # SUSPICIOUS → treated as tool_echo level (0.20)
        if tier == CorrectionTier.GROUNDED:
            suggested_trust = 1.0
        elif tier == CorrectionTier.UNGROUNDED:
            suggested_trust = 0.65
        else:
            suggested_trust = 0.20

        # Build human-readable reasoning string
        reasoning = (
            f"plausibility={plausibility:.2f} "
            f"(grounding={grounding_score:.2f} × {self.W_GROUNDING}, "
            f"specificity={specificity_score:.2f} × {self.W_SPECIFICITY}, "
            f"freq_penalty={frequency_penalty:.2f} × {self.W_FREQUENCY}); "
            f"retraction={'yes' if retraction_detected else 'no'}, "
            f"freq_anomaly={'yes' if frequency_anomaly else 'no'}, "
            f"tier={tier.value}"
        )

        return CorrectionScore(
            plausibility=plausibility,
            tier=tier,
            evidence_grounded=(grounding_score >= 0.5),
            specificity_score=specificity_score,
            frequency_anomaly=frequency_anomaly,
            retraction_detected=retraction_detected,
            suggested_trust=suggested_trust,
            reasoning=reasoning,
        )

    def _compute_grounding(
        self,
        node: ConsistencyNode,
        planning_nodes: List[ConsistencyNode],
        confirm_thresh: float,
    ) -> float:
        """
        Returns the maximum cosine similarity between the incoming correction
        and any planning-partition item.

        High similarity = the correction is topically grounded in existing evidence.
        Low similarity = the correction targets something not in evidence (suspicious).
        """
        if not planning_nodes:
            return 0.0

        sims = [
            float(safe_cosine_sim(node.embedding, pn.embedding))
            for pn in planning_nodes
        ]
        max_sim = max(sims)
        # Normalise: 0 at max_sim=0, 1 at max_sim >= confirm_thresh
        return min(1.0, max_sim / max(confirm_thresh, 1e-6))

    def get_and_increment_correction_count(self) -> int:
        """
        Returns the current correction count and increments it.
        Call this when a user correction is detected.
        """
        count = self._graph._user_correction_count
        self._graph._user_correction_count += 1
        return count
