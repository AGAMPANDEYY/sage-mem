import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from embedding import HashedTextEmbedder
from procedural import ProceduralDetector
from utils import safe_cosine_sim

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

def default_source_cred(source_type: str) -> float:
    return {
        "user": 1.0,
        "ocr_text": 0.60,
        "vision_caption": 0.58,
        "tool_output_text": 0.70,
        "self_summary": 0.45,
        "attacker": 0.10,
        "tool_echo": 0.20,
    }.get(source_type, 0.35)

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
    ) -> int:
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
        )
        self._next_id += 1
        if partition == "audit":
            self.audit_items.append(item)
        else:
            self.items.append(item)
        return item.item_id

    def consolidate(self, step: int) -> None:
        return

    def _iter_for_retrieval(self, *, for_planning: bool) -> Iterable[MemoryItem]:
        if for_planning:
            return self.items
        return list(self.items) + list(self.audit_items)

    def retrieve(
        self, *, query: str, step: int, for_planning: bool = True
    ) -> List[MemoryItem]:
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
        return out

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
            agree = 0
            disagree = 0
            for other in self.items:
                if other.item_id == it.item_id:
                    continue
                olbl = extract_label(other.text)
                if olbl is None:
                    continue
                if olbl == lbl:
                    agree += 1
                else:
                    disagree += 1
            consensus = (1.0 + agree) / (1.0 + agree + disagree)

        rel = self.w_source * src + self.w_decay * decay + self.w_consensus * consensus
        rel = max(0.0, min(1.0, float(rel)))
        score = float(sim) * rel
        if not math.isfinite(score):
            return 0.0
        return score

    def retrieve(
        self, *, query: str, step: int, for_planning: bool = True
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
        raw = [it for it in self.items if it.source_type != "self_summary"]
        keep_raw = raw[-self.keep_M :] if self.keep_M > 0 else []
        summaries = [it for it in self.items if it.source_type == "self_summary"]
        self.items = keep_raw + summaries[-1:]
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
    # Write-time trust threshold: items with trust below this are quarantined
    # to the audit partition and excluded from planning retrieval.
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

    def write(self, *, text: str, source_type: str, channel_id: str, step: int,
              parent_ids=(), partition: str = "planning", trust=None,
              actionable: bool = False, claimed_source=None, session_idx: Optional[int] = None) -> int:
        """Write-time gate: items with trust below threshold go to audit partition.

        This is the core write-boundary defense — untrusted observations are
        quarantined before they can enter planning memory, preventing them from
        appearing in retrieval context regardless of retrieval-time scoring.
        """
        computed_trust = _clip01(
            default_source_cred(source_type) if trust is None else float(trust)
        )
        # Route to audit if trust is too low (write-time quarantine)
        if computed_trust < self.WRITE_TRUST_THRESHOLD and partition == "planning":
            self.write_quarantine_count += 1
            return super().write(
                text=text, source_type=source_type, channel_id=channel_id,
                step=step, parent_ids=parent_ids, partition="audit",
                trust=computed_trust, actionable=False, claimed_source=claimed_source,
                session_idx=session_idx,
            )
        return super().write(
            text=text, source_type=source_type, channel_id=channel_id,
            step=step, parent_ids=parent_ids, partition=partition,
            trust=trust, actionable=actionable, claimed_source=claimed_source,
            session_idx=session_idx,
        )

    def constructor_guard(self, candidate: str, lineage: List[MemoryItem]) -> bool:
        if any(it.source_type == "attacker" for it in lineage):
            high_integrity = False
        else:
            avg_trust = float(np.mean([it.trust for it in lineage])) if lineage else 0.0
            high_integrity = avg_trust >= 0.65

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

        self.write(
            text=candidate,
            source_type="self_summary",
            channel_id="summary",
            step=step,
            parent_ids=[it.item_id for it in buf_items],
            trust=0.70,
            actionable=False,
        )
        raw = [it for it in self.items if it.source_type != "self_summary"]
        keep_raw = raw[-self.keep_M :] if self.keep_M > 0 else []
        summaries = [it for it in self.items if it.source_type == "self_summary"]
        self.items = keep_raw + summaries[-1:]
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
        self, *, query: str, step: int, for_planning: bool = True
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
        if partition == "planning" and trust < self.write_trust_threshold:
            self.write_quarantine_count += 1
            kwargs["partition"] = "audit"
            kwargs["actionable"] = False
        return super().write(**kwargs)


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
        self, *, query: str, step: int, for_planning: bool = True
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
                )
            )

        trusted = [it for it in out if it.trust >= 0.7]
        self.retrieval_trusted_n += len(trusted)
        self.retrieval_trusted_capture_n += sum(1 for it in trusted if it.source_type == "attacker")
        return out

    def close(self) -> None:
        try:
            self.client.delete_all(user_id=self.user_scope)
        except Exception:
            return
