import math
import re
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
        return f"SUMMARY: top_labels={label_part}\n{directive_part}".strip()

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
              actionable: bool = False, claimed_source=None) -> int:
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
            )
        return super().write(
            text=text, source_type=source_type, channel_id=channel_id,
            step=step, parent_ids=parent_ids, partition=partition,
            trust=trust, actionable=actionable, claimed_source=claimed_source,
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
        for it in items:
            lbl = extract_label(it.text)
            if lbl is not None:
                labels[lbl] = labels.get(lbl, 0) + 1
        top = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        label_part = ", ".join([f"{k}({v})" for k, v in top]) if top else "none"
        return f"SUMMARY: top_labels={label_part}"

    def write(self, **kwargs):  # type: ignore[override]
        parent_ids = list(kwargs.get("parent_ids", ()))
        source_type = str(kwargs.get("source_type", ""))
        trust = float(kwargs.get("trust", default_source_cred(source_type)))

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
        return super().write(**kwargs)