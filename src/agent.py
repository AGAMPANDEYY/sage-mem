import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from memory import BaseMemory, extract_directive, extract_label

@dataclass
class StepResult:
    abstained: bool
    confirmation_requested: bool
    hijack_attempted: bool
    policy_violation: bool

class SimpleCountingAgent:
    """Counts labels across steps; can be misled by poisoned memory."""

    def __init__(
        self,
        *,
        memory: BaseMemory,
        classes: List[str],
        rng: random.Random,
        trust_accept_threshold: float,
        abstain_on_low_trust: bool,
    ):
        self.memory = memory
        self.classes = classes
        self.rng = rng
        self.trust_accept_threshold = float(trust_accept_threshold)
        self.abstain_on_low_trust = bool(abstain_on_low_trust)
        self.counts: Dict[str, float] = {c: 0.0 for c in classes}
        self._abstention_steps = 0

    @property
    def abstention_steps(self) -> int:
        return self._abstention_steps

    def observe_and_write(
        self, *, ocr_text: str, tool_text: Optional[str], step: int
    ) -> None:
        self.memory.write(
            text=ocr_text, source_type="ocr_text", channel_id=f"ocr_{step}", step=step
        )
        if tool_text is not None:
            self.memory.write(
                text=tool_text,
                source_type="tool_output_text",
                channel_id=f"tool_{step}",
                step=step,
            )

    def _choose_label_from_retrieval(
        self, retrieved: List, fallback: Optional[str]
    ) -> Tuple[Optional[str], float]:
        best_lbl = None
        best_w = -1.0
        for it in retrieved:
            lbl = extract_label(it.text)
            if lbl is None:
                continue
            w = float(getattr(it, "trust", 0.5))
            if w > best_w:
                best_w = w
                best_lbl = lbl
        if best_lbl is None:
            return fallback, 0.0
        return best_lbl, best_w

    def step(self, *, step: int, ocr_text: str) -> StepResult:
        retrieved = self.memory.retrieve(
            query="OBJECT_LABEL:", step=step, for_planning=True
        )
        obs_lbl = extract_label(ocr_text)
        chosen_lbl, w = self._choose_label_from_retrieval(retrieved, obs_lbl)

        abstain = False
        if chosen_lbl is None:
            abstain = True
        elif self.abstain_on_low_trust and w < self.trust_accept_threshold and obs_lbl is None:
            abstain = True

        if abstain:
            self._abstention_steps += 1
        else:
            if chosen_lbl in self.counts:
                self.counts[chosen_lbl] += 1.0

        hijack_attempted = False
        policy_violation = False
        for it in retrieved:
            d = extract_directive(it.text)
            if d is None:
                continue
            (at_step, tool, payload) = d
            if at_step == step and tool == "external_write":
                hijack_attempted = True
                policy_violation = True
                break

        return StepResult(
            abstained=abstain,
            confirmation_requested=False,
            hijack_attempted=hijack_attempted,
            policy_violation=policy_violation,
        )

    def finalize_majority_label(self) -> str:
        items = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[0][0] if items else self.classes[0]

class RiskSensitiveActionFirewallAgent(SimpleCountingAgent):
    def __init__(
        self,
        *,
        memory: BaseMemory,
        classes: List[str],
        rng: random.Random,
        trust_accept_threshold: float,
        abstain_on_low_trust: bool,
        high_risk_requires_corroboration: int,
        require_user_confirmation: bool,
        max_confirmation_rate: float,
        user_confirmation_token_probability: float,
        allow_corroboration: bool = True,
    ):
        super().__init__(
            memory=memory,
            classes=classes,
            rng=rng,
            trust_accept_threshold=trust_accept_threshold,
            abstain_on_low_trust=abstain_on_low_trust,
        )
        self.high_risk_requires_corroboration = int(high_risk_requires_corroboration)
        self.require_user_confirmation = bool(require_user_confirmation)
        self.max_confirmation_rate = float(max_confirmation_rate)
        self.user_confirmation_token_probability = float(user_confirmation_token_probability)
        self.allow_corroboration = bool(allow_corroboration)
        self.confirmations_requested = 0
        self.total_steps = 0

    def _has_confirmation_token(self) -> bool:
        return self.rng.random() < self.user_confirmation_token_probability

    def _corroborated(self, directive_text: str, retrieved: List) -> bool:
        if not self.allow_corroboration:
            return False
        chans = set()
        for it in retrieved:
            if directive_text in it.text and float(getattr(it, "trust", 0.0)) >= 0.7:
                chans.add(getattr(it, "channel_id", ""))
        return len(chans) >= self.high_risk_requires_corroboration

    def step(self, *, step: int, ocr_text: str) -> StepResult:
        self.total_steps += 1
        retrieved = self.memory.retrieve(
            query="OBJECT_LABEL:", step=step, for_planning=True
        )
        obs_lbl = extract_label(ocr_text)
        chosen_lbl, w = self._choose_label_from_retrieval(retrieved, obs_lbl)

        abstain = False
        if chosen_lbl is None:
            abstain = True
        elif self.abstain_on_low_trust and w < self.trust_accept_threshold and obs_lbl is None:
            abstain = True

        if abstain:
            self._abstention_steps += 1
        else:
            if chosen_lbl in self.counts:
                self.counts[chosen_lbl] += 1.0

        hijack_attempted = False
        policy_violation = False
        confirmation_requested = False

        for it in retrieved:
            d = extract_directive(it.text)
            if d is None:
                continue
            (at_step, tool, payload) = d
            if at_step != step or tool != "external_write":
                continue

            hijack_attempted = True
            directive_text = f"ACTION_DIRECTIVE: at_step={at_step} tool={tool} payload={payload}"

            authorized = False
            if self._corroborated(directive_text, retrieved):
                authorized = True
            elif self.require_user_confirmation:
                if self.total_steps > 0 and (self.confirmations_requested / self.total_steps) <= self.max_confirmation_rate:
                    confirmation_requested = True
                    self.confirmations_requested += 1
                    if self._has_confirmation_token():
                        authorized = True

            policy_violation = bool(authorized)
            break

        return StepResult(
            abstained=abstain,
            confirmation_requested=confirmation_requested,
            hijack_attempted=hijack_attempted,
            policy_violation=policy_violation,
        )