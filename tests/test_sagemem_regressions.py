import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import SAGEMemConfig
from embedding import HashedTextEmbedder
from memory import ConstructorGuardedSandboxMemory
from trust_calibration import BayesianChannelTrust


class _StubDetector:
    def score(self, text: str) -> float:
        return 0.0

    def regex_hits(self, text: str) -> int:
        return 0


def _make_memory():
    cfg = SAGEMemConfig.from_file(REPO_ROOT / "configs" / "default_trust_config.json")
    embedder = HashedTextEmbedder(dim=32, seed=0)
    detector = _StubDetector()
    bayes = BayesianChannelTrust(cfg)
    mem = ConstructorGuardedSandboxMemory(
        embedder=embedder,
        top_k=4,
        consolidation_period_K=4,
        keep_last_M_raw=4,
        procedural_classifier_threshold=0.6,
        quarantine_on_fail=True,
        detector=detector,
        rewrite_on_fail=True,
        cfg=cfg,
        bayes_trust=bayes,
    )
    return cfg, bayes, mem


class SAGEMemRegressionTests(unittest.TestCase):
    def test_reactive_threshold_tightens_after_quarantines(self):
        cfg = SAGEMemConfig.from_file(REPO_ROOT / "configs" / "default_trust_config.json")
        bayes = BayesianChannelTrust(cfg)

        theta0 = bayes.reactive_threshold(0)
        theta3 = bayes.reactive_threshold(3)

        self.assertTrue(math.isclose(theta0, cfg.thresholds.write_trust_threshold))
        self.assertGreater(theta3, theta0)
        self.assertLessEqual(theta3, 1.0)

    def test_accepted_write_persists_computed_trust(self):
        _, bayes, mem = _make_memory()
        channel_id = "ch_tool_1"
        source_type = "tool_output_text"

        for _ in range(3):
            bayes.observe(channel_id, source_type, was_correct=False)

        expected_trust = bayes.get_trust(channel_id, source_type)
        self.assertLess(expected_trust, 0.7)

        item_id = mem.write(
            text="The venue capacity is 30,150 seats.",
            source_type=source_type,
            channel_id=channel_id,
            step=1,
            partition="planning",
            fact_key="venue_capacity",
            fact_value="30150",
        )

        stored = next(it for it in mem.items if it.item_id == item_id)
        self.assertTrue(
            math.isclose(stored.trust, expected_trust, rel_tol=1e-6, abs_tol=1e-6)
        )

    def test_bayesian_channel_trust_state_is_fresh_per_instance(self):
        cfg = SAGEMemConfig.from_file(REPO_ROOT / "configs" / "default_trust_config.json")
        first = BayesianChannelTrust(cfg)
        second = BayesianChannelTrust(cfg)

        first.observe("shared_channel", "tool_output_text", was_correct=False)
        first.observe("shared_channel", "tool_output_text", was_correct=False)

        self.assertLess(
            first.get_trust("shared_channel", "tool_output_text"),
            second.get_trust("shared_channel", "tool_output_text"),
        )

    def test_reactive_tightening_is_channel_local_in_memory(self):
        _, _, mem = _make_memory()

        for _ in range(3):
            mem._record_write_quarantine("ch_a", "tool_output_text")

        self.assertEqual(mem._channel_quarantine_count("ch_a", "tool_output_text"), 3)
        self.assertEqual(mem._channel_quarantine_count("ch_b", "tool_output_text"), 0)


if __name__ == "__main__":
    unittest.main()
