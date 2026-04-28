"""
Unit tests for monitoring classes: CoTMonitor, ToTMonitor, AhaMonitor.
"""

import pytest

from src.monitoring.cot_monitor import CoTMonitor
from src.monitoring.tot_monitor import ToTMonitor
from src.monitoring.aha_monitor import AhaMonitor


# ---------------------------------------------------------------------------
# Sample outputs
# ---------------------------------------------------------------------------

COT_TEXT = (
    "Step 1: Identify the variables. "
    "Step 2: Apply the formula. "
    "Step 3: Compute the result. "
    "Therefore: The answer is 42."
)

TOT_TEXT = (
    "Option 1: Use brute force — O(n^2). "
    "Option 2: Use dynamic programming — O(n). "
    "Option 3: Use a greedy algorithm — O(n log n). "
    "Best option: Option 2 for optimal time complexity."
)

AHA_TEXT = (
    "I was computing this incorrectly. "
    "Wait, actually — the constraint flips the inequality. "
    "I see now that the answer must be negative."
)

PLAIN_TEXT = "The answer is 5."


# ---------------------------------------------------------------------------
# CoTMonitor tests
# ---------------------------------------------------------------------------

class TestCoTMonitor:
    def test_extract_has_cot(self):
        m = CoTMonitor(log_every=0)
        trace = m.extract(COT_TEXT)
        assert trace.has_cot is True

    def test_extract_no_cot(self):
        m = CoTMonitor(log_every=0)
        trace = m.extract(PLAIN_TEXT)
        assert trace.has_cot is False

    def test_extract_step_count(self):
        m = CoTMonitor(log_every=0)
        trace = m.extract(COT_TEXT)
        assert trace.step_count >= 3

    def test_conclusion_detected(self):
        m = CoTMonitor(log_every=0)
        trace = m.extract(COT_TEXT)
        conclusions = [s for s in trace.steps if s.is_conclusion]
        assert len(conclusions) >= 1

    def test_update_returns_traces(self):
        m = CoTMonitor(log_every=0)
        traces = m.update([COT_TEXT, PLAIN_TEXT])
        assert len(traces) == 2

    def test_get_stats_after_update(self):
        m = CoTMonitor(log_every=0)
        m.update([COT_TEXT, PLAIN_TEXT])
        stats = m.get_stats()
        assert stats["total_samples"] == 2
        assert stats["cot_detected"] >= 1
        assert 0.0 <= stats["cot_rate"] <= 1.0

    def test_reset_clears_state(self):
        m = CoTMonitor(log_every=0)
        m.update([COT_TEXT])
        m.reset()
        stats = m.get_stats()
        assert stats["total_samples"] == 0

    def test_to_dict(self):
        m = CoTMonitor(log_every=0)
        trace = m.extract(COT_TEXT)
        d = trace.to_dict()
        assert "has_cot" in d
        assert "steps" in d

    def test_empty_stats(self):
        m = CoTMonitor(log_every=0)
        stats = m.get_stats()
        assert stats["total_samples"] == 0
        assert stats["cot_rate"] == 0.0

    def test_sample_ids_forwarded(self):
        m = CoTMonitor(log_every=0)
        traces = m.update([COT_TEXT], sample_ids=["id_001"])
        assert traces[0].sample_id == "id_001"


# ---------------------------------------------------------------------------
# ToTMonitor tests
# ---------------------------------------------------------------------------

class TestToTMonitor:
    def test_extract_has_tot(self):
        m = ToTMonitor(log_every=0)
        trace = m.extract(TOT_TEXT)
        assert trace.has_tot is True

    def test_extract_no_tot(self):
        m = ToTMonitor(log_every=0)
        trace = m.extract(PLAIN_TEXT)
        assert trace.has_tot is False

    def test_update_returns_traces(self):
        m = ToTMonitor(log_every=0)
        traces = m.update([TOT_TEXT, PLAIN_TEXT])
        assert len(traces) == 2

    def test_get_stats(self):
        m = ToTMonitor(log_every=0)
        m.update([TOT_TEXT, PLAIN_TEXT])
        stats = m.get_stats()
        assert stats["total_samples"] == 2
        assert 0.0 <= stats["tot_rate"] <= 1.0

    def test_reset(self):
        m = ToTMonitor(log_every=0)
        m.update([TOT_TEXT])
        m.reset()
        assert m.get_stats()["total_samples"] == 0

    def test_to_dict(self):
        m = ToTMonitor(log_every=0)
        trace = m.extract(TOT_TEXT)
        d = trace.to_dict()
        assert "has_tot" in d
        assert "nodes" in d

    def test_empty_stats(self):
        m = ToTMonitor(log_every=0)
        stats = m.get_stats()
        assert stats["total_samples"] == 0
        assert stats["tot_rate"] == 0.0


# ---------------------------------------------------------------------------
# AhaMonitor tests
# ---------------------------------------------------------------------------

class TestAhaMonitor:
    def test_extract_has_aha(self):
        m = AhaMonitor(log_every=0)
        trace = m.extract(AHA_TEXT)
        assert trace.has_aha is True

    def test_extract_no_aha(self):
        m = AhaMonitor(log_every=0)
        trace = m.extract(PLAIN_TEXT)
        assert trace.has_aha is False

    def test_update_returns_traces(self):
        m = AhaMonitor(log_every=0)
        traces = m.update([AHA_TEXT, PLAIN_TEXT])
        assert len(traces) == 2

    def test_get_stats(self):
        m = AhaMonitor(log_every=0)
        m.update([AHA_TEXT, PLAIN_TEXT])
        stats = m.get_stats()
        assert stats["total_samples"] == 2
        assert 0.0 <= stats["aha_rate"] <= 1.0

    def test_reset(self):
        m = AhaMonitor(log_every=0)
        m.update([AHA_TEXT])
        m.reset()
        assert m.get_stats()["total_samples"] == 0

    def test_to_dict(self):
        m = AhaMonitor(log_every=0)
        trace = m.extract(AHA_TEXT)
        d = trace.to_dict()
        assert "has_aha" in d
        assert "moments" in d

    def test_empty_stats(self):
        m = AhaMonitor(log_every=0)
        stats = m.get_stats()
        assert stats["total_samples"] == 0
        assert stats["aha_rate"] == 0.0

    def test_deduplication(self):
        m = AhaMonitor(log_every=0)
        # Text with two very close triggers should not double-count within 30 chars
        close_text = "Aha! I see now."
        trace = m.extract(close_text)
        # Should detect at least 1 but avoid massive over-counting
        assert trace.moment_count >= 1

    def test_pattern_breakdown_populated(self):
        m = AhaMonitor(log_every=0)
        m.update([AHA_TEXT])
        stats = m.get_stats()
        assert isinstance(stats["pattern_breakdown"], dict)
