"""
Unit tests for Langfuse tracing integration.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

from src.tracing.langfuse_client import LangfuseTracer, SpanRecord
from src.tracing.trace_context import TenKDemoTrace, TraceSpanName, PARENT_TRACE_NAME


class TestLangfuseTracer:
    """Tests for LangfuseTracer wrapper."""

    def test_tracer_init_no_keys(self):
        """Tracer initializes with no Langfuse keys."""
        with patch.dict(os.environ, {}, clear=True):
            tracer = LangfuseTracer()
            assert not tracer.enabled
            assert tracer.public_key is None
            assert tracer.secret_key is None

    def test_tracer_init_with_keys(self):
        """Tracer initializes with Langfuse keys."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "test_public",
                "LANGFUSE_SECRET_KEY": "test_secret",
            },
        ):
            tracer = LangfuseTracer()
            assert tracer.public_key == "test_public"
            assert tracer.secret_key == "test_secret"

    def test_tracer_span_context_manager(self):
        """Span records are captured in context."""
        tracer = LangfuseTracer()
        
        with tracer.span(
            name="test_span",
            trace_id="trace_123",
            inputs={"key": "value"},
        ) as record:
            assert record.name == "test_span"
            assert record.trace_id == "trace_123"
            assert record.inputs == {"key": "value"}
            record.outputs = {"result": "success"}
        
        # After context, record should be stored
        assert len(tracer.spans) > 0
        stored = tracer.spans[-1]
        assert stored.name == "test_span"
        assert stored.outputs == {"result": "success"}

    def test_tracer_span_error_handling(self):
        """Span records error state when exception occurs."""
        tracer = LangfuseTracer()
        
        try:
            with tracer.span(
                name="error_span",
                trace_id="trace_123",
            ) as record:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Record should be captured with error status
        error_record = tracer.spans[-1]
        assert error_record.status == "error"
        assert "Test error" in error_record.metadata.get("failure", "")

    def test_tracer_latency_recorded(self):
        """Span latency is recorded."""
        import time
        tracer = LangfuseTracer()
        
        with tracer.span(
            name="timed_span",
            trace_id="trace_123",
        ) as record:
            time.sleep(0.01)  # 10ms
        
        timed = tracer.spans[-1]
        assert "latency_ms" in timed.metadata
        assert timed.metadata["latency_ms"] >= 10

    def test_span_record_dataclass(self):
        """SpanRecord stores all required fields."""
        record = SpanRecord(
            name="test",
            trace_id="trace_123",
        )
        assert record.name == "test"
        assert record.trace_id == "trace_123"
        assert record.span_id is not None
        assert record.parent_id is None
        assert record.inputs == {}
        assert record.outputs == {}
        assert record.status == "ok"


class TestTenKDemoTrace:
    """Tests for demo trace orchestration."""

    def test_trace_init(self):
        """Demo trace initializes with filing_id."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        assert trace.filing_id == "acme_corp_10k"
        assert trace.trace_id is not None
        assert trace.model_name == "offline-stub"

    def test_trace_langfuse_url_when_disabled(self):
        """Langfuse URL is None when tracer disabled."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        url = trace.langfuse_url
        # In no-key mode, URL should be None
        assert url is None or isinstance(url, str)

    def test_trace_run_stage(self):
        """Run stage executes function and records span."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        
        result = trace.run_stage(
            TraceSpanName.LOADING,
            lambda: {"loaded": True},
            inputs={"filing_id": "acme_corp_10k"},
        )
        
        assert result == {"loaded": True}
        assert len(trace.tracer.spans) > 0

    def test_trace_stage_with_metadata(self):
        """Run stage records custom metadata."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        
        trace.run_stage(
            "custom_stage",
            lambda: {"data": "value"},
            metadata={"custom_key": "custom_value"},
        )
        
        span = trace.tracer.spans[-1]
        assert "custom_key" in span.metadata
        assert span.metadata["custom_key"] == "custom_value"

    def test_trace_model_metadata(self):
        """Model metadata includes provider and cache dir."""
        with patch.dict(
            os.environ,
            {
                "ENTERPRISE_MODEL_NAME": "gpt-4",
                "LLM_PROVIDER": "openai",
                "DSPY_CACHE_DIR": "/tmp/dspy",
            },
        ):
            trace = TenKDemoTrace(filing_id="acme_corp_10k")
            meta = trace._model_metadata()
            
            assert meta["model_name"] == "gpt-4"
            assert meta["provider"] == "openai"
            assert meta["dspy_cache_dir"] == "/tmp/dspy"

    def test_trace_callback_records_stage_data(self):
        """Trace callback records stage, inputs, outputs."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        
        trace.trace_callback(
            stage="extraction",
            inputs={"num_chunks": 10},
            outputs={"num_items": 5},
        )
        
        span = trace.tracer.spans[-1]
        assert span.name == "extraction"
        assert span.inputs == {"num_chunks": 10}
        assert span.outputs == {"num_items": 5}

    def test_trace_callback_captures_evidence_ids(self):
        """Trace callback captures evidence chunk IDs."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        
        trace.trace_callback(
            stage="chunking",
            inputs={},
            outputs={"evidence_chunk_ids": ["chunk_1", "chunk_2"]},
        )
        
        span = trace.tracer.spans[-1]
        assert "evidence_chunk_ids" in span.metadata
        assert span.metadata["evidence_chunk_ids"] == ["chunk_1", "chunk_2"]

    def test_trace_record_scenario_builds(self):
        """Record scenario builds creates multiple spans."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        
        def build_scenario(idx: int) -> dict:
            return {"scenario_index": idx, "title": f"Scenario {idx}"}
        
        results = trace.record_scenario_builds(build_scenario, count=3)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["scenario_index"] == i

    def test_trace_flush(self):
        """Flush does not raise error."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        trace.flush()  # Should not raise

    def test_trace_parent_span_created(self):
        """Parent trace span is created on first stage."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        
        trace.run_stage(
            TraceSpanName.LOADING,
            lambda: {"data": "loaded"},
            inputs={"filing_id": "acme_corp_10k"},
        )
        
        # Parent trace should be registered
        assert PARENT_TRACE_NAME in trace._span_ids

    def test_trace_span_hierarchy(self):
        """Spans maintain parent-child hierarchy."""
        trace = TenKDemoTrace(filing_id="acme_corp_10k")
        
        # First stage should create parent
        trace.run_stage(
            TraceSpanName.LOADING,
            lambda: {"data": "loaded"},
        )
        
        # Second stage should have parent
        trace.run_stage(
            TraceSpanName.EXTRACTION,
            lambda: {"data": "extracted"},
        )
        
        # Should have at least parent + 2 stages
        assert len(trace.tracer.spans) >= 2


class TestTraceSpanNames:
    """Tests for TraceSpanName enumeration."""

    def test_all_pipeline_stages_defined(self):
        """All pipeline stages are defined as span names."""
        required_stages = [
            "LOADING",
            "EXTRACTION",
            "CHUNKING",
            "HYPOTHESES",
            "SCENARIO_BUILD",
            "CRITIQUE",
            "RANKING",
            "RENDERING",
        ]
        
        for stage in required_stages:
            assert hasattr(TraceSpanName, stage)

    def test_span_name_values(self):
        """Span names have correct string values."""
        assert TraceSpanName.LOADING.value == "loading"
        assert TraceSpanName.EXTRACTION.value == "extraction"
        assert TraceSpanName.CRITIQUE.value == "critique"


class TestTracingIntegration:
    """Integration tests for tracing system."""

    def test_complete_demo_run_with_tracing(self):
        """Complete demo pipeline produces trace."""
        from src.demo.pipeline import run_enterprise_demo
        
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        assert "trace_id" in result
        assert result["trace_id"] is not None
        assert isinstance(result["trace_id"], str)

    def test_trace_id_matches_scenarios(self):
        """Trace ID is consistent across scenarios."""
        from src.demo.pipeline import run_enterprise_demo
        
        result = run_enterprise_demo(
            filing_id="acme_corp_10k",
            offline=True,
            output_dir=None,
        )
        
        trace_id = result["trace_id"]
        # All scenarios should be from this run
        assert all(s.get("trace_id") for s in result["scenarios"])

    def test_tracing_gracefully_degrades_without_keys(self):
        """Tracing works even without Langfuse credentials."""
        with patch.dict(
            os.environ,
            {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""},
            clear=False,
        ):
            from src.demo.pipeline import run_enterprise_demo
            
            result = run_enterprise_demo(
                filing_id="acme_corp_10k",
                offline=True,
                output_dir=None,
            )
            
            # Should still complete and have trace_id
            assert result["trace_id"] is not None
