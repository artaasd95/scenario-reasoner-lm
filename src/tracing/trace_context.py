"""
Parent trace contract for the enterprise risk 10-K demo.

Parent trace name: ``tenk_demo_run``
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.tracing.langfuse_client import LangfuseTracer, get_tracer


class TraceSpanName(str, Enum):
    LOADING = "loading"
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    HYPOTHESES = "hypotheses"
    SCENARIO_BUILD = "scenario_build"
    CRITIQUE = "critique"
    RANKING = "ranking"
    RENDERING = "rendering"


PARENT_TRACE_NAME = "tenk_demo_run"


@dataclass
class TenKDemoTrace:
    """
    Orchestrates spans for a single 10-K demo run.

    Spans: loading, extraction, chunking, hypotheses, five scenario builds,
    critique, ranking, rendering.
    """

    filing_id: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = field(
        default_factory=lambda: os.getenv("ENTERPRISE_MODEL_NAME", "offline-stub")
    )
    tracer: Optional[LangfuseTracer] = None
    _span_ids: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tracer is None:
            self.tracer = get_tracer()

    @property
    def langfuse_url(self) -> Optional[str]:
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
        if self.tracer and self.tracer.enabled:
            return f"{host}/trace/{self.trace_id}"
        return None

    def _model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dspy_cache_dir": os.getenv("DSPY_CACHE_DIR", ""),
            "provider": os.getenv("LLM_PROVIDER", "offline"),
        }

    def run_stage(
        self,
        span_name: TraceSpanName | str,
        fn: Callable[[], Any],
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        name = span_name.value if isinstance(span_name, TraceSpanName) else span_name
        meta = {**self._model_metadata(), **(metadata or {})}
        parent_id = self._span_ids.get(PARENT_TRACE_NAME)

        with self.tracer.span(
            name=name,
            trace_id=self.trace_id,
            parent_id=parent_id,
            inputs=inputs,
            metadata=meta,
        ) as record:
            if PARENT_TRACE_NAME not in self._span_ids and name == TraceSpanName.LOADING.value:
                self._start_parent_trace(inputs)

            result = fn()
            record.outputs = result if isinstance(result, dict) else {"result": result}
            self._span_ids[name] = record.span_id
            return result

    def _start_parent_trace(self, inputs: Optional[Dict[str, Any]]) -> None:
        if self.tracer._client is not None:
            try:
                self.tracer._client.trace(
                    id=self.trace_id,
                    name=PARENT_TRACE_NAME,
                    input=inputs or {"filing_id": self.filing_id},
                    metadata=self._model_metadata(),
                )
            except Exception:
                pass
        self._span_ids[PARENT_TRACE_NAME] = self.trace_id

    def record_scenario_builds(
        self,
        build_fn: Callable[[int], Dict[str, Any]],
        count: int = 5,
    ) -> List[Dict[str, Any]]:
        results = []
        for i in range(count):
            def _run(idx: int = i) -> Dict[str, Any]:
                return build_fn(idx)

            out = self.run_stage(
                f"{TraceSpanName.SCENARIO_BUILD.value}_{i + 1}",
                lambda: build_fn(i),
                inputs={"index": i + 1},
            )
            results.append(out if isinstance(out, dict) else {"built": True})
        return results

    def trace_callback(self, stage: str, inputs: dict, outputs: dict) -> None:
        """DSPy module callback compatible with pipeline stages."""
        meta = {}
        if "evidence_chunk_ids" in outputs:
            meta["evidence_chunk_ids"] = outputs["evidence_chunk_ids"]
        if "scores" in outputs:
            meta["scores"] = outputs["scores"]

        with self.tracer.span(
            name=stage,
            trace_id=self.trace_id,
            inputs=inputs,
            metadata=meta,
        ) as record:
            record.outputs = outputs

    def flush(self) -> None:
        self.tracer.flush()
