"""
Langfuse client wrapper with no-op fallback when credentials are absent.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional


@dataclass
class SpanRecord:
    """In-memory span record (used by no-op and Langfuse backends)."""

    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_ms: float = 0.0
    end_ms: Optional[float] = None
    status: str = "ok"
    retries: int = 0


class LangfuseTracer:
    """
    Thin Langfuse wrapper.

    Records inputs, outputs, model parameters, evidence chunk ids, scores,
    latency, retries, and failure state when available.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.enabled = (
            enabled
            if enabled is not None
            else bool(self.public_key and self.secret_key)
        )
        self._client = None
        self._spans: list[SpanRecord] = []

        if self.enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
            except ImportError:
                self.enabled = False

    @contextmanager
    def span(
        self,
        name: str,
        trace_id: str,
        parent_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanRecord, None, None]:
        record = SpanRecord(
            name=name,
            trace_id=trace_id,
            parent_id=parent_id,
            inputs=inputs or {},
            metadata=metadata or {},
            start_ms=time.time() * 1000,
        )
        lf_span = None
        if self._client is not None:
            try:
                lf_span = self._client.span(
                    name=name,
                    trace_id=trace_id,
                    parent_observation_id=parent_id,
                    input=inputs,
                    metadata=metadata,
                )
            except Exception:
                lf_span = None

        try:
            yield record
            record.status = "ok"
        except Exception as exc:
            record.status = "error"
            record.metadata["failure"] = str(exc)
            raise
        finally:
            record.end_ms = time.time() * 1000
            record.metadata["latency_ms"] = record.end_ms - record.start_ms
            self._spans.append(record)
            if lf_span is not None:
                try:
                    lf_span.end(
                        output=record.outputs,
                        metadata=record.metadata,
                        level="ERROR" if record.status == "error" else "DEFAULT",
                    )
                except Exception:
                    pass

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass

    @property
    def spans(self) -> list[SpanRecord]:
        return list(self._spans)


_tracer: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer
