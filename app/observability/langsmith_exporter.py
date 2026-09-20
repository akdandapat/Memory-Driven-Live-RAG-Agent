"""Optional LangSmith export.

Local SQLite tracing is always on and powers the UI. LangSmith is an *additional* sink, enabled
with LANGSMITH_TRACING=true and an API key.

The export mirrors the real shape of a run: one root chain, one child span per stage
(memory_retrieval, query_rewrite, planning, execution, sub_question:*, synthesis,
memory_update) and one child span per tool call, so the whole path is inspectable rather than a
single opaque root. LangSmith identifies runs by UUID, so internal ids are mapped to UUIDs here
rather than leaking LangSmith's requirements into the rest of the system.

The SDK is imported lazily, every payload is redacted, and no export failure can affect a
request.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.config import Settings
from app.security.sanitize import redact_secrets

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LangSmithExporter:
    def __init__(self, settings: Settings):
        self.enabled = False
        self.project = settings.langsmith_project
        self._client = None
        self._uuids: dict[str, uuid.UUID] = {}
        self._root: Optional[uuid.UUID] = None

        if not (settings.langsmith_tracing and settings.langsmith_api_key):
            return
        try:
            from langsmith import Client  # type: ignore

            self._client = Client(api_key=settings.langsmith_api_key,
                                  api_url=settings.langsmith_endpoint)
            self.enabled = True
            logger.info("LangSmith export enabled (project=%s)", self.project)
        except Exception as exc:  # noqa: BLE001 - observability must never break the request
            logger.warning("LangSmith export disabled: %s", exc)

    # --- id mapping --------------------------------------------------------
    def _uuid_for(self, key: str) -> uuid.UUID:
        """Stable UUID per internal id, so retries and updates address the same run."""
        if key not in self._uuids:
            self._uuids[key] = uuid.uuid4()
        return self._uuids[key]

    # --- root --------------------------------------------------------------
    def start_run(self, *, run_id: str, name: str, run_type: str, inputs: dict[str, Any],
                  parent_run_id: Optional[str] = None) -> None:
        if not self.enabled or self._client is None:
            return
        self._root = self._uuid_for(run_id)
        try:
            self._client.create_run(
                id=self._root, name=name, run_type=run_type,
                inputs=redact_secrets(inputs), project_name=self.project,
                start_time=_now(),
                parent_run_id=self._uuid_for(parent_run_id) if parent_run_id else None,
                extra={"metadata": {"internal_run_id": run_id}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("LangSmith create_run failed: %s", exc)

    def end_run(self, *, run_id: str, outputs: dict[str, Any], error: str = "") -> None:
        if not self.enabled or self._client is None:
            return
        try:
            self._client.update_run(
                self._uuid_for(run_id), outputs=redact_secrets(outputs),
                error=error or None, end_time=_now(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("LangSmith update_run failed: %s", exc)

    # --- child spans -------------------------------------------------------
    def record_child(
        self,
        *,
        child_id: str,
        name: str,
        run_type: str,
        inputs: Any,
        outputs: Any,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        error: str = "",
    ) -> None:
        """Write one completed child span under the current root.

        Called after the step finishes, so it creates and closes in one pair of calls - the
        step's real duration is preserved through start_time/end_time.
        """
        if not self.enabled or self._client is None or self._root is None:
            return
        child_uuid = self._uuid_for(child_id)
        started = start_time or _now()
        try:
            self._client.create_run(
                id=child_uuid, name=name, run_type=run_type,
                inputs=redact_secrets(_as_dict(inputs)),
                project_name=self.project, parent_run_id=self._root, start_time=started,
            )
            self._client.update_run(
                child_uuid, outputs=redact_secrets(_as_dict(outputs)),
                error=error or None, end_time=end_time or _now(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("LangSmith child span failed (%s): %s", name, exc)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}
