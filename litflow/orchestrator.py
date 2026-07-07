from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents import (
    CitationSnowballAgent,
    DeepSummaryAgent,
    EvidenceAcquisitionAgent,
    IntakeScopeAgent,
    MetadataNormalizeDedupAgent,
    ParallelSearchAgent,
    QualityAuditAgent,
    QueryStrategyAgent,
    RelevanceScreeningAgent,
)
from .connectors import BaseConnector, SemanticScholarConnector, default_connectors
from .models import Record, RunConfig
from .sdk_bridge import SDKAgentBridge
from .utils import ensure_dir, make_run_id, utc_now, write_json
from .writers import PackagingAgent


class Orchestrator:
    def __init__(self, connectors: dict[str, BaseConnector] | None = None) -> None:
        self.connectors = connectors or default_connectors()

    def run(self, config: RunConfig) -> dict[str, Any]:
        out_dir = Path(config.out)
        if out_dir.name in {"", "."} or str(out_dir).endswith("<run_id>"):
            out_dir = out_dir / make_run_id(config.topic)
        ensure_dir(out_dir)
        manifest = {
            "run_id": out_dir.name,
            "created_at": utc_now(),
            "topic": config.topic,
            "config": config.to_dict(),
            "connector_status": {name: connector.__class__.__name__ for name, connector in self.connectors.items()},
            "stages": [],
        }
        sdk_bridge = SDKAgentBridge(config)
        manifest["agents_sdk"] = sdk_bridge.status()
        write_json(out_dir / "run_manifest.json", manifest)

        scope = self._stage(manifest, out_dir, "intake_scope", lambda: IntakeScopeAgent().run(config, out_dir))
        query_plan = self._stage(manifest, out_dir, "query_strategy", lambda: QueryStrategyAgent(sdk_bridge).run(scope, config, out_dir))
        self._stage(manifest, out_dir, "parallel_search", lambda: ParallelSearchAgent(self.connectors).run(query_plan, config, out_dir))
        records, dedup_report = self._stage(manifest, out_dir, "normalize_dedup", lambda: MetadataNormalizeDedupAgent().run(out_dir))
        records = self._stage(manifest, out_dir, "screening", lambda: RelevanceScreeningAgent(sdk_bridge).run(records, scope, out_dir))
        semantic = self.connectors.get("semantic_scholar")
        snowball_connector = semantic if isinstance(semantic, SemanticScholarConnector) else None
        snowball_report = self._stage(manifest, out_dir, "citation_snowball", lambda: CitationSnowballAgent(snowball_connector).run(records, config, out_dir))
        if snowball_report.get("new_raw_records", 0):
            records, dedup_report = self._stage(manifest, out_dir, "normalize_dedup_after_snowball", lambda: MetadataNormalizeDedupAgent().run(out_dir))
            records = self._stage(manifest, out_dir, "screening_after_snowball", lambda: RelevanceScreeningAgent(sdk_bridge).run(records, scope, out_dir))
        evidence = self._stage(manifest, out_dir, "evidence_acquisition", lambda: EvidenceAcquisitionAgent().run(records, config, out_dir))
        summaries = self._stage(manifest, out_dir, "deep_summary", lambda: DeepSummaryAgent(sdk_bridge).run(records, evidence, config, out_dir))
        flags = self._stage(manifest, out_dir, "quality_audit", lambda: QualityAuditAgent(sdk_bridge).run(records, summaries, out_dir, config.topic))
        outputs = self._stage(manifest, out_dir, "packaging", lambda: PackagingAgent().run(records, summaries, flags, config, out_dir))
        manifest["outputs"] = outputs
        manifest["dedup_report"] = dedup_report
        manifest["completed_at"] = utc_now()
        write_json(out_dir / "run_manifest.json", manifest)
        return {"out_dir": str(out_dir), "outputs": outputs, "records": [r.to_dict() for r in records]}

    def _stage(self, manifest: dict[str, Any], out_dir: Path, name: str, func):
        started = utc_now()
        try:
            result = func()
            manifest["stages"].append({"name": name, "started_at": started, "completed_at": utc_now(), "status": "completed"})
            write_json(out_dir / "run_manifest.json", manifest)
            return result
        except Exception as exc:
            manifest["stages"].append({"name": name, "started_at": started, "completed_at": utc_now(), "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            write_json(out_dir / "run_manifest.json", manifest)
            raise
