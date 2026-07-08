from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents import (
    CitationSnowballAgent,
    CoverageAuditAgent,
    DeepSummaryAgent,
    EvidenceAcquisitionAgent,
    FullTextRequestAgent,
    IntakeScopeAgent,
    MetadataNormalizeDedupAgent,
    ParallelSearchAgent,
    QualityAuditAgent,
    QueryStrategyAgent,
    RelevanceScreeningAgent,
    write_manual_db_pack,
)
from .connectors import BaseConnector, SemanticScholarConnector, default_connectors
from .models import Record, RunConfig
from .sdk_bridge import SDKAgentBridge
from .utils import ensure_dir, make_run_id, read_json, read_jsonl, utc_now, write_json
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
        coverage_report = self._stage(manifest, out_dir, "coverage_audit", lambda: CoverageAuditAgent().run(records, query_plan, config, out_dir, allow_supplemental=True))
        if coverage_report.get("supplemental_queries"):
            query_plan = self._stage(manifest, out_dir, "query_strategy_supplemental", lambda: append_supplemental_queries(query_plan, coverage_report, config, out_dir))
            self._stage(manifest, out_dir, "parallel_search_supplemental", lambda: ParallelSearchAgent(self.connectors).run({"queries": coverage_report["supplemental_queries"]}, config, out_dir, append=True))
            records, dedup_report = self._stage(manifest, out_dir, "normalize_dedup_after_supplemental", lambda: MetadataNormalizeDedupAgent().run(out_dir))
            records = self._stage(manifest, out_dir, "screening_after_supplemental", lambda: RelevanceScreeningAgent(sdk_bridge).run(records, scope, out_dir))
            coverage_report = self._stage(manifest, out_dir, "coverage_audit_after_supplemental", lambda: CoverageAuditAgent().run(records, query_plan, config, out_dir, allow_supplemental=False))
        semantic = self.connectors.get("semantic_scholar")
        snowball_connector = semantic if isinstance(semantic, SemanticScholarConnector) else None
        snowball_report = self._stage(manifest, out_dir, "citation_snowball", lambda: CitationSnowballAgent(snowball_connector).run(records, config, out_dir))
        if snowball_report.get("new_raw_records", 0):
            records, dedup_report = self._stage(manifest, out_dir, "normalize_dedup_after_snowball", lambda: MetadataNormalizeDedupAgent().run(out_dir))
            records = self._stage(manifest, out_dir, "screening_after_snowball", lambda: RelevanceScreeningAgent(sdk_bridge).run(records, scope, out_dir))
            coverage_report = self._stage(manifest, out_dir, "coverage_audit_after_snowball", lambda: CoverageAuditAgent().run(records, query_plan, config, out_dir, allow_supplemental=False))
        evidence = self._stage(manifest, out_dir, "evidence_acquisition", lambda: EvidenceAcquisitionAgent().run(records, config, out_dir))
        summaries = self._stage(manifest, out_dir, "deep_summary", lambda: DeepSummaryAgent(sdk_bridge).run(records, evidence, config, out_dir))
        fulltext_requests = self._stage(manifest, out_dir, "fulltext_requests", lambda: FullTextRequestAgent().run(records, evidence, config, out_dir))
        flags = self._stage(manifest, out_dir, "quality_audit", lambda: QualityAuditAgent(sdk_bridge).run(records, summaries, out_dir, config.topic))
        outputs = self._stage(manifest, out_dir, "packaging", lambda: PackagingAgent().run(records, summaries, flags, config, out_dir))
        manifest["outputs"] = outputs
        manifest["dedup_report"] = dedup_report
        manifest["fulltext_requests"] = {"count": len(fulltext_requests)}
        manifest["completed_at"] = utc_now()
        write_json(out_dir / "run_manifest.json", manifest)
        return {"out_dir": str(out_dir), "outputs": outputs, "records": [r.to_dict() for r in records]}

    def resume_fulltext(self, config: RunConfig) -> dict[str, Any]:
        out_dir = Path(config.out)
        if not out_dir.exists():
            raise RuntimeError(f"Output directory does not exist: {out_dir}")
        manifest_path = out_dir / "run_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {
            "run_id": out_dir.name,
            "created_at": utc_now(),
            "topic": config.topic,
            "config": config.to_dict(),
            "connector_status": {},
            "stages": [],
        }
        previous_config = manifest.get("config") or {}
        if not config.topic:
            config.topic = manifest.get("topic") or previous_config.get("topic") or ""
        if config.output_format == "files" and previous_config.get("output_format"):
            config.output_format = previous_config.get("output_format")
        manifest["topic"] = config.topic
        manifest["resume_config"] = config.to_dict()
        sdk_bridge = SDKAgentBridge(config)
        manifest["agents_sdk"] = sdk_bridge.status()
        write_json(manifest_path, manifest)

        rows = read_jsonl(out_dir / "screened_records.jsonl")
        if not rows:
            raise RuntimeError(f"No screened_records.jsonl found in {out_dir}; run `python -m litflow run` first.")
        records = [Record.from_dict(row) for row in rows]
        evidence = self._stage(manifest, out_dir, "resume_evidence_acquisition", lambda: EvidenceAcquisitionAgent().run(records, config, out_dir))
        summaries = self._stage(manifest, out_dir, "resume_deep_summary", lambda: DeepSummaryAgent(sdk_bridge).run(records, evidence, config, out_dir, reuse_existing=True))
        fulltext_requests = self._stage(manifest, out_dir, "resume_fulltext_requests", lambda: FullTextRequestAgent().run(records, evidence, config, out_dir))
        flags = self._stage(manifest, out_dir, "resume_quality_audit", lambda: QualityAuditAgent(sdk_bridge).run(records, summaries, out_dir, config.topic))
        outputs = self._stage(manifest, out_dir, "resume_packaging", lambda: PackagingAgent().run(records, summaries, flags, config, out_dir))
        manifest["outputs"] = outputs
        manifest["fulltext_requests"] = {"count": len(fulltext_requests)}
        manifest["resume_completed_at"] = utc_now()
        write_json(manifest_path, manifest)
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


def append_supplemental_queries(query_plan: dict[str, Any], coverage_report: dict[str, Any], config: RunConfig, out_dir: Path) -> dict[str, Any]:
    existing = {(q.get("source"), q.get("query"), q.get("intent")) for q in query_plan.get("queries", [])}
    appended = []
    for query in coverage_report.get("supplemental_queries", []):
        key = (query.get("source"), query.get("query"), query.get("intent"))
        if key in existing:
            continue
        existing.add(key)
        appended.append(query)
    if appended:
        query_plan["queries"] = [*query_plan.get("queries", []), *appended]
        query_plan["supplemental_added"] = len(appended)
        query_plan["supplemental_reason"] = coverage_report.get("warnings", [])
        write_json(out_dir / "query_plan.json", query_plan)
        write_manual_db_pack(out_dir / "manual_db_search_pack.md", query_plan, config)
    return query_plan
