from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook

from litflow.agents import (
    DeepSummaryAgent,
    EvidenceAcquisitionAgent,
    IntakeScopeAgent,
    MetadataNormalizeDedupAgent,
    QualityAuditAgent,
    QueryStrategyAgent,
    RelevanceScreeningAgent,
)
from litflow.connectors import BaseConnector, SearchResult
from litflow.models import RawRecord, Record, RunConfig, SearchLogEntry
from litflow.orchestrator import Orchestrator
from litflow.sdk_bridge import LiteratureSummaryOutput, QABatchOutput, QAFlagOutput, QueryStrategyOutput, RelevanceBatchOutput, RelevanceDecision
from litflow.utils import read_json, read_jsonl, utc_now, write_json, write_jsonl
from litflow.writers import PackagingAgent


class FakeConnector(BaseConnector):
    def __init__(self, source: str, records: list[RawRecord]):
        self.source = source
        self.records = records

    def search(self, query: str, config: RunConfig, limit: int) -> SearchResult:
        rows = [RawRecord(**{**r.to_dict(), "source": self.source, "source_database": self.source}) for r in self.records[:limit]]
        return SearchResult(rows, SearchLogEntry(self.source, query, utc_now(), f"fake://{self.source}", len(rows)))


class FakeSDKBridge:
    enabled = True

    def refine_query_strategy(self, topic, deterministic_plan):
        return QueryStrategyOutput(
            expanded_terms=["personnel selection"],
            concept_groups={"task_terms": ["employee selection"]},
            base_queries=["LLM employee selection validity"],
            rationale="fake sdk query strategy",
        )

    def screen_relevance(self, topic, records):
        return RelevanceBatchOutput(
            decisions=[
                RelevanceDecision(record_id=record.record_id, relevance_score=3, inclusion_status="included", exclusion_reason="")
                for record in records
            ]
        )

    def summarize_record(self, topic, record, evidence_bundle, fallback_citation):
        return LiteratureSummaryOutput(
            record_id=record.record_id,
            citation=fallback_citation,
            title=record.title,
            evidence_level=record.evidence_level,
            abstract="SDK abstract summary",
            introduction="SDK introduction summary",
            method="SDK method summary limited to evidence",
            results_findings="SDK findings summary limited to evidence",
            conclusion="SDK conclusion summary",
            limitations="SDK limitations",
            topic_relevance="SDK relevance",
            follow_up="SDK follow up",
        )

    def audit_quality(self, topic, records, summaries, existing_flags):
        return QABatchOutput(flags=[QAFlagOutput(record_id=records[0].record_id, title=records[0].title, code="sdk_flag", message="fake sdk flag")])


class LitflowTests(unittest.TestCase):
    def test_korean_topic_expands_to_english_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(topic="LLM을 활용한 면접 평가와 역량 측정 연구", out=tmp)
            scope = IntakeScopeAgent().run(config, Path(tmp))
            expanded = " ".join(scope["expanded_terms"])
            self.assertIn("interview", expanded)
            self.assertIn("assessment", expanded)
            self.assertIn("competency", expanded)

    def test_dedup_merges_same_doi(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            raw_dir = out / "raw_results"
            raw_dir.mkdir()
            write_jsonl(
                raw_dir / "a.jsonl",
                [
                    RawRecord(source="openalex", source_id="oa1", title="A Study of AI Interview Assessment", doi="10.123/ABC", abstract="First abstract", source_database="openalex").to_dict(),
                    RawRecord(source="crossref", source_id="cr1", title="A Study of AI Interview Assessment", doi="https://doi.org/10.123/abc", abstract="", source_database="crossref").to_dict(),
                ],
            )
            records, report = MetadataNormalizeDedupAgent().run(out)
            self.assertEqual(len(records), 1)
            self.assertEqual(report["duplicate_count"], 1)
            self.assertEqual(set(records[0].source_database), {"openalex", "crossref"})

    def test_evidence_summary_marks_abstract_only_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            record = Record(
                record_id="r1",
                title="AI Literacy and Job Performance",
                abstract="This study examines AI literacy and job performance using survey data.",
                relevance_score=3,
                inclusion_status="included",
            )
            config = RunConfig(topic="AI literacy job performance", out=tmp)
            evidence = EvidenceAcquisitionAgent().run([record], config, out)
            summaries = DeepSummaryAgent().run([record], evidence, config, out)
            self.assertEqual(record.evidence_level, "abstract_only")
            self.assertIn("초록", summaries[0]["method"])
            self.assertIn("추정하지 않습니다", summaries[0]["results_findings"])

    def test_packaging_creates_required_xlsx_sheets_and_docx(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_json(out / "query_plan.json", {"queries": [{"source": "openalex", "query": "AI assessment", "intent": "test"}]})
            write_jsonl(out / "search_log.jsonl", [{"source": "fake", "query": "AI", "result_count": 1}])
            record = Record(
                record_id="r1",
                title="AI Assessment Validity",
                authors=["Kim"],
                year=2026,
                venue="Journal",
                abstract="Abstract",
                relevance_score=3,
                inclusion_status="included",
                evidence_level="abstract_only",
                source_database=["fake"],
            )
            summary = {
                "record_id": "r1",
                "citation": "Kim (2026). AI Assessment Validity.",
                "title": "AI Assessment Validity",
                "evidence_level": "abstract_only",
                "abstract": "Abstract",
                "introduction": "Intro",
                "method": "Method limited to abstract.",
                "results_findings": "Findings limited to abstract.",
                "conclusion": "Conclusion",
                "limitations": "Limitations",
                "topic_relevance": "Relevant",
                "follow_up": "Review full text.",
            }
            outputs = PackagingAgent().run([record], [summary], [], RunConfig(topic="AI assessment", out=tmp), out)
            wb = load_workbook(outputs["xlsx"], read_only=True)
            try:
                self.assertEqual(
                    set(["Run_Config", "Search_Log", "Queries", "All_Candidates", "Included", "Excluded", "Evidence_Level", "Deep_Summaries", "QA_Flags"]),
                    set(wb.sheetnames),
                )
            finally:
                wb.close()
            with ZipFile(outputs["docx"]) as zf:
                xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
                self.assertIn("w:tblGrid", xml)
                self.assertNotIn("https://", xml)

    def test_mock_integration_run_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                RawRecord(
                    source="fake",
                    source_id="1",
                    title="Large Language Models for Interview Assessment",
                    authors=["Lee"],
                    year=2025,
                    venue="Conference",
                    doi="10.1000/test",
                    abstract="Large language models are used for interview assessment validity and reliability.",
                    url="https://example.test/paper",
                    citation_count=10,
                    source_database="fake",
                )
            ]
            connectors = {name: FakeConnector(name, records) for name in ["openalex", "crossref", "semantic_scholar", "arxiv", "web"]}
            out = Path(tmp) / "run"
            result = Orchestrator(connectors).run(RunConfig(topic="LLM interview assessment validity", out=str(out), max_raw=25, max_deep=5, agent_mode="off"))
            self.assertTrue(Path(result["outputs"]["xlsx"]).exists())
            self.assertTrue(Path(result["outputs"]["docx"]).exists())
            manifest = read_json(out / "run_manifest.json")
            self.assertEqual(manifest["stages"][-1]["name"], "packaging")
            included = [r for r in read_jsonl(out / "screened_records.jsonl") if r["inclusion_status"] == "included"]
            self.assertEqual(len(included), 1)

    def test_sdk_assisted_agents_use_bridge_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            bridge = FakeSDKBridge()
            config = RunConfig(topic="LLM interview assessment", out=tmp, agent_mode="sdk")
            scope = IntakeScopeAgent().run(config, out)
            query_plan = QueryStrategyAgent(bridge).run(scope, config, out)
            self.assertTrue(query_plan["sdk_assisted"])
            self.assertIn("LLM employee selection validity", [q["query"] for q in query_plan["queries"]])
            record = Record(
                record_id="r1",
                title="Large Language Models for Interview Assessment",
                abstract="LLM interview assessment validity.",
            )
            records = RelevanceScreeningAgent(bridge).run([record], scope, out)
            self.assertEqual(records[0].relevance_score, 3)
            evidence = EvidenceAcquisitionAgent().run(records, config, out)
            summaries = DeepSummaryAgent(bridge).run(records, evidence, config, out)
            self.assertEqual(summaries[0]["abstract"], "SDK abstract summary")
            flags = QualityAuditAgent(bridge).run(records, summaries, out, config.topic)
            self.assertEqual(flags[-1]["code"], "sdk_flag")


if __name__ == "__main__":
    unittest.main()
