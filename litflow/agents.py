from __future__ import annotations

import math
import re
import urllib.error
import urllib.parse
import urllib.request
import csv
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .connectors import BaseConnector, SemanticScholarConnector
from .models import RawRecord, Record, RunConfig, SearchQuery
from .utils import (
    clean_text,
    compact_list,
    contains_korean,
    ensure_dir,
    append_jsonl,
    make_record_id,
    normalize_doi,
    normalize_title,
    read_json,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)
from .sdk_bridge import SDKAgentBridge


MAX_PDF_BYTES = 30 * 1024 * 1024
MAX_FULLTEXT_CHARS = 50000
MAX_SECTION_CHARS = 12000
FULLTEXT_EVIDENCE_LEVELS = {"fulltext_pdf", "user_provided_fulltext"}
REQUEST_HEADERS = {
    "User-Agent": "litflow/0.1 (mailto:research@example.com)",
    "Accept": "application/pdf,text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}
CROSSREF_PDF_CACHE: dict[str, list[str]] = {}


KOREAN_TERM_MAP = {
    "문헌": ["literature", "review"],
    "논문": ["paper", "study"],
    "면접": ["interview", "asynchronous video interview", "structured interview"],
    "채용": ["hiring", "recruitment", "employee selection", "personnel selection"],
    "선발": ["employee selection", "personnel selection", "selection assessment"],
    "평가": ["assessment", "evaluation", "scoring", "measurement"],
    "역량": ["competency", "competence", "capability"],
    "스킬": ["skill", "skills", "skill extraction"],
    "직무": ["job", "occupation", "job description", "work role"],
    "태그": ["tagging", "classification", "taxonomy"],
    "분류": ["classification", "taxonomy", "labeling"],
    "생성형": ["generative AI", "GenAI", "GAI"],
    "인공지능": ["artificial intelligence", "AI"],
    "사람": ["human", "human judgment"],
    "협업": ["collaboration", "human-AI collaboration", "AI-assisted decision making"],
    "의사결정": ["decision making", "decision support"],
    "리터러시": ["literacy", "AI literacy"],
    "성과": ["performance", "job performance", "productivity"],
    "타당도": ["validity", "validation"],
    "신뢰도": ["reliability"],
    "공정성": ["fairness", "bias"],
}

DEFAULT_DOMAIN_TERMS = [
    "large language model",
    "LLM",
    "generative AI",
    "artificial intelligence",
    "machine learning",
]

KOREAN_FALLBACK_TERM_MAP = {
    "문헌": ["literature", "review"],
    "논문": ["paper", "study"],
    "면접": ["interview", "asynchronous video interview", "structured interview"],
    "채용": ["hiring", "recruitment", "employee selection", "personnel selection"],
    "평가": ["assessment", "evaluation", "scoring", "measurement"],
    "역량": ["competency", "competence", "capability", "skill"],
    "스킬": ["skill", "skills", "skill extraction"],
    "직무": ["job", "occupation", "work role", "workplace"],
    "업무": ["work", "workplace", "work performance"],
    "성과": ["performance", "job performance", "productivity", "task performance"],
    "직무성과": ["job performance", "work performance", "employee performance", "productivity"],
    "생산성": ["productivity", "worker productivity", "task performance"],
    "태그": ["tagging", "classification", "taxonomy"],
    "분류": ["classification", "taxonomy", "labeling"],
    "생성형": ["generative AI", "GenAI", "GAI"],
    "인공지능": ["artificial intelligence", "AI"],
    "사람": ["human", "human judgment"],
    "협업": ["collaboration", "human-AI collaboration", "AI-assisted decision making"],
    "의사결정": ["decision making", "decision support"],
    "리터러시": ["literacy", "AI literacy"],
    "활용": ["use", "adoption", "usage", "capability"],
    "타당": ["validity", "validation"],
    "신뢰": ["reliability"],
    "공정": ["fairness", "bias"],
    "척도": ["scale development", "measurement scale", "validation"],
    "실증": ["empirical study", "survey", "experiment"],
    "검증": ["validation", "empirical validation"],
}

RECALL_QUERY_LIMITS = {"fast": 5, "balanced": 9, "high": 16}


class IntakeScopeAgent:
    name = "IntakeScopeAgent"

    def run(self, config: RunConfig, out_dir: Path) -> dict[str, Any]:
        topic_terms = extract_terms(config.topic)
        expanded = expand_terms(config.topic)
        scope = {
            "agent": self.name,
            "created_at": utc_now(),
            "topic": config.topic,
            "language": config.language,
            "year_from": config.year_from,
            "year_to": config.year_to,
            "contains_korean": contains_korean(config.topic),
            "core_terms": topic_terms,
            "expanded_terms": expanded,
            "include": [
                "peer-reviewed journal articles",
                "conference papers",
                "preprints",
                "book chapters",
                "technical reports",
                "survey/review papers",
                "empirical studies",
                "methodology papers",
            ],
            "exclude": [
                "non-scholarly blog posts unless used only for source discovery",
                "marketing pages without bibliographic metadata",
                "duplicate records after DOI/title matching",
            ],
            "default_evidence_policy": "Summaries must state evidence level and avoid unsupported Method/Results claims.",
        }
        write_json(out_dir / "scope.json", scope)
        return scope


class QueryStrategyAgent:
    name = "QueryStrategyAgent"

    def __init__(self, sdk_bridge: SDKAgentBridge | None = None):
        self.sdk_bridge = sdk_bridge

    def run(self, scope: dict[str, Any], config: RunConfig, out_dir: Path) -> dict[str, Any]:
        topic = scope["topic"]
        expanded = scope["expanded_terms"]
        concept_groups = make_concept_groups(topic, expanded)
        query_specs = make_query_specs(topic, concept_groups, config.recall_mode)
        sdk_rationale = ""
        sdk_assisted = False
        if self.sdk_bridge and self.sdk_bridge.enabled:
            deterministic_plan = {
                "expanded_terms": expanded,
                "concept_groups": concept_groups,
                "base_queries": [spec["query"] for spec in query_specs],
                "recall_mode": config.recall_mode,
            }
            try:
                sdk_output = self.sdk_bridge.refine_query_strategy(topic, deterministic_plan)
                if sdk_output.expanded_terms:
                    expanded = compact_list([*expanded, *sdk_output.expanded_terms])
                if sdk_output.concept_groups:
                    concept_groups = merge_concept_groups(concept_groups, sdk_output.concept_groups)
                if sdk_output.base_queries:
                    sdk_specs = [
                        {"query": query, "intent": f"sdk_refined_{idx + 1}", "facets": ["sdk_refined"]}
                        for idx, query in enumerate(sdk_output.base_queries)
                    ]
                    query_specs = [*sdk_specs, *make_query_specs(topic, concept_groups, config.recall_mode)]
                else:
                    query_specs = make_query_specs(topic, concept_groups, config.recall_mode)
                sdk_rationale = sdk_output.rationale
                sdk_assisted = True
                log_sdk_event(out_dir, self.name, "completed", "SDK refined query strategy.")
            except Exception as exc:
                log_sdk_event(out_dir, self.name, "fallback", f"{type(exc).__name__}: {exc}")
        source_queries: list[SearchQuery] = []
        public_sources = ["openalex", "crossref", "semantic_scholar", "arxiv"]
        round1_specs = query_specs[: RECALL_QUERY_LIMITS.get(config.recall_mode, 16)]
        for source in public_sources:
            for spec in round1_specs:
                source_queries.append(
                    SearchQuery(
                        source=source,
                        query=str(spec["query"]),
                        intent=str(spec["intent"]),
                        search_round=1,
                        facets=list(spec.get("facets") or []),
                    )
                )
        if config.web_search_provider == "serpapi":
            scholar_specs = query_specs[: max(3, min(len(query_specs), 10 if config.recall_mode == "high" else 5))]
            for spec in scholar_specs:
                source_queries.append(
                    SearchQuery(
                        source="serpapi_google_scholar",
                        query=str(spec["query"]),
                        intent="scholar_" + str(spec["intent"]),
                        search_round=2,
                        facets=list(spec.get("facets") or []),
                    )
                )
        elif config.web_search_provider == "none":
            for spec in round1_specs[:3]:
                source_queries.append(
                    SearchQuery(
                        source="web",
                        query=str(spec["query"]),
                        intent="manual_web_" + str(spec["intent"]),
                        search_round=2,
                        facets=list(spec.get("facets") or []),
                    )
                )
        plan = {
            "agent": self.name,
            "created_at": utc_now(),
            "topic": topic,
            "recall_mode": config.recall_mode,
            "web_search_provider": config.web_search_provider,
            "sdk_assisted": sdk_assisted,
            "sdk_rationale": sdk_rationale,
            "expanded_terms": expanded,
            "concept_groups": concept_groups,
            "query_specs": query_specs,
            "queries": [q.to_dict() for q in source_queries],
            "manual_databases": ["Google Scholar", "Scopus", "Web of Science", "ACM Digital Library", "IEEE Xplore", "Dimensions"],
        }
        write_json(out_dir / "query_plan.json", plan)
        write_manual_db_pack(out_dir / "manual_db_search_pack.md", plan, config)
        return plan


class ParallelSearchAgent:
    name = "ParallelSearchAgent"

    def __init__(self, connectors: dict[str, BaseConnector]):
        self.connectors = connectors

    def run(self, query_plan: dict[str, Any], config: RunConfig, out_dir: Path, append: bool = False) -> list[dict[str, Any]]:
        raw_dir = ensure_dir(out_dir / "raw_results")
        search_logs: list[dict[str, Any]] = read_jsonl(out_dir / "search_log.jsonl") if append else []
        source_records: dict[str, list[dict[str, Any]]] = {}
        if append:
            for path in sorted(raw_dir.glob("*.jsonl")):
                source_records[path.stem] = read_jsonl(path)
        queries = [SearchQuery(**q) for q in query_plan["queries"]]
        per_source_budget = max(1, min(config.per_source_cap, math.ceil(config.max_raw / max(1, len(self.connectors)))))
        source_counts: Counter[str] = Counter({source: len(rows) for source, rows in source_records.items()})
        for query in queries:
            if source_counts[query.source] >= per_source_budget:
                continue
            connector = self.connectors.get(query.source)
            if connector is None:
                continue
            remaining = per_source_budget - source_counts[query.source]
            result = connector.search(query.query, config, min(remaining, 50))
            for record in result.records:
                record.found_by = f"{query.source}:{query.intent}"
                record.search_round = query.search_round
                record.facet_matches = list(query.facets)
                record.extra = {
                    **record.extra,
                    "query": query.query,
                    "query_intent": query.intent,
                    "query_facets": list(query.facets),
                    "search_round": query.search_round,
                }
            source_counts[query.source] += len(result.records)
            source_records.setdefault(query.source, []).extend(record.to_dict() for record in result.records)
            search_logs.append(result.log.to_dict())
        for source, rows in source_records.items():
            write_jsonl(raw_dir / f"{source}.jsonl", rows)
        write_jsonl(out_dir / "search_log.jsonl", search_logs)
        return search_logs


class MetadataNormalizeDedupAgent:
    name = "MetadataNormalizeDedupAgent"

    def run(self, out_dir: Path) -> tuple[list[Record], dict[str, Any]]:
        raw_dir = out_dir / "raw_results"
        raw_records: list[RawRecord] = []
        for path in sorted(raw_dir.glob("*.jsonl")):
            for row in read_jsonl(path):
                raw_records.append(RawRecord(**row))

        records: list[Record] = []
        doi_index: dict[str, Record] = {}
        title_index: dict[str, Record] = {}
        duplicate_count = 0
        for raw in raw_records:
            if not clean_text(raw.title):
                continue
            doi = normalize_doi(raw.doi)
            title_key = normalize_title(raw.title)
            target = None
            if doi and doi in doi_index:
                target = doi_index[doi]
            elif title_key and title_key in title_index:
                target = title_index[title_key]
            elif title_key:
                target = fuzzy_find(title_key, title_index)
            if target is None:
                target = Record(
                    record_id=make_record_id(doi, raw.title),
                    title=clean_text(raw.title),
                    authors=raw.authors,
                    year=raw.year,
                    venue=clean_text(raw.venue),
                    doi=doi,
                    source_ids={raw.source: raw.source_id},
                    abstract=clean_text(raw.abstract),
                    url=raw.url,
                    citation_count=raw.citation_count,
                    source_database=[raw.source_database or raw.source],
                    open_access_pdf=raw.open_access_pdf,
                    found_by=compact_list([raw.found_by or raw.source]),
                    search_round=raw.search_round or 1,
                    facet_matches=compact_list(raw.facet_matches),
                    extra={"raw_sources": [raw.extra]},
                )
                records.append(target)
                if doi:
                    doi_index[doi] = target
                if title_key:
                    title_index[title_key] = target
            else:
                duplicate_count += 1
                merge_record(target, raw)

        rows = [record.to_dict() for record in records]
        write_jsonl(out_dir / "normalized_records.jsonl", rows)
        report = {
            "agent": self.name,
            "created_at": utc_now(),
            "raw_count": len(raw_records),
            "normalized_count": len(records),
            "duplicate_count": duplicate_count,
        }
        write_json(out_dir / "dedup_report.json", report)
        return records, report


class RelevanceScreeningAgent:
    name = "RelevanceScreeningAgent"

    def __init__(self, sdk_bridge: SDKAgentBridge | None = None):
        self.sdk_bridge = sdk_bridge

    def run(self, records: list[Record], scope: dict[str, Any], out_dir: Path) -> list[Record]:
        terms = screening_terms(scope)
        for record in records:
            score, reason = score_record(record, terms)
            record.relevance_score = score
            if score >= 2:
                record.inclusion_status = "included"
                record.exclusion_reason = ""
            else:
                record.inclusion_status = "excluded"
                record.exclusion_reason = reason
            record.summary_status = "pending" if record.inclusion_status == "included" else "not_applicable"
        if self.sdk_bridge and self.sdk_bridge.enabled and records:
            decisions: dict[str, Any] = {}
            for chunk in chunks(records, 20):
                try:
                    sdk_output = self.sdk_bridge.screen_relevance(scope["topic"], chunk)
                    for decision in sdk_output.decisions:
                        decisions[decision.record_id] = decision
                except Exception as exc:
                    log_sdk_event(out_dir, self.name, "fallback", f"{type(exc).__name__}: {exc}")
                    decisions = {}
                    break
            if decisions:
                for record in records:
                    decision = decisions.get(record.record_id)
                    if not decision:
                        continue
                    record.relevance_score = int(decision.relevance_score)
                    record.inclusion_status = "included" if decision.relevance_score >= 2 else "excluded"
                    record.exclusion_reason = "" if record.inclusion_status == "included" else decision.exclusion_reason
                    record.summary_status = "pending" if record.inclusion_status == "included" else "not_applicable"
                log_sdk_event(out_dir, self.name, "completed", f"SDK screened {len(decisions)} records.")
        write_jsonl(out_dir / "screened_records.jsonl", [record.to_dict() for record in records])
        return records


class CoverageAuditAgent:
    name = "CoverageAuditAgent"

    def run(
        self,
        records: list[Record],
        query_plan: dict[str, Any],
        config: RunConfig,
        out_dir: Path,
        allow_supplemental: bool = False,
    ) -> dict[str, Any]:
        concept_groups = query_plan.get("concept_groups", {})
        facet_counts = {}
        included = [record for record in records if record.inclusion_status == "included"]
        for record in records:
            record.facet_matches = match_record_facets(record, concept_groups, existing=record.facet_matches)
            record.coverage_warning = coverage_warning_for_record(record, concept_groups)
        for facet in ["ai_terms", "core_concepts", "outcome_terms", "context_terms", "method_terms", "quality_terms"]:
            facet_counts[facet] = {
                "all_records": sum(1 for record in records if facet in record.facet_matches),
                "included_records": sum(1 for record in included if facet in record.facet_matches),
            }
        combo_counts = {
            "core_outcome": count_facet_combo(included, ["core_concepts", "outcome_terms"]),
            "core_outcome_context": count_facet_combo(included, ["core_concepts", "outcome_terms", "context_terms"]),
            "core_method": count_facet_combo(included, ["core_concepts", "method_terms"]),
        }
        warnings = coverage_warnings(facet_counts, combo_counts, concept_groups, config)
        supplemental = []
        if allow_supplemental and config.recall_mode == "high" and warnings:
            supplemental = supplemental_queries(query_plan, concept_groups, config)
        report = {
            "agent": self.name,
            "created_at": utc_now(),
            "recall_mode": config.recall_mode,
            "web_search_provider": config.web_search_provider,
            "total_records": len(records),
            "included_records": len(included),
            "facet_counts": facet_counts,
            "combo_counts": combo_counts,
            "warnings": warnings,
            "supplemental_queries": supplemental,
        }
        write_json(out_dir / "coverage_audit.json", report)
        write_jsonl(out_dir / "screened_records.jsonl", [record.to_dict() for record in records])
        return report


class CitationSnowballAgent:
    name = "CitationSnowballAgent"

    def __init__(self, connector: SemanticScholarConnector | None):
        self.connector = connector

    def run(self, records: list[Record], config: RunConfig, out_dir: Path) -> dict[str, Any]:
        report = {"agent": self.name, "created_at": utc_now(), "rounds": [], "new_raw_records": 0}
        if self.connector is None:
            report["status"] = "skipped_no_semantic_scholar_connector"
            write_json(out_dir / "snowball_report.json", report)
            return report
        raw_dir = ensure_dir(out_dir / "raw_results")
        seeds = sorted(
            [r for r in records if r.relevance_score == 3 and r.source_ids.get("semantic_scholar")],
            key=lambda r: r.citation_count or 0,
            reverse=True,
        )[:15]
        all_new: list[dict[str, Any]] = []
        next_seed_ids: list[str] = []
        for seed in seeds:
            paper_id = seed.source_ids["semantic_scholar"]
            for relation in ["references", "citations"]:
                result = self.connector.expand(paper_id, relation, limit=10)
                for record in result.records:
                    record.found_by = f"citation_snowball:{relation}"
                    record.search_round = 3
                    record.facet_matches = ["citation_snowball"]
                    if record.source_id:
                        next_seed_ids.append(record.source_id)
                all_new.extend(record.to_dict() for record in result.records)
                report["rounds"].append(result.log.to_dict())
        if all_new:
            write_jsonl(raw_dir / "snowball_semantic_round1.jsonl", all_new)
        if all_new and len(all_new) >= max(1, int(len(records) * 0.1)):
            round2_rows: list[dict[str, Any]] = []
            for paper_id in compact_list(next_seed_ids)[:8]:
                for relation in ["references", "citations"]:
                    result = self.connector.expand(paper_id, relation, limit=5)
                    for record in result.records:
                        record.found_by = f"citation_snowball_round2:{relation}"
                        record.search_round = 3
                        record.facet_matches = ["citation_snowball"]
                    round2_rows.extend(record.to_dict() for record in result.records)
                    report["rounds"].append(result.log.to_dict())
            if round2_rows:
                write_jsonl(raw_dir / "snowball_semantic_round2.jsonl", round2_rows)
                all_new.extend(round2_rows)
        report["new_raw_records"] = len(all_new)
        report["status"] = "completed"
        write_json(out_dir / "snowball_report.json", report)
        return report


class EvidenceAcquisitionAgent:
    name = "EvidenceAcquisitionAgent"

    def run(self, records: list[Record], config: RunConfig, out_dir: Path) -> dict[str, Any]:
        evidence_dir = ensure_dir(out_dir / "evidence_bundles")
        pdf_dir = ensure_dir(out_dir / "fulltext_pdfs")
        user_dir = user_fulltext_dir(config, out_dir)
        ensure_dir(user_dir)
        user_files = list(user_dir.glob("*")) if user_dir.exists() else []
        included = [r for r in records if r.inclusion_status == "included"]
        included.sort(key=lambda r: (r.relevance_score, r.citation_count or 0), reverse=True)
        fulltext_targets = {r.record_id for r in included[: config.max_deep]}
        bundle_index: dict[str, dict[str, Any]] = {}
        for record in records:
            level = "metadata_only"
            sources = []
            fulltext: dict[str, Any] = {}
            if record.abstract:
                level = "abstract_only"
                sources.append("abstract")
            matched_user_files = match_user_files(record, user_files)
            if matched_user_files:
                level = "user_provided_fulltext"
                sources.append("user_provided_file")
                fulltext = extract_user_fulltext(record, matched_user_files)
            elif record.record_id in fulltext_targets:
                fulltext = acquire_pdf_fulltext(record, pdf_dir)
                if fulltext.get("status") == "completed":
                    level = "fulltext_pdf"
                    sources.append("pdf_fulltext")
                    record.open_access_pdf = fulltext.get("pdf_url", record.open_access_pdf)
                elif record.open_access_pdf:
                    sources.append("open_access_pdf_unextracted")
            record.evidence_level = level
            bundle = {
                "record_id": record.record_id,
                "title": record.title,
                "evidence_level": level,
                "sources": sources,
                "abstract_available": bool(record.abstract),
                "open_access_pdf": record.open_access_pdf,
                "user_files": [str(path) for path in matched_user_files],
                "fulltext_status": fulltext.get("status", "not_attempted"),
                "fulltext_error": fulltext.get("error", ""),
                "pdf_url": fulltext.get("pdf_url", record.open_access_pdf),
                "pdf_path": fulltext.get("pdf_path", ""),
                "fulltext_chars": fulltext.get("char_count", 0),
                "fulltext_excerpt": fulltext.get("excerpt", ""),
                "fulltext_sections": fulltext.get("sections", {}),
                "user_fulltext_dir": str(user_dir),
                "summary_constraints": summary_constraints(level),
            }
            bundle_index[record.record_id] = bundle
            write_json(evidence_dir / f"{record.record_id}.json", bundle)
        write_jsonl(out_dir / "screened_records.jsonl", [record.to_dict() for record in records])
        return bundle_index


class DeepSummaryAgent:
    name = "DeepSummaryAgent"

    def __init__(self, sdk_bridge: SDKAgentBridge | None = None):
        self.sdk_bridge = sdk_bridge

    def run(
        self,
        records: list[Record],
        evidence: dict[str, dict[str, Any]],
        config: RunConfig,
        out_dir: Path,
        reuse_existing: bool = False,
    ) -> list[dict[str, Any]]:
        included = [r for r in records if r.inclusion_status == "included"]
        fulltext_included = [r for r in included if is_fulltext_evidence(evidence.get(r.record_id, {}).get("evidence_level", r.evidence_level))]
        fulltext_included.sort(key=lambda r: (r.relevance_score, r.citation_count or 0), reverse=True)
        selected = fulltext_included[: config.max_deep]
        existing_by_id: dict[str, dict[str, Any]] = {}
        if reuse_existing:
            for summary in read_jsonl(out_dir / "summaries.jsonl"):
                record_id = summary.get("record_id", "")
                if is_fulltext_evidence(summary.get("evidence_level", "")):
                    existing_by_id[record_id] = summary
        summaries_by_id: dict[str, dict[str, Any]] = {}
        for record in selected:
            bundle = evidence.get(record.record_id, {})
            existing = existing_by_id.get(record.record_id)
            if existing and is_fulltext_evidence(bundle.get("evidence_level", record.evidence_level)):
                summaries_by_id[record.record_id] = existing
                record.summary_status = "completed"
                continue
            if self.sdk_bridge and self.sdk_bridge.enabled:
                try:
                    summary_model = self.sdk_bridge.summarize_record(config.topic, record, bundle, format_citation(record))
                    summary = summary_model.model_dump()
                    log_sdk_event(out_dir, self.name, "completed", f"SDK summarized {record.record_id}.")
                except Exception as exc:
                    log_sdk_event(out_dir, self.name, "fallback", f"{record.record_id}: {type(exc).__name__}: {exc}")
                    summary = build_summary(record, bundle)
            else:
                summary = build_summary(record, bundle)
            summaries_by_id[record.record_id] = summary
            record.summary_status = "completed"
        for record in records:
            if record.inclusion_status != "included":
                continue
            if record.summary_status == "completed":
                continue
            if not is_fulltext_evidence(evidence.get(record.record_id, {}).get("evidence_level", record.evidence_level)):
                record.summary_status = "needs_user_fulltext"
            else:
                record.summary_status = "deferred_max_deep"
        summaries = [summaries_by_id[record.record_id] for record in selected if record.record_id in summaries_by_id]
        write_jsonl(out_dir / "summaries.jsonl", summaries)
        write_jsonl(out_dir / "screened_records.jsonl", [record.to_dict() for record in records])
        return summaries


class FullTextRequestAgent:
    name = "FullTextRequestAgent"

    def run(self, records: list[Record], evidence: dict[str, dict[str, Any]], config: RunConfig, out_dir: Path) -> list[dict[str, Any]]:
        rows = fulltext_request_rows(records, evidence, config, out_dir)
        write_jsonl(out_dir / "fulltext_requests.jsonl", rows)
        write_fulltext_requests_csv(out_dir / "fulltext_requests.csv", rows)
        write_fulltext_requests_md(out_dir / "fulltext_requests.md", rows, config, out_dir)
        return rows


class QualityAuditAgent:
    name = "QualityAuditAgent"

    def __init__(self, sdk_bridge: SDKAgentBridge | None = None):
        self.sdk_bridge = sdk_bridge

    def run(self, records: list[Record], summaries: list[dict[str, Any]], out_dir: Path, topic: str = "") -> list[dict[str, Any]]:
        flags: list[dict[str, Any]] = []
        seen_doi: dict[str, str] = {}
        for record in records:
            if record.doi:
                if record.doi in seen_doi:
                    flags.append(flag(record, "duplicate_doi", f"Duplicate DOI also seen in {seen_doi[record.doi]}."))
                seen_doi[record.doi] = record.record_id
            if record.inclusion_status == "included" and record.evidence_level == "metadata_only":
                flags.append(flag(record, "weak_evidence", "Included record has metadata only."))
            if record.inclusion_status == "included" and not record.title:
                flags.append(flag(record, "missing_title", "Included record is missing title."))
        summary_by_id = {s["record_id"]: s for s in summaries}
        for record in records:
            if record.summary_status == "completed" and record.record_id not in summary_by_id:
                flags.append(flag(record, "missing_summary", "Summary status completed but no summary row exists."))
        if self.sdk_bridge and self.sdk_bridge.enabled:
            try:
                sdk_output = self.sdk_bridge.audit_quality(topic, records, summaries, flags)
                flags.extend(item.model_dump() for item in sdk_output.flags)
                log_sdk_event(out_dir, self.name, "completed", f"SDK generated {len(sdk_output.flags)} QA flags.")
            except Exception as exc:
                log_sdk_event(out_dir, self.name, "fallback", f"{type(exc).__name__}: {exc}")
        write_jsonl(out_dir / "qa_flags.jsonl", flags)
        coverage_report = read_json(out_dir / "coverage_audit.json") if (out_dir / "coverage_audit.json").exists() else {}
        write_coverage_audit(out_dir / "coverage_audit.md", records, flags, coverage_report, out_dir)
        return flags


def extract_terms(topic: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+|[가-힣]{2,}", topic)
    stop = {"and", "or", "the", "with", "for", "using", "study", "research", "관련", "주제", "문헌"}
    return compact_list(w for w in words if w.lower() not in stop)


def expand_terms(topic: str) -> list[str]:
    terms = extract_terms(topic)
    expanded = list(terms)
    for ko, en_terms in KOREAN_TERM_MAP.items():
        if ko in topic:
            expanded.extend(en_terms)
    for ko, en_terms in KOREAN_FALLBACK_TERM_MAP.items():
        if ko in topic:
            expanded.extend(en_terms)
    if re.search(r"\bLLM\b|large language model|생성형\s*AI|생성형|AI|인공지능|artificial intelligence|GenAI|GAI", topic, re.I):
        expanded.extend(DEFAULT_DOMAIN_TERMS)
    return [str(x) for x in compact_list(expanded)]


def make_concept_groups(topic: str, expanded: list[str]) -> dict[str, list[str]]:
    text = " ".join([topic, *expanded])
    ai_terms = [t for t in expanded if re.search(r"AI|LLM|language model|machine learning|artificial intelligence|GenAI|GAI", t, re.I)]
    core_terms = [t for t in expanded if re.search(r"literacy|competenc|capabil|skill|역량|리터러시", t, re.I)]
    if re.search(r"literacy|리터러시|활용\s*역량", text, re.I):
        core_terms.extend(["AI literacy", "generative AI literacy", "generative artificial intelligence literacy", "AI competency", "AI capability"])
    if re.search(r"interview|면접", text, re.I):
        core_terms.extend(["AI interview assessment", "automated interview evaluation", "asynchronous video interview"])
    if re.search(r"skill|스킬|직무|job", text, re.I):
        core_terms.extend(["skill extraction", "skill classification", "job skill taxonomy"])
    if not ai_terms:
        ai_terms = DEFAULT_DOMAIN_TERMS[:3]
    if not core_terms:
        core_terms = compact_list([*ai_terms, *expanded[:6]])
    outcome_terms = [t for t in expanded if re.search(r"performance|productivity|outcome|성과|생산성", t, re.I)]
    if re.search(r"performance|productivity|성과|생산성|직무성과", text, re.I):
        outcome_terms.extend(["job performance", "work performance", "employee performance", "worker productivity", "task performance", "productivity effects"])
    context_terms = [t for t in expanded if re.search(r"workplace|employee|worker|organization|직무|업무|조직|직원|근로", t, re.I)]
    if re.search(r"job|work|employee|직무|업무|조직|직원|근로", text, re.I):
        context_terms.extend(["employee", "worker", "workplace", "organization", "occupational", "professional"])
    method_terms = [t for t in expanded if re.search(r"scale|valid|reliab|empirical|survey|experiment|review|척도|타당|신뢰|실증|검증", t, re.I)]
    method_terms.extend(["scale development", "validation", "empirical study", "survey", "experiment", "systematic review"])
    return {
        "topic_terms": expanded[:20],
        "ai_terms": compact_list(ai_terms)[:8],
        "core_concepts": compact_list(core_terms)[:12],
        "outcome_terms": compact_list(outcome_terms)[:10],
        "context_terms": compact_list(context_terms)[:10],
        "method_terms": compact_list(method_terms)[:10],
        "task_terms": compact_list(core_terms)[:12],
        "quality_terms": ["validity", "reliability", "fairness", "bias", "human-AI collaboration", "systematic review"],
    }


def make_queries(topic: str, groups: dict[str, list[str]]) -> list[str]:
    return [str(spec["query"]) for spec in make_query_specs(topic, groups, "balanced")]


def make_query_specs(topic: str, groups: dict[str, list[str]], recall_mode: str = "high") -> list[dict[str, Any]]:
    ai = quote_or(groups.get("ai_terms", [])[:4])
    core = quote_or(groups.get("core_concepts", groups.get("task_terms", []))[:5])
    outcome = quote_or(groups.get("outcome_terms", [])[:5])
    context = quote_or(groups.get("context_terms", [])[:5])
    method = quote_or(groups.get("method_terms", [])[:5])
    quality = quote_or(groups.get("quality_terms", [])[:4])
    plain = " ".join(groups.get("topic_terms", [])[:8])
    specs: list[dict[str, Any]] = [{"query": topic, "intent": "user_topic", "facets": ["user_topic"]}]
    if ai and core:
        specs.append({"query": f"({ai}) AND ({core})", "intent": "core_concept", "facets": ["ai_terms", "core_concepts"]})
    if core and outcome:
        specs.append({"query": f"({core}) AND ({outcome})", "intent": "core_outcome", "facets": ["core_concepts", "outcome_terms"]})
    if ai and core and outcome:
        specs.append({"query": f"({ai}) AND ({core}) AND ({outcome})", "intent": "ai_core_outcome", "facets": ["ai_terms", "core_concepts", "outcome_terms"]})
    if core and outcome and context:
        specs.append({"query": f"({core}) AND ({outcome}) AND ({context})", "intent": "core_outcome_context", "facets": ["core_concepts", "outcome_terms", "context_terms"]})
    if core and method:
        specs.append({"query": f"({core}) AND ({method})", "intent": "core_method", "facets": ["core_concepts", "method_terms"]})
    if ai and core and method:
        specs.append({"query": f"({ai}) AND ({core}) AND ({method})", "intent": "ai_core_method", "facets": ["ai_terms", "core_concepts", "method_terms"]})
    if ai and core and quality:
        specs.append({"query": f"({ai}) AND ({core}) AND ({quality})", "intent": "quality_validity", "facets": ["ai_terms", "core_concepts", "quality_terms"]})
    if plain:
        specs.append({"query": f"{plain} systematic review OR survey", "intent": "review_survey", "facets": ["method_terms"]})
        specs.append({"query": f"{plain} empirical study validation", "intent": "empirical_validation", "facets": ["method_terms", "quality_terms"]})
    if recall_mode == "high":
        for core_term in groups.get("core_concepts", [])[:4]:
            for outcome_term in groups.get("outcome_terms", [])[:4]:
                specs.append(
                    {
                        "query": f'"{core_term}" "{outcome_term}"',
                        "intent": "exact_core_outcome",
                        "facets": ["core_concepts", "outcome_terms"],
                    }
                )
        for core_term in groups.get("core_concepts", [])[:3]:
            for method_term in groups.get("method_terms", [])[:3]:
                specs.append(
                    {
                        "query": f'"{core_term}" "{method_term}"',
                        "intent": "exact_core_method",
                        "facets": ["core_concepts", "method_terms"],
                    }
                )
    limit = RECALL_QUERY_LIMITS.get(recall_mode, RECALL_QUERY_LIMITS["high"])
    seen = set()
    unique = []
    for spec in specs:
        query = clean_text(spec["query"])
        if not query or query.lower() in seen:
            continue
        seen.add(query.lower())
        unique.append({"query": query, "intent": spec["intent"], "facets": list(spec.get("facets") or [])})
    return unique[:limit]


def quote_or(terms: list[str]) -> str:
    return " OR ".join(f'"{t}"' if " " in t else t for t in terms)


def write_manual_db_pack(path: Path, plan: dict[str, Any], config: RunConfig) -> None:
    concept_groups = plan.get("concept_groups", {})
    lines = [
        "# Manual Database Search Pack",
        "",
        f"Topic: {config.topic}",
        f"Recall mode: {config.recall_mode}",
        f"Web search provider: {config.web_search_provider}",
        "",
        "Use these queries in databases that are not fully automated in v1: Google Scholar, Scopus, Web of Science, ACM Digital Library, IEEE Xplore, and Dimensions.",
        "",
        "## Facet Map",
        "",
    ]
    for facet in ["ai_terms", "core_concepts", "outcome_terms", "context_terms", "method_terms", "quality_terms"]:
        values = concept_groups.get(facet, [])
        if values:
            lines.append(f"- {facet}: {', '.join(values[:12])}")
    lines.extend(["", "## Reproducible Queries", ""])
    for query in compact_list(q["query"] for q in plan["queries"] if q["source"] in {"openalex", "serpapi_google_scholar"}):
        lines.append(f"- `{query}`")
    if plan.get("supplemental_reason"):
        lines.extend(["", "## Coverage-Triggered Supplemental Reason", ""])
        for warning in plan["supplemental_reason"]:
            lines.append(f"- {warning.get('code')}: {warning.get('message')}")
    lines.extend(
        [
            "",
            "## Manual Follow-Up Checklist",
            "",
            "- Run the core_outcome and core_outcome_context queries in Google Scholar, Scopus, and Web of Science.",
            "- If results are education/student-heavy, rerun with workplace, employee, worker, organization, job performance, and productivity terms.",
            "- Export title, authors, year, venue, DOI, abstract, URL, citation count, references, and database source.",
            "",
            "Recommended export fields: title, authors, year, venue, DOI, abstract, URL, citation count, references, database source.",
            "Preferred export formats for future import: RIS, BibTeX, CSV, or XLSX.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def merge_concept_groups(base: dict[str, list[str]], refined: dict[str, list[str]]) -> dict[str, list[str]]:
    merged = {key: list(value) for key, value in base.items()}
    for key, values in refined.items():
        if not isinstance(values, list):
            continue
        merged[key] = compact_list([*merged.get(key, []), *[str(v) for v in values]])[:20]
    return merged


def chunks(values: list[Record], size: int) -> list[list[Record]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def log_sdk_event(out_dir: Path, stage: str, status: str, message: str) -> None:
    append_jsonl(
        out_dir / "sdk_usage_log.jsonl",
        [{"stage": stage, "status": status, "message": message, "timestamp": utc_now()}],
    )


def fuzzy_find(title_key: str, title_index: dict[str, Record]) -> Record | None:
    for existing, record in title_index.items():
        if SequenceMatcher(None, title_key, existing).ratio() >= 0.94:
            return record
    return None


def merge_record(target: Record, raw: RawRecord) -> None:
    if raw.source and raw.source_id:
        target.source_ids[raw.source] = raw.source_id
    if raw.source_database and raw.source_database not in target.source_database:
        target.source_database.append(raw.source_database)
    if not target.abstract and raw.abstract:
        target.abstract = clean_text(raw.abstract)
    if not target.url and raw.url:
        target.url = raw.url
    if not target.open_access_pdf and raw.open_access_pdf:
        target.open_access_pdf = raw.open_access_pdf
    if raw.citation_count is not None:
        target.citation_count = max(target.citation_count or 0, raw.citation_count)
    if raw.found_by:
        target.found_by = compact_list([*target.found_by, raw.found_by])
    elif raw.source:
        target.found_by = compact_list([*target.found_by, raw.source])
    target.search_round = min(target.search_round or raw.search_round or 1, raw.search_round or 1)
    target.facet_matches = compact_list([*target.facet_matches, *raw.facet_matches])
    target.authors = target.authors or raw.authors
    target.venue = target.venue or clean_text(raw.venue)
    target.year = target.year or raw.year
    target.extra.setdefault("raw_sources", []).append(raw.extra)


def screening_terms(scope: dict[str, Any]) -> list[str]:
    terms = []
    for value in scope.get("expanded_terms", []):
        text = clean_text(value).lower()
        if len(text) >= 2:
            terms.append(text)
    return compact_list(terms)


def score_record(record: Record, terms: list[str]) -> tuple[int, str]:
    haystack = f"{record.title} {record.abstract} {record.venue}".lower()
    hits = [term for term in terms if term.lower() in haystack]
    unique_hits = len(set(hits))
    if unique_hits >= 5:
        return 3, ""
    if unique_hits >= 3:
        return 2, ""
    if unique_hits >= 1:
        return 1, "Only weak topical overlap."
    return 0, "No topical overlap with expanded query terms."


def is_fulltext_evidence(level: str) -> bool:
    return str(level or "") in FULLTEXT_EVIDENCE_LEVELS


def user_fulltext_dir(config: RunConfig, out_dir: Path) -> Path:
    return Path(config.user_files) if config.user_files else out_dir / "user_fulltext"


def fulltext_request_rows(records: list[Record], evidence: dict[str, dict[str, Any]], config: RunConfig, out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    target_dir = user_fulltext_dir(config, out_dir)
    for record in records:
        if record.inclusion_status != "included":
            continue
        bundle = evidence.get(record.record_id, {})
        level = bundle.get("evidence_level", record.evidence_level)
        if is_fulltext_evidence(level):
            continue
        hint = access_hint(record, bundle)
        rows.append(
            {
                "record_id": record.record_id,
                "citation": format_citation(record),
                "title": record.title,
                "doi": record.doi,
                "url": record.url,
                "open_access_pdf": bundle.get("open_access_pdf", record.open_access_pdf),
                "evidence_level": level,
                "fulltext_status": bundle.get("fulltext_status", "not_attempted"),
                "fulltext_error": bundle.get("fulltext_error", ""),
                "access_hint": hint,
                "priority": access_priority(hint),
                "suggested_filename": suggested_fulltext_filename(record),
                "target_folder": str(target_dir),
                "search_queries": search_queries_for_fulltext(record),
            }
        )
    rows.sort(key=lambda row: (int(row["priority"]), str(row["citation"]).lower()))
    return rows


def access_hint(record: Record, bundle: dict[str, Any]) -> str:
    status = str(bundle.get("fulltext_status", ""))
    error = str(bundle.get("fulltext_error", "")).lower()
    open_pdf = bundle.get("open_access_pdf") or record.open_access_pdf
    if status == "failed" and open_pdf:
        return "download_failed"
    if status == "no_pdf_url":
        return "no_pdf_discovered"
    if open_pdf and status not in {"completed", "failed"}:
        return "maybe_oa_unextracted"
    if any(token in error for token in ["403", "401", "paywall", "forbidden", "unauthorized", "needaccess"]):
        return "likely_paywalled"
    if record.url and any(host in record.url.lower() for host in ["sciencedirect.com", "springer.com", "tandfonline.com", "wiley.com", "sagepub.com"]):
        return "unknown"
    return "unknown"


def access_priority(hint: str) -> int:
    return {
        "download_failed": 1,
        "maybe_oa_unextracted": 2,
        "no_pdf_discovered": 2,
        "unknown": 3,
        "likely_paywalled": 4,
    }.get(hint, 3)


def suggested_fulltext_filename(record: Record) -> str:
    base = record.doi.replace("/", "_").replace(".", "_") if record.doi else normalize_title(record.title)
    base = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", base).strip("._-")
    return (base[:120] or record.record_id) + ".pdf"


def search_queries_for_fulltext(record: Record) -> list[str]:
    queries = []
    if record.title:
        queries.extend([f'"{record.title}" pdf', f'"{record.title}" filetype:pdf'])
    if record.doi:
        queries.append(f'"{record.doi}" pdf')
    if record.authors and record.year:
        queries.append(f'"{record.title}" {record.authors[0]} {record.year} pdf')
    return compact_list(queries)[:4]


def write_fulltext_requests_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    headers = [
        "priority",
        "access_hint",
        "citation",
        "title",
        "doi",
        "url",
        "open_access_pdf",
        "evidence_level",
        "fulltext_status",
        "fulltext_error",
        "suggested_filename",
        "target_folder",
        "search_queries",
        "record_id",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") if not isinstance(row.get(header), list) else " | ".join(row[header]) for header in headers})


def write_fulltext_requests_md(path: Path, rows: list[dict[str, Any]], config: RunConfig, out_dir: Path) -> None:
    lines = [
        "# Full Text Needed",
        "",
        f"Topic: {config.topic}",
        f"Target folder: `{user_fulltext_dir(config, out_dir)}`",
        "",
        "Only records with extracted full text are summarized into Notion detail pages. Download accessible PDFs into the target folder using the suggested filename when possible, then run `python -m litflow resume-fulltext --out <same_out>`.",
        "",
    ]
    if not rows:
        lines.append("No full-text requests remain.")
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {idx}. {row['citation']}",
                "",
                f"- record_id: `{row['record_id']}`",
                f"- access_hint: `{row['access_hint']}` | priority: {row['priority']}",
                f"- suggested_filename: `{row['suggested_filename']}`",
                f"- DOI: {row['doi'] or 'n/a'}",
                f"- URL: {row['url'] or 'n/a'}",
                f"- open_access_pdf: {row['open_access_pdf'] or 'n/a'}",
                "- search_queries:",
            ]
        )
        for query in row["search_queries"]:
            lines.append(f"  - `{query}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def match_record_facets(record: Record, concept_groups: dict[str, list[str]], existing: list[str] | None = None) -> list[str]:
    haystack = record_haystack(record)
    facets = list(existing or [])
    for facet in ["ai_terms", "core_concepts", "outcome_terms", "context_terms", "method_terms", "quality_terms"]:
        terms = concept_groups.get(facet, [])
        if any(term_match(term, haystack) for term in terms):
            facets.append(facet)
    return compact_list(facets)


def record_haystack(record: Record) -> str:
    return f"{record.title} {record.abstract} {record.venue} {' '.join(record.source_database)}".lower()


def term_match(term: str, haystack: str) -> bool:
    text = clean_text(term).lower()
    if not text:
        return False
    if text in haystack:
        return True
    pieces = [piece for piece in re.split(r"[^a-z0-9]+", text) if len(piece) >= 4]
    return bool(pieces) and all(piece in haystack for piece in pieces[:3])


def coverage_warning_for_record(record: Record, concept_groups: dict[str, list[str]]) -> str:
    if record.inclusion_status != "included":
        return ""
    warnings = []
    if concept_groups.get("outcome_terms") and "outcome_terms" not in record.facet_matches:
        warnings.append("No explicit outcome/performance facet matched in available metadata.")
    if concept_groups.get("context_terms") and "context_terms" not in record.facet_matches:
        warnings.append("No explicit workplace/context facet matched in available metadata.")
    return " ".join(warnings)


def count_facet_combo(records: list[Record], facets: list[str]) -> int:
    return sum(1 for record in records if all(facet in record.facet_matches for facet in facets))


def coverage_warnings(
    facet_counts: dict[str, dict[str, int]],
    combo_counts: dict[str, int],
    concept_groups: dict[str, list[str]],
    config: RunConfig,
) -> list[dict[str, Any]]:
    threshold = 2 if config.recall_mode == "high" else 1
    warnings = []
    if concept_groups.get("outcome_terms") and combo_counts.get("core_outcome", 0) < threshold:
        warnings.append(
            {
                "code": "weak_core_outcome_coverage",
                "message": "Core concept + outcome/performance coverage is thin; add focused performance/productivity queries.",
                "count": combo_counts.get("core_outcome", 0),
                "threshold": threshold,
            }
        )
    if concept_groups.get("context_terms") and combo_counts.get("core_outcome_context", 0) < threshold:
        warnings.append(
            {
                "code": "weak_workplace_context_coverage",
                "message": "Workplace/employee context is under-covered relative to the topic facets.",
                "count": combo_counts.get("core_outcome_context", 0),
                "threshold": threshold,
            }
        )
    if concept_groups.get("method_terms") and facet_counts.get("method_terms", {}).get("included_records", 0) == 0:
        warnings.append(
            {
                "code": "weak_method_coverage",
                "message": "No included records clearly matched method/validation/review facets.",
                "count": 0,
                "threshold": 1,
            }
        )
    return warnings


def supplemental_queries(query_plan: dict[str, Any], concept_groups: dict[str, list[str]], config: RunConfig) -> list[dict[str, Any]]:
    existing = {(q.get("source"), clean_text(q.get("query")).lower()) for q in query_plan.get("queries", [])}
    sources = ["openalex", "crossref", "semantic_scholar"]
    if config.web_search_provider == "serpapi":
        sources.append("serpapi_google_scholar")
    core_terms = concept_groups.get("core_concepts", [])[:4]
    outcome_terms = concept_groups.get("outcome_terms", [])[:4]
    context_terms = concept_groups.get("context_terms", [])[:3]
    method_terms = concept_groups.get("method_terms", [])[:3]
    specs = []
    for core in core_terms:
        for outcome in outcome_terms:
            specs.append((f'"{core}" "{outcome}"', "supplemental_core_outcome", ["core_concepts", "outcome_terms"]))
    for core in core_terms[:3]:
        for context in context_terms:
            specs.append((f'"{core}" "{context}"', "supplemental_core_context", ["core_concepts", "context_terms"]))
    for core in core_terms[:3]:
        for method in method_terms:
            specs.append((f'"{core}" "{method}"', "supplemental_core_method", ["core_concepts", "method_terms"]))
    out = []
    for query, intent, facets in specs:
        for source in sources:
            key = (source, clean_text(query).lower())
            if key in existing:
                continue
            existing.add(key)
            out.append(SearchQuery(source=source, query=query, intent=intent, search_round=2, facets=facets).to_dict())
            if len(out) >= 24:
                return out
    return out


def match_user_files(record: Record, files: list[Path]) -> list[Path]:
    if not files:
        return []
    title_key = normalize_title(record.title)
    doi_key = record.doi.replace("/", "_").replace(".", "_")
    matched = []
    for path in files:
        name = normalize_title(path.stem)
        if doi_key and doi_key in path.stem.lower():
            matched.append(path)
        elif title_key and SequenceMatcher(None, title_key, name).ratio() >= 0.8:
            matched.append(path)
    return matched


def extract_user_fulltext(record: Record, files: list[Path]) -> dict[str, Any]:
    errors = []
    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                text = extract_pdf_text(path)
            elif path.suffix.lower() in {".txt", ".md"}:
                text = clean_fulltext(path.read_text(encoding="utf-8", errors="ignore"))
            else:
                errors.append(f"{path.name}: unsupported fulltext file type")
                continue
            if text:
                payload = fulltext_payload(text)
                payload.update({"status": "completed", "pdf_path": str(path), "source": "user_file"})
                return payload
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return {"status": "failed", "error": "; ".join(errors) or "No readable user fulltext file."}


def acquire_pdf_fulltext(record: Record, pdf_dir: Path) -> dict[str, Any]:
    urls = discover_pdf_urls(record)
    if not urls:
        return {"status": "no_pdf_url", "error": "No candidate PDF URL discovered."}
    errors = []
    for url in urls:
        try:
            pdf_path = pdf_dir / f"{record.record_id}.pdf"
            downloaded = download_pdf(url, pdf_path)
            text = extract_pdf_text(downloaded)
            if not text:
                raise RuntimeError("PDF text extraction returned no text.")
            payload = fulltext_payload(text)
            payload.update({"status": "completed", "pdf_url": url, "pdf_path": str(downloaded), "source": "pdf"})
            return payload
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    return {"status": "failed", "error": " | ".join(errors)}


def discover_pdf_urls(record: Record) -> list[str]:
    candidates: list[str] = []
    landing_pages: list[str] = []
    for url in [record.open_access_pdf, record.url]:
        if not url:
            continue
        if looks_like_pdf_url(url):
            candidates.append(url)
        elif is_arxiv_abs_url(url):
            candidates.append(arxiv_pdf_url(url))
        else:
            landing_pages.append(url)
    if record.doi:
        doi_url = "https://doi.org/" + record.doi
        landing_pages.append(doi_url)
        candidates.extend(known_publisher_pdf_urls(record.doi))
        candidates.extend(discover_crossref_pdf_urls(record.doi))
    for landing in compact_list(landing_pages):
        candidates.extend(discover_pdf_links_from_landing(landing))
    return [str(url) for url in compact_list(candidates)]


def known_publisher_pdf_urls(doi: str) -> list[str]:
    doi = normalize_doi(doi)
    if not doi:
        return []
    urls = []
    if doi.startswith("10.1145/"):
        urls.append(f"https://dl.acm.org/doi/pdf/{doi}")
    if doi.startswith("10.1080/"):
        urls.append(f"https://www.tandfonline.com/doi/pdf/{doi}")
    if doi.startswith("10.1007/"):
        urls.append(f"https://link.springer.com/content/pdf/{doi}.pdf")
    if doi.startswith("10.3389/"):
        urls.append(f"https://www.frontiersin.org/articles/{doi}/pdf")
    return urls


def discover_crossref_pdf_urls(doi: str) -> list[str]:
    doi = normalize_doi(doi)
    if not doi:
        return []
    if doi in CROSSREF_PDF_CACHE:
        return CROSSREF_PDF_CACHE[doi]
    endpoint = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        req = urllib.request.Request(endpoint, headers=REQUEST_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            import json

            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        exc.close()
        CROSSREF_PDF_CACHE[doi] = []
        return []
    except Exception:
        CROSSREF_PDF_CACHE[doi] = []
        return []
    message = data.get("message") or {}
    urls = []
    for item in message.get("link") or []:
        url = item.get("URL") or ""
        content_type = (item.get("content-type") or "").lower()
        if not url:
            continue
        if "pdf" in content_type or looks_like_pdf_url(url):
            urls.append(url)
        pii = extract_elsevier_pii(url)
        if pii:
            urls.append(f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true")
    result = [str(url) for url in compact_list(urls)]
    CROSSREF_PDF_CACHE[doi] = result
    return result


def extract_elsevier_pii(url: str) -> str:
    match = re.search(r"PII:([A-Za-z0-9]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"/pii/([A-Za-z0-9]+)", url)
    return match.group(1) if match else ""


def discover_pdf_links_from_landing(url: str) -> list[str]:
    try:
        req = urllib.request.Request(url, headers=REQUEST_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            final_url = resp.geturl()
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read(1_500_000)
    except urllib.error.HTTPError as exc:
        exc.close()
        return []
    except Exception:
        return []
    if data.startswith(b"%PDF") or "application/pdf" in content_type.lower():
        return [final_url]
    html = data.decode("utf-8", errors="ignore")
    links = []
    meta_patterns = [
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
    ]
    for pattern in meta_patterns:
        links.extend(re.findall(pattern, html, flags=re.I))
    for href in re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I):
        absolute = urllib.parse.urljoin(final_url, href.replace("&amp;", "&"))
        if (looks_like_pdf_url(absolute) or likely_pdf_link(absolute)) and allowed_landing_pdf_link(final_url, absolute):
            links.append(absolute)
    return [urllib.parse.urljoin(final_url, link.replace("&amp;", "&")) for link in compact_list(links)]


def looks_like_pdf_url(url: str) -> bool:
    lower = url.lower()
    return lower.endswith(".pdf") or ".pdf?" in lower or "/pdf/" in lower or "/pdf?" in lower or "openreview.net/pdf" in lower


def likely_pdf_link(url: str) -> bool:
    lower = url.lower()
    return any(fragment in lower for fragment in ["/doi/pdf", "/pdfft", "download=true", "download=1"])


def allowed_landing_pdf_link(landing_url: str, pdf_url: str) -> bool:
    landing_host = urllib.parse.urlparse(landing_url).netloc.lower()
    pdf_host = urllib.parse.urlparse(pdf_url).netloc.lower()
    if not landing_host or not pdf_host:
        return False
    if pdf_host == landing_host or pdf_host.endswith("." + landing_host):
        return True
    return any(fragment in pdf_url.lower() for fragment in ["openreview.net/pdf", "/doi/pdf", "/pdfft"])


def is_arxiv_abs_url(url: str) -> bool:
    return "arxiv.org/abs/" in url.lower()


def arxiv_pdf_url(url: str) -> str:
    paper_id = url.rstrip("/").split("/abs/", 1)[-1]
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def download_pdf(url: str, path: Path) -> Path:
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=45) as resp:
        content_type = resp.headers.get("Content-Type", "")
        data = resp.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise RuntimeError("PDF exceeds maximum download size.")
    if not data.startswith(b"%PDF") and "application/pdf" not in content_type.lower():
        raise RuntimeError(f"Response is not a PDF; content type={content_type or 'unknown'}.")
    ensure_dir(path.parent)
    path.write_bytes(data)
    return path


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("pypdf is not installed. Install it with `python -m pip install pypdf` or `python -m pip install -e .[pdf]`.") from exc
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return clean_fulltext("\n\n".join(pages))


def fulltext_payload(text: str) -> dict[str, Any]:
    clean = clean_fulltext(text)
    excerpt = clean[:MAX_FULLTEXT_CHARS]
    return {
        "char_count": len(clean),
        "excerpt": excerpt,
        "sections": extract_fulltext_sections(clean),
    }


def clean_fulltext(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_fulltext_sections(text: str) -> dict[str, str]:
    groups = {
        "abstract": ["abstract"],
        "introduction": ["introduction", "background"],
        "method": ["method", "methods", "methodology", "materials and methods", "study design"],
        "results_findings": ["results", "findings"],
        "discussion": ["discussion"],
        "conclusion": ["conclusion", "conclusions"],
        "limitations": ["limitations", "limitation"],
    }
    headings = []
    for key, aliases in groups.items():
        for alias in aliases:
            pattern = rf"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?{re.escape(alias)}s?\s*$"
            match = re.search(pattern, text)
            if match:
                headings.append((match.start(), match.end(), key))
                break
    headings.sort()
    sections: dict[str, str] = {}
    for idx, (start, end, key) in enumerate(headings):
        next_start = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        body = clean_fulltext(text[end:next_start])
        if body:
            sections[key] = body[:MAX_SECTION_CHARS]
    if not sections and text:
        sections["fulltext_excerpt"] = text[:MAX_FULLTEXT_CHARS]
    return sections


def summary_constraints(level: str) -> str:
    if level == "metadata_only":
        return "Only bibliographic metadata is available; do not summarize methods or findings."
    if level == "abstract_only":
        return "Abstract is available; summarize Method/Results only when explicitly stated in the abstract."
    if level == "fulltext_pdf":
        return "PDF full text was extracted; use extracted sections as primary evidence and note any missing sections."
    return "User-provided full text is available; use extracted text as primary evidence and note any missing sections."


def build_summary(record: Record, bundle: dict[str, Any]) -> dict[str, Any]:
    abstract = clean_text(record.abstract)
    level = bundle.get("evidence_level", record.evidence_level)
    base_note = summary_constraints(level)
    sections = bundle.get("fulltext_sections") or {}
    has_fulltext = level in {"fulltext_pdf", "user_provided_fulltext"} and bool(bundle.get("fulltext_excerpt") or sections)
    if abstract:
        abstract_summary = abstract
    elif sections.get("abstract"):
        abstract_summary = sections["abstract"]
    else:
        abstract_summary = "초록을 확인할 수 없습니다. 현재 요약은 메타데이터에 근거합니다."
    if has_fulltext:
        introduction_text = sections.get("introduction") or "추출된 원문에서 Introduction 섹션을 안정적으로 분리하지 못했습니다."
        method_text = sections.get("method") or "추출된 원문에서 Method 섹션을 안정적으로 분리하지 못했습니다. fulltext_excerpt를 확인해 수동 검토가 필요합니다."
        findings_text = sections.get("results_findings") or sections.get("discussion") or "추출된 원문에서 Results/Findings 섹션을 안정적으로 분리하지 못했습니다. fulltext_excerpt를 확인해 수동 검토가 필요합니다."
        conclusion_text = sections.get("conclusion") or "추출된 원문에서 Conclusion 섹션을 안정적으로 분리하지 못했습니다."
        limitations_text = sections.get("limitations") or base_note
    else:
        limited = level in {"metadata_only", "abstract_only"}
        introduction_text = "문제의식과 연구 맥락은 초록 및 제목에서 확인되는 범위로 제한해 파악해야 합니다."
        method_text = "확인 가능한 초록/메타데이터 안에서 방법 정보가 명시된 경우에만 해석해야 합니다." if limited else "사용자 제공 원문 기반으로 방법 섹션 확장이 가능합니다."
        findings_text = "초록에 명시된 발견만 요약 대상으로 삼습니다. 원문 확인 전에는 세부 결과를 추정하지 않습니다." if limited else "사용자 제공 원문 기반으로 결과 섹션 확장이 가능합니다."
        conclusion_text = "결론은 초록에 직접 제시된 주장으로 제한합니다. 고위험 의사결정에 사용하려면 원문 검토가 필요합니다."
        limitations_text = base_note
    relevance = f"검색 주제와의 관련성 점수는 {record.relevance_score}/3입니다. 제목, 초록, 출처 메타데이터의 키워드 중첩을 기준으로 자동 판정했습니다."
    purpose_text = sections.get("introduction") or introduction_text
    theory_text = sections.get("introduction") or sections.get("discussion") or "원문은 확보되었지만 이론적 배경 섹션을 자동으로 안정 분리하지 못했습니다."
    design_text = sections.get("method") or method_text
    measures_text = sections.get("method") or "측정도구, 변수, 지표 정보는 추출 원문에서 수동 재확인이 필요합니다."
    analysis_text = sections.get("method") or "분석 방법 정보는 추출 원문에서 수동 재확인이 필요합니다."
    findings_detail = sections.get("results_findings") or sections.get("discussion") or findings_text
    discussion_text = sections.get("discussion") or sections.get("conclusion") or conclusion_text
    return {
        "record_id": record.record_id,
        "citation": format_citation(record),
        "title": record.title,
        "evidence_level": level,
        "summary_status": "completed",
        "abstract": abstract_summary,
        "introduction": introduction_text,
        "research_purpose_questions": purpose_text,
        "theoretical_background": theory_text,
        "study_design_data_sample_context": design_text,
        "measures_variables_indicators": measures_text,
        "analysis_methods": analysis_text,
        "method": method_text,
        "results_findings": findings_text,
        "key_findings": findings_detail,
        "discussion_contribution": discussion_text,
        "conclusion": conclusion_text,
        "limitations": limitations_text,
        "topic_relevance": relevance,
        "follow_up": "core 문헌이면 원문 PDF 또는 기관 DB 원문을 확보해 Method, Measures, Validation 지표를 수동 검증하세요.",
    }


def format_citation(record: Record) -> str:
    author = record.authors[0] if record.authors else "Unknown"
    if len(record.authors) > 1:
        author = f"{author} et al."
    year = record.year or "n.d."
    return f"{author} ({year}). {record.title}."


def flag(record: Record, code: str, message: str) -> dict[str, Any]:
    return {"record_id": record.record_id, "title": record.title, "code": code, "message": message}


def write_coverage_audit(path: Path, records: list[Record], flags: list[dict[str, Any]], coverage_report: dict[str, Any] | None = None, out_dir: Path | None = None) -> None:
    included = [r for r in records if r.inclusion_status == "included"]
    by_evidence = Counter(r.evidence_level for r in included)
    by_source = Counter(src for r in records for src in r.source_database)
    by_round = Counter(r.search_round for r in records)
    by_found = Counter(found for r in records for found in r.found_by)
    coverage_report = coverage_report or {}
    lines = [
        "# Coverage Audit",
        "",
        f"Generated at: {utc_now()}",
        f"Total normalized records: {len(records)}",
        f"Included records: {len(included)}",
        f"Recall mode: {coverage_report.get('recall_mode', 'unknown')}",
        f"Web search provider: {coverage_report.get('web_search_provider', 'unknown')}",
        "",
        "## What This Result Can And Cannot Guarantee",
        "",
        "This run improves recall by combining public scholarly APIs, optional Google Scholar-style web search, coverage auditing, and citation snowballing. It still cannot guarantee a complete universe of literature because paid databases and publisher search indexes are not exhaustively queried through official connectors.",
        "",
        "## Facet Coverage",
    ]
    facet_counts = coverage_report.get("facet_counts") or {}
    if facet_counts:
        lines.append("| Facet | All Records | Included Records |")
        lines.append("|---|---:|---:|")
        for facet, counts in facet_counts.items():
            lines.append(f"| {facet} | {counts.get('all_records', 0)} | {counts.get('included_records', 0)} |")
    else:
        lines.append("- No facet coverage report available.")
    combo_counts = coverage_report.get("combo_counts") or {}
    lines.extend(["", "## Key Facet Combinations"])
    if combo_counts:
        for key, count in combo_counts.items():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- No facet combination report available.")
    lines.extend(["", "## Coverage Warnings"])
    warnings = coverage_report.get("warnings") or []
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning.get('code')}: {warning.get('message')} (count={warning.get('count')}, threshold={warning.get('threshold')})")
    else:
        lines.append("- No coverage warnings generated.")
    lines.extend(
        [
            "",
            "## Search Rounds",
        ]
    )
    for key, count in sorted(by_round.items()):
        lines.append(f"- round {key}: {count} normalized records")
    if out_dir and (out_dir / "search_log.jsonl").exists():
        logs = read_jsonl(out_dir / "search_log.jsonl")
        result_by_source = Counter()
        query_by_source = Counter()
        for log in logs:
            source = log.get("source", "unknown")
            query_by_source[source] += 1
            result_by_source[source] += int(log.get("result_count") or 0)
        lines.extend(["", "## Search Log Summary"])
        for source in sorted(query_by_source):
            lines.append(f"- {source}: {query_by_source[source]} queries, {result_by_source[source]} raw results")
    lines.extend(
        [
            "",
            "## Found By",
        ]
    )
    if by_found:
        for key, count in by_found.most_common(30):
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- No found_by metadata available.")
    lines.extend(
        [
            "",
            "## Evidence Levels",
        ]
    )
    for key, count in sorted(by_evidence.items()):
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## Sources")
    for key, count in sorted(by_source.items()):
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## QA Flags")
    if flags:
        for item in flags:
            lines.append(f"- {item['code']}: {item['title']} - {item['message']}")
    else:
        lines.append("- No QA flags generated.")
    path.write_text("\n".join(lines), encoding="utf-8")
