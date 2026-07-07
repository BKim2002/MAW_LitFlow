from __future__ import annotations

import math
import re
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
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)
from .sdk_bridge import SDKAgentBridge


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
        base_queries = make_queries(topic, concept_groups)
        sdk_rationale = ""
        sdk_assisted = False
        if self.sdk_bridge and self.sdk_bridge.enabled:
            deterministic_plan = {"expanded_terms": expanded, "concept_groups": concept_groups, "base_queries": base_queries}
            try:
                sdk_output = self.sdk_bridge.refine_query_strategy(topic, deterministic_plan)
                if sdk_output.expanded_terms:
                    expanded = compact_list([*expanded, *sdk_output.expanded_terms])
                if sdk_output.concept_groups:
                    concept_groups = merge_concept_groups(concept_groups, sdk_output.concept_groups)
                if sdk_output.base_queries:
                    base_queries = compact_list([*sdk_output.base_queries, *base_queries])[:8]
                sdk_rationale = sdk_output.rationale
                sdk_assisted = True
                log_sdk_event(out_dir, self.name, "completed", "SDK refined query strategy.")
            except Exception as exc:
                log_sdk_event(out_dir, self.name, "fallback", f"{type(exc).__name__}: {exc}")
        source_queries: list[SearchQuery] = []
        sources = ["openalex", "crossref", "semantic_scholar", "arxiv", "web"]
        for source in sources:
            for idx, query in enumerate(base_queries):
                source_queries.append(SearchQuery(source=source, query=query, intent=f"broad_query_{idx + 1}"))
        plan = {
            "agent": self.name,
            "created_at": utc_now(),
            "topic": topic,
            "sdk_assisted": sdk_assisted,
            "sdk_rationale": sdk_rationale,
            "expanded_terms": expanded,
            "concept_groups": concept_groups,
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

    def run(self, query_plan: dict[str, Any], config: RunConfig, out_dir: Path) -> list[dict[str, Any]]:
        raw_dir = ensure_dir(out_dir / "raw_results")
        search_logs: list[dict[str, Any]] = []
        source_records: dict[str, list[dict[str, Any]]] = {}
        queries = [SearchQuery(**q) for q in query_plan["queries"]]
        per_source_budget = max(1, min(config.per_source_cap, math.ceil(config.max_raw / max(1, len(self.connectors)))))
        source_counts: Counter[str] = Counter()
        for query in queries:
            if source_counts[query.source] >= per_source_budget:
                continue
            connector = self.connectors.get(query.source)
            if connector is None:
                continue
            remaining = per_source_budget - source_counts[query.source]
            result = connector.search(query.query, config, min(remaining, 50))
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
        for seed in seeds:
            paper_id = seed.source_ids["semantic_scholar"]
            for relation in ["references", "citations"]:
                result = self.connector.expand(paper_id, relation, limit=10)
                all_new.extend(record.to_dict() for record in result.records)
                report["rounds"].append(result.log.to_dict())
        if all_new:
            write_jsonl(raw_dir / "snowball_semantic_round1.jsonl", all_new)
        report["new_raw_records"] = len(all_new)
        report["status"] = "completed"
        write_json(out_dir / "snowball_report.json", report)
        return report


class EvidenceAcquisitionAgent:
    name = "EvidenceAcquisitionAgent"

    def run(self, records: list[Record], config: RunConfig, out_dir: Path) -> dict[str, Any]:
        evidence_dir = ensure_dir(out_dir / "evidence_bundles")
        user_files = list(Path(config.user_files).glob("*")) if config.user_files and Path(config.user_files).exists() else []
        bundle_index: dict[str, dict[str, Any]] = {}
        for record in records:
            level = "metadata_only"
            sources = []
            if record.abstract:
                level = "abstract_only"
                sources.append("abstract")
            if record.open_access_pdf:
                level = "fulltext_pdf"
                sources.append("open_access_pdf")
            matched_user_files = match_user_files(record, user_files)
            if matched_user_files:
                level = "user_provided_fulltext"
                sources.append("user_provided_file")
            record.evidence_level = level
            bundle = {
                "record_id": record.record_id,
                "title": record.title,
                "evidence_level": level,
                "sources": sources,
                "abstract_available": bool(record.abstract),
                "open_access_pdf": record.open_access_pdf,
                "user_files": [str(path) for path in matched_user_files],
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

    def run(self, records: list[Record], evidence: dict[str, dict[str, Any]], config: RunConfig, out_dir: Path) -> list[dict[str, Any]]:
        included = [r for r in records if r.inclusion_status == "included"]
        included.sort(key=lambda r: (r.relevance_score, r.citation_count or 0), reverse=True)
        selected = included[: config.max_deep]
        summaries: list[dict[str, Any]] = []
        for record in selected:
            bundle = evidence.get(record.record_id, {})
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
            summaries.append(summary)
            record.summary_status = "completed"
        for record in records:
            if record.inclusion_status == "included" and record.summary_status != "completed":
                record.summary_status = "deferred_max_deep"
        write_jsonl(out_dir / "summaries.jsonl", summaries)
        write_jsonl(out_dir / "screened_records.jsonl", [record.to_dict() for record in records])
        return summaries


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
        write_coverage_audit(out_dir / "coverage_audit.md", records, flags)
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
    if re.search(r"\bLLM\b|large language model|생성형|AI|인공지능", topic, re.I):
        expanded.extend(DEFAULT_DOMAIN_TERMS)
    return [str(x) for x in compact_list(expanded)]


def make_concept_groups(topic: str, expanded: list[str]) -> dict[str, list[str]]:
    ai_terms = [t for t in expanded if re.search(r"AI|LLM|language model|machine learning|artificial intelligence|GenAI", t, re.I)]
    methods = [t for t in expanded if re.search(r"assessment|evaluation|scoring|classification|extraction|decision|literacy|interview|skill", t, re.I)]
    if not ai_terms:
        ai_terms = DEFAULT_DOMAIN_TERMS[:3]
    if not methods:
        methods = expanded[:6]
    return {
        "topic_terms": expanded[:20],
        "ai_terms": compact_list(ai_terms)[:8],
        "task_terms": compact_list(methods)[:12],
        "quality_terms": ["validity", "reliability", "fairness", "bias", "human-AI collaboration", "systematic review"],
    }


def make_queries(topic: str, groups: dict[str, list[str]]) -> list[str]:
    ai = quote_or(groups["ai_terms"][:4])
    task = quote_or(groups["task_terms"][:5])
    quality = quote_or(groups["quality_terms"][:4])
    plain = " ".join(groups["topic_terms"][:8])
    queries = [
        topic,
        f"({ai}) AND ({task})",
        f"({ai}) AND ({task}) AND ({quality})",
        f"{plain} systematic review OR survey",
        f"{plain} empirical study validation",
    ]
    return compact_list(queries)


def quote_or(terms: list[str]) -> str:
    return " OR ".join(f'"{t}"' if " " in t else t for t in terms)


def write_manual_db_pack(path: Path, plan: dict[str, Any], config: RunConfig) -> None:
    lines = [
        "# Manual Database Search Pack",
        "",
        f"Topic: {config.topic}",
        "",
        "Use these queries in databases that are not fully automated in v1: Google Scholar, Scopus, Web of Science, ACM Digital Library, IEEE Xplore, and Dimensions.",
        "",
    ]
    for query in compact_list(q["query"] for q in plan["queries"] if q["source"] == "openalex"):
        lines.append(f"- `{query}`")
    lines.extend(
        [
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


def summary_constraints(level: str) -> str:
    if level == "metadata_only":
        return "Only bibliographic metadata is available; do not summarize methods or findings."
    if level == "abstract_only":
        return "Abstract is available; summarize Method/Results only when explicitly stated in the abstract."
    if level == "fulltext_pdf":
        return "Open-access PDF URL is available; v1 records availability and uses metadata/abstract unless full text is provided locally."
    return "User-provided full text is available; detailed summary can be expanded by a full-text parser."


def build_summary(record: Record, bundle: dict[str, Any]) -> dict[str, Any]:
    abstract = clean_text(record.abstract)
    level = bundle.get("evidence_level", record.evidence_level)
    base_note = summary_constraints(level)
    if abstract:
        abstract_summary = abstract
    else:
        abstract_summary = "초록을 확인할 수 없습니다. 현재 요약은 메타데이터에 근거합니다."
    limited = level in {"metadata_only", "abstract_only", "fulltext_pdf"}
    method_text = "확인 가능한 초록/메타데이터 안에서 방법 정보가 명시된 경우에만 해석해야 합니다." if limited else "사용자 제공 원문 기반으로 방법 섹션 확장이 가능합니다."
    findings_text = "초록에 명시된 발견만 요약 대상으로 삼습니다. 원문 확인 전에는 세부 결과를 추정하지 않습니다." if limited else "사용자 제공 원문 기반으로 결과 섹션 확장이 가능합니다."
    relevance = f"검색 주제와의 관련성 점수는 {record.relevance_score}/3입니다. 제목, 초록, 출처 메타데이터의 키워드 중첩을 기준으로 자동 판정했습니다."
    return {
        "record_id": record.record_id,
        "citation": format_citation(record),
        "title": record.title,
        "evidence_level": level,
        "summary_status": "completed",
        "abstract": abstract_summary,
        "introduction": "문제의식과 연구 맥락은 초록 및 제목에서 확인되는 범위로 제한해 파악해야 합니다.",
        "method": method_text,
        "results_findings": findings_text,
        "conclusion": "결론은 초록에 직접 제시된 주장으로 제한합니다. 고위험 의사결정에 사용하려면 원문 검토가 필요합니다.",
        "limitations": base_note,
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


def write_coverage_audit(path: Path, records: list[Record], flags: list[dict[str, Any]]) -> None:
    included = [r for r in records if r.inclusion_status == "included"]
    by_evidence = Counter(r.evidence_level for r in included)
    by_source = Counter(src for r in records for src in r.source_database)
    lines = [
        "# Coverage Audit",
        "",
        f"Generated at: {utc_now()}",
        f"Total normalized records: {len(records)}",
        f"Included records: {len(included)}",
        "",
        "## Evidence Levels",
    ]
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
