from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from .models import Record, RunConfig


class QueryStrategyOutput(BaseModel):
    expanded_terms: list[str] = Field(default_factory=list)
    concept_groups: dict[str, list[str]] = Field(default_factory=dict)
    base_queries: list[str] = Field(default_factory=list)
    rationale: str = ""


class RelevanceDecision(BaseModel):
    record_id: str
    relevance_score: int = Field(ge=0, le=3)
    inclusion_status: str
    exclusion_reason: str = ""


class RelevanceBatchOutput(BaseModel):
    decisions: list[RelevanceDecision] = Field(default_factory=list)


class LiteratureSummaryOutput(BaseModel):
    record_id: str
    citation: str
    title: str
    evidence_level: str
    summary_status: str = "completed"
    abstract: str
    introduction: str
    research_purpose_questions: str = ""
    theoretical_background: str = ""
    study_design_data_sample_context: str = ""
    measures_variables_indicators: str = ""
    analysis_methods: str = ""
    method: str
    results_findings: str
    key_findings: str = ""
    discussion_contribution: str = ""
    conclusion: str
    limitations: str
    topic_relevance: str
    follow_up: str


class QAFlagOutput(BaseModel):
    record_id: str
    title: str
    code: str
    message: str


class QABatchOutput(BaseModel):
    flags: list[QAFlagOutput] = Field(default_factory=list)


class SDKAgentBridge:
    """Optional wrapper around openai-agents-python.

    The rest of litflow remains deterministic and file-artifact based. This bridge
    only replaces judgment-heavy stages when the SDK and an API key are available.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.mode = config.agent_mode
        self.model = config.agent_model or os.environ.get("LITFLOW_AGENT_MODEL")
        self.available = False
        self.enabled = False
        self.reason = ""
        self._Agent = None
        self._Runner = None
        self._ModelSettings = None
        self._init_sdk()

    def _init_sdk(self) -> None:
        if self.mode == "off":
            self.reason = "agent_mode=off"
            return
        try:
            from agents import Agent, ModelSettings, Runner
        except Exception as exc:
            self.reason = f"openai-agents not importable: {type(exc).__name__}: {exc}"
            if self.mode == "sdk":
                raise RuntimeError(self.reason) from exc
            return
        self.available = True
        if not os.environ.get("OPENAI_API_KEY"):
            self.reason = "OPENAI_API_KEY is not set"
            if self.mode == "sdk":
                raise RuntimeError(self.reason)
            return
        self._Agent = Agent
        self._Runner = Runner
        self._ModelSettings = ModelSettings
        self.enabled = True
        self.reason = "enabled"

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "available": self.available,
            "enabled": self.enabled,
            "model": self.model or "agents-sdk-default",
            "reason": self.reason,
        }

    def _agent(self, name: str, instructions: str, output_type: type[BaseModel]):
        kwargs: dict[str, Any] = {
            "name": name,
            "instructions": instructions,
            "output_type": output_type,
        }
        if self.model:
            kwargs["model"] = self.model
        return self._Agent(**kwargs)

    def _run(self, agent, payload: dict[str, Any]) -> BaseModel:
        result = self._Runner.run_sync(agent, json.dumps(payload, ensure_ascii=False), max_turns=4)
        output = result.final_output
        if isinstance(output, BaseModel):
            return output
        if isinstance(output, str):
            return agent.output_type.model_validate_json(output)
        return agent.output_type.model_validate(output)

    def refine_query_strategy(self, topic: str, deterministic_plan: dict[str, Any]) -> QueryStrategyOutput:
        agent = self._agent(
            "Literature Query Strategy Agent",
            (
                "You design scholarly literature-search strategies. "
                "Return concise but broad English query terms for academic databases. "
                "Preserve the user's research intent, include synonyms across adjacent fields, "
                "and avoid adding unrelated domains. Output only the requested structured object."
            ),
            QueryStrategyOutput,
        )
        return self._run(agent, {"topic": topic, "deterministic_plan": deterministic_plan})

    def screen_relevance(self, topic: str, records: list[Record]) -> RelevanceBatchOutput:
        agent = self._agent(
            "Literature Relevance Screening Agent",
            (
                "You screen scholarly records for relevance to the user's topic. "
                "Score each record: 0 exclude, 1 peripheral, 2 relevant, 3 core. "
                "Use title, abstract, venue, and metadata only. Do not over-include weak matches. "
                "Return Korean exclusion reasons when excluded."
            ),
            RelevanceBatchOutput,
        )
        payload = {
            "topic": topic,
            "records": [
                {
                    "record_id": r.record_id,
                    "title": r.title,
                    "abstract": r.abstract,
                    "venue": r.venue,
                    "year": r.year,
                    "source_database": r.source_database,
                    "found_by": r.found_by,
                    "search_round": r.search_round,
                    "facet_matches": r.facet_matches,
                    "deterministic_score": r.relevance_score,
                }
                for r in records
            ],
        }
        return self._run(agent, payload)

    def summarize_record(self, topic: str, record: Record, evidence_bundle: dict[str, Any], fallback_citation: str) -> LiteratureSummaryOutput:
        agent = self._agent(
            "Literature Deep Summary Agent",
            (
                "You write detailed Korean structured summaries of scholarly papers. "
                "Use only the provided evidence. If fulltext_sections or fulltext_excerpt are present, "
                "use that extracted full text as the primary evidence. Summarize at a level where a "
                "researcher can understand the paper without opening it: purpose and questions, theory, "
                "design/data/sample/context, measures and indicators, analysis methods, key findings, "
                "discussion and contribution, limitations, and relevance to the user's topic. "
                "Do not invent methods, results, variables, or conclusions beyond the available full text."
            ),
            LiteratureSummaryOutput,
        )
        payload = {
            "topic": topic,
            "record": {
                "record_id": record.record_id,
                "title": record.title,
                "authors": record.authors,
                "year": record.year,
                "venue": record.venue,
                "abstract": record.abstract,
                "evidence_level": record.evidence_level,
                "relevance_score": record.relevance_score,
                "citation": fallback_citation,
            },
            "evidence_bundle": evidence_bundle,
        }
        return self._run(agent, payload)

    def audit_quality(self, topic: str, records: list[Record], summaries: list[dict[str, Any]], existing_flags: list[dict[str, Any]]) -> QABatchOutput:
        agent = self._agent(
            "Literature QA Agent",
            (
                "You audit literature-search outputs. Flag unsupported method/results claims, "
                "evidence-level mismatches, duplicate leakage, missing metadata, and relevance errors. "
                "Return only concrete issues. Use short machine-readable codes."
            ),
            QABatchOutput,
        )
        payload = {
            "topic": topic,
            "records": [
                {
                    "record_id": r.record_id,
                    "title": r.title,
                    "evidence_level": r.evidence_level,
                    "inclusion_status": r.inclusion_status,
                    "relevance_score": r.relevance_score,
                }
                for r in records
            ],
            "summaries": summaries,
            "existing_flags": existing_flags,
        }
        return self._run(agent, payload)
