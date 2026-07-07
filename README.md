# litflow

`litflow` is a local, file-artifact based workflow for broad literature discovery.
It implements the planned multi-agent architecture as deterministic Python agents
that hand off JSON/JSONL/Markdown/XLSX/DOCX artifacts through a run directory.

## Run

```powershell
python -m litflow run --topic "LLM을 활용한 면접 평가와 역량 측정 연구" --out outputs/litflow-demo --max-raw 100 --max-deep 20
```

Use the OpenAI Agents SDK for query strategy, relevance screening, deep summaries,
and QA when `openai-agents` and `OPENAI_API_KEY` are available:

```powershell
python -m litflow run --topic "LLM을 활용한 면접 평가와 역량 측정 연구" --out outputs/litflow-demo --max-raw 100 --max-deep 20 --agent-mode auto
```

Agent modes:

- `--agent-mode auto`: use the Agents SDK when installed and authenticated; otherwise fall back to deterministic logic.
- `--agent-mode sdk`: require the Agents SDK and `OPENAI_API_KEY`; fail fast if unavailable.
- `--agent-mode off`: disable SDK calls and use deterministic logic only.

Optional model override:

```powershell
python -m litflow run --topic "AI literacy and job performance" --out outputs/ai-literacy --agent-mode sdk --agent-model gpt-5.4-mini
```

## Outputs

- `run_manifest.json`
- `scope.json`
- `query_plan.json`
- `manual_db_search_pack.md`
- `search_log.jsonl`
- `raw_results/*.jsonl`
- `normalized_records.jsonl`
- `dedup_report.json`
- `screened_records.jsonl`
- `evidence_bundles/*.json`
- `summaries.jsonl`
- `qa_flags.jsonl`
- `coverage_audit.md`
- `literature_review.xlsx`
- `literature_review.docx`

## Notes

- Public automated connectors: OpenAlex, Crossref, Semantic Scholar, arXiv.
- General web search and paid scholarly databases are represented through
  reproducible manual search strings in `manual_db_search_pack.md`.
- DOCX output intentionally omits URLs to avoid fragile Word hyperlink handling;
  URLs are preserved in the XLSX workbook.
