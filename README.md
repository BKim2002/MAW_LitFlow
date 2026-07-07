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

## CLI Options

All options are passed after `python -m litflow run`. Option order does not
matter as long as each option value comes right after the option name.

Required options:

| Option | What it means | Example |
|---|---|---|
| `--topic` | Literature-search topic or user request. Korean topics are expanded into English search terms. | `--topic "LLM을 활용한 면접 평가와 역량 측정 연구"` |
| `--out` | Run output directory. Intermediate artifacts and final outputs are written here. Reusing the same directory can update the same Notion run if `notion_manifest.json` exists. | `--out outputs/litflow-demo` |

Search scope and volume:

| Option | Default | When to use |
|---|---:|---|
| `--recall-mode` | `high` | Controls search breadth: `fast`, `balanced`, or `high`. `high` runs broader facet queries, coverage audit, supplemental queries, and snowballing. |
| `--web-search-provider` | `serpapi` | Use `serpapi` for Google Scholar-style round-2 recall expansion, or `none` to skip web/scholar API calls. |
| `--web-search-token-env` | `SERPAPI_API_KEY` | Environment-variable name that stores the SerpAPI key. If missing, litflow logs a warning and continues with public scholarly APIs. |
| `--year-from` | none | Limit results to papers published from this year onward. Example: `--year-from 2020`. |
| `--year-to` | none | Limit results to papers published up to this year. Example: `--year-to 2026`. |
| `--language` | `ko` | Output-language setting for generated artifacts. Metadata such as titles, venues, and DOI values keep their original notation. |
| `--max-raw` | `1500` | Upper budget for the raw candidate pool before deduplication. Use a smaller value for smoke tests. |
| `--max-deep` | `50` | Maximum number of included records to receive long structured summaries, PDF full-text extraction attempts, and paper-level Notion pages. |
| `--user-files` | none | Directory containing user-provided full-text files. Matching PDF/TXT/MD files are extracted and raise the evidence level to `user_provided_fulltext`. |

Agents SDK options:

| Option | Default | When to use |
|---|---:|---|
| `--agent-mode` | `auto` | `auto` uses the Agents SDK when `openai-agents` and `OPENAI_API_KEY` are available; `sdk` requires it; `off` disables SDK calls. |
| `--agent-model` | SDK default | Override the SDK model for judgment-heavy stages. Example: `--agent-model gpt-5.4-mini`. |

Output options:

| Option | Default | When to use |
|---|---:|---|
| `--output-format` | `files` | `files` creates DOCX/XLSX, `notion` creates Notion pages only, and `both` creates both. |
| `--notion-parent` | none | Parent Notion page URL or ID for the first Notion run. Required when creating a Notion run without an existing manifest. |
| `--notion-run-page` | none | Existing Notion run hub page URL or ID to update explicitly. Useful when `notion_manifest.json` is unavailable. |
| `--notion-token-env` | `NOTION_TOKEN` | Environment-variable name that stores the Notion integration token. Change this only if you use a different token variable name. |

PowerShell example with several options:

```powershell
python -m litflow run `
  --topic "LLM을 활용한 면접 평가와 역량 측정 연구" `
  --out outputs/litflow-demo `
  --year-from 2020 `
  --year-to 2026 `
  --recall-mode high `
  --web-search-provider serpapi `
  --max-raw 500 `
  --max-deep 30 `
  --agent-mode auto `
  --output-format notion `
  --notion-parent "https://app.notion.com/p/LitSearch-3963ec09355b80d08e6ff804d8b01bab?source=copy_link"
```

## Notion Output

Install the optional Notion dependency and set a Notion integration token:

```powershell
python -m pip install -e ".[notion]"
$env:NOTION_TOKEN = "secret_..."
```

Share the target parent page with the Notion integration, then run:

```powershell
python -m litflow run --topic "AI literacy and job performance" --out outputs/ai-literacy --output-format notion --notion-parent "https://www.notion.so/..."
```

Output modes:

- `--output-format files`: create the local XLSX and DOCX files only. This is the default.
- `--output-format notion`: create or update a Notion run hub page and paper subpages only.
- `--output-format both`: create local files and Notion pages.

Notion reruns use `notion_manifest.json` inside the output directory. If the
manifest exists, litflow updates the same hub and paper pages. You can also pass
`--notion-run-page` to update an existing hub page explicitly.

The Notion output uses native Notion blocks to keep the result calm and scannable:
summary callouts, dividers, clear heading hierarchy, a paper-link map, and
evidence-level callouts on each paper page.

## PDF Full Text

For included records that are selected for deep summary, litflow tries to discover
and download an accessible PDF, extract text with `pypdf`, and pass extracted
sections to the summary agent. When extraction succeeds, the record evidence level
becomes `fulltext_pdf`; otherwise it falls back to abstract or metadata evidence.
Publisher login walls, bot protection, and non-PDF HTML full text can still prevent
automatic extraction even when the paper is readable in a browser.

## High-Recall Search

The default `--recall-mode high` changes search from a one-pass query into a
facet-based recall workflow. Litflow decomposes the topic into concept facets
such as core concepts, outcomes, workplace/context terms, and method terms. For
example, a Korean topic about generative AI literacy and job performance creates
queries that include terms such as `AI literacy`, `generative AI literacy`,
`job performance`, `productivity`, `employee`, `workplace`, `scale development`,
and `validation`.

Search proceeds in rounds:

- Round 1: OpenAlex, Crossref, Semantic Scholar, and arXiv.
- Round 2: optional SerpAPI Google Scholar-style search and coverage-triggered supplemental queries.
- Round 3: Semantic Scholar citation/reference snowballing from core records.

Every candidate keeps `found_by`, `search_round`, `facet_matches`, and
`coverage_warning` fields. These fields appear in `screened_records.jsonl`, the
XLSX candidate sheets, and Notion paper pages. `coverage_audit.md` summarizes
facet coverage, weak concept combinations, search rounds, source counts, and
remaining manual database gaps.

To enable SerpAPI in PowerShell:

```powershell
$env:SERPAPI_API_KEY = "your-serpapi-key"
python -m litflow run --topic "AI literacy and job performance" --out outputs/ai-literacy --recall-mode high --web-search-provider serpapi
```

If `SERPAPI_API_KEY` is not set, the workflow does not fail. It writes a clear
warning to `search_log.jsonl` and continues with the public scholarly connectors.

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
- `fulltext_pdfs/*.pdf` when PDF extraction succeeds
- `summaries.jsonl`
- `qa_flags.jsonl`
- `coverage_audit.md`
- `literature_review.xlsx`
- `literature_review.docx`
- `notion_manifest.json` when `--output-format notion` or `both` is used

## Notes

- Public automated connectors: OpenAlex, Crossref, Semantic Scholar, arXiv.
- Optional web/scholar connector: SerpAPI Google Scholar-style search when
  `SERPAPI_API_KEY` is available.
- Paid scholarly databases are represented through reproducible manual search
  strings in `manual_db_search_pack.md`.
- DOCX output intentionally omits URLs to avoid fragile Word hyperlink handling;
  URLs are preserved in the XLSX workbook.
- Notion output uses the official Notion API through `notion-client`, not the
  Codex Notion connector. The Notion integration must have access to the parent
  page.
