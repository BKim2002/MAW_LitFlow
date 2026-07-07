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
| `--year-from` | none | Limit results to papers published from this year onward. Example: `--year-from 2020`. |
| `--year-to` | none | Limit results to papers published up to this year. Example: `--year-to 2026`. |
| `--language` | `ko` | Output-language setting for generated artifacts. Metadata such as titles, venues, and DOI values keep their original notation. |
| `--max-raw` | `1500` | Upper budget for the raw candidate pool before deduplication. Use a smaller value for smoke tests. |
| `--max-deep` | `50` | Maximum number of included records to receive long structured summaries and paper-level Notion pages. |
| `--user-files` | none | Directory containing user-provided full-text files. Matching files raise the evidence level to `user_provided_fulltext`. |

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
- `notion_manifest.json` when `--output-format notion` or `both` is used

## Notes

- Public automated connectors: OpenAlex, Crossref, Semantic Scholar, arXiv.
- General web search and paid scholarly databases are represented through
  reproducible manual search strings in `manual_db_search_pack.md`.
- DOCX output intentionally omits URLs to avoid fragile Word hyperlink handling;
  URLs are preserved in the XLSX workbook.
- Notion output uses the official Notion API through `notion-client`, not the
  Codex Notion connector. The Notion integration must have access to the parent
  page.
