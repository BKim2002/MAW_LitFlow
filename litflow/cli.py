from __future__ import annotations

import argparse
from pathlib import Path

from .models import RunConfig
from .orchestrator import Orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="litflow", description="Local multi-agent literature search workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a literature search workflow.")
    run.add_argument("--topic", required=True, help="Research topic or user request.")
    run.add_argument("--out", required=True, help="Output directory, e.g. outputs/<run_id>.")
    run.add_argument("--year-from", type=int, default=None)
    run.add_argument("--year-to", type=int, default=None)
    run.add_argument("--language", default="ko")
    run.add_argument("--max-raw", type=int, default=1500)
    run.add_argument("--max-deep", type=int, default=50)
    run.add_argument("--user-files", default=None)
    run.add_argument("--agent-mode", choices=["auto", "sdk", "off"], default="auto", help="Use Agents SDK for judgment-heavy stages: auto, sdk, or off.")
    run.add_argument("--agent-model", default=None, help="Optional OpenAI model name for Agents SDK runs. Defaults to the SDK/provider default.")
    run.add_argument("--recall-mode", choices=["fast", "balanced", "high"], default="high", help="Search recall depth. high runs broader facet queries and recall audits.")
    run.add_argument("--web-search-provider", choices=["serpapi", "none"], default="serpapi", help="Optional web/scholar search provider for round-2 recall expansion.")
    run.add_argument("--web-search-token-env", default="SERPAPI_API_KEY", help="Environment variable that contains the web search provider API key.")
    run.add_argument("--output-format", choices=["files", "notion", "both"], default="files", help="Final packaging target: local DOCX/XLSX files, Notion, or both.")
    run.add_argument("--notion-parent", default=None, help="Notion parent page URL or ID for creating a new run hub page.")
    run.add_argument("--notion-run-page", default=None, help="Existing Notion run hub page URL or ID to update.")
    run.add_argument("--notion-token-env", default="NOTION_TOKEN", help="Environment variable that contains the Notion integration token.")
    resume = sub.add_parser("resume-fulltext", help="Reuse an existing run and process newly provided full-text files without rerunning search.")
    resume.add_argument("--out", required=True, help="Existing output directory from a previous run.")
    resume.add_argument("--topic", default="", help="Optional topic override. Defaults to the previous run topic.")
    resume.add_argument("--language", default="ko")
    resume.add_argument("--max-deep", type=int, default=50)
    resume.add_argument("--user-files", default=None, help="Directory containing user-provided PDF/TXT/MD full text. Defaults to <out>/user_fulltext.")
    resume.add_argument("--agent-mode", choices=["auto", "sdk", "off"], default="auto", help="Use Agents SDK for newly summarized full-text records.")
    resume.add_argument("--agent-model", default=None, help="Optional OpenAI model name for Agents SDK runs.")
    resume.add_argument("--output-format", choices=["files", "notion", "both"], default="files", help="Final packaging target for the resumed run.")
    resume.add_argument("--notion-parent", default=None, help="Notion parent page URL or ID for creating a new run hub page if no manifest exists.")
    resume.add_argument("--notion-run-page", default=None, help="Existing Notion run hub page URL or ID to update.")
    resume.add_argument("--notion-token-env", default="NOTION_TOKEN", help="Environment variable that contains the Notion integration token.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = RunConfig(
            topic=args.topic,
            out=str(Path(args.out)),
            year_from=args.year_from,
            year_to=args.year_to,
            language=args.language,
            max_raw=args.max_raw,
            max_deep=args.max_deep,
            user_files=args.user_files,
            agent_mode=args.agent_mode,
            agent_model=args.agent_model,
            recall_mode=args.recall_mode,
            web_search_provider=args.web_search_provider,
            web_search_token_env=args.web_search_token_env,
            output_format=args.output_format,
            notion_parent=args.notion_parent,
            notion_run_page=args.notion_run_page,
            notion_token_env=args.notion_token_env,
        )
        result = Orchestrator().run(config)
        print(f"litflow completed: {result['out_dir']}")
        for key, value in result["outputs"].items():
            print(f"{key.upper()}: {value}")
        return 0
    if args.command == "resume-fulltext":
        config = RunConfig(
            topic=args.topic,
            out=str(Path(args.out)),
            language=args.language,
            max_deep=args.max_deep,
            user_files=args.user_files,
            agent_mode=args.agent_mode,
            agent_model=args.agent_model,
            output_format=args.output_format,
            notion_parent=args.notion_parent,
            notion_run_page=args.notion_run_page,
            notion_token_env=args.notion_token_env,
        )
        result = Orchestrator().resume_fulltext(config)
        print(f"litflow resume-fulltext completed: {result['out_dir']}")
        for key, value in result["outputs"].items():
            print(f"{key.upper()}: {value}")
        return 0
    return 2
