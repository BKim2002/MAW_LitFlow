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
    run.add_argument("--output-format", choices=["files", "notion", "both"], default="files", help="Final packaging target: local DOCX/XLSX files, Notion, or both.")
    run.add_argument("--notion-parent", default=None, help="Notion parent page URL or ID for creating a new run hub page.")
    run.add_argument("--notion-run-page", default=None, help="Existing Notion run hub page URL or ID to update.")
    run.add_argument("--notion-token-env", default="NOTION_TOKEN", help="Environment variable that contains the Notion integration token.")
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
    return 2
