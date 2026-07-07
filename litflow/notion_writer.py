from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .agents import format_citation
from .models import Record, RunConfig
from .utils import read_json, read_jsonl, utc_now, write_json

RICH_TEXT_LIMIT = 1900
BLOCK_APPEND_LIMIT = 100
NOTION_ID_RE = re.compile(
    r"([0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


class NotionPackagingAgent:
    name = "NotionPackagingAgent"

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    def run(self, records: list[Record], summaries: list[dict[str, Any]], flags: list[dict[str, Any]], config: RunConfig, out_dir: Path) -> dict[str, str]:
        client = self.client or make_notion_client(config.notion_token_env)
        manifest_path = out_dir / "notion_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        hub_id = resolve_hub_id(config, manifest)
        hub_url = manifest.get("hub_url", "")
        hub_title = trim_title(f"Literature Review - {config.topic}")

        if hub_id:
            update_page_title(client, hub_id, hub_title)
            clear_page_content(client, hub_id)
        else:
            if not config.notion_parent:
                raise RuntimeError("--notion-parent is required for the first Notion run unless --notion-run-page or notion_manifest.json is available.")
            parent_id = parse_notion_page_id(config.notion_parent)
            hub = create_page(client, parent_id, hub_title, [])
            hub_id = hub["id"]
            hub_url = hub.get("url", notion_page_url(hub_id))

        record_by_id = {record.record_id: record for record in records}
        child_pages = dict(manifest.get("child_pages") or {})
        current_children: dict[str, dict[str, str]] = {}
        for summary in summaries:
            record_id = summary.get("record_id", "")
            record = record_by_id.get(record_id) or record_from_summary(summary)
            child_title = trim_title(summary.get("citation") or summary.get("title") or record.title or record_id)
            child_blocks = summary_record_to_blocks(summary, record)
            existing = child_pages.get(record_id) or {}
            page_id = existing.get("page_id")
            if page_id:
                update_page_title(client, page_id, child_title)
                clear_page_content(client, page_id)
                append_blocks(client, page_id, child_blocks)
                page_url = existing.get("url") or notion_page_url(page_id)
            else:
                page = create_page(client, hub_id, child_title, child_blocks)
                page_id = page["id"]
                page_url = page.get("url", notion_page_url(page_id))
            current_children[record_id] = {"page_id": page_id, "url": page_url, "title": child_title}

        child_pages.update(current_children)
        hub_blocks = hub_page_blocks(records, summaries, flags, config, out_dir, current_children)
        append_blocks(client, hub_id, hub_blocks)

        new_manifest = {
            "hub_page_id": hub_id,
            "hub_url": hub_url or notion_page_url(hub_id),
            "updated_at": utc_now(),
            "output_format": config.output_format,
            "topic": config.topic,
            "child_pages": child_pages,
        }
        write_json(manifest_path, new_manifest)
        return {"notion": new_manifest["hub_url"], "notion_manifest": str(manifest_path)}


def make_notion_client(token_env: str):
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"{token_env} is not set. Create a Notion integration token and set it before using --output-format notion.")
    try:
        from notion_client import Client
    except Exception as exc:
        raise RuntimeError("notion-client is not installed. Install it with `python -m pip install notion-client` or `python -m pip install -e .[notion]`.") from exc
    return Client(auth=token)


def resolve_hub_id(config: RunConfig, manifest: dict[str, Any]) -> str:
    if config.notion_run_page:
        return parse_notion_page_id(config.notion_run_page)
    hub_id = manifest.get("hub_page_id")
    return str(hub_id) if hub_id else ""


def parse_notion_page_id(value: str) -> str:
    text = unquote(str(value or "").strip())
    matches = NOTION_ID_RE.findall(text)
    if not matches:
        raise ValueError(f"Could not parse a Notion page ID from: {value}")
    return matches[-1].replace("-", "")


def create_page(client: Any, parent_page_id: str, title: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    page = client.pages.create(parent={"page_id": parent_page_id}, properties=title_properties(title))
    append_blocks(client, page["id"], blocks)
    return page


def update_page_title(client: Any, page_id: str, title: str) -> None:
    client.pages.update(page_id=page_id, properties=title_properties(title))


def title_properties(title: str) -> dict[str, Any]:
    return {"title": [{"type": "text", "text": {"content": trim_title(title)}}]}


def clear_page_content(client: Any, page_id: str) -> None:
    cursor = None
    while True:
        kwargs = {"block_id": page_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        response = client.blocks.children.list(**kwargs)
        for child in response.get("results", []):
            if child.get("type") == "child_page":
                continue
            client.blocks.delete(block_id=child["id"])
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")


def append_blocks(client: Any, page_id: str, blocks: list[dict[str, Any]]) -> None:
    for chunk in chunked(blocks, BLOCK_APPEND_LIMIT):
        if chunk:
            client.blocks.children.append(block_id=page_id, children=chunk)


def hub_page_blocks(
    records: list[Record],
    summaries: list[dict[str, Any]],
    flags: list[dict[str, Any]],
    config: RunConfig,
    out_dir: Path,
    child_pages: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    included = [record for record in records if record.inclusion_status == "included"]
    query_plan = read_json(out_dir / "query_plan.json") if (out_dir / "query_plan.json").exists() else {"queries": []}
    search_logs = read_jsonl(out_dir / "search_log.jsonl")
    manifest = read_json(out_dir / "run_manifest.json") if (out_dir / "run_manifest.json").exists() else {}
    source_counts = Counter(log.get("source", "unknown") for log in search_logs)
    result_counts = Counter()
    for log in search_logs:
        result_counts[log.get("source", "unknown")] += int(log.get("result_count") or 0)

    blocks: list[dict[str, Any]] = [
        callout(
            "\n".join(
                [
                    "실행 요약",
                    f"주제: {config.topic}",
                    f"전체 후보 {len(records)}건 | 포함 문헌 {len(included)}건 | 장문 요약 {len(summaries)}건 | QA 플래그 {len(flags)}건",
                    f"생성 시각: {utc_now()}",
                ]
            ),
            icon="📚",
            color="gray_background",
        ),
        divider(),
        heading("핵심 문헌 맵", 1),
    ]
    if summaries:
        for idx, summary in enumerate(summaries, start=1):
            child = child_pages.get(summary.get("record_id", ""), {})
            label = f"{idx}. {summary.get('citation') or summary.get('title') or summary.get('record_id')}"
            blocks.append(numbered(label, child.get("url")))
    else:
        blocks.append(paragraph("장문 요약 대상 문헌이 없습니다."))

    blocks.extend(
        [
            divider(),
            heading("조사 범위와 실행 설정", 1),
            callout(
                "공개 커넥터 기반 자동 검색 결과입니다. Google Scholar, Scopus, Web of Science, ACM Digital Library, IEEE Xplore 등은 manual_db_search_pack.md의 재현 가능한 검색식으로 별도 확인이 필요합니다.",
                icon="ℹ️",
                color="gray_background",
            ),
        ]
    )
    for key, value in config.to_dict().items():
        blocks.append(bulleted(f"{key}: {value}"))
    sdk_status = manifest.get("agents_sdk")
    if sdk_status:
        blocks.append(bulleted(f"agents_sdk: {sdk_status}"))

    blocks.extend([divider(), heading("검색식 요약", 1)])
    for query in query_plan.get("queries", [])[:12]:
        blocks.append(bulleted(f"[{query.get('source')}] {query.get('query')}"))
    if not query_plan.get("queries"):
        blocks.append(paragraph("검색식 정보가 없습니다."))

    blocks.extend([divider(), heading("검색 로그 요약", 1)])
    if search_logs:
        for source in sorted(source_counts):
            blocks.append(bulleted(f"{source}: {source_counts[source]} queries, {result_counts[source]} raw results"))
    else:
        blocks.append(paragraph("검색 로그가 없습니다."))

    blocks.extend([divider(), heading("커버리지 한계", 1)])
    coverage_path = out_dir / "coverage_audit.md"
    if coverage_path.exists():
        coverage_lines = []
        for line in coverage_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                coverage_lines.append(line.strip().lstrip("# "))
        blocks.append(callout("\n".join(coverage_lines) if coverage_lines else "커버리지 한계 정보가 없습니다.", icon="🔎", color="gray_background"))
    else:
        blocks.append(paragraph("coverage_audit.md가 아직 생성되지 않았습니다."))

    blocks.extend([divider(), heading("QA 플래그", 1)])
    if flags:
        blocks.append(callout(f"검토가 필요한 QA 플래그 {len(flags)}건이 있습니다. 아래 목록에서 record_id와 code를 기준으로 확인하세요.", icon="⚠️", color="gray_background"))
        for item in flags[:50]:
            blocks.append(bulleted(f"{item.get('record_id', '')} | {item.get('code', '')}: {item.get('message', '')}"))
    else:
        blocks.append(callout("QA 플래그가 없습니다.", icon="✅", color="gray_background"))
    return blocks


def summary_record_to_blocks(summary: dict[str, Any], record: Record) -> list[dict[str, Any]]:
    metadata = [
        ("Citation", summary.get("citation") or format_citation(record)),
        ("Title", summary.get("title") or record.title),
        ("Authors", "; ".join(record.authors)),
        ("Year", record.year or ""),
        ("Venue", record.venue),
        ("DOI", record.doi),
        ("URL", record.url),
        ("Evidence Level", summary.get("evidence_level") or record.evidence_level),
        ("Relevance Score", record.relevance_score),
        ("Source Databases", "; ".join(record.source_database)),
        ("Record ID", record.record_id),
    ]
    sections = [
        ("Abstract", summary.get("abstract", "")),
        ("Introduction", summary.get("introduction", "")),
        ("Method", summary.get("method", "")),
        ("Results/Findings", summary.get("results_findings", "")),
        ("Conclusion", summary.get("conclusion", "")),
        ("Limitations", summary.get("limitations", "")),
        ("사용자의 주제와의 관련성", summary.get("topic_relevance", "")),
        ("후속 검토 필요성", summary.get("follow_up", "")),
    ]
    evidence_level = summary.get("evidence_level") or record.evidence_level
    blocks: list[dict[str, Any]] = [
        callout(
            f"Evidence Level: {evidence_level}\n이 페이지의 Method/Results 해석은 확보된 근거 수준을 넘지 않도록 제한됩니다.",
            icon="🔎",
            color="gray_background",
        ),
        divider(),
        heading("문헌 정보", 1),
    ]
    for label, value in metadata:
        if value not in ("", None, []):
            blocks.append(paragraph(f"{label}: {value}", link=str(value) if label == "URL" and value else None))
    blocks.extend([divider(), heading("구조화 요약", 1)])
    for label, value in sections:
        blocks.append(heading(label, 2))
        blocks.extend(paragraph_blocks(str(value or "확인 가능한 내용 없음")))
    return blocks


def record_from_summary(summary: dict[str, Any]) -> Record:
    return Record(
        record_id=summary.get("record_id", "missing-record-id"),
        title=summary.get("title", ""),
        evidence_level=summary.get("evidence_level", "metadata_only"),
    )


def paragraph_blocks(text: str) -> list[dict[str, Any]]:
    chunks = split_text(text or "", RICH_TEXT_LIMIT * 20)
    return [paragraph(chunk) for chunk in chunks] if chunks else [paragraph("")]


def paragraph(text: str, link: str | None = None) -> dict[str, Any]:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_text(text, link=link)}}


def callout(text: str, icon: str = "ℹ️", color: str = "gray_background") -> dict[str, Any]:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": rich_text(text),
            "icon": {"type": "emoji", "emoji": icon},
            "color": color,
        },
    }


def divider() -> dict[str, Any]:
    return {"object": "block", "type": "divider", "divider": {}}


def heading(text: str, level: int) -> dict[str, Any]:
    block_type = f"heading_{max(1, min(level, 3))}"
    return {"object": "block", "type": block_type, block_type: {"rich_text": rich_text(text)}}


def bulleted(text: str) -> dict[str, Any]:
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rich_text(text)}}


def numbered(text: str, link: str | None = None) -> dict[str, Any]:
    return {"object": "block", "type": "numbered_list_item", "numbered_list_item": {"rich_text": rich_text(text, link=link)}}


def rich_text(text: Any, link: str | None = None) -> list[dict[str, Any]]:
    chunks = split_text(str(text or ""), RICH_TEXT_LIMIT)
    if not chunks:
        chunks = [""]
    items = []
    for chunk in chunks:
        text_payload: dict[str, Any] = {"content": chunk}
        if link:
            text_payload["link"] = {"url": link}
        items.append({"type": "text", "text": text_payload})
    return items


def split_text(text: str, limit: int) -> list[str]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not text:
        return []
    chunks = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def chunked(values: list[Any], size: int) -> list[list[Any]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def trim_title(title: str, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", str(title or "Untitled")).strip()
    return text[:limit].rstrip() or "Untitled"


def notion_page_url(page_id: str) -> str:
    return f"https://www.notion.so/{str(page_id).replace('-', '')}"
