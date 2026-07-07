from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from .models import RawRecord, RunConfig, SearchLogEntry
from .utils import clean_text, normalize_doi, utc_now

USER_AGENT = "litflow/0.1 (mailto:research@example.com)"


@dataclass
class SearchResult:
    records: list[RawRecord]
    log: SearchLogEntry


class BaseConnector:
    source = "base"
    base_url = ""

    def search(self, query: str, config: RunConfig, limit: int) -> SearchResult:
        raise NotImplementedError

    def _get_json(self, url: str) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _safe_error(self, exc: BaseException) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            return f"HTTP {exc.code}: {exc.reason}"
        if isinstance(exc, urllib.error.URLError):
            return f"URL error: {exc.reason}"
        return f"{type(exc).__name__}: {exc}"


class OpenAlexConnector(BaseConnector):
    source = "openalex"
    base_url = "https://api.openalex.org/works"

    def search(self, query: str, config: RunConfig, limit: int) -> SearchResult:
        params = {"search": query, "per-page": str(min(limit, 200))}
        filters = []
        if config.year_from:
            filters.append(f"from_publication_date:{config.year_from}-01-01")
        if config.year_to:
            filters.append(f"to_publication_date:{config.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        endpoint = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        try:
            data = self._get_json(endpoint)
            records = [self._parse_work(item) for item in data.get("results", [])]
            return SearchResult(records, SearchLogEntry(self.source, query, utc_now(), endpoint, len(records)))
        except Exception as exc:
            return SearchResult([], SearchLogEntry(self.source, query, utc_now(), endpoint, 0, self._safe_error(exc)))

    def _parse_work(self, item: dict[str, Any]) -> RawRecord:
        authors = []
        for auth in item.get("authorships") or []:
            name = ((auth.get("author") or {}).get("display_name") or "").strip()
            if name:
                authors.append(name)
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        open_access = item.get("open_access") or {}
        pdf = open_access.get("oa_url") or ""
        return RawRecord(
            source=self.source,
            source_id=item.get("id") or "",
            title=clean_text(item.get("title")),
            authors=authors,
            year=item.get("publication_year"),
            venue=clean_text(source.get("display_name") or item.get("host_venue", {}).get("display_name")),
            doi=normalize_doi(item.get("doi")),
            abstract=inverted_index_to_text(item.get("abstract_inverted_index") or {}),
            url=item.get("doi") or item.get("id") or "",
            citation_count=item.get("cited_by_count"),
            source_database=self.source,
            open_access_pdf=pdf if pdf and pdf.lower().endswith(".pdf") else "",
            extra={"type": item.get("type"), "open_access": open_access},
        )


class CrossrefConnector(BaseConnector):
    source = "crossref"
    base_url = "https://api.crossref.org/works"

    def search(self, query: str, config: RunConfig, limit: int) -> SearchResult:
        params = {"query.bibliographic": query, "rows": str(min(limit, 100))}
        filters = []
        if config.year_from:
            filters.append(f"from-pub-date:{config.year_from}-01-01")
        if config.year_to:
            filters.append(f"until-pub-date:{config.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        endpoint = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        try:
            data = self._get_json(endpoint)
            items = (data.get("message") or {}).get("items") or []
            records = [self._parse_work(item) for item in items]
            return SearchResult(records, SearchLogEntry(self.source, query, utc_now(), endpoint, len(records)))
        except Exception as exc:
            return SearchResult([], SearchLogEntry(self.source, query, utc_now(), endpoint, 0, self._safe_error(exc)))

    def _parse_work(self, item: dict[str, Any]) -> RawRecord:
        authors = []
        for author in item.get("author") or []:
            name = " ".join([author.get("given", ""), author.get("family", "")]).strip()
            if name:
                authors.append(name)
        year = None
        date_parts = ((item.get("published-print") or item.get("published-online") or item.get("issued") or {}).get("date-parts") or [])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        title = clean_text((item.get("title") or [""])[0])
        return RawRecord(
            source=self.source,
            source_id=item.get("DOI") or item.get("URL") or title,
            title=title,
            authors=authors,
            year=year,
            venue=clean_text((item.get("container-title") or [""])[0]),
            doi=normalize_doi(item.get("DOI")),
            abstract=clean_text(item.get("abstract")),
            url=item.get("URL") or "",
            citation_count=item.get("is-referenced-by-count"),
            source_database=self.source,
            extra={"type": item.get("type"), "publisher": item.get("publisher")},
        )


class SemanticScholarConnector(BaseConnector):
    source = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    fields = ",".join(
        [
            "paperId",
            "title",
            "authors",
            "year",
            "venue",
            "abstract",
            "url",
            "externalIds",
            "citationCount",
            "referenceCount",
            "openAccessPdf",
            "publicationTypes",
        ]
    )

    def search(self, query: str, config: RunConfig, limit: int) -> SearchResult:
        params = {"query": query, "limit": str(min(limit, 100)), "fields": self.fields}
        endpoint = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        try:
            data = self._get_json(endpoint)
            records = [self._parse_paper(item) for item in data.get("data", [])]
            return SearchResult(records, SearchLogEntry(self.source, query, utc_now(), endpoint, len(records)))
        except Exception as exc:
            return SearchResult([], SearchLogEntry(self.source, query, utc_now(), endpoint, 0, self._safe_error(exc)))

    def expand(self, paper_id: str, relation: str, limit: int = 20) -> SearchResult:
        fields = "paperId,title,authors,year,venue,abstract,url,externalIds,citationCount,openAccessPdf"
        endpoint = f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(paper_id)}?fields={relation}.{fields}"
        try:
            data = self._get_json(endpoint)
            records = []
            for item in data.get(relation, [])[:limit]:
                paper = item.get("citingPaper") or item.get("citedPaper") or item
                if paper:
                    records.append(self._parse_paper(paper))
            return SearchResult(records, SearchLogEntry(self.source, f"{relation}:{paper_id}", utc_now(), endpoint, len(records)))
        except Exception as exc:
            return SearchResult([], SearchLogEntry(self.source, f"{relation}:{paper_id}", utc_now(), endpoint, 0, self._safe_error(exc)))

    def _parse_paper(self, item: dict[str, Any]) -> RawRecord:
        external = item.get("externalIds") or {}
        pdf = item.get("openAccessPdf") or {}
        return RawRecord(
            source=self.source,
            source_id=item.get("paperId") or external.get("DOI") or clean_text(item.get("title")),
            title=clean_text(item.get("title")),
            authors=[clean_text(a.get("name")) for a in item.get("authors") or [] if a.get("name")],
            year=item.get("year"),
            venue=clean_text(item.get("venue")),
            doi=normalize_doi(external.get("DOI")),
            abstract=clean_text(item.get("abstract")),
            url=item.get("url") or "",
            citation_count=item.get("citationCount"),
            source_database=self.source,
            open_access_pdf=pdf.get("url") or "",
            extra={"externalIds": external, "publicationTypes": item.get("publicationTypes")},
        )


class ArxivConnector(BaseConnector):
    source = "arxiv"
    base_url = "https://export.arxiv.org/api/query"

    def search(self, query: str, config: RunConfig, limit: int) -> SearchResult:
        arxiv_query = "all:" + query.replace('"', "")
        params = {"search_query": arxiv_query, "start": "0", "max_results": str(min(limit, 100))}
        endpoint = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(endpoint, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml = resp.read()
            root = ET.fromstring(xml)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            records = [self._parse_entry(entry) for entry in root.findall("atom:entry", ns)]
            time.sleep(0.2)
            return SearchResult(records, SearchLogEntry(self.source, query, utc_now(), endpoint, len(records)))
        except Exception as exc:
            return SearchResult([], SearchLogEntry(self.source, query, utc_now(), endpoint, 0, self._safe_error(exc)))

    def _parse_entry(self, entry: ET.Element) -> RawRecord:
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        title = clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        authors = [clean_text(a.findtext("atom:name", default="", namespaces=ns)) for a in entry.findall("atom:author", ns)]
        published = entry.findtext("atom:published", default="", namespaces=ns)
        year = int(published[:4]) if published[:4].isdigit() else None
        entry_id = entry.findtext("atom:id", default="", namespaces=ns)
        pdf = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf = link.attrib.get("href", "")
                break
        doi = ""
        doi_node = entry.find("arxiv:doi", ns)
        if doi_node is not None:
            doi = normalize_doi(doi_node.text)
        return RawRecord(
            source=self.source,
            source_id=entry_id,
            title=title,
            authors=[a for a in authors if a],
            year=year,
            venue="arXiv",
            doi=doi,
            abstract=clean_text(entry.findtext("atom:summary", default="", namespaces=ns)),
            url=entry_id,
            citation_count=None,
            source_database=self.source,
            open_access_pdf=pdf,
            extra={"published": published},
        )


class WebSearchConnector(BaseConnector):
    source = "web"

    def search(self, query: str, config: RunConfig, limit: int) -> SearchResult:
        endpoint = "manual-web-search:" + urllib.parse.quote_plus(query)
        log = SearchLogEntry(
            source=self.source,
            query=query,
            timestamp=utc_now(),
            endpoint=endpoint,
            result_count=0,
            error="No free general web-search API configured; query written to manual_db_search_pack.md.",
        )
        return SearchResult([], log)


class SerpApiGoogleScholarConnector(BaseConnector):
    source = "serpapi_google_scholar"
    base_url = "https://serpapi.com/search.json"

    def search(self, query: str, config: RunConfig, limit: int) -> SearchResult:
        token = os.environ.get(config.web_search_token_env)
        params = {
            "engine": "google_scholar",
            "q": query,
            "num": str(min(limit, 20)),
        }
        endpoint = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        if not token:
            log = SearchLogEntry(
                self.source,
                query,
                utc_now(),
                endpoint,
                0,
                f"{config.web_search_token_env} is not set; SerpAPI Google Scholar search skipped.",
            )
            return SearchResult([], log)
        params["api_key"] = token
        endpoint_with_key = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        endpoint_for_log = endpoint_with_key.replace(token, "***")
        try:
            data = self._get_json(endpoint_with_key)
            if data.get("error"):
                return SearchResult([], SearchLogEntry(self.source, query, utc_now(), endpoint_for_log, 0, str(data["error"])))
            rows = data.get("organic_results") or []
            records = [self._parse_result(item) for item in rows[:limit]]
            return SearchResult(records, SearchLogEntry(self.source, query, utc_now(), endpoint_for_log, len(records)))
        except Exception as exc:
            return SearchResult([], SearchLogEntry(self.source, query, utc_now(), endpoint_for_log, 0, self._safe_error(exc)))

    def _parse_result(self, item: dict[str, Any]) -> RawRecord:
        publication = item.get("publication_info") or {}
        resources = item.get("resources") or []
        pdf = ""
        for resource in resources:
            link = resource.get("link") or ""
            file_format = str(resource.get("file_format") or "").lower()
            if link and ("pdf" in file_format or link.lower().endswith(".pdf")):
                pdf = link
                break
        summary = clean_text(publication.get("summary"))
        year = parse_year(summary) or parse_year(item.get("snippet"))
        authors = []
        for author in publication.get("authors") or []:
            name = clean_text(author.get("name"))
            if name:
                authors.append(name)
        cited_by = ((item.get("inline_links") or {}).get("cited_by") or {}).get("total")
        return RawRecord(
            source=self.source,
            source_id=item.get("result_id") or item.get("link") or item.get("title") or "",
            title=clean_text(item.get("title")),
            authors=authors or parse_authors_from_scholar_summary(summary),
            year=year,
            venue=summary,
            doi="",
            abstract=clean_text(item.get("snippet")),
            url=item.get("link") or "",
            citation_count=int(cited_by) if isinstance(cited_by, int) or str(cited_by).isdigit() else None,
            source_database=self.source,
            open_access_pdf=pdf,
            extra={"publication_info": publication, "inline_links": item.get("inline_links"), "resources": resources},
        )


def parse_year(text: Any) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", str(text or ""))
    return int(match.group(0)) if match else None


def parse_authors_from_scholar_summary(summary: str) -> list[str]:
    if not summary:
        return []
    before_dash = re.split(r"\s+-\s+", summary, maxsplit=1)[0]
    pieces = [clean_text(piece) for piece in re.split(r",| and ", before_dash) if clean_text(piece)]
    return pieces[:8]


def inverted_index_to_text(index: dict[str, list[int]]) -> str:
    if not index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            words.append((position, word))
    return " ".join(word for _, word in sorted(words))


def default_connectors() -> dict[str, BaseConnector]:
    return {
        "openalex": OpenAlexConnector(),
        "crossref": CrossrefConnector(),
        "semantic_scholar": SemanticScholarConnector(),
        "arxiv": ArxivConnector(),
        "web": WebSearchConnector(),
        "serpapi_google_scholar": SerpApiGoogleScholarConnector(),
    }
