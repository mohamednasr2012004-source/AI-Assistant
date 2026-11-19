import re
from urllib.parse import urlparse

from langchain_core.documents import Document
from tavily import TavilyClient

from config.config import TAVILY_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

PROPERTY_DOMAINS = ["wasalt.sa", "darglobal.co.uk", "darglobal.com"]
MAX_RESULTS = 5

BRAND_PATTERN = re.compile(r"wasalt(?:\.sa)?|darglobal(?:\.co\.uk|\.com)?", re.IGNORECASE)


def neutralize_brand(text: str) -> str:
    return BRAND_PATTERN.sub("the property website", text or "")


class TavilyFallback:
    def __init__(self):
        self.client = (
            TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
        )

    def _search(self, query: str, include_domains: list[str] | None) -> dict:
        return self.client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            include_domains=include_domains,
            max_results=MAX_RESULTS,
        )

    def _to_documents(self, results: list[dict]) -> list[Document]:
        documents = []
        for index, result in enumerate(results, start=1):
            url = result.get("url", "")
            host = urlparse(url).netloc.replace("www.", "")
            title = neutralize_brand(result.get("title", "Web result"))
            content = neutralize_brand(
                result.get("content", "") or result.get("title", "")
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "id": url or f"tavily-{index}",
                        "title": title,
                        "source": host,
                        "url": url,
                        "city": "",
                    },
                )
            )
        return documents

    def _format_answer(self, query: str, payload: dict, documents: list[Document]) -> str:
        answer = neutralize_brand((payload.get("answer") or "").strip())
        header = (
            "The AI model services are unavailable right now, so I used a live web "
            f"search instead. Results for: {query}"
        )

        if answer:
            body = answer
        elif documents:
            parts = []
            for document in documents:
                snippet = document.page_content.strip().replace("\n", " ")
                parts.append(
                    f"- **{document.metadata['title']}** — "
                    f"{document.metadata['url']}\n  {snippet[:400]}"
                )
            body = "\n".join(parts)
        else:
            body = "The web search did not return any results. Please try again later."

        return f"{header}\n\n{body}"

    def build_result(self, query: str) -> dict:
        if self.client is None:
            logger.warning("Tavily fallback requested but TAVILY_API_KEY is not set")
            return {
                "answer": (
                    "The AI model services are unavailable and web search is not "
                    "configured. Please try again later."
                ),
                "sources": [],
                "filters": {},
                "fallback": "none",
            }

        try:
            payload = self._search(query, PROPERTY_DOMAINS)
            results = payload.get("results") or []
            if not results:
                payload = self._search(query, None)
                results = payload.get("results") or []

            documents = self._to_documents(results)
            answer = self._format_answer(query, payload, documents)
            logger.info("Tavily fallback returned %s results", len(documents))

            return {
                "answer": answer,
                "sources": documents,
                "filters": {},
                "fallback": "tavily",
            }
        except Exception as error:
            logger.error("Tavily fallback failed: %s", error)
            return {
                "answer": (
                    "The AI model services are unavailable and the web search failed "
                    "as well. Please try again later."
                ),
                "sources": [],
                "filters": {},
                "fallback": "none",
            }
