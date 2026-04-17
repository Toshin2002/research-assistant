import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient

from app.config import settings

_tavily = TavilyClient(api_key=settings.tavily_api_key)


# ── Tool implementations ───────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web. Returns a list of {title, url, content} dicts."""
    response = _tavily.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=False,
    )
    return [
        {
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]


def fetch_page(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return clean text, stripped of HTML."""
    headers = {"User-Agent": "ResearchAgent/1.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"[fetch_page error: {exc}]"

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text  = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    clean = "\n".join(lines)

    return clean[:max_chars] if len(clean) > max_chars else clean


# ── Registry ───────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    "web_search": web_search,
    "fetch_page": fetch_page,
}


# ── Schemas (Groq / OpenAI format) ────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information on a topic. "
                "Use this to find relevant sources, facts, and recent data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch and read the full text of a specific web page by URL. "
                "Use this after web_search to get more detail from a source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the page to fetch",
                    },
                },
                "required": ["url"],
            },
        },
    },
]
