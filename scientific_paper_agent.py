"""
BiodiversIA – Scientific Paper Agent
Drafts Q1-quality scientific papers on marine biodiversity, ecology,
and conservation by searching PubMed and Google Scholar.
"""

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import anthropic
import requests
from anthropic import RateLimitError, APIStatusError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_EMAIL = "biodiversia@research.org"   # required by NCBI for polite usage
MODEL = "claude-sonnet-4-6"

# Token budget per call – keeps each request well under the 30k/min limit
MAX_TOKENS_PER_CALL = 3000
# Minimum seconds to wait between consecutive Claude API calls
MIN_CALL_INTERVAL = 10

# ---------------------------------------------------------------------------
# PubMed helpers (NCBI E-utilities – no API key needed)
# ---------------------------------------------------------------------------

def _ncbi_get(endpoint: str, params: dict) -> requests.Response:
    """Rate-limited NCBI request (max ~3 req/s without API key)."""
    params.setdefault("tool", "BiodiversIA")
    params.setdefault("email", NCBI_EMAIL)
    time.sleep(0.35)
    return requests.get(f"{NCBI_BASE}/{endpoint}", params=params, timeout=20)


def search_pubmed(query: str, max_results: int = 8) -> dict:
    """
    Search PubMed and return a list of PMIDs with basic metadata.

    Args:
        query: Free-text search query (supports MeSH terms and boolean operators).
        max_results: Maximum number of results to retrieve.

    Returns:
        dict with keys 'count', 'pmids', 'query_translation'.
    """
    r = _ncbi_get("esearch.fcgi", {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
        "usehistory": "y",
    })
    r.raise_for_status()
    data = r.json()["esearchresult"]
    return {
        "count": int(data.get("count", 0)),
        "pmids": data.get("idlist", []),
        "query_translation": data.get("querytranslation", ""),
    }


def fetch_pubmed_abstracts(pmids: list[str]) -> list[dict]:
    """
    Fetch titles, abstracts, authors, journal and year for a list of PMIDs.

    Args:
        pmids: List of PubMed IDs (strings).

    Returns:
        List of dicts with keys: pmid, title, abstract, authors, journal, year, doi.
    """
    if not pmids:
        return []

    ids = ",".join(pmids[:6])  # cap at 6 to keep tool-result tokens low
    r = _ncbi_get("efetch.fcgi", {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml",
        "rettype": "abstract",
    })
    r.raise_for_status()

    root = ET.fromstring(r.text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        # Title
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else "N/A"

        # Abstract
        abstract_texts = article.findall(".//AbstractText")
        abstract = " ".join(
            ("".join(el.itertext())) for el in abstract_texts
        ).strip() or "Abstract not available."

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "")
            init = author.findtext("Initials", "")
            if last:
                authors.append(f"{last} {init}".strip())

        # Journal & year
        journal = article.findtext(".//Journal/Title") or article.findtext(".//ISOAbbreviation") or "N/A"
        year = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//PubDate/MedlineDate", "")[:4]
            or "N/A"
        )

        # PMID
        pmid = article.findtext(".//PMID") or "N/A"

        # DOI
        doi = "N/A"
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text or "N/A"
                break

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract[:700],    # hard cap to limit token usage
            "authors": authors[:4],
            "journal": journal,
            "year": year,
            "doi": doi,
        })

    return articles


# ---------------------------------------------------------------------------
# Google Scholar helper (via scholarly – no API key needed)
# ---------------------------------------------------------------------------

def search_google_scholar(query: str, max_results: int = 5) -> list[dict]:
    """
    Search Google Scholar using the scholarly library.

    Args:
        query: Search query string.
        max_results: Maximum number of results.

    Returns:
        List of dicts with keys: title, authors, year, abstract, cited_by, url.
    """
    try:
        from scholarly import scholarly as sc

        results = []
        search_gen = sc.search_pubs(query)
        for i, pub in enumerate(search_gen):
            if i >= max_results:
                break
            bib = pub.get("bib", {})
            results.append({
                "title": bib.get("title", "N/A"),
                "authors": bib.get("author", [])[:4],
                "year": bib.get("pub_year", "N/A"),
                "abstract": bib.get("abstract", "Abstract not available.")[:600],
                "cited_by": pub.get("num_citations", 0),
                "url": pub.get("pub_url", "N/A"),
                "journal": bib.get("venue", bib.get("journal", "N/A")),
            })
            time.sleep(0.5)
        return results
    except ImportError:
        return [{"error": "scholarly library not installed. Run: pip install scholarly"}]
    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_pubmed",
        "description": (
            "Search PubMed (MEDLINE) for peer-reviewed scientific articles. "
            "Supports boolean operators (AND, OR, NOT) and MeSH field tags like "
            "[MeSH], [tiab], [au]. Returns article count and PMIDs. "
            "Use specific queries combining species names, habitats, and topics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "PubMed search query. Example: "
                        "'marine biodiversity[MeSH] AND coral reef conservation AND climate change'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 8, max 15).",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_pubmed_abstracts",
        "description": (
            "Fetch full titles, abstracts, authors, journal names and DOIs "
            "for a list of PubMed IDs (PMIDs). Call this after search_pubmed "
            "to read the actual content of the most relevant papers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pmids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of PubMed IDs returned by search_pubmed (max 10 per call).",
                },
            },
            "required": ["pmids"],
        },
    },
    {
        "name": "search_google_scholar",
        "description": (
            "Search Google Scholar for academic papers, including grey literature "
            "and interdisciplinary work. Complements PubMed searches with broader "
            "coverage. Returns titles, abstracts, citation counts and URLs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string for Google Scholar.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5, max 8).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_paper_section",
        "description": (
            "Store a completed draft section of the scientific paper. "
            "Call this once you have written each section to accumulate the full paper. "
            "Sections: title, abstract, keywords, introduction, methods, results, "
            "discussion, conclusions, references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": [
                        "title",
                        "abstract",
                        "keywords",
                        "introduction",
                        "methods",
                        "results",
                        "discussion",
                        "conclusions",
                        "references",
                    ],
                    "description": "Name of the paper section.",
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown text of the section.",
                },
            },
            "required": ["section", "content"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, inputs: dict, paper_sections: dict) -> str:
    """Execute a tool and return its result as a JSON string."""
    if name == "search_pubmed":
        result = search_pubmed(inputs["query"], min(inputs.get("max_results", 8), 15))
        return json.dumps(result, ensure_ascii=False)

    elif name == "fetch_pubmed_abstracts":
        result = fetch_pubmed_abstracts(inputs["pmids"])
        return json.dumps(result, ensure_ascii=False)

    elif name == "search_google_scholar":
        result = search_google_scholar(inputs["query"], min(inputs.get("max_results", 5), 8))
        return json.dumps(result, ensure_ascii=False)

    elif name == "write_paper_section":
        section = inputs["section"]
        content = inputs["content"]
        paper_sections[section] = content
        return json.dumps({"status": "saved", "section": section, "chars": len(content)})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior marine ecologist and scientific writer. Draft a complete Q1 journal paper \
(suitable for *Global Change Biology*, *Biological Conservation*, or *Frontiers in Marine Science*).

WORKFLOW – follow exactly in this order:
1. search_pubmed (2-3 targeted queries, max_results=8 each)
2. fetch_pubmed_abstracts (top 5-6 PMIDs per search)
3. search_google_scholar (1-2 queries, max_results=5)
4. write_paper_section for each section IN ORDER:
   title → abstract → keywords → introduction → methods → results → discussion → conclusions → references

PAPER SECTIONS:
- title: specific, ≤18 words
- abstract: ≤250 words, structured (Background / Methods / Results / Conclusions)
- keywords: 6 terms
- introduction: 500-700 words, cite ≥8 real papers found above, funnel structure
- methods: 400-600 words, systematic review methodology, inclusion criteria
- results: 500-700 words, synthesized findings with quantitative data from papers
- discussion: 600-800 words, interpretation, limitations, conservation policy
- conclusions: 120-180 words, main findings + future directions
- references: full APA list of ALL cited papers (only papers found via search tools)

STYLE: formal scientific English, inline citations as (Author et al., Year),
past tense for methods/results, present for established facts.\
"""


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_paper_agent(
    topic: str = "marine biodiversity loss and conservation strategies in the context of climate change",
    progress_callback=None,
) -> dict:
    """
    Run the agentic loop to research and draft a scientific paper.

    Args:
        topic: Research topic or question to focus the paper on.
        progress_callback: Optional callable(message: str) for real-time updates.

    Returns:
        dict with key 'sections' (the paper sections dict) and 'full_paper' (markdown string).
    """
    client = anthropic.Anthropic()
    paper_sections: dict[str, str] = {}
    messages = [
        {
            "role": "user",
            "content": (
                f"Please research and draft a complete Q1 scientific paper on the following topic:\n\n"
                f"**{topic}**\n\n"
                "Start with an extensive literature search on PubMed and Google Scholar, "
                "then write each section of the paper using the write_paper_section tool. "
                "Make sure to cite the real papers you find."
            ),
        }
    ]

    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    def _call_claude(msgs: list) -> anthropic.types.Message:
        """Call Claude with exponential backoff on 429 rate-limit errors."""
        wait = 20  # initial wait in seconds
        for attempt in range(6):
            try:
                return client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS_PER_CALL,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=msgs,
                )
            except RateLimitError:
                if attempt == 5:
                    raise
                _log(f"  [Rate limit] Waiting {wait}s before retry {attempt + 1}/5...")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            except APIStatusError as e:
                if e.status_code == 529:  # overloaded
                    if attempt == 5:
                        raise
                    _log(f"  [API overloaded] Waiting {wait}s...")
                    time.sleep(wait)
                    wait = min(wait * 2, 120)
                else:
                    raise

    _last_call_time = [0.0]  # mutable reference to allow mutation in closure

    def _rate_limited_call(msgs: list) -> anthropic.types.Message:
        elapsed = time.time() - _last_call_time[0]
        if elapsed < MIN_CALL_INTERVAL:
            gap = MIN_CALL_INTERVAL - elapsed
            _log(f"  [Pacing] Waiting {gap:.1f}s to respect rate limit...")
            time.sleep(gap)
        result = _call_claude(msgs)
        _last_call_time[0] = time.time()
        return result

    iteration = 0
    max_iterations = 40   # safety cap

    while iteration < max_iterations:
        iteration += 1
        _log(f"[Iteration {iteration}] Calling Claude...")

        response = _rate_limited_call(messages)

        # Append assistant message
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            _log("Agent finished.")
            break

        if response.stop_reason != "tool_use":
            _log(f"Unexpected stop_reason: {response.stop_reason}")
            break

        # Process all tool calls in this response
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_inputs = block.input
            _log(f"  -> Tool: {tool_name}({json.dumps({k: v for k, v in tool_inputs.items() if k != 'content'})[:120]})")

            result_str = dispatch_tool(tool_name, tool_inputs, paper_sections)
            _log(f"     Result preview: {result_str[:200]}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    # ---------------------------------------------------------------------------
    # Assemble full paper
    # ---------------------------------------------------------------------------
    section_order = [
        "title", "abstract", "keywords", "introduction",
        "methods", "results", "discussion", "conclusions", "references",
    ]
    section_headers = {
        "title": "",  # title has no header prefix
        "abstract": "## Abstract",
        "keywords": "## Keywords",
        "introduction": "## 1. Introduction",
        "methods": "## 2. Materials and Methods",
        "results": "## 3. Results",
        "discussion": "## 4. Discussion",
        "conclusions": "## 5. Conclusions",
        "references": "## References",
    }

    parts = [
        f"# BiodiversIA – Scientific Paper Agent\n"
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC*\n"
        f"*Topic: {topic}*\n\n---\n"
    ]

    for sec in section_order:
        if sec in paper_sections:
            header = section_headers[sec]
            content = paper_sections[sec]
            if header:
                parts.append(f"\n{header}\n\n{content}\n")
            else:
                parts.append(f"\n{content}\n")

    full_paper = "\n".join(parts)

    return {"sections": paper_sections, "full_paper": full_paper}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BiodiversIA Scientific Paper Agent")
    parser.add_argument(
        "--topic",
        default=(
            "Impact of climate change on marine biodiversity in coastal ecosystems: "
            "a systematic review of conservation strategies and ecosystem resilience"
        ),
        help="Research topic for the paper",
    )
    parser.add_argument(
        "--output",
        default="scientific_paper_output.md",
        help="Output markdown file path",
    )
    args = parser.parse_args()

    print(f"Starting paper agent for topic:\n  {args.topic}\n")
    result = run_paper_agent(topic=args.topic)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result["full_paper"])

    print(f"\nPaper saved to: {args.output}")
    print(f"Sections written: {list(result['sections'].keys())}")
