"""
BiodiversIA – Scientific Paper Agent
Drafts Q1-quality scientific papers on marine biodiversity, ecology,
and conservation.

Architecture: each paper section is written in its OWN independent API call
so the input token count never accumulates across sections.
The literature search phase is kept lightweight and optional.
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

NCBI_BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_EMAIL  = "biodiversia@research.org"
MODEL       = "claude-sonnet-4-6"

# Keep each request small to stay under 30k tokens/min on free tier
MAX_TOKENS_OUTPUT = 2000
# Pause between consecutive Claude calls (seconds)
INTER_CALL_PAUSE  = 15

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _claude_call(client: anthropic.Anthropic, **kwargs) -> anthropic.types.Message:
    """Call Claude with exponential backoff on 429 / 529 errors."""
    wait = 30
    for attempt in range(6):
        try:
            return client.messages.create(**kwargs)
        except RateLimitError:
            if attempt == 5:
                raise
            print(f"  [Rate limit] Waiting {wait}s (attempt {attempt+1}/5)…")
            time.sleep(wait)
            wait = min(wait * 2, 120)
        except APIStatusError as e:
            if e.status_code == 529:
                if attempt == 5:
                    raise
                print(f"  [Overloaded] Waiting {wait}s…")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                raise

# ---------------------------------------------------------------------------
# PubMed helpers (NCBI E-utilities – no API key needed)
# ---------------------------------------------------------------------------

def _ncbi_get(endpoint: str, params: dict) -> requests.Response:
    params.setdefault("tool", "BiodiversIA")
    params.setdefault("email", NCBI_EMAIL)
    time.sleep(0.4)
    return requests.get(f"{NCBI_BASE}/{endpoint}", params=params, timeout=20)


def search_pubmed(query: str, max_results: int = 6) -> dict:
    r = _ncbi_get("esearch.fcgi", {
        "db": "pubmed", "term": query,
        "retmax": max_results, "retmode": "json", "sort": "relevance",
    })
    r.raise_for_status()
    data = r.json()["esearchresult"]
    return {
        "count": int(data.get("count", 0)),
        "pmids": data.get("idlist", []),
    }


def fetch_pubmed_abstracts(pmids: list) -> list:
    if not pmids:
        return []
    ids = ",".join(pmids[:5])
    r = _ncbi_get("efetch.fcgi", {
        "db": "pubmed", "id": ids, "retmode": "xml", "rettype": "abstract",
    })
    r.raise_for_status()
    root = ET.fromstring(r.text)
    articles = []
    for art in root.findall(".//PubmedArticle"):
        title_el = art.find(".//ArticleTitle")
        title    = "".join(title_el.itertext()) if title_el is not None else "N/A"
        abstract = " ".join(
            "".join(el.itertext()) for el in art.findall(".//AbstractText")
        ).strip()[:500] or "N/A"
        authors  = [
            f"{a.findtext('LastName','')} {a.findtext('Initials','')}".strip()
            for a in art.findall(".//Author") if a.findtext("LastName")
        ][:3]
        journal  = art.findtext(".//Journal/Title") or "N/A"
        year     = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate","")[:4] or "N/A"
        pmid     = art.findtext(".//PMID") or "N/A"
        doi      = next(
            (el.text for el in art.findall(".//ArticleId") if el.get("IdType") == "doi"),
            "N/A"
        )
        articles.append({
            "pmid": pmid, "title": title, "abstract": abstract,
            "authors": authors, "journal": journal, "year": year, "doi": doi,
        })
    return articles


def search_google_scholar(query: str, max_results: int = 4) -> list:
    try:
        from scholarly import scholarly as sc
        results = []
        for i, pub in enumerate(sc.search_pubs(query)):
            if i >= max_results:
                break
            bib = pub.get("bib", {})
            results.append({
                "title":    bib.get("title", "N/A"),
                "authors":  bib.get("author", [])[:3],
                "year":     bib.get("pub_year", "N/A"),
                "abstract": bib.get("abstract", "N/A")[:500],
                "journal":  bib.get("venue", "N/A"),
                "cited_by": pub.get("num_citations", 0),
            })
            time.sleep(0.5)
        return results
    except ImportError:
        return [{"error": "scholarly not installed"}]
    except Exception as e:
        return [{"error": str(e)}]

# ---------------------------------------------------------------------------
# Literature search phase (lightweight, optional)
# ---------------------------------------------------------------------------

def collect_references(topic: str, log=print) -> list:
    """
    Run a compact literature search on PubMed (+ Scholar if available).
    Returns a compact reference list (dicts) to pass to the drafting phase.
    Keeps total retrieved text well under 5k tokens.
    """
    refs = []

    queries = [
        f"{topic} marine biodiversity conservation",
        f"{topic} climate change marine ecosystem",
    ]

    for q in queries:
        log(f"  PubMed: {q[:70]}…")
        try:
            res   = search_pubmed(q, max_results=5)
            pmids = res.get("pmids", [])[:4]
            arts  = fetch_pubmed_abstracts(pmids)
            refs.extend(arts)
            log(f"    → {len(arts)} abstracts fetched")
        except Exception as e:
            log(f"    PubMed error: {e}")

    log("  Google Scholar search…")
    scholar = search_google_scholar(f"{topic} marine conservation", max_results=4)
    if not any("error" in r for r in scholar):
        refs.extend(scholar)
        log(f"    → {len(scholar)} Scholar results")

    # Deduplicate by title
    seen, unique = set(), []
    for r in refs:
        t = r.get("title", "")
        if t and t not in seen:
            seen.add(t)
            unique.append(r)

    return unique[:12]  # hard cap to limit tokens


def _format_refs_for_prompt(refs: list) -> str:
    """Format reference list as a compact string for injection into prompts."""
    if not refs:
        return "(No references retrieved – use your training knowledge.)"
    lines = []
    for i, r in enumerate(refs, 1):
        authors = ", ".join(r.get("authors", [])) or "Unknown"
        lines.append(
            f"[{i}] {authors} ({r.get('year','?')}). {r.get('title','?')}. "
            f"{r.get('journal','?')}. PMID:{r.get('pmid','N/A')}. "
            f"Abstract: {r.get('abstract','N/A')[:300]}"
        )
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Section-by-section drafting  (each section = one independent API call)
# ---------------------------------------------------------------------------

_ROLE = (
    "You are a senior marine ecologist and science writer. "
    "Write in formal scientific English. "
    "Cite papers as (Author et al., Year). "
    "Use past tense for methods/results; present for established facts."
)

def _draft_section(
    client: anthropic.Anthropic,
    section: str,
    topic: str,
    refs_text: str,
    context: dict,
    log=print,
) -> str:
    """Draft a single section in one API call with no accumulated history."""

    # Build a tight, section-specific prompt
    if section == "title":
        prompt = (
            f"Write ONLY the title of a Q1 scientific paper on:\n{topic}\n\n"
            "Requirements: ≤18 words, specific, includes habitat type, main variable, "
            "and geographic scope. Output the title only, no quotes."
        )

    elif section == "abstract":
        prompt = (
            f"Write a structured abstract (≤250 words) for a Q1 paper titled:\n"
            f"\"{context.get('title','')}\"\n\n"
            f"Topic: {topic}\n\n"
            "Use four labelled paragraphs: Background | Methods | Results | Conclusions.\n"
            "Include quantitative data where possible. Cite 2-3 papers from the list.\n\n"
            f"Available references:\n{refs_text}"
        )

    elif section == "keywords":
        prompt = (
            f"List exactly 6 keywords for a paper titled:\n\"{context.get('title','')}\"\n"
            "Output as a comma-separated line. Use MeSH/EMTREE terms where possible."
        )

    elif section == "introduction":
        prompt = (
            f"Write the Introduction (550-750 words) for a Q1 paper titled:\n"
            f"\"{context.get('title','')}\"\n\n"
            "Structure: broad global context → specific knowledge gap → objectives.\n"
            "Cite ≥8 papers from the reference list below (or from your own knowledge "
            "if the list is short). End with a clear statement of the paper's objectives.\n\n"
            f"Available references:\n{refs_text}"
        )

    elif section == "methods":
        prompt = (
            f"Write the Materials and Methods section (450-600 words) for:\n"
            f"\"{context.get('title','')}\"\n\n"
            "This is a systematic review / meta-analysis. Cover: search strategy, "
            "databases (PubMed, Web of Science, Google Scholar), inclusion/exclusion criteria, "
            "data extraction, quality assessment (PRISMA-inspired), and statistical synthesis.\n"
            "Write in past tense."
        )

    elif section == "results":
        prompt = (
            f"Write the Results section (550-750 words) for:\n"
            f"\"{context.get('title','')}\"\n\n"
            "Organise findings under 3-4 sub-headings. Extract and report quantitative "
            "data (percentages, effect sizes, species counts, temperature anomalies, etc.) "
            "from the references below. Cite papers inline.\n\n"
            f"Available references:\n{refs_text}"
        )

    elif section == "discussion":
        prompt = (
            f"Write the Discussion section (650-850 words) for:\n"
            f"\"{context.get('title','')}\"\n\n"
            "Cover: interpretation of results, comparison with prior studies, ecological "
            "mechanisms, study limitations, conservation policy implications, and "
            "management recommendations. Cite ≥6 papers.\n\n"
            f"Available references:\n{refs_text}"
        )

    elif section == "conclusions":
        prompt = (
            f"Write the Conclusions section (130-180 words) for:\n"
            f"\"{context.get('title','')}\"\n\n"
            "Summarise the 3 main findings, state their conservation significance, "
            "and suggest 2-3 future research directions. No new citations."
        )

    elif section == "references":
        prompt = (
            f"Format a complete References list in APA 7th edition for the paper:\n"
            f"\"{context.get('title','')}\"\n\n"
            "Include ALL papers cited in the sections above. Use the details below "
            "for the retrieved papers; fill in full citations from your training knowledge "
            "for any additional ones mentioned in the text.\n\n"
            f"Retrieved papers:\n{refs_text}\n\n"
            "Output only the formatted reference list, numbered, one per line."
        )

    else:
        return f"(Section '{section}' not implemented)"

    log(f"  Drafting: {section}…")
    time.sleep(INTER_CALL_PAUSE)

    response = _claude_call(
        client,
        model=MODEL,
        max_tokens=MAX_TOKENS_OUTPUT,
        system=_ROLE,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

SECTIONS = [
    "title", "abstract", "keywords", "introduction",
    "methods", "results", "discussion", "conclusions", "references",
]

def run_paper_agent(
    topic: str = (
        "Impact of climate change on marine biodiversity in coastal ecosystems: "
        "conservation strategies and ecosystem resilience"
    ),
    search_literature: bool = True,
    progress_callback=None,
) -> dict:
    """
    Draft a complete Q1 scientific paper section by section.

    Each section is a separate, independent API call – the conversation history
    never accumulates, so input tokens stay small on every request.

    Args:
        topic: Research topic for the paper.
        search_literature: If True, search PubMed/Scholar first (uses ~4 extra
            API calls). If False, Claude uses its training knowledge directly.
        progress_callback: Optional callable(str) for live progress messages.

    Returns:
        dict with 'sections' (dict) and 'full_paper' (markdown str).
    """
    log = progress_callback or print
    client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Phase 1: Literature search (optional)
    # ------------------------------------------------------------------
    refs_text = ""
    if search_literature:
        log("[1/2] Searching literature (PubMed + Google Scholar)…")
        refs = collect_references(topic, log=log)
        refs_text = _format_refs_for_prompt(refs)
        log(f"  Literature collected: {len(refs_text.splitlines())} reference entries")
    else:
        log("[1/2] Skipping search – using Claude training knowledge.")
        refs_text = "(Use your training knowledge to cite real, published papers.)"

    # ------------------------------------------------------------------
    # Phase 2: Draft each section independently
    # ------------------------------------------------------------------
    log("[2/2] Drafting paper sections…")
    sections: dict = {}
    context: dict = {}

    for sec in SECTIONS:
        text = _draft_section(client, sec, topic, refs_text, context, log=log)
        sections[sec] = text
        context[sec] = text          # make earlier sections available to later ones
        log(f"  ✓ {sec} ({len(text.split())} words)")

    # ------------------------------------------------------------------
    # Assemble full paper
    # ------------------------------------------------------------------
    headers = {
        "title":        "",
        "abstract":     "## Abstract",
        "keywords":     "## Keywords",
        "introduction": "## 1. Introduction",
        "methods":      "## 2. Materials and Methods",
        "results":      "## 3. Results",
        "discussion":   "## 4. Discussion",
        "conclusions":  "## 5. Conclusions",
        "references":   "## References",
    }

    parts = [
        f"# BiodiversIA – Scientific Paper\n"
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        f"*Topic: {topic}*\n\n---"
    ]
    for sec in SECTIONS:
        if sec not in sections:
            continue
        h = headers[sec]
        parts.append(f"\n{h}\n\n{sections[sec]}" if h else f"\n# {sections[sec]}")

    return {"sections": sections, "full_paper": "\n".join(parts)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BiodiversIA Scientific Paper Agent")
    parser.add_argument("--topic",  default=(
        "Impact of ocean warming and acidification on coral reef biodiversity: "
        "global patterns, tipping points, and marine conservation strategies"
    ))
    parser.add_argument("--output", default="scientific_paper_output.md")
    parser.add_argument("--no-search", action="store_true",
                        help="Skip literature search; use Claude knowledge only")
    args = parser.parse_args()

    result = run_paper_agent(
        topic=args.topic,
        search_literature=not args.no_search,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result["full_paper"])
    print(f"\nPaper saved → {args.output}")
    print(f"Sections: {list(result['sections'].keys())}")
