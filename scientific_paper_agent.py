"""
BiodiversIA – Scientific Paper Agent

Flujo (rápido y sin desperdiciar API):
  1. Busca 15 papers en PubMed  →  API de NCBI, gratis, ~5 segundos
  2. UNA sola llamada a Claude  →  genera el paper Q1 completo
"""

import time
import xml.etree.ElementTree as ET
from datetime import datetime

import anthropic
import requests
from anthropic import RateLimitError, APIStatusError

MODEL     = "claude-haiku-4-5-20251001"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# ---------------------------------------------------------------------------
# PubMed  (API pública de NCBI, sin clave, rápida)
# ---------------------------------------------------------------------------

def buscar_pubmed(query: str, n: int = 15) -> list:
    try:
        r = requests.get(f"{NCBI_BASE}/esearch.fcgi", params={
            "db": "pubmed", "term": query, "retmax": n,
            "retmode": "json", "sort": "relevance",
            "tool": "BiodiversIA", "email": "biodiversia@research.org",
        }, timeout=15)
        pmids = r.json()["esearchresult"].get("idlist", [])
        if not pmids:
            return []

        time.sleep(0.4)
        r2 = requests.get(f"{NCBI_BASE}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(pmids),
            "retmode": "xml", "rettype": "abstract",
            "tool": "BiodiversIA", "email": "biodiversia@research.org",
        }, timeout=20)

        papers = []
        for art in ET.fromstring(r2.text).findall(".//PubmedArticle"):
            title = "".join((art.find(".//ArticleTitle") or ET.Element("x")).itertext()).strip()
            ab    = " ".join("".join(el.itertext()) for el in art.findall(".//AbstractText")).strip()[:400]
            auths = [f"{a.findtext('LastName','')} {a.findtext('Initials','')}".strip()
                     for a in art.findall(".//Author") if a.findtext("LastName")][:3]
            year  = art.findtext(".//PubDate/Year") or "?"
            jour  = art.findtext(".//Journal/Title") or "?"
            pmid  = art.findtext(".//PMID") or "?"
            doi   = next((el.text for el in art.findall(".//ArticleId")
                          if el.get("IdType") == "doi"), "")
            if title:
                papers.append({
                    "pmid": pmid, "doi": doi, "title": title,
                    "authors": auths, "year": year, "journal": jour, "abstract": ab,
                })
        return papers
    except Exception as e:
        return []


def _formato(papers: list) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        auths = ", ".join(p.get("authors", []))
        lines.append(
            f"[{i}] {auths} ({p.get('year','?')}). \"{p.get('title','?')}\". "
            f"{p.get('journal','?')}."
            + (f" DOI:{p['doi']}." if p.get("doi") else "")
            + (f" Abstract: {p.get('abstract','')}" if p.get("abstract") else "")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude con retry
# ---------------------------------------------------------------------------

def _llamar_claude(client, system, prompt, max_tokens=4000, log=print):
    wait = 30
    for attempt in range(6):
        try:
            return client.messages.create(
                model=MODEL, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except RateLimitError:
            if attempt == 5:
                raise
            log(f"  [Rate limit] Esperando {wait}s…")
            time.sleep(wait)
            wait = min(wait * 2, 120)
        except APIStatusError as e:
            if e.status_code == 529:
                if attempt == 5:
                    raise
                log(f"  [Sobrecarga] Esperando {wait}s…")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                raise


# ---------------------------------------------------------------------------
# Agente principal
# ---------------------------------------------------------------------------

def run_paper_agent(
    topic: str = "Impact of climate change on coral reef biodiversity and conservation",
    search_literature: bool = True,
    progress_callback=None,
) -> dict:
    log = progress_callback or print
    client = anthropic.Anthropic()

    # Paso 1 — PubMed (sin API de Claude, ~5-10 segundos)
    papers = []
    if search_literature:
        log("🔍 Buscando en PubMed (15 papers más relevantes)…")
        papers = buscar_pubmed(topic, n=15)
        log(f"   ✓ {len(papers)} papers encontrados")
    else:
        log("ℹ️  Sin búsqueda — Claude usará su conocimiento.")

    refs = _formato(papers) if papers else "(Claude: cita papers reales de tu conocimiento.)"

    # Paso 2 — UNA llamada a Claude
    log("✍️  Generando paper Q1 (1 llamada a Claude, ~30 segundos)…")

    system = (
        "Expert marine ecologist and scientific writer. "
        "Formal scientific English. Cite as (Author et al., Year). "
        "Past tense for methods/results; present for established facts."
    )

    prompt = f"""Write a complete Q1 scientific review paper on:

TOPIC: {topic}

PAPERS FROM PUBMED (use these as your primary citations):
{refs}

Use the exact markdown headers below:

# [Title — ≤18 words, specific]

## Abstract
Background / Methods / Results / Conclusions. ≤220 words. 2-3 citations.

## Keywords
6 MeSH-style terms, comma-separated.

## 1. Introduction
500-600 words. Global context → knowledge gap → objectives. ≥8 inline citations.

## 2. Materials and Methods
400-500 words. Systematic review design, PRISMA criteria. Past tense.

## 3. Results
500-600 words. 3 sub-headings. Quantitative findings, inline citations.

## 4. Discussion
550-650 words. Interpretation, mechanisms, limitations, conservation policy. ≥5 citations.

## 5. Conclusions
120-150 words. 3 main findings, 2 future research directions.

## References
APA 7th. Numbered list. Prioritise the PubMed papers above; add others from your knowledge as needed.

Write the complete paper now."""

    resp  = _llamar_claude(client, system, prompt, max_tokens=4000, log=log)
    paper = resp.content[0].text.strip()

    # Parsear secciones
    import re
    patterns = {
        "title":        r"^#\s+(.+?)$",
        "abstract":     r"## Abstract\n(.+?)(?=\n## |\Z)",
        "keywords":     r"## Keywords\n(.+?)(?=\n## |\Z)",
        "introduction": r"## 1\. Introduction\n(.+?)(?=\n## |\Z)",
        "methods":      r"## 2\. Materials and Methods\n(.+?)(?=\n## |\Z)",
        "results":      r"## 3\. Results\n(.+?)(?=\n## |\Z)",
        "discussion":   r"## 4\. Discussion\n(.+?)(?=\n## |\Z)",
        "conclusions":  r"## 5\. Conclusions\n(.+?)(?=\n## |\Z)",
        "references":   r"## References\n(.+?)(?=\n## |\Z)",
    }
    sections = {}
    for key, pat in patterns.items():
        m = re.search(pat, paper, re.DOTALL | re.MULTILINE)
        if m:
            sections[key] = m.group(1).strip()

    header = (
        f"# BiodiversIA – Scientific Paper\n"
        f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}*  \n"
        f"*Tema: {topic}*  \n"
        f"*Papers PubMed usados: {len(papers)}*\n\n---\n\n"
    )

    log("✅ Paper completado.")
    return {"sections": sections, "full_paper": header + paper}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="Ocean warming effects on coral reef biodiversity and conservation")
    p.add_argument("--output", default="scientific_paper_output.md")
    p.add_argument("--no-search", action="store_true")
    args = p.parse_args()
    result = run_paper_agent(topic=args.topic, search_literature=not args.no_search)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result["full_paper"])
    print(f"Paper guardado → {args.output}")
