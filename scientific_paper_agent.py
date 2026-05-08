"""
BiodiversIA – Scientific Paper Agent

Flujo:
  1. Busca 15 papers en PubMed  (sin API de Claude)
  2. Busca 10 papers en Google Scholar  (sin API de Claude)
  3. UNA sola llamada a Claude con esos papers → paper Q1 completo
"""

import time
import xml.etree.ElementTree as ET
from datetime import datetime

import anthropic
import requests
from anthropic import RateLimitError, APIStatusError

MODEL = "claude-haiku-4-5-20251001"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

def buscar_pubmed(query: str, n: int = 15) -> list:
    """Devuelve lista de dicts con los n papers más relevantes de PubMed."""
    try:
        r = requests.get(f"{NCBI_BASE}/esearch.fcgi", params={
            "db": "pubmed", "term": query, "retmax": n,
            "retmode": "json", "sort": "relevance",
            "tool": "BiodiversIA", "email": "biodiversia@research.org",
        }, timeout=15)
        pmids = r.json()["esearchresult"].get("idlist", [])
        if not pmids:
            return []

        time.sleep(0.5)
        r2 = requests.get(f"{NCBI_BASE}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(pmids),
            "retmode": "xml", "rettype": "abstract",
            "tool": "BiodiversIA", "email": "biodiversia@research.org",
        }, timeout=20)

        root = ET.fromstring(r2.text)
        papers = []
        for art in root.findall(".//PubmedArticle"):
            title = "".join((art.find(".//ArticleTitle") or ET.Element("x")).itertext()).strip()
            ab    = " ".join("".join(el.itertext()) for el in art.findall(".//AbstractText")).strip()[:400]
            auths = [f"{a.findtext('LastName','')} {a.findtext('Initials','')}".strip()
                     for a in art.findall(".//Author") if a.findtext("LastName")][:3]
            year  = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate","")[:4] or "?"
            jour  = art.findtext(".//Journal/Title") or "?"
            pmid  = art.findtext(".//PMID") or "?"
            doi   = next((el.text for el in art.findall(".//ArticleId")
                          if el.get("IdType") == "doi"), "")
            papers.append({
                "source": "PubMed", "pmid": pmid, "doi": doi,
                "title": title, "authors": auths, "year": year,
                "journal": jour, "abstract": ab,
            })
        return papers
    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# Google Scholar
# ---------------------------------------------------------------------------

def buscar_scholar(query: str, n: int = 10) -> list:
    """Devuelve lista de dicts con los n papers más relevantes de Google Scholar."""
    try:
        from scholarly import scholarly as sc
        papers = []
        for i, pub in enumerate(sc.search_pubs(query)):
            if i >= n:
                break
            bib = pub.get("bib", {})
            papers.append({
                "source": "Scholar",
                "title":    bib.get("title", "?"),
                "authors":  bib.get("author", [])[:3],
                "year":     bib.get("pub_year", "?"),
                "journal":  bib.get("venue", bib.get("journal", "?")),
                "abstract": bib.get("abstract", "")[:400],
                "cited_by": pub.get("num_citations", 0),
                "url":      pub.get("pub_url", ""),
            })
            time.sleep(0.4)
        return papers
    except ImportError:
        return [{"error": "scholarly no instalado: pip install scholarly"}]
    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# Formatear papers para el prompt
# ---------------------------------------------------------------------------

def _formato_papers(papers: list) -> str:
    lineas = []
    i = 1
    for p in papers:
        if "error" in p:
            continue
        auths = ", ".join(p.get("authors", [])) if isinstance(p.get("authors"), list) else str(p.get("authors",""))
        ab    = p.get("abstract","")
        cita  = f"{auths} ({p.get('year','?')})"
        lineas.append(
            f"[{i}] {cita}. \"{p.get('title','?')}\". "
            f"{p.get('journal','?')}. "
            + (f"DOI:{p['doi']}. " if p.get("doi") else "")
            + (f"Abstract: {ab}" if ab else "")
        )
        i += 1
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Llamada a Claude con retry
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
            log(f"  [Rate limit] Esperando {wait}s (intento {attempt+1}/5)…")
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
    """
    1. Busca en PubMed (15 papers) y Google Scholar (10 papers) — sin API Claude.
    2. Genera el paper completo en UNA sola llamada a Claude.
    """
    log = progress_callback or print
    client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Fase 1: Búsqueda (cero tokens de Claude)
    # ------------------------------------------------------------------
    pubmed_papers  = []
    scholar_papers = []

    if search_literature:
        log("🔍 Buscando en PubMed (15 papers más relevantes)…")
        pubmed_papers = buscar_pubmed(topic, n=15)
        ok_pm = [p for p in pubmed_papers if "error" not in p]
        log(f"   ✓ {len(ok_pm)} papers encontrados en PubMed")

        log("🔍 Buscando en Google Scholar (10 papers más relevantes)…")
        scholar_papers = buscar_scholar(topic, n=10)
        ok_sc = [p for p in scholar_papers if "error" not in p]
        log(f"   ✓ {len(ok_sc)} papers encontrados en Google Scholar")
    else:
        log("ℹ️ Búsqueda omitida – Claude usará su conocimiento de entrenamiento.")

    todos = [p for p in pubmed_papers + scholar_papers if "error" not in p]
    refs_block = _formato_papers(todos)

    if refs_block:
        refs_section = f"\n\nPAPERS ENCONTRADOS (usa estas citas en el paper):\n{refs_block}"
    else:
        refs_section = "\n\n(Usa tu conocimiento de entrenamiento para citar papers reales publicados.)"

    # ------------------------------------------------------------------
    # Fase 2: UNA sola llamada a Claude
    # ------------------------------------------------------------------
    log("✍️ Generando paper Q1 completo (1 llamada a Claude)…")

    system = (
        "You are an expert marine ecologist and scientific writer. "
        "Write in formal scientific English. "
        "Cite inline as (Author et al., Year). "
        "Past tense for methods/results; present for established facts."
    )

    prompt = f"""Write a complete Q1 scientific review paper on:

TOPIC: {topic}
{refs_section}

Write ALL sections with these exact markdown headers:

# [Title here]

## Abstract
(≤220 words. Paragraphs: Background / Methods / Results / Conclusions. Quantitative data. 2-3 citations.)

## Keywords
(6 terms, comma-separated)

## 1. Introduction
(500-600 words. Global context → knowledge gap → objectives. ≥8 inline citations.)

## 2. Materials and Methods
(400-500 words. Systematic review: databases, PRISMA criteria, data extraction. Past tense.)

## 3. Results
(500-600 words. 3 sub-headings. Quantitative findings with inline citations.)

## 4. Discussion
(550-650 words. Interpretation, mechanisms, limitations, conservation policy. ≥5 citations.)

## 5. Conclusions
(120-150 words. 3 main findings, conservation significance, 2 future directions.)

## References
(APA 7th edition. All cited papers numbered, one per line. Prioritize the papers listed above.)

Write the complete paper now."""

    resp = _llamar_claude(client, system, prompt, max_tokens=4000, log=log)
    full_paper = resp.content[0].text.strip()

    # ------------------------------------------------------------------
    # Parsear secciones para la UI
    # ------------------------------------------------------------------
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
        m = re.search(pat, full_paper, re.DOTALL | re.MULTILINE)
        if m:
            sections[key] = m.group(1).strip()

    header = (
        f"# BiodiversIA – Scientific Paper\n"
        f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        f"*Tema: {topic}*\n"
        f"*Papers encontrados: {len(todos)} (PubMed: {len([p for p in pubmed_papers if 'error' not in p])}, "
        f"Scholar: {len([p for p in scholar_papers if 'error' not in p])})*\n\n---\n\n"
    )

    log("✅ Paper generado correctamente.")
    return {"sections": sections, "full_paper": header + full_paper}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="Ocean warming effects on coral reef biodiversity and marine conservation")
    p.add_argument("--output", default="scientific_paper_output.md")
    p.add_argument("--no-search", action="store_true")
    args = p.parse_args()

    result = run_paper_agent(topic=args.topic, search_literature=not args.no_search)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result["full_paper"])
    print(f"\nPaper guardado → {args.output}")
