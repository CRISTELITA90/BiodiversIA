"""
BiodiversIA – Scientific Paper Agent
Una sola llamada a la API genera el paper completo.
"""

import time
from datetime import datetime

import anthropic
from anthropic import RateLimitError, APIStatusError

MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _claude_call(client, system, prompt, max_tokens=4000):
    wait = 30
    for attempt in range(6):
        try:
            return client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except RateLimitError:
            if attempt == 5:
                raise
            print(f"  [Rate limit] Esperando {wait}s…")
            time.sleep(wait)
            wait = min(wait * 2, 120)
        except APIStatusError as e:
            if e.status_code == 529:
                if attempt == 5:
                    raise
                print(f"  [Sobrecarga] Esperando {wait}s…")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                raise

# ---------------------------------------------------------------------------
# PubMed (solo para enriquecer, totalmente opcional)
# ---------------------------------------------------------------------------

def _quick_pubmed(topic: str) -> str:
    """Busca 3 papers en PubMed y devuelve un bloque de texto compacto."""
    try:
        import requests, xml.etree.ElementTree as ET, time as t

        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        q = f"{topic} marine biodiversity conservation"[:100]

        r = requests.get(f"{base}/esearch.fcgi",
                         params={"db":"pubmed","term":q,"retmax":4,"retmode":"json"},
                         timeout=10)
        pmids = r.json()["esearchresult"].get("idlist", [])[:3]
        if not pmids:
            return ""

        t.sleep(0.4)
        r2 = requests.get(f"{base}/efetch.fcgi",
                          params={"db":"pubmed","id":",".join(pmids),
                                  "retmode":"xml","rettype":"abstract"},
                          timeout=15)
        root = ET.fromstring(r2.text)
        lines = []
        for art in root.findall(".//PubmedArticle"):
            title = "".join((art.find(".//ArticleTitle") or ET.Element("x")).itertext())
            ab    = " ".join("".join(el.itertext()) for el in art.findall(".//AbstractText"))[:300]
            auths = [f"{a.findtext('LastName','')} {a.findtext('Initials','')}".strip()
                     for a in art.findall(".//Author") if a.findtext("LastName")][:2]
            year  = art.findtext(".//PubDate/Year") or "?"
            lines.append(f"- {', '.join(auths)} ({year}). {title}. {ab}")
        return "\n".join(lines)
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Main: genera el paper completo en UNA sola llamada
# ---------------------------------------------------------------------------

def run_paper_agent(
    topic: str = "Impact of climate change on coral reef biodiversity and conservation",
    search_literature: bool = False,
    progress_callback=None,
) -> dict:

    log = progress_callback or print
    client = anthropic.Anthropic()

    # Referencias opcionales de PubMed (muy compactas)
    refs_block = ""
    if search_literature:
        log("Buscando referencias en PubMed…")
        refs_block = _quick_pubmed(topic)
        if refs_block:
            log(f"  3 referencias encontradas.")
            refs_block = f"\n\nReal papers found (use these as citations):\n{refs_block}"

    log("Generando paper completo (una sola llamada)…")

    system = "Expert marine ecologist and scientific writer. Write in formal scientific English."

    prompt = f"""Write a complete Q1 scientific review paper on:

TOPIC: {topic}
{refs_block}

Write ALL sections in order, clearly labelled with markdown headers:

# [TITLE]
(≤18 words, specific, includes habitat + variable + scope)

## Abstract
(≤220 words. Paragraphs: Background / Methods / Results / Conclusions.
Include quantitative data. Cite 2-3 real papers as Author et al., Year.)

## Keywords
(6 terms, comma-separated)

## 1. Introduction
(500-600 words. Funnel: global context → knowledge gap → objectives.
Cite ≥8 real published papers inline as Author et al., Year.)

## 2. Materials and Methods
(400-500 words. Systematic review design: databases, search terms,
PRISMA inclusion/exclusion criteria, data extraction. Past tense.)

## 3. Results
(500-600 words. 3 sub-headings. Quantitative findings with inline citations.)

## 4. Discussion
(550-650 words. Interpretation, mechanisms, limitations, conservation policy.
Cite ≥5 papers.)

## 5. Conclusions
(120-150 words. 3 main findings, conservation significance, 2 future directions.)

## References
(APA 7th. All cited papers numbered, one per line.)

Write the complete paper now."""

    response = _claude_call(client, system, prompt, max_tokens=4000)
    full_paper = response.content[0].text.strip()

    # Dividir secciones para la UI
    import re
    sections = {}
    patterns = {
        "title":        r"#\s+(.+?)(?=\n##|\Z)",
        "abstract":     r"## Abstract\n(.+?)(?=\n##|\Z)",
        "keywords":     r"## Keywords\n(.+?)(?=\n##|\Z)",
        "introduction": r"## 1\. Introduction\n(.+?)(?=\n##|\Z)",
        "methods":      r"## 2\. Materials and Methods\n(.+?)(?=\n##|\Z)",
        "results":      r"## 3\. Results\n(.+?)(?=\n##|\Z)",
        "discussion":   r"## 4\. Discussion\n(.+?)(?=\n##|\Z)",
        "conclusions":  r"## 5\. Conclusions\n(.+?)(?=\n##|\Z)",
        "references":   r"## References\n(.+?)(?=\n##|\Z)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, full_paper, re.DOTALL)
        if m:
            sections[key] = m.group(1).strip()

    header = (
        f"# BiodiversIA – Scientific Paper\n"
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        f"*Topic: {topic}*\n\n---\n\n"
    )
    log("✓ Paper generado correctamente.")
    return {"sections": sections, "full_paper": header + full_paper}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default=(
        "Ocean warming effects on coral reef biodiversity and marine conservation strategies"
    ))
    p.add_argument("--output", default="scientific_paper_output.md")
    p.add_argument("--search", action="store_true", help="Buscar en PubMed primero")
    args = p.parse_args()

    result = run_paper_agent(topic=args.topic, search_literature=args.search)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result["full_paper"])
    print(f"\nPaper guardado → {args.output}")
