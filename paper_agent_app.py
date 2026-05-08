"""
BiodiversIA – Scientific Paper Agent UI (simplificada)
"""

import os
import threading
import queue
from datetime import datetime

import streamlit as st
from scientific_paper_agent import run_paper_agent

st.set_page_config(
    page_title="BiodiversIA – Paper Agent",
    page_icon="🌊",
    layout="centered",
)

st.title("🌊 BiodiversIA – Generador de Paper Científico Q1")
st.caption("Biodiversidad marina · Ecología · Conservación")
st.info(
    "**Cómo funciona (rápido, ~40 segundos en total):**  \n"
    "1️⃣ Busca los **15 papers más relevantes en PubMed** — API de NCBI, gratis, ~5 seg  \n"
    "2️⃣ **Una sola llamada a Claude** genera el paper Q1 completo con esas referencias — ~30 seg",
    icon="💡"
)

st.markdown("---")

# --- API Key ---
api_key = st.text_input("🔑 Anthropic API Key", type="password",
                        placeholder="sk-ant-...")

# --- Tema ---
TEMAS = {
    "Selecciona un tema predefinido…": "",
    "Arrecifes de coral y cambio climático": "Effects of climate change on coral reef biodiversity: bleaching, species loss and conservation strategies",
    "Áreas Marinas Protegidas": "Effectiveness of Marine Protected Areas in conserving marine biodiversity under global change",
    "Biodiversidad de aguas profundas": "Deep-sea biodiversity under threat: impacts of mining, trawling and climate change on benthic communities",
    "Praderas de posidonia y carbono azul": "Seagrass meadows as biodiversity hotspots and blue carbon sinks: decline, conservation and restoration",
    "Desoxigenación del océano": "Ocean deoxygenation effects on marine biodiversity: mechanisms, projections and conservation implications",
    "Bosques de kelp": "Kelp forest biodiversity loss and restoration under ocean warming and acidification",
    "Pesca y ecosistemas costeros": "Overfishing and habitat degradation effects on coastal marine biodiversity and ecosystem-based management",
}

col1, col2 = st.columns([1, 1])
with col1:
    eleccion = st.selectbox("Tema predefinido", list(TEMAS.keys()))
with col2:
    journal = st.selectbox("Revista objetivo", [
        "Global Change Biology",
        "Biological Conservation",
        "Frontiers in Marine Science",
        "Marine Ecology Progress Series",
        "Nature Communications",
    ])

topic_default = TEMAS.get(eleccion, "")
topic = st.text_area("✏️ Tema del paper (editable)", value=topic_default, height=90,
                     placeholder="Describe el tema del paper en inglés…")

st.markdown("---")

# --- Botón ---
generar = st.button("🚀 Generar Paper Q1", type="primary",
                    disabled=not api_key or not topic)

if not api_key:
    st.warning("Introduce tu API Key para continuar.")
elif not topic:
    st.warning("Selecciona o escribe un tema.")

# --- Ejecución ---
if "resultado" not in st.session_state:
    st.session_state.resultado = None

if generar:
    os.environ["ANTHROPIC_API_KEY"] = api_key
    st.session_state.resultado = None

    log_box  = st.empty()
    progress = st.progress(0)
    logs     = []
    q        = queue.Queue()
    result_c = {}

    def _run():
        def _log(m):
            q.put(m)
        try:
            result_c["ok"] = run_paper_agent(
                topic=topic,
                search_literature=True,   # busca PubMed+Scholar (sin Claude), luego 1 llamada
                progress_callback=_log,
            )
        except Exception as e:
            result_c["err"] = str(e)
        finally:
            q.put("__DONE__")

    threading.Thread(target=_run, daemon=True).start()

    steps = ["Conectando con Claude…", "Redactando introducción…",
             "Redactando métodos y resultados…", "Redactando discusión…",
             "Compilando paper…"]
    step_i = 0

    while True:
        try:
            msg = q.get(timeout=0.5)
        except queue.Empty:
            if step_i < len(steps):
                log_box.info(f"⏳ {steps[min(step_i, len(steps)-1)]}")
            continue

        if msg == "__DONE__":
            break
        logs.append(msg)
        step_i += 1
        progress.progress(min(step_i * 15, 90))

    progress.progress(100)

    if "err" in result_c:
        st.error(f"Error: {result_c['err']}")
    else:
        st.session_state.resultado = result_c["ok"]
        st.rerun()

# --- Resultado ---
if st.session_state.resultado:
    r = st.session_state.resultado
    sections = r["sections"]
    full     = r["full_paper"]

    st.success("✅ Paper generado correctamente")

    words = len(full.split())
    c1, c2, c3 = st.columns(3)
    c1.metric("Secciones", len(sections))
    c2.metric("Palabras", f"{words:,}")
    c3.metric("Calidad", "Q1")

    st.markdown("---")

    # Descargas
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ Descargar .md", full.encode(),
                           f"paper_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                           "text/markdown", use_container_width=True)
    with d2:
        plain = full.replace("**","").replace("*","").replace("#","")
        st.download_button("⬇️ Descargar .txt", plain.encode(),
                           f"paper_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                           "text/plain", use_container_width=True)

    st.markdown("---")

    # Tabs por sección
    orden  = ["title","abstract","keywords","introduction","methods",
               "results","discussion","conclusions","references"]
    labels = ["📌 Título","📄 Abstract","🔑 Keywords","1️⃣ Introducción",
               "2️⃣ Métodos","3️⃣ Resultados","4️⃣ Discusión","5️⃣ Conclusiones","📚 Referencias"]

    visibles = [(l, s) for l, s in zip(labels, orden) if s in sections]
    if visibles:
        tabs = st.tabs([l for l, _ in visibles])
        for tab, (_, sec) in zip(tabs, visibles):
            with tab:
                st.markdown(sections[sec])

    with st.expander("📄 Ver paper completo (markdown)"):
        st.text_area("", full, height=500)

st.markdown("---")
st.caption("BiodiversIA · Cristela Yosmary Moreno García · Canarias")
