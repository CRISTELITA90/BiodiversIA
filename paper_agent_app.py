"""
BiodiversIA – Scientific Paper Agent UI
Streamlit interface for the AI scientific paper drafting agent.
"""

import os
import threading
import queue
import time
from datetime import datetime

import streamlit as st

from scientific_paper_agent import run_paper_agent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BiodiversIA – Paper Agent",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0a3d62 0%, #1e6091 50%, #2980b9 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; margin: 0.5rem 0 0; opacity: 0.9; }

    .section-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    .section-card h3 { color: #1e6091; margin-top: 0; }

    .status-box {
        background: #f0f8ff;
        border-left: 4px solid #2980b9;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.82rem;
        max-height: 260px;
        overflow-y: auto;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e6091, #2980b9);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .metric-card .number { font-size: 2rem; font-weight: bold; }
    .metric-card .label  { font-size: 0.85rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🌊 BiodiversIA – Scientific Paper Agent</h1>
    <p>AI-powered Q1 scientific paper drafting on marine biodiversity, ecology & conservation<br>
    Literature sourced from <strong>PubMed</strong> &amp; <strong>Google Scholar</strong></p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Biodiversity.jpg/320px-Biodiversity.jpg",
             use_container_width=True, caption="Marine Biodiversity")

    st.header("⚙️ Configuration")

    predefined_topics = {
        "Custom topic...": "",
        "Climate change & coral reefs": (
            "Effects of climate change on coral reef biodiversity and resilience: "
            "a systematic review of bleaching events, species loss, and conservation strategies"
        ),
        "Marine Protected Areas efficacy": (
            "Effectiveness of Marine Protected Areas in conserving biodiversity "
            "and supporting ecosystem recovery in the context of global change"
        ),
        "Deep-sea biodiversity threats": (
            "Biodiversity of deep-sea ecosystems under threat: impacts of deep-sea mining, "
            "trawling, and climate change on benthic communities"
        ),
        "Seagrass & blue carbon": (
            "Seagrass meadows as biodiversity hotspots and blue carbon sinks: "
            "global decline, conservation priorities, and restoration approaches"
        ),
        "Ocean deoxygenation impacts": (
            "Ocean deoxygenation and its cascading effects on marine biodiversity: "
            "mechanisms, projections, and conservation implications"
        ),
        "Kelp forest ecology": (
            "Kelp forest ecosystem dynamics, biodiversity loss, and restoration "
            "under warming ocean temperatures and ocean acidification"
        ),
        "Coastal fisheries & ecosystems": (
            "Overfishing and habitat degradation effects on coastal marine biodiversity: "
            "ecosystem-based management and conservation approaches"
        ),
    }

    topic_choice = st.selectbox(
        "Select a topic or enter your own:",
        list(predefined_topics.keys()),
    )

    if topic_choice == "Custom topic...":
        topic = st.text_area(
            "Research topic",
            height=120,
            placeholder=(
                "E.g.: 'Impact of ocean acidification on mollusc diversity "
                "in Mediterranean coastal ecosystems: mechanisms and conservation responses'"
            ),
        )
    else:
        topic = predefined_topics[topic_choice]
        st.text_area("Topic (editable):", value=topic, height=100, key="topic_display")
        topic = st.session_state.get("topic_display", topic)

    st.markdown("---")
    st.header("📋 Journal Target")
    journal = st.selectbox(
        "Target journal style",
        [
            "Global Change Biology (IF ~11)",
            "Nature Communications (IF ~17)",
            "Marine Ecology Progress Series (IF ~3)",
            "Biological Conservation (IF ~7)",
            "Frontiers in Marine Science (IF ~4)",
            "Conservation Biology (IF ~5)",
        ],
    )

    st.markdown("---")
    st.header("🔑 API Key")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Required to run the agent. Get yours at console.anthropic.com",
    )
    st.caption("Key is used only for this session and never stored.")

    st.markdown("---")
    st.markdown(
        "**Author:** Cristela Yosmary Moreno García  \n"
        "📍 Canarias · Bioinformática & IA aplicada a conservación"
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "paper_result" not in st.session_state:
    st.session_state.paper_result = None
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "running" not in st.session_state:
    st.session_state.running = False

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Topic to research")
    if topic:
        st.info(topic)
    else:
        st.warning("Please select or enter a research topic in the sidebar.")

with col2:
    st.subheader("🎯 Selected journal")
    st.success(journal)

st.markdown("---")

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

run_col, info_col = st.columns([1, 2])

with run_col:
    run_btn = st.button(
        "🚀 Generate Scientific Paper",
        type="primary",
        disabled=st.session_state.running or not topic or not api_key,
        use_container_width=True,
    )

with info_col:
    st.markdown("""
    **The agent will:**
    1. Search PubMed with multiple targeted queries
    2. Fetch and read abstracts of the most relevant papers
    3. Search Google Scholar for complementary literature
    4. Draft all paper sections sequentially
    5. Compile the full Q1-quality manuscript
    """)

# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------

if run_btn:
    os.environ["ANTHROPIC_API_KEY"] = api_key
    st.session_state.paper_result = None
    st.session_state.log_messages = []
    st.session_state.running = True

    log_queue: queue.Queue = queue.Queue()
    result_container: dict = {}

    def _run():
        def _log(msg):
            log_queue.put(msg)

        try:
            result = run_paper_agent(topic=topic, progress_callback=_log)
            result_container["result"] = result
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            log_queue.put("__DONE__")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Live progress display
    st.subheader("⚙️ Agent progress")
    log_placeholder = st.empty()
    progress_bar = st.progress(0)

    sections_done = 0
    total_sections = 9
    logs = []

    while True:
        try:
            msg = log_queue.get(timeout=0.5)
        except queue.Empty:
            log_placeholder.markdown(
                '<div class="status-box">' +
                "<br>".join(f"▸ {m}" for m in logs[-30:]) +
                "</div>",
                unsafe_allow_html=True,
            )
            time.sleep(0.2)
            continue

        if msg == "__DONE__":
            break

        logs.append(msg)
        if "write_paper_section" in msg and "Tool:" in msg:
            sections_done += 1
            progress_bar.progress(min(sections_done / total_sections, 1.0))

        log_placeholder.markdown(
            '<div class="status-box">' +
            "<br>".join(f"▸ {m}" for m in logs[-30:]) +
            "</div>",
            unsafe_allow_html=True,
        )

    progress_bar.progress(1.0)

    if "error" in result_container:
        st.error(f"Agent error: {result_container['error']}")
        st.session_state.running = False
        st.stop()

    st.session_state.paper_result = result_container["result"]
    st.session_state.running = False
    st.rerun()

# ---------------------------------------------------------------------------
# Display result
# ---------------------------------------------------------------------------

if st.session_state.paper_result:
    result = st.session_state.paper_result
    sections = result["sections"]
    full_paper = result["full_paper"]

    st.success("✅ Paper successfully generated!")

    # Metrics
    word_count = len(full_paper.split())
    ref_count = full_paper.count("(20") + full_paper.count("(19")  # rough estimate
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="number">{len(sections)}</div><div class="label">Sections written</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="number">{word_count:,}</div><div class="label">Total words</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="number">{ref_count}</div><div class="label">Approx. citations</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="number">Q1</div><div class="label">Target quality</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Download buttons
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="⬇️ Download paper (.md)",
            data=full_paper.encode("utf-8"),
            file_name=f"scientific_paper_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl_col2:
        # Plain text version
        plain = full_paper.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
        st.download_button(
            label="⬇️ Download paper (.txt)",
            data=plain.encode("utf-8"),
            file_name=f"scientific_paper_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")

    # Section viewer
    section_order = [
        "title", "abstract", "keywords", "introduction",
        "methods", "results", "discussion", "conclusions", "references",
    ]
    section_labels = {
        "title": "📌 Title",
        "abstract": "📄 Abstract",
        "keywords": "🔑 Keywords",
        "introduction": "1️⃣ Introduction",
        "methods": "2️⃣ Materials & Methods",
        "results": "3️⃣ Results",
        "discussion": "4️⃣ Discussion",
        "conclusions": "5️⃣ Conclusions",
        "references": "📚 References",
    }

    tabs = st.tabs([section_labels[s] for s in section_order if s in sections])
    visible = [s for s in section_order if s in sections]

    for tab, sec in zip(tabs, visible):
        with tab:
            st.markdown(sections[sec])

    st.markdown("---")
    with st.expander("📄 Full paper (raw markdown)", expanded=False):
        st.text_area("", full_paper, height=600)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<small>BiodiversIA Scientific Paper Agent · "
    "Powered by Claude (Anthropic) · PubMed NCBI E-utilities · Google Scholar · "
    "Developed by Cristela Yosmary Moreno García</small>",
    unsafe_allow_html=True,
)
