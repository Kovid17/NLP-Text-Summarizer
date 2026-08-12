"""
app/app.py  —  Interactive Text Summarization Demo (Streamlit)
Run: streamlit run app/app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background removed to respect Streamlit native theme */

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin: 6px 0;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card h2 { color: #a78bfa; font-size: 1.8rem; margin: 0; }
.metric-card p  { color: rgba(255,255,255,0.7); margin: 4px 0 0 0; font-size: 0.85rem; }

/* New Hero banner */
.new-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 40px 20px 20px;
    font-family: 'Inter', sans-serif;
}
.trusted-by {
    font-size: 0.75rem;
    color: #888;
    margin-bottom: 60px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.trusted-logos {
    display: flex;
    justify-content: center;
    gap: 20px;
    color: #444;
    font-weight: 500;
    font-size: 1.1rem;
    margin-top: 10px;
}
.trusted-logos span.active { color: #fff; }

.feature-badge {
    background: rgba(255, 80, 80, 0.1);
    color: #ff9999;
    border: 1px solid rgba(255, 80, 80, 0.2);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 20px;
}

.new-hero h1 {
    font-size: 3.5rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.1;
    max-width: 800px;
    margin: 0 0 20px 0;
}
.new-hero p.subtitle {
    font-size: 1.05rem;
    color: #888;
    max-width: 600px;
    line-height: 1.6;
    margin: 0 0 40px 0;
}

.hero-actions {
    display: flex;
    gap: 24px;
    align-items: center;
    margin-bottom: 50px;
}
.btn-primary-hero {
    background: linear-gradient(135deg, #ff6b6b, #ff4757);
    color: white !important;
    text-decoration: none;
    padding: 12px 28px;
    border-radius: 30px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.btn-primary-hero:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(255, 71, 87, 0.3);
}
.btn-secondary-hero {
    color: #ccc !important;
    text-decoration: none;
    font-weight: 500;
    font-size: 0.95rem;
    transition: color 0.2s;
}
.btn-secondary-hero:hover { color: #fff !important; }

/* Glow around the app */
.app-container-glow {
    position: relative;
    width: 100%;
}
.glow-effect {
    position: absolute;
    top: 10%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 60%;
    height: 300px;
    background: radial-gradient(circle, rgba(255,71,87,0.12) 0%, rgba(0,0,0,0) 70%);
    filter: blur(60px);
    z-index: 0;
    pointer-events: none;
}
.content-wrapper {
    position: relative;
    z-index: 1;
    background: rgba(15, 15, 15, 0.6);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
}

/* Summary output box */
.summary-box {
    background: rgba(167,139,250,0.12);
    border: 1px solid rgba(167,139,250,0.4);
    border-radius: 14px;
    padding: 20px 24px;
    color: #e2e8f0;
    font-size: 0.98rem;
    line-height: 1.7;
    min-height: 320px;
}

/* Phase badge */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-1 { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
.badge-2 { background: rgba(96,165,250,0.2); color: #60a5fa; border: 1px solid #60a5fa; }
.badge-3 { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid #fbbf24; }

/* Headings */
h1, h2, h3 { color: #fff !important; }
p, li, label { color: rgba(255,255,255,0.85) !important; }

/* Input / TextArea */
textarea, input {
    background: rgba(255,255,255,0.07) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(124,58,237,0.4);
}

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "CNN/DailyMail  (News — Multi-sentence)":    ("facebook/bart-large-cnn",          1),
    "SAMSum  (Dialogue — Conversation)":          ("philschmid/bart-large-cnn-samsum", 1),
    "XSum  (BBC News — One-line)":                ("facebook/bart-large-xsum",          2),
    "Gigaword  (Headlines)":                      ("google/pegasus-gigaword",           2),
    "ArXiv  (Scientific Papers)":                 ("google/pegasus-arxiv",              3),
}

SAMPLE_TEXTS = {
    "News Article": (
        "The United Nations Climate Change Conference concluded on Saturday with a landmark "
        "agreement signed by 195 countries. The deal commits developed nations to providing "
        "$100 billion annually to developing countries to help them transition to clean energy "
        "and adapt to climate impacts. Scientists called it a historic step forward, though "
        "environmental groups warned that the pledges remain insufficient to limit global "
        "warming to 1.5 degrees Celsius above pre-industrial levels. World leaders praised "
        "the consensus-based process that allowed even the most vulnerable island nations "
        "to have their voices heard in the final text."
    ),
    "Dialogue / Chat": (
        "Alice: Hey Bob, did you finish the project report?\n"
        "Bob: Not yet, I'm stuck on the data analysis section.\n"
        "Alice: I can help. I already did something similar last week.\n"
        "Bob: That would be amazing! Can we meet at 3 PM?\n"
        "Alice: Sure, I'll bring my laptop. We should be done in an hour.\n"
        "Bob: Perfect. Thanks so much, Alice!"
    ),
    "Scientific Abstract": (
        "We present a novel transformer-based architecture for abstractive text summarization "
        "that incorporates hierarchical attention mechanisms at the sentence and document level. "
        "Our model, trained on a corpus of 3.8 million news-science articles, achieves state-"
        "of-the-art ROUGE-1 scores of 46.2 on the CNN/DailyMail benchmark, outperforming "
        "previous models by 2.1 points. We demonstrate that hierarchical encoding significantly "
        "reduces hallucination errors and improves factual consistency as measured by the "
        "FactCC metric. Ablation studies confirm the contribution of each architectural "
        "component to overall performance."
    ),
}


# ── Model loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str):
    return pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        device=-1,               # CPU; change to 0 for GPU
        framework="pt",
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    st.markdown("---")

    selected_model_label = st.selectbox(
        "📦 Model / Dataset",
        list(MODEL_REGISTRY.keys()),
        index=0,
    )
    model_id, phase = MODEL_REGISTRY[selected_model_label]

    phase_labels = {1: "Phase 1 — Foundation", 2: "Phase 2 — Expansion", 3: "Phase 3 — Advanced"}
    badge_cls    = {1: "badge-1", 2: "badge-2", 3: "badge-3"}
    st.markdown(
        f'<span class="badge {badge_cls[phase]}">{phase_labels[phase]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Model:** `{model_id}`")

    st.markdown("---")
    st.markdown("### ⚙️ Generation Settings")
    gen_mode  = st.radio("Mode", ["Beam Search", "Sampling"], horizontal=True)
    num_beams = st.slider("Num Beams", 1, 8, 4) if gen_mode == "Beam Search" else 1
    temperature = st.slider("Temperature", 0.5, 1.5, 0.7, 0.05) if gen_mode == "Sampling" else 1.0
    max_length  = st.slider("Max Summary Length", 30, 256, 128)
    min_length  = st.slider("Min Summary Length", 5,  64,  20)




# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="text-align: center; color: white; margin-bottom: 2rem; font-family: 'Inter', sans-serif; font-weight: 700;">TEXT-Summarizer</h1>
<div class="app-container-glow">
    <div class="glow-effect"></div>
    <div class="content-wrapper">
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["✨ Summarize", "📊 Model Comparison", "📚 Dataset Explorer"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Summarize
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📄 Input Text")
        sample_choice = st.selectbox("Load a sample", ["— custom input —"] + list(SAMPLE_TEXTS.keys()))
        default_text  = SAMPLE_TEXTS.get(sample_choice, "")
        user_text = st.text_area(
            "Paste or type your text here:",
            value=default_text,
            height=320,
            placeholder="Enter news article, conversation, scientific text …",
        )

        word_count = len(user_text.split()) if user_text.strip() else 0
        st.caption(f"📏 {word_count} words")

        summarize_btn = st.button("⚡ Generate Summary", use_container_width=True)

    with col_right:
        st.markdown("### 📋 Generated Summary")

        if summarize_btn:
            if not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                with st.spinner(f"Loading `{model_id}` and generating …"):
                    try:
                        t0       = time.time()
                        pipe     = load_pipeline(model_id)
                        result   = pipe(
                            user_text,
                            max_length=max_length,
                            min_length=min_length,
                            num_beams=num_beams,
                            temperature=temperature if gen_mode == "Sampling" else 1.0,
                            do_sample=(gen_mode == "Sampling"),
                            early_stopping=True,
                            no_repeat_ngram_size=3,
                        )
                        summary  = result[0]["summary_text"]
                        elapsed  = time.time() - t0

                        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

                        # Quick stats
                        st.markdown("---")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f'<div class="metric-card"><h2>{len(summary.split())}</h2><p>Summary Words</p></div>', unsafe_allow_html=True)
                        with c2:
                            ratio = round(word_count / max(len(summary.split()), 1), 1)
                            st.markdown(f'<div class="metric-card"><h2>{ratio}×</h2><p>Compression Ratio</p></div>', unsafe_allow_html=True)
                        with c3:
                            st.markdown(f'<div class="metric-card"><h2>{elapsed:.1f}s</h2><p>Inference Time</p></div>', unsafe_allow_html=True)

                        # ROUGE against reference (if available)
                        st.markdown("#### 🎯 ROUGE Score (vs. your reference)")
                        ref_text = st.text_area("Optional: paste reference summary to compute ROUGE", height=80, key="ref_box")
                        if ref_text.strip():
                            from src.evaluator import Evaluator
                            scores = Evaluator().compute([summary], [ref_text.strip()])
                            cols = st.columns(len(scores))
                            for col, (metric, val) in zip(cols, scores.items()):
                                with col:
                                    st.markdown(f'<div class="metric-card"><h2>{val}</h2><p>{metric.upper()}</p></div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.info("💡 Make sure you have internet access to download the model on first run.")
        else:
            st.markdown("""
            <div style="
                height:320px; display:flex; align-items:center; justify-content:center;
                background: rgba(255,255,255,0.04); border-radius: 14px;
                border: 1px dashed rgba(255,255,255,0.2); color: rgba(255,255,255,0.4);
                font-size: 1rem;
            ">
                ← Enter text and click Generate Summary
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 ROUGE Score Comparison Across Datasets")
    st.markdown("*Typical benchmark ROUGE scores for pre-trained models (literature values)*")

    # Benchmark data from published papers
    benchmark_data = {
        "Dataset":    ["CNN/DailyMail", "SAMSum", "XSum",   "Gigaword", "ArXiv"],
        "Model":      ["BART-Large-CNN","BART-SAMSum","BART-XSum","Pegasus","Pegasus"],
        "ROUGE-1":    [44.16,           53.22,        45.14,     39.12,    44.21],
        "ROUGE-2":    [21.28,           28.97,        22.27,     19.86,    17.06],
        "ROUGE-L":    [40.90,           48.60,        37.25,     36.24,    38.96],
        "BLEU":       [18.5,            22.1,         14.3,      16.8,     11.2],
    }
    df = pd.DataFrame(benchmark_data)

    # Bar chart
    fig = px.bar(
        df.melt(id_vars=["Dataset", "Model"], value_vars=["ROUGE-1", "ROUGE-2", "ROUGE-L"]),
        x="Dataset", y="value", color="variable", barmode="group",
        title="ROUGE-1 / ROUGE-2 / ROUGE-L by Dataset",
        color_discrete_sequence=["#a78bfa", "#60a5fa", "#34d399"],
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend_title_text="Metric",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("---")
    st.markdown("### 🕸️ Model Performance Radar")
    radar_model = st.selectbox("Select Model for Radar", df["Dataset"].tolist())
    row = df[df["Dataset"] == radar_model].iloc[0]

    categories  = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
    vals        = [row[c] for c in categories]
    vals        += vals[:1]
    angles      = [i / len(categories) * 360 for i in range(len(categories))]
    angles      += angles[:1]

    fig_radar = go.Figure(go.Scatterpolar(
        r=vals, theta=categories + [categories[0]],
        fill="toself",
        line_color="#a78bfa",
        fillcolor="rgba(167,139,250,0.2)",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 55], color="white"),
            angularaxis=dict(color="white"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title=f"{radar_model} — {row['Model']}",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Table
    st.markdown("### 📋 Full Benchmark Table")
    st.dataframe(
        df.style.background_gradient(subset=["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"], cmap="Purples"),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📚 Dataset Overview")

    dataset_info = [
        {"Phase": "1 🟢", "Dataset": "CNN/DailyMail",  "Size": "~287K",  "Domain": "News (General)",   "Type": "Article → Multi-sentence", "Model": "BART-Large-CNN"},
        {"Phase": "1 🟢", "Dataset": "SAMSum",          "Size": "~16K",   "Domain": "Dialogue / Chat",  "Type": "Conversation → Summary",  "Model": "BART-SAMSum"},
        {"Phase": "2 🔵", "Dataset": "XSum",            "Size": "~226K",  "Domain": "BBC News",         "Type": "Article → One-line",       "Model": "BART-Large-XSum"},
        {"Phase": "2 🔵", "Dataset": "Gigaword",        "Size": "~3.8M",  "Domain": "News Headlines",   "Type": "Sentence → Headline",      "Model": "Pegasus-Gigaword"},
        {"Phase": "3 🟡", "Dataset": "ArXiv",           "Size": "Large",  "Domain": "Scientific",       "Type": "Paper → Abstract",         "Model": "Pegasus-ArXiv"},
        {"Phase": "3 🟡", "Dataset": "Reddit TIFU",     "Size": "~4M",    "Domain": "Social Media",     "Type": "Story → TL;DR",            "Model": "BART-Large (FT)"},
        {"Phase": "3 🔴", "Dataset": "ACLSum",          "Size": "~250",   "Domain": "NLP Research",     "Type": "Paper → Summary",          "Model": "LED-Base-16384"},
    ]
    df_datasets = pd.DataFrame(dataset_info)
    st.dataframe(df_datasets, use_container_width=True, hide_index=True)

    # Dataset → Model flowchart as Plotly Sankey
    st.markdown("---")
    st.markdown("### 🔀 Dataset → Model Mapping")

    all_nodes   = ["CNN/DM", "SAMSum", "XSum", "Gigaword", "ArXiv", "Reddit", "ACLSum",
                   "BART-CNN", "BART-SAMSum", "BART-XSum", "Pegasus-GW", "Pegasus-ArXiv", "BART-FT", "LED"]
    source_idx  = [0, 1, 2, 3, 4, 5, 6]
    target_idx  = [7, 8, 9, 10, 11, 12, 13]
    values      = [287, 16, 226, 3800, 200, 4000, 250]

    fig_sankey = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=all_nodes,
            color=["#a78bfa"]*7 + ["#60a5fa"]*7,
        ),
        link=dict(source=source_idx, target=target_idx, value=values,
                  color=["rgba(167,139,250,0.4)"]*7),
    ))
    fig_sankey.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        title_text="Dataset → Model Flow (size ∝ dataset samples)",
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)
