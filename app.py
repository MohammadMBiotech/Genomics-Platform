"""
Interactive Population Genomics Platform — Home Page

A comprehensive Streamlit application for SNP-based population genetics
analysis with focus on plant genomics research.
"""

import streamlit as st
from utils import get_geno_from_session, get_meta_from_session, display_data_summary


# ═══════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="Interactive Population Genomics Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════
# CUSTOM STYLING
# ═══════════════════════════════════════════
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5 0%, #43A047 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .mod-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #1E88E5;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-loaded {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .status-missing {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 5px solid #ffc107;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .workflow-step {
        background: #e3f2fd;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #1E88E5;
        margin-right: 10px;
    }
    /* Show Home in sidebar with better label */
    [data-testid="stSidebarNav"] li:first-child a p::before {
        content: "🏠 ";
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════
st.markdown(
    '<p class="main-header">🧬 Interactive Population Genomics Platform</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Comprehensive SNP-based population genetics '
    'analysis for plant genomics research</p>',
    unsafe_allow_html=True,
)

st.markdown("---")


# ═══════════════════════════════════════════
# DATA STATUS CHECK
# ═══════════════════════════════════════════
geno, marker_info = get_geno_from_session(warn_if_missing=False)
meta = get_meta_from_session()

if geno is not None:
    st.markdown('<div class="status-loaded">', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("✅ Samples loaded", f"{geno.shape[0]:,}")
    sc2.metric("✅ Markers loaded", f"{geno.shape[1]:,}")
    missing_pct = geno.isna().sum().sum() / geno.size * 100
    sc3.metric("Missing data", f"{missing_pct:.1f}%")
    sc4.metric("Metadata",
                "✅ Loaded" if meta is not None else "⚠️ Not loaded")
    st.markdown('</div>', unsafe_allow_html=True)

    display_data_summary(geno, marker_info, meta)
else:
    st.markdown('<div class="status-missing">', unsafe_allow_html=True)
    st.markdown(
        "### ⚠️ No data loaded yet\n"
        "👉 Start by visiting **📁 Upload Metadata** in the sidebar."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# WELCOME
# ═══════════════════════════════════════════
st.header("👋 Welcome")
st.markdown("""
This platform provides a **complete suite of population genomics analyses**
designed for plant researchers, breeders, and evolutionary biologists.

**Supported input formats:**
- **Numeric (0/1/2 dosage)** — CSV, TSV, TXT, Excel
- **HapMap** — Standard `.hmp.txt` format
- **VCF** — Variant Call Format (v4.x)
- **Compressed** — All formats support `.gz`
""")


# ═══════════════════════════════════════════
# ANALYSIS MODULES
# ═══════════════════════════════════════════
st.header("📂 Analysis Modules")
st.markdown(
    "Navigate to any module from the sidebar. "
    "Modules are organized by analysis category."
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="mod-box">', unsafe_allow_html=True)
    st.markdown("""
### 🧬 Core Genomics

- **📁 Upload & Metadata** — Multi-format SNP input
- **🧹 Quality Control** — Missing, MAF, HWE filtering
- **🧬 SNP Statistics** — Allele freq, PIC, per-marker
- **🌿 Genetic Diversity** — Ho, He, Fis, Na, Ne
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="mod-box">', unsafe_allow_html=True)
    st.markdown("""
### 📊 Population Analysis

- **🧩 Population Structure** — PCA, STRUCTURE, ΔK
- **🌳 Phylogenetics** — NJ, ML, Fitch, bootstrap
- **🎯 Clustering** — Hierarchical, K-means, DBSCAN
- **👥 Kinship & Relatedness** — VanRaden, IBS
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="mod-box">', unsafe_allow_html=True)
    st.markdown("""
### 🔬 Advanced

- **🔗 Linkage Disequilibrium** — r²/D', decay
- **🌍 Geographic Genetics** — Fst, DEST, Nm, IBD
- **🤖 Machine Learning** — 9 selection methods
- **📑 Reports & Export** — Summaries, formats
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# WORKFLOW
# ═══════════════════════════════════════════
st.markdown("---")
st.header("🚀 Recommended Workflow")

workflow_steps = [
    ("1", "📁 Upload data", "Upload genotype file + optional metadata"),
    ("2", "🧹 Quality Control", "Filter markers by MAF/missing/HWE"),
    ("3", "🧬 Explore diversity", "Per-marker stats, population diversity"),
    ("4", "🧩 Population structure", "PCA + STRUCTURE"),
    ("5", "🌳 Phylogeny", "Build tree with bootstrap"),
    ("6", "🌍 Geographic patterns", "Fst, IBD, IBE"),
    ("7", "🔗 LD & selection", "LD decay, selection detection"),
    ("8", "🤖 Machine learning", "Classification, feature selection"),
    ("9", "📑 Report & export", "Generate report, download"),
]

for step_num, step_title, step_desc in workflow_steps:
    st.markdown(
        f'<div style="display:flex; align-items:center; padding:0.5rem 0;">'
        f'<span class="workflow-step">{step_num}</span>'
        f'<span><b>{step_title}</b> — {step_desc}</span></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════
# QUICK START
# ═══════════════════════════════════════════
st.markdown("---")
st.header("🎯 Quick Start")

qs1, qs2 = st.columns(2)
with qs1:
    st.success("""
### 🆕 New Users
1. Click **📁 Upload Metadata** in sidebar
2. Upload your genotype file
3. Optionally upload metadata
4. Explore each module in order
    """)

with qs2:
    st.info("""
### 🔬 Experienced Users
- Jump to your analysis of interest
- Use **Machine Learning** for selection detection
- Use **Geographic Genetics** for landscape genomics
- Use **Reports** for comprehensive summary
    """)


# ═══════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; padding:1rem;">
🧬 <b>Interactive Population Genomics Platform</b><br>
Built with Streamlit · Powered by scikit-learn, scipy, numpy, plotly<br>
For plant genomics research and beyond
</div>
""", unsafe_allow_html=True)