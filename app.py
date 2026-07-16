import streamlit as st

st.set_page_config(
    page_title="Interactive Population Genomics Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {font-size:2.5rem;font-weight:bold;color:#1E88E5;text-align:center;margin-bottom:.5rem;}
    .sub-header {font-size:1.2rem;color:#666;text-align:center;margin-bottom:2rem;}
    .mod-box {background:#f8f9fa;border-left:4px solid #1E88E5;padding:1rem 1.5rem;border-radius:6px;margin-bottom:1rem;}
    [data-testid="stSidebarNav"] li:first-child {display:none;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧬 Interactive Population Genomics Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive SNP-based population genetics analysis for plant genomics research</p>', unsafe_allow_html=True)
st.markdown("---")

st.header("👋 Welcome")
st.markdown("""
This platform supports **Numeric (0/1/2)**, **HapMap**, and **VCF** genotype formats.  
Upload your genotype data and optional metadata to access all analysis modules.
""")

st.header("📂 Analysis Modules")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="mod-box">', unsafe_allow_html=True)
    st.markdown("""### 🧬 Core Genomics
- **Upload & Metadata** — Multi-format SNP input
- **Quality Control** — Missing rate, MAF, HWE
- **SNP Statistics** — Allele frequencies, PIC
- **Genetic Diversity** — He, Ho, Fis, Na, Ne, Shannon
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="mod-box">', unsafe_allow_html=True)
    st.markdown("""### 📊 Population Analysis
- **Population Structure** — PCA, PCoA, STRUCTURE
- **Phylogenetics** — NJ, ML, Parsimony trees
- **Clustering** — Hierarchical, K-means, DBSCAN
- **Kinship & Relatedness** — IBS/IBD matrices
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="mod-box">', unsafe_allow_html=True)
    st.markdown("""### 🔬 Advanced
- **Linkage Disequilibrium** — LD decay, haplotype blocks
- **Geographic Genetics** — Mantel, IBD, IBE
- **Machine Learning** — Classification, clustering
- **Reports & Export** — Interactive summaries
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.header("🚀 Quick Start")
st.markdown("""
1. Go to **Upload & Metadata** → upload genotype file (Numeric/HapMap/VCF)
2. Optionally upload metadata CSV (Sample ID, Species, Origin, Group)
3. Navigate to any analysis module
4. Configure parameters and run
5. Download results and publication-quality plots
""")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;padding:1rem;">🧬 Interactive Population Genomics Platform | Built with Streamlit</div>', unsafe_allow_html=True)