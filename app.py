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
    .feature-item {
        padding: 0.5rem 0;
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
    [data-testid="stSidebarNav"] li:first-child { display: none; }
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

    # Also show in sidebar
    display_data_summary(geno, marker_info, meta)
else:
    st.markdown('<div class="status-missing">', unsafe_allow_html=True)
    st.markdown(
        "### ⚠️ No data loaded yet\n"
        "👉 Start by visiting **📁 Upload & Metadata** in the sidebar."
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

**Key advantages:**
- 🚀 **Fast** — Vectorized computations (50-100× faster than naive implementations)
- 📊 **Publication-quality** — Interactive Plotly figures with export
- 🎯 **Comprehensive** — 13 analysis modules covering the full workflow
- 🔬 **Rigorous** — Uses standard formulas from GWASTools, PLINK, qqman
- 🌐 **Interactive** — Everything runs in your browser, no installation needed
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

- **📁 Upload & Metadata** — Multi-format SNP input with auto-detection
- **🧹 Quality Control** — Missing rate, MAF, HWE filtering
- **🧬 SNP Statistics** — Allele frequencies, PIC, per-marker stats
- **🌿 Genetic Diversity** — Ho, He, Fis, Na, Ne, Shannon index

*Foundation for all downstream analyses.*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="mod-box">', unsafe_allow_html=True)
    st.markdown("""
### 📊 Population Analysis

- **🧩 Population Structure** — PCA (2D/3D/Biplot), PCoA, STRUCTURE (Evanno ΔK, CLUMPP), fastStructure
- **🌳 Phylogenetics** — NJ, UPGMA, ML (JC69/K2P), Fitch parsimony, bootstrap
- **🎯 Clustering** — Hierarchical, K-means, DBSCAN with real population labels
- **👥 Kinship & Relatedness** — VanRaden, IBS, Astle-Balding matrices

*Discover population structure and relationships.*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="mod-box">', unsafe_allow_html=True)
    st.markdown("""
### 🔬 Advanced Analyses

- **🔗 Linkage Disequilibrium** — r²/D', Hill-Weir decay, Gabriel blocks
- **🌍 Geographic Genetics** — Fst, Gst, DEST, Nm, IBD, IBE, gene flow network
- **🤖 Machine Learning** — Classification, feature selection, **9 selection detection methods**
- **📑 Reports & Export** — Comprehensive summaries, multi-format export

*Publication-ready advanced analyses.*
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# HIGHLIGHTED FEATURES
# ═══════════════════════════════════════════
st.markdown("---")
st.header("✨ Highlighted Features")

fc1, fc2 = st.columns(2)

with fc1:
    st.markdown("""
### 🎯 Selection Detection Suite
Detect SNPs under natural selection using **9 complementary methods**:

- **Fst outlier** — population-based differentiation
- **PCAdapt** — PC-based Mahalanobis outliers
- **Extended ROH** — with FROH and length classes
- **Tajima's D** — sliding window neutrality test
- **iHS-like** — extended homozygosity
- **BayeScan-like** — Fst decomposition (α coefficient)
- **LFMM-like** — Genotype-Environment Association
- **Selective sweep (CLR)** — composite likelihood ratio
- **Combined outlier** — consensus across methods with overlap matrix
    """)

with fc2:
    st.markdown("""
### 🧩 Advanced Population Structure

- **CLUMPP-like alignment** — Hungarian algorithm for consensus Q-matrix
- **Evanno's ΔK** — automatic optimal K selection
- **Multiple K metrics** — BIC + AIC + ΔK
- **Distruct-style plots** — with population separator lines
- **Bayesian GMM** — fastStructure alternative
- **Confusion matrix** — cluster vs true population

### 🌳 Phylogenetics with Bootstrap
- **Real ML** — JC69 & Kimura K2P substitution models
- **True Fitch parsimony** — character-based algorithm
- **Bootstrap support** — resampling with 50-500 replicates
- **Newick with annotations** — importable to iTOL/FigTree
    """)


# ═══════════════════════════════════════════
# WORKFLOW
# ═══════════════════════════════════════════
st.markdown("---")
st.header("🚀 Recommended Workflow")

st.markdown("""
Follow this workflow for a complete population genomics analysis:
""")

workflow_steps = [
    ("1", "📁 Upload data",
     "Upload genotype file (Numeric/HapMap/VCF) + optional metadata"),
    ("2", "🧹 Quality Control",
     "Filter markers by MAF/missing/HWE, filter samples by missing rate"),
    ("3", "🧬 Explore diversity",
     "Compute per-marker stats, per-population diversity indices"),
    ("4", "🧩 Population structure",
     "PCA + STRUCTURE to identify populations and admixture"),
    ("5", "🌳 Phylogeny",
     "Build phylogenetic tree with bootstrap support"),
    ("6", "🌍 Geographic patterns",
     "Fst, IBD, IBE analyses with metadata"),
    ("7", "🔗 LD & selection",
     "LD decay, selection detection (9 methods)"),
    ("8", "🤖 Machine learning",
     "Classification, feature selection for GWAS candidates"),
    ("9", "📑 Report & export",
     "Generate comprehensive report, export all results"),
]

for step_num, step_title, step_desc in workflow_steps:
    st.markdown(
        f'<div style="display:flex; align-items:center; padding:0.5rem 0;">'
        f'<span class="workflow-step">{step_num}</span>'
        f'<span><b>{step_title}</b> — {step_desc}</span></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════
# INPUT FORMATS
# ═══════════════════════════════════════════
st.markdown("---")
st.header("📄 Input File Formats")

fmt_tabs = st.tabs(["Numeric (0/1/2)", "HapMap", "VCF", "Metadata"])

with fmt_tabs[0]:
    st.markdown("""
### Numeric Genotype Format
Rows = samples, Columns = markers, Values = allele dosage (0, 1, 2):
    """)
    st.code("""Sample_ID,SNP_1,SNP_2,SNP_3,SNP_4,SNP_5
Sample_001,0,1,2,0,1
Sample_002,1,1,0,2,0
Sample_003,2,0,1,1,2
...""", language="csv")
    st.markdown("""
- **0** = homozygous reference (e.g., AA)
- **1** = heterozygous (e.g., AT)
- **2** = homozygous alternate (e.g., TT)
- Missing values as empty, `NA`, or `NaN`
- Supported extensions: `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`, `.gz`
    """)

with fmt_tabs[1]:
    st.markdown("""
### HapMap Format
Standard hmp.txt with sample columns after metadata:
    """)
    st.code("""rs#	alleles	chrom	pos	strand	assembly#	center	protLSID	assayLSID	panelLSID	QCcode	Sample_001	Sample_002	Sample_003
SNP_1	A/T	1	1000	+	NA	NA	NA	NA	NA	NA	AA	AT	TT
SNP_2	G/C	1	2000	+	NA	NA	NA	NA	NA	NA	GG	GC	CC
...""", language="text")
    st.markdown("""
- Tab-separated
- First 11 columns = marker metadata
- Remaining columns = samples (genotype codes like `AA`, `AT`, `TT`)
- Missing values: `NN`, `NA`, `--`, `??`, `.`
- Supported: `.hmp.txt`, `.txt`, `.tsv`, `.gz`
    """)

with fmt_tabs[2]:
    st.markdown("""
### VCF Format
Standard VCF v4.x with GT field:
    """)
    st.code("""##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample_001	Sample_002	Sample_003
1	1000	SNP_1	A	T	.	PASS	.	GT	0/0	0/1	1/1
1	2000	SNP_2	G	C	.	PASS	.	GT	0/0	1/1	0/1
...""", language="text")
    st.markdown("""
- Tab-separated
- Supports phased (`|`) and unphased (`/`) genotypes
- Multi-allelic sites: alt alleles counted (0/0=0, 0/1=1, 1/1=2, 1/2=2)
- Missing: `./.` or `.`
- Supported: `.vcf`, `.vcf.gz`
    """)

with fmt_tabs[3]:
    st.markdown("""
### Metadata Format (CSV/Excel)
Provide sample information for population-based analyses:
    """)
    st.code("""Sample_ID,Species,Population,Origin,Latitude,Longitude,Rainfall,Yield
Sample_001,Wheat,Pop_A,Iran,35.5,52.3,250,45.2
Sample_002,Wheat,Pop_A,Iran,34.9,51.8,240,47.1
Sample_003,Wheat,Pop_B,Turkey,39.9,32.8,400,52.3
Sample_004,Wheat,Pop_B,Turkey,40.2,33.1,410,50.8
...""", language="csv")
    st.markdown("""
**Recommended columns:**
- `Sample_ID` — must match genotype sample IDs
- `Species` / `Population` / `Group` — for grouping analyses
- `Origin` / `Country` — for geographic analyses
- `Latitude` / `Longitude` — for IBD, maps
- Environmental variables — for IBE, LFMM
- Phenotypic traits — for GWAS, feature selection

Supported: `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`
    """)


# ═══════════════════════════════════════════
# TIPS & TRICKS
# ═══════════════════════════════════════════
st.markdown("---")
st.header("💡 Tips for Best Results")

tips_cols = st.columns(3)

with tips_cols[0]:
    st.info("""
**Sample size**
- N ≥ 50 for most analyses
- N ≥ 100 for STRUCTURE
- N ≥ 200 for GWAS-like analyses
    """)

with tips_cols[1]:
    st.info("""
**Marker density**
- ≥ 500 markers for basic analyses
- ≥ 5,000 for LD & selection
- ≥ 50,000 for genome-wide scans
    """)

with tips_cols[2]:
    st.info("""
**Quality thresholds**
- MAF ≥ 0.05
- Missing per marker ≤ 20%
- Missing per sample ≤ 30%
- HWE p ≥ 10⁻³ (for QC)
    """)


# ═══════════════════════════════════════════
# CITATION / REFERENCES
# ═══════════════════════════════════════════
st.markdown("---")
st.header("📚 Methods & References")

with st.expander("View statistical methods used"):
    st.markdown("""
### Population Genetics
- **Fst (Weir & Cockerham)** — Weir & Cockerham (1984)
- **Nei's Gst** — Nei (1973)
- **Jost's DEST** — Jost (2008)
- **Nm (migrants)** — Wright (1931)
- **AMOVA** — Excoffier et al. (1992)

### Population Structure
- **PCA** — Standard eigendecomposition
- **STRUCTURE** — Pritchard et al. (2000) [GMM approximation]
- **Evanno's ΔK** — Evanno et al. (2005)
- **CLUMPP alignment** — Jakobsson & Rosenberg (2007) [Hungarian algorithm]

### Phylogenetics
- **Neighbor Joining** — Saitou & Nei (1987)
- **UPGMA** — Sokal & Michener (1958)
- **Jukes-Cantor 1969** — Jukes & Cantor (1969)
- **Kimura K2P** — Kimura (1980)
- **Fitch Parsimony** — Fitch (1971)

### Selection Detection
- **PCAdapt** — Duforet-Frebourg et al. (2016)
- **Tajima's D** — Tajima (1989)
- **BayeScan** — Foll & Gaggiotti (2008)
- **LFMM** — Frichot et al. (2013)
- **CLR sweep** — Nielsen et al. (2005)

### Linkage Disequilibrium
- **Hill & Weir decay** — Hill & Weir (1988)
- **Gabriel blocks** — Gabriel et al. (2002)

### Machine Learning
- **Lasso** — Tibshirani (1996)
- **Random Forest** — Breiman (2001)
- **XGBoost** — Chen & Guestrin (2016)
- **UMAP** — McInnes et al. (2018)
    """)


# ═══════════════════════════════════════════
# FAQ
# ═══════════════════════════════════════════
with st.expander("❓ Frequently Asked Questions"):
    st.markdown("""
**Q: My data is very large (>100k SNPs, >1000 samples). Will it work?**
A: Yes! The platform uses vectorized operations. However, some analyses (STRUCTURE, phylogenetics) may be slow. Consider filtering to informative markers first.

**Q: Can I use this for animal genomics?**
A: Absolutely! All methods work equally well for any diploid species.

**Q: Are the results identical to PLINK / STRUCTURE / etc.?**
A: The core statistics (Fst, PCA, etc.) are identical. Some heuristics (e.g., STRUCTURE via GMM) are approximations for speed but validated to give similar biological conclusions.

**Q: How do I cite this platform?**
A: Please cite the underlying methods (see References tab). The platform itself can be cited as a tool used for the analysis.

**Q: My session was lost when I refreshed. Why?**
A: Streamlit stores session data in memory. Refreshing clears it. Re-upload your files.

**Q: Can I download all results at once?**
A: Yes! Go to **💾 Export Results** for a ZIP bundle of all outputs.
    """)


# ═══════════════════════════════════════════
# QUICK START
# ═══════════════════════════════════════════
st.markdown("---")
st.header("🚀 Quick Start")

qs1, qs2 = st.columns(2)
with qs1:
    st.success("""
### 🆕 New Users
1. Click **📁 Upload & Metadata** in sidebar
2. Upload your genotype file
3. Optionally upload metadata
4. Explore each module in order
    """)

with qs2:
    st.info("""
### 🔬 Experienced Users
- Jump directly to your analysis of interest
- Use **Machine Learning → Selection Detection** for candidate gene discovery
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