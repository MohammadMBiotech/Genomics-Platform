"""
utils.py — Utility helpers for the Interactive Population Genomics Platform.

Contains:
  - Session state management
  - Genotype matrix operations (allele freqs, MAF, He, Ho, Fis, PIC)
  - Missing data handling (imputation, filtering)
  - Distance matrices (Hamming, IBS, genetic distances)
  - Population genetics statistics
  - File I/O helpers
  - Statistical utilities (multiple testing correction, normalization)
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


# ═══════════════════════════════════════════
# SESSION STATE MANAGEMENT
# ═══════════════════════════════════════════
def get_geno_from_session(warn_if_missing=True):
    """
    Retrieve genotype matrix from session state.

    Args:
        warn_if_missing: If True, display warning when no data loaded

    Returns:
        tuple: (genotype_df, marker_info_df) or (None, None)
    """
    geno = st.session_state.get("genotype_matrix", None)
    marker_info = st.session_state.get("marker_info", None)
    if geno is None and warn_if_missing:
        st.warning(
            "⚠️ No genotype data loaded. Please go to "
            "**📁 Upload & Metadata** first."
        )
    return geno, marker_info


def get_meta_from_session():
    """Retrieve metadata DataFrame from session state (or None)."""
    return st.session_state.get("metadata", None)


def clear_session_data():
    """Clear all data-related session state."""
    keys_to_clear = ["genotype_matrix", "marker_info", "metadata"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def store_result_in_session(key, value):
    """Store any analysis result in session state for cross-page access."""
    st.session_state[key] = value


def get_result_from_session(key, default=None):
    """Retrieve stored result from session state."""
    return st.session_state.get(key, default)


# ═══════════════════════════════════════════
# DOWNLOAD HELPERS
# ═══════════════════════════════════════════
def download_plotly_html(fig, file_name="plot.html",
                          label="📥 Download Plot (HTML)", key=None):
    """
    Create a Streamlit download button for a Plotly figure.

    Args:
        fig: Plotly figure object
        file_name: Output filename
        label: Button label
        key: Unique streamlit widget key
    """
    if fig is None:
        return
    try:
        html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
        st.download_button(
            label, html_str, file_name, "text/html", key=key,
        )
    except Exception as e:
        st.error(f"Failed to prepare plot download: {e}")


def download_dataframe(df, file_name="data.csv",
                        label="📥 Download CSV", index=False, key=None):
    """
    Create a Streamlit download button for a DataFrame.

    Args:
        df: pandas DataFrame
        file_name: Output filename
        label: Button label
        index: Whether to include the DataFrame index
        key: Unique streamlit widget key
    """
    if df is None or len(df) == 0:
        return
    try:
        csv_str = df.to_csv(index=index)
        st.download_button(label, csv_str, file_name, "text/csv", key=key)
    except Exception as e:
        st.error(f"Failed to prepare CSV download: {e}")


def download_excel(dfs_dict, file_name="results.xlsx",
                    label="📥 Download Excel", key=None):
    """
    Create a Streamlit download button for multiple DataFrames as Excel.

    Args:
        dfs_dict: Dict {sheet_name: DataFrame}
        file_name: Output filename
        label: Button label
        key: Unique widget key
    """
    if not dfs_dict:
        return
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for sheet_name, df in dfs_dict.items():
                # Sheet names max 31 chars
                clean_name = str(sheet_name)[:31]
                df.to_excel(writer, sheet_name=clean_name, index=False)
        buf.seek(0)
        st.download_button(
            label, buf, file_name,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key,
        )
    except Exception as e:
        st.error(f"Failed to prepare Excel: {e}")


# ═══════════════════════════════════════════
# MISSING DATA HANDLING
# ═══════════════════════════════════════════
def impute_missing(geno, method="mean"):
    """
    Impute missing values in a genotype matrix.

    Args:
        geno: DataFrame of shape (n_samples, n_markers) with 0/1/2 codes
        method: One of 'mean', 'median', 'zero', 'mode', 'drop'

    Returns:
        Imputed DataFrame
    """
    if geno is None or len(geno) == 0:
        return geno

    if method == "mean":
        # Column-wise mean (per marker)
        col_means = geno.mean()
        return geno.fillna(col_means)
    elif method == "median":
        col_medians = geno.median()
        return geno.fillna(col_medians)
    elif method == "zero":
        return geno.fillna(0)
    elif method == "mode":
        # Use most common value per marker
        modes = geno.mode()
        if len(modes) == 0:
            return geno.fillna(0)
        mode_vals = modes.iloc[0]
        return geno.fillna(mode_vals)
    elif method == "drop":
        return geno.dropna(axis=1, how="any")
    else:
        return geno.fillna(geno.mean())


def filter_by_missing(geno, max_missing_marker=0.2, max_missing_sample=0.3,
                       marker_info=None):
    """
    Filter markers and samples by missing rate.

    Args:
        geno: Genotype DataFrame
        max_missing_marker: Max allowed missing rate per marker
        max_missing_sample: Max allowed missing rate per sample
        marker_info: Optional marker info DataFrame (subset accordingly)

    Returns:
        tuple: (filtered_geno, filtered_marker_info, report)
    """
    initial_shape = geno.shape

    # Filter markers
    miss_per_marker = geno.isna().mean(axis=0)
    keep_markers = miss_per_marker <= max_missing_marker
    geno_filt = geno.loc[:, keep_markers]

    # Filter samples
    miss_per_sample = geno_filt.isna().mean(axis=1)
    keep_samples = miss_per_sample <= max_missing_sample
    geno_filt = geno_filt.loc[keep_samples]

    # Filter marker info
    if marker_info is not None:
        marker_info_filt = marker_info[
            marker_info["Marker"].astype(str).isin(
                geno_filt.columns.astype(str))
        ].reset_index(drop=True)
    else:
        marker_info_filt = None

    report = {
        "initial_samples": initial_shape[0],
        "initial_markers": initial_shape[1],
        "final_samples": geno_filt.shape[0],
        "final_markers": geno_filt.shape[1],
        "removed_samples": initial_shape[0] - geno_filt.shape[0],
        "removed_markers": initial_shape[1] - geno_filt.shape[1],
    }

    return geno_filt, marker_info_filt, report


# ═══════════════════════════════════════════
# ALLELE FREQUENCY & DIVERSITY
# ═══════════════════════════════════════════
def calc_allele_freq(geno):
    """
    Calculate allele frequencies from 0/1/2 coded genotype matrix.

    Returns:
        tuple: (p, q) — allele frequencies as Series indexed by marker
        p = frequency of alt allele (0 = homozygous ref → adds 0)
        q = 1 - p
    """
    p = geno.mean(axis=0) / 2.0
    q = 1 - p
    return p, q


def calc_maf(geno):
    """
    Calculate Minor Allele Frequency per marker.

    Returns:
        pandas Series of MAF values indexed by marker
    """
    p, q = calc_allele_freq(geno)
    maf = pd.Series(np.minimum(p.values, q.values), index=geno.columns)
    return maf


def calc_missing_rate(geno, axis=0):
    """
    Calculate missing rate.

    Args:
        axis: 0 = per marker, 1 = per sample

    Returns:
        pandas Series of missing rates
    """
    return geno.isna().mean(axis=axis)


def calc_het_obs(geno):
    """
    Calculate observed heterozygosity per marker.
    Ho = proportion of samples with genotype 1 (heterozygous).

    Returns:
        pandas Series of Ho values
    """
    n_valid = geno.notna().sum(axis=0)
    n_het = (geno == 1).sum(axis=0)
    ho = pd.Series(
        np.where(n_valid > 0, n_het / n_valid, np.nan),
        index=geno.columns,
    )
    return ho


def calc_het_exp(geno):
    """
    Calculate expected heterozygosity per marker: He = 2pq

    Returns:
        pandas Series of He values
    """
    p, q = calc_allele_freq(geno)
    return 2 * p * q


def calc_pic(geno):
    """
    Calculate Polymorphism Information Content per marker.
    PIC = 1 - Σp²ᵢ - Σ Σ 2p²ᵢp²ⱼ
    For biallelic: PIC = 1 - p² - q² - 2p²q²

    Returns:
        pandas Series of PIC values
    """
    p, q = calc_allele_freq(geno)
    pic = 1 - p**2 - q**2 - 2 * p**2 * q**2
    return pic


def calc_fis(geno):
    """
    Calculate inbreeding coefficient Fis per marker.
    Fis = 1 - (Ho / He)

    Returns:
        pandas Series of Fis values (NaN where He = 0)
    """
    ho = calc_het_obs(geno)
    he = calc_het_exp(geno)
    with np.errstate(divide="ignore", invalid="ignore"):
        fis = 1 - (ho / he.replace(0, np.nan))
    return fis


def calc_shannon_diversity(geno):
    """
    Calculate Shannon's diversity index per marker.
    H = -Σ pᵢ ln(pᵢ)

    Returns:
        pandas Series of Shannon indices
    """
    p, q = calc_allele_freq(geno)
    p_safe = p.replace(0, np.nan)
    q_safe = q.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(p_safe * np.log(p_safe) + q_safe * np.log(q_safe))
    return H


def calc_effective_alleles(geno):
    """
    Calculate effective number of alleles per marker.
    Ne = 1 / Σpᵢ² = 1 / (p² + q²)

    Returns:
        pandas Series of Ne values
    """
    p, q = calc_allele_freq(geno)
    denom = (p ** 2 + q ** 2).replace(0, np.nan)
    return 1.0 / denom


# ═══════════════════════════════════════════
# POPULATION-LEVEL STATISTICS
# ═══════════════════════════════════════════
def calc_pairwise_fst(geno1, geno2, method="Weir-Cockerham"):
    """
    Calculate pairwise Fst between two populations.

    Args:
        geno1, geno2: Genotype DataFrames for each population
        method: 'Weir-Cockerham' (default) or 'Nei'

    Returns:
        float: mean Fst across markers
    """
    p1, _ = calc_allele_freq(geno1)
    p2, _ = calc_allele_freq(geno2)

    pbar = (p1 + p2) / 2
    Ht = 2 * pbar * (1 - pbar)
    Hs = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2

    with np.errstate(divide="ignore", invalid="ignore"):
        fst = (Ht - Hs) / Ht.replace(0, np.nan)

    return float(fst.mean())


def calc_pairwise_dest(geno1, geno2):
    """
    Calculate Jost's D (DEST) between two populations.
    D = ((Ht - Hs) / (1 - Hs)) × (n / (n - 1))

    Returns:
        float: mean DEST across markers
    """
    p1, _ = calc_allele_freq(geno1)
    p2, _ = calc_allele_freq(geno2)

    n = 2  # two populations
    pbar = (p1 + p2) / 2
    Ht = 2 * pbar * (1 - pbar)
    Hs = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2

    with np.errstate(divide="ignore", invalid="ignore"):
        dest = ((Ht - Hs) / (1 - Hs).replace(0, np.nan)) * (n / (n - 1))

    return float(dest.mean())


def calc_nm_from_fst(fst):
    """
    Estimate number of migrants per generation from Fst.
    Wright's formula: Nm = (1 - Fst) / (4 × Fst)

    Returns:
        float: Nm (inf if fst ≤ 0, 0 if fst ≥ 1)
    """
    if fst <= 0 or np.isnan(fst):
        return np.inf
    if fst >= 1:
        return 0.0
    return (1 - fst) / (4 * fst)


# ═══════════════════════════════════════════
# HARDY-WEINBERG EQUILIBRIUM
# ═══════════════════════════════════════════
def hwe_chi2_test(geno_column):
    """
    Chi-square test for Hardy-Weinberg equilibrium at one marker.

    Args:
        geno_column: pandas Series of 0/1/2 genotypes for one marker

    Returns:
        float: p-value (NaN if invalid)
    """
    from scipy.stats import chisquare

    counts = geno_column.dropna()
    n = len(counts)
    if n == 0:
        return np.nan

    n0 = (counts == 0).sum()
    n1 = (counts == 1).sum()
    n2 = (counts == 2).sum()

    p = (2 * n0 + n1) / (2 * n)
    q = 1 - p

    exp0 = p ** 2 * n
    exp1 = 2 * p * q * n
    exp2 = q ** 2 * n

    observed = [n0, n1, n2]
    expected = [max(exp0, 1e-10), max(exp1, 1e-10), max(exp2, 1e-10)]

    try:
        _, pval = chisquare(observed, f_exp=expected)
        return pval
    except Exception:
        return np.nan


# ═══════════════════════════════════════════
# DISTANCE MATRICES
# ═══════════════════════════════════════════
def compute_ibs_matrix(geno):
    """
    Compute Identity By State (IBS) similarity matrix.
    IBS(i,j) = mean over markers of (2 - |g_i - g_j|) / 2

    Returns:
        numpy 2D array (n_samples × n_samples)
    """
    geno_np = geno.fillna(1).values.astype(float)
    n = geno_np.shape[0]
    IBS = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            diff = np.abs(geno_np[i] - geno_np[j])
            ibs = np.mean((2 - diff) / 2)
            IBS[i, j] = IBS[j, i] = ibs
    return IBS


def compute_kinship_vanraden(geno):
    """
    Compute VanRaden (2008) genomic relationship matrix (GRM).
    G = (M - P)(M - P)' / (2 * Σ p(1-p))

    Returns:
        numpy 2D array (n_samples × n_samples)
    """
    M = geno.fillna(geno.mean()).values.astype(float)
    p = M.mean(axis=0) / 2
    P = 2 * p
    Z = M - P
    denom = 2 * np.sum(p * (1 - p))
    if denom < 1e-9:
        denom = 1e-9
    return Z @ Z.T / denom


def compute_distance_matrix(geno, metric="euclidean"):
    """
    Compute pairwise distance matrix between samples.

    Args:
        geno: Genotype DataFrame
        metric: any scipy pdist metric

    Returns:
        numpy 2D array
    """
    geno_imp = impute_missing(geno, "mean")
    D = squareform(pdist(geno_imp.values, metric=metric))
    return D


# ═══════════════════════════════════════════
# STATISTICAL UTILITIES
# ═══════════════════════════════════════════
def bonferroni_correction(pvals, alpha=0.05):
    """
    Bonferroni multiple testing correction.

    Returns:
        tuple: (adjusted_pvals, significant_mask)
    """
    pvals = np.asarray(pvals)
    n = len(pvals)
    adjusted = np.minimum(pvals * n, 1.0)
    significant = adjusted < alpha
    return adjusted, significant


def fdr_bh_correction(pvals, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Returns:
        tuple: (adjusted_pvals, significant_mask)
    """
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adjusted = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonicity
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    # Un-order
    final = np.empty(n)
    final[order] = adjusted
    significant = final < alpha
    return final, significant


def quantile_normalize(df):
    """
    Quantile normalization of DataFrame columns.

    Returns:
        Normalized DataFrame with same shape
    """
    rank_mean = df.stack().groupby(df.rank(method="first").stack().astype(int)).mean()
    return df.rank(method="min").stack().astype(int).map(rank_mean).unstack()


def standardize_matrix(X, axis=0):
    """
    Z-score standardize matrix along specified axis.

    Args:
        X: numpy array or DataFrame
        axis: 0 (per column) or 1 (per row)

    Returns:
        Standardized array
    """
    if hasattr(X, "values"):
        X_np = X.values.astype(float)
    else:
        X_np = np.asarray(X, dtype=float)

    mean = np.nanmean(X_np, axis=axis, keepdims=True)
    std = np.nanstd(X_np, axis=axis, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)  # avoid division by zero

    return (X_np - mean) / std


# ═══════════════════════════════════════════
# LINKAGE DISEQUILIBRIUM (utility for other modules)
# ═══════════════════════════════════════════
def compute_r2(g1, g2):
    """
    Compute r² between two SNP columns using correlation.

    Args:
        g1, g2: 1D numpy arrays (n_samples,)

    Returns:
        float: r² value (NaN if invalid)
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    mask = ~(np.isnan(g1) | np.isnan(g2))
    if mask.sum() < 5:
        return np.nan
    a, b = g1[mask], g2[mask]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    if np.isnan(r):
        return np.nan
    return r * r


def compute_dprime(g1, g2):
    """
    Compute D' between two SNP columns.

    Args:
        g1, g2: 1D numpy arrays (n_samples,)

    Returns:
        float: |D'| value (NaN if invalid)
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    mask = ~(np.isnan(g1) | np.isnan(g2))
    if mask.sum() < 5:
        return np.nan
    a, b = g1[mask], g2[mask]

    pA = a.mean() / 2.0
    pB = b.mean() / 2.0

    if pA in (0, 1) or pB in (0, 1):
        return np.nan

    D = np.cov(a, b)[0, 1] / 2.0
    if D >= 0:
        Dmax = min(pA * (1 - pB), (1 - pA) * pB)
    else:
        Dmax = min(pA * pB, (1 - pA) * (1 - pB))

    if Dmax < 1e-9:
        return np.nan
    return abs(D / Dmax)


# ═══════════════════════════════════════════
# CHROMOSOME SORTING
# ═══════════════════════════════════════════
def chromosome_sort_key(chrom_name):
    """
    Sort key for chromosome names.
    Handles numeric (1, 2, ..., 22), sex (X, Y), and mitochondrial (MT).
    """
    x = str(chrom_name).replace("chr", "").replace("Chr", "")
    if x.isdigit():
        return (0, int(x))
    order_map = {"X": 100, "Y": 101, "M": 102, "MT": 102, "NA": 999}
    return (1, order_map.get(x.upper(), 999))


def sort_chromosomes(chrom_list):
    """Sort a list of chromosome names properly."""
    return sorted(set(chrom_list), key=chromosome_sort_key)


# ═══════════════════════════════════════════
# METADATA HELPERS
# ═══════════════════════════════════════════
def build_sample_pop_map(meta, sample_col, pop_col):
    """
    Build a dictionary mapping sample IDs to population labels.

    Args:
        meta: metadata DataFrame
        sample_col: name of sample ID column
        pop_col: name of population column

    Returns:
        dict: {sample_id (str): population (str)}
    """
    if meta is None or sample_col not in meta.columns or pop_col not in meta.columns:
        return {}
    return dict(zip(
        meta[sample_col].astype(str),
        meta[pop_col].astype(str),
    ))


def get_samples_by_population(pop_map, sample_index, pop_name):
    """
    Get list of sample IDs belonging to a specific population.

    Args:
        pop_map: dict from build_sample_pop_map
        sample_index: iterable of sample IDs (e.g., geno.index)
        pop_name: target population

    Returns:
        list: sample IDs in that population
    """
    return [s for s in sample_index if pop_map.get(str(s)) == pop_name]


# ═══════════════════════════════════════════
# HAVERSINE DISTANCE
# ═══════════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two lat/lon points in km.
    """
    R = 6371.0  # Earth radius km
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def haversine_matrix(lats, lons):
    """
    Build pairwise haversine distance matrix from arrays of lat/lon.

    Returns:
        numpy 2D array (n × n) of distances in km
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            D[i, j] = D[j, i] = d
    return D


# ═══════════════════════════════════════════
# LOGGING / INFO
# ═══════════════════════════════════════════
def display_data_summary(geno, marker_info=None, meta=None):
    """
    Display a compact summary of loaded data in the sidebar.
    """
    with st.sidebar:
        st.markdown("### 📊 Loaded Data")
        if geno is not None:
            st.metric("Samples", f"{geno.shape[0]:,}")
            st.metric("Markers", f"{geno.shape[1]:,}")
            missing = geno.isna().sum().sum() / geno.size * 100
            st.metric("Missing %", f"{missing:.1f}%")
        if marker_info is not None and "Chrom" in marker_info.columns:
            n_chr = marker_info["Chrom"].nunique()
            st.metric("Chromosomes", n_chr)
        if meta is not None:
            st.metric("Metadata rows", meta.shape[0])