"""
Utility helpers for the Population Genomics Platform.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


def get_geno_from_session():
    """Retrieve genotype matrix from session state."""
    geno = st.session_state.get("genotype_matrix", None)
    marker_info = st.session_state.get("marker_info", None)
    if geno is None:
        st.warning("⚠️ No genotype data loaded. Please go to **Upload & Metadata** first.")
    return geno, marker_info


def get_meta_from_session():
    """Retrieve metadata from session state."""
    return st.session_state.get("metadata", None)


def download_plotly_html(fig, file_name="plot.html", label="Download Plot (HTML)", key=None):
    if fig is None:
        return
    st.download_button(label, fig.to_html(include_plotlyjs="cdn", full_html=True),
                       file_name, "text/html", key=key)


def download_dataframe(df, file_name="data.csv", label="Download CSV", index=False, key=None):
    if df is None:
        return
    st.download_button(label, df.to_csv(index=index), file_name, "text/csv", key=key)


def impute_missing(geno, method="mean"):
    """Impute missing values in genotype matrix."""
    if method == "mean":
        return geno.fillna(geno.mean())
    elif method == "median":
        return geno.fillna(geno.median())
    elif method == "zero":
        return geno.fillna(0)
    elif method == "mode":
        return geno.fillna(geno.mode().iloc[0])
    else:
        return geno.dropna(axis=1)


def calc_allele_freq(geno):
    """Calculate allele frequencies from 0/1/2 coded genotype matrix."""
    p = geno.mean(axis=0) / 2.0  # frequency of alt allele
    q = 1 - p
    return p, q


def calc_maf(geno):
    """Calculate minor allele frequency for each marker."""
    p, q = calc_allele_freq(geno)
    maf = pd.Series(np.minimum(p, q), index=geno.columns)
    return maf


def calc_missing_rate(geno, axis=0):
    """Calculate missing rate per marker (axis=0) or per sample (axis=1)."""
    return geno.isna().mean(axis=axis)


def calc_het_obs(geno):
    """Calculate observed heterozygosity per marker (proportion of 1s)."""
    return (geno == 1).sum(axis=0) / geno.notna().sum(axis=0)


def calc_het_exp(geno):
    """Calculate expected heterozygosity per marker: 2pq."""
    p, q = calc_allele_freq(geno)
    return 2 * p * q


def calc_pic(geno):
    """Calculate Polymorphism Information Content per marker."""
    p, q = calc_allele_freq(geno)
    pic = 1 - p**2 - q**2 - 2 * p**2 * q**2
    return pic


def calc_fis(geno):
    """Calculate inbreeding coefficient Fis per marker."""
    ho = calc_het_obs(geno)
    he = calc_het_exp(geno)
    fis = 1 - (ho / he.replace(0, np.nan))
    return fis