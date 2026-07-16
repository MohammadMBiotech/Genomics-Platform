"""
parsers.py — Parse genotype data from Numeric (0/1/2), HapMap, and VCF formats
into a unified numeric matrix (samples × markers) with marker metadata.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st


def detect_format(uploaded_file):
    """Detect genotype file format from content."""
    uploaded_file.seek(0)
    first_lines = []
    for i, line in enumerate(uploaded_file):
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="ignore")
        first_lines.append(line.strip())
        if i >= 20:
            break
    uploaded_file.seek(0)

    header = first_lines[0] if first_lines else ""

    # VCF detection
    if header.startswith("##fileformat=VCF") or header.startswith("#CHROM"):
        return "VCF"
    for line in first_lines:
        if line.startswith("##fileformat=VCF") or line.startswith("#CHROM"):
            return "VCF"

    # HapMap detection
    hapmap_cols = ["rs#", "alleles", "chrom", "pos", "strand",
                   "assembly#", "center", "protLSID", "assayLSID",
                   "panelLSID", "QCcode"]
    if any(col in header.lower() for col in [c.lower() for c in hapmap_cols[:4]]):
        return "HapMap"

    # Numeric detection (default)
    return "Numeric"


def parse_numeric(uploaded_file, has_header=True, sample_col=None):
    """
    Parse numeric genotype file (0, 1, 2 coded).
    Rows = samples, Columns = markers (or vice versa).
    """
    uploaded_file.seek(0)
    if has_header:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    else:
        df = pd.read_csv(uploaded_file, sep=None, engine="python", header=None)

    if sample_col and sample_col in df.columns:
        df = df.set_index(sample_col)

    # Ensure numeric
    geno = df.select_dtypes(include=[np.number])

    # Check orientation: if rows >> cols, likely markers in rows → transpose
    if geno.shape[0] > geno.shape[1] * 3:
        st.info("Auto-transposed: detected markers in rows, samples in columns.")
        geno = geno.T

    # Marker metadata (minimal)
    marker_info = pd.DataFrame({
        "Marker": geno.columns.astype(str),
        "Chrom": "NA",
        "Pos": np.arange(len(geno.columns)),
    })

    geno.columns = marker_info["Marker"].values
    geno.index = geno.index.astype(str)

    return geno, marker_info


def parse_hapmap(uploaded_file):
    """Parse HapMap format genotype file."""
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep="\t")

    # Standard HapMap columns
    meta_cols = ["rs#", "alleles", "chrom", "pos", "strand",
                 "assembly#", "center", "protLSID", "assayLSID",
                 "panelLSID", "QCcode"]
    meta_cols_present = [c for c in meta_cols if c in df.columns]
    sample_cols = [c for c in df.columns if c not in meta_cols_present]

    if len(sample_cols) == 0:
        st.error("No sample columns found in HapMap file.")
        return None, None

    marker_info = df[meta_cols_present].copy()
    marker_info = marker_info.rename(columns={"rs#": "Marker", "chrom": "Chrom", "pos": "Pos"})
    if "Marker" not in marker_info.columns:
        marker_info["Marker"] = [f"SNP_{i}" for i in range(len(marker_info))]

    # Convert genotypes to numeric
    geno_raw = df[sample_cols].copy()

    # Build allele lookup from 'alleles' column
    if "alleles" in df.columns:
        alleles = df["alleles"].str.split("/", expand=True)
        ref_alleles = alleles[0] if alleles.shape[1] >= 1 else pd.Series(["A"] * len(df))
        alt_alleles = alleles[1] if alleles.shape[1] >= 2 else pd.Series(["T"] * len(df))
    else:
        ref_alleles = pd.Series(["A"] * len(df))
        alt_alleles = pd.Series(["T"] * len(df))

    # Convert each genotype call to 0/1/2
    geno_numeric = pd.DataFrame(index=sample_cols, columns=marker_info["Marker"].values, dtype=float)

    for i in range(len(df)):
        ref = ref_alleles.iloc[i]
        alt = alt_alleles.iloc[i]
        marker_name = marker_info["Marker"].iloc[i]

        for sample in sample_cols:
            call = str(geno_raw.iloc[i][sample]).strip()
            if len(call) == 2:
                a1, a2 = call[0], call[1]
                count = 0
                if a1 == alt:
                    count += 1
                if a2 == alt:
                    count += 1
                if a1 not in [ref, alt] or a2 not in [ref, alt]:
                    geno_numeric.loc[sample, marker_name] = np.nan
                else:
                    geno_numeric.loc[sample, marker_name] = count
            elif call in ["NN", "NA", "N", "--", "??", "."]:
                geno_numeric.loc[sample, marker_name] = np.nan
            else:
                geno_numeric.loc[sample, marker_name] = np.nan

    geno_numeric = geno_numeric.astype(float)
    return geno_numeric, marker_info


def parse_vcf(uploaded_file):
    """Parse VCF format genotype file (simplified)."""
    uploaded_file.seek(0)
    lines = []
    header_line = None

    for line in uploaded_file:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="ignore")
        line = line.strip()
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            header_line = line
            continue
        if header_line:
            lines.append(line)

    if header_line is None:
        st.error("VCF file does not contain a valid #CHROM header line.")
        return None, None

    cols = header_line.lstrip("#").split("\t")
    data = [l.split("\t") for l in lines if l]

    df = pd.DataFrame(data, columns=cols)

    # Standard VCF columns
    vcf_meta = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    sample_cols = [c for c in df.columns if c not in vcf_meta]

    if len(sample_cols) == 0:
        st.error("No sample columns found in VCF file.")
        return None, None

    marker_info = pd.DataFrame({
        "Marker": df["ID"].values if "ID" in df.columns else [f"SNP_{i}" for i in range(len(df))],
        "Chrom": df["CHROM"].values if "CHROM" in df.columns else "NA",
        "Pos": pd.to_numeric(df["POS"], errors="coerce").values if "POS" in df.columns else np.arange(len(df)),
        "REF": df["REF"].values if "REF" in df.columns else "N",
        "ALT": df["ALT"].values if "ALT" in df.columns else "N",
    })

    # Replace '.' marker IDs
    mask = (marker_info["Marker"] == ".") | (marker_info["Marker"].isna())
    marker_info.loc[mask, "Marker"] = [f"SNP_{i}" for i in range(mask.sum())]

    # Parse GT field
    format_col = df["FORMAT"].values if "FORMAT" in df.columns else None
    geno_numeric = pd.DataFrame(index=sample_cols, columns=marker_info["Marker"].values, dtype=float)

    for i in range(len(df)):
        fmt = format_col[i].split(":") if format_col is not None else ["GT"]
        gt_idx = fmt.index("GT") if "GT" in fmt else 0

        marker_name = marker_info["Marker"].iloc[i]
        for sample in sample_cols:
            fields = str(df.iloc[i][sample]).split(":")
            if gt_idx < len(fields):
                gt = fields[gt_idx]
                gt = gt.replace("|", "/")
                if gt in ["./.", ".", "./0", "0/."]:
                    geno_numeric.loc[sample, marker_name] = np.nan
                else:
                    alleles = gt.split("/")
                    try:
                        geno_numeric.loc[sample, marker_name] = sum(int(a) for a in alleles if a != ".")
                    except ValueError:
                        geno_numeric.loc[sample, marker_name] = np.nan
            else:
                geno_numeric.loc[sample, marker_name] = np.nan

    geno_numeric = geno_numeric.astype(float)
    return geno_numeric, marker_info


def load_genotype_data(key_prefix="geno"):
    """
    Main entry point: file uploader + format detection + parsing.
    Returns (genotype_matrix, marker_info) or (None, None).
    Stores results in st.session_state for cross-page access.
    """
    uploaded = st.file_uploader(
        "Upload Genotype File (Numeric 0/1/2, HapMap, or VCF)",
        type=["csv", "tsv", "txt", "hmp", "vcf", "xlsx"],
        key=f"{key_prefix}_uploader",
    )

    if uploaded is None:
        # Check session state for previously loaded data
        if "genotype_matrix" in st.session_state and st.session_state["genotype_matrix"] is not None:
            return st.session_state["genotype_matrix"], st.session_state["marker_info"]
        return None, None

    fmt = detect_format(uploaded)
    st.info(f"🔍 Detected format: **{fmt}**")

    fmt_override = st.radio(
        "Confirm or override format:",
        ["Auto-detected: " + fmt, "Numeric (0/1/2)", "HapMap", "VCF"],
        index=0, horizontal=True, key=f"{key_prefix}_fmt",
    )

    if "Numeric" in fmt_override:
        fmt = "Numeric"
    elif "HapMap" in fmt_override:
        fmt = "HapMap"
    elif "VCF" in fmt_override:
        fmt = "VCF"

    with st.spinner(f"Parsing {fmt} file..."):
        if fmt == "Numeric":
            geno, marker_info = parse_numeric(uploaded)
        elif fmt == "HapMap":
            geno, marker_info = parse_hapmap(uploaded)
        elif fmt == "VCF":
            geno, marker_info = parse_vcf(uploaded)
        else:
            geno, marker_info = parse_numeric(uploaded)

    if geno is None:
        st.error("Failed to parse genotype file.")
        return None, None

    # Store in session state
    st.session_state["genotype_matrix"] = geno
    st.session_state["marker_info"] = marker_info

    st.success(f"✅ Loaded: **{geno.shape[0]}** samples × **{geno.shape[1]}** markers")
    return geno, marker_info


def load_metadata(key_prefix="meta"):
    """Load optional metadata CSV. Returns DataFrame or None."""
    uploaded = st.file_uploader(
        "Upload Metadata (CSV/Excel with Sample ID, Species, Origin, Group, etc.)",
        type=["csv", "tsv", "txt", "xlsx"],
        key=f"{key_prefix}_uploader",
    )

    if uploaded is None:
        if "metadata" in st.session_state and st.session_state["metadata"] is not None:
            return st.session_state["metadata"]
        return None

    fname = uploaded.name.lower()
    uploaded.seek(0)

    if fname.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(uploaded)
        if len(xls.sheet_names) > 1:
            sheet = st.selectbox("Select sheet", xls.sheet_names, key=f"{key_prefix}_sheet")
        else:
            sheet = xls.sheet_names[0]
        meta = pd.read_excel(uploaded, sheet_name=sheet)
    else:
        meta = pd.read_csv(uploaded, sep=None, engine="python")

    st.session_state["metadata"] = meta
    st.success(f"✅ Metadata loaded: **{meta.shape[0]}** rows × **{meta.shape[1]}** columns")
    return meta