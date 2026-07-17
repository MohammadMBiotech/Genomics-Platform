"""
Export Results — Publication Quality
─────────────────────────────────────
Comprehensive export module supporting multiple formats:
  - Numeric (CSV, TSV, Excel)
  - HapMap
  - VCF
  - PLINK (PED + MAP files)
  - STRUCTURE format
  - Excel multi-sheet analysis bundle
  - Compressed (gzip) versions
  - Full analysis ZIP bundle with README
  - Subset export (select samples/markers)
"""

import io
import gzip
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from utils import (
    get_geno_from_session, get_meta_from_session,
    calc_allele_freq, calc_maf, calc_het_obs, calc_het_exp,
    calc_pic, calc_fis, calc_missing_rate,
    calc_shannon_diversity, calc_effective_alleles,
    build_sample_pop_map,
    download_dataframe, download_excel,
)

st.title("💾 Export Results")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


# ═══════════════════════════════════════════
# METADATA CONFIGURATION
# ═══════════════════════════════════════════
pop_map = {}
if meta is not None:
    with st.expander("🔧 Metadata Configuration (optional)"):
        mc1, mc2 = st.columns(2)
        with mc1:
            sample_col = st.selectbox("Sample ID column",
                                        meta.columns.tolist(),
                                        key="ex_samcol")
        with mc2:
            pop_col_opt = st.selectbox("Population column",
                                         ["None"] + meta.columns.tolist(),
                                         key="ex_popcol")
            pop_col = None if pop_col_opt == "None" else pop_col_opt

        if pop_col:
            pop_map = build_sample_pop_map(meta, sample_col, pop_col)


# ═══════════════════════════════════════════
# SESSION SUMMARY
# ═══════════════════════════════════════════
st.subheader("📊 Current Data Summary")

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Samples", f"{geno.shape[0]:,}")
sc2.metric("Markers", f"{geno.shape[1]:,}")
sc3.metric("Missing %",
             f"{geno.isna().sum().sum() / geno.size * 100:.2f}%")
sc4.metric("Metadata", "✅ Yes" if meta is not None else "❌ No")


# ═══════════════════════════════════════════
# CONVERSION FUNCTIONS
# ═══════════════════════════════════════════
def numeric_to_hapmap(geno_df, marker_info_df=None):
    """Convert numeric genotype (0/1/2) to HapMap format."""
    n_markers = geno_df.shape[1]

    # Get REF/ALT from marker_info if available, else default A/T
    if marker_info_df is not None and \
        "REF" in marker_info_df.columns and \
        "ALT" in marker_info_df.columns:
        ref_alleles = marker_info_df["REF"].astype(str).values
        alt_alleles = marker_info_df["ALT"].astype(str).values
    else:
        ref_alleles = np.array(["A"] * n_markers)
        alt_alleles = np.array(["T"] * n_markers)

    alleles_col = [f"{r}/{a}" for r, a in zip(ref_alleles, alt_alleles)]

    hmp = pd.DataFrame({
        "rs#": geno_df.columns.astype(str),
        "alleles": alleles_col,
        "chrom": (marker_info_df["Chrom"].astype(str).values
                    if marker_info_df is not None
                    and "Chrom" in marker_info_df.columns
                    else ["NA"] * n_markers),
        "pos": (marker_info_df["Pos"].values
                  if marker_info_df is not None
                  and "Pos" in marker_info_df.columns
                  else np.arange(n_markers)),
        "strand": "+",
        "assembly#": "NA", "center": "NA",
        "protLSID": "NA", "assayLSID": "NA",
        "panelLSID": "NA", "QCcode": "NA",
    })

    # Convert numeric to calls
    calls_matrix = np.full((n_markers, len(geno_df.index)), "NN",
                             dtype=object)
    geno_values = geno_df.values

    for j in range(n_markers):
        ref = ref_alleles[j]
        alt = alt_alleles[j]
        for i in range(len(geno_df.index)):
            v = geno_values[i, j]
            if pd.isna(v):
                calls_matrix[j, i] = "NN"
            elif v == 0:
                calls_matrix[j, i] = ref + ref
            elif v == 1:
                calls_matrix[j, i] = ref + alt
            elif v == 2:
                calls_matrix[j, i] = alt + alt

    calls_df = pd.DataFrame(calls_matrix,
                              columns=geno_df.index.astype(str))
    result = pd.concat([hmp.reset_index(drop=True),
                          calls_df.reset_index(drop=True)], axis=1)
    return result


def numeric_to_vcf(geno_df, marker_info_df=None):
    """Convert numeric genotype to VCF format string."""
    lines = [
        "##fileformat=VCFv4.2",
        f"##source=Interactive_Population_Genomics_Platform",
        f"##fileDate={datetime.now().strftime('%Y%m%d')}",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    ]

    header = ("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(geno_df.index.astype(str)))
    lines.append(header)

    n_markers = geno_df.shape[1]

    if marker_info_df is not None:
        chr_arr = (marker_info_df["Chrom"].astype(str).values
                     if "Chrom" in marker_info_df.columns
                     else ["NA"] * n_markers)
        pos_arr = (marker_info_df["Pos"].values
                     if "Pos" in marker_info_df.columns
                     else np.arange(n_markers))
        ref_arr = (marker_info_df["REF"].astype(str).values
                     if "REF" in marker_info_df.columns
                     else ["A"] * n_markers)
        alt_arr = (marker_info_df["ALT"].astype(str).values
                     if "ALT" in marker_info_df.columns
                     else ["T"] * n_markers)
    else:
        chr_arr = ["NA"] * n_markers
        pos_arr = np.arange(n_markers)
        ref_arr = ["A"] * n_markers
        alt_arr = ["T"] * n_markers

    geno_values = geno_df.values

    def _num_to_gt(v):
        if pd.isna(v):
            return "./."
        v_int = int(round(v))
        if v_int == 0:
            return "0/0"
        elif v_int == 1:
            return "0/1"
        elif v_int == 2:
            return "1/1"
        return "./."

    for j, snp in enumerate(geno_df.columns):
        pos_val = int(pos_arr[j]) if not pd.isna(pos_arr[j]) else 0
        row = [str(chr_arr[j]), str(pos_val), str(snp),
                ref_arr[j], alt_arr[j], ".", "PASS", ".", "GT"]
        for i in range(len(geno_df.index)):
            row.append(_num_to_gt(geno_values[i, j]))
        lines.append("\t".join(row))

    return "\n".join(lines)


def numeric_to_plink_ped_map(geno_df, marker_info_df=None,
                                pop_map=None):
    """
    Convert numeric genotype to PLINK PED + MAP files.
    Returns: (ped_content, map_content)
    """
    n_markers = geno_df.shape[1]

    if marker_info_df is not None:
        chr_arr = (marker_info_df["Chrom"].astype(str).values
                     if "Chrom" in marker_info_df.columns
                     else ["0"] * n_markers)
        pos_arr = (marker_info_df["Pos"].values
                     if "Pos" in marker_info_df.columns
                     else np.arange(n_markers))
    else:
        chr_arr = ["0"] * n_markers
        pos_arr = np.arange(n_markers)

    # MAP file: chr, snp_id, genetic_distance, position
    map_lines = []
    for j, snp in enumerate(geno_df.columns):
        map_lines.append(f"{chr_arr[j]}\t{snp}\t0\t{int(pos_arr[j])}")

    map_content = "\n".join(map_lines)

    # PED file
    def _num_to_alleles(v):
        if pd.isna(v):
            return ("0", "0")
        v_int = int(round(v))
        if v_int == 0:
            return ("A", "A")
        elif v_int == 1:
            return ("A", "T")
        elif v_int == 2:
            return ("T", "T")
        return ("0", "0")

    geno_values = geno_df.values
    ped_lines = []

    for i, s in enumerate(geno_df.index.astype(str)):
        # FID: use population if available, else sample ID
        if pop_map and s in pop_map:
            fid = str(pop_map[s]).replace(" ", "_")
        else:
            fid = s
        # PED: FID IID PID MID SEX PHENO alleles...
        row = [fid, s, "0", "0", "0", "-9"]
        for j in range(n_markers):
            a1, a2 = _num_to_alleles(geno_values[i, j])
            row.extend([a1, a2])
        ped_lines.append("\t".join(row))

    ped_content = "\n".join(ped_lines)
    return ped_content, map_content


def numeric_to_structure(geno_df, pop_map=None):
    """
    Convert to STRUCTURE input format.
    Each individual takes 2 rows (one per allele).
    """
    n_samples = geno_df.shape[0]
    n_markers = geno_df.shape[1]

    # Header: marker names
    header = "\t".join([""] * 2 + geno_df.columns.astype(str).tolist())

    lines = [header]

    geno_values = geno_df.values

    for i, s in enumerate(geno_df.index.astype(str)):
        pop_id = "1"  # default
        if pop_map and s in pop_map:
            # Use hash of pop name as integer ID (STRUCTURE needs integers)
            unique_pops = sorted(set(pop_map.values()))
            pop_id = str(unique_pops.index(pop_map[s]) + 1)

        # Two rows per individual (one per allele)
        row1 = [s, pop_id]
        row2 = [s, pop_id]

        for j in range(n_markers):
            v = geno_values[i, j]
            if pd.isna(v):
                a1, a2 = "-9", "-9"
            elif v == 0:
                a1, a2 = "1", "1"  # AA
            elif v == 1:
                a1, a2 = "1", "2"  # AT
            elif v == 2:
                a1, a2 = "2", "2"  # TT
            else:
                a1, a2 = "-9", "-9"
            row1.append(a1)
            row2.append(a2)

        lines.append("\t".join(row1))
        lines.append("\t".join(row2))

    return "\n".join(lines)


# ═══════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════
tab_basic, tab_convert, tab_bundle, tab_subset, tab_analyses = st.tabs([
    "📄 Basic Downloads",
    "🔄 Format Conversion",
    "📦 Full Bundle",
    "✂️ Subset Export",
    "📊 Analysis Results",
])


# ═══════════════════════════════════════════
# TAB 1 — Basic Downloads
# ═══════════════════════════════════════════
with tab_basic:
    st.subheader("📄 Basic Data Downloads")
    st.write("Download the currently loaded data in standard formats.")

    st.markdown("### 🧬 Genotype Data")

    bc1, bc2, bc3 = st.columns(3)

    with bc1:
        st.download_button(
            "📥 Genotype (CSV)",
            geno.to_csv(),
            "genotype_matrix.csv", "text/csv",
            key="dl_geno_csv",
            use_container_width=True,
        )

    with bc2:
        st.download_button(
            "📥 Genotype (TSV)",
            geno.to_csv(sep="\t"),
            "genotype_matrix.tsv", "text/tab-separated-values",
            key="dl_geno_tsv",
            use_container_width=True,
        )

    with bc3:
        # Compressed CSV
        buf_gz = io.BytesIO()
        with gzip.GzipFile(fileobj=buf_gz, mode="wb") as gz:
            gz.write(geno.to_csv().encode("utf-8"))
        buf_gz.seek(0)
        st.download_button(
            "📥 Genotype (CSV.gz)",
            buf_gz,
            "genotype_matrix.csv.gz", "application/gzip",
            key="dl_geno_gz",
            use_container_width=True,
        )

    st.markdown("### 📋 Metadata & Marker Info")

    bc4, bc5 = st.columns(2)

    with bc4:
        if marker_info is not None:
            st.download_button(
                "📥 Marker Information (CSV)",
                marker_info.to_csv(index=False),
                "marker_info.csv", "text/csv",
                key="dl_mi_csv",
                use_container_width=True,
            )
        else:
            st.info("ℹ️ Marker info not available.")

    with bc5:
        if meta is not None:
            st.download_button(
                "📥 Metadata (CSV)",
                meta.to_csv(index=False),
                "metadata.csv", "text/csv",
                key="dl_meta_csv",
                use_container_width=True,
            )
        else:
            st.info("ℹ️ Metadata not loaded.")

    # Excel bundle
    st.markdown("### 📊 Excel Multi-Sheet Bundle")

    if st.button("🚀 Create Excel bundle", key="excel_bundle_basic",
                 use_container_width=True):
        sheets_dict = {
            "Genotype": geno.reset_index(),
        }
        if marker_info is not None:
            sheets_dict["Marker_Info"] = marker_info
        if meta is not None:
            sheets_dict["Metadata"] = meta

        download_excel(sheets_dict, "genomics_data.xlsx",
                        label="📥 Download Excel bundle",
                        key="dl_excel_basic")


# ═══════════════════════════════════════════
# TAB 2 — Format Conversion
# ═══════════════════════════════════════════
with tab_convert:
    st.subheader("🔄 Convert Between Genotype Formats")
    st.write(
        "Convert your genotype data to different standard formats used by "
        "population genetics software."
    )

    fmt_out = st.selectbox(
        "Export format",
        ["Numeric CSV", "Numeric TSV", "HapMap",
         "VCF", "PLINK PED + MAP", "STRUCTURE"],
        key="ex_fmt",
        help=(
            "• **Numeric CSV/TSV**: Standard 0/1/2 dosage format\n"
            "• **HapMap**: Standard hmp.txt format\n"
            "• **VCF**: Variant Call Format v4.2\n"
            "• **PLINK PED + MAP**: For PLINK software\n"
            "• **STRUCTURE**: For STRUCTURE software"
        ),
    )

    compress_output = st.checkbox("Compress output (gzip)", False,
                                    key="ex_compress")

    if st.button("🚀 Generate export", key="ex_gen",
                 use_container_width=True):

        with st.spinner(f"Converting to {fmt_out}..."):
            if fmt_out == "Numeric CSV":
                content = geno.to_csv()
                mime = "text/csv"
                name = "genotype_numeric.csv"
                extra_files = None

            elif fmt_out == "Numeric TSV":
                content = geno.to_csv(sep="\t")
                mime = "text/tab-separated-values"
                name = "genotype_numeric.tsv"
                extra_files = None

            elif fmt_out == "HapMap":
                hmp_df = numeric_to_hapmap(geno, marker_info)
                content = hmp_df.to_csv(sep="\t", index=False)
                mime = "text/plain"
                name = "genotype.hmp.txt"
                extra_files = None

            elif fmt_out == "VCF":
                content = numeric_to_vcf(geno, marker_info)
                mime = "text/plain"
                name = "genotype.vcf"
                extra_files = None

            elif fmt_out == "PLINK PED + MAP":
                ped_content, map_content = numeric_to_plink_ped_map(
                    geno, marker_info, pop_map)
                content = ped_content
                mime = "text/plain"
                name = "genotype.ped"
                extra_files = {"genotype.map": map_content}

            elif fmt_out == "STRUCTURE":
                content = numeric_to_structure(geno, pop_map)
                mime = "text/plain"
                name = "genotype.str"
                extra_files = None

        # Preview
        with st.expander("👁️ Preview first 5 lines"):
            preview_lines = content.split("\n")[:5]
            st.code("\n".join(preview_lines), language="text")

        # Compress if requested
        if compress_output:
            buf_c = io.BytesIO()
            with gzip.GzipFile(fileobj=buf_c, mode="wb") as gz:
                gz.write(content.encode("utf-8"))
            buf_c.seek(0)
            st.download_button(
                f"📥 Download {name}.gz",
                buf_c, f"{name}.gz", "application/gzip",
                key="ex_dl_button_gz",
                use_container_width=True,
            )
        else:
            st.download_button(
                f"📥 Download {name}",
                content, name, mime,
                key="ex_dl_button",
                use_container_width=True,
            )

        # If PED, also offer MAP
        if extra_files:
            for extra_name, extra_content in extra_files.items():
                st.download_button(
                    f"📥 Download {extra_name}",
                    extra_content, extra_name, "text/plain",
                    key=f"ex_dl_{extra_name}",
                    use_container_width=True,
                )

        st.success(f"✅ Export ready: {name}"
                    + (".gz" if compress_output else ""))

        # Show format info
        st.markdown(f"### About {fmt_out} format")
        format_info = {
            "Numeric CSV": "Standard CSV where rows are samples and columns are markers. Values: 0 (ref/ref), 1 (ref/alt), 2 (alt/alt).",
            "Numeric TSV": "Tab-separated version of Numeric CSV.",
            "HapMap": "Standard HapMap format. Includes 11 metadata columns (rs#, alleles, chrom, pos, strand, etc.) followed by sample genotype calls (e.g., AA, AT, TT).",
            "VCF": "Variant Call Format v4.2. The standard format for variant data. GT field encodes genotypes as 0/0, 0/1, 1/1.",
            "PLINK PED + MAP": "PLINK v1 format. PED file contains sample IDs and genotypes; MAP file contains marker positions. Use with `plink --file genotype`.",
            "STRUCTURE": "STRUCTURE software format. Each individual takes 2 rows (one per allele). Population IDs are converted to integers.",
        }
        st.info(format_info.get(fmt_out, ""))


# ═══════════════════════════════════════════
# TAB 3 — Full Bundle
# ═══════════════════════════════════════════
with tab_bundle:
    st.subheader("📦 Download Full Analysis Bundle")
    st.write(
        "Bundle all data + statistics + reports into a single ZIP file."
    )

    # Options
    st.markdown("### Choose contents")
    bc1, bc2 = st.columns(2)
    with bc1:
        include_geno = st.checkbox("Genotype matrix (CSV)", True,
                                     key="bundle_geno")
        include_meta = st.checkbox("Metadata (CSV)",
                                     meta is not None,
                                     key="bundle_meta")
        include_marker = st.checkbox("Marker info (CSV)",
                                       marker_info is not None,
                                       key="bundle_marker")
        include_stats = st.checkbox("Per-marker statistics (CSV)",
                                      True, key="bundle_stats")
    with bc2:
        include_hapmap = st.checkbox("HapMap format", False,
                                       key="bundle_hapmap")
        include_vcf = st.checkbox("VCF format", False,
                                    key="bundle_vcf")
        include_ped = st.checkbox("PLINK PED + MAP", False,
                                    key="bundle_ped")
        include_report = st.checkbox("Markdown report", True,
                                       key="bundle_report")

    if st.button("🚀 Build ZIP bundle",
                 use_container_width=True, key="ex_zip"):
        with st.spinner("Building bundle..."):
            buf = io.BytesIO()

            with zipfile.ZipFile(buf, "w",
                                    zipfile.ZIP_DEFLATED) as zf:

                # ─── Data folder ───
                if include_geno:
                    zf.writestr("01_data/genotype_matrix.csv",
                                 geno.to_csv())
                if include_marker and marker_info is not None:
                    zf.writestr("01_data/marker_info.csv",
                                 marker_info.to_csv(index=False))
                if include_meta and meta is not None:
                    zf.writestr("01_data/metadata.csv",
                                 meta.to_csv(index=False))

                # ─── Statistics folder ───
                if include_stats:
                    maf = calc_maf(geno)
                    ho = calc_het_obs(geno)
                    he = calc_het_exp(geno)
                    pic = calc_pic(geno)
                    fis = calc_fis(geno)
                    shannon = calc_shannon_diversity(geno)
                    ne = calc_effective_alleles(geno)
                    miss = calc_missing_rate(geno, axis=0)

                    stats = pd.DataFrame({
                        "Marker": geno.columns,
                        "MAF": maf.values,
                        "PIC": pic.values,
                        "Ho": ho.values,
                        "He": he.values,
                        "Fis": fis.values,
                        "Shannon_I": shannon.values,
                        "Ne_alleles": ne.values,
                        "MissingRate": miss.values,
                    })
                    if marker_info is not None:
                        stats = stats.merge(marker_info,
                                              on="Marker", how="left")
                    zf.writestr("02_statistics/snp_statistics.csv",
                                 stats.to_csv(index=False))

                    # Sample stats
                    sample_stats = pd.DataFrame({
                        "Sample": geno.index.astype(str),
                        "Missing_rate": calc_missing_rate(
                            geno, axis=1).values,
                        "Het_rate": ((geno == 1).sum(axis=1).values /
                                       np.maximum(geno.notna().sum(axis=1).values, 1)),
                    })
                    zf.writestr("02_statistics/sample_statistics.csv",
                                 sample_stats.to_csv(index=False))

                    # Per-population stats
                    if pop_map:
                        pop_stats_rows = []
                        for pop in sorted(set(pop_map.values())):
                            samples_pop = [s for s in geno.index.astype(str)
                                            if pop_map.get(s) == pop]
                            if len(samples_pop) < 2:
                                continue
                            sub = geno.loc[samples_pop]
                            pop_stats_rows.append({
                                "Population": pop,
                                "N_samples": len(samples_pop),
                                "Mean_MAF": calc_maf(sub).mean(),
                                "Mean_He": calc_het_exp(sub).mean(),
                                "Mean_Ho": calc_het_obs(sub).mean(),
                                "Mean_PIC": calc_pic(sub).mean(),
                                "Mean_Fis": calc_fis(sub).mean(),
                            })
                        if pop_stats_rows:
                            zf.writestr("02_statistics/per_population_stats.csv",
                                         pd.DataFrame(pop_stats_rows).to_csv(
                                             index=False))

                # ─── Alternative formats folder ───
                if include_hapmap:
                    hmp_df = numeric_to_hapmap(geno, marker_info)
                    zf.writestr("03_formats/genotype.hmp.txt",
                                 hmp_df.to_csv(sep="\t", index=False))
                if include_vcf:
                    vcf_content = numeric_to_vcf(geno, marker_info)
                    zf.writestr("03_formats/genotype.vcf", vcf_content)
                if include_ped:
                    ped_c, map_c = numeric_to_plink_ped_map(
                        geno, marker_info, pop_map)
                    zf.writestr("03_formats/genotype.ped", ped_c)
                    zf.writestr("03_formats/genotype.map", map_c)

                # ─── Report ───
                if include_report:
                    if not include_stats:
                        maf = calc_maf(geno)
                        he = calc_het_exp(geno)
                        pic = calc_pic(geno)

                    report = f"""# Population Genomics Analysis Bundle

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Samples:** {geno.shape[0]:,}
- **Markers:** {geno.shape[1]:,}
- **Missing rate:** {geno.isna().sum().sum() / geno.size * 100:.2f}%
- **Populations:** {len(set(pop_map.values())) if pop_map else 'N/A'}

## Bundle Contents

### 01_data/
- `genotype_matrix.csv` — Full 0/1/2 genotype matrix
- `marker_info.csv` — Marker chromosome/position (if available)
- `metadata.csv` — Sample metadata (if available)

### 02_statistics/
- `snp_statistics.csv` — Per-SNP MAF, PIC, He, Ho, Fis, Shannon, Ne, missing
- `sample_statistics.csv` — Per-sample missing rate, heterozygosity
- `per_population_stats.csv` — Per-population diversity indices (if metadata)

### 03_formats/
- `genotype.hmp.txt` — HapMap format
- `genotype.vcf` — VCF v4.2 format
- `genotype.ped` + `.map` — PLINK format

## Genome-wide Summary
- Mean MAF: {maf.mean():.4f}
- Mean He: {he.mean():.4f}
- Mean PIC: {pic.mean():.4f}

## Citation
If you use these results in a publication, please cite:
- The underlying statistical methods (see README of the platform)
- Interactive Population Genomics Platform

## License
This data is provided for research purposes only.
"""
                    zf.writestr("README.md", report)

            buf.seek(0)

        # Show what's in the bundle
        with st.expander("📋 Bundle contents"):
            st.code("""
population_genomics_bundle.zip
├── README.md
├── 01_data/
│   ├── genotype_matrix.csv
│   ├── marker_info.csv
│   └── metadata.csv
├── 02_statistics/
│   ├── snp_statistics.csv
│   ├── sample_statistics.csv
│   └── per_population_stats.csv
└── 03_formats/
    ├── genotype.hmp.txt
    ├── genotype.vcf
    ├── genotype.ped
    └── genotype.map
""", language="text")

        st.download_button(
            "📥 Download bundle.zip",
            buf, "population_genomics_bundle.zip",
            "application/zip",
            key="ex_zip_dl",
            use_container_width=True,
        )
        st.success("✅ Bundle ready for download!")


# ═══════════════════════════════════════════
# TAB 4 — Subset Export
# ═══════════════════════════════════════════
with tab_subset:
    st.subheader("✂️ Export Subset of Data")
    st.write("Select specific samples or markers to export.")

    st.markdown("### 👤 Sample Selection")
    subset_method_s = st.radio(
        "How to select samples?",
        ["All samples", "By list", "By population", "First N"],
        horizontal=True, key="subset_sample_method",
    )

    selected_samples = geno.index.tolist()

    if subset_method_s == "By list":
        default_samples = geno.index.tolist()[:5]
        selected_samples = st.multiselect(
            "Choose samples",
            geno.index.tolist(),
            default=default_samples,
            key="subset_sample_list",
        )
    elif subset_method_s == "By population" and pop_map:
        pops_avail = sorted(set(pop_map.values()))
        selected_pops = st.multiselect(
            "Choose populations",
            pops_avail,
            default=pops_avail[:2],
            key="subset_pop_list",
        )
        selected_samples = [s for s in geno.index.astype(str)
                             if pop_map.get(s) in selected_pops]
    elif subset_method_s == "First N":
        n_first = st.slider("First N samples", 1, geno.shape[0],
                              min(20, geno.shape[0]),
                              key="subset_n_first")
        selected_samples = geno.index.tolist()[:n_first]

    st.info(f"Selected: **{len(selected_samples):,}** samples")

    st.markdown("### 🧬 Marker Selection")
    subset_method_m = st.radio(
        "How to select markers?",
        ["All markers", "By list", "By chromosome",
         "Top N by MAF", "First N"],
        horizontal=True, key="subset_marker_method",
    )

    selected_markers = geno.columns.tolist()

    if subset_method_m == "By list":
        default_markers = geno.columns.tolist()[:5]
        selected_markers = st.multiselect(
            "Choose markers",
            geno.columns.tolist(),
            default=default_markers,
            key="subset_marker_list",
        )
    elif subset_method_m == "By chromosome" and \
        marker_info is not None and "Chrom" in marker_info.columns:
        chr_avail = sorted(marker_info["Chrom"].astype(str).unique())
        selected_chrs = st.multiselect(
            "Choose chromosomes", chr_avail,
            default=chr_avail[:1],
            key="subset_chr_list",
        )
        markers_in_chr = marker_info[
            marker_info["Chrom"].astype(str).isin(selected_chrs)
        ]["Marker"].astype(str).tolist()
        selected_markers = [m for m in geno.columns.astype(str)
                              if m in markers_in_chr]
    elif subset_method_m == "Top N by MAF":
        n_top = st.slider("Top N markers", 10,
                            min(1000, geno.shape[1]),
                            min(100, geno.shape[1]),
                            key="subset_n_maf")
        maf_all = calc_maf(geno)
        selected_markers = maf_all.nlargest(n_top).index.tolist()
    elif subset_method_m == "First N":
        n_first_m = st.slider("First N markers", 10, geno.shape[1],
                                min(100, geno.shape[1]),
                                key="subset_n_first_m")
        selected_markers = geno.columns.tolist()[:n_first_m]

    st.info(f"Selected: **{len(selected_markers):,}** markers")

    # Preview
    if len(selected_samples) > 0 and len(selected_markers) > 0:
        subset_geno = geno.loc[selected_samples, selected_markers]

        st.markdown(f"### 👁️ Preview subset ({subset_geno.shape[0]} × {subset_geno.shape[1]})")
        st.dataframe(subset_geno.iloc[:10, :10],
                      use_container_width=True)

        st.download_button(
            f"📥 Download subset ({subset_geno.shape[0]}×{subset_geno.shape[1]})",
            subset_geno.to_csv(),
            "genotype_subset.csv", "text/csv",
            key="dl_subset",
            use_container_width=True,
        )


# ═══════════════════════════════════════════
# TAB 5 — Analysis Results
# ═══════════════════════════════════════════
with tab_analyses:
    st.subheader("📊 Analysis Results Export")
    st.write("Export cached analysis results from other modules.")

    st.markdown("### Available cached results in session:")

    # Check what's in session
    session_keys_of_interest = [
        ("kin_matrix", "Kinship matrix"),
        ("fst_matrix", "Fst matrix"),
        ("struct_consensus_Q", "STRUCTURE consensus Q"),
        ("struct_stats_df", "STRUCTURE model statistics"),
        ("fst_outlier_df", "Fst outlier analysis"),
        ("pcadapt_outlier_df", "PCAdapt outliers"),
        ("ihs_outlier_df", "iHS-like outliers"),
        ("bayescan_outlier_df", "BayeScan-like outliers"),
        ("lfmm_outlier_df", "LFMM-like outliers"),
        ("tajimas_d_df", "Tajima's D windows"),
    ]

    found_results = []
    for key, name in session_keys_of_interest:
        if key in st.session_state and \
            st.session_state[key] is not None:
            found_results.append((key, name))

    if not found_results:
        st.info(
            "ℹ️ No cached analysis results found in session. "
            "Please run analyses (Population Structure, Kinship, "
            "Selection Detection, etc.) in other modules first."
        )
    else:
        st.success(f"✅ Found {len(found_results)} cached results")

        for key, name in found_results:
            with st.expander(f"📊 {name}"):
                result = st.session_state[key]

                if isinstance(result, pd.DataFrame):
                    st.dataframe(result.head(20),
                                  use_container_width=True)
                    download_dataframe(result, f"{key}.csv",
                                        key=f"dl_analysis_{key}")
                elif isinstance(result, np.ndarray):
                    st.write(f"Shape: {result.shape}")
                    st.dataframe(pd.DataFrame(result).head(10),
                                  use_container_width=True)
                    download_dataframe(
                        pd.DataFrame(result), f"{key}.csv",
                        key=f"dl_analysis_{key}",
                    )
                elif isinstance(result, dict):
                    for sub_key, sub_val in list(result.items())[:5]:
                        st.write(f"**{sub_key}:**")
                        if isinstance(sub_val, (pd.DataFrame,
                                                  np.ndarray)):
                            st.dataframe(pd.DataFrame(sub_val).head(5))
                        else:
                            st.write(sub_val)


# ═══════════════════════════════════════════
# SESSION STATE INFO
# ═══════════════════════════════════════════
st.markdown("---")
st.subheader("📋 Session State Summary")

session_info = {
    "Genotype loaded": "✅ Yes" if geno is not None else "❌ No",
    "Marker info loaded": "✅ Yes" if marker_info is not None else "❌ No",
    "Metadata loaded": "✅ Yes" if meta is not None else "❌ No",
    "Populations configured": ("✅ " + str(len(set(pop_map.values())))
                                if pop_map else "❌ No"),
    "Samples": f"{geno.shape[0]:,}" if geno is not None else "0",
    "Markers": f"{geno.shape[1]:,}" if geno is not None else "0",
    "Missing rate":
        (f"{geno.isna().sum().sum() / geno.size * 100:.2f}%"
          if geno is not None else "N/A"),
}
info_df = pd.DataFrame(list(session_info.items()),
                        columns=["Item", "Status"])
st.table(info_df)