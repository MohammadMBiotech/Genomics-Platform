"""
Machine Learning & Selection Detection — Publication Quality
────────────────────────────────────────────────────────────
Features:
  - Classification (RF, LR, SVM, XGBoost, Gradient Boosting)
  - Feature Selection (Lasso, Ridge)
  - Selection Detection:
      * Extended ROH (with FROH, total ROH)
      * Fst outlier scan
      * PCAdapt
      * Tajima's D sliding window
      * iHS-like (Extended Haplotype Homozygosity)
      * BayeScan-like
      * LFMM-like (Genotype-Environment Association)
      * Selective sweep (CLR-like)
      * Combined outlier detection
  - Unsupervised UMAP
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score,
)
from scipy.stats import chi2, f as f_dist

from utils import (
    get_geno_from_session, get_meta_from_session,
    impute_missing, calc_maf, calc_het_exp, calc_allele_freq,
    download_plotly_html, download_dataframe,
)

st.title("🤖 Machine Learning & Selection Detection")
st.markdown("---")

geno, marker_info = get_geno_from_session()
if geno is None:
    st.stop()

meta = get_meta_from_session()


# ═══════════════════════════════════════════
# GLOBAL: Metadata column selector
# ═══════════════════════════════════════════
global_sam_col = None
global_pop_col = None

if meta is not None:
    st.subheader("🔧 Metadata Configuration")
    gc1, gc2 = st.columns(2)
    with gc1:
        global_sam_col = st.selectbox(
            "Sample ID column",
            meta.columns.tolist(),
            key="ml_samcol_g",
        )
    with gc2:
        global_pop_col = st.selectbox(
            "Default Target / Population column",
            meta.columns.tolist(),
            key="ml_popcol_g",
        )
    st.info(f"✅ Using **{global_sam_col}** as Sample ID, "
             f"**{global_pop_col}** as default target.")
    st.markdown("---")


tab_clf, tab_fs, tab_sel, tab_um = st.tabs([
    "🏷️ Classification",
    "🎯 Feature Selection",
    "🎯 Selection Detection",
    "🗺️ Unsupervised (UMAP)",
])


# =========================================================
# TAB 1 — Classification
# =========================================================
with tab_clf:
    st.subheader("Classification: Predict Population/Species from SNPs")

    if meta is None:
        st.warning("Metadata required for classification.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            target_col = st.selectbox("Target column (label)",
                                        meta.columns.tolist(),
                                        index=meta.columns.tolist().index(global_pop_col)
                                        if global_pop_col in meta.columns.tolist() else 0,
                                        key="ml_clf_target")
        with c2:
            sam_col = st.selectbox("Sample ID column",
                                     meta.columns.tolist(),
                                     index=meta.columns.tolist().index(global_sam_col)
                                     if global_sam_col in meta.columns.tolist() else 0,
                                     key="ml_clf_sam")

        method = st.selectbox(
            "Classifier",
            ["Random Forest", "Logistic Regression", "SVM",
             "XGBoost", "Gradient Boosting"],
            key="ml_clf_method",
        )

        # Hyperparameters
        rf_n, rf_d = 100, None
        if method == "Random Forest":
            rf_n = st.slider("N estimators", 10, 500, 100, 10, key="ml_rf_n")
            rf_d = st.selectbox("Max depth", [None, 5, 10, 15, 20], key="ml_rf_d")

        lr_C = 1.0
        if method == "Logistic Regression":
            lr_C = st.slider("Regularization C", 0.01, 10.0, 1.0, 0.01, key="ml_lr_c")

        svm_C, svm_k = 1.0, "rbf"
        if method == "SVM":
            svm_C = st.slider("C", 0.01, 10.0, 1.0, 0.01, key="ml_svm_c")
            svm_k = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="ml_svm_k")

        xgb_lr, xgb_n, xgb_d = 0.1, 100, 6
        if method == "XGBoost":
            xgb_lr = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01, key="ml_xgb_lr")
            xgb_n = st.slider("N estimators", 10, 500, 100, 10, key="ml_xgb_n")
            xgb_d = st.slider("Max depth", 2, 15, 6, key="ml_xgb_d")

        use_pca_clf = st.checkbox("PCA pre-reduction (recommended)", True, key="ml_use_pca")
        n_pcs_clf = st.slider("N PCs", 2, 50, 20, key="ml_npcs") if use_pca_clf else None

        c3, c4, c5 = st.columns(3)
        with c3:
            test_sz = st.slider("Test %", 10, 50, 20, 5, key="ml_ts") / 100
        with c4:
            rs = int(st.number_input("Random state", value=42, key="ml_rs"))
        with c5:
            cv_k = st.slider("CV folds", 2, 10, 5, key="ml_cv")

        if st.button("🚀 Train & Evaluate Classifier",
                     use_container_width=True, key="ml_clf_run"):
            pop_map = dict(zip(meta[sam_col].astype(str),
                                meta[target_col].astype(str)))
            samples_use = [s for s in geno.index.astype(str)
                            if s in pop_map and pd.notna(pop_map.get(s))]

            X_raw = geno.loc[samples_use]
            y_raw = np.array([pop_map[s] for s in samples_use])

            X_imp = impute_missing(X_raw, "mean").values
            X_scaled = StandardScaler().fit_transform(X_imp)

            if use_pca_clf:
                pca_pre = PCA(n_components=min(n_pcs_clf, X_scaled.shape[1]))
                X_use = pca_pre.fit_transform(X_scaled)
            else:
                X_use = X_scaled

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            cnames = le.classes_.astype(str)
            nc = len(cnames)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_use, y, test_size=test_sz, random_state=rs, stratify=y,
            )

            if method == "Random Forest":
                model = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d,
                                                  random_state=rs, n_jobs=-1)
            elif method == "Logistic Regression":
                model = LogisticRegression(C=lr_C, max_iter=2000, random_state=rs)
            elif method == "SVM":
                model = SVC(C=svm_C, kernel=svm_k, probability=True, random_state=rs)
            elif method == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=rs)
            elif method == "XGBoost":
                try:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        learning_rate=xgb_lr, n_estimators=xgb_n,
                        max_depth=xgb_d, random_state=rs, n_jobs=-1,
                        verbosity=0, use_label_encoder=False,
                        eval_metric="mlogloss" if nc > 2 else "logloss")
                except ImportError:
                    st.error("XGBoost not installed.")
                    st.stop()

            with st.spinner(f"Training {method}..."):
                model.fit(X_tr, y_tr)
                y_pr = model.predict(X_te)
                try:
                    y_pp = model.predict_proba(X_te)
                except Exception:
                    y_pp = None

                avg = "weighted" if nc > 2 else "binary"
                acc = accuracy_score(y_te, y_pr)
                prec = precision_score(y_te, y_pr, average=avg, zero_division=0)
                rec = recall_score(y_te, y_pr, average=avg, zero_division=0)
                f1 = f1_score(y_te, y_pr, average=avg, zero_division=0)

                skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=rs)
                cv_acc = cross_val_score(model, X_use, y, cv=skf, scoring="accuracy")

            st.success(f"✅ {method} trained.")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("Precision", f"{prec:.4f}")
            m3.metric("Recall", f"{rec:.4f}")
            m4.metric("F1", f"{f1:.4f}")
            m5.metric(f"CV Acc ({cv_k}-fold)",
                        f"{cv_acc.mean():.4f}±{cv_acc.std():.4f}")

            st.subheader("Detailed classification report")
            rpt = classification_report(y_te, y_pr, target_names=cnames, output_dict=True)
            st.dataframe(pd.DataFrame(rpt).T.style.format("{:.4f}"),
                          use_container_width=True)

            cm = confusion_matrix(y_te, y_pr)
            fig_cm = px.imshow(cm, x=cnames, y=cnames, text_auto=True,
                                color_continuous_scale="Blues",
                                title="Confusion Matrix",
                                labels=dict(x="Predicted", y="Actual"))
            fig_cm.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_cm, use_container_width=True)

            if y_pp is not None:
                st.subheader("ROC Curve")
                fig_roc = go.Figure()
                if nc == 2:
                    fpr, tpr, _ = roc_curve(y_te, y_pp[:, 1])
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"AUC={auc(fpr, tpr):.4f}",
                        line=dict(color="steelblue", width=2)))
                else:
                    for i, cn in enumerate(cnames):
                        yb = (y_te == i).astype(int)
                        if yb.sum() == 0:
                            continue
                        fpr, tpr, _ = roc_curve(yb, y_pp[:, i])
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"{cn} AUC={auc(fpr, tpr):.4f}"))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="gray"), name="Random"))
                fig_roc.update_layout(
                    xaxis_title="FPR", yaxis_title="TPR",
                    template="plotly_white", height=500, title="ROC Curves")
                st.plotly_chart(fig_roc, use_container_width=True)

            if method in ["Random Forest", "XGBoost", "Gradient Boosting"]:
                st.subheader("Feature Importance (top 30)")
                if use_pca_clf:
                    feat_names = [f"PC{i+1}" for i in range(X_use.shape[1])]
                else:
                    feat_names = geno.columns.astype(str).tolist()

                imp = pd.DataFrame({
                    "Feature": feat_names,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=False).head(30)

                fig_imp = px.bar(imp, x="Importance", y="Feature",
                                  orientation="h", color="Importance",
                                  color_continuous_scale="viridis",
                                  title="Top 30 features")
                fig_imp.update_layout(template="plotly_white", height=650,
                                         yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)
                download_dataframe(imp, "feature_importance.csv", key="dl_ml_imp")


# =========================================================
# TAB 2 — Feature Selection
# =========================================================
with tab_fs:
    st.subheader("Feature Selection with Lasso (L1 penalty)")
    st.write(
        "Identifies most informative SNPs for predicting a phenotype/trait."
    )

    if meta is None:
        st.warning("Metadata required.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            target_fs = st.selectbox(
                "Target trait (numeric)",
                meta.select_dtypes(include=[np.number]).columns.tolist(),
                key="fs_target",
            )
        with c2:
            sam_fs = st.selectbox("Sample ID column",
                                    meta.columns.tolist(),
                                    index=meta.columns.tolist().index(global_sam_col)
                                    if global_sam_col in meta.columns.tolist() else 0,
                                    key="fs_sam")

        alpha_choice = st.selectbox("Alpha selection",
                                      ["Auto (LassoCV)", "Manual"],
                                      key="fs_alpha_ch")
        if alpha_choice == "Manual":
            alpha_fs = st.slider("Alpha", 0.001, 5.0, 0.1, 0.001, key="fs_alpha")

        if st.button("🚀 Run Lasso feature selection",
                     use_container_width=True, key="fs_run"):
            trait_map = dict(zip(meta[sam_fs].astype(str),
                                   meta[target_fs].astype(float)))
            samples_use = [s for s in geno.index.astype(str)
                            if s in trait_map and pd.notna(trait_map.get(s))]

            X_raw = geno.loc[samples_use]
            y = np.array([trait_map[s] for s in samples_use])

            X_imp = impute_missing(X_raw, "mean").values
            X_scaled = StandardScaler().fit_transform(X_imp)

            with st.spinner("Fitting Lasso..."):
                if alpha_choice == "Auto (LassoCV)":
                    model_fs = LassoCV(cv=5, random_state=42,
                                        max_iter=10000, n_jobs=-1)
                else:
                    model_fs = Lasso(alpha=alpha_fs, max_iter=10000)
                model_fs.fit(X_scaled, y)

            coefs = model_fs.coef_
            selected_mask = np.abs(coefs) > 1e-8
            selected_snps = geno.columns[selected_mask]

            st.success(f"✅ Selected **{selected_mask.sum()}** SNPs out of {len(coefs)}.")

            if alpha_choice == "Auto (LassoCV)":
                st.info(f"Optimal α = {model_fs.alpha_:.6f}")

            imp_df = pd.DataFrame({
                "SNP": geno.columns.astype(str),
                "Coefficient": coefs,
                "Abs_Coef": np.abs(coefs),
            }).sort_values("Abs_Coef", ascending=False)

            top_coef = imp_df.head(50)
            fig_c = px.bar(top_coef, x="Coefficient", y="SNP",
                            orientation="h", color="Coefficient",
                            color_continuous_scale="RdBu_r",
                            title="Top 50 selected SNPs by Lasso")
            fig_c.update_layout(template="plotly_white", height=800,
                                 yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_c, use_container_width=True)

            st.dataframe(imp_df.head(200), use_container_width=True)
            download_dataframe(imp_df, "lasso_selected_snps.csv", key="dl_fs")

# =========================================================
# TAB 3 — SELECTION DETECTION (Comprehensive)
# =========================================================
with tab_sel:
    st.subheader("🎯 Selection Detection")
    st.write(
        "Comprehensive suite for detecting SNPs under natural selection using "
        "multiple complementary approaches."
    )

    method_sel = st.selectbox(
        "Detection method",
        [
            "Fst outlier (population-based)",
            "PCAdapt (PC-based)",
            "Extended ROH (Runs of Homozygosity)",
            "Tajima's D (sliding window)",
            "iHS-like (Extended Haplotype Homozygosity)",
            "BayeScan-like (Fst decomposition)",
            "LFMM-like (Genotype-Environment Association)",
            "Selective Sweep (CLR-like)",
            "🧩 Combined outlier detection",
        ],
        key="sel_method",
    )

    # ═══════════════════════════════════════════
    # METHOD 1: Fst outlier
    # ═══════════════════════════════════════════
    if method_sel == "Fst outlier (population-based)":
        if meta is None:
            st.warning("Metadata required.")
        else:
            sc1, sc2 = st.columns(2)
            with sc1:
                pop_sel = st.selectbox("Population column",
                                        meta.columns.tolist(),
                                        index=meta.columns.tolist().index(global_pop_col)
                                        if global_pop_col in meta.columns.tolist() else 0,
                                        key="sel_pop")
            with sc2:
                sam_sel = st.selectbox("Sample ID column",
                                        meta.columns.tolist(),
                                        index=meta.columns.tolist().index(global_sam_col)
                                        if global_sam_col in meta.columns.tolist() else 0,
                                        key="sel_sam")

            fst_pct = st.slider("Top % as outliers", 1, 20, 5, key="sel_fst_pct")

            if st.button("🚀 Detect Fst outliers", key="sel_fst_run"):
                pop_map_s = dict(zip(meta[sam_sel].astype(str),
                                       meta[pop_sel].astype(str)))
                pops_here = sorted(set(pop_map_s.values()))

                with st.spinner("Computing per-marker Fst..."):
                    Ht_marker = calc_het_exp(geno)
                    hs_list = []
                    weights = []
                    for p in pops_here:
                        samples_p = [s for s in geno.index
                                        if pop_map_s.get(str(s)) == p]
                        if len(samples_p) < 2:
                            continue
                        Hs_p = calc_het_exp(geno.loc[samples_p])
                        hs_list.append(Hs_p * len(samples_p))
                        weights.append(len(samples_p))

                    if hs_list:
                        Hs_marker = pd.concat(hs_list, axis=1).sum(axis=1) / sum(weights)
                        Fst_marker = (Ht_marker - Hs_marker) / Ht_marker.replace(0, np.nan)
                    else:
                        st.error("Not enough populations.")
                        st.stop()

                fst_df = pd.DataFrame({
                    "Marker": geno.columns,
                    "Fst": Fst_marker.values,
                })

                if marker_info is not None:
                    fst_df = fst_df.merge(marker_info, on="Marker", how="left")

                fst_df = fst_df.dropna(subset=["Fst"])
                thresh = np.percentile(fst_df["Fst"], 100 - fst_pct)
                fst_df["Outlier"] = fst_df["Fst"] >= thresh

                n_outl = int(fst_df["Outlier"].sum())
                st.success(f"✅ Detected **{n_outl}** outlier SNPs "
                            f"(Fst ≥ {thresh:.4f}).")

                if "Chrom" in fst_df.columns:
                    fst_df_plot = fst_df.copy()
                    fst_df_plot["Chrom"] = fst_df_plot["Chrom"].astype(str)
                    chr_order = sorted(fst_df_plot["Chrom"].unique())
                    fst_df_plot["Chrom"] = pd.Categorical(
                        fst_df_plot["Chrom"], categories=chr_order, ordered=True)
                    fst_df_plot = fst_df_plot.sort_values(["Chrom", "Pos"])
                    fst_df_plot["Order"] = np.arange(len(fst_df_plot))

                    fig_man = px.scatter(
                        fst_df_plot, x="Order", y="Fst", color="Chrom",
                        title="Per-marker Fst (Manhattan-style)",
                        hover_data=["Marker", "Pos"],
                    )
                    fig_man.add_hline(y=thresh, line_dash="dash",
                                       line_color="red",
                                       annotation_text=f"Outlier ({fst_pct}%)")
                    fig_man.update_layout(template="plotly_white",
                                            height=550, showlegend=False)
                    st.plotly_chart(fig_man, use_container_width=True)

                fig_fst_dist = px.histogram(
                    fst_df, x="Fst", nbins=60,
                    title="Fst distribution across markers")
                fig_fst_dist.add_vline(x=thresh, line_dash="dash", line_color="red")
                fig_fst_dist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_fst_dist, use_container_width=True)

                st.subheader("Top outlier SNPs")
                st.dataframe(fst_df[fst_df["Outlier"]].sort_values(
                    "Fst", ascending=False).head(50),
                    use_container_width=True)
                download_dataframe(fst_df, "fst_outliers.csv", key="dl_sel_fst")

                # Save for combined method
                st.session_state["fst_outlier_df"] = fst_df

    # ═══════════════════════════════════════════
    # METHOD 2: PCAdapt
    # ═══════════════════════════════════════════
    elif method_sel == "PCAdapt (PC-based)":
        st.write("Identifies SNPs strongly correlated with top PCs (Mahalanobis-based).")

        pc1, pc2 = st.columns(2)
        with pc1:
            k_pcs = st.slider("Number of PCs", 2, 20, 5, key="sel_pca_k")
        with pc2:
            pct_out = st.slider("Top % as outliers", 1, 20, 5, key="sel_pca_pct")

        if st.button("🚀 Run PCAdapt-like", key="sel_pca_run"):
            with st.spinner("Running PCA + Mahalanobis..."):
                imp = impute_missing(geno, "mean").values
                X_sc = StandardScaler().fit_transform(imp)
                pca_a = PCA(n_components=k_pcs)
                pca_a.fit(X_sc)

                loadings = pca_a.components_
                loadings_z = (loadings - loadings.mean(axis=1, keepdims=True)) / \
                              loadings.std(axis=1, keepdims=True)
                stat = (loadings_z ** 2).sum(axis=0)
                pvals = 1 - chi2.cdf(stat, df=k_pcs)

            adapt_df = pd.DataFrame({
                "Marker": geno.columns,
                "Statistic": stat,
                "P_value": pvals,
                "NegLog10P": -np.log10(np.clip(pvals, 1e-300, 1)),
            })

            if marker_info is not None:
                adapt_df = adapt_df.merge(marker_info, on="Marker", how="left")

            thr = np.percentile(adapt_df["Statistic"], 100 - pct_out)
            adapt_df["Outlier"] = adapt_df["Statistic"] >= thr

            n_out = int(adapt_df["Outlier"].sum())
            st.success(f"✅ Detected **{n_out}** outlier SNPs.")

            fig_pca_out = px.histogram(adapt_df, x="Statistic", nbins=60,
                                          title="PCAdapt statistic distribution")
            fig_pca_out.add_vline(x=thr, line_dash="dash", line_color="red")
            fig_pca_out.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_pca_out, use_container_width=True)

            if "Chrom" in adapt_df.columns:
                ad_plot = adapt_df.dropna(subset=["Chrom"]).copy()
                ad_plot["Chrom"] = ad_plot["Chrom"].astype(str)
                ad_plot["Order"] = np.arange(len(ad_plot))
                fig_ad_man = px.scatter(
                    ad_plot, x="Order", y="NegLog10P", color="Chrom",
                    title="PCAdapt Manhattan plot",
                    hover_data=["Marker", "Pos"],
                )
                fig_ad_man.add_hline(y=-np.log10(0.001), line_dash="dash",
                                       line_color="red")
                fig_ad_man.update_layout(template="plotly_white", height=500,
                                           showlegend=False)
                st.plotly_chart(fig_ad_man, use_container_width=True)

            st.dataframe(adapt_df[adapt_df["Outlier"]].sort_values(
                "Statistic", ascending=False).head(50),
                use_container_width=True)
            download_dataframe(adapt_df, "pcadapt_outliers.csv",
                                key="dl_sel_pca")

            st.session_state["pcadapt_outlier_df"] = adapt_df

    # ═══════════════════════════════════════════
    # METHOD 3: Extended ROH
    # ═══════════════════════════════════════════
    elif method_sel == "Extended ROH (Runs of Homozygosity)":
        st.write(
            "Advanced ROH detection with **FROH** (fraction of genome in ROH), "
            "**length classes**, and **total ROH per sample**."
        )

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            win_size = st.slider("Window size (markers)", 10, 500, 50, key="sel_roh_ws")
        with rc2:
            min_het = st.slider("Max heterozygosity", 0.0, 0.5, 0.05, 0.01, key="sel_roh_h")
        with rc3:
            min_len = st.slider("Min ROH length (markers)", 5, 100, 20, key="sel_roh_ml")

        rc4, rc5 = st.columns(2)
        with rc4:
            merge_gap = st.slider("Merge ROH within gap (markers)",
                                    0, 20, 5, key="sel_roh_mg")
        with rc5:
            classify_lengths = st.checkbox("Classify by length", True,
                                             key="sel_roh_cl")

        if st.button("🚀 Scan Extended ROH", key="sel_roh_run"):
            with st.spinner("Detecting ROH segments..."):
                G = geno.fillna(1).values
                n_samples, n_markers = G.shape

                all_roh = []

                for si in range(n_samples):
                    sample_name = geno.index[si]
                    genotype = G[si]

                    # Sliding window heterozygosity
                    het_by_pos = np.array([
                        np.mean(genotype[i:i+win_size] == 1)
                        for i in range(0, n_markers - win_size + 1)
                    ])

                    # Find continuous low-het regions
                    is_low = het_by_pos <= min_het

                    # Identify runs
                    runs = []
                    start = None
                    for i, low in enumerate(is_low):
                        if low and start is None:
                            start = i
                        elif not low and start is not None:
                            runs.append((start, i - 1 + win_size - 1))
                            start = None
                    if start is not None:
                        runs.append((start, len(is_low) - 1 + win_size - 1))

                    # Merge close runs
                    if runs:
                        merged = [runs[0]]
                        for s, e in runs[1:]:
                            if s - merged[-1][1] <= merge_gap:
                                merged[-1] = (merged[-1][0], e)
                            else:
                                merged.append((s, e))
                        runs = merged

                    # Filter by min length
                    for s, e in runs:
                        length_markers = e - s + 1
                        if length_markers >= min_len:
                            roh_entry = {
                                "Sample": sample_name,
                                "Start_marker_idx": s,
                                "End_marker_idx": e,
                                "N_markers": length_markers,
                                "Het_proportion": np.mean(genotype[s:e+1] == 1),
                            }

                            # Add physical positions if available
                            if marker_info is not None and "Pos" in marker_info.columns:
                                positions = marker_info["Pos"].values
                                if s < len(positions) and e < len(positions):
                                    try:
                                        pos_s = float(positions[s])
                                        pos_e = float(positions[e])
                                        roh_entry["Start_pos"] = pos_s
                                        roh_entry["End_pos"] = pos_e
                                        roh_entry["Length_bp"] = pos_e - pos_s
                                    except (ValueError, TypeError):
                                        pass

                            all_roh.append(roh_entry)

            roh_df = pd.DataFrame(all_roh)

            if len(roh_df) == 0:
                st.warning("No ROH segments detected.")
            else:
                st.success(f"✅ Detected **{len(roh_df)}** ROH segments "
                            f"across {roh_df['Sample'].nunique()} samples.")

                # Summary
                summary_cols = st.columns(4)
                summary_cols[0].metric("Total ROH", len(roh_df))
                summary_cols[1].metric("Samples with ROH", roh_df["Sample"].nunique())
                summary_cols[2].metric("Mean N markers",
                                         f"{roh_df['N_markers'].mean():.1f}")
                if "Length_bp" in roh_df.columns:
                    summary_cols[3].metric("Mean length (bp)",
                                             f"{roh_df['Length_bp'].mean():,.0f}")

                # FROH per sample
                st.subheader("📊 F_ROH — Fraction of Genome in ROH")
                per_sample = roh_df.groupby("Sample").agg({
                    "N_markers": ["sum", "count", "mean"],
                }).reset_index()
                per_sample.columns = ["Sample", "Total_ROH_markers",
                                        "N_ROH_segments", "Mean_ROH_length"]
                per_sample["FROH"] = per_sample["Total_ROH_markers"] / n_markers

                st.dataframe(per_sample.style.format({
                    "FROH": "{:.4f}",
                    "Mean_ROH_length": "{:.1f}",
                }), use_container_width=True)

                # FROH distribution
                fig_froh = px.histogram(
                    per_sample, x="FROH", nbins=30,
                    title="FROH distribution across samples")
                fig_froh.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_froh, use_container_width=True)

                # FROH per sample bar
                fig_froh_bar = px.bar(
                    per_sample.sort_values("FROH", ascending=False).head(50),
                    x="Sample", y="FROH",
                    title="Top 50 samples by FROH",
                    color="FROH", color_continuous_scale="Reds",
                )
                fig_froh_bar.update_layout(template="plotly_white", height=500,
                                              xaxis_tickangle=90)
                st.plotly_chart(fig_froh_bar, use_container_width=True)

                # Length classification
                if classify_lengths:
                    st.subheader("📏 ROH Length Classes")
                    if "Length_bp" in roh_df.columns:
                        def _class(l):
                            if l < 500_000:
                                return "Short (<0.5 Mb)"
                            elif l < 1_500_000:
                                return "Medium (0.5-1.5 Mb)"
                            elif l < 5_000_000:
                                return "Long (1.5-5 Mb)"
                            else:
                                return "Very long (>5 Mb)"
                        roh_df["Length_class"] = roh_df["Length_bp"].apply(_class)
                    else:
                        def _class_m(n):
                            if n < 30:
                                return "Short (<30 markers)"
                            elif n < 100:
                                return "Medium (30-100)"
                            elif n < 300:
                                return "Long (100-300)"
                            else:
                                return "Very long (>300)"
                        roh_df["Length_class"] = roh_df["N_markers"].apply(_class_m)

                    class_counts = roh_df["Length_class"].value_counts().reset_index()
                    class_counts.columns = ["Length_class", "Count"]

                    fig_class = px.pie(
                        class_counts, values="Count", names="Length_class",
                        title="ROH length class distribution")
                    fig_class.update_layout(template="plotly_white", height=450)
                    st.plotly_chart(fig_class, use_container_width=True)

                # Per-sample bar
                st.subheader("N ROH segments per sample")
                per_sample_bar = per_sample.sort_values(
                    "N_ROH_segments", ascending=False).head(50)
                fig_ns = px.bar(per_sample_bar, x="Sample",
                                 y="N_ROH_segments",
                                 title="Number of ROH segments (top 50)")
                fig_ns.update_layout(template="plotly_white", height=450,
                                       xaxis_tickangle=90)
                st.plotly_chart(fig_ns, use_container_width=True)

                st.dataframe(roh_df.head(200), use_container_width=True)
                download_dataframe(roh_df, "roh_segments.csv", key="dl_sel_roh")
                download_dataframe(per_sample, "roh_per_sample.csv",
                                    key="dl_sel_roh_ps")

    # ═══════════════════════════════════════════
    # METHOD 4: Tajima's D (sliding window)
    # ═══════════════════════════════════════════
    elif method_sel == "Tajima's D (sliding window)":
        st.write(
            "**Tajima's D** — neutrality test comparing observed to expected "
            "genetic diversity. Values ≠ 0 suggest departure from neutrality."
        )
        st.info(
            "• **D < -2**: recent selective sweep or population expansion\n"
            "• **D > +2**: balancing selection or population contraction\n"
            "• **D ≈ 0**: neutral evolution"
        )

        td1, td2 = st.columns(2)
        with td1:
            win_td = st.slider("Window size (markers)", 20, 500, 100, 10, key="td_win")
        with td2:
            step_td = st.slider("Step size (markers)", 5, 100, 25, 5, key="td_step")

        if st.button("🚀 Compute Tajima's D", key="td_run"):
            with st.spinner("Computing Tajima's D windows..."):
                G = geno.fillna(1).values.astype(float)
                n_samples, n_markers = G.shape

                def tajimas_d(G_win):
                    """Compute Tajima's D for a window of SNPs."""
                    n = G_win.shape[0]
                    if n < 4:
                        return np.nan

                    # Number of segregating sites (polymorphic markers)
                    p = np.mean(G_win, axis=0) / 2
                    seg = np.sum((p > 0) & (p < 1))
                    if seg < 3:
                        return np.nan

                    # Watterson's theta (S / a1)
                    a1 = np.sum(1 / np.arange(1, n))
                    theta_w = seg / a1

                    # Nucleotide diversity (pi) — average pairwise differences
                    pi = 0
                    for j in range(G_win.shape[1]):
                        pj = p[j]
                        if 0 < pj < 1:
                            pi += 2 * pj * (1 - pj) * n / (n - 1)
                    pi = pi / G_win.shape[1] if G_win.shape[1] > 0 else 0

                    # Variance components
                    a2 = np.sum(1 / np.arange(1, n) ** 2)
                    b1 = (n + 1) / (3 * (n - 1))
                    b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
                    c1 = b1 - 1 / a1
                    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
                    e1 = c1 / a1
                    e2 = c2 / (a1 ** 2 + a2)

                    var_d = e1 * seg + e2 * seg * (seg - 1)
                    if var_d <= 0:
                        return np.nan

                    D = (pi - theta_w) / np.sqrt(var_d)
                    return D

                windows = []
                for start in range(0, n_markers - win_td + 1, step_td):
                    end = start + win_td
                    G_win = G[:, start:end]
                    D = tajimas_d(G_win)

                    entry = {
                        "Window_start": start,
                        "Window_end": end,
                        "Tajimas_D": D,
                    }

                    if marker_info is not None and "Pos" in marker_info.columns and "Chrom" in marker_info.columns:
                        if end - 1 < len(marker_info):
                            entry["Chrom"] = str(marker_info["Chrom"].iloc[start])
                            entry["Start_pos"] = float(marker_info["Pos"].iloc[start])
                            entry["End_pos"] = float(marker_info["Pos"].iloc[end - 1])
                            entry["Mid_pos"] = (entry["Start_pos"] + entry["End_pos"]) / 2

                    windows.append(entry)

            td_df = pd.DataFrame(windows).dropna(subset=["Tajimas_D"])

            if len(td_df) == 0:
                st.warning("No valid windows.")
            else:
                st.success(f"✅ Computed Tajima's D for **{len(td_df)}** windows.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean D", f"{td_df['Tajimas_D'].mean():.4f}")
                m2.metric("Windows D < -2",
                            f"{(td_df['Tajimas_D'] < -2).sum()}")
                m3.metric("Windows D > +2",
                            f"{(td_df['Tajimas_D'] > 2).sum()}")
                m4.metric("Windows |D| > 1",
                            f"{(td_df['Tajimas_D'].abs() > 1).sum()}")

                # Manhattan-style plot
                if "Chrom" in td_df.columns:
                    td_df["Chrom"] = td_df["Chrom"].astype(str)
                    fig_td = px.scatter(
                        td_df, x="Mid_pos" if "Mid_pos" in td_df.columns
                        else "Window_start",
                        y="Tajimas_D", color="Chrom",
                        title="Tajima's D across genome",
                        hover_data=[c for c in ["Chrom", "Start_pos", "End_pos"]
                                     if c in td_df.columns],
                    )
                else:
                    fig_td = px.scatter(
                        td_df, x="Window_start", y="Tajimas_D",
                        title="Tajima's D across windows")

                fig_td.add_hline(y=0, line_dash="dot", line_color="black")
                fig_td.add_hline(y=-2, line_dash="dash", line_color="red",
                                   annotation_text="D=-2 (sweep)")
                fig_td.add_hline(y=2, line_dash="dash", line_color="blue",
                                   annotation_text="D=+2 (balancing)")
                fig_td.update_layout(template="plotly_white", height=500,
                                       showlegend=False)
                st.plotly_chart(fig_td, use_container_width=True)

                fig_td_dist = px.histogram(
                    td_df, x="Tajimas_D", nbins=50,
                    title="Distribution of Tajima's D values")
                fig_td_dist.add_vline(x=0, line_dash="dot", line_color="black")
                fig_td_dist.add_vline(x=-2, line_dash="dash", line_color="red")
                fig_td_dist.add_vline(x=2, line_dash="dash", line_color="blue")
                fig_td_dist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_td_dist, use_container_width=True)

                st.subheader("Extreme windows")
                extreme = td_df[td_df["Tajimas_D"].abs() > 2].sort_values(
                    "Tajimas_D")
                st.dataframe(extreme, use_container_width=True)
                download_dataframe(td_df, "tajimas_d_windows.csv",
                                    key="dl_td")

                st.session_state["tajimas_d_df"] = td_df

    # ═══════════════════════════════════════════
    # METHOD 5: iHS-like
    # ═══════════════════════════════════════════
    elif method_sel == "iHS-like (Extended Haplotype Homozygosity)":
        st.write(
            "**iHS-like** — measures extended homozygosity around focal SNPs. "
            "Extreme values suggest recent positive selection."
        )
        st.info(
            "This is an approximation using genotype dosages "
            "(true iHS requires phased haplotypes)."
        )

        ih1, ih2 = st.columns(2)
        with ih1:
            window_flank = st.slider("Flanking window (markers)", 5, 100, 25,
                                       key="ihs_win")
        with ih2:
            top_pct_ihs = st.slider("Top % as outliers", 1, 20, 5, key="ihs_pct")

        if st.button("🚀 Compute iHS-like", key="ihs_run"):
            with st.spinner("Computing extended homozygosity..."):
                G = geno.fillna(1).values.astype(float)
                n_samples, n_markers = G.shape

                ihs_scores = []
                for m in range(n_markers):
                    start = max(0, m - window_flank)
                    end = min(n_markers, m + window_flank + 1)
                    window = G[:, start:end]

                    # Homozygosity: proportion of samples with 0 or 2 (not 1)
                    hom_freq = np.mean((window == 0) | (window == 2))

                    # Ancestral vs derived - simplified as MAF-based
                    p = G[:, m].mean() / 2
                    if p < 0.05 or p > 0.95:
                        ihs_scores.append(np.nan)
                        continue

                    # Log ratio: high hom → strong selection signal
                    if 0 < hom_freq < 1:
                        ihs = np.log(hom_freq / (1 - hom_freq))
                    else:
                        ihs = np.nan

                    ihs_scores.append(ihs)

                ihs_scores = np.array(ihs_scores)
                # Standardize |iHS|
                valid = ~np.isnan(ihs_scores)
                if valid.sum() > 0:
                    ihs_std = np.copy(ihs_scores)
                    ihs_std[valid] = (ihs_scores[valid] - ihs_scores[valid].mean()) / \
                                      ihs_scores[valid].std()
                else:
                    ihs_std = ihs_scores

            ihs_df = pd.DataFrame({
                "Marker": geno.columns,
                "iHS_like": ihs_std,
                "Abs_iHS": np.abs(ihs_std),
            })

            if marker_info is not None:
                ihs_df = ihs_df.merge(marker_info, on="Marker", how="left")

            ihs_df = ihs_df.dropna(subset=["iHS_like"])
            thr_ihs = np.percentile(ihs_df["Abs_iHS"], 100 - top_pct_ihs)
            ihs_df["Outlier"] = ihs_df["Abs_iHS"] >= thr_ihs

            n_ihs = int(ihs_df["Outlier"].sum())
            st.success(f"✅ Detected **{n_ihs}** iHS-like outliers.")

            fig_ihs = px.histogram(ihs_df, x="iHS_like", nbins=60,
                                     title="iHS-like distribution")
            fig_ihs.add_vline(x=-thr_ihs, line_dash="dash", line_color="red")
            fig_ihs.add_vline(x=thr_ihs, line_dash="dash", line_color="red")
            fig_ihs.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_ihs, use_container_width=True)

            if "Chrom" in ihs_df.columns:
                ihs_plot = ihs_df.copy()
                ihs_plot["Chrom"] = ihs_plot["Chrom"].astype(str)
                ihs_plot["Order"] = np.arange(len(ihs_plot))
                fig_ihs_man = px.scatter(
                    ihs_plot, x="Order", y="Abs_iHS", color="Chrom",
                    title="iHS-like Manhattan plot",
                    hover_data=["Marker", "Pos"] if "Pos" in ihs_plot.columns else ["Marker"],
                )
                fig_ihs_man.add_hline(y=thr_ihs, line_dash="dash",
                                        line_color="red")
                fig_ihs_man.update_layout(template="plotly_white", height=500,
                                            showlegend=False)
                st.plotly_chart(fig_ihs_man, use_container_width=True)

            st.dataframe(ihs_df[ihs_df["Outlier"]].sort_values(
                "Abs_iHS", ascending=False).head(50),
                use_container_width=True)
            download_dataframe(ihs_df, "ihs_outliers.csv", key="dl_ihs")

            st.session_state["ihs_outlier_df"] = ihs_df

    # ═══════════════════════════════════════════
    # METHOD 6: BayeScan-like
    # ═══════════════════════════════════════════
    elif method_sel == "BayeScan-like (Fst decomposition)":
        st.write(
            "**BayeScan-like** — decomposes Fst into locus-specific vs "
            "population-specific components. Loci with elevated locus-specific "
            "component are candidates for selection."
        )

        if meta is None:
            st.warning("Metadata required.")
        else:
            bs1, bs2 = st.columns(2)
            with bs1:
                pop_bs = st.selectbox("Population column",
                                       meta.columns.tolist(),
                                       index=meta.columns.tolist().index(global_pop_col)
                                       if global_pop_col in meta.columns.tolist() else 0,
                                       key="bs_pop")
            with bs2:
                sam_bs = st.selectbox("Sample ID column",
                                       meta.columns.tolist(),
                                       index=meta.columns.tolist().index(global_sam_col)
                                       if global_sam_col in meta.columns.tolist() else 0,
                                       key="bs_sam")

            pct_bs = st.slider("Top % as outliers", 1, 20, 5, key="bs_pct")

            if st.button("🚀 Run BayeScan-like", key="bs_run"):
                pop_map = dict(zip(meta[sam_bs].astype(str),
                                    meta[pop_bs].astype(str)))
                pops = sorted(set(pop_map.values()))

                with st.spinner("Computing locus-specific Fst deviations..."):
                    # Compute per-marker Fst
                    Ht = calc_het_exp(geno)
                    hs_list = []
                    for p in pops:
                        samples_p = [s for s in geno.index
                                       if pop_map.get(str(s)) == p]
                        if len(samples_p) < 2:
                            continue
                        Hs_p = calc_het_exp(geno.loc[samples_p])
                        hs_list.append(Hs_p)

                    if len(hs_list) < 2:
                        st.error("Need ≥ 2 populations.")
                        st.stop()

                    Hs = pd.concat(hs_list, axis=1).mean(axis=1)
                    Fst_marker = (Ht - Hs) / Ht.replace(0, np.nan)

                    # Global mean Fst
                    global_mean_fst = Fst_marker.mean()

                    # Locus-specific deviation (α coefficient in BayeScan)
                    # High α → locus under selection
                    alpha = np.log(Fst_marker / (1 - Fst_marker + 1e-10)) - \
                             np.log(global_mean_fst / (1 - global_mean_fst + 1e-10))

                    # Simulated posterior probability using magnitude
                    # (Real BayeScan uses MCMC — this is an approximation)
                    alpha_abs = np.abs(alpha)
                    posterior_prob = alpha_abs / (1 + alpha_abs)

                bs_df = pd.DataFrame({
                    "Marker": geno.columns,
                    "Fst": Fst_marker.values,
                    "Alpha": alpha.values,
                    "Posterior_prob": posterior_prob.values,
                })

                if marker_info is not None:
                    bs_df = bs_df.merge(marker_info, on="Marker", how="left")

                bs_df = bs_df.dropna(subset=["Alpha"])
                thr_bs = np.percentile(bs_df["Posterior_prob"], 100 - pct_bs)
                bs_df["Outlier"] = bs_df["Posterior_prob"] >= thr_bs

                # Interpretation
                bs_df["Selection_type"] = "Neutral"
                bs_df.loc[(bs_df["Outlier"]) & (bs_df["Alpha"] > 0), "Selection_type"] = "Positive (divergent)"
                bs_df.loc[(bs_df["Outlier"]) & (bs_df["Alpha"] < 0), "Selection_type"] = "Balancing"

                n_bs = int(bs_df["Outlier"].sum())
                n_pos = int((bs_df["Selection_type"] == "Positive (divergent)").sum())
                n_bal = int((bs_df["Selection_type"] == "Balancing").sum())

                st.success(f"✅ Detected **{n_bs}** candidate SNPs")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Total outliers", n_bs)
                mc2.metric("Positive selection", n_pos)
                mc3.metric("Balancing selection", n_bal)

                # Plot: α vs Fst (BayeScan-style)
                fig_bs = px.scatter(
                    bs_df, x="Fst", y="Alpha", color="Selection_type",
                    title="BayeScan-like plot: α vs Fst",
                    hover_data=["Marker"] +
                                (["Chrom", "Pos"] if "Chrom" in bs_df.columns else []),
                    color_discrete_map={
                        "Neutral": "lightgray",
                        "Positive (divergent)": "red",
                        "Balancing": "blue",
                    },
                )
                fig_bs.add_hline(y=0, line_dash="dot", line_color="black")
                fig_bs.update_layout(template="plotly_white", height=550)
                st.plotly_chart(fig_bs, use_container_width=True)

                st.dataframe(bs_df[bs_df["Outlier"]].sort_values(
                    "Posterior_prob", ascending=False).head(50),
                    use_container_width=True)
                download_dataframe(bs_df, "bayescan_like_outliers.csv",
                                    key="dl_bs")

                st.session_state["bayescan_outlier_df"] = bs_df

    # ═══════════════════════════════════════════
    # METHOD 7: LFMM-like (GEA)
    # ═══════════════════════════════════════════
    elif method_sel == "LFMM-like (Genotype-Environment Association)":
        st.write(
            "**LFMM-like** — tests associations between SNPs and environmental "
            "variables while controlling for population structure via PCs."
        )

        if meta is None:
            st.warning("Metadata required.")
        else:
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                env_col = st.selectbox(
                    "Environmental variable",
                    meta.select_dtypes(include=[np.number]).columns.tolist(),
                    key="lfmm_env")
            with lc2:
                sam_lfmm = st.selectbox(
                    "Sample ID column",
                    meta.columns.tolist(),
                    index=meta.columns.tolist().index(global_sam_col)
                    if global_sam_col in meta.columns.tolist() else 0,
                    key="lfmm_sam")
            with lc3:
                n_latent = st.slider("Latent factors (K)", 1, 10, 3,
                                       key="lfmm_k")

            pct_lfmm = st.slider("Top % as outliers", 1, 20, 5, key="lfmm_pct")

            if st.button("🚀 Run LFMM-like", key="lfmm_run"):
                env_map = dict(zip(meta[sam_lfmm].astype(str),
                                    meta[env_col].astype(float)))
                samples_use = [s for s in geno.index.astype(str)
                                if s in env_map and pd.notna(env_map.get(s))]

                X_geno = geno.loc[samples_use].values
                y_env = np.array([env_map[s] for s in samples_use])

                with st.spinner("Fitting latent factor model..."):
                    X_imp = impute_missing(pd.DataFrame(X_geno,
                                                          columns=geno.columns),
                                            "mean").values
                    X_sc = StandardScaler().fit_transform(X_imp)

                    # Estimate latent factors from PCA
                    pca_lf = PCA(n_components=n_latent)
                    latent = pca_lf.fit_transform(X_sc)

                    # For each SNP, fit: y_env ~ SNP + latent_factors
                    pvals = []
                    betas = []
                    for j in range(X_sc.shape[1]):
                        try:
                            snp = X_sc[:, j].reshape(-1, 1)
                            covariates = np.hstack([snp, latent])
                            reg = LinearRegression()
                            reg.fit(covariates, y_env)
                            beta = reg.coef_[0]

                            # F-test for SNP coefficient
                            y_pred = reg.predict(covariates)
                            residuals = y_env - y_pred
                            n_obs = len(y_env)
                            k = covariates.shape[1]
                            rss = np.sum(residuals ** 2)
                            se = np.sqrt(rss / (n_obs - k - 1))

                            # Simple t-test approximation
                            snp_var = np.var(snp)
                            if snp_var > 0 and se > 0:
                                t_stat = beta / (se / np.sqrt(snp_var * n_obs))
                                # p-value from t-distribution (approx via chi²)
                                pval = 2 * (1 - chi2.cdf(t_stat ** 2, df=1))
                            else:
                                pval = 1.0

                            pvals.append(pval)
                            betas.append(beta)
                        except Exception:
                            pvals.append(np.nan)
                            betas.append(np.nan)

                lfmm_df = pd.DataFrame({
                    "Marker": geno.columns,
                    "Beta": betas,
                    "P_value": pvals,
                    "NegLog10P": -np.log10(np.clip(pvals, 1e-300, 1)),
                })

                if marker_info is not None:
                    lfmm_df = lfmm_df.merge(marker_info, on="Marker", how="left")

                lfmm_df = lfmm_df.dropna(subset=["P_value"])
                thr_lfmm = np.percentile(lfmm_df["NegLog10P"], 100 - pct_lfmm)
                lfmm_df["Outlier"] = lfmm_df["NegLog10P"] >= thr_lfmm

                n_lfmm = int(lfmm_df["Outlier"].sum())
                st.success(f"✅ Detected **{n_lfmm}** SNPs associated with **{env_col}**.")

                # Manhattan
                if "Chrom" in lfmm_df.columns:
                    lp = lfmm_df.copy()
                    lp["Chrom"] = lp["Chrom"].astype(str)
                    lp["Order"] = np.arange(len(lp))
                    fig_lfmm = px.scatter(
                        lp, x="Order", y="NegLog10P", color="Chrom",
                        title=f"LFMM-like Manhattan plot for {env_col}",
                        hover_data=["Marker", "Pos"] if "Pos" in lp.columns else ["Marker"],
                    )
                    fig_lfmm.add_hline(y=thr_lfmm, line_dash="dash",
                                        line_color="red")
                    fig_lfmm.update_layout(template="plotly_white",
                                             height=550, showlegend=False)
                    st.plotly_chart(fig_lfmm, use_container_width=True)

                fig_lfmm_dist = px.histogram(
                    lfmm_df, x="NegLog10P", nbins=50,
                    title="-log10(p) distribution")
                fig_lfmm_dist.add_vline(x=thr_lfmm, line_dash="dash",
                                          line_color="red")
                fig_lfmm_dist.update_layout(template="plotly_white",
                                              height=400)
                st.plotly_chart(fig_lfmm_dist, use_container_width=True)

                st.dataframe(lfmm_df[lfmm_df["Outlier"]].sort_values(
                    "NegLog10P", ascending=False).head(50),
                    use_container_width=True)
                download_dataframe(lfmm_df, "lfmm_outliers.csv", key="dl_lfmm")

                st.session_state["lfmm_outlier_df"] = lfmm_df

    # ═══════════════════════════════════════════
    # METHOD 8: Selective Sweep (CLR-like)
    # ═══════════════════════════════════════════
    elif method_sel == "Selective Sweep (CLR-like)":
        st.write(
            "**CLR-like sweep detection** — sliding-window statistic based on "
            "the deviation of allele frequency spectrum from neutral expectation."
        )

        cs1, cs2 = st.columns(2)
        with cs1:
            win_cs = st.slider("Window size (markers)", 20, 500, 100, 10,
                                 key="cs_win")
        with cs2:
            step_cs = st.slider("Step (markers)", 5, 100, 25, 5, key="cs_step")

        if st.button("🚀 Detect selective sweeps", key="cs_run"):
            with st.spinner("Scanning for selective sweeps..."):
                G = geno.fillna(1).values.astype(float)
                n_samples, n_markers = G.shape

                sweep_scores = []

                for start in range(0, n_markers - win_cs + 1, step_cs):
                    end = start + win_cs
                    G_win = G[:, start:end]

                    # Allele frequency spectrum
                    p_arr = np.mean(G_win, axis=0) / 2
                    maf_arr = np.minimum(p_arr, 1 - p_arr)

                    # Under selection: excess of rare or high-frequency alleles
                    n_rare = np.sum(maf_arr < 0.1)
                    n_common = np.sum(maf_arr >= 0.3)

                    # Expected under neutral (Wright-Fisher): ~equal
                    expected_ratio = 0.5
                    total_valid = n_rare + n_common
                    if total_valid == 0:
                        sweep_scores.append({
                            "Window_start": start,
                            "Window_end": end,
                            "CLR_stat": np.nan,
                        })
                        continue

                    obs_ratio = n_rare / total_valid

                    # Composite likelihood-like ratio
                    if 0 < obs_ratio < 1:
                        clr = 2 * total_valid * (
                            obs_ratio * np.log(obs_ratio / expected_ratio) +
                            (1 - obs_ratio) * np.log((1 - obs_ratio) / (1 - expected_ratio))
                        )
                    else:
                        clr = np.nan

                    entry = {
                        "Window_start": start,
                        "Window_end": end,
                        "CLR_stat": clr,
                        "N_rare": n_rare,
                        "N_common": n_common,
                    }

                    if marker_info is not None and "Chrom" in marker_info.columns:
                        if end - 1 < len(marker_info):
                            entry["Chrom"] = str(marker_info["Chrom"].iloc[start])
                            if "Pos" in marker_info.columns:
                                entry["Start_pos"] = float(marker_info["Pos"].iloc[start])
                                entry["End_pos"] = float(marker_info["Pos"].iloc[end - 1])
                                entry["Mid_pos"] = (entry["Start_pos"] + entry["End_pos"]) / 2

                    sweep_scores.append(entry)

                sweep_df = pd.DataFrame(sweep_scores).dropna(subset=["CLR_stat"])

            if len(sweep_df) == 0:
                st.warning("No valid windows.")
            else:
                # Threshold at 95th percentile
                thr_cs = np.percentile(sweep_df["CLR_stat"], 95)
                sweep_df["Sweep_candidate"] = sweep_df["CLR_stat"] >= thr_cs

                n_sweep = int(sweep_df["Sweep_candidate"].sum())
                st.success(f"✅ Detected **{n_sweep}** candidate sweep windows.")

                # Manhattan
                if "Chrom" in sweep_df.columns:
                    fig_cs = px.scatter(
                        sweep_df,
                        x="Mid_pos" if "Mid_pos" in sweep_df.columns else "Window_start",
                        y="CLR_stat", color="Chrom",
                        title="CLR-like sweep statistic across genome",
                        hover_data=["Window_start", "N_rare", "N_common"],
                    )
                else:
                    fig_cs = px.scatter(sweep_df, x="Window_start", y="CLR_stat",
                                          title="CLR-like sweep statistic")

                fig_cs.add_hline(y=thr_cs, line_dash="dash", line_color="red",
                                   annotation_text=f"95th percentile ({thr_cs:.2f})")
                fig_cs.update_layout(template="plotly_white", height=500,
                                       showlegend=False)
                st.plotly_chart(fig_cs, use_container_width=True)

                fig_cs_dist = px.histogram(sweep_df, x="CLR_stat", nbins=50,
                                              title="CLR distribution")
                fig_cs_dist.add_vline(x=thr_cs, line_dash="dash",
                                        line_color="red")
                fig_cs_dist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_cs_dist, use_container_width=True)

                st.subheader("Top candidate sweep windows")
                st.dataframe(sweep_df[sweep_df["Sweep_candidate"]].sort_values(
                    "CLR_stat", ascending=False).head(50),
                    use_container_width=True)
                download_dataframe(sweep_df, "selective_sweeps.csv",
                                    key="dl_cs")

    # ═══════════════════════════════════════════
    # METHOD 9: Combined outlier detection
    # ═══════════════════════════════════════════
    else:  # Combined
        st.write(
            "**Combined outlier detection** — combines results from previously-run "
            "methods to identify robust selection candidates. Run individual methods "
            "first, then use this to find SNPs detected by multiple approaches."
        )

        available_methods = []
        if "fst_outlier_df" in st.session_state:
            available_methods.append("Fst")
        if "pcadapt_outlier_df" in st.session_state:
            available_methods.append("PCAdapt")
        if "ihs_outlier_df" in st.session_state:
            available_methods.append("iHS")
        if "bayescan_outlier_df" in st.session_state:
            available_methods.append("BayeScan")
        if "lfmm_outlier_df" in st.session_state:
            available_methods.append("LFMM")

        if len(available_methods) < 2:
            st.warning(
                "⚠️ Please run at least 2 different selection methods first "
                "(Fst, PCAdapt, iHS, BayeScan, or LFMM). "
                f"Currently available: {available_methods if available_methods else 'None'}"
            )
        else:
            selected_methods = st.multiselect(
                "Methods to combine",
                available_methods,
                default=available_methods,
                key="comb_methods",
            )

            min_methods = st.slider(
                "Min number of methods for consensus",
                1, len(selected_methods), max(2, len(selected_methods) // 2),
                key="comb_min",
            )

            if st.button("🚀 Combine outliers", key="comb_run"):
                all_outliers = {}
                for m in selected_methods:
                    key = f"{m.lower()}_outlier_df"
                    if m == "Fst":
                        df = st.session_state["fst_outlier_df"]
                    elif m == "PCAdapt":
                        df = st.session_state["pcadapt_outlier_df"]
                    elif m == "iHS":
                        df = st.session_state["ihs_outlier_df"]
                    elif m == "BayeScan":
                        df = st.session_state["bayescan_outlier_df"]
                    elif m == "LFMM":
                        df = st.session_state["lfmm_outlier_df"]

                    outlier_snps = set(df[df["Outlier"]]["Marker"].astype(str))
                    all_outliers[m] = outlier_snps

                # Count how many methods detected each SNP
                all_snps = set()
                for s in all_outliers.values():
                    all_snps |= s

                snp_counts = {snp: 0 for snp in all_snps}
                snp_methods = {snp: [] for snp in all_snps}
                for m, snps in all_outliers.items():
                    for snp in snps:
                        snp_counts[snp] += 1
                        snp_methods[snp].append(m)

                combined_df = pd.DataFrame({
                    "Marker": list(snp_counts.keys()),
                    "N_methods": list(snp_counts.values()),
                    "Methods": [",".join(snp_methods[s])
                                 for s in snp_counts.keys()],
                })
                combined_df["Robust_candidate"] = combined_df["N_methods"] >= min_methods

                if marker_info is not None:
                    combined_df = combined_df.merge(
                        marker_info, on="Marker", how="left")

                combined_df = combined_df.sort_values("N_methods", ascending=False)

                n_robust = int(combined_df["Robust_candidate"].sum())
                st.success(f"✅ **{n_robust}** SNPs detected by ≥ {min_methods} methods.")

                # Method overlap
                cc1, cc2 = st.columns(2)
                cc1.metric("Total unique outliers", len(combined_df))
                cc2.metric(f"Robust (≥ {min_methods} methods)", n_robust)

                # Method count distribution
                fig_meth = px.histogram(
                    combined_df, x="N_methods",
                    title="Distribution of methods detecting each SNP",
                    nbins=len(selected_methods) + 1,
                )
                fig_meth.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_meth, use_container_width=True)

                # Venn-like: methods overlap
                if len(selected_methods) <= 4:
                    st.markdown("### Method Overlap Matrix")
                    overlap_mat = np.zeros((len(selected_methods),
                                              len(selected_methods)))
                    for i, m1 in enumerate(selected_methods):
                        for j, m2 in enumerate(selected_methods):
                            overlap = all_outliers[m1] & all_outliers[m2]
                            overlap_mat[i, j] = len(overlap)

                    fig_ov = px.imshow(
                        overlap_mat,
                        x=selected_methods, y=selected_methods,
                        text_auto=True, color_continuous_scale="Greens",
                        title="Number of SNPs detected by each method pair")
                    fig_ov.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_ov, use_container_width=True)

                st.subheader("Robust selection candidates")
                st.dataframe(combined_df[combined_df["Robust_candidate"]].head(100),
                              use_container_width=True)
                download_dataframe(combined_df,
                                    "combined_selection_outliers.csv",
                                    key="dl_comb")


# =========================================================
# TAB 4 — UMAP
# =========================================================
with tab_um:
    st.subheader("UMAP Non-linear Dimensionality Reduction")
    st.write("Visualize sample structure with UMAP.")

    c1, c2, c3 = st.columns(3)
    with c1:
        nn = st.slider("N neighbors", 2, 100, 15, key="um_nn")
    with c2:
        md = st.slider("Min distance", 0.0, 1.0, 0.1, 0.01, key="um_md")
    with c3:
        nc_um = st.radio("Components", [2, 3], key="um_nc")

    color_col_u = None
    if meta is not None:
        color_col_u = st.selectbox("Color by (metadata)",
                                     ["None"] + meta.columns.tolist(),
                                     index=meta.columns.tolist().index(global_pop_col) + 1
                                     if global_pop_col in meta.columns.tolist() else 0,
                                     key="um_color")
        sam_col_u = st.selectbox("Sample ID column",
                                  meta.columns.tolist(),
                                  index=meta.columns.tolist().index(global_sam_col)
                                  if global_sam_col in meta.columns.tolist() else 0,
                                  key="um_sam")

    if st.button("🚀 Run UMAP", use_container_width=True, key="um_run"):
        try:
            from umap import UMAP
        except ImportError:
            st.error("UMAP not installed.")
            st.stop()

        imp = impute_missing(geno, "mean").values
        X_sc = StandardScaler().fit_transform(imp)

        with st.spinner("Running UMAP..."):
            reducer = UMAP(n_neighbors=nn, min_dist=md,
                            n_components=nc_um, random_state=42)
            emb = reducer.fit_transform(X_sc)

        um_df = pd.DataFrame(emb, columns=[f"UMAP_{i+1}" for i in range(nc_um)])
        um_df["Sample"] = geno.index.astype(str)

        if color_col_u and color_col_u != "None":
            m_sub = meta[[sam_col_u, color_col_u]].drop_duplicates()
            m_sub[sam_col_u] = m_sub[sam_col_u].astype(str)
            um_df = um_df.merge(m_sub, left_on="Sample",
                                  right_on=sam_col_u, how="left")

        color_arg = color_col_u if (color_col_u and color_col_u != "None") else None

        if nc_um == 2:
            fig_um = px.scatter(um_df, x="UMAP_1", y="UMAP_2",
                                 color=color_arg, hover_data=["Sample"],
                                 title="UMAP 2D projection")
            fig_um.update_traces(marker=dict(size=9,
                                               line=dict(width=0.5,
                                                          color="darkslategrey")))
        else:
            fig_um = px.scatter_3d(um_df, x="UMAP_1", y="UMAP_2", z="UMAP_3",
                                    color=color_arg, hover_data=["Sample"],
                                    title="UMAP 3D projection")
            fig_um.update_traces(marker=dict(size=5))

        fig_um.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig_um, use_container_width=True)

        download_plotly_html(fig_um, "umap_projection.html", key="dl_um_html")
        download_dataframe(um_df, "umap_embedding.csv", key="dl_um_csv")