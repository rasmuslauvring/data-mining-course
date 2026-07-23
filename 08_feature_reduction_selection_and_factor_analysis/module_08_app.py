from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
SAMPLE_DATASETS = {
    "Synthetic customer feature selection sample": DATA_DIR / "feature_selection_sample.csv",
    "Wine classification sample": DATA_DIR / "wine_classification_sample.csv",
}

st.set_page_config(page_title="Feature Reduction, Selection and Factor Analysis", layout="wide")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": "#F8F9FB",
    "axes.facecolor": "#F8F9FB",
    "axes.edgecolor": "#D0D7DE",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: #F7F9FC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 12px;}
.small-note {font-size: 0.88rem; color: #475569;}
</style>
""", unsafe_allow_html=True)

st.title("Feature Reduction, Selection and Factor Analysis")
st.caption("Workflow: upload a dataset, choose target and numeric features, then compare filter, wrapper, embedded, PCA and factor-analysis outputs.")

[tab_feature_reduction_selection_fa] = st.tabs(["Feature Reduction, Selection and Factor Analysis"])

# ----------------------------- helpers -----------------------------
def coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    converted = df.copy()
    for col in converted.columns:
        s0 = converted[col]
        if not (pd.api.types.is_object_dtype(s0) or pd.api.types.is_string_dtype(s0)):
            continue
        s = s0.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA, "NA": pd.NA})
        non_missing = s.notna()
        if non_missing.sum() == 0:
            continue
        numeric_like = s[non_missing].str.match(r"^[+-]?[0-9\s.,]+$", na=False).mean()
        if numeric_like < min_ratio:
            continue
        candidates = [
            pd.to_numeric(s, errors="coerce"),
            pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce"),
            pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce"),
        ]
        best = max(candidates, key=lambda x: x[non_missing].notna().mean())
        if best[non_missing].notna().mean() >= min_ratio:
            converted[col] = best
    return converted


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    separators = [None, ";", ",", "\t", "|"]
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    best_df, best_score, last_exc = None, (-1, -1), None
    for encoding in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, engine="python", encoding=encoding)
                score = (df.shape[1], df.shape[0])
                if score > best_score:
                    best_df, best_score = df, score
            except Exception as exc:
                last_exc = exc
    if best_df is None:
        raise ValueError(f"Could not parse CSV file. Last error: {last_exc}")
    return coerce_numeric_like_columns(best_df)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def highlight_null_values(table: pd.DataFrame):
    """Return a styled dataframe where null values are highlighted in light red."""
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return table

    def _style_null(value):
        return "background-color: #FFE0E0; color: #7F1D1D; font-weight: 600;" if pd.isna(value) else ""

    styler = table.style
    if hasattr(styler, "map"):
        return styler.map(_style_null)
    return styler.applymap(_style_null)


def display_dataframe(table: pd.DataFrame, **kwargs):
    """Display dataframes consistently and highlight null values when present."""
    st.dataframe(highlight_null_values(table), **kwargs)


def make_feature_matrix(df: pd.DataFrame, features: list[str]):
    X_raw = df[features].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imp)
    return X_raw, X_imp, X_scaled


def plot_barh(table, label_col, value_col, title, top_n=20):
    fig, ax = plt.subplots(figsize=(8, 5))
    if table.empty:
        ax.text(0.5, 0.5, "No results available", ha="center", va="center")
        ax.set_axis_off()
    else:
        chart = table.head(top_n).iloc[::-1]
        ax.barh(chart[label_col].astype(str), chart[value_col])
        ax.set_title(title)
        ax.set_xlabel(value_col)
    fig.tight_layout()
    return fig


def plot_corr_heatmap(df, features):
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(features) < 2:
        ax.text(0.5, 0.5, "Need at least two numeric features", ha="center", va="center")
        ax.set_axis_off()
    else:
        corr = df[features].corr()
        sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, center=0, cmap="coolwarm", annot=len(features) <= 12, fmt=".2f", linewidths=.4)
        ax.set_title("Correlation heatmap")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_pca_variance(explained, threshold):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, explained["explained_variance_ratio"], label="Individual")
    ax.plot(x, explained["cumulative_explained_variance"], marker="o", label="Cumulative")
    ax.axhline(threshold, linestyle="--", linewidth=1.5, label=f"{threshold:.0%} reference")
    ax.set_title("PCA explained variance")
    ax.set_xlabel("Component number")
    ax.set_ylabel("Variance ratio")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_scree(eigenvalues):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(eigenvalues) + 1)
    ax.plot(x, eigenvalues, marker="o")
    ax.axhline(1.0, linestyle="--", linewidth=1.2, label="Eigenvalue = 1")
    ax.set_title("Scree plot: PCA eigenvalues")
    ax.set_xlabel("Component number")
    ax.set_ylabel("Eigenvalue")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pc_scatter(pc_df, color_col=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    if color_col and color_col in pc_df.columns:
        sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue=color_col, ax=ax, s=55)
    else:
        sns.scatterplot(data=pc_df, x="PC1", y="PC2", ax=ax, s=55)
    ax.set_title("2D PCA projection")
    fig.tight_layout()
    return fig


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(np.dot(Phi.T, Lambda**3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return np.dot(Phi, R)


with tab_feature_reduction_selection_fa:
    st.subheader("Batch feature reduction, selection and factor analysis")
    st.caption(
        "Workflow: upload a dataset, choose the target and numeric features, then compare filter, wrapper, embedded, PCA and factor-analysis outputs."
    )

    # ----------------------------- state and input -----------------------------
    if "fsr_uploader_key_version" not in st.session_state:
        st.session_state["fsr_uploader_key_version"] = 0
    if "fsr_run_requested" not in st.session_state:
        st.session_state["fsr_run_requested"] = False

    uploader_key = f"fsr_uploader_{st.session_state['fsr_uploader_key_version']}"

    uploader_col, sample_col, download_col, clear_col = st.columns([5.5, 2.6, 1.5, 1], gap="small", vertical_alignment="bottom")
    with uploader_col:
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"], key=uploader_key)
    with sample_col:
        sample_choice = st.selectbox("Sample dataset", list(SAMPLE_DATASETS.keys()))
        sample_path = SAMPLE_DATASETS[sample_choice]
    with download_col:
        st.write("")
        st.download_button("Download sample", data=sample_path.read_bytes(), file_name=sample_path.name, mime="text/csv", width="stretch", icon=":material/download:")
    with clear_col:
        st.write("")
        if st.button("Clear", width="stretch", icon=":material/clear:", disabled=uploaded_file is None):
            st.session_state["fsr_uploader_key_version"] += 1
            st.session_state["fsr_run_requested"] = False
            st.rerun()

    if uploaded_file is None:
        st.info("Upload a CSV/XLSX file. You can also download a sample dataset above and upload it.")
        st.stop()

    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            excel = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", excel.sheet_names)
            df = coerce_numeric_like_columns(pd.read_excel(uploaded_file, sheet_name=sheet_name))
        else:
            df = read_csv_flexible(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        st.stop()

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("No suitable numeric feature set found. The app needs at least two numeric columns.")
        st.stop()

    left_col, right_col = st.columns([1, 2], gap="large")
    with left_col:
        st.subheader("Setup")
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns):,}")
        st.write(f"**Numeric columns:** {len(numeric_columns):,}")
        st.write(f"**Missing cells:** {int(df.isna().sum().sum()):,}")

        default_target = "target_purchase" if "target_purchase" in df.columns else ("target" if "target" in df.columns else "None")
        target_options = ["None"] + df.columns.tolist()
        target_choice = st.selectbox("Optional target column", target_options, index=target_options.index(default_target) if default_target in target_options else 0)
        target_col = None if target_choice == "None" else target_choice

        default_features = [c for c in numeric_columns if c != target_col]
        feature_cols = st.multiselect("Numeric features to analyze", numeric_columns, default=default_features)
        st.caption("Use numeric predictors. Median imputation and scaling are applied internally where required.")

        if target_col:
            task_type = st.radio("Target task type", ["Classification", "Regression"], horizontal=True)
        else:
            task_type = "Unsupervised"
            st.caption("No target selected: model-based feature selection will be disabled, but PCA and factor analysis can run.")

        st.divider()
        st.write("**Method controls**")
        variance_threshold = st.number_input("Low-variance threshold", min_value=0.0, value=0.01, step=0.01, format="%.4f")
        k_best = st.slider("Number of top features/components to keep", 1, max(1, len(feature_cols)), min(6, max(1, len(feature_cols))))
        corr_threshold = st.slider("High-correlation reference |r|", 0.50, 1.00, 0.90, 0.05)
        pca_threshold = st.slider("PCA cumulative variance target", 0.50, 0.99, 0.90, 0.01)
        factor_n = st.slider("Number of factors", 1, max(1, min(6, len(feature_cols))), min(3, max(1, min(6, len(feature_cols)))))
        use_varimax = st.checkbox("Rotate factor loadings with varimax", value=True)

        st.subheader("Dataset preview")
        display_dataframe(df.head(10), width="stretch")
        run_clicked = st.button("Run analysis", type="primary", width="stretch")
        if run_clicked:
            st.session_state["fsr_run_requested"] = True

    if not st.session_state["fsr_run_requested"]:
        st.stop()

    if len(feature_cols) < 2:
        st.warning("Select at least two numeric features before running the analysis.")
        st.stop()

    with right_col:
        st.subheader("Results")
        X_raw, X_imp, X_scaled = make_feature_matrix(df, feature_cols)
        rows_used = len(df)

        y = None
        if target_col:
            target_series = df[target_col]
            valid_target = target_series.notna()
            if valid_target.sum() < 10:
                st.warning("The selected target has too few non-missing rows for model-based methods.")
                target_col = None
            else:
                X_raw = X_raw.loc[valid_target]
                X_imp, X_scaled = make_feature_matrix(df.loc[valid_target], feature_cols)[1:]
                y = target_series.loc[valid_target]
                rows_used = valid_target.sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows used", f"{rows_used:,}")
        c2.metric("Input features", f"{len(feature_cols):,}")
        c3.metric("Missing cells", f"{int(df[feature_cols].isna().sum().sum()):,}")
        c4.metric("Target", target_col if target_col else "None")

        tabs = st.tabs(["Data checks", "Feature selection", "PCA reduction", "Factor analysis", "Processed output"])

        with tabs[0]:
            st.caption("Fast checks before selection or reduction. Correlation is useful for review, but it should not be the only removal rule.")
            q = pd.DataFrame({
                "feature": feature_cols,
                "dtype": [str(df[c].dtype) for c in feature_cols],
                "missing_count": df[feature_cols].isna().sum().values,
                "missing_percent": (df[feature_cols].isna().mean().values * 100).round(2),
                "variance_raw": df[feature_cols].var(numeric_only=True).values,
                "unique_values": df[feature_cols].nunique().values,
            }).sort_values("variance_raw")
            display_dataframe(q, width="stretch")
            ch1, ch2 = st.columns(2, gap="large")
            with ch1:
                fig = plot_barh(q.sort_values("variance_raw", ascending=False), "feature", "variance_raw", "Feature variance", top_n=20)
                st.pyplot(fig); plt.close(fig)
            with ch2:
                show_cols = feature_cols[:18]
                fig = plot_corr_heatmap(df, show_cols)
                st.pyplot(fig); plt.close(fig)

            corr = df[feature_cols].corr().abs()
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            pairs = corr.where(mask).stack().reset_index()
            pairs.columns = ["feature_a", "feature_b", "abs_correlation"]
            pairs = pairs[pairs["abs_correlation"] >= corr_threshold].sort_values("abs_correlation", ascending=False)
            st.write("**High-correlation pairs**")
            display_dataframe(pairs, width="stretch")

        with tabs[1]:
            st.caption("Feature selection keeps original variables. The app includes filter, wrapper and embedded examples covered in the class material.")
            selection_tables = []
            if target_col is None or y is None:
                st.info("Select a target column to run ANOVA/F-test, RFE, LASSO/Ridge, Random Forest and Boruta-style checks.")
            else:
                if task_type == "Classification":
                    y_model = y.astype("category").cat.codes if not pd.api.types.is_numeric_dtype(y) else y
                    if pd.Series(y_model).nunique() < 2:
                        st.warning("The target must contain at least two classes.")
                        st.stop()
                    score_func = f_classif
                    base_estimator = LogisticRegression(max_iter=3000, solver="lbfgs")
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
                    metric_name = "accuracy"
                else:
                    y_model = pd.to_numeric(y, errors="coerce")
                    good = pd.Series(y_model).notna()
                    X_scaled = X_scaled[good]
                    X_imp = X_imp[good]
                    y_model = y_model[good]
                    score_func = f_regression
                    base_estimator = RidgeCV(alphas=np.logspace(-3, 3, 15))
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
                    metric_name = "r2"

                k_eff = min(k_best, len(feature_cols))
                var = X_raw.var(numeric_only=True).reindex(feature_cols).fillna(0)
                variance_kept = var[var >= variance_threshold].index.tolist()

                skb = SelectKBest(score_func=score_func, k="all").fit(X_scaled, y_model)
                filter_scores = pd.DataFrame({"feature": feature_cols, "score": skb.scores_, "p_value": skb.pvalues_}).replace([np.inf, -np.inf], np.nan).sort_values("score", ascending=False)
                filter_selected = filter_scores.head(k_eff)["feature"].tolist()

                rfe = RFE(base_estimator, n_features_to_select=k_eff)
                rfe.fit(X_scaled, y_model)
                rfe_table = pd.DataFrame({"feature": feature_cols, "rfe_rank": rfe.ranking_, "selected": rfe.support_}).sort_values(["rfe_rank", "feature"])
                rfe_selected = rfe_table.loc[rfe_table["selected"], "feature"].tolist()

                if task_type == "Classification":
                    y_binary = (pd.Series(y_model).astype(int) == pd.Series(y_model).astype(int).mode()[0]).astype(int)
                    lasso = LassoCV(cv=3, random_state=RANDOM_STATE, max_iter=10000).fit(X_scaled, y_binary)
                    ridge = RidgeCV(alphas=np.logspace(-3, 3, 15), cv=3).fit(X_scaled, y_binary)
                else:
                    lasso = LassoCV(cv=3, random_state=RANDOM_STATE, max_iter=10000).fit(X_scaled, y_model)
                    ridge = RidgeCV(alphas=np.logspace(-3, 3, 15), cv=3).fit(X_scaled, y_model)
                lasso_table = pd.DataFrame({"feature": feature_cols, "lasso_coefficient": lasso.coef_, "selected_non_zero": np.abs(lasso.coef_) > 1e-6}).sort_values("lasso_coefficient", key=lambda s: abs(s), ascending=False)
                ridge_table = pd.DataFrame({"feature": feature_cols, "ridge_coefficient": ridge.coef_}).sort_values("ridge_coefficient", key=lambda s: abs(s), ascending=False)

                rf_model.fit(X_imp, y_model)
                rf_table = pd.DataFrame({"feature": feature_cols, "importance": rf_model.feature_importances_}).sort_values("importance", ascending=False)
                rf_selected = rf_table.head(k_eff)["feature"].tolist()

                rng = np.random.default_rng(RANDOM_STATE)
                X_shadow = pd.DataFrame(X_imp, columns=feature_cols)
                for col in feature_cols:
                    X_shadow[f"shadow_{col}"] = rng.permutation(X_shadow[col].values)
                shadow_model = RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1) if task_type == "Classification" else RandomForestRegressor(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1)
                shadow_model.fit(X_shadow, y_model)
                boruta_imp = pd.DataFrame({"feature": X_shadow.columns, "importance": shadow_model.feature_importances_})
                shadow_max = boruta_imp.loc[boruta_imp.feature.str.startswith("shadow_"), "importance"].max()
                boruta_table = boruta_imp[~boruta_imp.feature.str.startswith("shadow_")].copy()
                boruta_table["shadow_max"] = shadow_max
                boruta_table["decision"] = np.where(boruta_table["importance"] > shadow_max, "confirmed", "tentative_or_rejected")
                boruta_table = boruta_table.sort_values("importance", ascending=False)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_model, test_size=.30, random_state=RANDOM_STATE, stratify=y_model if task_type == "Classification" and pd.Series(y_model).nunique() > 1 else None)
                model_scores = []
                subsets = {"All features": feature_cols, "Filter top k": filter_selected, "Wrapper RFE": rfe_selected, "Embedded RF top k": rf_selected}
                for name, subset in subsets.items():
                    idx = [feature_cols.index(c) for c in subset]
                    model = LogisticRegression(max_iter=3000) if task_type == "Classification" else RidgeCV(alphas=np.logspace(-3, 3, 15))
                    model.fit(X_train[:, idx], y_train)
                    pred = model.predict(X_test[:, idx])
                    val = accuracy_score(y_test, pred) if task_type == "Classification" else r2_score(y_test, pred)
                    model_scores.append({"method": name, "features": len(subset), metric_name: round(float(val), 4), "selected_features": ", ".join(subset)})
                model_scores = pd.DataFrame(model_scores)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Filter top k", len(filter_selected))
                m2.metric("RFE selected", len(rfe_selected))
                m3.metric("LASSO non-zero", int(lasso_table["selected_non_zero"].sum()))
                m4.metric("Boruta confirmed", int((boruta_table["decision"] == "confirmed").sum()))

                st.write("**Model comparison**")
                display_dataframe(model_scores, width="stretch")
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    fig = plot_barh(filter_scores, "feature", "score", "Filter scores", top_n=20)
                    st.pyplot(fig); plt.close(fig)
                with col_b:
                    fig = plot_barh(rf_table, "feature", "importance", "Random Forest importance", top_n=20)
                    st.pyplot(fig); plt.close(fig)

                st.write("**Detailed selection outputs**")
                t1, t2, t3, t4, t5 = st.tabs(["Filter", "Wrapper RFE", "LASSO", "Ridge", "Boruta-style"])
                with t1: display_dataframe(filter_scores, width="stretch")
                with t2: display_dataframe(rfe_table, width="stretch")
                with t3: display_dataframe(lasso_table, width="stretch"); st.caption(f"LASSO alpha: {lasso.alpha_:.5f}. Non-zero coefficients indicate selected candidates.")
                with t4: display_dataframe(ridge_table, width="stretch"); st.caption(f"Ridge alpha: {ridge.alpha_:.5f}. Ridge shrinks coefficients but usually keeps all variables.")
                with t5: display_dataframe(boruta_table, width="stretch"); st.caption("Boruta-style check: original features are compared with randomized shadow features.")

        with tabs[2]:
            st.caption("PCA creates synthetic components from numeric features. It is feature reduction, not feature selection.")
            pca = PCA().fit(X_scaled)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            n_threshold = int(np.argmax(cumulative >= pca_threshold) + 1)
            n_kaiser = int(np.sum(pca.explained_variance_ > 1)) or 1
            explained = pd.DataFrame({
                "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_explained_variance": cumulative,
                "eigenvalue": pca.explained_variance_,
            })
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Components for target variance", n_threshold)
            p2.metric("Variance captured", f"{cumulative[n_threshold-1]:.1%}")
            p3.metric("Kaiser > 1", n_kaiser)
            p4.metric("Original features", len(feature_cols))
            display_dataframe(explained, width="stretch")
            col_a, col_b = st.columns(2, gap="large")
            with col_a:
                fig = plot_pca_variance(explained, pca_threshold)
                st.pyplot(fig); plt.close(fig)
            with col_b:
                fig = plot_scree(pca.explained_variance_)
                st.pyplot(fig); plt.close(fig)

            if len(feature_cols) >= 2:
                pca2 = PCA(n_components=2).fit(X_scaled)
                comps = pca2.transform(X_scaled)
                pc_df = pd.DataFrame({"PC1": comps[:, 0], "PC2": comps[:, 1]})
                if target_col and y is not None:
                    pc_df[target_col] = y.astype(str).values
                fig = plot_pc_scatter(pc_df, target_col if target_col else None)
                st.pyplot(fig); plt.close(fig)
                loadings = pd.DataFrame(pca.components_.T[:, :min(5, len(feature_cols))], index=feature_cols, columns=[f"PC{i+1}_loading" for i in range(min(5, len(feature_cols)))])
                st.write("**Component loadings**")
                display_dataframe(loadings, width="stretch")

        with tabs[3]:
            st.caption("Factor Analysis estimates latent factors behind observed variables. Loadings indicate which variables move with the same hidden factor.")
            if len(feature_cols) < factor_n:
                st.warning("The number of factors cannot exceed the number of selected features.")
            else:
                fa = FactorAnalysis(n_components=factor_n, random_state=RANDOM_STATE)
                factor_scores = fa.fit_transform(X_scaled)
                loadings = fa.components_.T
                if use_varimax and factor_n > 1:
                    loadings = varimax(loadings)
                factor_columns = [f"Factor_{i+1}" for i in range(factor_n)]
                factor_loadings = pd.DataFrame(loadings, index=feature_cols, columns=factor_columns)
                communalities = pd.DataFrame({"feature": feature_cols, "communality_sum_squared_loadings": (factor_loadings**2).sum(axis=1).values}).sort_values("communality_sum_squared_loadings", ascending=False)
                f1, f2, f3 = st.columns(3)
                f1.metric("Factors", factor_n)
                f2.metric("Rotation", "Varimax" if use_varimax and factor_n > 1 else "None")
                f3.metric("Average communality", f"{communalities['communality_sum_squared_loadings'].mean():.2f}")
                st.write("**Factor loadings**")
                display_dataframe(factor_loadings, width="stretch")
                fig, ax = plt.subplots(figsize=(8, max(4, .35 * len(feature_cols))))
                sns.heatmap(factor_loadings, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
                ax.set_title("Factor loadings heatmap")
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
                st.write("**Communalities**")
                display_dataframe(communalities, width="stretch")

        with tabs[4]:
            st.caption("Download an enriched dataset with PCA components and factor scores. Original columns are preserved.")
            pca_out = PCA(n_components=n_threshold).fit_transform(X_scaled)
            pca_cols = [f"PC{i+1}" for i in range(n_threshold)]
            out_df = df.copy()
            aligned_index = X_raw.index if hasattr(X_raw, "index") else df.index
            for i, col in enumerate(pca_cols):
                out_df.loc[aligned_index, col] = pca_out[:, i]
            if len(feature_cols) >= factor_n:
                fa = FactorAnalysis(n_components=factor_n, random_state=RANDOM_STATE)
                fs = fa.fit_transform(X_scaled)
                for i in range(factor_n):
                    out_df.loc[aligned_index, f"Factor_{i+1}"] = fs[:, i]
            display_dataframe(out_df.head(30), width="stretch")
            st.download_button("Download enriched dataset", data=to_csv_bytes(out_df), file_name="feature_reduction_selection_factor_analysis_output.csv", mime="text/csv", width="stretch", icon=":material/download:")
