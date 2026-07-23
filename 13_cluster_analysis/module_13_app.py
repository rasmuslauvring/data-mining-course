from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
SAMPLE_DATASETS = {
    "Bank customer segmentation": DATA_DIR / "sample_customer_segmentation.csv",
}

st.set_page_config(page_title="Cluster Analysis", layout="wide")
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

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background:#F7F9FC;border:1px solid #E2E8F0;padding:14px;border-radius:12px;}
.small-note {font-size:.88rem;color:#475569;line-height:1.4;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Cluster Analysis")
st.caption(
    "Workflow: upload a dataset, select numeric features, configure K-Means or Hierarchical Clustering, "
    "then evaluate, visualize, profile, and download the resulting clusters."
)
[main_tab] = st.tabs(["Batch Cluster Analysis"])


# -----------------------------------------------------------------------------
# Data and display helpers
# -----------------------------------------------------------------------------
def coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    """Convert numeric-looking text columns, including common US/EU number formats."""
    converted = df.copy()
    missing_tokens = {
        "": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA,
        "NULL": pd.NA, "NA": pd.NA, "$null$": pd.NA,
    }
    for col in converted.columns:
        series = converted[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        text = series.astype("string").str.strip().replace(missing_tokens)
        non_missing = text.notna()
        if int(non_missing.sum()) == 0:
            continue
        numeric_like_ratio = text[non_missing].str.match(r"^[+-]?[0-9\s.,]+$", na=False).mean()
        if numeric_like_ratio < min_ratio:
            continue
        candidates = [
            pd.to_numeric(text, errors="coerce"),
            pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce"),
            pd.to_numeric(
                text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            ),
        ]
        best = max(candidates, key=lambda values: values[non_missing].notna().mean())
        if best[non_missing].notna().mean() >= min_ratio:
            converted[col] = best
    return converted


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Read CSV files with delimiter and encoding fallbacks."""
    best_df, best_score, last_error = None, (-1, -1), None
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        for separator in [None, ";", ",", "\t", "|"]:
            try:
                uploaded_file.seek(0)
                candidate = pd.read_csv(
                    uploaded_file,
                    sep=separator,
                    engine="python",
                    encoding=encoding,
                )
                score = (candidate.shape[1], candidate.shape[0])
                if score > best_score:
                    best_df, best_score = candidate, score
            except Exception as exc:
                last_error = exc
    if best_df is None:
        raise ValueError(f"Could not parse the CSV file. Last error: {last_error}")
    return coerce_numeric_like_columns(best_df)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        return coerce_numeric_like_columns(pd.read_excel(uploaded_file))
    return read_csv_flexible(uploaded_file)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(how="all").copy()
    unnamed = [col for col in cleaned.columns if str(col).lower().startswith("unnamed")]
    return cleaned.drop(columns=unnamed, errors="ignore")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def highlight_null_values(table: pd.DataFrame):
    if table is None or table.empty:
        return table
    style_fn = lambda value: "background-color:#FFE0E0;color:#7F1D1D;font-weight:600;" if pd.isna(value) else ""
    styler = table.style
    return styler.map(style_fn) if hasattr(styler, "map") else styler.applymap(style_fn)


def display_dataframe(table: pd.DataFrame, **kwargs) -> None:
    st.dataframe(highlight_null_values(table), **kwargs)


def file_signature(uploaded_file) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(raw).hexdigest()


# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------
def prepare_features(
    df: pd.DataFrame,
    features: list[str],
    missing_strategy: str,
    scale_features: bool,
) -> tuple[pd.DataFrame, np.ndarray, SimpleImputer, StandardScaler | None]:
    X_raw = df[features].replace([np.inf, -np.inf], np.nan).copy()
    strategy = "median" if missing_strategy == "Median" else "mean"
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X_raw)
    scaler = StandardScaler() if scale_features else None
    X_model = scaler.fit_transform(X_imputed) if scaler else X_imputed
    return X_raw, X_model, imputer, scaler


def safe_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return {"Silhouette": np.nan, "Calinski-Harabasz": np.nan, "Davies-Bouldin": np.nan}
    return {
        "Silhouette": silhouette_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
    }


def candidate_k_table(X: np.ndarray, max_k: int, n_init: int, random_state: int) -> pd.DataFrame:
    rows = []
    upper = min(max_k, len(X) - 1)
    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X)
        metrics = safe_cluster_metrics(X, labels)
        rows.append({
            "K": k,
            "Inertia / WCSS": model.inertia_,
            **metrics,
        })
    return pd.DataFrame(rows)


def cluster_profile_tables(
    original_df: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
    X_model: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = original_df[features].copy()
    raw["cluster"] = labels
    raw_profile = raw.groupby("cluster")[features].mean().round(3)
    counts = pd.Series(labels).value_counts().sort_index()
    raw_profile.insert(0, "records", counts.reindex(raw_profile.index).values)
    raw_profile.insert(1, "share_percent", (raw_profile["records"] / len(labels) * 100).round(1))

    standardized = pd.DataFrame(X_model, columns=features)
    standardized["cluster"] = labels
    standardized_profile = standardized.groupby("cluster")[features].mean()
    return raw_profile, standardized_profile


def plot_pca_projection(X: np.ndarray, labels: np.ndarray, features: list[str]):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    chart_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": labels.astype(str)})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=chart_df, x="PC1", y="PC2", hue="cluster", palette="Set2", s=55, alpha=.8, ax=ax)
    ax.set_title(f"Cluster projection using PCA ({pca.explained_variance_ratio_.sum():.1%} variance shown)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, pca.explained_variance_ratio_.sum()


def plot_cluster_sizes(labels: np.ndarray):
    sizes = pd.Series(labels).value_counts().sort_index().rename_axis("cluster").reset_index(name="records")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=sizes, x="cluster", y="records", ax=ax)
    ax.set_title("Cluster sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of records")
    for position, value in enumerate(sizes["records"]):
        ax.text(position, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    return fig, sizes


def plot_k_diagnostics(table: pd.DataFrame, selected_k: int):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(table["K"], table["Inertia / WCSS"], marker="o")
    ax.axvline(selected_k, linestyle="--", linewidth=1.4, label=f"Selected K = {selected_k}")
    ax.set_title("Elbow chart: within-cluster variation")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia / WCSS")
    ax.set_xticks(table["K"])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_silhouette_by_k(table: pd.DataFrame, selected_k: int):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(table["K"], table["Silhouette"], marker="o")
    ax.axvline(selected_k, linestyle="--", linewidth=1.4, label=f"Selected K = {selected_k}")
    ax.set_title("Average silhouette score by K")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(table["K"])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_silhouette_distribution(X: np.ndarray, labels: np.ndarray):
    values = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10
    for cluster_id in sorted(np.unique(labels)):
        cluster_values = np.sort(values[labels == cluster_id])
        size = len(cluster_values)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values, alpha=.75)
        ax.text(-0.05, y_lower + size / 2, str(cluster_id))
        y_lower = y_upper + 10
    ax.axvline(values.mean(), linestyle="--", label=f"Average = {values.mean():.3f}")
    ax.set_title("Silhouette values by observation")
    ax.set_xlabel("Silhouette value")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_profile_heatmap(profile: pd.DataFrame):
    fig_width = max(8, min(14, 1.15 * len(profile.columns)))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    sns.heatmap(profile, annot=True, fmt=".2f", center=0, cmap="vlag", linewidths=.5, ax=ax)
    ax.set_title("Cluster profile relative to the overall feature scale")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return fig


def plot_dendrogram(X: np.ndarray, linkage_method: str, max_rows: int, random_state: int):
    rng = np.random.default_rng(random_state)
    if len(X) > max_rows:
        indices = np.sort(rng.choice(len(X), size=max_rows, replace=False))
        X_plot = X[indices]
        note = f"Dendrogram uses a reproducible sample of {max_rows:,} records for readability."
    else:
        X_plot = X
        note = f"Dendrogram uses all {len(X):,} records."
    matrix = linkage(X_plot, method=linkage_method, metric="euclidean")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    dendrogram(matrix, truncate_mode="lastp", p=min(40, len(X_plot)), leaf_rotation=90, ax=ax)
    ax.set_title(f"Hierarchical clustering dendrogram ({linkage_method} linkage)")
    ax.set_xlabel("Merged groups / observations")
    ax.set_ylabel("Linkage distance")
    fig.tight_layout()
    return fig, note


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for key, default in {
    "cluster_uploader_version": 0,
    "cluster_results": None,
    "cluster_file_signature": None,
    "cluster_run_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


with main_tab:
    st.subheader("Batch Cluster Analysis from file")
    st.caption(
        "Download a sample or upload your own CSV/XLSX file. Configure the analysis on the left, "
        "then click Run to create the clusters."
    )

    uploader_key = f"cluster_uploader_{st.session_state['cluster_uploader_version']}"
    upload_col, sample_col, download_col, clear_col = st.columns(
        [5.5, 2.2, 1.4, 1], gap="small", vertical_alignment="bottom"
    )

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key=uploader_key,
        )
    with sample_col:
        sample_choice = st.selectbox("Sample Dataset", list(SAMPLE_DATASETS), key="cluster_sample_choice")
    with download_col:
        sample_path = SAMPLE_DATASETS[sample_choice]
        sample_bytes = sample_path.read_bytes() if sample_path.exists() else b""
        st.download_button(
            "Download Sample",
            data=sample_bytes,
            file_name=sample_path.name,
            mime="text/csv",
            width="stretch",
            disabled=not sample_bytes,
        )
    with clear_col:
        if st.button("Clear", width="stretch", disabled=uploaded_file is None):
            st.session_state["cluster_uploader_version"] += 1
            st.session_state["cluster_results"] = None
            st.session_state["cluster_file_signature"] = None
            st.session_state["cluster_run_error"] = None
            st.rerun()

    if uploaded_file is None:
        st.info("Upload a CSV/XLSX file. You can also download the sample dataset above and upload it for practice.")
        st.stop()

    try:
        current_signature = file_signature(uploaded_file)
        df = clean_frame(read_uploaded_file(uploaded_file))
    except Exception as exc:
        st.error(f"The file could not be loaded: {exc}")
        st.stop()

    if st.session_state["cluster_file_signature"] != current_signature:
        st.session_state["cluster_results"] = None
        st.session_state["cluster_run_error"] = None
        st.session_state["cluster_file_signature"] = current_signature

    if df.empty:
        st.error("The uploaded dataset is empty after removing blank rows.")
        st.stop()

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    useful_numeric = [col for col in numeric_columns if df[col].dropna().nunique() > 1]
    if len(useful_numeric) < 2:
        st.error("At least two non-constant numeric columns are required for cluster analysis.")
        st.stop()

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown("#### Setup")
            method = st.selectbox(
                "Clustering method",
                ["K-Means", "Hierarchical Clustering"],
                help="K-Means creates centroid-based groups. Hierarchical clustering builds nested groups and supports a dendrogram.",
            )

            default_features = [c for c in ["AGE", "YEARSEMPLOYED", "INCOME", "CARDDEBT", "OTHERDEBT"] if c in useful_numeric]
            if len(default_features) < 2:
                default_features = useful_numeric[: min(5, len(useful_numeric))]
            features = st.multiselect(
                "Numeric features",
                useful_numeric,
                default=default_features,
                help="Choose variables that define what similarity means. Identifier columns should usually be excluded.",
            )

            st.markdown("#### Parameters")
            max_allowed_k = max(2, min(12, len(df) - 1))
            n_clusters = st.slider("Number of clusters", 2, max_allowed_k, min(4, max_allowed_k))

            linkage_method = "ward"
            if method == "Hierarchical Clustering":
                linkage_method = st.selectbox(
                    "Linkage method",
                    ["ward", "complete", "average", "single"],
                    help="Ward is a common default for compact numeric clusters. Single linkage can be sensitive to chaining.",
                )

            with st.expander("Advanced settings"):
                scale_features = st.checkbox(
                    "Standardize selected features",
                    value=True,
                    help="Recommended because distance-based methods can be dominated by variables with larger numerical ranges.",
                )
                missing_strategy = st.selectbox("Missing-value imputation", ["Median", "Mean"])
                random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=RANDOM_STATE, step=1)
                n_init = st.slider("K-Means initializations", 5, 100, 30, 5, disabled=method != "K-Means")
                max_candidate_k = st.slider(
                    "Maximum K for diagnostics",
                    3,
                    max_allowed_k,
                    max(n_clusters, min(10, max_allowed_k)),
                    disabled=method != "K-Means",
                )
                dendrogram_rows = st.slider(
                    "Maximum dendrogram records",
                    50,
                    min(500, len(df)),
                    min(250, len(df)),
                    25,
                    disabled=method != "Hierarchical Clustering",
                )

            run_clicked = st.button("Run cluster analysis", type="primary", width="stretch")
            st.markdown(
                '<p class="small-note">Cluster labels are generated only after Run is clicked. Changing controls does not automatically refit the model.</p>',
                unsafe_allow_html=True,
            )

    if run_clicked:
        if len(features) < 2:
            st.session_state["cluster_results"] = None
            st.session_state["cluster_run_error"] = "Select at least two numeric features before running the analysis."
        elif len(df) <= n_clusters:
            st.session_state["cluster_results"] = None
            st.session_state["cluster_run_error"] = "The number of records must be larger than the number of clusters."
        else:
            try:
                X_raw, X_model, imputer, scaler = prepare_features(
                    df, features, missing_strategy, scale_features
                )

                if np.any(np.nanstd(X_model, axis=0) == 0):
                    raise ValueError(
                        "At least one selected feature has no usable variation after preprocessing."
                    )

                candidate_table = None
                dendrogram_data = None
                if method == "K-Means":
                    model = KMeans(
                        n_clusters=n_clusters,
                        init="k-means++",
                        n_init=int(n_init),
                        random_state=int(random_state),
                    )
                    labels = model.fit_predict(X_model)
                    inertia = float(model.inertia_)
                    candidate_table = candidate_k_table(
                        X_model, int(max_candidate_k), int(n_init), int(random_state)
                    )
                else:
                    model = AgglomerativeClustering(
                        n_clusters=n_clusters, linkage=linkage_method
                    )
                    labels = model.fit_predict(X_model)
                    inertia = np.nan
                    dendrogram_data = plot_dendrogram(
                        X_model,
                        linkage_method,
                        int(dendrogram_rows),
                        int(random_state),
                    )

                metrics = safe_cluster_metrics(X_model, labels)
                enriched = df.copy()
                enriched["cluster"] = labels
                raw_profile, standardized_profile = cluster_profile_tables(
                    df, features, labels, X_model
                )
                pca_figure, pca_variance = plot_pca_projection(
                    X_model, labels, features
                )
                size_figure, size_table = plot_cluster_sizes(labels)
                silhouette_figure = plot_silhouette_distribution(X_model, labels)
                heatmap_figure = plot_profile_heatmap(standardized_profile)

                st.session_state["cluster_results"] = {
                    "method": method,
                    "features": features,
                    "n_clusters": n_clusters,
                    "scale_features": scale_features,
                    "missing_strategy": missing_strategy,
                    "linkage_method": linkage_method,
                    "labels": labels,
                    "metrics": metrics,
                    "inertia": inertia,
                    "candidate_table": candidate_table,
                    "dendrogram_data": dendrogram_data,
                    "enriched": enriched,
                    "raw_profile": raw_profile,
                    "standardized_profile": standardized_profile,
                    "pca_figure": pca_figure,
                    "pca_variance": pca_variance,
                    "size_figure": size_figure,
                    "size_table": size_table,
                    "silhouette_figure": silhouette_figure,
                    "heatmap_figure": heatmap_figure,
                }
                st.session_state["cluster_run_error"] = None
            except Exception as exc:
                st.session_state["cluster_results"] = None
                st.session_state["cluster_run_error"] = (
                    f"The cluster analysis could not be completed: {exc}"
                )

    results = st.session_state.get("cluster_results")
    run_error = st.session_state.get("cluster_run_error")

    with right_col:
        if run_error:
            st.error(run_error)

        if results is None:
            st.markdown("#### Dataset preview")
            st.caption(
                f"{len(df):,} rows × {df.shape[1]:,} columns · "
                f"{len(useful_numeric)} usable numeric columns"
            )
            display_dataframe(df.head(20), width="stretch", height=390)
            st.info(
                "Configure the analysis on the left and click Run cluster analysis. "
                "Results will replace this preview."
            )
        else:
            st.markdown("#### Results")
            st.caption(
                f"Saved run: {results['method']} · {results['n_clusters']} clusters · "
                f"{len(results['features'])} features · "
                f"scaling {'on' if results['scale_features'] else 'off'}."
            )

            metric_cols = st.columns(5)
            metric_cols[0].metric("Records", f"{len(results['enriched']):,}")
            metric_cols[1].metric("Clusters", results["n_clusters"])
            metric_cols[2].metric(
                "Silhouette", f"{results['metrics']['Silhouette']:.3f}"
            )
            metric_cols[3].metric(
                "Calinski-Harabasz",
                f"{results['metrics']['Calinski-Harabasz']:.1f}",
            )
            metric_cols[4].metric(
                "Davies-Bouldin", f"{results['metrics']['Davies-Bouldin']:.3f}"
            )

            overview_tab, evaluation_tab, profile_tab, output_tab = st.tabs(
                ["Overview", "Evaluation", "Cluster Profiles", "Processed Output"]
            )

            with overview_tab:
                st.pyplot(results["pca_figure"], width="stretch")
                st.caption(
                    "PCA is used only for the two-dimensional display. The clustering "
                    "uses the complete prepared feature set."
                )

                size_col, table_col = st.columns([1.15, 1], gap="large")
                with size_col:
                    st.pyplot(results["size_figure"], width="stretch")
                with table_col:
                    st.markdown("##### Cluster sizes")
                    display_dataframe(
                        results["size_table"], width="stretch", hide_index=True
                    )
                    st.caption(
                        "Very small clusters may represent meaningful niches, outliers, "
                        "or an over-segmented solution."
                    )

            with evaluation_tab:
                st.markdown("##### Internal validation")
                validation_table = pd.DataFrame({
                    "Metric": [
                        "Silhouette score",
                        "Calinski-Harabasz index",
                        "Davies-Bouldin index",
                    ],
                    "Value": [
                        results["metrics"]["Silhouette"],
                        results["metrics"]["Calinski-Harabasz"],
                        results["metrics"]["Davies-Bouldin"],
                    ],
                    "Interpretation": [
                        "Higher is better; values near 0 indicate overlap.",
                        "Higher suggests compact, separated groups; compare solutions on the same data.",
                        "Lower is better; compare solutions on the same data.",
                    ],
                })
                display_dataframe(
                    validation_table.round(4), width="stretch", hide_index=True
                )

                st.pyplot(results["silhouette_figure"], width="stretch")
                st.caption(
                    "Negative silhouette values indicate observations that may fit "
                    "another cluster better."
                )

                if results["method"] == "K-Means" and results["candidate_table"] is not None:
                    st.markdown("##### Choosing K")
                    st.pyplot(
                        plot_k_diagnostics(
                            results["candidate_table"], results["n_clusters"]
                        ),
                        width="stretch",
                    )
                    st.pyplot(
                        plot_silhouette_by_k(
                            results["candidate_table"], results["n_clusters"]
                        ),
                        width="stretch",
                    )
                    with st.expander("View diagnostic values"):
                        display_dataframe(
                            results["candidate_table"].round(4),
                            width="stretch",
                            hide_index=True,
                        )
                    st.info(
                        "Use the elbow and silhouette as evidence, not as automatic "
                        "answers. The number of clusters should also be interpretable "
                        "and useful for the business problem."
                    )
                elif results["dendrogram_data"] is not None:
                    dendrogram_figure, dendrogram_note = results["dendrogram_data"]
                    st.markdown("##### Dendrogram")
                    st.pyplot(dendrogram_figure, width="stretch")
                    st.caption(
                        dendrogram_note
                        + " Large vertical jumps can suggest plausible cut levels, "
                        "but expert judgment remains necessary."
                    )

            with profile_tab:
                st.markdown("##### Raw-unit cluster profile")
                st.caption(
                    "Means are shown in the original units so the groups can be "
                    "interpreted and named carefully."
                )
                display_dataframe(
                    results["raw_profile"].reset_index(),
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("##### Relative profile heatmap")
                st.pyplot(results["heatmap_figure"], width="stretch")
                st.caption(
                    "Positive values are above the overall prepared-data average; "
                    "negative values are below it. Cluster numbers have no inherent "
                    "business meaning."
                )
                st.warning(
                    "Avoid labels such as ‘loyal’, ‘profitable’, or ‘high risk’ unless "
                    "the selected variables provide evidence for those concepts."
                )

            with output_tab:
                st.markdown("##### Enriched dataset preview")
                st.caption("The original rows are preserved and a cluster column is added.")
                display_dataframe(
                    results["enriched"].head(100),
                    width="stretch",
                    height=420,
                )

                download_col1, download_col2 = st.columns(2)
                with download_col1:
                    st.download_button(
                        "Download clustered dataset",
                        data=to_csv_bytes(results["enriched"]),
                        file_name="clustered_dataset.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                with download_col2:
                    st.download_button(
                        "Download cluster profile",
                        data=to_csv_bytes(results["raw_profile"].reset_index()),
                        file_name="cluster_profile.csv",
                        mime="text/csv",
                        width="stretch",
                    )

                with st.expander("Practical cautions"):
                    st.markdown(
                        "- Scaling strongly affects distance-based clustering.\n"
                        "- Outliers can pull K-Means centroids and can form small hierarchical groups.\n"
                        "- Irrelevant variables change the meaning of similarity.\n"
                        "- A visually attractive PCA chart is not proof of quality in the full feature space.\n"
                        "- Validate the profiles with domain knowledge before using them for decisions."
                    )
