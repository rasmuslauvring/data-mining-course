from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from scipy.stats import boxcox_llf, gaussian_kde, probplot
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from statsmodels.stats.diagnostic import lilliefors
import statsmodels.api as sm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Transforming Variables", layout="wide")

APP_DIR = Path(__file__).parent
SAMPLE_DATASETS = {
    "Insurance": Path("data") / "insurance.csv",
    "Iris": Path("data") / "iris.csv",
    "Wine Quality - Red": Path("data") / "winequality-red.csv",
    "Wine Quality - White": Path("data") / "winequality-white.csv",
}

CHART_FIGSIZE = (6, 4)
ALPHA = 0.05


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def clean_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
        .lower()
    )


def _coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    """Convert numeric-like text columns (including locale formats) into numeric dtype."""
    converted = df.copy()

    for col in converted.columns:
        series = converted[col]
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue

        s = series.astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        non_missing = s.notna()
        if int(non_missing.sum()) == 0:
            continue

        numeric_like_ratio = s[non_missing].str.match(r"^[+-]?[0-9\s.,]+$", na=False).mean()
        if numeric_like_ratio < min_ratio:
            continue

        candidates = {
            "plain": pd.to_numeric(s, errors="coerce"),
            "us": pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce"),
            "eu": pd.to_numeric(
                s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            ),
        }

        best_values, best_ratio = None, -1.0
        for values in candidates.values():
            ratio = values[non_missing].notna().mean()
            if ratio > best_ratio:
                best_values, best_ratio = values, ratio

        if best_values is not None and best_ratio >= min_ratio:
            converted[col] = best_values

    return converted


def _read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Read CSV files with automatic delimiter and encoding fallback."""
    separators = [None, ";", ",", "\t", "|"]
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    best_df = None
    best_score = (-1, -1)
    last_exc = None

    for encoding in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                kwargs = {"engine": "python", "encoding": encoding}
                if sep is None:
                    kwargs["sep"] = None
                else:
                    kwargs["sep"] = sep

                candidate = pd.read_csv(uploaded_file, **kwargs)
                score = (candidate.shape[1], candidate.shape[0])
                if score > best_score:
                    best_df = candidate
                    best_score = score
            except Exception as exc:
                last_exc = exc

    if best_df is None:
        raise ValueError(f"Could not parse CSV file. Last error: {last_exc}")

    return _coerce_numeric_like_columns(best_df)


def read_uploaded_file(uploaded_file, sheet_name=None) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return _coerce_numeric_like_columns(pd.read_excel(uploaded_file, sheet_name=sheet_name))
    return _read_csv_flexible(uploaded_file)


def safe_shapiro(x: np.ndarray) -> tuple[float, float, str]:
    """Return Shapiro-Wilk result. For large n, use a reproducible sample."""
    note = ""
    test_x = x
    if len(x) > 5000:
        rng = np.random.default_rng(42)
        test_x = rng.choice(x, size=5000, replace=False)
        note = "Sampled 5,000 rows for Shapiro-Wilk."
    w, p = stats.shapiro(test_x)
    return float(w), float(p), note


def normality_tests(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    result = {
        "n": len(x),
        "shapiro_stat": np.nan,
        "shapiro_p": np.nan,
        "lilliefors_stat": np.nan,
        "lilliefors_p": np.nan,
        "ks_stat": np.nan,
        "ks_p": np.nan,
        "note": "",
    }

    if len(x) < 3 or np.nanstd(x) == 0:
        result["note"] = "Not enough variation for normality tests."
        return result

    try:
        w, p, note = safe_shapiro(x)
        result["shapiro_stat"] = w
        result["shapiro_p"] = p
        result["note"] = note
    except Exception as exc:
        result["note"] = f"Shapiro-Wilk failed: {exc}"

    try:
        stat, p = lilliefors(x)
        result["lilliefors_stat"] = float(stat)
        result["lilliefors_p"] = float(p)
    except Exception as exc:
        result["note"] = (result["note"] + f" Lilliefors failed: {exc}").strip()

    try:
        z = (x - np.mean(x)) / np.std(x, ddof=1)
        stat, p = stats.kstest(z, "norm")
        result["ks_stat"] = float(stat)
        result["ks_p"] = float(p)
    except Exception as exc:
        result["note"] = (result["note"] + f" KS failed: {exc}").strip()

    return result


def variable_profile(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "n": len(x),
        "missing_removed": None,
        "mean": float(np.mean(x)) if len(x) else np.nan,
        "median": float(np.median(x)) if len(x) else np.nan,
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "min": float(np.min(x)) if len(x) else np.nan,
        "max": float(np.max(x)) if len(x) else np.nan,
        "skewness": float(stats.skew(x, bias=False)) if len(x) > 2 else np.nan,
        "kurtosis": float(stats.kurtosis(x, bias=False)) if len(x) > 3 else np.nan,
    }


def boxcox_profile(x: np.ndarray, lmbda_grid=None):
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("Box-Cox requires all values to be strictly positive.")
    if lmbda_grid is None:
        lmbda_grid = np.linspace(-2, 2, 161)
    llf = np.array([boxcox_llf(lmbda, x) for lmbda in lmbda_grid])
    best_idx = int(np.argmax(llf))
    return lmbda_grid, llf, float(lmbda_grid[best_idx])


def apply_transformations(series: pd.Series, selected_methods: list[str], minmax_range: tuple[float, float]) -> tuple[pd.DataFrame, dict]:
    """Create transformed columns for the selected numeric series."""
    x = series.astype(float)
    valid_mask = x.notna()
    x_valid = x.loc[valid_mask].to_numpy().reshape(-1, 1)
    x_vector = x.loc[valid_mask].to_numpy(dtype=float)

    transformed = pd.DataFrame(index=series.index)
    metadata = {}

    if len(x_vector) < 3:
        raise ValueError("Please select a column with at least 3 non-missing numeric values.")

    if "Min-max scaling" in selected_methods:
        scaler = MinMaxScaler(feature_range=minmax_range)
        col = "minmax"
        transformed[col] = np.nan
        transformed.loc[valid_mask, col] = scaler.fit_transform(x_valid).ravel()
        metadata[col] = {"method": "Min-max scaling", "type": "Linear", "lambda": None, "status": "Applied"}

    if "Z-score standardization" in selected_methods:
        scaler = StandardScaler()
        col = "zscore"
        transformed[col] = np.nan
        transformed.loc[valid_mask, col] = scaler.fit_transform(x_valid).ravel()
        metadata[col] = {"method": "Z-score standardization", "type": "Linear", "lambda": None, "status": "Applied"}

    if "Log transformation" in selected_methods:
        col = "log"
        transformed[col] = np.nan
        if np.all(x_vector > 0):
            transformed.loc[valid_mask, col] = np.log10(x_vector)
            metadata[col] = {"method": "Log transformation", "type": "Non-linear", "lambda": None, "status": "Applied with log10(x)"}
        else:
            metadata[col] = {"method": "Log transformation", "type": "Non-linear", "lambda": None, "status": "Skipped: requires strictly positive values"}

    if "Box-Cox" in selected_methods:
        col = "boxcox"
        transformed[col] = np.nan
        if np.all(x_vector > 0):
            bc = PowerTransformer(method="box-cox", standardize=False)
            transformed.loc[valid_mask, col] = bc.fit_transform(x_valid).ravel()
            metadata[col] = {
                "method": "Box-Cox",
                "type": "Non-linear",
                "lambda": float(bc.lambdas_[0]),
                "status": "Applied",
            }
        else:
            metadata[col] = {"method": "Box-Cox", "type": "Non-linear", "lambda": None, "status": "Skipped: requires strictly positive values"}

    if "Yeo-Johnson" in selected_methods:
        col = "yeojohnson"
        transformed[col] = np.nan
        yj = PowerTransformer(method="yeo-johnson", standardize=False)
        transformed.loc[valid_mask, col] = yj.fit_transform(x_valid).ravel()
        metadata[col] = {
            "method": "Yeo-Johnson",
            "type": "Non-linear",
            "lambda": float(yj.lambdas_[0]),
            "status": "Applied",
        }

    return transformed, metadata


def _hist_with_kde(ax, data: np.ndarray, color: str, title: str, xlabel: str) -> None:
    ax.hist(data, bins=25, density=True, color=color, edgecolor="white", linewidth=0.8, alpha=0.75)
    x_grid = np.linspace(data.min(), data.max(), 500)
    kde = gaussian_kde(data)
    ax.plot(x_grid, kde(x_grid), color="#d62728", linewidth=2.5, label="Density")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_histogram(original: np.ndarray, transformed: np.ndarray, original_label: str, transformed_label: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#F8F9FB")
    for ax in axes:
        ax.set_facecolor("#F8F9FB")
    _hist_with_kde(axes[0], original, "#4C78A8", f"Original: {original_label}", original_label)
    _hist_with_kde(axes[1], transformed, "#59A14F", f"Transformed: {transformed_label}", transformed_label)
    fig.tight_layout()
    return fig


def _qqplot_probplot(ax, data: np.ndarray, title: str) -> None:
    (osm, osr), (slope, intercept, _) = probplot(data, dist="norm")
    ax.scatter(osm, osr, s=18, alpha=0.8, color="#4C78A8", zorder=3)
    ax.plot(osm, slope * np.array(osm) + intercept, color="#E4572E", linewidth=2, label="Reference line")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_qqplot(original: np.ndarray, transformed: np.ndarray, original_label: str, transformed_label: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#F8F9FB")
    for ax in axes:
        ax.set_facecolor("#F8F9FB")
    _qqplot_probplot(axes[0], original, f"QQ plot: {original_label}")
    _qqplot_probplot(axes[1], transformed, f"QQ plot: {transformed_label}")
    fig.tight_layout()
    return fig


def make_boxcox_profile_plot(x: np.ndarray):
    grid, llf, best_lambda = boxcox_profile(x)
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    fig.patch.set_facecolor("#F8F9FB")
    ax.set_facecolor("#F8F9FB")
    ax.plot(grid, llf, color="#4C78A8", linewidth=2)
    ax.axvline(best_lambda, color="#E4572E", linestyle="--", linewidth=2)
    ax.set_title("Box-Cox log-likelihood profile", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Log-likelihood")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, best_lambda


def interpret_results(results_df: pd.DataFrame, metadata: dict) -> list[str]:
    messages = []
    comparable = results_df.dropna(subset=["skewness_abs", "shapiro_p"])

    if not comparable.empty:
        original_row = comparable[comparable["method"] == "Original"]
        transformed_rows = comparable[comparable["method"] != "Original"]

        if not transformed_rows.empty:
            best_skew = transformed_rows.sort_values("skewness_abs", ascending=True).iloc[0]
            best_shapiro = transformed_rows.sort_values("shapiro_p", ascending=False).iloc[0]

            messages.append(
                f"Lowest absolute skewness after transformation: **{best_skew['method']}** "
                f"({best_skew['skewness_abs']:.3f})."
            )
            messages.append(
                f"Highest Shapiro-Wilk p-value after transformation: **{best_shapiro['method']}** "
                f"(p = {best_shapiro['shapiro_p']:.4g})."
            )

            if not original_row.empty:
                original_p = float(original_row.iloc[0]["shapiro_p"])
                improved = transformed_rows[transformed_rows["shapiro_p"] > original_p]
                if not improved.empty:
                    messages.append("At least one transformation improved the Shapiro-Wilk p-value compared with the original variable.")
                else:
                    messages.append("None of the selected transformations improved the Shapiro-Wilk p-value compared with the original variable.")

    skipped = [m["method"] for m in metadata.values() if str(m.get("status", "")).startswith("Skipped")]
    if skipped:
        messages.append(f"Skipped methods: **{', '.join(skipped)}** due to input restrictions.")

    messages.append("Remember: better normality is only useful when the model or diagnostic workflow requires it.")
    return messages


def build_normality_conclusion(shapiro_p: float, lilliefors_p: float, ks_p: float, alpha: float = 0.05) -> str:
    pvals = [p for p in [shapiro_p, lilliefors_p, ks_p] if pd.notna(p)]
    if not pvals:
        return "Normality: not assessed"

    pass_count = sum(p >= alpha for p in pvals)
    if pass_count == len(pvals):
        return "Normality: yes"
    if pass_count > 0:
        return "Normality: mixed"
    return "Normality: no"


# -----------------------------------------------------------------------------
# Streamlit state
# -----------------------------------------------------------------------------

if "transform_uploader_key_version" not in st.session_state:
    st.session_state["transform_uploader_key_version"] = 0
if "transformation_requested" not in st.session_state:
    st.session_state["transformation_requested"] = False


# -----------------------------------------------------------------------------
# App layout
# -----------------------------------------------------------------------------

st.title("Transforming Variables")

[tab_transform] = st.tabs(["Transformation analysis"])

with tab_transform:
    st.subheader("Batch transformation analysis from file")
    st.caption(
        "Workflow: upload or use a sample dataset, select one numeric column, choose transformations, "
        "then compare distributions, QQ plots, and normality tests."
    )

    uploader_key = f"transform_uploader_{st.session_state['transform_uploader_key_version']}"

    uploader_col, sample_select_col, sample_download_col, clear_col = st.columns(
        [5.5, 2.2, 1.4, 1], gap="small", vertical_alignment="bottom"
    )

    with uploader_col:
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key=uploader_key,
        )

    with sample_select_col:
        sample_choice = st.selectbox("Sample dataset", list(SAMPLE_DATASETS.keys()))
        sample_rel_path = SAMPLE_DATASETS[sample_choice]
        sample_path = APP_DIR / sample_rel_path
        if not sample_path.exists():
            sample_path = APP_DIR / sample_rel_path.name

    with sample_download_col:
        st.write("")
        st.download_button(
            "Download sample",
            data=sample_path.read_bytes() if sample_path.exists() else b"",
            file_name=sample_path.name,
            mime="text/csv",
            width="stretch",
            disabled=not sample_path.exists(),
            icon=":material/download:",
        )

    with clear_col:
        st.write("")
        clear_file = st.button(
            "Clear",
            width="stretch",
            icon=":material/clear:",
            disabled=uploaded_file is None,
        )

    if clear_file:
        if uploader_key in st.session_state:
            del st.session_state[uploader_key]
        st.session_state["transform_uploader_key_version"] += 1
        st.session_state["transformation_requested"] = False
        st.rerun()

    if uploaded_file is None:
        st.info("Upload a CSV/XLSX file. You can also download one of the sample datasets above and upload it for practice.")
        st.stop()

with tab_transform:
    file_suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if file_suffix in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", excel_file.sheet_names)
            df = read_uploaded_file(uploaded_file, sheet_name=sheet_name)
        else:
            df = read_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

    if df.empty:
        st.warning("The uploaded file contains no rows.")
        st.stop()

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        st.warning("No numeric columns were found in the uploaded dataset.")
        st.stop()

    df_filtered = df  # may be reassigned by the category filter below

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Setup")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns):,}")
        st.write(f"**Numeric columns:** {len(numeric_columns):,}")

        selected_column = st.selectbox("Select numeric column", numeric_columns)

        categorical_cols = [
            col for col in df.columns
            if df[col].dtype == object
            or str(df[col].dtype) == "category"
            or (col not in numeric_columns and df[col].nunique() <= 50)
        ]

        st.markdown("**Category filter (optional)**")
        filter_column = st.selectbox(
            "Filter rows by column",
            ["(No filter)"] + categorical_cols,
        )

        if filter_column != "(No filter)":
            unique_vals = sorted(df[filter_column].dropna().unique().tolist(), key=str)
            filter_value = st.selectbox(f"Select value for '{filter_column}'", unique_vals)
            df_filtered = df[df[filter_column] == filter_value].copy()
            st.write(f"Rows after filter: **{len(df_filtered):,}**")
            if df_filtered.empty:
                st.error("No rows match the selected filter. Choose a different value.")
                st.stop()
        else:
            df_filtered = df

        missing_count = int(df_filtered[selected_column].isna().sum())
        st.write(f"**Missing values in selected column:** {missing_count:,}")

        selected_series = df_filtered[selected_column].dropna().astype(float)
        min_val = float(selected_series.min())
        max_val = float(selected_series.max())
        can_log = bool((selected_series > 0).all())
        can_boxcox = can_log

        st.markdown("**Quick validation**")
        st.write(f"Min: **{min_val:,.4f}**")
        st.write(f"Max: **{max_val:,.4f}**")

        if can_boxcox:
            st.success("Box-Cox and log transformation are possible because all selected values are strictly positive.")
        else:
            st.warning("Log and Box-Cox require strictly positive values. Yeo-Johnson can still be used.")

        st.subheader("Transformation options")
        transformation_mode = st.selectbox(
            "Choose transformation mode",
            [
                "Compare all valid transformations",
                "Min-max scaling",
                "Z-score standardization",
                "Log transformation",
                "Box-Cox",
                "Yeo-Johnson",
            ],
        )

        if transformation_mode == "Compare all valid transformations":
            selected_methods = [
                "Min-max scaling",
                "Z-score standardization",
                "Log transformation",
                "Box-Cox",
                "Yeo-Johnson",
            ]
        else:
            selected_methods = [transformation_mode]

        st.markdown("**Min-max range**")
        range_col_1, range_col_2 = st.columns(2)
        with range_col_1:
            minmax_low = st.number_input("New minimum", value=0.0, step=0.5, format="%.4f")
        with range_col_2:
            minmax_high = st.number_input("New maximum", value=1.0, step=0.5, format="%.4f")

        if minmax_high <= minmax_low:
            st.error("The new maximum must be greater than the new minimum.")
            st.stop()

        st.subheader("Dataset preview")
        st.dataframe(df.head(10), width="stretch")

        run_transformation_clicked = st.button("Run transformation analysis", type="primary", width="stretch")

    if run_transformation_clicked:
        st.session_state["transformation_requested"] = True

    if not st.session_state["transformation_requested"]:
        st.stop()

    try:
        transformed_df, metadata = apply_transformations(
            df_filtered[selected_column],
            selected_methods=selected_methods,
            minmax_range=(minmax_low, minmax_high),
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    original_clean = df_filtered[selected_column].dropna().astype(float).to_numpy()
    base_name = clean_column_name(selected_column)

    analysis_df = df_filtered.copy()
    output_columns = []

    for short_name in transformed_df.columns:
        status = metadata.get(short_name, {}).get("status", "")
        if status.startswith("Applied"):
            output_col = f"{base_name}_{short_name}"
            analysis_df[output_col] = transformed_df[short_name]
            output_columns.append(output_col)

    rows = []
    original_profile = variable_profile(original_clean)
    original_tests = normality_tests(original_clean)
    rows.append({
        "method": "Original",
        "type": "Original",
        "status": "Reference",
        "lambda": np.nan,
        "n": original_profile["n"],
        "mean": original_profile["mean"],
        "std": original_profile["std"],
        "min": original_profile["min"],
        "max": original_profile["max"],
        "skewness": original_profile["skewness"],
        "skewness_abs": abs(original_profile["skewness"]),
        "kurtosis": original_profile["kurtosis"],
        "shapiro_p": original_tests["shapiro_p"],
        "lilliefors_p": original_tests["lilliefors_p"],
        "ks_p": original_tests["ks_p"],
        "note": " ".join(
            part for part in [
                original_tests["note"],
                build_normality_conclusion(
                    original_tests["shapiro_p"],
                    original_tests["lilliefors_p"],
                    original_tests["ks_p"],
                    alpha=ALPHA,
                ),
            ] if part
        ),
    })

    for short_name, meta in metadata.items():
        output_col = f"{base_name}_{short_name}"
        if output_col in analysis_df.columns:
            x_t = analysis_df[output_col].dropna().astype(float).to_numpy()
            profile = variable_profile(x_t)
            tests = normality_tests(x_t)
            rows.append({
                "method": meta["method"],
                "type": meta["type"],
                "status": meta["status"],
                "lambda": meta["lambda"] if meta["lambda"] is not None else np.nan,
                "n": profile["n"],
                "mean": profile["mean"],
                "std": profile["std"],
                "min": profile["min"],
                "max": profile["max"],
                "skewness": profile["skewness"],
                "skewness_abs": abs(profile["skewness"]),
                "kurtosis": profile["kurtosis"],
                "shapiro_p": tests["shapiro_p"],
                "lilliefors_p": tests["lilliefors_p"],
                "ks_p": tests["ks_p"],
                "note": " ".join(
                    part for part in [
                        tests["note"],
                        build_normality_conclusion(
                            tests["shapiro_p"],
                            tests["lilliefors_p"],
                            tests["ks_p"],
                            alpha=ALPHA,
                        ),
                    ] if part
                ),
            })
        else:
            rows.append({
                "method": meta["method"],
                "type": meta["type"],
                "status": meta["status"],
                "lambda": np.nan,
                "n": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "skewness": np.nan,
                "skewness_abs": np.nan,
                "kurtosis": np.nan,
                "shapiro_p": np.nan,
                "lilliefors_p": np.nan,
                "ks_p": np.nan,
                "note": "Method was not applied. Normality: n/a",
            })

    results_df = pd.DataFrame(rows)

    with right_col:
        st.subheader("Results")

        metric_cols = st.columns(4)
        metric_cols[0].metric("Original skewness", f"{original_profile['skewness']:.3f}")
        metric_cols[1].metric("Original mean", f"{original_profile['mean']:,.3f}")
        metric_cols[2].metric("Original std dev", f"{original_profile['std']:,.3f}")
        metric_cols[3].metric("Valid rows", f"{original_profile['n']:,}")

        st.markdown("**Interpretation guide**")
        st.write("- H0 (null hypothesis): the variable is normally distributed.")
        st.write("- Decision rule: p-value >= 0.05 means you do not reject H0; p-value < 0.05 means evidence against H0.")
        for message in interpret_results(results_df, metadata):
            st.write(f"- {message}")

        st.markdown("**Summary table**")
        display_cols = [
            "method", "mean", "std", "min", "max",
            "skewness", "shapiro_p", "lilliefors_p", "ks_p", "note"
        ]
        st.dataframe(
            results_df[display_cols].style.format({
                "lambda": "{:.4f}",
                "mean": "{:,.4f}",
                "std": "{:,.4f}",
                "min": "{:,.4f}",
                "max": "{:,.4f}",
                "skewness": "{:.4f}",
                "shapiro_p": "{:.4g}",
                "lilliefors_p": "{:.4g}",
                "ks_p": "{:.4g}",
            }),
            width="stretch",
        )

        if output_columns:
            chart_column = st.selectbox(
                "Select transformed column for charts",
                ["(No transformation)"] + output_columns,
            )

            if chart_column == "(No transformation)":
                transformed_clean = original_clean
                chart_label = f"{selected_column} (no transformation)"
            else:
                transformed_clean = analysis_df[chart_column].dropna().astype(float).to_numpy()
                chart_label = chart_column

            st.caption(f"Original: {selected_column} | Transformed: {chart_label}")

            st.write("**Histogram comparison**")
            fig_hist = make_histogram(original_clean, transformed_clean, selected_column, chart_label)
            st.pyplot(fig_hist)
            plt.close(fig_hist)

            st.write("**QQ plot comparison**")
            fig_qq = make_qqplot(original_clean, transformed_clean, selected_column, chart_label)
            st.pyplot(fig_qq)
            plt.close(fig_qq)

            if chart_label.endswith("_boxcox"):
                boxcox_lambda = metadata.get("boxcox", {}).get("lambda")
                if boxcox_lambda is not None:
                    st.caption(f"Box-Cox lambda (estimated): {boxcox_lambda:.4f}")

            st.write("**Enriched dataset preview**")
            st.dataframe(analysis_df.head(20), width="stretch")

            st.download_button(
                label="Download enriched CSV",
                data=to_csv_bytes(analysis_df),
                file_name=f"{base_name}_transformation_analysis.csv",
                mime="text/csv",
                width="stretch",
                icon=":material/download:",
            )
        else:
            st.warning("No transformation columns were created. Check the method restrictions above.")

        with st.expander("Concept reminder", expanded=True):
            st.markdown(
                """
                - **Min-max scaling** and **z-score standardization** are linear transformations. They change scale, not shape.
                - **Log**, **Box-Cox**, and **Yeo-Johnson** are non-linear transformations. They can change distribution shape.
                - **Box-Cox** and standard **log** require strictly positive values.
                - **Yeo-Johnson** can handle positive, zero, and negative values.
                - Always validate with plots and tests. Do not transform by habit.
                """
            )
