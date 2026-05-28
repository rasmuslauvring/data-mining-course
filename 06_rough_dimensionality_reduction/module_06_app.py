from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler


APP_DIR = Path(__file__).parent
SAMPLE_DATASETS = {
    "Manufacturing / sensor sample": APP_DIR / "data" / "rough_dimensionality_reduction_sample.csv",
}

st.set_page_config(page_title="Rough Dimensionality Reduction", layout="wide")

# =========================================================
# Global visualization style
# =========================================================
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

st.title("Rough Dimensionality Reduction")
st.caption(
    "Workflow: upload or use a sample dataset, inspect missing values, variance and correlations, "
    "then remove features that are likely to add noise or redundancy before modeling."
)

# =========================================================
# Helper functions
# =========================================================
def _coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    """Convert numeric-like text columns, including common US/EU formats, into numeric dtype."""
    converted = df.copy()

    for col in converted.columns:
        series = converted[col]
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue

        s = series.astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA, "NA": pd.NA})
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


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Read CSV files with delimiter and encoding fallback."""
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
                kwargs["sep"] = sep if sep is not None else None
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


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2),
        "Data Type": [str(df[col].dtype) for col in df.columns],
    }).sort_values("Missing %", ascending=False)


def highlight_missing_values(df: pd.DataFrame):
    """Highlight missing values in a preview table.

    Light red cells indicate values that are missing in the displayed dataset.
    """
    style_fn = lambda value: "background-color: #FFE0E0" if pd.isna(value) else ""
    styler = df.style

    # pandas >= 2.1 uses Styler.map; older versions use Styler.applymap.
    if hasattr(styler, "map"):
        return styler.map(style_fn)

    return styler.applymap(style_fn)


def variance_summary(df: pd.DataFrame, numeric_columns: list[str], scaling_mode: str) -> pd.DataFrame:
    if not numeric_columns:
        return pd.DataFrame(columns=["Column", "Variance", "Missing %", "Scaling"])

    numeric_df = df[numeric_columns].copy()

    # Variance cannot be computed on missing values. Use the observed values only for raw variance.
    if scaling_mode == "Raw values":
        variances = numeric_df.var(skipna=True)
    else:
        # Scaling needs complete input. Median fill is used only to support comparable variance diagnostics.
        filled = numeric_df.fillna(numeric_df.median(numeric_only=True))
        if scaling_mode == "Standardized values":
            scaled = StandardScaler().fit_transform(filled)
        else:
            scaled = MinMaxScaler().fit_transform(filled)
        variances = pd.Series(scaled.var(axis=0), index=numeric_columns)

    return pd.DataFrame({
        "Column": variances.index,
        "Variance": variances.values,
        "Missing %": (df[variances.index].isna().mean().values * 100).round(2),
        "Scaling": scaling_mode,
    }).sort_values("Variance", ascending=True)


def find_high_corr_pairs(df: pd.DataFrame, numeric_columns: list[str], threshold: float) -> pd.DataFrame:
    if len(numeric_columns) < 2:
        return pd.DataFrame(columns=["Feature A", "Feature B", "Correlation", "Abs Correlation"])

    corr = df[numeric_columns].corr().abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).stack().reset_index()
    pairs.columns = ["Feature A", "Feature B", "Abs Correlation"]

    signed_corr = df[numeric_columns].corr()
    pairs["Correlation"] = pairs.apply(lambda row: signed_corr.loc[row["Feature A"], row["Feature B"]], axis=1)
    pairs = pairs.loc[pairs["Abs Correlation"] >= threshold].sort_values("Abs Correlation", ascending=False)
    return pairs[["Feature A", "Feature B", "Correlation", "Abs Correlation"]]


def choose_correlation_drops(
    df: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    target_column: str | None,
    protected_columns: list[str],
) -> list[str]:
    """Choose one feature from each high-correlation pair to remove."""
    to_drop: list[str] = []
    protected = set(protected_columns)

    target_corr = None
    if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        candidate_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_column]
        target_corr = df[candidate_columns + [target_column]].corr()[target_column].abs().drop(labels=[target_column])

    for _, row in high_corr_pairs.iterrows():
        a = row["Feature A"]
        b = row["Feature B"]

        if a in protected or b in protected:
            if a in protected and b not in protected:
                candidate = b
            elif b in protected and a not in protected:
                candidate = a
            else:
                continue
        elif target_corr is not None and a in target_corr.index and b in target_corr.index:
            # Keep the feature that is more related to the target.
            candidate = a if target_corr[a] < target_corr[b] else b
        else:
            # Simple deterministic fallback: remove the second feature in the pair.
            candidate = b

        if candidate not in to_drop:
            to_drop.append(candidate)

    return to_drop


def plot_missing_percentages(summary_df: pd.DataFrame, threshold: float):
    chart_df = summary_df.sort_values("Missing %", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    if chart_df.empty:
        ax.text(0.5, 0.5, "No columns found", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(data=chart_df, x="Column", y="Missing %", ax=ax, color="#4C78A8")
        ax.axhline(threshold, color="#E4572E", linestyle="--", linewidth=1.5, label="Threshold")
        ax.set_title("Missing Values by Feature")
        ax.set_ylabel("Missing %")
        ax.set_xlabel("Feature")
        ax.tick_params(axis="x", rotation=65)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_variance_histogram(var_df: pd.DataFrame, threshold: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    values = var_df["Variance"].dropna()
    if values.empty:
        ax.text(0.5, 0.5, "No numeric variance available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(values, bins=min(30, max(5, len(values))), color="#4C78A8", edgecolor="white", linewidth=1)
        ax.axvline(threshold, color="#E4572E", linestyle="--", linewidth=1.5, label="Threshold")
        ax.set_title("Histogram of Feature Variance")
        ax.set_xlabel("Variance")
        ax.set_ylabel("Number of features")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str]):
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(numeric_columns) < 2:
        ax.text(0.5, 0.5, "Need at least two numeric columns", ha="center", va="center")
        ax.set_axis_off()
    else:
        corr = df[numeric_columns].corr()
        sns.heatmap(
            corr,
            ax=ax,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="coolwarm",
            annot=len(numeric_columns) <= 12,
            fmt=".2f",
            square=False,
            linewidths=0.4,
            cbar=True,
        )
        ax.set_title("Correlation Heatmap")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def plot_feature_count(before_count: int, after_count: int):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Before", "After"], [before_count, after_count], color=["#4C78A8", "#72B7B2"])
    ax.set_title("Feature Count Before vs After")
    ax.set_ylabel("Number of columns")
    for i, value in enumerate([before_count, after_count]):
        ax.text(i, value, str(value), ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    return fig


# =========================================================
# Session state
# =========================================================
if "rdr_uploader_key_version" not in st.session_state:
    st.session_state["rdr_uploader_key_version"] = 0
if "rdr_analysis_requested" not in st.session_state:
    st.session_state["rdr_analysis_requested"] = False

# =========================================================
# App layout
# =========================================================
[tab_feature_removal] = st.tabs(["Batch feature removal"])

with tab_feature_removal:
    st.subheader("Batch rough dimensionality reduction from file")
    st.caption(
        "Methods: missing-value filtering, optional row filtering, low-variance filtering, and high-correlation filtering. "
        "Thresholds are practical starting points and should be adjusted after reviewing the results."
    )

    uploader_key = f"rdr_uploader_{st.session_state['rdr_uploader_key_version']}"

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
        sample_path = SAMPLE_DATASETS[sample_choice]
        if not sample_path.exists():
            sample_path = APP_DIR / sample_path.name

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
        st.session_state["rdr_uploader_key_version"] += 1
        st.session_state["rdr_analysis_requested"] = False
        st.rerun()

    if uploaded_file is None:
        st.info("Upload a CSV/XLSX file. You can also download the sample dataset above and upload it for practice.")
        st.stop()

with tab_feature_removal:
    file_suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if file_suffix in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", excel_file.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            df = _coerce_numeric_like_columns(df)
        else:
            df = read_csv_flexible(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

    if df.empty:
        st.warning("The uploaded file contains no rows.")
        st.stop()

    numeric_columns_all = df.select_dtypes(include=[np.number]).columns.tolist()
    all_columns = df.columns.tolist()

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Setup")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns):,}")
        st.write(f"**Numeric columns:** {len(numeric_columns_all):,}")
        st.write(f"**Missing cells:** {int(df.isna().sum().sum()):,}")

        protected_columns = st.multiselect(
            "Columns to protect from removal",
            all_columns,
            default=[col for col in all_columns if col.lower() in ["id", "wafer_id", "target", "defect", "timestamp"]],
            help="Use this for IDs, timestamps, or target columns that should remain in the output.",
        )

        target_column_options = ["None"] + numeric_columns_all
        default_target_index = target_column_options.index("defect") if "defect" in target_column_options else 0
        target_choice = st.selectbox(
            "Optional numeric target for correlation decisions",
            target_column_options,
            index=default_target_index,
            help="When a highly correlated pair is found, the app keeps the feature with stronger relationship to this target.",
        )
        target_column = None if target_choice == "None" else target_choice

        st.divider()
        st.write("**Feature removal controls**")

        apply_missing_filter = st.checkbox("Remove features with many missing values", value=True)
        missing_threshold = st.slider(
            "Maximum allowed missing % per feature",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=5.0,
            disabled=not apply_missing_filter,
        )
        st.caption("Interpretation: features above this threshold are removed. Use proportions rather than absolute counts.")

        apply_row_filter = st.checkbox("Optionally remove rows with too many missing values", value=False)
        row_missing_threshold = st.slider(
            "Maximum allowed missing % per row",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=5.0,
            disabled=not apply_row_filter,
        )
        st.caption("Practical caution: complete-case analysis can remove too many rows. Use row filtering carefully.")

        apply_variance_filter = st.checkbox("Remove low-variance numeric features", value=True)
        scaling_mode = st.selectbox(
            "Variance calculation mode",
            ["Raw values", "Standardized values", "Min-max scaled values"],
            disabled=not apply_variance_filter,
        )
        variance_threshold = st.number_input(
            "Minimum variance threshold",
            min_value=0.0,
            value=0.01 if scaling_mode != "Raw values" else 0.0001,
            step=0.01 if scaling_mode != "Raw values" else 0.0001,
            format="%.6f",
            disabled=not apply_variance_filter,
        )
        st.caption("Interpretation: features below this threshold are removed. Scaling can make variance comparisons more fair.")

        apply_correlation_filter = st.checkbox("Remove highly correlated numeric features", value=True)
        correlation_threshold = st.slider(
            "High-correlation threshold |r|",
            min_value=0.50,
            max_value=1.00,
            value=0.90,
            step=0.05,
            disabled=not apply_correlation_filter,
        )
        st.caption("Interpretation: correlated features may be redundant, but correlation does not prove causality.")

        st.subheader("Dataset preview")
        st.caption("Light red cells indicate missing values.")
        st.dataframe(highlight_missing_values(df.head(10)), width="stretch")

        run_clicked = st.button("Run reduction", type="primary", width="stretch")

    if run_clicked:
        st.session_state["rdr_analysis_requested"] = True

    if not st.session_state["rdr_analysis_requested"]:
        st.stop()

    # =========================================================
    # Processing
    # =========================================================
    original_df = df.copy()
    working_df = df.copy()
    removal_log = []

    if apply_row_filter:
        row_missing_pct = working_df.isna().mean(axis=1) * 100
        rows_before = len(working_df)
        working_df = working_df.loc[row_missing_pct <= row_missing_threshold].copy()
        rows_removed = rows_before - len(working_df)
    else:
        rows_removed = 0

    missing_summary_df = missing_summary(working_df)
    missing_drop_cols = []
    if apply_missing_filter:
        missing_drop_cols = missing_summary_df.loc[
            (missing_summary_df["Missing %"] > missing_threshold)
            & (~missing_summary_df["Column"].isin(protected_columns)),
            "Column",
        ].tolist()
        for col in missing_drop_cols:
            removal_log.append({"Column": col, "Reason": "High missing values", "Detail": f"> {missing_threshold:.1f}% missing"})
        working_df = working_df.drop(columns=missing_drop_cols)

    numeric_columns_after_missing = working_df.select_dtypes(include=[np.number]).columns.tolist()
    var_df = variance_summary(working_df, numeric_columns_after_missing, scaling_mode)
    variance_drop_cols = []
    if apply_variance_filter and not var_df.empty:
        variance_drop_cols = var_df.loc[
            (var_df["Variance"] < variance_threshold)
            & (~var_df["Column"].isin(protected_columns)),
            "Column",
        ].tolist()
        for col in variance_drop_cols:
            var_value = var_df.loc[var_df["Column"] == col, "Variance"].iloc[0]
            removal_log.append({"Column": col, "Reason": "Low variance", "Detail": f"variance = {var_value:.6f}"})
        working_df = working_df.drop(columns=variance_drop_cols)

    numeric_columns_after_variance = working_df.select_dtypes(include=[np.number]).columns.tolist()
    corr_input_columns = [col for col in numeric_columns_after_variance if col != target_column]
    high_corr_pairs = find_high_corr_pairs(working_df, corr_input_columns, correlation_threshold)
    correlation_drop_cols = []
    if apply_correlation_filter and not high_corr_pairs.empty:
        correlation_drop_cols = choose_correlation_drops(
            working_df,
            high_corr_pairs,
            target_column=target_column,
            protected_columns=protected_columns,
        )
        for col in correlation_drop_cols:
            removal_log.append({"Column": col, "Reason": "High correlation", "Detail": f"|r| >= {correlation_threshold:.2f}"})
        working_df = working_df.drop(columns=correlation_drop_cols, errors="ignore")

    processed_df = working_df.copy()
    removal_log_df = pd.DataFrame(removal_log)
    if removal_log_df.empty:
        removal_log_df = pd.DataFrame(columns=["Column", "Reason", "Detail"])

    total_removed_columns = len(original_df.columns) - len(processed_df.columns)
    total_removed_rows = len(original_df) - len(processed_df)

    with right_col:
        st.subheader("Results")

        r1 = st.columns(4)
        r1[0].metric("Columns Before", f"{len(original_df.columns):,}")
        r1[1].metric("Columns After", f"{len(processed_df.columns):,}")
        r1[2].metric("Columns Removed", f"{total_removed_columns:,}")
        r1[3].metric("Rows Removed", f"{total_removed_rows:,}")

        r2 = st.columns(4)
        r2[0].metric("High Missing", f"{len(missing_drop_cols):,}")
        r2[1].metric("Low Variance", f"{len(variance_drop_cols):,}")
        r2[2].metric("High Correlation", f"{len(correlation_drop_cols):,}")
        r2[3].metric("Remaining Missing", f"{int(processed_df.isna().sum().sum()):,}")

        st.info(
            "Interpretation: this is a rough feature-removal step. Review the removed columns before modeling, "
            "especially when a feature has clear business meaning."
        )

        chart_col_1, chart_col_2 = st.columns(2, gap="large")
        with chart_col_1:
            st.write("**Missing values**")
            fig_missing = plot_missing_percentages(missing_summary_df, missing_threshold if apply_missing_filter else 100.0)
            st.pyplot(fig_missing)
            plt.close(fig_missing)

        with chart_col_2:
            st.write("**Variance distribution**")
            fig_variance = plot_variance_histogram(var_df, variance_threshold if apply_variance_filter else 0.0)
            st.pyplot(fig_variance)
            plt.close(fig_variance)

        chart_col_3, chart_col_4 = st.columns(2, gap="large")
        with chart_col_3:
            st.write("**Correlation heatmap**")
            heatmap_columns = [col for col in numeric_columns_after_variance if col in processed_df.columns or col in correlation_drop_cols]
            heatmap_columns = heatmap_columns[:18]
            fig_corr = plot_correlation_heatmap(working_df.drop(columns=correlation_drop_cols, errors="ignore") if False else original_df, [c for c in original_df.select_dtypes(include=[np.number]).columns if c in heatmap_columns])
            st.pyplot(fig_corr)
            plt.close(fig_corr)

        with chart_col_4:
            st.write("**Feature count**")
            fig_count = plot_feature_count(len(original_df.columns), len(processed_df.columns))
            st.pyplot(fig_count)
            plt.close(fig_count)

        st.write("**Removed feature log**")
        if removal_log_df.empty:
            st.success("No columns were removed with the selected thresholds.")
        else:
            st.dataframe(removal_log_df, width="stretch")

        st.write("**High-correlation pairs detected before correlation removal**")
        if high_corr_pairs.empty:
            st.success("No numeric feature pairs exceeded the selected correlation threshold.")
        else:
            st.dataframe(high_corr_pairs.round(4), width="stretch")

        st.write("**Missing value summary**")
        st.dataframe(missing_summary_df, width="stretch", height=260)

        st.write("**Variance summary**")
        st.dataframe(var_df.round(6), width="stretch", height=260)

        st.write("**Processed dataset preview**")
        st.caption("Light red cells indicate missing values that remain after feature removal.")
        st.dataframe(highlight_missing_values(processed_df.head(20)), width="stretch")

        st.download_button(
            label="Download reduced CSV",
            data=to_csv_bytes(processed_df),
            file_name="rough_dimensionality_reduction_output.csv",
            mime="text/csv",
            width="stretch",
        )

        st.download_button(
            label="Download removed feature log",
            data=to_csv_bytes(removal_log_df),
            file_name="rough_dimensionality_reduction_removed_features.csv",
            mime="text/csv",
            width="stretch",
        )
