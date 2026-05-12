from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

# IterativeImputer is still experimental in scikit-learn.
# This import is required before importing IterativeImputer.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


APP_DIR = Path(__file__).parent

SAMPLE_DATASETS = {
    "Simple example": APP_DIR / "data" / "simple_missing_values_sample.csv",
    "Bank / customer example": APP_DIR / "data" / "bank_customers_missing_values_sample.csv",
}

st.set_page_config(page_title="Missing Value Imputation", layout="wide")

# =========================================================
# Global visualization style
# =========================================================
# This applies a consistent visual style to all matplotlib charts.
# It avoids repeating the same styling lines inside every chart function.

sns.set_theme(
    style="whitegrid",
    context="notebook",
)

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

st.title("Missing Value Imputation")


# =========================================================
# Helper functions
# =========================================================

def _coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    """
    Convert numeric-like text columns into numeric dtype.

    This is useful when numbers are read as text because of CSV export settings,
    regional formats, or Excel formatting.
    """
    converted = df.copy()

    for col in converted.columns:
        series = converted[col]

        # Skip columns that are already numeric.
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue

        # Standardize common empty or missing labels.
        s = series.astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA, "NA": pd.NA})

        non_missing = s.notna()
        if int(non_missing.sum()) == 0:
            continue

        # Check whether most values look numeric.
        numeric_like_ratio = s[non_missing].str.match(r"^[+-]?[0-9\s.,]+$", na=False).mean()
        if numeric_like_ratio < min_ratio:
            continue

        # Try common numeric formats.
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
    """
    Read CSV files with automatic delimiter and encoding fallback.
    """
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

                # Prefer the parse with more columns and rows.
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
    """Convert a DataFrame into downloadable CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact missing-value summary table."""
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": df.isna().sum().values,
            "Missing %": (df.isna().mean().values * 100).round(2),
            "Data Type": [str(df[col].dtype) for col in df.columns],
        }
    ).sort_values("Missing %", ascending=False)


def highlight_missing_values(df: pd.DataFrame):
    """
    Highlight missing values in red.

    Red = value was missing in the uploaded dataset.
    """
    style_fn = lambda value: "background-color: #FFE0E0" if pd.isna(value) else ""
    styler = df.style

    # pandas >= 2.1 uses Styler.map; older versions use Styler.applymap.
    if hasattr(styler, "map"):
        return styler.map(style_fn)

    return styler.applymap(style_fn)


def highlight_imputed_values(original_df: pd.DataFrame, imputed_df: pd.DataFrame):
    """
    Highlight imputed values in green.

    Green = value was originally missing and has now been filled.
    """
    style_df = pd.DataFrame("", index=imputed_df.index, columns=imputed_df.columns)

    for row_idx in imputed_df.index:
        for col_name in imputed_df.columns:
            if (
                col_name in original_df.columns
                and row_idx in original_df.index
                and pd.isna(original_df.loc[row_idx, col_name])
                and not pd.isna(imputed_df.loc[row_idx, col_name])
            ):
                style_df.loc[row_idx, col_name] = "background-color: #DDF7DD"

    return imputed_df.style.apply(lambda _: style_df, axis=None)


def plot_missing_values(summary_df: pd.DataFrame):
    """Create a bar chart of missing percentages."""
    chart_df = summary_df[summary_df["Missing Count"] > 0].copy()

    fig, ax = plt.subplots(figsize=(6, 4))

    if chart_df.empty:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        sns.barplot(
            data=chart_df,
            x="Column",
            y="Missing %",
            ax=ax,
            color="#4C78A8",
        )
        ax.set_title("Missing Values by Column")
        ax.set_ylabel("Missing %")
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig


def plot_histogram_before_after(original_df: pd.DataFrame, imputed_df: pd.DataFrame, column: str):
    """Compare one numeric column before and after imputation."""
    fig, ax = plt.subplots(
        figsize=(10, 4),
        dpi=200,
    )

    sns.histplot(
        original_df[column].dropna(),
        bins=20,
        ax=ax,
        color="#4C78A8",
        alpha=0.55,
        label="Before",
    )

    sns.histplot(
        imputed_df[column].dropna(),
        bins=20,
        ax=ax,
        color="#E4572E",
        alpha=0.45,
        label="After",
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str], title: str):
    """Create a simple correlation heatmap."""
    fig, ax = plt.subplots(figsize=(6, 4))

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
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=True,
            square=False,
        )

        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    return fig


def variance_comparison(original_df: pd.DataFrame, imputed_df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Compare variance before and after imputation."""
    rows = []

    for col in numeric_columns:
        before_var = original_df[col].var(skipna=True)
        after_var = imputed_df[col].var(skipna=True)
        rows.append(
            {
                "Column": col,
                "Variance Before": before_var,
                "Variance After": after_var,
                "Difference": after_var - before_var,
            }
        )

    return pd.DataFrame(rows)


def impute_mean_mode(df: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    """
    Numeric: mean.
    Categorical: mode.
    """
    result = df.copy()

    if numeric_columns:
        result[numeric_columns] = SimpleImputer(strategy="mean").fit_transform(result[numeric_columns])

    if categorical_columns:
        result[categorical_columns] = SimpleImputer(strategy="most_frequent").fit_transform(
            result[categorical_columns].astype("object")
        )

    return result


def impute_median_mode(df: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    """
    Numeric: median.
    Categorical: mode.
    """
    result = df.copy()

    if numeric_columns:
        result[numeric_columns] = SimpleImputer(strategy="median").fit_transform(result[numeric_columns])

    if categorical_columns:
        result[categorical_columns] = SimpleImputer(strategy="most_frequent").fit_transform(
            result[categorical_columns].astype("object")
        )

    return result


def impute_mode_only(df: pd.DataFrame, selected_columns: list[str]) -> pd.DataFrame:
    """All selected columns: mode."""
    result = df.copy()

    if selected_columns:
        result[selected_columns] = SimpleImputer(strategy="most_frequent").fit_transform(
            result[selected_columns].astype("object")
        )

    return result


def impute_knn_mode(df: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str], k: int) -> pd.DataFrame:
    """
    Numeric: KNN.
    Categorical: mode.

    KNN uses distances, so numeric values are standardized before imputation
    and then converted back to their original scale.
    """
    result = df.copy()

    if numeric_columns:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(result[numeric_columns])
        imputed_scaled = KNNImputer(n_neighbors=k).fit_transform(scaled)
        result[numeric_columns] = scaler.inverse_transform(imputed_scaled)

    if categorical_columns:
        result[categorical_columns] = SimpleImputer(strategy="most_frequent").fit_transform(
            result[categorical_columns].astype("object")
        )

    return result


def impute_mice_style_mode(df: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str], max_iter: int) -> pd.DataFrame:
    """
    Numeric: MICE-style iterative imputation.
    Categorical: mode.

    This is a classroom-friendly approximation using scikit-learn's IterativeImputer.
    """
    result = df.copy()

    if numeric_columns:
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=42,
            initial_strategy="mean",
        )
        result[numeric_columns] = imputer.fit_transform(result[numeric_columns])

    if categorical_columns:
        result[categorical_columns] = SimpleImputer(strategy="most_frequent").fit_transform(
            result[categorical_columns].astype("object")
        )

    return result


# =========================================================
# Session state
# =========================================================

if "imputation_uploader_key_version" not in st.session_state:
    st.session_state["imputation_uploader_key_version"] = 0

if "imputation_requested" not in st.session_state:
    st.session_state["imputation_requested"] = False


# =========================================================
# App layout
# =========================================================

[tab_imputation] = st.tabs(["Batch imputation"])

with tab_imputation:
    st.subheader("Batch missing-value imputation from file")
    st.caption(
        "Workflow: upload or use a sample dataset, select columns, choose an imputation method, "
        "then compare the dataset before and after imputation."
    )

    uploader_key = f"imputation_uploader_{st.session_state['imputation_uploader_key_version']}"

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
        st.session_state["imputation_uploader_key_version"] += 1
        st.session_state["imputation_requested"] = False
        st.rerun()

    if uploaded_file is None:
        st.info("Upload a CSV/XLSX file. You can also download a sample dataset above and upload it for practice.")
        st.stop()


with tab_imputation:
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
    categorical_columns_all = [col for col in df.columns if col not in numeric_columns_all]
    missing_columns = [col for col in df.columns if df[col].isna().any()]

    if not missing_columns:
        st.success("No missing values were found in the uploaded dataset.")
        st.dataframe(df.head(20), width="stretch")
        st.stop()

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Setup")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Columns:** {len(df.columns)}")
        st.write(f"**Missing cells:** {int(df.isna().sum().sum())}")

        selected_columns = st.multiselect(
            "Select columns to impute",
            df.columns.tolist(),
            default=missing_columns,
        )

        selected_numeric_columns = [col for col in selected_columns if col in numeric_columns_all]
        selected_categorical_columns = [col for col in selected_columns if col in categorical_columns_all]

        method = st.selectbox(
            "Select imputation method",
            [
                "Mean + mode",
                "Median + mode",
                "Mode only",
                "KNN + mode",
                "MICE-style iterative + mode",
            ],
        )

        k_value = 5
        max_iter = 10

        if method == "Mean + mode":
            st.write("**Method:** numeric = mean, categorical = mode")

        elif method == "Median + mode":
            st.write("**Method:** numeric = median, categorical = mode")

        elif method == "Mode only":
            st.write("**Method:** selected columns = most frequent value")

        elif method == "KNN + mode":
            st.write("**Method:** numeric = KNN, categorical = mode")
            k_value = st.slider("Number of neighbors (k)", min_value=1, max_value=10, value=5)

        else:
            st.write("**Method:** numeric = MICE-style iterative, categorical = mode")
            max_iter = st.slider("Number of iterations", min_value=3, max_value=30, value=10)

        add_missing_indicators = st.checkbox(
            "Add missing indicator columns",
            value=False,
        )

        st.subheader("Dataset preview")
        st.caption("Red cells are missing values.")
        st.dataframe(highlight_missing_values(df.head(10)), width="stretch")

        run_imputation_clicked = st.button("Run imputation", type="primary", width="stretch")

    if run_imputation_clicked:
        st.session_state["imputation_requested"] = True

    if not st.session_state["imputation_requested"]:
        st.stop()

    if not selected_columns:
        st.warning("Please select at least one column to impute.")
        st.stop()

    if method == "Mean + mode":
        imputed_df = impute_mean_mode(df, selected_numeric_columns, selected_categorical_columns)

    elif method == "Median + mode":
        imputed_df = impute_median_mode(df, selected_numeric_columns, selected_categorical_columns)

    elif method == "Mode only":
        imputed_df = impute_mode_only(df, selected_columns)

    elif method == "KNN + mode":
        imputed_df = impute_knn_mode(df, selected_numeric_columns, selected_categorical_columns, k=k_value)

    else:
        imputed_df = impute_mice_style_mode(
            df,
            selected_numeric_columns,
            selected_categorical_columns,
            max_iter=max_iter,
        )

    if add_missing_indicators:
        for col in selected_columns:
            imputed_df[f"{col}_was_missing"] = df[col].isna()

    missing_summary_df = missing_summary(df)
    imputed_cells_count = int(df[selected_columns].isna().sum().sum())
    remaining_missing_count = int(imputed_df[selected_columns].isna().sum().sum())
    filled_cells_count = imputed_cells_count - remaining_missing_count

    with right_col:
        st.subheader("Results")

        r1 = st.columns(4)
        r1[0].metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
        r1[1].metric("Cells Filled", f"{filled_cells_count:,}")
        r1[2].metric("Remaining Missing", f"{remaining_missing_count:,}")
        r1[3].metric("Columns Treated", f"{len(selected_columns):,}")

        r2 = st.columns(4)
        r2[0].metric("Numeric Columns", f"{len(selected_numeric_columns):,}")
        r2[1].metric("Categorical Columns", f"{len(selected_categorical_columns):,}")
        r2[2].metric("Rows", f"{len(df):,}")
        r2[3].metric("Method", method)

        chart_col_1, chart_col_2 = st.columns(2, gap="large")

        with chart_col_1:
            st.write("**Missing values**")
            fig_missing = plot_missing_values(missing_summary_df)
            st.pyplot(fig_missing)
            plt.close(fig_missing)

        with chart_col_2:
            st.write("**Missing summary**")
            st.dataframe(missing_summary_df, width="stretch", height=330)

        st.write("**Before and after preview**")
        before_col, after_col = st.columns(2, gap="large")

        with before_col:
            st.caption("Before: red cells were missing.")
            st.dataframe(highlight_missing_values(df.head(20)), width="stretch", height=420)

        with after_col:
            st.caption("After: green cells were imputed.")
            st.dataframe(highlight_imputed_values(df.head(20), imputed_df.head(20)), width="stretch", height=420)

        if selected_numeric_columns:
            st.write("**Distribution comparison**")
            selected_visual_column = st.selectbox("Select numeric column for histogram", selected_numeric_columns)

            fig_hist = plot_histogram_before_after(df, imputed_df, selected_visual_column)
            st.pyplot(fig_hist, width="stretch")
            plt.close(fig_hist)

            st.write("**Variance comparison**")
            variance_df = variance_comparison(df, imputed_df, selected_numeric_columns)
            st.dataframe(variance_df, width="stretch")

        if len(selected_numeric_columns) >= 2:
            st.write("**Correlation heatmaps**")
            heatmap_col_1, heatmap_col_2 = st.columns(2, gap="large")

            with heatmap_col_1:
                fig_corr_before = plot_correlation_heatmap(df, selected_numeric_columns, "Before imputation")
                st.pyplot(fig_corr_before)
                plt.close(fig_corr_before)

            with heatmap_col_2:
                fig_corr_after = plot_correlation_heatmap(imputed_df, selected_numeric_columns, "After imputation")
                st.pyplot(fig_corr_after)
                plt.close(fig_corr_after)

        st.write("**Imputed dataset preview**")
        st.dataframe(imputed_df.head(20), width="stretch")

        st.download_button(
            label="Download imputed CSV",
            data=to_csv_bytes(imputed_df),
            file_name="missing_value_imputation_analysis.csv",
            mime="text/csv",
            width="stretch",
        )
