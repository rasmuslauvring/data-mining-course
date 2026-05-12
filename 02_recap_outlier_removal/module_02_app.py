from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import zscore

APP_DIR = Path(__file__).parent
SAMPLE_DATA_PATH = APP_DIR / "data" / "outlier_sample_dataset.csv"
if not SAMPLE_DATA_PATH.exists():
    SAMPLE_DATA_PATH = APP_DIR / "outlier_sample_dataset.csv"

st.title("Outlier Removal")


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

        best_name, best_values, best_ratio = None, None, -1.0
        for name, values in candidates.items():
            ratio = values[non_missing].notna().mean()
            if ratio > best_ratio:
                best_name, best_values, best_ratio = name, values, ratio

        if best_values is not None and best_ratio >= min_ratio:
            converted[col] = best_values

    return converted


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
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

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

if "outlier_uploader_key_version" not in st.session_state:
    st.session_state["outlier_uploader_key_version"] = 0
if "outlier_detection_requested" not in st.session_state:
    st.session_state["outlier_detection_requested"] = False

[tab_outlier] = st.tabs(["Batch outlier detection"])

with tab_outlier:
    st.subheader("Batch outlier detection from file")
    st.caption(
        "Workflow: batch outlier detection. Inputs: CSV/XLSX dataset and one numeric column. "
        "Methods: IQR (1.5), Z-score (3.0), or domain bounds. "
        "Outputs: flagged rows plus downloadable enriched dataset."
    )

    uploader_key = f"outlier_uploader_{st.session_state['outlier_uploader_key_version']}"

    uploader_col, sample_col, clear_col = st.columns([6, 1.4, 1], gap="small", vertical_alignment="bottom")

    with uploader_col:
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key=uploader_key,
        )

    with sample_col:
        st.write("")
        st.download_button(
            "Sample CSV",
            data=SAMPLE_DATA_PATH.read_bytes() if SAMPLE_DATA_PATH.exists() else b"",
            file_name=SAMPLE_DATA_PATH.name,
            mime="text/csv",
            width="stretch",
            disabled=not SAMPLE_DATA_PATH.exists(),
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
        st.session_state["outlier_uploader_key_version"] += 1
        st.session_state["outlier_detection_requested"] = False
        st.rerun()

    if uploaded_file is None:
        st.stop()

with tab_outlier:
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

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        st.warning("No numeric columns were found in the uploaded dataset.")
        st.stop()

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Setup")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Columns:** {len(df.columns)}")

        selected_column = st.selectbox("Select numeric column", numeric_columns)
        missing_count = int(df[selected_column].isna().sum())
        st.write(f"**Missing values in selected column:** {missing_count}")

        method = st.selectbox(
            "Select detection method",
            ["IQR (1.5)", "Z-score (3.0)", "Domain rule"]
        )

        add_score_column = False
        lower_bound = None
        upper_bound = None

        if method == "Z-score (3.0)":
            st.write("**Method:** Z-score with fixed threshold = 3.0")
            add_score_column = st.checkbox("Add z-score column", value=True)

        elif method == "IQR (1.5)":
            st.write("**Method:** IQR with fixed multiplier = 1.5")

        else:
            st.write("**Method:** Flag values outside user-defined bounds")
            lower_bound = st.number_input("Lower bound", value=0.0, step=1.0, format="%.4f")
            upper_bound = st.number_input("Upper bound", value=100.0, step=1.0, format="%.4f")

        st.subheader("Dataset preview")
        st.dataframe(df.head(10), width="stretch")

        run_detection_clicked = st.button("Run detection", type="primary", width="stretch")

    if run_detection_clicked:
        st.session_state["outlier_detection_requested"] = True

    if not st.session_state["outlier_detection_requested"]:
        st.stop()

    analysis_df = df.copy()
    flag_column_name = f"{selected_column}_outlier_flag"
    score_column_name = None

    series = df[selected_column].dropna()

    if method == "Z-score (3.0)":
        score_column_name = f"{selected_column}_z_score"
        mask = analysis_df[selected_column].notna()

        analysis_df.loc[mask, score_column_name] = zscore(
            analysis_df.loc[mask, selected_column],
            nan_policy="omit"
        )
        analysis_df.loc[~mask, score_column_name] = np.nan

        analysis_df[flag_column_name] = analysis_df[score_column_name].abs() > 3.0
        analysis_df[flag_column_name] = analysis_df[flag_column_name].fillna(False)

        if not add_score_column:
            analysis_df = analysis_df.drop(columns=[score_column_name])
            score_column_name = None

    elif method == "IQR (1.5)":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr

        analysis_df[flag_column_name] = (
            (analysis_df[selected_column] < lower_iqr) |
            (analysis_df[selected_column] > upper_iqr)
        )
        analysis_df[flag_column_name] = analysis_df[flag_column_name].fillna(False)

    else:
        analysis_df[flag_column_name] = (
            (analysis_df[selected_column] < lower_bound) |
            (analysis_df[selected_column] > upper_bound)
        )
        analysis_df[flag_column_name] = analysis_df[flag_column_name].fillna(False)

    flagged_rows_df = analysis_df.loc[analysis_df[flag_column_name]].copy()

    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std())
    min_val = float(series.min())
    max_val = float(series.max())
    outlier_count = int(analysis_df[flag_column_name].sum())
    outlier_pct = (outlier_count / len(analysis_df)) * 100 if len(analysis_df) else 0

    with right_col:
        st.subheader("Results")

        r1 = st.columns(4)
        r1[0].metric("Mean", f"{mean_val:,.3f}")
        r1[1].metric("Median", f"{median_val:,.3f}")
        r1[2].metric("Std Dev", f"{std_val:,.3f}")
        r1[3].metric("Outlier Count", f"{outlier_count:,}")

        r2 = st.columns(4)
        r2[0].metric("Outlier %", f"{outlier_pct:.2f}%")
        r2[1].metric("Min", f"{min_val:,.3f}")
        r2[2].metric("Max", f"{max_val:,.3f}")
        r2[3].metric("Column", selected_column)

        chart_col_1, chart_col_2 = st.columns(2, gap="large")
        CHART_FIGSIZE = (6, 4)

        with chart_col_1:
            st.write("**Histogram**")
            fig_hist, ax_hist = plt.subplots(figsize=CHART_FIGSIZE)
            fig_hist.patch.set_facecolor("#F8F9FB")
            ax_hist.set_facecolor("#F8F9FB")

            ax_hist.hist(
                series,
                bins=20,
                color="#4C78A8",
                edgecolor="white",
                linewidth=1,
                alpha=0.9
            )

            flagged_series = analysis_df.loc[
                analysis_df[flag_column_name], selected_column
            ].dropna()

            if not flagged_series.empty:
                ax_hist.hist(
                    flagged_series,
                    bins=20,
                    color="#E4572E",
                    edgecolor="white",
                    linewidth=1,
                    alpha=0.75
                )

            ax_hist.set_title(f"Histogram of {selected_column}", fontsize=12, fontweight="bold")
            ax_hist.set_xlabel(selected_column)
            ax_hist.set_ylabel("Frequency")
            ax_hist.grid(axis="y", linestyle="--", alpha=0.25)
            ax_hist.spines["top"].set_visible(False)
            ax_hist.spines["right"].set_visible(False)

            fig_hist.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        with chart_col_2:
            st.write("**Boxplot**")
            fig_box, ax_box = plt.subplots(figsize=CHART_FIGSIZE)
            fig_box.patch.set_facecolor("#F8F9FB")
            ax_box.set_facecolor("#F8F9FB")

            ax_box.boxplot(
                series,
                vert=False,
                patch_artist=True,
                boxprops=dict(facecolor="#A0CBE8", edgecolor="#4C78A8", linewidth=1.5),
                medianprops=dict(color="#E4572E", linewidth=2),
                whiskerprops=dict(color="#4C78A8", linewidth=1.2),
                capprops=dict(color="#4C78A8", linewidth=1.2),
                flierprops=dict(
                    marker="o",
                    markerfacecolor="#E4572E",
                    markeredgecolor="white",
                    markersize=7,
                    alpha=0.9
                )
            )

            ax_box.set_title(f"Boxplot of {selected_column}", fontsize=12, fontweight="bold")
            ax_box.set_xlabel(selected_column)
            ax_box.grid(axis="x", linestyle="--", alpha=0.25)
            ax_box.spines["top"].set_visible(False)
            ax_box.spines["right"].set_visible(False)

            fig_box.tight_layout()
            st.pyplot(fig_box)
            plt.close(fig_box)

        if method == "IQR (1.5)":
            st.write(f"**IQR bounds:** lower = {lower_iqr:,.3f}, upper = {upper_iqr:,.3f}")

        if method == "Domain rule":
            st.write(f"**Domain bounds:** lower = {lower_bound:,.3f}, upper = {upper_bound:,.3f}")

        st.write("**Flagged rows preview**")
        if flagged_rows_df.empty:
            st.success("No rows were flagged as possible outliers.")
        else:
            st.dataframe(flagged_rows_df, width="stretch")

        st.write("**Enriched dataset preview**")
        st.dataframe(analysis_df.head(20), width="stretch")

        st.download_button(
            label="Download enriched CSV",
            data=to_csv_bytes(analysis_df),
            file_name=f"{selected_column}_outlier_analysis.csv",
            mime="text/csv",
            width="stretch"
        )