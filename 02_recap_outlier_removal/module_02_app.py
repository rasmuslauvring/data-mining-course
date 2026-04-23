from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import zscore

st.title("Outlier Management Explorer")
st.write(
    "Upload a dataset, select one numeric column, detect possible outliers using "
    "**Z-score**, **IQR**, or a **domain rule**, and download the enriched file."
)

st.info(
    "Flagged values are not automatically wrong. Review business context before "
    "removing or changing them."
)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.stop()

file_suffix = Path(uploaded_file.name).suffix.lower()

if file_suffix in [".xlsx", ".xls"]:
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select sheet", excel_file.sheet_names)
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
else:
    df = pd.read_csv(uploaded_file)

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

    run_detection = st.button("Run detection", type="primary", width="stretch")

if not run_detection:
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