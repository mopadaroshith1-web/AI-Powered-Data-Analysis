
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="AI Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Data Analysis Dashboard")
st.write("Upload, clean, visualize, and analyze your data.")

# 1. Upload dataset
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "json"]
)

if uploaded_file is None:
    st.info("Upload a CSV, Excel, or JSON file to begin.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    if df.empty or len(df.columns) == 0:
        st.error("The uploaded file contains no data.")
        st.stop()

    if len(df.columns) > 200 or len(df) > 100000:
        st.error("For this starter version, use at most 200 columns and 100,000 rows.")
        st.stop()

    if not df.columns.is_unique:
        st.error("Your dataset has duplicate column names.")
        st.stop()

    df.columns = df.columns.map(str)

except Exception as error:
    st.error(f"Unable to read this file: {error}")
    st.stop()

st.success(f"Successfully loaded: {uploaded_file.name}")

# 2. Preprocessing controls
st.sidebar.header("Data Cleaning")

fill_missing = st.sidebar.checkbox(
    "Fill missing values",
    value=False
)

remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows",
    value=True
)

data = df.copy()
missing_before = int(data.isna().sum().sum())
duplicates_before = int(data.duplicated().sum())

if remove_duplicates:
    data = data.drop_duplicates()

if fill_missing:
    for column in data.columns:
        if data[column].isna().any():
            if pd.api.types.is_numeric_dtype(data[column]):
                median = data[column].median()
                if pd.notna(median):
                    data[column] = data[column].fillna(median)
            else:
                mode = data[column].mode(dropna=True)
                if not mode.empty:
                    data[column] = data[column].fillna(mode.iloc[0])

st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(data):,}")
c2.metric("Columns", f"{len(data.columns):,}")
c3.metric("Missing Values", f"{data.isna().sum().sum():,}")
c4.metric("Duplicates Removed", f"{duplicates_before if remove_duplicates else 0:,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Preview",
    "Statistics",
    "Visualizations",
    "AI Insights",
    "Search & Filter"
])

# 3. Preview and data types
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.head(100), use_container_width=True)

    st.subheader("Column Information")
    column_info = pd.DataFrame({
        "Column": data.columns,
        "Data Type": [str(t) for t in data.dtypes],
        "Missing Values": data.isna().sum().values
    })
    st.dataframe(column_info, use_container_width=True)

    st.write("Missing values before cleaning:", missing_before)
    st.write("Duplicate rows found before cleaning:", duplicates_before)

    st.download_button(
        "Download Cleaned CSV",
        data.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

# 4. Statistical summary
with tab2:
    st.subheader("Statistical Summary")
    st.dataframe(
        data.describe(include="all").T,
        use_container_width=True
    )

    numeric_data = data.select_dtypes(include=np.number)
    if len(numeric_data.columns) >= 2:
        st.subheader("Correlation Heatmap")
        corr = numeric_data.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Between Numeric Columns"
        )
        st.plotly_chart(fig, use_container_width=True)

# 5. Interactive charts
with tab3:
    st.subheader("Explore Your Data")

    columns = list(data.columns)
    numeric_columns = list(
        data.select_dtypes(include=np.number).columns
    )
    categorical_columns = list(
        data.select_dtypes(exclude=np.number).columns
    )

    chart_type = st.selectbox(
        "Choose Chart",
        ["Bar Chart", "Line Chart", "Pie Chart",
         "Histogram", "Scatter Plot"]
    )

    if chart_type == "Histogram":
        if numeric_columns:
            x = st.selectbox("Numeric Column", numeric_columns)
            fig = px.histogram(data, x=x, title=f"Distribution of {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns are available.")

    elif chart_type == "Scatter Plot":
        if len(numeric_columns) >= 2:
            x = st.selectbox("X-axis", numeric_columns)
            y_options = [c for c in numeric_columns if c != x]
            y = st.selectbox("Y-axis", y_options)
            fig = px.scatter(data, x=x, y=y, title=f"{y} vs {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Scatter plots need two numeric columns.")

    elif chart_type in ["Bar Chart", "Pie Chart"]:
        if categorical_columns and numeric_columns:
            x = st.selectbox("Category", categorical_columns)
            y = st.selectbox("Value", numeric_columns)

            grouped = (
                data.groupby(x, dropna=False)[y]
                .sum()
                .reset_index()
                .sort_values(y, ascending=False)
                .head(20)
            )

            if chart_type == "Bar Chart":
                fig = px.bar(grouped, x=x, y=y, title=f"{y} by {x}")
            else:
                fig = px.pie(grouped, names=x, values=y, title=f"{y} by {x}")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("This chart needs a category and a numeric column.")

    elif chart_type == "Line Chart":
        if numeric_columns:
            x = st.selectbox("X-axis", columns)
            y = st.selectbox("Value", numeric_columns)
            fig = px.line(data, x=x, y=y, title=f"{y} by {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns are available.")

# 6. Automated, rule-based insights
with tab4:
    st.subheader("Automated Data Insights")

    st.write(f"The dataset contains {len(data):,} rows and "
             f"{len(data.columns):,} columns.")

    if data.isna().sum().sum() > 0:
        st.warning(
            f"There are {data.isna().sum().sum():,} missing cells. "
            "Consider cleaning them before analysis."
        )
    else:
        st.success("No missing values remain in the processed dataset.")

    if duplicates_before:
        st.write(f"Found {duplicates_before} duplicate rows before cleaning.")

    for column in data.select_dtypes(include=np.number).columns:
        series = data[column].dropna()
        if series.empty:
            continue

        st.write(
            f"**{column}:** Mean = {series.mean():.2f}, "
            f"minimum = {series.min():.2f}, "
            f"maximum = {series.max():.2f}."
        )

        if len(series) >= 5 and series.nunique() > 1:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = ((series < q1 - 1.5 * iqr) |
                        (series > q3 + 1.5 * iqr)).sum()

            if outliers:
                st.write(
                    f"Potential outliers in {column}: {outliers}. "
                    "Review these records before drawing conclusions."
                )

    st.caption(
        "These are automatically generated statistical observations, "
        "not conclusions from a trained generative AI model."
    )

# 7. Search and filtering
with tab5:
    st.subheader("Search and Filter Records")

    search = st.text_input("Search across all columns")
    filtered = data.copy()

    if search:
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(
                search, case=False, na=False, regex=False
            )
        ).any(axis=1)
        filtered = filtered[mask]

    selected_column = st.selectbox(
        "Filter by column",
        ["None"] + list(data.columns)
    )

    if selected_column != "None":
        values = data[selected_column].dropna().unique().tolist()
        if values and len(values) <= 500:
            chosen = st.multiselect(
                "Select values",
                options=values
            )
            if chosen:
                filtered = filtered[
                    filtered[selected_column].isin(chosen)
                ]

    st.write(f"Matching records: {len(filtered):,}")
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        "Download Filtered Data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )

st.sidebar.caption("AI Data Analysis Dashboard | Student Project")
