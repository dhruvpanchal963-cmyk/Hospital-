# ============================================================
# 📊 DataVerse AI
# Intelligent Data Analyst Assistant
# Author : Dhruv Panchal
# Version : 1.0
# ============================================================

# -------------------------
# Core Libraries
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import joblib

# -------------------------
# Visualization Libraries
# -------------------------
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# -------------------------
# Machine Learning
# -------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -------------------------
# Report Generation
# -------------------------
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -------------------------
# Ignore Warnings
# -------------------------
import warnings
warnings.filterwarnings("ignore")
# ============================================================
# SECTION 2 : PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="📊 DataVerse AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/",
        "Report a Bug": "https://github.com/",
        "About": """
        # 📊 DataVerse AI

        ### Intelligent Data Analyst Assistant

        Upload any CSV or Excel dataset and perform:

        ✅ Data Cleaning
        ✅ Exploratory Data Analysis (EDA)
        ✅ Interactive Visualizations
        ✅ Machine Learning
        ✅ Predictions
        ✅ PDF Report Generation

        Developed using ❤️ with Streamlit.
        """
    }
)

# ------------------------------------------------------------
# Create Required Folders (Automatically)
# ------------------------------------------------------------

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ------------------------------------------------------------
# App Constants
# ------------------------------------------------------------

APP_NAME = "📊 DataVerse AI"
APP_VERSION = "Version 1.0"
AUTHOR = "Dhruv Panchal"

PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#0F172A"
SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#F59E0B"
DANGER_COLOR = "#DC2626"

# ------------------------------------------------------------
# Sidebar Logo & Header
# ------------------------------------------------------------

st.sidebar.image(
    "https://raw.githubusercontent.com/streamlit/brand/main/logos/mark/streamlit-mark-color.png",
    width=90
)

st.sidebar.title("📊 DataVerse AI")
st.sidebar.caption("Intelligent Data Analyst Assistant")

st.sidebar.markdown("---")
# ============================================================
# SECTION 3 : CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* -----------------------------------------------------------
Main App Background
----------------------------------------------------------- */

.stApp{
    background-color:#F5F7FA;
}

/* -----------------------------------------------------------
Hide Streamlit Branding
----------------------------------------------------------- */

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* -----------------------------------------------------------
Main Title
----------------------------------------------------------- */

.main-title{
    font-size:42px;
    font-weight:800;
    color:#2563EB;
    text-align:center;
    margin-bottom:5px;
}

.sub-title{
    text-align:center;
    color:#64748B;
    font-size:18px;
    margin-bottom:30px;
}

/* -----------------------------------------------------------
Dashboard Cards
----------------------------------------------------------- */

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 3px 12px rgba(0,0,0,0.08);
    margin-bottom:20px;
    transition:0.3s;
}

.card:hover{
    transform:translateY(-3px);
    box-shadow:0px 6px 18px rgba(0,0,0,0.15);
}

/* -----------------------------------------------------------
Metric Cards
----------------------------------------------------------- */

.metric-card{
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    color:white;
    padding:18px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 5px 15px rgba(0,0,0,0.15);
}

/* -----------------------------------------------------------
Buttons
----------------------------------------------------------- */

.stButton>button{

    width:100%;
    border-radius:10px;
    border:none;
    padding:12px;
    font-weight:bold;
    color:white;
    background:#2563EB;
    transition:0.3s;
}

.stButton>button:hover{

    background:#1D4ED8;
    transform:scale(1.02);

}

/* -----------------------------------------------------------
Sidebar
----------------------------------------------------------- */

section[data-testid="stSidebar"]{

    background:#0F172A;
}

section[data-testid="stSidebar"] *{

    color:white;

}

/* -----------------------------------------------------------
File Uploader
----------------------------------------------------------- */

[data-testid="stFileUploader"]{

    border:2px dashed #2563EB;
    border-radius:15px;
    padding:15px;

}

/* -----------------------------------------------------------
DataFrame
----------------------------------------------------------- */

[data-testid="stDataFrame"]{

    border-radius:10px;

}

/* -----------------------------------------------------------
Success Box
----------------------------------------------------------- */

.success-box{

    background:#DCFCE7;
    color:#166534;
    padding:15px;
    border-radius:10px;
    font-weight:600;

}

/* -----------------------------------------------------------
Warning Box
----------------------------------------------------------- */

.warning-box{

    background:#FEF3C7;
    color:#92400E;
    padding:15px;
    border-radius:10px;
    font-weight:600;

}

/* -----------------------------------------------------------
Footer
----------------------------------------------------------- */

.footer{

    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:14px;

}

</style>
""", unsafe_allow_html=True)
# ============================================================
# SECTION 4 : SESSION STATE MANAGEMENT
# ============================================================

# Dataset Storage
if "df" not in st.session_state:
    st.session_state.df = None

if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

# Uploaded File Information
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

# Machine Learning
if "model" not in st.session_state:
    st.session_state.model = None

if "target_column" not in st.session_state:
    st.session_state.target_column = None

if "problem_type" not in st.session_state:
    st.session_state.problem_type = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

# Reports
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# Statistics
if "statistics" not in st.session_state:
    st.session_state.statistics = {}

# Visualization Cache
if "plots" not in st.session_state:
    st.session_state.plots = {}

# Dataset Information
if "dataset_info" not in st.session_state:
    st.session_state.dataset_info = {}

# Navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

# Theme
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# User Status
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# ------------------------------------------------------------
# Helper Function : Reset Application
# ------------------------------------------------------------

def reset_application():
    """Reset all stored data."""

    st.session_state.df = None
    st.session_state.clean_df = None
    st.session_state.uploaded_file = None
    st.session_state.file_name = ""
    st.session_state.model = None
    st.session_state.target_column = None
    st.session_state.problem_type = None
    st.session_state.predictions = None
    st.session_state.report_generated = False
    st.session_state.statistics = {}
    st.session_state.plots = {}
    st.session_state.dataset_info = {}
    st.session_state.data_loaded = False

# ------------------------------------------------------------
# Helper Function : Dataset Loaded Check
# ------------------------------------------------------------

def dataset_available():
    """Returns True if dataset exists."""

    return (
        st.session_state.df is not None
        or st.session_state.clean_df is not None
    )
  # ============================================================
# SECTION 5 : SIDEBAR NAVIGATION
# ============================================================

# -----------------------------
# Navigation Menu
# -----------------------------
st.sidebar.header("🧭 Navigation")

pages = [
    "🏠 Home",
    "📂 Upload Dataset",
    "📋 Dataset Overview",
    "🧹 Data Cleaning",
    "📊 Data Visualization",
    "📈 Statistics",
    "🤖 Machine Learning",
    "🎯 Prediction",
    "📄 Report Generator",
    "⬇ Download Center",
    "ℹ About"
]

selected_page = st.sidebar.radio(
    "Select a Module",
    pages,
    index=pages.index(st.session_state.current_page)
)

# Store Current Page
st.session_state.current_page = selected_page

st.sidebar.markdown("---")

# -----------------------------
# Dataset Status
# -----------------------------
st.sidebar.subheader("📌 Dataset Status")

if st.session_state.df is not None:
    st.sidebar.success("✅ Dataset Loaded")

    st.sidebar.write(f"**File:** {st.session_state.file_name}")
    st.sidebar.write(f"**Rows:** {st.session_state.df.shape[0]}")
    st.sidebar.write(f"**Columns:** {st.session_state.df.shape[1]}")

else:
    st.sidebar.warning("⚠ No Dataset Uploaded")

st.sidebar.markdown("---")

# -----------------------------
# Quick Actions
# -----------------------------
st.sidebar.subheader("⚡ Quick Actions")

if st.sidebar.button("🔄 Reset Application"):

    reset_application()

    st.success("Application Reset Successfully!")

    st.rerun()

# -----------------------------
# Theme Selector
# -----------------------------
theme = st.sidebar.selectbox(
    "🎨 Theme",
    ["Light", "Dark"],
    index=0
)

st.session_state.theme = theme

st.sidebar.markdown("---")

# -----------------------------
# Project Information
# -----------------------------
st.sidebar.subheader("📊 Project")

st.sidebar.info(
    """
**DataVerse AI**

Intelligent Data Analyst Assistant

Version : 1.0

Developed using
Python • Streamlit • Machine Learning
"""
)

st.sidebar.markdown("---")

# -----------------------------
# Footer
# -----------------------------
st.sidebar.caption("© 2026 Dhruv Panchal")
# ============================================================
# SECTION 6 : HOME DASHBOARD
# ============================================================

if selected_page == "🏠 Home":

    # --------------------------------------------------------
    # Hero Section
    # --------------------------------------------------------

    st.markdown("""
    <h1 class="main-title">📊 DataVerse AI</h1>
    <p class="sub-title">
    Intelligent Data Analyst Assistant for Data Cleaning,
    Visualization, Machine Learning & Report Generation
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------------------------
    # Welcome Message
    # --------------------------------------------------------

    st.info("""
    👋 **Welcome to DataVerse AI!**

    Upload any **CSV** or **Excel** dataset and perform complete
    Data Analysis with interactive visualizations, machine learning,
    statistical insights, and downloadable reports.
    """)

    # --------------------------------------------------------
    # KPI Cards
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📂 Supported Formats", "CSV & Excel")

    with col2:
        st.metric("📊 Charts Available", "10+")

    with col3:
        st.metric("🤖 ML Algorithms", "6+")

    with col4:
        st.metric("📄 Export Reports", "PDF / CSV")

    st.markdown("---")

    # --------------------------------------------------------
    # Feature Cards
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">
        <h3>📈 Data Analytics</h3>

        ✔ Data Cleaning<br>
        ✔ Missing Value Detection<br>
        ✔ Duplicate Removal<br>
        ✔ Statistical Summary<br>
        ✔ Correlation Analysis

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">
        <h3>🤖 Machine Learning</h3>

        ✔ Classification<br>
        ✔ Regression<br>
        ✔ Prediction<br>
        ✔ Model Evaluation<br>
        ✔ Feature Importance

        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # Workflow
    # --------------------------------------------------------

    st.subheader("🚀 Data Analysis Workflow")

    st.success("""
    📂 Upload Dataset

          ↓

    🧹 Clean Data

          ↓

    📊 Explore Data

          ↓

    📈 Visualize

          ↓

    🤖 Train ML Model

          ↓

    🔮 Predict

          ↓

    📄 Generate Report
    """)

    # --------------------------------------------------------
    # Why Choose DataVerse AI
    # --------------------------------------------------------

    st.subheader("⭐ Why Choose DataVerse AI?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### ⚡ Fast

        Analyze datasets in seconds using optimized Python libraries.
        """)

    with col2:
        st.markdown("""
        ### 📊 Interactive

        Create professional charts and dashboards with Plotly.
        """)

    with col3:
        st.markdown("""
        ### 🤖 AI Powered

        Train Machine Learning models with just a few clicks.
        """)

    st.markdown("---")

    # --------------------------------------------------------
    # Getting Started
    # --------------------------------------------------------

    st.subheader("📌 Getting Started")

    st.markdown("""
    **Step 1:** Go to **📂 Upload Dataset**

    **Step 2:** Upload a CSV or Excel file

    **Step 3:** Explore the dataset

    **Step 4:** Clean missing values

    **Step 5:** Create visualizations

    **Step 6:** Train a Machine Learning model

    **Step 7:** Generate predictions and reports
    """)

    st.markdown("---")

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    st.caption(
        "© 2026 DataVerse AI | Developed by Dhruv Panchal | Python • Streamlit • Machine Learning"
    )
  # ============================================================
# SECTION 7 : UPLOAD DATASET
# ============================================================

if selected_page == "📂 Upload Dataset":

    st.title("📂 Upload Dataset")
    st.write("Upload your **CSV** or **Excel (.xlsx)** dataset to begin analysis.")

    st.markdown("---")

    # --------------------------------------------------------
    # File Upload
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose a Dataset",
        type=["csv", "xlsx"],
        help="Supported formats: CSV and Excel"
    )

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)

            # Save in Session State
            st.session_state.df = df
            st.session_state.clean_df = df.copy()
            st.session_state.file_name = uploaded_file.name
            st.session_state.uploaded_file = uploaded_file
            st.session_state.data_loaded = True

            st.success("✅ Dataset uploaded successfully!")

            st.markdown("---")

            # ------------------------------------------------
            # Dataset Information
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📄 Rows", df.shape[0])

            with col2:
                st.metric("📊 Columns", df.shape[1])

            with col3:
                memory = df.memory_usage(deep=True).sum() / 1024
                st.metric("💾 Size (KB)", f"{memory:.2f}")

            st.markdown("---")

            # ------------------------------------------------
            # Column Information
            # ------------------------------------------------

            st.subheader("📋 Dataset Columns")

            column_info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum().values
            })

            st.dataframe(
                column_info,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")

            # ------------------------------------------------
            # Dataset Preview
            # ------------------------------------------------

            st.subheader("👀 Dataset Preview")

            rows = st.slider(
                "Select Number of Rows",
                5,
                min(100, len(df)),
                10
            )

            st.dataframe(
                df.head(rows),
                use_container_width=True
            )

            st.markdown("---")

            # ------------------------------------------------
            # Dataset Summary
            # ------------------------------------------------

            st.subheader("📈 Dataset Summary")

            st.dataframe(
                df.describe(include="all").fillna("-"),
                use_container_width=True
            )

        except Exception as e:

            st.error(f"❌ Error while reading file:\n\n{e}")

    else:

        st.info("⬆ Please upload a CSV or Excel file to continue.")
      # ============================================================
# SECTION 8 : DATASET OVERVIEW
# ============================================================

if selected_page == "📋 Dataset Overview":

    st.title("📋 Dataset Overview")

    if not dataset_available():
        st.warning("⚠ Please upload a dataset first.")
        st.stop()

    # Use cleaned dataset if available
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
    else:
        df = st.session_state.df

    # --------------------------------------------------------
    # Basic Statistics
    # --------------------------------------------------------

    total_rows = df.shape[0]
    total_columns = df.shape[1]
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    numeric_columns = len(df.select_dtypes(include=np.number).columns)
    categorical_columns = len(df.select_dtypes(exclude=np.number).columns)

    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # --------------------------------------------------------
    # KPI Cards
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("📄 Rows", total_rows)

    with c2:
        st.metric("📊 Columns", total_columns)

    with c3:
        st.metric("❗ Missing Values", missing_values)

    with c4:
        st.metric("🔁 Duplicate Rows", duplicate_rows)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🔢 Numeric Columns", numeric_columns)

    with c2:
        st.metric("🔤 Categorical Columns", categorical_columns)

    with c3:
        st.metric("💾 Memory (MB)", f"{memory_usage:.2f}")

    st.markdown("---")

    # --------------------------------------------------------
    # Dataset Information
    # --------------------------------------------------------

    st.subheader("📑 Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Missing Values Chart
    # --------------------------------------------------------

    st.subheader("📉 Missing Values Analysis")

    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing"]

    missing_df = missing_df[missing_df["Missing"] > 0]

    if len(missing_df) > 0:

        fig = px.bar(
            missing_df,
            x="Column",
            y="Missing",
            color="Missing",
            title="Missing Values by Column"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.success("✅ No Missing Values Found!")

    st.markdown("---")

    # --------------------------------------------------------
    # Data Type Distribution
    # --------------------------------------------------------

    st.subheader("📊 Data Type Distribution")

    dtype_counts = df.dtypes.astype(str).value_counts()

    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index,
        title="Column Data Types"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------
    # Dataset Quality Score
    # --------------------------------------------------------

    st.subheader("⭐ Dataset Quality Score")

    score = 100

    score -= min(missing_values, 50)
    score -= min(duplicate_rows, 20)

    score = max(score, 0)

    st.progress(score / 100)

    st.metric("Quality Score", f"{score}%")

    if score >= 90:
        st.success("Excellent Dataset Quality")

    elif score >= 70:
        st.info("Good Dataset Quality")

    elif score >= 50:
        st.warning("Average Dataset Quality")

    else:
        st.error("Poor Dataset Quality")

    st.markdown("---")

    # --------------------------------------------------------
    # First & Last Records
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🔝 First 5 Records")
        st.dataframe(df.head(), use_container_width=True)

    with col2:

        st.subheader("🔚 Last 5 Records")
        st.dataframe(df.tail(), use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------
    # Column Selector
    # --------------------------------------------------------

    st.subheader("🔍 Explore a Column")

    selected_column = st.selectbox(
        "Select a Column",
        df.columns
    )

    st.write("**Data Type:**", df[selected_column].dtype)
    st.write("**Unique Values:**", df[selected_column].nunique())
    st.write("**Missing Values:**", df[selected_column].isnull().sum())

    st.dataframe(
        df[[selected_column]].head(10),
        use_container_width=True
    )
  # ============================================================
# SECTION 9 : DATA CLEANING
# ============================================================

if selected_page == "🧹 Data Cleaning":

    st.title("🧹 Data Cleaning")

    if not dataset_available():
        st.warning("⚠ Please upload a dataset first.")
        st.stop()

    # Use cleaned dataset if available
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df.copy()
    else:
        df = st.session_state.df.copy()

    st.info("Apply data cleaning operations and save the cleaned dataset.")

    st.markdown("---")

    # ========================================================
    # Dataset Summary
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("---")

    # ========================================================
    # Remove Duplicate Rows
    # ========================================================

    st.subheader("🔁 Remove Duplicate Rows")

    duplicates = df.duplicated().sum()

    st.write(f"Duplicate Rows Found: **{duplicates}**")

    if st.button("Remove Duplicates"):

        before = len(df)

        df = df.drop_duplicates()

        removed = before - len(df)

        st.session_state.clean_df = df

        st.success(f"✅ {removed} duplicate rows removed.")

    st.markdown("---")

    # ========================================================
    # Handle Missing Values
    # ========================================================

    st.subheader("❗ Handle Missing Values")

    strategy = st.selectbox(
        "Choose Method",
        [
            "Mean",
            "Median",
            "Mode",
            "Drop Rows"
        ]
    )

    if st.button("Apply Missing Value Treatment"):

        temp = df.copy()

        if strategy == "Mean":

            numeric = temp.select_dtypes(include=np.number).columns

            temp[numeric] = temp[numeric].fillna(
                temp[numeric].mean()
            )

        elif strategy == "Median":

            numeric = temp.select_dtypes(include=np.number).columns

            temp[numeric] = temp[numeric].fillna(
                temp[numeric].median()
            )

        elif strategy == "Mode":

            for col in temp.columns:
                temp[col].fillna(
                    temp[col].mode()[0],
                    inplace=True
                )

        elif strategy == "Drop Rows":

            temp.dropna(inplace=True)

        st.session_state.clean_df = temp

        st.success("✅ Missing values handled successfully.")

    st.markdown("---")

    # ========================================================
    # Rename Columns
    # ========================================================

    st.subheader("✏ Rename Column")

    old_column = st.selectbox(
        "Select Column",
        df.columns
    )

    new_column = st.text_input(
        "Enter New Column Name"
    )

    if st.button("Rename Column"):

        if new_column.strip() != "":

            df.rename(
                columns={old_column: new_column},
                inplace=True
            )

            st.session_state.clean_df = df

            st.success("✅ Column renamed successfully.")

    st.markdown("---")

    # ========================================================
    # Change Data Type
    # ========================================================

    st.subheader("🔄 Change Data Type")

    dtype_column = st.selectbox(
        "Column",
        df.columns,
        key="dtype"
    )

    dtype_option = st.selectbox(
        "Convert To",
        [
            "int",
            "float",
            "str"
        ]
    )

    if st.button("Convert Data Type"):

        try:

            df[dtype_column] = df[dtype_column].astype(dtype_option)

            st.session_state.clean_df = df

            st.success("✅ Data type converted successfully.")

        except Exception as e:

            st.error(e)

    st.markdown("---")

    # ========================================================
    # Dataset Preview
    # ========================================================

    st.subheader("📋 Cleaned Dataset Preview")

    st.dataframe(
        st.session_state.clean_df.head(20),
        use_container_width=True
    )

    st.markdown("---")

    # ========================================================
    # Download Cleaned Dataset
    # ========================================================

    st.subheader("⬇ Download Cleaned Dataset")

    csv = st.session_state.clean_df.to_csv(index=False)

    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    st.success("🎉 Data Cleaning Completed Successfully!")
  # ============================================================
# SECTION 10 : STATISTICS DASHBOARD
# ============================================================

if selected_page == "📈 Statistics":

    st.title("📈 Statistical Analysis Dashboard")

    if not dataset_available():
        st.warning("⚠ Please upload a dataset first.")
        st.stop()

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df.copy()
    else:
        df = st.session_state.df.copy()

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        st.error("❌ No numeric columns available.")
        st.stop()

    # --------------------------------------------------------
    # Select Numeric Column
    # --------------------------------------------------------

    st.subheader("📌 Select Numeric Column")

    column = st.selectbox(
        "Choose a Column",
        numeric_df.columns
    )

    series = numeric_df[column]

    st.markdown("---")

    # --------------------------------------------------------
    # Basic Statistics
    # --------------------------------------------------------

    mean = series.mean()
    median = series.median()
    mode = series.mode().iloc[0]
    std = series.std()
    variance = series.var()
    minimum = series.min()
    maximum = series.max()

    q1 = series.quantile(0.25)
    q2 = series.quantile(0.50)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    skewness = series.skew()
    kurtosis = series.kurt()

    # --------------------------------------------------------
    # KPI Cards
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Mean", f"{mean:.2f}")

    with c2:
        st.metric("Median", f"{median:.2f}")

    with c3:
        st.metric("Mode", f"{mode:.2f}")

    with c4:
        st.metric("Std Dev", f"{std:.2f}")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Variance", f"{variance:.2f}")

    with c2:
        st.metric("Minimum", f"{minimum:.2f}")

    with c3:
        st.metric("Maximum", f"{maximum:.2f}")

    with c4:
        st.metric("Range", f"{maximum-minimum:.2f}")

    st.markdown("---")

    # --------------------------------------------------------
    # Quartiles
    # --------------------------------------------------------

    st.subheader("📊 Quartile Information")

    qdf = pd.DataFrame({
        "Statistic": [
            "Q1 (25%)",
            "Q2 (Median)",
            "Q3 (75%)",
            "Inter Quartile Range"
        ],
        "Value": [
            round(q1,2),
            round(q2,2),
            round(q3,2),
            round(iqr,2)
        ]
    })

    st.dataframe(
        qdf,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Distribution Shape
    # --------------------------------------------------------

    st.subheader("📈 Distribution Shape")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Skewness", round(skewness,3))

    with c2:
        st.metric("Kurtosis", round(kurtosis,3))

    st.markdown("---")

    # --------------------------------------------------------
    # Complete Statistical Summary
    # --------------------------------------------------------

    st.subheader("📋 Statistical Summary")

    summary = pd.DataFrame({

        "Statistic":[
            "Count",
            "Mean",
            "Median",
            "Mode",
            "Standard Deviation",
            "Variance",
            "Minimum",
            "Maximum",
            "Range",
            "Q1",
            "Q2",
            "Q3",
            "IQR",
            "Skewness",
            "Kurtosis"
        ],

        "Value":[
            series.count(),
            mean,
            median,
            mode,
            std,
            variance,
            minimum,
            maximum,
            maximum-minimum,
            q1,
            q2,
            q3,
            iqr,
            skewness,
            kurtosis
        ]

    })

    st.dataframe(
        summary.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Five Number Summary
    # --------------------------------------------------------

    st.subheader("📦 Five Number Summary")

    five = pd.DataFrame({

        "Minimum":[minimum],
        "Q1":[q1],
        "Median":[median],
        "Q3":[q3],
        "Maximum":[maximum]

    })

    st.dataframe(
        five.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.success("✅ Statistical Analysis Completed Successfully.")
  # ============================================================
# SECTION 11 : DATA VISUALIZATION STUDIO
# ============================================================

if selected_page == "📊 Data Visualization":

    st.title("📊 Interactive Data Visualization Studio")

    if not dataset_available():
        st.warning("⚠ Please upload a dataset first.")
        st.stop()

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df.copy()
    else:
        df = st.session_state.df.copy()

    st.info("Create beautiful interactive charts using Plotly.")

    st.markdown("---")

    # --------------------------------------------------------
    # Chart Selection
    # --------------------------------------------------------

    chart_type = st.selectbox(
        "📈 Select Chart Type",
        [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Histogram",
            "Box Plot",
            "Pie Chart",
            "Area Chart",
            "Violin Plot",
            "Bubble Chart"
        ]
    )

    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    x_col = st.selectbox("Select X-axis", columns)

    y_col = None

    if chart_type != "Pie Chart":
        y_col = st.selectbox("Select Y-axis", numeric_columns)

    st.markdown("---")

    # --------------------------------------------------------
    # Generate Chart
    # --------------------------------------------------------

    fig = None

    if chart_type == "Bar Chart":

        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=x_col,
            title=f"{y_col} vs {x_col}"
        )

    elif chart_type == "Line Chart":

        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            markers=True,
            title=f"{y_col} vs {x_col}"
        )

    elif chart_type == "Scatter Plot":

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=x_col,
            title=f"{y_col} vs {x_col}"
        )

    elif chart_type == "Histogram":

        fig = px.histogram(
            df,
            x=x_col,
            nbins=30,
            title=f"Histogram of {x_col}"
        )

    elif chart_type == "Box Plot":

        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            color=x_col,
            title=f"Box Plot of {y_col}"
        )

    elif chart_type == "Pie Chart":

        value_col = st.selectbox(
            "Select Values",
            numeric_columns
        )

        fig = px.pie(
            df,
            names=x_col,
            values=value_col,
            title=f"{value_col} Distribution"
        )

    elif chart_type == "Area Chart":

        fig = px.area(
            df,
            x=x_col,
            y=y_col,
            title=f"{y_col} Area Chart"
        )

    elif chart_type == "Violin Plot":

        fig = px.violin(
            df,
            x=x_col,
            y=y_col,
            box=True,
            title=f"Violin Plot of {y_col}"
        )

    elif chart_type == "Bubble Chart":

        size_col = st.selectbox(
            "Bubble Size",
            numeric_columns
        )

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            size=size_col,
            color=x_col,
            title="Bubble Chart"
        )

    # --------------------------------------------------------
    # Display Chart
    # --------------------------------------------------------

    if fig is not None:

        fig.update_layout(
            template="plotly_white",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------------

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr()

        heatmap = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True
        )

    else:

        st.info("At least two numeric columns are required.")

    st.markdown("---")

    # --------------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.success("✅ Visualization Generated Successfully!")
  # ============================================================
# SECTION 12 : MACHINE LEARNING STUDIO
# ============================================================

if selected_page == "🤖 Machine Learning":

    st.title("🤖 Machine Learning Studio")

    if not dataset_available():
        st.warning("⚠ Please upload a dataset first.")
        st.stop()

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df.copy()
    else:
        df = st.session_state.df.copy()

    st.info("Train a Machine Learning model in just a few clicks.")

    st.markdown("---")

    # --------------------------------------------------------
    # Target Column
    # --------------------------------------------------------

    target = st.selectbox(
        "🎯 Select Target Column",
        df.columns
    )

    X = df.drop(columns=[target])
    y = df[target]

    # --------------------------------------------------------
    # Encode Categorical Features
    # --------------------------------------------------------

    X = pd.get_dummies(X)

    if y.dtype == "object":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # --------------------------------------------------------
    # Detect Problem Type
    # --------------------------------------------------------

    if len(np.unique(y)) <= 15:

        problem = "Classification"

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "Support Vector Machine": SVC()
        }

    else:

        problem = "Regression"

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor()
        }

    st.success(f"Detected Problem Type : **{problem}**")

    st.markdown("---")

    # --------------------------------------------------------
    # Model Selection
    # --------------------------------------------------------

    model_name = st.selectbox(
        "Choose Algorithm",
        list(models.keys())
    )

    test_size = st.slider(
        "Test Size (%)",
        10,
        40,
        20
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Train Model
    # --------------------------------------------------------

    if st.button("🚀 Train Model"):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size/100,
            random_state=42
        )

        model = models[model_name]

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        st.session_state.model = model
        st.session_state.target_column = target
        st.session_state.problem_type = problem

        st.success("✅ Model Trained Successfully!")

        st.markdown("---")

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        st.subheader("📊 Model Performance")

        if problem == "Classification":

            accuracy = accuracy_score(y_test, prediction)
            precision = precision_score(
                y_test,
                prediction,
                average="weighted",
                zero_division=0
            )
            recall = recall_score(
                y_test,
                prediction,
                average="weighted",
                zero_division=0
            )
            f1 = f1_score(
                y_test,
                prediction,
                average="weighted",
                zero_division=0
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{accuracy:.3f}")
            c2.metric("Precision", f"{precision:.3f}")
            c3.metric("Recall", f"{recall:.3f}")
            c4.metric("F1 Score", f"{f1:.3f}")

            st.markdown("---")

            cm = confusion_matrix(y_test, prediction)

            fig = px.imshow(
                cm,
                text_auto=True,
                title="Confusion Matrix",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            mae = mean_absolute_error(y_test, prediction)
            mse = mean_squared_error(y_test, prediction)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, prediction)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("MAE", f"{mae:.3f}")
            c2.metric("MSE", f"{mse:.3f}")
            c3.metric("RMSE", f"{rmse:.3f}")
            c4.metric("R² Score", f"{r2:.3f}")

        st.markdown("---")

        # ----------------------------------------------------
        # Feature Importance
        # ----------------------------------------------------

        if hasattr(model, "feature_importances_"):

            st.subheader("⭐ Feature Importance")

            importance = pd.DataFrame({

                "Feature": X.columns,
                "Importance": model.feature_importances_

            })

            importance = importance.sort_values(
                by="Importance",
                ascending=False
            )

            fig = px.bar(
                importance.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Important Features"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown("---")

        # ----------------------------------------------------
        # Save Model
        # ----------------------------------------------------

        joblib.dump(
            model,
            "models/trained_model.pkl"
        )

        st.success("💾 Model saved successfully.")

        st.markdown("---")

        # ----------------------------------------------------
        # Prediction Preview
        # ----------------------------------------------------

        result = pd.DataFrame({

            "Actual": y_test,
            "Predicted": prediction

        })

        st.subheader("📋 Prediction Preview")

        st.dataframe(
            result.head(20),
            use_container_width=True
        )
      # ============================================================
# SECTION 13 : PREDICTION STUDIO
# ============================================================

if selected_page == "🎯 Prediction":

    st.title("🎯 AI Prediction Studio")

    # --------------------------------------------------------
    # Check Model
    # --------------------------------------------------------

    if st.session_state.model is None:

        st.warning("⚠ Please train a Machine Learning model first.")

        st.stop()

    model = st.session_state.model

    target = st.session_state.target_column

    problem = st.session_state.problem_type

    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df.copy()
    else:
        df = st.session_state.df.copy()

    st.success(f"✅ Loaded Model : {problem}")

    st.markdown("---")

    # --------------------------------------------------------
    # Input Features
    # --------------------------------------------------------

    st.subheader("📝 Enter Feature Values")

    X = df.drop(columns=[target])

    X = pd.get_dummies(X)

    user_data = {}

    for column in X.columns:

        user_data[column] = st.number_input(
            column,
            value=0.0,
            format="%.4f"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button("🚀 Predict"):

        input_df = pd.DataFrame([user_data])

        input_df = input_df.reindex(
            columns=X.columns,
            fill_value=0
        )

        prediction = model.predict(input_df)

        st.session_state.predictions = prediction

        st.success("✅ Prediction Completed")

        st.markdown("---")

        # ----------------------------------------------------
        # Prediction Result
        # ----------------------------------------------------

        st.subheader("🎯 Prediction Result")

        if problem == "Classification":

            st.metric(
                "Predicted Class",
                prediction[0]
            )

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(input_df)

                confidence = np.max(probability) * 100

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                fig = px.bar(
                    x=range(len(probability[0])),
                    y=probability[0],
                    labels={
                        "x":"Class",
                        "y":"Probability"
                    },
                    title="Prediction Probability"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            st.metric(
                "Predicted Value",
                round(float(prediction[0]),3)
            )

        st.markdown("---")

        # ----------------------------------------------------
        # Input Summary
        # ----------------------------------------------------

        st.subheader("📋 Input Summary")

        summary = pd.DataFrame({

            "Feature":input_df.columns,
            "Value":input_df.iloc[0]

        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # ----------------------------------------------------
        # Download Prediction
        # ----------------------------------------------------

        result = pd.DataFrame({

            "Prediction":prediction

        })

        csv = result.to_csv(index=False)

        st.download_button(

            "⬇ Download Prediction",

            data=csv,

            file_name="prediction.csv",

            mime="text/csv"

        )

        st.success("🎉 Prediction Report Ready!")
      # ============================================================
# SECTION 14 : REPORT GENERATOR
# ============================================================

if selected_page == "📄 Report Generator":

    st.title("📄 AI Report Generator")

    if not dataset_available():

        st.warning("⚠ Please upload a dataset first.")
        st.stop()

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df.copy()
    else:
        df = st.session_state.df.copy()

    st.info("Generate a professional PDF report of your analysis.")

    st.markdown("---")

    # --------------------------------------------------------
    # Report Information
    # --------------------------------------------------------

    report_title = st.text_input(
        "Report Title",
        "Data Analysis Report"
    )

    analyst = st.text_input(
        "Analyst Name",
        "Dhruv Panchal"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Generate PDF
    # --------------------------------------------------------

    if st.button("📄 Generate PDF Report"):

        pdf_path = "reports/DataVerse_Report.pdf"

        c = canvas.Canvas(
            pdf_path,
            pagesize=letter
        )

        width, height = letter

        y = height - 40

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        c.setFont("Helvetica-Bold",18)
        c.drawString(40,y,report_title)

        y -= 25

        c.setFont("Helvetica",11)
        c.drawString(
            40,
            y,
            f"Prepared By : {analyst}"
        )

        y -= 20

        c.drawString(
            40,
            y,
            f"Dataset : {st.session_state.file_name}"
        )

        y -= 30

        # ----------------------------------------------------
        # Dataset Summary
        # ----------------------------------------------------

        c.setFont("Helvetica-Bold",14)
        c.drawString(40,y,"Dataset Summary")

        y -= 20

        c.setFont("Helvetica",11)

        c.drawString(
            50,
            y,
            f"Rows : {df.shape[0]}"
        )

        y -= 18

        c.drawString(
            50,
            y,
            f"Columns : {df.shape[1]}"
        )

        y -= 18

        c.drawString(
            50,
            y,
            f"Missing Values : {df.isnull().sum().sum()}"
        )

        y -= 18

        c.drawString(
            50,
            y,
            f"Duplicate Rows : {df.duplicated().sum()}"
        )

        y -= 30

        # ----------------------------------------------------
        # Numeric Statistics
        # ----------------------------------------------------

        c.setFont("Helvetica-Bold",14)

        c.drawString(
            40,
            y,
            "Numeric Summary"
        )

        y -= 20

        numeric = df.select_dtypes(include=np.number)

        if not numeric.empty:

            for col in numeric.columns:

                if y < 70:

                    c.showPage()

                    y = height - 40

                mean = round(numeric[col].mean(),2)

                std = round(numeric[col].std(),2)

                minimum = round(numeric[col].min(),2)

                maximum = round(numeric[col].max(),2)

                c.setFont("Helvetica",10)

                c.drawString(
                    50,
                    y,
                    f"{col}"
                )

                y -= 15

                c.drawString(
                    70,
                    y,
                    f"Mean : {mean}"
                )

                y -= 15

                c.drawString(
                    70,
                    y,
                    f"Std Dev : {std}"
                )

                y -= 15

                c.drawString(
                    70,
                    y,
                    f"Min : {minimum}"
                )

                y -= 15

                c.drawString(
                    70,
                    y,
                    f"Max : {maximum}"
                )

                y -= 25

        # ----------------------------------------------------
        # ML Information
        # ----------------------------------------------------

        if st.session_state.model is not None:

            if y < 120:

                c.showPage()

                y = height - 40

            c.setFont("Helvetica-Bold",14)

            c.drawString(
                40,
                y,
                "Machine Learning"
            )

            y -= 20

            c.setFont("Helvetica",11)

            c.drawString(
                50,
                y,
                f"Problem Type : {st.session_state.problem_type}"
            )

            y -= 18

            c.drawString(
                50,
                y,
                f"Target Column : {st.session_state.target_column}"
            )

            y -= 30

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if st.session_state.predictions is not None:

            c.setFont("Helvetica-Bold",14)

            c.drawString(
                40,
                y,
                "Latest Prediction"
            )

            y -= 20

            c.setFont("Helvetica",11)

            c.drawString(
                50,
                y,
                str(st.session_state.predictions[0])
            )

            y -= 25

        # ----------------------------------------------------
        # Footer
        # ----------------------------------------------------

        c.setFont("Helvetica",9)

        c.drawString(
            40,
            30,
            "Generated by DataVerse AI"
        )

        c.save()

        st.session_state.report_generated = True

        st.success("✅ PDF Report Generated Successfully!")

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        with open(pdf_path,"rb") as pdf:

            st.download_button(

                "⬇ Download PDF Report",

                data=pdf,

                file_name="DataVerse_Report.pdf",

                mime="application/pdf"

            )
          # ============================================================
# SECTION 15 : DOWNLOAD CENTER
# ============================================================

if selected_page == "⬇ Download Center":

    st.title("⬇ Download Center")

    st.info("Download your datasets, trained model, predictions, and reports from one place.")

    st.markdown("---")

    # ========================================================
    # Cleaned Dataset
    # ========================================================

    st.subheader("📂 Cleaned Dataset")

    if st.session_state.clean_df is not None:

        csv = st.session_state.clean_df.to_csv(index=False)

        st.download_button(
            label="⬇ Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    else:

        st.warning("No cleaned dataset available.")

    st.markdown("---")

    # ========================================================
    # Trained Machine Learning Model
    # ========================================================

    st.subheader("🤖 Trained Model")

    model_path = "models/trained_model.pkl"

    if os.path.exists(model_path):

        with open(model_path, "rb") as model_file:

            st.download_button(
                label="💾 Download Trained Model (.pkl)",
                data=model_file,
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )

    else:

        st.warning("Train a model first.")

    st.markdown("---")

    # ========================================================
    # Prediction Results
    # ========================================================

    st.subheader("🎯 Prediction Results")

    if st.session_state.predictions is not None:

        prediction_df = pd.DataFrame({
            "Prediction": st.session_state.predictions
        })

        csv_prediction = prediction_df.to_csv(index=False)

        st.download_button(
            label="📊 Download Prediction CSV",
            data=csv_prediction,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

        st.dataframe(
            prediction_df,
            use_container_width=True
        )

    else:

        st.warning("No prediction results available.")

    st.markdown("---")

    # ========================================================
    # PDF Report
    # ========================================================

    st.subheader("📄 PDF Report")

    report_path = "reports/DataVerse_Report.pdf"

    if os.path.exists(report_path):

        with open(report_path, "rb") as pdf:

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf,
                file_name="DataVerse_Report.pdf",
                mime="application/pdf"
            )

    else:

        st.warning("Generate a report first.")

    st.markdown("---")

    # ========================================================
    # Dataset Information
    # ========================================================

    st.subheader("📋 Current Project Summary")

    if dataset_available():

        current_df = (
            st.session_state.clean_df
            if st.session_state.clean_df is not None
            else st.session_state.df
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Rows",
                current_df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                current_df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                current_df.isnull().sum().sum()
            )

    st.markdown("---")

    # ========================================================
    # Download Status
    # ========================================================

    st.success("🎉 Your project files are ready for download!")

    st.info("""
    Available Downloads

    ✅ Cleaned Dataset (.csv)

    ✅ Trained ML Model (.pkl)

    ✅ Prediction Results (.csv)

    ✅ PDF Analysis Report
    """)
  # ============================================================
# SECTION 16 : ABOUT
# ============================================================

if selected_page == "ℹ About":

    st.title("ℹ About DataVerse AI")

    st.markdown("""
    ## 📊 DataVerse AI

    **DataVerse AI** is an intelligent data analysis platform built
    using **Python**, **Streamlit**, and **Machine Learning**.

    It enables users to upload datasets, clean data, perform
    exploratory data analysis, visualize information, train machine
    learning models, generate predictions, and export professional
    reports—all from a single user-friendly dashboard.
    """)

    st.markdown("---")

    # ========================================================
    # Project Information
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🚀 Project Details")

        st.write("**Project Name:** DataVerse AI")
        st.write("**Version:** 1.0")
        st.write("**Platform:** Streamlit")
        st.write("**Language:** Python")
        st.write("**Category:** Data Analytics & Machine Learning")

    with col2:

        st.subheader("👨‍💻 Developer")

        st.write("**Name:** Dhruv Panchal")
        st.write("**Role:** Data Science Student")
        st.write("**Country:** India")
        st.write("**Project Type:** Academic & Portfolio")

    st.markdown("---")

    # ========================================================
    # Features
    # ========================================================

    st.subheader("⭐ Features")

    features = [
        "📂 CSV & Excel Dataset Upload",
        "🧹 Data Cleaning",
        "📋 Dataset Overview",
        "📊 Interactive Visualizations",
        "📈 Statistical Analysis",
        "🤖 Machine Learning",
        "🎯 AI Prediction",
        "📄 PDF Report Generation",
        "⬇ Download Center",
        "📱 Responsive Dashboard"
    ]

    for feature in features:
        st.write(feature)

    st.markdown("---")

    # ========================================================
    # Technologies
    # ========================================================

    st.subheader("🛠 Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.info("""
        **Frontend**
        - Streamlit
        - HTML
        - CSS
        """)

    with tech2:
        st.info("""
        **Backend**
        - Python
        - Pandas
        - NumPy
        """)

    with tech3:
        st.info("""
        **Machine Learning**
        - Scikit-learn
        - Joblib
        - Plotly
        """)

    st.markdown("---")

    # ========================================================
    # Python Libraries
    # ========================================================

    st.subheader("📚 Python Libraries")

    libraries = [
        "Pandas",
        "NumPy",
        "Streamlit",
        "Plotly",
        "Matplotlib",
        "Scikit-learn",
        "Joblib",
        "ReportLab"
    ]

    st.dataframe(
        pd.DataFrame({"Library": libraries}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # ========================================================
    # Application Workflow
    # ========================================================

    st.subheader("🔄 Workflow")

    st.success("""
    Upload Dataset

            ↓

    Data Cleaning

            ↓

    Data Analysis

            ↓

    Data Visualization

            ↓

    Machine Learning

            ↓

    Prediction

            ↓

    Generate PDF Report

            ↓

    Download Results
    """)

    st.markdown("---")

    # ========================================================
    # Thank You
    # ========================================================

    st.subheader("🙏 Thank You")

    st.write("""
    Thank you for using **DataVerse AI**.

    This project demonstrates the integration of Data Analytics,
    Data Visualization, and Machine Learning into a single
    interactive platform. It is designed as a portfolio project
    and learning resource for students, researchers, and data
    professionals.
    """)

    st.success("⭐ Thank you for exploring DataVerse AI!")
  # ============================================================
# SECTION 17 : PROFESSIONAL FOOTER
# ============================================================

st.markdown("---")

# ============================================================
# Dashboard Status
# ============================================================

with st.expander("📊 Application Status", expanded=False):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset",
            "Loaded" if st.session_state.df is not None else "Not Loaded"
        )

    with col2:
        st.metric(
            "ML Model",
            "Ready" if st.session_state.model is not None else "Not Trained"
        )

    with col3:
        st.metric(
            "Prediction",
            "Available" if st.session_state.predictions is not None else "None"
        )

    with col4:
        st.metric(
            "Report",
            "Generated" if st.session_state.report_generated else "Pending"
        )

# ============================================================
# System Information
# ============================================================

st.markdown("### ⚙ System Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.info("""
**Application**

📊 DataVerse AI

Version : 1.0
""")

with info2:
    st.info("""
**Technology**

🐍 Python

🎈 Streamlit

🤖 Scikit-Learn
""")

with info3:
    st.info("""
**Visualization**

📈 Plotly

📊 Pandas

🔢 NumPy
""")

# ============================================================
# Feature Summary
# ============================================================

st.markdown("### 🚀 Available Modules")

features = [
    "🏠 Home Dashboard",
    "📂 Dataset Upload",
    "📋 Dataset Overview",
    "🧹 Data Cleaning",
    "📈 Statistics",
    "📊 Data Visualization",
    "🤖 Machine Learning",
    "🎯 AI Prediction",
    "📄 PDF Report Generator",
    "⬇ Download Center",
    "ℹ About"
]

col1, col2 = st.columns(2)

half = len(features) // 2

with col1:
    for feature in features[:half]:
        st.success(feature)

with col2:
    for feature in features[half:]:
        st.success(feature)

# ============================================================
# Performance Summary
# ============================================================

st.markdown("### 📈 Performance")

performance = pd.DataFrame({

    "Feature":[
        "Dataset Upload",
        "Cleaning",
        "Visualization",
        "Machine Learning",
        "Prediction",
        "Report Generation"
    ],

    "Status":[
        "✅ Ready",
        "✅ Ready",
        "✅ Ready",
        "✅ Ready",
        "✅ Ready",
        "✅ Ready"
    ]

})

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# Final Footer
# ============================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;padding:25px;'>

<h2 style='color:#2563EB;'>
📊 DataVerse AI
</h2>

<p style='font-size:18px;'>

An Intelligent Data Analysis & Machine Learning Platform

</p>

<p>

Built with ❤️ using Python, Streamlit, Pandas,
Plotly and Scikit-Learn

</p>

<hr>

<p>

© 2026 Dhruv Panchal

</p>

<p>

Version 1.0

</p>

</div>
""",
unsafe_allow_html=True
)

# ============================================================
# Application Completed
# ============================================================

st.balloons()

st.success(
"""
🎉 Congratulations!

DataVerse AI is fully loaded and ready to use.

Upload your dataset and begin your complete
Data Analytics and Machine Learning workflow.
"""
)
