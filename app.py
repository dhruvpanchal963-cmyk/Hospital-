# ============================================================
# 📊 DataVerse AI
# Intelligent Data Analyst Assistant
# Author : Dhruv Panchal
# Version : 2.0 (UI Upgrade)
# ============================================================

# =========================
# Core Libraries
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import warnings

warnings.filterwarnings("ignore")


# =========================
# Visualization
# =========================

import plotly.express as px
import plotly.graph_objects as go


# =========================
# Machine Learning
# =========================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


# =========================
# PDF Report
# =========================

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataVerse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FOLDERS
# ============================================================

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

APP_NAME = "📊 DataVerse AI"
VERSION = "2.0"
AUTHOR = "Dhruv Panchal"


# ============================================================
# PROFESSIONAL UI CSS
# ============================================================

st.markdown("""

<style>

body {
    background-color:#0f172a;
}


.main-title {

    font-size:48px;
    font-weight:800;
    color:#2563eb;
    text-align:center;

}


.sub-title {

    font-size:20px;
    text-align:center;
    color:#64748b;

}


.card {

    padding:25px;
    border-radius:20px;
    background:#ffffff;
    box-shadow:0px 5px 20px rgba(0,0,0,0.1);
    margin:10px;

}


div[data-testid="stMetric"] {

    background:#ffffff;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 3px 15px rgba(0,0,0,0.08);

}


.stButton button {

    width:100%;
    border-radius:12px;
    height:45px;
    font-weight:bold;

}


footer {

    visibility:hidden;

}


</style>

""", unsafe_allow_html=True)



# ============================================================
# SESSION STATE
# ============================================================

default_values = {

    "df":None,
    "clean_df":None,
    "file_name":"",
    "model":None,
    "target_column":None,
    "problem_type":None,
    "predictions":None,
    "report_generated":False

}


for key,value in default_values.items():

    if key not in st.session_state:

        st.session_state[key]=value



# ============================================================
# RESET FUNCTION
# ============================================================


def reset_application():

    for key,value in default_values.items():

        st.session_state[key]=value



# ============================================================
# DATASET CHECK
# ============================================================


def dataset_available():

    return (
        st.session_state.df is not None
    )



# ============================================================
# SIDEBAR
# ============================================================


st.sidebar.markdown(
"""
# 📊 DataVerse AI

### Intelligent Data Analyst Assistant

"""
)


st.sidebar.markdown("---")


pages=[

"🏠 Home",
"📂 Upload Dataset",
"📋 Dataset Overview",
"🧹 Data Cleaning",
"📊 Visualization",
"📈 Statistics",
"🤖 Machine Learning",
"🎯 Prediction",
"📄 Report Generator",
"⬇ Download Center",
"ℹ About"

]


selected_page = st.sidebar.radio(

    "Navigation",

    pages

)



st.sidebar.markdown("---")


if st.sidebar.button("🔄 Reset Application"):

    reset_application()

    st.success("Application Reset Successfully")

    st.rerun()



st.sidebar.markdown("---")


if st.session_state.df is not None:

    st.sidebar.success("✅ Dataset Loaded")

    st.sidebar.write(
        "File:",
        st.session_state.file_name
    )

    st.sidebar.write(
        "Rows:",
        st.session_state.df.shape[0]
    )

    st.sidebar.write(
        "Columns:",
        st.session_state.df.shape[1]
    )


else:

    st.sidebar.warning(
        "⚠ No Dataset Uploaded"
    )


st.sidebar.markdown("---")


st.sidebar.info(

"""
**DataVerse AI**

Python  
Streamlit  
Machine Learning  

Developer:
Dhruv Panchal

"""

)
# ============================================================
# HOME DASHBOARD
# ============================================================

if selected_page == "🏠 Home":


    st.markdown(
    """
    <h1 class="main-title">
    📊 DataVerse AI
    </h1>

    <p class="sub-title">
    Intelligent Data Analyst Assistant
    </p>

    """,
    unsafe_allow_html=True
    )


    st.info(
    """
    Welcome to DataVerse AI.

    Upload any CSV or Excel dataset and perform:

    ✅ Data Cleaning  
    ✅ Data Visualization  
    ✅ Statistical Analysis  
    ✅ Machine Learning  
    ✅ Prediction  
    ✅ PDF Reports

    """
    )


    st.markdown("---")


    c1,c2,c3,c4 = st.columns(4)


    with c1:
        st.metric(
            "📂 Dataset",
            "CSV / Excel"
        )


    with c2:
        st.metric(
            "📊 Visualization",
            "10+ Charts"
        )


    with c3:
        st.metric(
            "🤖 ML Models",
            "6+ Algorithms"
        )


    with c4:
        st.metric(
            "📄 Reports",
            "PDF Export"
        )


    st.markdown("---")


    col1,col2 = st.columns(2)


    with col1:

        st.markdown(
        """
        <div class="card">

        <h3>📈 Data Analytics</h3>

        ✔ Data Cleaning  
        ✔ Missing Value Handling  
        ✔ Statistical Analysis  
        ✔ Interactive Charts  

        </div>

        """,
        unsafe_allow_html=True
        )


    with col2:

        st.markdown(
        """
        <div class="card">

        <h3>🤖 AI Machine Learning</h3>

        ✔ Classification  
        ✔ Regression  
        ✔ Prediction  
        ✔ Model Evaluation  

        </div>

        """,
        unsafe_allow_html=True
        )



    st.subheader(
        "🚀 Workflow"
    )


    st.success(
    """
    Upload Dataset

            ↓

    Clean Data

            ↓

    Analyze Dataset

            ↓

    Create Visualization

            ↓

    Train ML Model

            ↓

    Generate Prediction

            ↓

    Export Report

    """
    )



# ============================================================
# UPLOAD DATASET
# ============================================================


elif selected_page == "📂 Upload Dataset":


    st.title(
        "📂 Upload Dataset"
    )


    uploaded_file = st.file_uploader(

        "Upload CSV or Excel File",

        type=[
            "csv",
            "xlsx"
        ]

    )


    if uploaded_file:


        try:


            if uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(
                    uploaded_file
                )


            else:

                df = pd.read_excel(
                    uploaded_file
                )



            st.session_state.df = df

            st.session_state.clean_df = df.copy()

            st.session_state.file_name = uploaded_file.name



            st.success(
                "Dataset Uploaded Successfully"
            )


            st.markdown("---")


            c1,c2,c3 = st.columns(3)


            with c1:

                st.metric(
                    "Rows",
                    df.shape[0]
                )


            with c2:

                st.metric(
                    "Columns",
                    df.shape[1]
                )


            with c3:

                st.metric(
                    "Missing Values",
                    df.isnull().sum().sum()
                )



            st.markdown("---")


            st.subheader(
                "Dataset Preview"
            )


            st.dataframe(

                df.head(20),

                use_container_width=True

            )


            st.subheader(
                "Column Information"
            )


            info = pd.DataFrame({

                "Column":df.columns,

                "Datatype":
                df.dtypes.astype(str),

                "Missing":
                df.isnull().sum()

            })


            st.dataframe(

                info,

                use_container_width=True,

                hide_index=True

            )


        except Exception as e:


            st.error(
                f"Error: {e}"
            )



    else:

        st.info(
            "Please upload a dataset"
        )



# ============================================================
# DATASET OVERVIEW
# ============================================================


elif selected_page == "📋 Dataset Overview":


    st.title(
        "📋 Dataset Overview"
    )


    if not dataset_available():

        st.warning(
            "Upload dataset first"
        )

        st.stop()



    df = st.session_state.clean_df



    c1,c2,c3,c4 = st.columns(4)



    with c1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with c2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with c3:

        st.metric(
            "Missing",
            df.isnull().sum().sum()
        )


    with c4:

        st.metric(
            "Duplicates",
            df.duplicated().sum()
        )



    st.markdown("---")



    st.subheader(
        "📊 Data Types"
    )


    dtype = pd.DataFrame({

        "Column":df.columns,

        "Type":
        df.dtypes.astype(str),

        "Unique":
        df.nunique()

    })


    st.dataframe(

        dtype,

        use_container_width=True,

        hide_index=True

    )



    st.markdown("---")



    st.subheader(
        "📉 Missing Value Analysis"
    )


    missing = (
        df.isnull()
        .sum()
        .reset_index()
    )


    missing.columns=[
        "Column",
        "Missing"
    ]



    missing = missing[
        missing["Missing"]>0
    ]



    if len(missing)>0:


        fig = px.bar(

            missing,

            x="Column",

            y="Missing",

            title="Missing Values"

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


    else:


        st.success(
            "No Missing Values Found"
        )



    st.markdown("---")



    st.subheader(
        "👀 Dataset Sample"
    )


    st.dataframe(

        df.head(10),

        use_container_width=True

    )
    # ============================================================
# DATA CLEANING
# ============================================================


elif selected_page == "🧹 Data Cleaning":


    st.title("🧹 Data Cleaning Studio")


    if not dataset_available():

        st.warning("Upload dataset first")

        st.stop()



    df = st.session_state.clean_df.copy()



    st.metric(
        "Current Rows",
        df.shape[0]
    )


    st.markdown("---")



    # Remove duplicates

    st.subheader("🔁 Duplicate Removal")


    duplicates = df.duplicated().sum()


    st.write(
        f"Duplicate Rows: {duplicates}"
    )



    if st.button("Remove Duplicates"):


        df = df.drop_duplicates()


        st.session_state.clean_df = df


        st.success(
            "Duplicates Removed"
        )



    st.markdown("---")



    # Missing values


    st.subheader(
        "❗ Missing Value Treatment"
    )


    method = st.selectbox(

        "Choose Method",

        [
            "Mean",
            "Median",
            "Mode",
            "Drop Rows"
        ]

    )



    if st.button(
        "Apply Cleaning"
    ):


        temp = df.copy()



        if method == "Mean":

            numeric = temp.select_dtypes(
                include=np.number
            ).columns

            temp[numeric] = temp[numeric].fillna(
                temp[numeric].mean()
            )



        elif method == "Median":

            numeric = temp.select_dtypes(
                include=np.number
            ).columns

            temp[numeric] = temp[numeric].fillna(
                temp[numeric].median()
            )



        elif method == "Mode":


            for col in temp.columns:

                temp[col] = temp[col].fillna(
                    temp[col].mode()[0]
                )



        else:

            temp.dropna(
                inplace=True
            )



        st.session_state.clean_df = temp


        st.success(
            "Cleaning Completed"
        )



    st.markdown("---")



    st.subheader(
        "📋 Clean Dataset"
    )


    st.dataframe(

        st.session_state.clean_df.head(20),

        use_container_width=True

    )



    csv = st.session_state.clean_df.to_csv(
        index=False
    )


    st.download_button(

        "⬇ Download Clean Dataset",

        csv,

        "clean_dataset.csv",

        "text/csv"

    )



# ============================================================
# VISUALIZATION
# ============================================================


elif selected_page == "📊 Visualization":


    st.title(
        "📊 Visualization Studio"
    )


    if not dataset_available():

        st.warning(
            "Upload dataset first"
        )

        st.stop()



    df = st.session_state.clean_df



    chart = st.selectbox(

        "Select Chart",

        [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Histogram",
            "Box Plot",
            "Pie Chart"
        ]

    )



    columns = df.columns.tolist()


    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()



    x = st.selectbox(
        "X Axis",
        columns
    )



    y = None


    if chart != "Pie Chart":

        y = st.selectbox(
            "Y Axis",
            numeric
        )



    fig=None



    if chart=="Bar Chart":


        fig=px.bar(
            df,
            x=x,
            y=y
        )



    elif chart=="Line Chart":


        fig=px.line(
            df,
            x=x,
            y=y
        )



    elif chart=="Scatter Plot":


        fig=px.scatter(
            df,
            x=x,
            y=y
        )



    elif chart=="Histogram":


        fig=px.histogram(
            df,
            x=x
        )



    elif chart=="Box Plot":


        fig=px.box(
            df,
            y=y
        )



    elif chart=="Pie Chart":


        value=st.selectbox(
            "Value Column",
            numeric
        )


        fig=px.pie(
            df,
            names=x,
            values=value
        )



    if fig:


        fig.update_layout(
            height=600,
            template="plotly_white"
        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )



    st.subheader(
        "🔥 Correlation Heatmap"
    )


    numeric_df=df.select_dtypes(
        include=np.number
    )


    if numeric_df.shape[1]>=2:


        heat=px.imshow(

            numeric_df.corr(),

            text_auto=True

        )


        st.plotly_chart(

            heat,

            use_container_width=True

        )



# ============================================================
# STATISTICS
# ============================================================


elif selected_page == "📈 Statistics":


    st.title(
        "📈 Statistics Dashboard"
    )


    if not dataset_available():

        st.warning(
            "Upload dataset first"
        )

        st.stop()



    df=st.session_state.clean_df



    numeric=df.select_dtypes(
        include=np.number
    )



    if numeric.empty:


        st.error(
            "No Numeric Columns"
        )

        st.stop()



    column=st.selectbox(

        "Select Column",

        numeric.columns

    )



    data=numeric[column]



    c1,c2,c3,c4=st.columns(4)



    c1.metric(
        "Mean",
        round(data.mean(),2)
    )


    c2.metric(
        "Median",
        round(data.median(),2)
    )


    c3.metric(
        "Minimum",
        round(data.min(),2)
    )


    c4.metric(
        "Maximum",
        round(data.max(),2)
    )



    st.subheader(
        "Complete Summary"
    )


    st.dataframe(

        data.describe(),

        use_container_width=True

    )



# ============================================================
# MACHINE LEARNING
# ============================================================


elif selected_page == "🤖 Machine Learning":


    st.title(
        "🤖 Machine Learning Studio"
    )


    if not dataset_available():

        st.warning(
            "Upload dataset first"
        )

        st.stop()



    df=st.session_state.clean_df



    target=st.selectbox(

        "Select Target Column",

        df.columns

    )



    X=df.drop(
        columns=[target]
    )


    y=df[target]



    X=pd.get_dummies(X)



    if y.dtype=="object":

        encoder=LabelEncoder()

        y=encoder.fit_transform(y)



    if len(np.unique(y))<=15:


        problem="Classification"


        models={

            "Logistic Regression":
            LogisticRegression(max_iter=1000),

            "Decision Tree":
            DecisionTreeClassifier(),

            "Random Forest":
            RandomForestClassifier(),

            "KNN":
            KNeighborsClassifier(),

            "SVM":
            SVC()

        }


    else:


        problem="Regression"


        models={

            "Linear Regression":
            LinearRegression(),

            "Random Forest Regressor":
            RandomForestRegressor()

        }



    st.info(
        f"Problem Type: {problem}"
    )



    model_name=st.selectbox(

        "Choose Model",

        models.keys()

    )



    if st.button(
        "🚀 Train Model"
    ):


        X_train,X_test,y_train,y_test=train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42

        )



        model=models[model_name]


        model.fit(
            X_train,
            y_train
        )


        pred=model.predict(
            X_test
        )



        st.session_state.model=model

        st.session_state.target_column=target

        st.session_state.problem_type=problem



        st.success(
            "Model Trained Successfully"
        )



        if problem=="Classification":


            st.metric(
                "Accuracy",
                round(
                    accuracy_score(
                        y_test,
                        pred
                    ),
                    3
                )
            )


        else:


            st.metric(
                "R2 Score",
                round(
                    r2_score(
                        y_test,
                        pred
                    ),
                    3
                )
            )



        joblib.dump(

            model,

            "models/trained_model.pkl"

        )
        # ============================================================
# AI PREDICTION
# ============================================================


elif selected_page == "🎯 Prediction":


    st.title(
        "🎯 AI Prediction Studio"
    )


    if st.session_state.model is None:

        st.warning(
            "Please train a model first"
        )

        st.stop()



    if not dataset_available():

        st.warning(
            "Upload dataset first"
        )

        st.stop()



    model = st.session_state.model

    target = st.session_state.target_column



    df = st.session_state.clean_df



    X = df.drop(
        columns=[target]
    )


    X = pd.get_dummies(X)



    st.subheader(
        "Enter Feature Values"
    )



    user_input={}



    for col in X.columns:


        user_input[col]=st.number_input(

            col,

            value=0.0

        )



    if st.button(
        "🚀 Predict"
    ):


        input_df=pd.DataFrame(
            [user_input]
        )


        input_df=input_df.reindex(

            columns=X.columns,

            fill_value=0

        )


        prediction=model.predict(
            input_df
        )



        st.session_state.predictions=prediction



        st.success(
            "Prediction Completed"
        )



        st.metric(

            "Prediction Result",

            str(prediction[0])

        )



        result=pd.DataFrame({

            "Prediction":
            prediction

        })



        st.download_button(

            "⬇ Download Prediction",

            result.to_csv(index=False),

            "prediction.csv",

            "text/csv"

        )



# ============================================================
# REPORT GENERATOR
# ============================================================


elif selected_page == "📄 Report Generator":


    st.title(
        "📄 PDF Report Generator"
    )



    if not dataset_available():

        st.warning(
            "Upload dataset first"
        )

        st.stop()



    df=st.session_state.clean_df



    if st.button(
        "Generate PDF Report"
    ):



        path="reports/DataVerse_Report.pdf"



        pdf=canvas.Canvas(

            path,

            pagesize=letter

        )



        width,height=letter



        y=height-50



        pdf.setFont(
            "Helvetica-Bold",
            18
        )


        pdf.drawString(

            50,

            y,

            "DataVerse AI Report"

        )



        y-=40



        pdf.setFont(
            "Helvetica",
            12
        )


        information=[

            f"Dataset : {st.session_state.file_name}",

            f"Rows : {df.shape[0]}",

            f"Columns : {df.shape[1]}",

            f"Missing Values : {df.isnull().sum().sum()}",

            f"Duplicate Rows : {df.duplicated().sum()}"

        ]



        for item in information:


            pdf.drawString(

                50,

                y,

                item

            )

            y-=25



        if st.session_state.model:


            pdf.drawString(

                50,

                y,

                "Machine Learning Model Available"

            )



        pdf.save()



        st.session_state.report_generated=True



        st.success(
            "PDF Generated Successfully"
        )



        with open(path,"rb") as file:


            st.download_button(

                "⬇ Download PDF",

                file,

                "DataVerse_Report.pdf",

                "application/pdf"

            )



# ============================================================
# DOWNLOAD CENTER
# ============================================================


elif selected_page == "⬇ Download Center":


    st.title(
        "⬇ Download Center"
    )



    if st.session_state.clean_df is not None:



        st.download_button(

            "Download Dataset",

            st.session_state.clean_df.to_csv(
                index=False
            ),

            "dataset.csv",

            "text/csv"

        )



    else:

        st.info(
            "No Dataset Available"
        )



    if os.path.exists(
        "models/trained_model.pkl"
    ):


        with open(
            "models/trained_model.pkl",
            "rb"
        ) as model:


            st.download_button(

                "Download ML Model",

                model,

                "trained_model.pkl"

            )



    if os.path.exists(
        "reports/DataVerse_Report.pdf"
    ):


        with open(
            "reports/DataVerse_Report.pdf",
            "rb"
        ) as pdf:


            st.download_button(

                "Download PDF Report",

                pdf,

                "DataVerse_Report.pdf"

            )



# ============================================================
# ABOUT
# ============================================================


elif selected_page == "ℹ About":


    st.title(
        "ℹ About DataVerse AI"
    )



    st.markdown(
    """

## 📊 DataVerse AI

DataVerse AI is an intelligent data analysis platform.

Features:

✅ Dataset Upload  
✅ Data Cleaning  
✅ Visualization  
✅ Statistics  
✅ Machine Learning  
✅ Prediction  
✅ PDF Reports  


Technology:

🐍 Python

🎈 Streamlit

📊 Pandas

🤖 Scikit-Learn

📈 Plotly


Developer:

**Dhruv Panchal**

Version: 2.0


"""
    )



# ============================================================
# FINAL FOOTER
# ============================================================


st.markdown("---")


st.markdown(

"""

<center>

<h3>
📊 DataVerse AI
</h3>

<p>
Intelligent Data Analyst Assistant
</p>

<p>
Built with Python • Streamlit • Machine Learning
</p>

<p>
© 2026 Dhruv Panchal
</p>

</center>

""",

unsafe_allow_html=True

)
