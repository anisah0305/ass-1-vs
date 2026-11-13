"""Malaysia Data Insights — JIE42403 Scientific Visualisation.

Three research objectives, using OpenDOSM state-level merged datasets.
Run: streamlit run app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE SETTINGS AND THEME
# ============================================================
st.set_page_config(
    page_title="Malaysia Data Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GREEN = "#287C62"
DARK_GREEN = "#184B3C"
ORANGE = "#D38A49"
YEARS_WITH_INCOME = [2016, 2019, 2020, 2022, 2024]

st.markdown(
    """
    <style>
        .block-container {max-width: 1120px; padding-top: 1.3rem;}
        h1, h2, h3 {color: #184B3C;}
        div[data-testid="stMetric"] {
            background-color: #F4F9F6;
            border: 1px solid #DBE7DF;
            border-radius: 10px;
            padding: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. LOAD DATA — FILENAMES MATCH YOUR GITHUB REPOSITORY
# ============================================================
@st.cache_data
def load_data():
    folder = Path(__file__).resolve().parent
    filenames = {
        "obj1": "df_merged_income_birth (1).csv",
        "obj2": "df_merged_income_rates.csv",
        "obj3": "df_merged_birth_rates_2016_2025.csv",
    }
    required = {
        "obj1": {"state", "date", "income_mean", "abs"},
        "obj2": {"state", "date", "income_mean", "completion", "stage"},
        "obj3": {"state", "date", "abs", "completion", "stage"},
    }
    frames = {}

    for name, filename in filenames.items():
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {filename}")

        df = pd.read_csv(path)
        missing = required[name] - set(df.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {', '.join(sorted(missing))}")

        df["state"] = df["state"].fillna("").astype(str).str.strip()
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
        df = df.loc[
            df["state"].ne("")
            & df["state"].str.casefold().ne("malaysia")
            & df["year"].notna()
        ].copy()
        df["year"] = df["year"].astype(int)

        for field in ("abs", "rate", "income_mean", "income_median", "completion"):
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors="coerce")

        if "sex" in df.columns:
            df["sex"] = df["sex"].astype(str).str.strip().str.lower()
            df = df.loc[df["sex"].eq("both")].copy()

        if "stage" in df.columns:
            df["stage"] = df["stage"].astype(str).str.strip()

        frames[name] = df

    return frames["obj1"], frames["obj2"], frames["obj3"]
