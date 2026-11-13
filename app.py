from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure
from bokeh.resources import CDN


# ==========================================
# 1. DASHBOARD SETTINGS
# THREE SEPARATE PAGES - NO FILTERS
# ==========================================

st.set_page_config(
    page_title="Malaysia Data Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

GREEN = "#287C62"
DARK_GREEN = "#184B3C"
ORANGE = "#D38A49"

sns.set_theme(
    style="whitegrid",
    font_scale=1.0
)

st.markdown("""
<style>
.block-container {
    max-width: 1120px;
    padding-top: 1.4rem;
}

h1, h2, h3 {
    color: #184B3C;
}

div[data-testid="stMetric"] {
    background-color: #F4F9F6;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD YOUR THREE MERGED DATASETS
# ==========================================

@st.cache_data
def load_data():

    folder = Path(__file__).resolve().parent

    filenames = {
        "obj1": "df_merged_income_birth (1).csv",
        "obj2": "df_merged_income_rates.csv",
        "obj3": "df_merged_birth_rates_2016_2025.csv"
    }

    frames = {}

    for key, filename in filenames.items():

        df = pd.read_csv(folder / filename)

        df["state"] = (
            df["state"].astype(str).str.strip()
        )

        df["year"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        ).dt.year

        df = df.dropna(
            subset=["state", "year"]
        )

        df = df[
            df["state"] != "Malaysia"
        ].copy()

        df["year"] = df["year"].astype(int)
