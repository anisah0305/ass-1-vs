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

        for col in [
            "income_mean",
            "rate",
            "completion"
        ]:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        if "sex" in df.columns:

            df["sex"] = (
                df["sex"].astype(str).str.strip()
            )

            df = df[
                df["sex"] == "both"
            ].copy()

        if "stage" in df.columns:

            df["stage"] = (
                df["stage"].astype(str).str.strip()
            )

        frames[key] = df

    return (
        frames["obj1"],
        frames["obj2"],
        frames["obj3"]
    )


try:
    df1, df2, df3 = load_data()

except (FileNotFoundError, KeyError, ValueError) as error:

    st.error(
        f"CSV loading error: {error}"
    )

    st.info(
        "Ensure all three merged CSV files "
        "are uploaded to your GitHub repository."
    )

    st.stop()


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def pearson(data, x, y):

    valid = data[[x, y]].dropna()

    if (
        len(valid) < 3
        or valid[x].nunique() < 2
        or valid[y].nunique() < 2
    ):
        return float("nan")

    return float(
        valid[x].corr(valid[y])
    )


def relationship(r):

    if pd.isna(r):
        return "undefined"

    if abs(r) < 0.10:
        strength = "negligible"

    elif abs(r) < 0.30:
        strength = "weak"

    elif abs(r) < 0.70:
        strength = "moderate"

    else:
        strength = "strong"

    if r > 0:
        direction = "positive"

    elif r < 0:
        direction = "negative"

    else:
        direction = "zero"

    return f"{strength} {direction}"


def summary_cards(items):

    columns = st.columns(3)

    for col, (label, value) in zip(
        columns,
        items
    ):

        with col:

            st.metric(
                label,
                value,
                border=True
            )


def data_preview(data, columns):

    with st.expander(
        "View Data Used in Visualization"
    ):

        st.dataframe(
            data[columns],
            hide_index=True,
            width="stretch"
        )


# ==========================================
# PAGE 1: OBJECTIVE 1
# SCATTER PLOT
# ==========================================

def objective1():

    st.header(
        "Objective 1 — Annual Live Births and Household Income"
    )

    st.write(
        "**Research Objective:** To identify the relationship "
        "between annual live births and household income."
    )

    # Use available income survey years
    data = df1[
        df1["year"].between(2016, 2024)
    ].dropna(
        subset=["income_mean", "rate"]
    ).copy()

    if data.empty:

        st.warning(
            "No valid observations for Objective 1."
        )

        return

    data["Survey Year"] = (
        data["year"].astype(str)
    )

    years = sorted(
        data["Survey Year"].unique()
    )

    # Calculate Pearson correlation
    r = pearson(
        data,
        "income_mean",
        "rate"
    )

    # ======================================
    # SUMMARY BOXES
    # ======================================

    summary_cards([
        (
            "State-Year Observations",
            str(len(data))
        ),
        (
            "Average Birth Rate",
            f"{data['rate'].mean():.2f} / 1,000"
        ),
        (
            "Pearson Correlation (r)",
            f"{r:+.3f}"
        )
    ])

    # ======================================
    # VISUALIZATION 1
    # ======================================

    st.subheader(
        "Visualization 1 — Scatter Plot"
    )

    # Different colour for each survey year
    year_colours = {
        "2016": "#584F96",
        "2019": "#357EB3",
        "2020": "#329B8F",
        "2022": "#D58A47",
        "2024": "#91B849"
    }

    palette = {
        year: year_colours[year]
        for year in years
    }

    fig1, ax1 = plt.subplots(
        figsize=(10, 5.6)
    )

    sns.scatterplot(
        data=data,
        x="income_mean",
        y="rate",
        hue="Survey Year",
        hue_order=years,
        palette=palette,
        marker="o",
        s=90,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.7,
        ax=ax1
    )

    # Add regression line
    sns.regplot(
        data=data,
        x="income_mean",
        y="rate",
        scatter=False,
        ci=None,
        line_kws={
            "color": "#424A47",
            "linestyle": "--",
            "linewidth": 2
        },
        ax=ax1
    )

    # Chart title
    ax1.set_title(
        "Mean Household Income and Birth Rate (2016–2024)",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    # Axis labels
    ax1.set_xlabel(
        "Mean Monthly Household Income (RM)"
    )

    ax1.set_ylabel(
        "Birth Rate (births per 1,000 people)"
    )

    ax1.xaxis.set_major_formatter(
        StrMethodFormatter("{x:,.0f}")
    )

    # ======================================
    # LEGEND: YEAR LABELS ONLY
    # ======================================

    legend_items = []

    for year in years:

        item = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=year_colours[year],
            markeredgecolor="white",
            markersize=10,
            label=str(year)
        )

        legend_items.append(item)

    ax1.legend(
        handles=legend_items,
        title="Survey Year",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=10,
        title_fontsize=11
    )

    sns.despine(ax=ax1)

    fig1.tight_layout()

    st.pyplot(
        fig1,
        width="stretch"
    )

    plt.close(fig1)

    st.caption(
        "Each point represents a Malaysian state "
        "or federal territory in a survey year. "
        "Different colours represent different years. "
        "The dashed regression line indicates the "
        "overall linear trend."
    )

    # ======================================
    # INTERPRETATION
    # ======================================

    st.subheader(
        "Interpretation and Discussion"
    )

    st.write(
        f"The scatter plot includes **{len(data)} "
        "state-year observations** from 2016, "
        "2019, 2020, 2022 and 2024. "
        "The relationship between mean monthly "
        "household income and live birth rate "
        f"is **{relationship(r)}**, with a "
        f"Pearson correlation coefficient of "
        f"**{r:+.3f}**. The different colours "
        "represent survey years, while the "
        "regression line displays the overall "
        "linear relationship. Since several "
        "years are combined, changes over time "
        "may affect the overall correlation. "
        "The result does not establish causation."
    )

    # ======================================
    # DATA PREVIEW
    # ======================================

    data_preview(
        data.sort_values(
            ["year", "state"]
        ),
        [
            "state",
            "year",
            "income_mean",
            "rate"
        ]
    )


# ==========================================
# PAGE 2: OBJECTIVE 2
# GROUPED BAR CHART
# ==========================================

def objective2():

    st.header(
        "Objective 2 — School Completion Rates and Household Income"
    )

    st.write(
        "**Research Objective:** To examine the relationship "
        "between school completion rates and household income."
    )

    data = df2[
        df2["year"].between(2016, 2024)
    ].dropna(
        subset=[
            "income_mean",
            "completion",
            "stage"
        ]
    ).copy()

    stage_names = {
        "primary": "Primary",
        "secondary_lower": "Lower secondary",
        "secondary_upper": "Upper secondary"
    }

    stage_order = [
        "Primary",
        "Lower secondary",
        "Upper secondary"
    ]

    data["School stage"] = (
        data["stage"].map(stage_names)
    )

    data = data.dropna(
        subset=["School stage"]
    )

    if data.empty:

        st.warning(
            "No valid observations for Objective 2."
        )

        return

    # ======================================
    # SUMMARY BOXES
    # ======================================

    stage_means = data.groupby(
        "School stage"
    )["completion"].mean()

    summary_cards([
        (
            "Primary Completion",
            f"{stage_means['Primary']:.2f}%"
        ),
        (
            "Lower Secondary Completion",
            f"{stage_means['Lower secondary']:.2f}%"
        ),
        (
            "Upper Secondary Completion",
            f"{stage_means['Upper secondary']:.2f}%"
        )
    ])

    # ======================================
    # CREATE INCOME GROUPS
    # ======================================

    income_states = data[
        ["state", "year", "income_mean"]
    ].drop_duplicates(
        ["state", "year"]
    ).copy()

    income_order = [
        "Lowest",
        "Lower-middle",
        "Upper-middle",
        "Highest"
    ]

    # Create relative income quartiles
    # separately for each survey year
    income_states["Income group"] = (
        income_states.groupby("year")["income_mean"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                4,
                labels=income_order
            )
        )
    )

    # Merge income groups with school data
    data = data.merge(
        income_states[
            ["state", "year", "Income group"]
        ],
        on=["state", "year"],
        how="left",
        validate="many_to_one"
    )

    # Calculate average completion by group and stage
    grouped = data.groupby(
        ["Income group", "School stage"],
        observed=True
    )["completion"].mean().reset_index()

    # ======================================
    # VISUALIZATION 2
    # ======================================

    st.subheader(
        "Visualization 2 — Grouped Bar Chart"
    )

    fig2, ax2 = plt.subplots(
        figsize=(10.4, 5.7)
    )

    sns.barplot(
        data=grouped,
        x="Income group",
        y="completion",
        hue="School stage",
        order=income_order,
        hue_order=stage_order,
        palette=[
            "#BCDCCE",
            "#73B49C",
            "#287C62"
        ],
        errorbar=None,
        ax=ax2
    )

    # Add percentage labels above bars
    for bars in ax2.containers:

        ax2.bar_label(
            bars,
            fmt="%.1f%%",
            fontsize=9,
            padding=3
        )

    # Chart title
    ax2.set_title(
        "School Completion by Mean Income Group (2016–2024)",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    # Axis labels
    ax2.set_xlabel(
        "Relative Mean Household Income Group"
    )

    ax2.set_ylabel(
        "Average School Completion Rate (%)"
    )

    ax2.set_ylim(
        0,
        max(
            112,
            grouped["completion"].max() * 1.16
        )
    )

    # Legend
    ax2.legend(
        title="School Stage",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False
    )

    ax2.grid(
        axis="y",
        alpha=0.2
    )

    sns.despine(ax=ax2)

    fig2.tight_layout()

    st.pyplot(
        fig2,
        width="stretch"
    )

    plt.close(fig2)

    st.caption(
        "Each bar represents the average school "
        "completion rate for an income group and "
        "school stage. Income groups are relative "
        "state quartiles within each year, not "
        "official B40, M40 or T20 categories."
    )

    # ======================================
    # INTERPRETATION
    # ======================================

    comparison = grouped.pivot(
        index="Income group",
        columns="School stage",
        values="completion"
    )

    upper_difference = (
        comparison.loc[
            "Highest",
            "Upper secondary"
        ]
        -
        comparison.loc[
            "Lowest",
            "Upper secondary"
        ]
    )

    st.subheader(
        "Interpretation and Discussion"
    )

    st.write(
        "The grouped bar chart compares school "
        "completion rates across three educational "
        "stages and four relative household-income "
        "groups. The study covers 2016, 2019, 2020, "
        "2022 and 2024. For upper-secondary education, "
        "the difference between the highest and "
        "lowest income groups is "
        f"**{upper_difference:+.2f} percentage points**. "
        "The patterns differ across educational "
        "stages. These group comparisons show "
        "associations rather than causal effects."
    )

    # ======================================
    # DATA PREVIEW
    # ======================================

    data_preview(
        data.sort_values(
            ["year", "state", "School stage"]
        ),
        [
            "state",
            "year",
            "income_mean",
            "School stage",
            "completion",
            "Income group"
        ]
    )


# ==========================================
# PAGE 3: OBJECTIVE 3
# INTERACTIVE LINE CHART
# ==========================================

def objective3():

    st.header(
        "Objective 3 — Annual Live Births and School Completion Rates"
    )

    st.write(
        "**Research Objective:** To analyse the relationship "
        "between annual live births and school completion rates."
    )

    data = df3[
        df3["year"].between(2016, 2025)
    ].dropna(
        subset=[
            "rate",
            "completion",
            "stage"
        ]
    ).copy()

    if data.empty:

        st.warning(
            "No valid observations for Objective 3."
        )

        return

    # Birth rates repeat for each school stage
    # Aggregate once per state-year
    state_year = data.groupby(
        ["state", "year"],
        as_index=False
    ).agg(
        birth_rate=("rate", "first"),
        completion_rate=("completion", "mean")
    )

    # Calculate yearly averages
    trend = state_year.groupby(
        "year",
        as_index=False
    ).agg(
        birth_rate=("birth_rate", "mean"),
        completion_rate=("completion_rate", "mean")
    ).sort_values("year")

    # ======================================
    # NORMALIZE VALUES: 2016 = 100
    # ======================================

    birth_base = float(
        trend.loc[
            trend["year"] == 2016,
            "birth_rate"
        ].iloc[0]
    )

    completion_base = float(
        trend.loc[
            trend["year"] == 2016,
            "completion_rate"
        ].iloc[0]
    )

    trend["birth_index"] = (
        trend["birth_rate"] / birth_base * 100
    )

    trend["completion_index"] = (
        trend["completion_rate"] / completion_base * 100
    )

    latest = trend.iloc[-1]

    latest_states = state_year[
        state_year["year"] == int(latest["year"])
    ]

    r3 = pearson(
        latest_states,
        "birth_rate",
        "completion_rate"
    )

    # ======================================
    # SUMMARY BOXES
    # ======================================

    summary_cards([
        (
            "2025 Average Birth Rate",
            f"{latest['birth_rate']:.2f} / 1,000"
        ),
        (
            "2025 Average Completion",
            f"{latest['completion_rate']:.2f}%"
        ),
        (
            "2025 Pearson Correlation (r)",
            f"{r3:+.3f}"
        )
    ])

    # ======================================
    # VISUALIZATION 3
    # ======================================

    st.subheader(
        "Visualization 3 — Interactive Line Chart"
    )

    source = ColumnDataSource(trend)

    minimum = min(
        trend["birth_index"].min(),
        trend["completion_index"].min(),
        100
    )

    maximum = max(
        trend["birth_index"].max(),
        trend["completion_index"].max(),
        100
    )

    chart = figure(
        title=(
            "Birth Rate and School Completion Trends (2016–2025)"
        ),
        height=430,
        sizing_mode="stretch_width",
        x_range=(2015.5, 2025.7),
        y_range=(minimum - 7, maximum + 7),
        x_axis_label="Year",
        y_axis_label="Relative Index (2016 = 100)",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    # Line 1: Birth rate
    chart.line(
        "year",
        "birth_index",
        source=source,
        color=GREEN,
        line_width=3,
        legend_label="Birth Rate"
    )

    birth_points = chart.scatter(
        "year",
        "birth_index",
        source=source,
        color=GREEN,
        size=10
    )

    # Line 2: School completion
    chart.line(
        "year",
        "completion_index",
        source=source,
        color=ORANGE,
        line_width=3,
        legend_label="School Completion"
    )

    completion_points = chart.scatter(
        "year",
        "completion_index",
        source=source,
        color=ORANGE,
        size=10
    )

    # ======================================
    # INTERACTIVE HOVER LABELS
    # ======================================

    chart.add_tools(
        HoverTool(
            renderers=[birth_points],
            tooltips=[
                ("Year", "@year"),
                (
                    "Birth Rate",
                    "@birth_rate{0.00} per 1,000"
                ),
                (
                    "Index",
                    "@birth_index{0.0}"
                )
            ]
        )
    )

    chart.add_tools(
        HoverTool(
            renderers=[completion_points],
            tooltips=[
                ("Year", "@year"),
                (
                    "School Completion",
                    "@completion_rate{0.00}%"
                ),
                (
                    "Index",
                    "@completion_index{0.0}"
                )
            ]
        )
    )

    # Reference line
    chart.add_layout(
        Span(
            location=100,
            dimension="width",
            line_color="#81918A",
            line_dash="dashed"
        )
    )

    chart.xaxis.ticker = list(
        range(2016, 2026)
    )

    chart.ygrid.grid_line_alpha = 0.2

    chart.legend.location = "bottom_left"

    chart.legend.click_policy = "hide"

    chart.toolbar.logo = None

    # Display Bokeh chart
    chart_html = file_html(
        chart,
        CDN,
        "Objective 3 Line Chart"
    )

    components.html(
        chart_html,
        height=480,
        scrolling=False
    )

    st.caption(
        "The green line represents birth rates, "
        "while the orange line represents school "
        "completion. Both use 2016 = 100 to compare "
        "relative changes. Hover over the markers "
        "to view the original values."
    )

    # ======================================
    # INTERPRETATION
    # ======================================

    birth_change = (
        latest["birth_index"] - 100
    )

    completion_change = (
        latest["completion_rate"] - completion_base
    )

    st.subheader(
        "Interpretation and Discussion"
    )

    st.write(
        f"From 2016 to 2025, the average state "
        "birth rate changed by "
        f"**{birth_change:+.1f}%**, while the average "
        "school completion rate changed by "
        f"**{completion_change:+.2f} percentage points**. "
        "The line chart illustrates the relative "
        "changes in both indicators over time. "
        "The state-level Pearson correlation "
        f"in 2025 was **{r3:+.3f}**, indicating a "
        f"**{relationship(r3)}** relationship. "
        "The observed trends do not establish causation."
    )

    # ======================================
    # DATA PREVIEW
    # ======================================

    with st.expander(
        "View Objective 3 Yearly Data"
    ):

        st.dataframe(
            trend.round(2),
            hide_index=True,
            width="stretch"
        )


# ==========================================
# 4. DASHBOARD HEADER
# ==========================================

st.markdown("""
<div style="
    background:#184B3C;
    color:white;
    padding:24px 28px;
    border-radius:15px;
    margin-bottom:12px;
">
    <div style="
        font-size:12px;
        font-weight:700;
        letter-spacing:1.2px;
        color:#DDECE4;
    ">
        JIE42403 · SCIENTIFIC VISUALISATION
    </div>

    <div style="
        font-size:32px;
        font-weight:700;
        margin:5px 0;
    ">
        Malaysia Data Insights
    </div>

    <div style="color:#E4F2EB">
        Three Research Objectives · Three Separate Pages
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. MULTIPAGE NAVIGATION
# ==========================================

pages = st.navigation(
    [
        st.Page(
            objective1,
            title="1 · Births & Income",
            icon="📊",
            default=True
        ),

        st.Page(
            objective2,
            title="2 · Education & Income",
            icon="🎓"
        ),

        st.Page(
            objective3,
            title="3 · Births & Education",
            icon="📈"
        )
    ],
    position="top"
)

pages.run()


# ==========================================
# 6. DATA SOURCES AND LIMITATIONS
# ==========================================

st.divider()

with st.expander(
    "Data Sources and Methodology"
):

    st.write(
        "**Data Sources:** OpenDOSM state-level "
        "annual live births, household income and "
        "school completion datasets."
    )

    st.write(
        "**Methodology:** Objective 1 examines "
        "mean household income and live birth rates "
        "between 2016 and 2024. Objective 2 compares "
        "school completion across income groups "
        "and three educational stages. Objective 3 "
        "compares annual birth rates and school "
        "completion trends from 2016 to 2025."
    )

    st.write(
        "**Limitations:** Household income is reported "
        "in nominal RM. School completion rates may "
        "exceed 100% under the official indicator "
        "definition. The analyses use unweighted "
        "state averages, and correlation does not "
        "demonstrate causation."
    )

    st.markdown(
        "[OpenDOSM Live Births]"
        "(https://open.dosm.gov.my/data-catalogue/births_annual_state)"
        " · "
        "[OpenDOSM Household Income]"
        "(https://open.dosm.gov.my/data-catalogue/hh_income_state)"
        " · "
        "[OpenDOSM School Completion]"
        "(https://open.dosm.gov.my/data-catalogue/completion_school_state)"
    )

st.caption(
    "JIE42403 Scientific Visualisation | "
    "Malaysia Data Insights | OpenDOSM"
)
