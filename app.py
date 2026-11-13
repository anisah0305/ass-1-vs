
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


# ==========================================================
# 1. APP SETTINGS AND DESIGN
# ==========================================================

st.set_page_config(
    page_title="Malaysia Data Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN = "#287C62"
DARK_GREEN = "#184B3C"
ORANGE = "#D38A49"

sns.set_theme(style="whitegrid", font_scale=1.0)

st.markdown("""
<style>
.block-container {
    max-width: 1120px;
    padding-top: 1.6rem;
}

h1, h2, h3 {
    color: #184B3C;
}

div[data-testid="stMetric"] {
    background: #F4F9F6;
}

section[data-testid="stSidebar"] {
    background: #F2F7F4;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 2. LOAD YOUR EXISTING GITHUB CSV FILES
# ==========================================================

@st.cache_data
def load_data():

    folder = Path(__file__).resolve().parent

    filenames = {
        "obj1": "df_merged_income_birth (1).csv",
        "obj2": "df_merged_income_rates.csv",
        "obj3": "df_merged_birth_rates_2016_2025.csv",
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
        ).copy()

        df = df[
            df["state"].str.lower() != "malaysia"
        ].copy()

        df["year"] = df["year"].astype(int)

        for column in (
            "income_mean",
            "abs",
            "rate",
            "completion"
        ):

            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        if "sex" in df.columns:

            df["sex"] = (
                df["sex"]
                .astype(str)
                .str.strip()
                .str.lower()
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
        f"Could not load the datasets: {error}"
    )

    st.info(
        "Check that app.py and all three merged "
        "CSV files are in the same GitHub folder."
    )

    st.stop()


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

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

    direction = (
        "positive"
        if r > 0
        else "negative"
        if r < 0
        else "zero"
    )

    return f"{strength} {direction}"


def summary_cards(items):

    columns = st.columns(3)

    for column, (label, value) in zip(
        columns,
        items
    ):

        with column:

            st.metric(
                label,
                value,
                border=True
            )


def data_preview(data, columns):

    with st.expander(
        "View Data Used in This Visualisation"
    ):

        st.dataframe(
            data[columns],
            hide_index=True,
            width="stretch"
        )


# ==========================================================
# OBJECTIVE 1:
# ANNUAL LIVE BIRTHS VS HOUSEHOLD INCOME
# ==========================================================

def objective1():

    st.header(
        "Objective 1 — Annual Live Births and Household Income"
    )

    st.write(
        "**Research Objective:** To identify the "
        "relationship between annual live births "
        "and household income."
    )

    data = df1[
        df1["year"].between(2016, 2024)
    ].dropna(
        subset=["income_mean", "abs"]
    ).copy()

    if data.empty:

        st.warning(
            "No valid data is available for Objective 1."
        )
        return

    data["Survey Year"] = (
        data["year"].astype(str)
    )

    years = sorted(
        data["Survey Year"].unique()
    )

    r = pearson(
        data,
        "income_mean",
        "abs"
    )

    years_text = (
        ", ".join(years[:-1]) + " and " + years[-1]
        if len(years) > 1
        else years[0]
    )

    # Summary boxes
    summary_cards([
        (
            "Area-Year Observations",
            f"{len(data):,}"
        ),
        (
            "Average Births per Area-Year",
            f"{data['abs'].mean():,.2f}"
        ),
        (
            "Pearson Correlation (r)",
            f"{r:+.3f}"
        ),
    ])

    # Scatter plot
    st.subheader(
        "Visualisation 1 — Scatter Plot"
    )

    year_colours = {
        "2016": "#584F96",
        "2019": "#357EB3",
        "2020": "#329B8F",
        "2022": "#D58A47",
        "2024": "#91B849",
    }

    palette = {
        year: year_colours.get(year, GREEN)
        for year in years
    }

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    sns.scatterplot(
        data=data,
        x="income_mean",
        y="abs",
        hue="Survey Year",
        hue_order=years,
        palette=palette,
        s=100,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )

    # Regression line
    sns.regplot(
        data=data,
        x="income_mean",
        y="abs",
        scatter=False,
        ci=None,
        line_kws={
            "color": "#59665F",
            "linestyle": "--",
            "linewidth": 1.9,
        },
        ax=ax,
    )

    ax.set_title(
        "Household Income vs. Annual Live Births (2016–2024)",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel(
        "Mean Monthly Household Income (RM)"
    )

    ax.set_ylabel(
        "Annual Live Births (Number of Births)"
    )

    ax.xaxis.set_major_formatter(
        StrMethodFormatter("{x:,.0f}")
    )

    ax.yaxis.set_major_formatter(
        StrMethodFormatter("{x:,.0f}")
    )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=9,
            markerfacecolor=palette[year],
            markeredgecolor="white",
            label=year
        )
        for year in years
    ]

    ax.legend(
        handles=legend_handles,
        title="Survey Year",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.grid(alpha=0.23)

    sns.despine(ax=ax)

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

    st.caption(
        "Each dot represents one state or Federal "
        "Territory in a survey year. Colours distinguish "
        "survey years; the dashed line shows the "
        "pooled linear trend."
    )

    # Interpretation
    st.subheader(
        "Interpretation and Discussion"
    )

    st.write(
        f"The scatter plot compared **{len(data)} "
        "area-year observations** from "
        f"{years_text}. The Pearson correlation "
        f"was **{r:+.3f}**, indicating a "
        f"**{relationship(r)}** linear relationship "
        "between mean monthly household income "
        "and the number of live births. The wide "
        "spread of points showed that areas with "
        "similar incomes could have very different "
        "birth totals. Population size may partly "
        "explain these differences, so the result "
        "should not be interpreted as a causal "
        "effect of income on births."
    )

    data_preview(
        data.sort_values(
            ["year", "state"]
        ),
        [
            "state",
            "year",
            "income_mean",
            "abs"
        ]
    )


# ==========================================================
# OBJECTIVE 2:
# SCHOOL COMPLETION VS HOUSEHOLD INCOME
# ==========================================================

def objective2():

    st.header(
        "Objective 2 — School Completion Rates and Household Income"
    )

    st.write(
        "**Research Objective:** To examine the "
        "relationship between school completion "
        "rates and household income."
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
        "secondary_upper": "Upper secondary",
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
            "No valid data is available for Objective 2."
        )
        return

    # Summary boxes
    means = data.groupby(
        "School stage"
    )["completion"].mean()

    summary_cards([
        (
            "Primary Completion",
            f"{means['Primary']:.2f}%"
        ),
        (
            "Lower Secondary Completion",
            f"{means['Lower secondary']:.2f}%"
        ),
        (
            "Upper Secondary Completion",
            f"{means['Upper secondary']:.2f}%"
        ),
    ])

    # Create income groups
    income_order = [
        "Lowest",
        "Lower-middle",
        "Upper-middle",
        "Highest"
    ]

    incomes = data[
        ["state", "year", "income_mean"]
    ].drop_duplicates(
        ["state", "year"]
    ).copy()

    incomes["Income group"] = (
        incomes.groupby("year")["income_mean"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                4,
                labels=income_order
            )
        )
    )

    data = data.merge(
        incomes[
            ["state", "year", "Income group"]
        ],
        on=["state", "year"],
        how="left",
        validate="many_to_one",
    )

    grouped = data.groupby(
        ["Income group", "School stage"],
        observed=True
    )["completion"].mean().reset_index()

    # Grouped bar chart
    st.subheader(
        "Visualisation 2 — Grouped Bar Chart"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5.7)
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
        ax=ax,
    )

    # Percentage labels
    for bars in ax.containers:

        ax.bar_label(
            bars,
            fmt="%.1f%%",
            fontsize=9,
            padding=3
        )

    ax.set_title(
        "School Completion by Relative Income Group (2016–2024)",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel(
        "Relative State Income Group (Quartile Within Each Year)"
    )

    ax.set_ylabel(
        "Mean School Completion Rate (%)"
    )

    ax.set_ylim(
        0,
        max(
            112,
            grouped["completion"].max() * 1.16
        )
    )

    ax.legend(
        title="School Stage",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    sns.despine(ax=ax)

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

    st.caption(
        "The four income categories rank states "
        "separately within each survey year. "
        "They are NOT official household B40, "
        "M40 or T20 classifications. Bars show "
        "averages across the five income-survey years."
    )

    # Interpretation
    comparison = grouped.pivot(
        index="Income group",
        columns="School stage",
        values="completion"
    )

    gap = (
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
        "The grouped bar chart compared completion "
        "rates for **three education stages** across "
        "four relative state income groups. The mean "
        "completion rates were generally high, "
        "although differences varied between stages. "
        "For upper-secondary schooling, the "
        "highest-income group differed from the "
        "lowest-income group by "
        f"**{gap:+.2f} percentage points**. "
        "The groupings describe state-level patterns "
        "rather than outcomes for individual households. "
        "They do not establish a causal relationship."
    )

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


# ==========================================================
# OBJECTIVE 3:
# ANNUAL LIVE BIRTHS VS SCHOOL COMPLETION
# ==========================================================

def objective3():

    st.header(
        "Objective 3 — Annual Live Births and School Completion Rates"
    )

    st.write(
        "**Research Objective:** To analyse the "
        "relationship between annual live births "
        "and school completion rates."
    )

    data = df3[
        df3["year"].between(2016, 2025)
    ].dropna(
        subset=[
            "abs",
            "completion",
            "stage"
        ]
    ).copy()

    if data.empty:

        st.warning(
            "No valid data is available for Objective 3."
        )
        return

    # Birth values repeat for each school stage.
    # Count each state's births only once per year.

    by_state_year = data.groupby(
        ["state", "year"],
        as_index=False
    ).agg(
        annual_births=("abs", "first"),
        mean_completion=("completion", "mean"),
    )

    # Calculate annual totals and mean completion.
    trend = by_state_year.groupby(
        "year",
        as_index=False
    ).agg(
        total_births=("annual_births", "sum"),
        mean_completion=("mean_completion", "mean"),
    ).sort_values("year")

    if 2016 not in trend["year"].values:

        st.warning(
            "The 2016 baseline is missing from Objective 3."
        )
        return

    # Use 2016 as the reference year
    base_births = float(
        trend.loc[
            trend["year"] == 2016,
            "total_births"
        ].iloc[0]
    )

    base_completion = float(
        trend.loc[
            trend["year"] == 2016,
            "mean_completion"
        ].iloc[0]
    )

    if base_births == 0 or base_completion == 0:

        st.warning(
            "The 2016 index cannot be calculated "
            "because a baseline is zero."
        )
        return

    trend["birth_index"] = (
        trend["total_births"]
        / base_births
        * 100
    )

    trend["completion_index"] = (
        trend["mean_completion"]
        / base_completion
        * 100
    )

    latest = trend.iloc[-1]

    # Correlation across ten annual observations
    r_yearly = pearson(
        trend,
        "total_births",
        "mean_completion"
    )

    # Summary boxes
    summary_cards([
        (
            "2025 Total Live Births",
            f"{latest['total_births']:,.0f}"
        ),
        (
            "2025 Mean School Completion",
            f"{latest['mean_completion']:.2f}%"
        ),
        (
            "2016–2025 Annual Pearson (r)",
            f"{r_yearly:+.3f}"
        ),
    ])

    # Interactive line chart
    st.subheader(
        "Visualisation 3 — Interactive Line Chart"
    )

    source = ColumnDataSource(trend)

    low = min(
        trend["birth_index"].min(),
        trend["completion_index"].min(),
        100
    )

    high = max(
        trend["birth_index"].max(),
        trend["completion_index"].max(),
        100
    )

    chart = figure(
        title=(
            "Annual Live Births and School Completion "
            "Trends (2016–2025)"
        ),
        height=430,
        sizing_mode="stretch_width",
        x_range=(2015.5, 2025.7),
        y_range=(low - 7, high + 7),
        x_axis_label="Year",
        y_axis_label="Relative Index (2016 = 100)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Live births line
    chart.line(
        "year",
        "birth_index",
        source=source,
        color=GREEN,
        line_width=3,
        legend_label="Annual Live Births",
    )

    birth_points = chart.scatter(
        "year",
        "birth_index",
        source=source,
        color=GREEN,
        size=10,
    )

    # School completion line
    chart.line(
        "year",
        "completion_index",
        source=source,
        color=ORANGE,
        line_width=3,
        legend_label="School Completion",
    )

    completion_points = chart.scatter(
        "year",
        "completion_index",
        source=source,
        color=ORANGE,
        size=10,
    )

    # Hover labels for live births
    chart.add_tools(
        HoverTool(
            renderers=[birth_points],
            tooltips=[
                ("Year", "@year"),
                (
                    "Total Live Births",
                    "@total_births{0,0}"
                ),
                (
                    "Birth Index",
                    "@birth_index{0.0}"
                ),
            ],
        )
    )

    # Hover labels for school completion
    chart.add_tools(
        HoverTool(
            renderers=[completion_points],
            tooltips=[
                ("Year", "@year"),
                (
                    "Average Completion",
                    "@mean_completion{0.00}%"
                ),
                (
                    "Completion Index",
                    "@completion_index{0.0}"
                ),
            ],
        )
    )

    # Reference line at index 100
    chart.add_layout(
        Span(
            location=100,
            dimension="width",
            line_color="#81918A",
            line_dash="dashed",
        )
    )

    chart.xaxis.ticker = list(
        range(2016, 2026)
    )

    chart.ygrid.grid_line_alpha = 0.20

    chart.legend.location = "bottom_left"

    chart.legend.click_policy = "hide"

    chart.toolbar.logo = None

    # Display interactive Bokeh chart
    components.html(
        file_html(
            chart,
            CDN,
            "Annual Births and School Completion"
        ),
        height=490,
        scrolling=False,
    )

    st.caption(
        "The green line shows total annual live "
        "births across the 16 states and Federal "
        "Territories; the orange line shows the "
        "unweighted mean school completion rate. "
        "Each series is indexed to 100 in 2016 "
        "so its relative change can be compared. "
        "Hover over a point for the original value."
    )

    # Interpretation
    birth_change = (
        latest["total_births"]
        / base_births
        - 1
    ) * 100

    completion_change = (
        latest["mean_completion"]
        - base_completion
    )

    st.subheader(
        "Interpretation and Discussion"
    )

    st.write(
        "Between 2016 and 2025, annual live births "
        f"changed by **{birth_change:+.1f}%**, "
        "while mean school completion changed by "
        f"**{completion_change:+.2f} percentage points**. "
        "The Pearson correlation across the "
        "**10 annual totals and mean completion values** "
        f"was **{r_yearly:+.3f}** "
        f"({relationship(r_yearly)}). "
        "The two series use different units, "
        "so the line chart shows their relative "
        "change rather than comparing their raw "
        "magnitudes. This descriptive correlation "
        "does not account for time trends, "
        "population changes or other factors "
        "and must not be interpreted as "
        "evidence of causation."
    )

    with st.expander(
        "View Objective 3 Annual Data"
    ):

        st.dataframe(
            trend.round(2),
            hide_index=True,
            width="stretch"
        )


# ==========================================================
# 4. DASHBOARD HEADER
# ==========================================================

st.html("""
<div style="
    background:#184B3C;
    color:white;
    padding:20px 24px;
    border-radius:12px;
">
    <div style="
        font-size:13px;
        font-weight:700;
        color:#DDECE4;
    ">
        JIE42403 · SCIENTIFIC VISUALISATION
    </div>

    <div style="
        font-size:30px;
        font-weight:700;
        color:white;
        margin-top:8px;
    ">
        Malaysia Data Insights
    </div>
</div>
""")


# ==========================================================
# 5. LEFT SIDEBAR NAVIGATION
# ==========================================================

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
        ),
    ],
    position="sidebar",
)

pages.run()


# ==========================================================
# 6. DATA SOURCES AND LIMITATIONS
# ==========================================================

st.divider()

with st.expander(
    "Data Sources and Methodology"
):

    st.write(
        "**Data Sources:** OpenDOSM (Department "
        "of Statistics Malaysia) annual live "
        "births by state, household income by "
        "state and school completion by state."
    )

    st.write(
        "**Methods:** Objective 1 uses 80 "
        "area-year observations from the five "
        "income survey years (2016, 2019, "
        "2020, 2022 and 2024). Objective 2 "
        "compares three school stages across "
        "relative state-income quartiles. "
        "Objective 3 compares total annual "
        "live births and mean school completion "
        "using a 2016-based index for 2016–2025."
    )

    st.write(
        "**Limitations:** Household income is "
        "in nominal RM. The analyses are at "
        "state/Federal Territory level, not "
        "at household or individual level. "
        "Completion percentages use unweighted "
        "state averages and may exceed 100% "
        "under the official indicator definition. "
        "Correlation is not causation."
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
