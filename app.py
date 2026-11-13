
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------
# 1. PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="Malaysia Data Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN = "#287C62"
DARK = "#184B3C"
SURVEY_YEARS = [2016, 2019, 2020, 2022, 2024]

st.markdown(
    """
    <style>
      .block-container {max-width: 1150px; padding-top: 1.5rem;}
      h1, h2, h3 {color: #184B3C;}
      div[data-testid="stMetric"] {
        background: #F1F8F4;
        border: 1px solid #DDECE4;
        border-radius: 12px;
        padding: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 2. LOAD THREE CSV FILES
# -------------------------------------------------
@st.cache_data
def load_datasets():
    root = Path(__file__).resolve().parent

    files = {
        "income_birth": "df_merged_income_birth (1).csv",
        "income_school": "df_merged_income_rates.csv",
        "birth_school": "df_merged_birth_rates_2016_2025.csv",
    }

    frames = {}

    for key, name in files.items():
        path = root / name

        if not path.exists():
            raise FileNotFoundError(f"Missing CSV file: {name}")

        df = pd.read_csv(path)

        df["year"] = pd.to_datetime(
            df["date"], errors="coerce"
        ).dt.year

        df["state"] = df["state"].astype(str).str.strip()

        df = df.dropna(subset=["year"])

        df = df[
            df["state"].str.lower() != "malaysia"
        ].copy()

        df["year"] = df["year"].astype(int)

        for col in ["income_mean", "abs", "completion"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                )

        if "sex" in df.columns:
            df = df[
                df["sex"].astype(str).str.lower().eq("both")
            ]

        frames[key] = df

    return (
        frames["income_birth"],
        frames["income_school"],
        frames["birth_school"]
    )


try:
    df1, df2, df3 = load_datasets()

except (
    FileNotFoundError,
    KeyError,
    ValueError,
    pd.errors.ParserError
) as error:

    st.error(f"Unable to load the datasets: {error}")

    st.info(
        "Upload all three merged CSV files to "
        "the same GitHub folder as app.py."
    )

    st.stop()


# -------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------
def correlation(df, x, y):

    clean = df[[x, y]].dropna()

    if (
        len(clean) < 3
        or clean[x].nunique() < 2
        or clean[y].nunique() < 2
    ):
        return float("nan")

    return clean[x].corr(clean[y])


def strength(r):

    if pd.isna(r):
        return "undefined"

    if abs(r) < 0.10:
        level = "negligible"

    elif abs(r) < 0.30:
        level = "weak"

    elif abs(r) < 0.70:
        level = "moderate"

    else:
        level = "strong"

    direction = (
        "positive" if r > 0
        else "negative" if r < 0
        else "zero"
    )

    return f"{level} {direction}"


def metrics(items):

    columns = st.columns(3)

    for col, (label, value) in zip(columns, items):
        col.metric(label, value)


# -------------------------------------------------
# OBJECTIVE 1
# ANNUAL LIVE BIRTHS AND HOUSEHOLD INCOME
# -------------------------------------------------
def objective1():

    st.header(
        "Objective 1: Annual Live Births and Household Income"
    )

    st.write(
        "**Research Objective:** To identify the relationship "
        "between annual live births and household income."
    )

    data = df1[
        df1["year"].isin(SURVEY_YEARS)
    ].dropna(
        subset=["state", "income_mean", "abs"]
    ).copy()

    if data.empty:
        st.warning("No valid data for Objective 1.")
        return

    # YEAR SELECTION
    chosen = st.selectbox(
        "Select Survey Year",
        ["All Years"] + SURVEY_YEARS,
        key="obj1_year",
    )

    if chosen == "All Years":
        current = data.copy()

    else:
        current = data[
            data["year"] == chosen
        ].copy()

    r = correlation(
        current, "income_mean", "abs"
    )

    # SUMMARY BOXES
    metrics([
        (
            "Area-Year Observations",
            f"{len(current):,}"
        ),
        (
            "Average Births per Area-Year",
            f"{current['abs'].mean():,.2f}"
        ),
        (
            "Pearson Correlation (r)",
            f"{r:+.3f}"
        ),
    ])

    # INTERACTIVE SCATTER PLOT
    st.subheader(
        "Interactive Scatter Plot"
    )

    colors = {
        2016: "#6654A4",
        2019: "#377CA8",
        2020: "#299A8B",
        2022: "#D58A47",
        2024: "#84AD50",
    }

    fig = go.Figure()

    for year in sorted(
        current["year"].unique()
    ):

        d = current[
            current["year"] == year
        ]

        custom = np.column_stack([
            d["state"],
            d["year"].astype(str)
        ])

        fig.add_trace(
            go.Scatter(
                x=d["income_mean"],
                y=d["abs"],
                customdata=custom,
                mode="markers",
                name=str(year),

                marker=dict(
                    size=12,
                    color=colors[year],
                    opacity=0.85,
                    line=dict(
                        color="white",
                        width=1
                    )
                ),

                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Year: %{customdata[1]}"
                    "<br>Mean Monthly Household Income: "
                    "RM%{x:,.0f}"
                    "<br>Annual Live Births: %{y:,.0f}"
                    "<extra></extra>"
                )
            )
        )

    # REGRESSION LINE
    if current["income_mean"].nunique() >= 2:

        slope, intercept = np.polyfit(
            current["income_mean"],
            current["abs"],
            1
        )

        x_line = np.linspace(
            data["income_mean"].min(),
            data["income_mean"].max(),
            100
        )

        y_line = slope * x_line + intercept

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Regression Line",
                line=dict(
                    color="#253449",
                    width=2,
                    dash="dash"
                ),
                hoverinfo="skip"
            )
        )

    # GRAPH DESIGN
    fig.update_layout(
        title=(
            "Household Income vs. Annual Live Births "
            f"— {chosen}"
        ),

        xaxis_title=(
            "Mean Monthly Household Income (RM)"
        ),

        yaxis_title=(
            "Annual Live Births (Number of Births)"
        ),

        legend_title="Survey Year",
        template="plotly_white",
        height=560,
        hovermode="closest",

        margin=dict(
            l=35,
            r=25,
            t=70,
            b=35
        )
    )

    fig.update_xaxes(
        range=[
            data["income_mean"].min() - 700,
            data["income_mean"].max() + 700
        ],
        tickprefix="RM",
        tickformat=","
    )

    fig.update_yaxes(
        range=[
            0,
            data["abs"].max() * 1.10
        ],
        tickformat=","
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Each point represents a Malaysian state or "
        "Federal Territory in one survey year. "
        "Hover over the points to identify the state, "
        "year, household income and annual live births. "
        "The dashed line represents the linear trend "
        "for the selected observations."
    )

    # KEY FINDINGS
    st.subheader("Key Findings")

    high = current.loc[
        current["abs"].idxmax()
    ]

    low = current.loc[
        current["abs"].idxmin()
    ]

    left, right = st.columns(2)

    with left:

        st.metric(
            (
                f"Highest Births: {high['state']} "
                f"({int(high['year'])})"
            ),
            f"{high['abs']:,.0f}"
        )

    with right:

        st.metric(
            (
                f"Lowest Births: {low['state']} "
                f"({int(low['year'])})"
            ),
            f"{low['abs']:,.0f}"
        )

    # INTERPRETATION
    st.subheader(
        "Interpretation and Discussion"
    )

    st.write(
        f"The selected dataset contained "
        f"**{len(current)} area-year observations**. "
        f"The Pearson correlation coefficient of "
        f"**{r:+.3f}** indicated a "
        f"**{strength(r)} linear association** "
        f"between household income and annual live births. "
        f"States with similar household income levels "
        f"could record considerably different birth totals. "
        f"The highest number of births was recorded in "
        f"**{high['state']} ({int(high['year'])})**, "
        f"while the lowest was recorded in "
        f"**{low['state']} ({int(low['year'])})**. "
        f"Differences in population size and age structure "
        f"may have contributed to these variations. "
        f"These results describe an association rather "
        f"than a causal relationship."
    )

    # DATA TABLE
    with st.expander(
        "View Objective 1 Data"
    ):

        display = current[
            [
                "state",
                "year",
                "income_mean",
                "abs"
            ]
        ].rename(
            columns={
                "state": "State",
                "year": "Year",
                "income_mean": (
                    "Mean Monthly Income (RM)"
                ),
                "abs": "Annual Live Births"
            }
        )

        st.dataframe(
            display.sort_values(
                ["Year", "State"]
            ),
            hide_index=True,
            use_container_width=True
        )


# -------------------------------------------------
# OBJECTIVE 2
# SCHOOL COMPLETION RATES AND HOUSEHOLD INCOME
# -------------------------------------------------
def objective2():

    st.header(
        "Objective 2: School Completion Rates "
        "and Household Income"
    )

    st.write(
        "**Research Objective:** To examine the relationship "
        "between school completion rates and household income."
    )

    data = df2[
        df2["year"].isin(SURVEY_YEARS)
    ].dropna(
        subset=[
            "state",
            "income_mean",
            "completion",
            "stage"
        ]
    ).copy()

    stage_names = {
        "primary": "Primary",
        "secondary_lower": "Lower Secondary",
        "secondary_upper": "Upper Secondary"
    }

    stages = [
        "Primary",
        "Lower Secondary",
        "Upper Secondary"
    ]

    data["School Stage"] = (
        data["stage"].map(stage_names)
    )

    data = data.dropna(
        subset=["School Stage"]
    )

    if data.empty:
        st.warning("No valid data for Objective 2.")
        return

    # SUMMARY BOXES
    means = data.groupby(
        "School Stage"
    )["completion"].mean()

    metrics([
        (
            "Primary Completion",
            f"{means['Primary']:.2f}%"
        ),
        (
            "Lower Secondary Completion",
            f"{means['Lower Secondary']:.2f}%"
        ),
        (
            "Upper Secondary Completion",
            f"{means['Upper Secondary']:.2f}%"
        ),
    ])

    # CREATE RELATIVE INCOME GROUPS
    unique_income = data[
        ["state", "year", "income_mean"]
    ].drop_duplicates(
        ["state", "year"]
    ).copy()

    order = [
        "Lowest",
        "Lower-middle",
        "Upper-middle",
        "Highest"
    ]

    unique_income["Income Group"] = (
        unique_income.groupby(
            "year"
        )["income_mean"].transform(
            lambda x: pd.qcut(
                x.rank(method="first"),
                4,
                labels=order
            )
        )
    )

    data = data.merge(
        unique_income[
            [
                "state",
                "year",
                "Income Group"
            ]
        ],
        on=["state", "year"],
        how="left",
        validate="many_to_one"
    )

    grouped = data.groupby(
        [
            "Income Group",
            "School Stage"
        ],
        observed=True
    )["completion"].mean().reset_index()

    grouped["Label"] = grouped[
        "completion"
    ].map(
        lambda x: f"{x:.1f}%"
    )

    # GROUPED BAR CHART
    st.subheader(
        "Grouped Bar Chart: "
        "School Completion by Income Group"
    )

    bars = alt.Chart(
        grouped
    ).mark_bar().encode(

        x=alt.X(
            "Income Group:N",
            sort=order,
            title="Relative Household Income Group"
        ),

        xOffset=alt.XOffset(
            "School Stage:N",
            sort=stages
        ),

        y=alt.Y(
            "completion:Q",
            title="Mean School Completion Rate (%)",
            scale=alt.Scale(
                domain=[0, 110]
            )
        ),

        color=alt.Color(
            "School Stage:N",
            title="School Stage",
            scale=alt.Scale(
                domain=stages,
                range=[
                    "#BCDCCE",
                    "#73B49C",
                    GREEN
                ]
            )
        ),

        tooltip=[
            alt.Tooltip("Income Group:N"),
            alt.Tooltip("School Stage:N"),
            alt.Tooltip(
                "completion:Q",
                format=".2f",
                title="Completion (%)"
            )
        ]
    )

    labels = alt.Chart(
        grouped
    ).mark_text(
        dy=-7,
        fontSize=10
    ).encode(

        x=alt.X(
            "Income Group:N",
            sort=order
        ),

        xOffset=alt.XOffset(
            "School Stage:N",
            sort=stages
        ),

        y="completion:Q",
        text="Label:N"
    )

    st.altair_chart(
        (bars + labels).properties(
            height=400
        ),
        use_container_width=True
    )

    st.caption(
        "Income groups represent relative household "
        "income quartiles among Malaysian states "
        "within each selected year. "
        "They are not official B40, M40 or T20 "
        "household income classifications."
    )

    # INTERPRETATION
    comparison = grouped.pivot(
        index="Income Group",
        columns="School Stage",
        values="completion"
    )

    difference = (
        comparison.loc[
            "Highest",
            "Upper Secondary"
        ]
        -
        comparison.loc[
            "Lowest",
            "Upper Secondary"
        ]
    )

    st.subheader(
        "Interpretation and Discussion"
    )

    st.write(
        "School completion rates remained generally "
        "high across different income groups and "
        "education stages. The grouped bar chart "
        "allowed comparisons between Primary, "
        "Lower Secondary and Upper Secondary education. "
        "The difference in Upper Secondary completion "
        "between the highest and lowest relative "
        f"income groups was **{difference:+.2f} "
        "percentage points**. These findings "
        "identified differences between income groups "
        "but did not establish that household income "
        "directly caused changes in school completion."
    )

    with st.expander(
        "View Objective 2 Data"
    ):

        st.dataframe(
            data[
                [
                    "state",
                    "year",
                    "income_mean",
                    "School Stage",
                    "completion",
                    "Income Group"
                ]
            ].sort_values(
                ["year", "state"]
            ),
            hide_index=True,
            use_container_width=True
        )


# -------------------------------------------------
# OBJECTIVE 3
# ANNUAL LIVE BIRTHS AND SCHOOL COMPLETION RATES
# -------------------------------------------------
def objective3():

    st.header(
        "Objective 3: Annual Live Births "
        "and School Completion Rates"
    )

    st.write(
        "**Research Objective:** To analyse the relationship "
        "between annual live births and school completion rates."
    )

    data = df3[
        df3["year"].between(
            2016, 2025
        )
    ].dropna(
        subset=[
            "state",
            "abs",
            "completion",
            "stage"
        ]
    ).copy()

    if data.empty:
        st.warning("No valid data for Objective 3.")
        return

    # EACH STATE-YEAR HAS THREE SCHOOL STAGES.
    # BIRTH COUNTS MUST ONLY BE COUNTED ONCE.
    state_year = data.groupby(
        ["state", "year"],
        as_index=False
    ).agg(

        births=("abs", "first"),

        completion=(
            "completion",
            "mean"
        )
    )

    yearly = state_year.groupby(
        "year",
        as_index=False
    ).agg(

        births=("births", "sum"),

        completion=(
            "completion",
            "mean"
        )

    ).sort_values("year")

    if 2016 not in yearly["year"].values:
        st.warning(
            "A 2016 baseline is needed for "
            "the indexed line chart."
        )
        return

    # INDEX VALUES TO 2016 = 100
    baseline = yearly[
        yearly["year"] == 2016
    ].iloc[0]

    yearly["Birth Index"] = (
        yearly["births"]
        / baseline["births"]
        * 100
    )

    yearly["Completion Index"] = (
        yearly["completion"]
        / baseline["completion"]
        * 100
    )

    r = correlation(
        yearly,
        "births",
        "completion"
    )

    latest = yearly.iloc[-1]

    # SUMMARY BOXES
    metrics([
        (
            "2025 Total Live Births",
            f"{latest['births']:,.0f}"
        ),
        (
            "2025 Mean School Completion",
            f"{latest['completion']:.2f}%"
        ),
        (
            "Pearson r (2016–2025)",
            f"{r:+.3f}"
        ),
    ])

    st.subheader(
        "Interactive Line Chart: "
        "Trends from 2016 to 2025"
    )

    # PREPARE BIRTHS SERIES
    births = yearly[
        [
            "year",
            "Birth Index",
            "births"
        ]
    ].rename(
        columns={
            "Birth Index": "Index",
            "births": "Value"
        }
    )

    births["Series"] = (
        "Annual Live Births"
    )

    births["Original Value"] = (
        births["Value"].map(
            lambda x: f"{x:,.0f} births"
        )
    )

    # PREPARE COMPLETION SERIES
    completion = yearly[
        [
            "year",
            "Completion Index",
            "completion"
        ]
    ].rename(
        columns={
            "Completion Index": "Index",
            "completion": "Value"
        }
    )

    completion["Series"] = (
        "School Completion"
    )

    completion["Original Value"] = (
        completion["Value"].map(
            lambda x: f"{x:.2f}%"
        )
    )

    lines = pd.concat(
        [
            births,
            completion
        ],
        ignore_index=True
    )

    # INTERACTIVE LINE CHART
    chart = alt.Chart(
        lines
    ).mark_line(
        point=True,
        strokeWidth=3
    ).encode(

        x=alt.X(
            "year:Q",
            title="Year",
            axis=alt.Axis(
                values=list(
                    range(2016, 2026)
                ),
                format="d"
            ),
            scale=alt.Scale(
                domain=[2016, 2025]
            )
        ),

        y=alt.Y(
            "Index:Q",
            title=(
                "Relative Index "
                "(2016 = 100)"
            ),
            scale=alt.Scale(
                domain=[
                    lines["Index"].min() - 6,
                    lines["Index"].max() + 6
                ]
            )
        ),

        color=alt.Color(
            "Series:N",
            scale=alt.Scale(
                domain=[
                    "Annual Live Births",
                    "School Completion"
                ],
                range=[
                    GREEN,
                    "#D38A49"
                ]
            )
        ),

        tooltip=[
            alt.Tooltip(
                "year:O",
                title="Year"
            ),
            alt.Tooltip(
                "Series:N"
            ),
            alt.Tooltip(
                "Original Value:N"
            ),
            alt.Tooltip(
                "Index:Q",
                format=".1f"
            )
        ]
    )

    baseline_line = alt.Chart(
        pd.DataFrame({
            "Index": [100]
        })
    ).mark_rule(
        color="gray",
        strokeDash=[5, 5]
    ).encode(
        y="Index:Q"
    )

    st.altair_chart(
        (
            chart + baseline_line
        ).properties(
            height=420
        ).interactive(),
        use_container_width=True
    )

    st.caption(
        "Both series are indexed to 2016 = 100 "
        "because annual birth counts and school "
        "completion percentages use different units. "
        "Hover over the points to view their "
        "original values."
    )

    # INTERPRETATION
    births_change = (
        latest["Birth Index"] - 100
    )

    comp_change = (
        latest["completion"]
        - baseline["completion"]
    )

    st.subheader(
        "Interpretation and Discussion"
    )

    st.write(
        f"From 2016 to 2025, annual live births "
        f"changed by **{births_change:+.1f}%**, "
        f"while average school completion changed by "
        f"**{comp_change:+.2f} percentage points**. "
        f"The Pearson correlation across the ten "
        f"yearly observations was **{r:+.3f}**, "
        f"indicating a **{strength(r)} association** "
        f"between the two annual measures. "
        f"These findings compared aggregate time trends "
        f"and did not demonstrate a causal relationship."
    )

    with st.expander(
        "View Objective 3 Yearly Data"
    ):

        st.dataframe(
            yearly.round(2),
            hide_index=True,
            use_container_width=True
        )


# -------------------------------------------------
# DASHBOARD HEADER
# -------------------------------------------------
st.markdown(
    """
    <div style="
        background:#184B3C;
        color:white;
        padding:22px 25px;
        border-radius:14px;
        margin-bottom:20px;
    ">
      <div style="
          font-size:13px;
          color:#DDECE4;
          font-weight:700;
      ">
        JIE42403 · SCIENTIFIC VISUALISATION
      </div>

      <div style="
          font-size:29px;
          font-weight:700;
          margin-top:8px;
      ">
        Malaysia Data Insights
      </div>

      <div style="
          color:#E4F2EB;
          margin-top:5px;
      ">
        Household Income · Annual Live Births ·
        School Completion
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
pages = st.navigation(
    [
        st.Page(
            objective1,
            title="Objective 1: Births & Income",
            icon="📊",
            default=True
        ),

        st.Page(
            objective2,
            title="Objective 2: School & Income",
            icon="🎓"
        ),

        st.Page(
            objective3,
            title="Objective 3: Births & School",
            icon="📈"
        )
    ],
    position="sidebar"
)

pages.run()


# -------------------------------------------------
# DATA SOURCES
# -------------------------------------------------
st.divider()

with st.expander(
    "Data Sources and Methodology"
):

    st.write(
        "Sources: OpenDOSM Malaysian state-level "
        "annual live births, mean household income "
        "and school completion datasets. "
        "Objectives 1 and 2 use selected income "
        "survey years (2016, 2019, 2020, 2022 "
        "and 2024). Objective 3 covers 2016–2025. "
        "Birth counts reflect population size, "
        "while school completion averages "
        "are not population-weighted."
    )

    st.markdown(
        "[Live Births]"
        "(https://open.dosm.gov.my/data-catalogue/births_annual_state)"
        " | "
        "[Household Income]"
        "(https://open.dosm.gov.my/data-catalogue/hh_income_state)"
        " | "
        "[School Completion]"
        "(https://open.dosm.gov.my/data-catalogue/completion_school_state)"
    )

st.caption(
    "JIE42403 Scientific Visualisation | "
    "Malaysia Data Insights | OpenDOSM"
)
