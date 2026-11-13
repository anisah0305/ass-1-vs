from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import streamlit as st

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Malaysia Data Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

GREEN = "#287C62"
DARK_GREEN = "#184B3C"
ORANGE = "#D38A49"

YEARS = [2016, 2019, 2020, 2022, 2024]

st.markdown("""
<style>
.block-container {
    max-width: 1120px;
    padding-top: 1.3rem;
}
h1, h2, h3 {
    color: #184B3C;
}
div[data-testid="stMetric"] {
    background-color: #F4F9F6;
    border: 1px solid #DDECE4;
    border-radius: 12px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. LOAD DATASETS
# =====================================================
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

        path = folder / filename

        if not path.exists():
            raise FileNotFoundError(filename)

        df = pd.read_csv(path)

        df["state"] = (
            df["state"].astype(str).str.strip()
        )

        df["year"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        ).dt.year

        df = df.dropna(subset=["year"])

        df = df[
            df["state"].str.lower() != "malaysia"
        ].copy()

        df["year"] = df["year"].astype(int)

        for col in [
            "income_mean",
            "abs",
            "completion"
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        if "sex" in df.columns:
            df = df[
                df["sex"].astype(str)
                .str.lower()
                .eq("both")
            ].copy()

        frames[key] = df

    return (
        frames["obj1"],
        frames["obj2"],
        frames["obj3"]
    )


try:
    df1, df2, df3 = load_data()

except Exception as e:
    st.error(f"Unable to load CSV files: {e}")
    st.stop()


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================
def pearson(data, x, y):

    valid = data[[x, y]].dropna()

    if (
        len(valid) < 3
        or valid[x].nunique() < 2
        or valid[y].nunique() < 2
    ):
        return float("nan")

    return valid[x].corr(valid[y])


def correlation_label(r):

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


def summary_cards(items):

    cols = st.columns(3)

    for col, (title, value) in zip(cols, items):
        col.metric(title, value)


# =====================================================
# OBJECTIVE 1
# ANNUAL LIVE BIRTHS VS HOUSEHOLD INCOME
# ONE GRAPH - NO FILTER
# =====================================================
def objective1():

    st.header(
        "Objective 1: Annual Live Births "
        "and Household Income"
    )

    st.write(
        "**Research Objective:** To identify the "
        "relationship between annual live births "
        "and household income."
    )

    # Prepare data
    data = df1[
        df1["year"].isin(YEARS)
    ].dropna(
        subset=[
            "state",
            "income_mean",
            "abs"
        ]
    ).copy()

    if data.empty:
        st.warning("No valid data available.")
        return

    # Statistical calculations
    r = pearson(
        data,
        "income_mean",
        "abs"
    )

    avg_births = data["abs"].mean()

    # Summary boxes
    summary_cards([
        (
            "Area-Year Observations",
            f"{len(data):,}"
        ),
        (
            "Average Births per Area-Year",
            f"{avg_births:,.2f}"
        ),
        (
            "Pearson Correlation (r)",
            f"{r:+.3f}"
        )
    ])

    # -------------------------------------------------
    # ONE ENHANCED INTERACTIVE SCATTER PLOT
    # -------------------------------------------------
    st.subheader(
        "Visualisation 1 — Enhanced Scatter Plot"
    )

    colors = {
        2016: "#6654A4",
        2019: "#377CA8",
        2020: "#299A8B",
        2022: "#D58A47",
        2024: "#84AD50"
    }

    fig = go.Figure()

    # Plot all five years without filtering
    for year in YEARS:

        group = data[
            data["year"] == year
        ].copy()

        custom = np.column_stack([
            group["state"].astype(str),
            group["year"].astype(str)
        ])

        fig.add_trace(
            go.Scatter(
                x=group["income_mean"],
                y=group["abs"],
                mode="markers",
                name=str(year),
                customdata=custom,

                marker=dict(
                    size=11,
                    color=colors[year],
                    opacity=0.87,
                    line=dict(
                        color="white",
                        width=1
                    )
                ),

                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Year: %{customdata[1]}"
                    "<br>Mean Monthly Income: RM%{x:,.0f}"
                    "<br>Annual Live Births: %{y:,.0f}"
                    "<extra></extra>"
                )
            )
        )

    # -------------------------------------------------
    # HIGHLIGHT MAIN CLUSTER
    # -------------------------------------------------
    cluster = data[
        data["income_mean"].between(
            4000, 8000
        )
        &
        data["abs"].between(
            15000, 40000
        )
    ]

    fig.add_shape(
        type="rect",
        x0=4000,
        x1=8000,
        y0=15000,
        y1=40000,
        fillcolor="rgba(40,124,98,0.075)",
        line=dict(
            color="rgba(40,124,98,0.55)",
            width=1,
            dash="dot"
        ),
        layer="below"
    )

    # -------------------------------------------------
    # REGRESSION LINE
    # -------------------------------------------------
    x = data["income_mean"].to_numpy()
    y = data["abs"].to_numpy()

    slope, intercept = np.polyfit(
        x, y, 1
    )

    x_line = np.linspace(
        x.min(),
        x.max(),
        100
    )

    y_line = slope * x_line + intercept

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Overall Regression Line",
            line=dict(
                color="#263D35",
                width=2,
                dash="dash"
            ),
            hoverinfo="skip"
        )
    )

    # -------------------------------------------------
    # LABEL IMPORTANT OBSERVATIONS
    # -------------------------------------------------
    highest = data.loc[
        data["abs"].idxmax()
    ]

    lowest = data.loc[
        data["abs"].idxmin()
    ]

    fig.add_annotation(
        x=float(highest["income_mean"]),
        y=float(highest["abs"]),
        text=(
            f"{highest['state']} "
            f"{int(highest['year'])}"
            f"<br>{highest['abs']:,.0f} births"
        ),
        showarrow=True,
        ax=55,
        ay=-32,
        arrowhead=2,
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#DDECE4",
        font=dict(
            size=10,
            color="#263D35"
        )
    )

    # Compare two states with similar high incomes
    selangor = data[
        (data["state"].str.lower() == "selangor")
        &
        (data["year"] == 2024)
    ]

    putrajaya = data[
        data["state"].str.contains(
            "putrajaya",
            case=False,
            na=False
        )
        &
        (data["year"] == 2024)
    ]

    if (
        len(selangor) == 1
        and len(putrajaya) == 1
    ):

        selected_points = [
            (
                selangor.iloc[0],
                "Selangor 2024",
                -85,
                -35
            ),
            (
                putrajaya.iloc[0],
                "Putrajaya 2024",
                -100,
                -42
            )
        ]

        for row, label, ax, ay in selected_points:

            fig.add_annotation(
                x=float(row["income_mean"]),
                y=float(row["abs"]),
                text=label,
                showarrow=True,
                ax=ax,
                ay=ay,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="#DDECE4",
                font=dict(
                    size=10,
                    color="#263D35"
                )
            )

    # -------------------------------------------------
    # CHART DESIGN
    # -------------------------------------------------
    fig.update_layout(

        title=(
            "Household Income vs. Annual Live Births "
            "(2016–2024)"
        ),

        xaxis_title=(
            "Mean Monthly Household Income (RM)"
        ),

        yaxis_title=(
            "Annual Live Births (Number of Births)"
        ),

        legend_title="Survey Year",
        template="plotly_white",
        height=595,
        hovermode="closest",

        margin=dict(
            l=45,
            r=35,
            t=75,
            b=55
        ),

        font=dict(size=12)
    )

    fig.update_xaxes(
        tickprefix="RM",
        tickformat=",",
        range=[
            max(
                0,
                data["income_mean"].min() - 700
            ),
            data["income_mean"].max() + 700
        ]
    )

    fig.update_yaxes(
        tickformat=",",
        range=[
            0,
            data["abs"].max() * 1.20
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Each point represents one Malaysian state "
        "or Federal Territory in a selected survey year. "
        "Colours distinguish the five years. "
        "The shaded rectangle highlights a densely "
        "populated region of the scatter plot, "
        "while the dashed line represents the overall "
        "linear regression trend. Hover over points "
        "to view individual state information."
    )

    # -------------------------------------------------
    # KEY FINDINGS
    # -------------------------------------------------
    st.subheader(
        "Patterns, Trends and Insights"
    )

    st.write(
        f"**1. Main cluster and data distribution:** "
        f"The highlighted region contains "
        f"**{len(cluster)} out of {len(data)} observations** "
        f"({len(cluster)/len(data)*100:.1f}%). "
        f"These observations have household incomes "
        f"between RM4,000 and RM8,000 and annual "
        f"live births between 15,000 and 40,000. "
        f"This concentration shows that states "
        f"with similar income levels can still "
        f"record different birth totals."
    )

    st.write(
        f"**2. Highest and lowest observations:** "
        f"The highest recorded observation was "
        f"**{highest['state']} in "
        f"{int(highest['year'])}**, "
        f"with **{highest['abs']:,.0f} live births**. "
        f"The lowest was **{lowest['state']} in "
        f"{int(lowest['year'])}**, "
        f"with **{lowest['abs']:,.0f} live births**. "
        f"This difference highlights the wide "
        f"variation between Malaysian areas."
    )

    if (
        len(selangor) == 1
        and len(putrajaya) == 1
    ):

        s = selangor.iloc[0]
        p = putrajaya.iloc[0]

        st.write(
            f"**3. Comparison between similar income levels:** "
            f"In 2024, Selangor recorded "
            f"**{s['abs']:,.0f} live births** "
            f"with a mean monthly household income "
            f"of **RM{s['income_mean']:,.0f}**. "
            f"In comparison, Putrajaya recorded "
            f"**{p['abs']:,.0f} live births** "
            f"with an income of "
            f"**RM{p['income_mean']:,.0f}**. "
            f"Despite having similar household "
            f"income levels, the birth totals "
            f"were substantially different. "
            f"Population size may contribute "
            f"to this variation."
        )

    # -------------------------------------------------
    # COMPARE 2016 AND 2024
    # -------------------------------------------------
    yearly = data.groupby(
        "year"
    ).agg(
        mean_income=(
            "income_mean",
            "mean"
        ),
        total_births=(
            "abs",
            "sum"
        )
    )

    if (
        2016 in yearly.index
        and 2024 in yearly.index
    ):

        income_2016 = yearly.loc[
            2016, "mean_income"
        ]

        income_2024 = yearly.loc[
            2024, "mean_income"
        ]

        births_2016 = yearly.loc[
            2016, "total_births"
        ]

        births_2024 = yearly.loc[
            2024, "total_births"
        ]

        st.write(
            f"**4. Changes between 2016 and 2024:** "
            f"The unweighted average of state-level "
            f"mean household incomes increased from "
            f"**RM{income_2016:,.0f}** in 2016 "
            f"to **RM{income_2024:,.0f}** in 2024. "
            f"Meanwhile, combined annual live births "
            f"decreased from **{births_2016:,.0f}** "
            f"to **{births_2024:,.0f}**. "
            f"These changes describe overall "
            f"trends but do not establish that "
            f"income caused births to decline."
        )

    st.write(
        f"**5. Overall relationship:** "
        f"The Pearson correlation coefficient "
        f"of **{r:+.3f}** indicated a "
        f"**{correlation_label(r)} linear relationship** "
        f"between household income and annual live births. "
        f"The almost horizontal regression line "
        f"supported this finding. Overall, household "
        f"income alone was not a strong indicator "
        f"of annual live-birth totals. Differences "
        f"in population size, age structure and "
        f"other demographic characteristics may "
        f"also influence the observed results."
    )

    # -------------------------------------------------
    # DATA TABLE
    # -------------------------------------------------
    with st.expander(
        "View Objective 1 Dataset"
    ):

        display = data[
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


# =====================================================
# OBJECTIVE 2
# SCHOOL COMPLETION AND HOUSEHOLD INCOME
# =====================================================
def objective2():

    st.header(
        "Objective 2: School Completion Rates "
        "and Household Income"
    )

    st.write(
        "**Research Objective:** To examine the "
        "relationship between school completion "
        "rates and household income."
    )

    data = df2[
        df2["year"].isin(YEARS)
    ].dropna(
        subset=[
            "income_mean",
            "completion",
            "stage"
        ]
    ).copy()

    stage_map = {
        "primary": "Primary",
        "secondary_lower": "Lower Secondary",
        "secondary_upper": "Upper Secondary"
    }

    stage_order = [
        "Primary",
        "Lower Secondary",
        "Upper Secondary"
    ]

    data["School Stage"] = (
        data["stage"].map(stage_map)
    )

    data = data.dropna(
        subset=["School Stage"]
    )

    if data.empty:
        st.warning("No valid data available.")
        return

    means = data.groupby(
        "School Stage"
    )["completion"].mean()

    summary_cards([
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
        )
    ])

    # Relative income quartiles
    income_data = data[
        [
            "state",
            "year",
            "income_mean"
        ]
    ].drop_duplicates(
        ["state", "year"]
    ).copy()

    income_order = [
        "Lowest",
        "Lower-middle",
        "Upper-middle",
        "Highest"
    ]

    income_data["Income Group"] = (
        income_data.groupby(
            "year"
        )["income_mean"].transform(
            lambda x: pd.qcut(
                x.rank(method="first"),
                4,
                labels=income_order
            )
        )
    )

    data = data.merge(
        income_data[
            ["state", "year", "Income Group"]
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

    grouped["Label"] = (
        grouped["completion"].map(
            lambda x: f"{x:.1f}%"
        )
    )

    # GROUPED BAR CHART
    st.subheader(
        "Visualisation 2 — Grouped Bar Chart"
    )

    bars = alt.Chart(
        grouped
    ).mark_bar().encode(

        x=alt.X(
            "Income Group:N",
            sort=income_order,
            title="Relative Household Income Group"
        ),

        xOffset=alt.XOffset(
            "School Stage:N",
            sort=stage_order
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
            scale=alt.Scale(
                domain=stage_order,
                range=[
                    "#BCDCCE",
                    "#73B49C",
                    GREEN
                ]
            )
        ),

        tooltip=[
            "Income Group:N",
            "School Stage:N",
            alt.Tooltip(
                "completion:Q",
                format=".2f"
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
            sort=income_order
        ),

        xOffset=alt.XOffset(
            "School Stage:N",
            sort=stage_order
        ),

        y="completion:Q",
        text="Label:N"
    )

    st.altair_chart(
        (bars + labels).properties(
            title=(
                "School Completion Rates "
                "by Household Income Group"
            ),
            height=400
        ),
        use_container_width=True
    )

    st.caption(
        "Income groups represent relative "
        "state income quartiles within each "
        "survey year. They are not official "
        "B40, M40 or T20 classifications."
    )

    # INTERPRETATION
    st.subheader(
        "Interpretation and Discussion"
    )

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

    st.write(
        f"The grouped bar chart compared school "
        f"completion rates across four relative "
        f"income groups and three education stages. "
        f"Completion rates remained generally high, "
        f"although differences were observed between "
        f"education stages. The difference in Upper "
        f"Secondary completion between the highest "
        f"and lowest income groups was "
        f"**{difference:+.2f} percentage points**. "
        f"These results describe differences between "
        f"groups rather than a causal effect of income."
    )


# =====================================================
# OBJECTIVE 3
# ANNUAL LIVE BIRTHS AND SCHOOL COMPLETION
# =====================================================
def objective3():

    st.header(
        "Objective 3: Annual Live Births "
        "and School Completion Rates"
    )

    st.write(
        "**Research Objective:** To analyse the "
        "relationship between annual live births "
        "and school completion rates."
    )

    data = df3[
        df3["year"].between(
            2016, 2025
        )
    ].dropna(
        subset=[
            "abs",
            "completion",
            "stage"
        ]
    ).copy()

    if data.empty:
        st.warning("No valid data available.")
        return

    # Count births once per state-year
    state_year = data.groupby(
        ["state", "year"],
        as_index=False
    ).agg(
        births=("abs", "first"),
        completion=("completion", "mean")
    )

    yearly = state_year.groupby(
        "year",
        as_index=False
    ).agg(
        births=("births", "sum"),
        completion=("completion", "mean")
    ).sort_values("year")

    base = yearly[
        yearly["year"] == 2016
    ].iloc[0]

    yearly["Birth Index"] = (
        yearly["births"]
        / base["births"]
        * 100
    )

    yearly["Completion Index"] = (
        yearly["completion"]
        / base["completion"]
        * 100
    )

    latest = yearly.iloc[-1]

    latest_states = state_year[
        state_year["year"] == 2025
    ]

    r = pearson(
        latest_states,
        "births",
        "completion"
    )

    summary_cards([
        (
            "2025 Total Live Births",
            f"{latest['births']:,.0f}"
        ),
        (
            "2025 Mean School Completion",
            f"{latest['completion']:.2f}%"
        ),
        (
            "2025 Pearson r (Across Areas)",
            f"{r:+.3f}"
        )
    ])

    # INTERACTIVE LINE CHART
    st.subheader(
        "Visualisation 3 — Interactive Line Chart"
    )

    birth_series = yearly[
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

    birth_series["Indicator"] = (
        "Annual Live Births"
    )

    birth_series["Original Value"] = (
        birth_series["Value"].map(
            lambda x: f"{x:,.0f} births"
        )
    )

    completion_series = yearly[
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

    completion_series["Indicator"] = (
        "School Completion"
    )

    completion_series["Original Value"] = (
        completion_series["Value"].map(
            lambda x: f"{x:.2f}%"
        )
    )

    plot_data = pd.concat(
        [
            birth_series,
            completion_series
        ],
        ignore_index=True
    )

    chart = alt.Chart(
        plot_data
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
                "Relative Index (2016 = 100)"
            ),
            scale=alt.Scale(
                domain=[
                    plot_data["Index"].min() - 7,
                    plot_data["Index"].max() + 7
                ]
            )
        ),

        color=alt.Color(
            "Indicator:N",
            scale=alt.Scale(
                domain=[
                    "Annual Live Births",
                    "School Completion"
                ],
                range=[
                    GREEN,
                    ORANGE
                ]
            )
        ),

        tooltip=[
            "year:O",
            "Indicator:N",
            "Original Value:N",
            alt.Tooltip(
                "Index:Q",
                format=".1f"
            )
        ]
    )

    st.altair_chart(
        chart.properties(
            title=(
                "Annual Live Births and School "
                "Completion Trends (2016–2025)"
            ),
            height=420
        ).interactive(),
        use_container_width=True
    )

    st.caption(
        "Both measures are indexed to 2016 = 100 "
        "because annual live births and school "
        "completion percentages have different "
        "measurement units."
    )

    # INTERPRETATION
    st.subheader(
        "Interpretation and Discussion"
    )

    birth_change = (
        latest["Birth Index"] - 100
    )

    completion_change = (
        latest["completion"]
        - base["completion"]
    )

    st.write(
        f"Between 2016 and 2025, annual live "
        f"births changed by "
        f"**{birth_change:+.1f}%**, while average "
        f"school completion changed by "
        f"**{completion_change:+.2f} percentage points**. "
        f"The 2025 correlation across Malaysian "
        f"states and Federal Territories was "
        f"**{r:+.3f}**, indicating a "
        f"**{correlation_label(r)} association**. "
        f"These findings describe observed patterns "
        f"and do not establish causation."
    )


# =====================================================
# 4. DASHBOARD HEADER
# =====================================================
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


# =====================================================
# 5. NAVIGATION
# =====================================================
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


# =====================================================
# 6. DATA SOURCES
# =====================================================
st.divider()

with st.expander(
    "Data Sources and Methodology"
):

    st.write(
        "Data source: OpenDOSM Malaysian state-level "
        "annual live births, household income and "
        "school completion datasets. Objectives 1 "
        "and 2 use selected years between 2016 "
        "and 2024. Objective 3 covers 2016–2025. "
        "Annual birth counts are affected by "
        "population size, and associations should "
        "not be interpreted as causal."
    )

st.caption(
    "JIE42403 Scientific Visualisation | "
    "Malaysia Data Insights | OpenDOSM"
)
