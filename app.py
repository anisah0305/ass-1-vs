
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

from bokeh.embed import file_html
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Label,
    Span
)
from bokeh.plotting import figure
from bokeh.resources import CDN


# ==========================================
# 1. DASHBOARD SETTINGS
# NO SIDEBAR, NO TABS, NO FILTERS
# ==========================================

st.set_page_config(
    page_title="Malaysia Data Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

GREEN = "#247D61"
DARK_GREEN = "#174737"
MINT = "#DDECE4"
ORANGE = "#D58A47"

sns.set_theme(
    style="whitegrid",
    font_scale=1.05
)

st.markdown("""
<style>
.block-container {
    max-width: 1150px;
    padding-top: 1.5rem;
}

h1, h2, h3 {
    color: #174737;
}

div[data-testid="stMetric"] {
    background-color: #F5FAF7;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD YOUR THREE MERGED DATASETS
# ==========================================

@st.cache_data
def load_data():

    folder = Path(__file__).resolve().parent

    paths = {
        "obj1": "df_merged_income_birth (1).csv",
        "obj2": "df_merged_income_rates.csv",
        "obj3": "df_merged_birth_rates_2016_2025.csv"
    }

    frames = {}

    for key, filename in paths.items():

        df = pd.read_csv(
            folder / filename
        )

        # Clean state names
        df["state"] = (
            df["state"].astype(str).str.strip()
        )

        # Extract year
        df["year"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        ).dt.year

        df = df.dropna(
            subset=["year", "state"]
        )

        # Exclude nationwide aggregate
        df = df.loc[
            df["state"] != "Malaysia"
        ].copy()

        df["year"] = df["year"].astype(int)

        # Convert numerical columns
        for col in (
            "income_mean",
            "rate",
            "completion"
        ):

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # Use both genders
        if "sex" in df.columns:

            df["sex"] = (
                df["sex"].astype(str).str.strip()
            )

            df = df.loc[
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

except (
    FileNotFoundError,
    KeyError,
    ValueError
) as error:

    st.error(
        f"Unable to load a merged CSV file: {error}"
    )

    st.info(
        "Check that all three CSV filenames "
        "match your GitHub files."
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


def strength(r):

    if pd.isna(r):
        return "undefined"

    if abs(r) < 0.1:
        level = "negligible"

    elif abs(r) < 0.3:
        level = "weak"

    elif abs(r) < 0.7:
        level = "moderate"

    else:
        level = "strong"

    if r > 0:
        direction = "positive"

    elif r < 0:
        direction = "negative"

    else:
        direction = "zero"

    return f"{level} {direction}"


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


# ==========================================
# 4. DASHBOARD TITLE
# ==========================================

st.markdown(f"""
<div style="
    background:{DARK_GREEN};
    padding:26px 28px;
    border-radius:16px;
    color:white;
">
    <div style="
        font-size:12px;
        letter-spacing:1.4px;
        color:{MINT};
        font-weight:700;
    ">
        JIE42403 · SCIENTIFIC VISUALISATION
    </div>

    <div style="
        font-size:33px;
        font-weight:750;
        margin:6px 0;
    ">
        Malaysia Data Insights
    </div>

    <div style="color:#E5F1EA;">
        Live Births · Mean Household Income ·
        School Completion
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

st.subheader("Research Objectives")

st.write(
    "**Objective 1:** To identify the relationship "
    "between annual live births and household income."
)

st.write(
    "**Objective 2:** To examine the relationship "
    "between school completion rates and household income."
)

st.write(
    "**Objective 3:** To analyse the relationship "
    "between annual live births and school completion rates."
)


# ==========================================
# OBJECTIVE 1: SCATTER PLOT
# 2016-2024 | MEAN INCOME | BIRTH RATE
# ==========================================

st.divider()

st.header(
    "Objective 1 — Live Births and Household Income"
)

st.write(
    "**Objective:** To identify the relationship "
    "between annual live births and household income."
)

# Use all available survey years
a = df1.loc[
    df1["year"].between(2016, 2024)
].dropna(
    subset=["income_mean", "rate"]
).copy()

a["Year"] = a["year"].astype(str)

years1 = sorted(
    a["year"].unique()
)

r1 = pearson(
    a,
    "income_mean",
    "rate"
)

# SUMMARY BOXES
summary_cards([
    (
        "State-Year Observations",
        f"{len(a)}"
    ),
    (
        "Average Birth Rate",
        f"{a['rate'].mean():.2f} per 1,000"
    ),
    (
        "Pearson Correlation (r)",
        f"{r1:+.3f}"
    )
])

# ==========================================
# VISUALIZATION 1
# ==========================================

st.subheader(
    "Visualization 1: Scatter Plot"
)

st.caption(
    "Mean Monthly Household Income vs "
    "Live Birth Rate | 2016–2024"
)

fig1, ax1 = plt.subplots(
    figsize=(10, 5.6)
)

sns.scatterplot(
    data=a,
    x="income_mean",
    y="rate",
    hue="Year",
    style="Year",
    hue_order=[
        str(y) for y in years1
    ],
    style_order=[
        str(y) for y in years1
    ],
    palette="viridis",
    s=92,
    alpha=0.9,
    edgecolor="white",
    ax=ax1
)

# Regression line
sns.regplot(
    data=a,
    x="income_mean",
    y="rate",
    scatter=False,
    ci=None,
    line_kws={
        "color": ORANGE,
        "linewidth": 2.3,
        "linestyle": "--"
    },
    ax=ax1
)

# CHART TITLE
ax1.set_title(
    "Mean Household Income and Birth Rate (2016–2024)",
    fontsize=15,
    fontweight="bold",
    pad=15
)

# X AND Y LABELS
ax1.set_xlabel(
    "Mean Monthly Household Income (RM)"
)

ax1.set_ylabel(
    "Live Birth Rate (births per 1,000 population)"
)

ax1.xaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

# LEGEND LABELS
ax1.legend(
    title="Survey Year",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

sns.despine(ax=ax1)

fig1.tight_layout()

st.pyplot(
    fig1,
    width="stretch"
)

plt.close(fig1)

st.caption(
    "Each marker represents one Malaysian state "
    "or federal territory in a survey year. "
    "The dashed line represents the overall "
    "linear trend across the available years."
)

# INTERPRETATION
st.subheader(
    "Interpretation and Discussion"
)

st.write(
    f"The scatter plot contains **{len(a)} state-year "
    f"observations**, covering the available survey "
    f"years from **2016 to 2024**. "
    f"The Pearson correlation coefficient is "
    f"**{r1:+.3f}**, indicating a "
    f"**{strength(r1)}** linear relationship "
    "between mean household income and birth rate. "
    "Because the data combine multiple years, "
    "the observed association may also reflect "
    "changes over time. Correlation does not "
    "establish causation."
)

with st.expander(
    "Objective 1 — Data Preview"
):

    st.dataframe(
        a[
            [
                "state",
                "year",
                "income_mean",
                "rate"
            ]
        ].sort_values(
            ["year", "state"]
        ),
        hide_index=True
    )


# ==========================================
# OBJECTIVE 2: GROUPED BAR CHART
# 2016-2024 | MEAN INCOME | ALL STAGES
# ==========================================

st.divider()

st.header(
    "Objective 2 — School Completion and Household Income"
)

st.write(
    "**Objective:** To examine the relationship "
    "between school completion rates and household income."
)

b = df2.loc[
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

b["School stage"] = (
    b["stage"].map(stage_names)
)

b = b.dropna(
    subset=["School stage"]
)

# SUMMARY BOXES
summary_cards([
    (
        "Primary Completion",
        f"{b.loc[b['School stage'] == 'Primary', 'completion'].mean():.2f}%"
    ),
    (
        "Lower-Secondary Completion",
        f"{b.loc[b['School stage'] == 'Lower secondary', 'completion'].mean():.2f}%"
    ),
    (
        "Upper-Secondary Completion",
        f"{b.loc[b['School stage'] == 'Upper secondary', 'completion'].mean():.2f}%"
    )
])

# ==========================================
# CREATE INCOME GROUPS
# ==========================================

income_groups = b[
    ["state", "year", "income_mean"]
].drop_duplicates(
    ["state", "year"]
).copy()

group_order = [
    "Lowest",
    "Lower-middle",
    "Upper-middle",
    "Highest"
]

# Divide states into 4 income groups
# separately for each year
income_groups["Income group"] = (
    income_groups.groupby("year")["income_mean"]
    .transform(
        lambda s: pd.qcut(
            s.rank(method="first"),
            4,
            labels=group_order
        )
    )
)

# Add groups to school records
b = b.merge(
    income_groups[
        ["state", "year", "Income group"]
    ],
    on=["state", "year"],
    how="left",
    validate="many_to_one"
)

# Calculate average completion
bar_data = b.groupby(
    ["Income group", "School stage"],
    observed=True
)["completion"].mean().reset_index()

# ==========================================
# VISUALIZATION 2
# ==========================================

st.subheader(
    "Visualization 2: Grouped Bar Chart"
)

st.caption(
    "Average School Completion by Mean Household "
    "Income Group and School Stage | 2016–2024"
)

fig2, ax2 = plt.subplots(
    figsize=(10.5, 5.8)
)

sns.barplot(
    data=bar_data,
    x="Income group",
    y="completion",
    hue="School stage",
    order=group_order,
    hue_order=stage_order,
    palette=[
        "#B5DDD0",
        "#69AF97",
        GREEN
    ],
    errorbar=None,
    ax=ax2
)

# PERCENTAGE LABELS ABOVE EACH BAR
for bars in ax2.containers:

    ax2.bar_label(
        bars,
        fmt="%.1f%%",
        fontsize=9,
        padding=3
    )

# CHART TITLE
ax2.set_title(
    "School Completion by Mean Household Income Group (2016–2024)",
    fontsize=15,
    fontweight="bold",
    pad=15
)

# X AND Y LABELS
ax2.set_xlabel(
    "Relative Mean Household Income Group"
)

ax2.set_ylabel(
    "Average School Completion Rate (%)"
)

# Keep zero baseline
ax2.set_ylim(
    0,
    max(
        112,
        float(
            bar_data["completion"].max()
        ) * 1.16
    )
)

# STAGE LEGEND
ax2.legend(
    title="School Stage",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
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
    "The four income groups represent relative "
    "income quartiles within each year. "
    "They are not official B40, M40 or T20 categories. "
    "Percentage labels show mean completion rates."
)

# ==========================================
# OBJECTIVE 2 INTERPRETATION
# ==========================================

st.subheader(
    "Interpretation and Discussion"
)

comparison = bar_data.pivot(
    index="Income group",
    columns="School stage",
    values="completion"
)

upper_diff = float(
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

st.write(
    "The grouped bar chart compares school "
    "completion rates across the **primary, lower "
    "secondary and upper secondary** stages. "
    "The data cover **2016, 2019, 2020, 2022 "
    "and 2024**. "
    "Mean household income was used to classify "
    "states into four relative income groups "
    "for each year."
)

st.write(
    "For upper-secondary education, the "
    "highest-income group has an average "
    "completion rate that is "
    f"**{upper_diff:+.2f} percentage points** "
    "different from the lowest-income group. "
    "Patterns vary across educational stages. "
    "These comparisons describe associations "
    "rather than causal effects."
)

with st.expander(
    "Objective 2 — Data Preview"
):

    st.dataframe(
        b[
            [
                "state",
                "year",
                "income_mean",
                "School stage",
                "completion",
                "Income group"
            ]
        ].sort_values(
            ["year", "state"]
        ),
        hide_index=True
    )


# ==========================================
# OBJECTIVE 3: INTERACTIVE LINE CHART
# 2016-2025 | BIRTH RATE | COMPLETION
# ==========================================

st.divider()

st.header(
    "Objective 3 — Live Births and School Completion"
)

st.write(
    "**Objective:** To analyse the relationship "
    "between annual live births and school completion rates."
)

c = df3.loc[
    df3["year"].between(2016, 2025)
].dropna(
    subset=[
        "rate",
        "completion",
        "stage"
    ]
).copy()

# Birth rate is repeated for 3 school stages.
# Aggregate once per state and year.
state_year = c.groupby(
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

# ==========================================
# NORMALIZE BOTH VARIABLES
# BASE YEAR 2016 = 100
# ==========================================

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

trend["Birth index"] = (
    trend["birth_rate"] / birth_base * 100
)

trend["Completion index"] = (
    trend["completion_rate"]
    / completion_base * 100
)

latest = trend.iloc[-1]

latest_states = state_year.loc[
    state_year["year"] == int(latest["year"])
]

r3 = pearson(
    latest_states,
    "birth_rate",
    "completion_rate"
)

# SUMMARY BOXES
summary_cards([
    (
        "2025 Average Birth Rate",
        f"{latest['birth_rate']:.2f} per 1,000"
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

# ==========================================
# VISUALIZATION 3
# ==========================================

st.subheader(
    "Visualization 3: Interactive Line Chart"
)

st.caption(
    "Birth Rate and School Completion Trends | "
    "2016–2025 | Index (2016 = 100)"
)

source = ColumnDataSource(trend)

low = float(
    min(
        trend["Birth index"].min(),
        trend["Completion index"].min(),
        100
    )
)

high = float(
    max(
        trend["Birth index"].max(),
        trend["Completion index"].max(),
        100
    )
)

plot = figure(
    height=430,
    sizing_mode="stretch_width",
    x_range=(2015.5, 2026.5),
    y_range=(low - 7, high + 7),
    x_axis_label="Year",
    y_axis_label="Relative Index (2016 = 100)",
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

# LINE 1: BIRTH RATE
plot.line(
    "year",
    "Birth index",
    source=source,
    color=GREEN,
    line_width=3,
    legend_label="Birth Rate"
)

birth_pts = plot.scatter(
    "year",
    "Birth index",
    source=source,
    color=GREEN,
    size=10
)

# LINE 2: SCHOOL COMPLETION
plot.line(
    "year",
    "Completion index",
    source=source,
    color=ORANGE,
    line_width=3,
    legend_label="School Completion"
)

completion_pts = plot.scatter(
    "year",
    "Completion index",
    source=source,
    color=ORANGE,
    size=10
)

# ==========================================
# INTERACTIVE DATA LABELS
# ==========================================

plot.add_tools(
    HoverTool(
        renderers=[birth_pts],
        tooltips=[
            ("Year", "@year"),
            (
                "Birth Rate",
                "@birth_rate{0.00} per 1,000"
            ),
            (
                "Birth Index",
                "@{Birth index}{0.0}"
            )
        ]
    )
)

plot.add_tools(
    HoverTool(
        renderers=[completion_pts],
        tooltips=[
            ("Year", "@year"),
            (
                "School Completion",
                "@completion_rate{0.00}%"
            ),
            (
                "Completion Index",
                "@{Completion index}{0.0}"
            )
        ]
    )
)

# Reference line at 100
plot.add_layout(
    Span(
        location=100,
        dimension="width",
        line_color="#84958D",
        line_dash="dashed"
    )
)

# LABEL THE FINAL POINT OF BOTH LINES
plot.add_layout(
    Label(
        x=2025,
        y=float(latest["Birth index"]),
        text=f"Birth {latest['Birth index']:.1f}",
        x_offset=12,
        y_offset=-6,
        text_color=GREEN,
        text_font_size="11px"
    )
)

plot.add_layout(
    Label(
        x=2025,
        y=float(latest["Completion index"]),
        text=f"Completion {latest['Completion index']:.1f}",
        x_offset=12,
        y_offset=5,
        text_color=ORANGE,
        text_font_size="11px"
    )
)

plot.xaxis.ticker = list(
    range(2016, 2026)
)

plot.ygrid.grid_line_alpha = 0.2

# LINE LEGEND
plot.legend.location = "bottom_left"

plot.legend.click_policy = "hide"

plot.toolbar.logo = None

# Display Bokeh chart in Streamlit
components.html(
    file_html(
        plot,
        CDN,
        "Birth Rate and School Completion Trends"
    ),
    height=490,
    scrolling=False
)

st.caption(
    "Green line: Birth rate. "
    "Orange line: School completion. "
    "Both use 2016 = 100 for fair comparison "
    "of relative changes. Hover over the markers "
    "to view the original values."
)

# ==========================================
# OBJECTIVE 3 INTERPRETATION
# ==========================================

birth_change = float(
    latest["Birth index"] - 100
)

completion_change = float(
    latest["completion_rate"] - completion_base
)

st.subheader(
    "Interpretation and Discussion"
)

st.write(
    "From **2016 to 2025**, the average state "
    "birth rate changed from "
    f"**{birth_base:.2f}** to "
    f"**{latest['birth_rate']:.2f} births per 1,000 "
    "population**, representing a change of "
    f"**{birth_change:+.1f}%**."
)

st.write(
    "Meanwhile, the mean school completion rate "
    "across all three stages changed from "
    f"**{completion_base:.2f}%** to "
    f"**{latest['completion_rate']:.2f}%**, "
    "a difference of "
    f"**{completion_change:+.2f} percentage points**."
)

st.write(
    "Across states in 2025, the Pearson correlation "
    "between birth rate and average school completion "
    f"was **{r3:+.3f}**, representing a "
    f"**{strength(r3)}** linear relationship. "
    "The line chart shows changes over time, "
    "but similar or opposing trends do not "
    "establish causation."
)

with st.expander(
    "Objective 3 — Annual Data Preview"
):

    st.dataframe(
        trend.round(2),
        hide_index=True
    )


# ==========================================
# 5. DATA SOURCES AND LIMITATIONS
# ==========================================

st.divider()

with st.expander(
    "Data Sources, Methodology and Limitations"
):

    st.write(
        "**Data Sources:** OpenDOSM annual live births, "
        "household income and school completion "
        "datasets at state level."
    )

    st.write(
        "**Methodology:** Objective 1 analyses "
        "state-year observations for five survey years. "
        "Objective 2 divides states into four income "
        "groups within each year and compares three "
        "school stages. Objective 3 compares average "
        "birth rates and school completion trends "
        "using a 2016 baseline index."
    )

    st.write(
        "**Limitations:** Household income is reported "
        "in nominal RM. School completion rates can "
        "exceed 100% under the indicator definition. "
        "State averages are unweighted. Statistical "
        "relationships do not prove causation."
    )

    st.markdown(
        "[OpenDOSM Births]"
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
    "Three Objectives | Three Labelled Visualizations"
)
