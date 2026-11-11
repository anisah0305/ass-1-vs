from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure
from bokeh.resources import CDN
from matplotlib.ticker import StrMethodFormatter


# ==========================================
# 1. DASHBOARD CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Malaysia Data Insights",
    page_icon="📊",
    layout="wide"
)

GREEN = "#268567"
DARK_GREEN = "#174737"
MINT = "#DDECE4"
ORANGE = "#DB8653"

sns.set_theme(
    style="whitegrid",
    font_scale=1.05
)

st.markdown("""
<style>
.block-container {
    max-width: 1120px;
    padding-top: 1.4rem;
}

h1, h2, h3 {
    color: #174737;
}

div[data-testid="stMetric"] {
    background-color: #F6FAF8;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD THE THREE MERGED DATASETS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_datasets():

    files = {
        "Objective 1": "objective1_merged.csv",
        "Objective 2": "objective2_merged.csv",
        "Objective 3": "objective3_merged.csv"
    }

    required = {
        "Objective 1": {
            "state", "date", "income_mean",
            "income_median", "abs", "rate"
        },
        "Objective 2": {
            "state", "date", "income_mean",
            "income_median", "stage", "sex",
            "completion"
        },
        "Objective 3": {
            "state", "date", "abs", "rate",
            "stage", "sex", "completion"
        }
    }

    loaded = {}

    for objective, filename in files.items():

        path = BASE_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {filename}"
            )

        df = pd.read_csv(path)

        missing = required[objective] - set(df.columns)

        if missing:
            raise ValueError(
                f"{filename} is missing columns: {missing}"
            )

        # Standardize state names
        df["state"] = (
            df["state"].astype(str).str.strip()
        )

        # Extract year from date
        df["year"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        ).dt.year

        # Convert numeric values
        numeric_columns = [
            "income_mean",
            "income_median",
            "abs",
            "rate",
            "completion"
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # Clean categories
        if "stage" in df.columns:
            df["stage"] = (
                df["stage"].astype(str).str.strip()
            )

        if "sex" in df.columns:
            df["sex"] = (
                df["sex"].astype(str).str.strip()
            )

        # Remove invalid dates and national totals
        df = df.dropna(subset=["year"])

        df = df[
            df["state"] != "Malaysia"
        ].copy()

        df["year"] = df["year"].astype(int)

        loaded[objective] = df

    return (
        loaded["Objective 1"],
        loaded["Objective 2"],
        loaded["Objective 3"]
    )


try:
    births_income, school_income, births_school = (
        load_datasets()
    )

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


# School datasets contain both genders only
school_income = school_income[
    school_income["sex"] == "both"
].copy()

births_school = births_school[
    births_school["sex"] == "both"
].copy()


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def pearson_r(data, x, y):

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


def describe_r(r):

    if pd.isna(r):
        return "not available"

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


def show_summary_cards(metrics):

    columns = st.columns(3)

    for col, (label, value) in zip(
        columns,
        metrics
    ):

        with col:
            st.metric(
                label,
                value,
                border=True
            )


def show_data_preview(data, columns, filename):

    with st.expander("View Data Used for Visualization"):

        st.dataframe(
            data[columns],
            hide_index=True,
            width="stretch"
        )

        st.download_button(
            label="Download Data as CSV",
            data=data[columns].to_csv(
                index=False
            ).encode("utf-8"),
            file_name=filename,
            mime="text/csv"
        )


# ==========================================
# 4. DASHBOARD HEADER
# ==========================================

st.markdown(f"""
<div style="
    background:{DARK_GREEN};
    padding:28px;
    border-radius:16px;
    margin-bottom:18px;
    color:white;
">

    <div style="
        font-size:12px;
        letter-spacing:1.5px;
        color:{MINT};
        font-weight:700;
    ">
        JIE42403 · SCIENTIFIC VISUALISATION
    </div>

    <div style="
        font-size:34px;
        font-weight:750;
        margin-top:6px;
    ">
        Malaysia Data Insights
    </div>

    <div style="
        font-size:15px;
        color:#E6F1EA;
    ">
        Live Births · Household Income ·
        School Completion Rates
    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. RESEARCH OBJECTIVES
# ==========================================

with st.expander(
    "Research Objectives",
    expanded=True
):

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
# 6. SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Analysis Filters")

available_years = sorted(
    set(births_income["year"])
    & set(school_income["year"])
)

if not available_years:
    st.error("No matching survey years found.")
    st.stop()

year = st.sidebar.selectbox(
    "Income Survey Year",
    available_years,
    index=len(available_years) - 1
)

income_choices = {
    "income_median": "Median Household Income",
    "income_mean": "Mean Household Income"
}

income_col = st.sidebar.selectbox(
    "Household Income Measure",
    list(income_choices),
    format_func=income_choices.get
)

stage_choices = {
    "secondary_upper": "Upper Secondary",
    "secondary_lower": "Lower Secondary",
    "primary": "Primary"
}

stage = st.sidebar.selectbox(
    "School Completion Stage",
    list(stage_choices),
    format_func=stage_choices.get
)

st.sidebar.info(
    "Objectives 1 and 2 use the selected income survey year. "
    "Objective 3 analyses 2016–2025. "
    "School completion data represent both genders."
)


# ==========================================
# 7. CREATE THREE DASHBOARD TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "01 · Births & Income",
    "02 · Education & Income",
    "03 · Births & Education"
])


# ==========================================
# OBJECTIVE 1: SCATTER PLOT
# ==========================================

with tab1:

    st.header(
        "Objective 1 — Annual Live Births and Household Income"
    )

    st.write(
        "**Research Objective:** To identify the relationship "
        "between annual live births and household income."
    )

    data1 = births_income[
        births_income["year"] == year
    ].dropna(
        subset=[income_col, "rate"]
    ).copy()

    if (
        len(data1) < 3
        or data1[income_col].nunique() < 2
    ):

        st.warning(
            "Insufficient valid observations "
            "for the selected year."
        )

    else:

        # Calculate correlation
        r1 = pearson_r(
            data1,
            income_col,
            "rate"
        )

        # SUMMARY BOXES
        show_summary_cards([
            (
                "Average Birth Rate",
                f"{data1['rate'].mean():.2f} per 1,000"
            ),
            (
                "Pearson Correlation (r)",
                f"{r1:+.3f}"
            ),
            (
                "States Compared",
                str(data1["state"].nunique())
            )
        ])

        st.caption(
            f"Year: {year} · "
            f"{income_choices[income_col]} · "
            f"{describe_r(r1)} relationship"
        )

        # ONE VISUALIZATION
        st.subheader(
            "Scatter Plot: Household Income vs Live Birth Rate"
        )

        fig1, ax1 = plt.subplots(
            figsize=(10, 5)
        )

        sns.regplot(
            data=data1,
            x=income_col,
            y="rate",
            ci=None,
            scatter_kws={
                "s": 85,
                "color": GREEN,
                "alpha": 0.85,
                "edgecolor": "white"
            },
            line_kws={
                "color": ORANGE,
                "linewidth": 2.5
            },
            ax=ax1
        )

        ax1.set_xlabel(
            f"{income_choices[income_col]} per Month (RM)"
        )

        ax1.set_ylabel(
            "Live Birth Rate (per 1,000 Population)"
        )

        ax1.xaxis.set_major_formatter(
            StrMethodFormatter("{x:,.0f}")
        )

        sns.despine(ax=ax1)

        fig1.tight_layout()

        st.pyplot(
            fig1,
            width="stretch"
        )

        plt.close(fig1)

        st.caption(
            "Each green dot represents one Malaysian state "
            "or federal territory. The orange regression "
            "line shows the overall linear trend."
        )

        # INTERPRETATION
        st.subheader("Interpretation and Discussion")

        st.write(
            f"In **{year}**, the relationship between "
            f"household income and annual live birth rate "
            f"was **{describe_r(r1)}**, with a Pearson "
            f"correlation coefficient of **{r1:+.3f}**. "
            f"The analysis included "
            f"**{data1['state'].nunique()} states and "
            "federal territories**. Birth rate was used "
            "instead of total birth count to allow "
            "fairer comparisons between states with "
            "different population sizes. Correlation "
            "does not imply causation."
        )

        # DATA PREVIEW
        show_data_preview(
            data1.sort_values("state"),
            [
                "state",
                "year",
                income_col,
                "rate",
                "abs"
            ],
            "objective1_chart_data.csv"
        )


# ==========================================
# OBJECTIVE 2: SEABORN BAR CHART
# ==========================================

with tab2:

    st.header(
        "Objective 2 — School Completion Rates and Household Income"
    )

    st.write(
        "**Research Objective:** To examine the relationship "
        "between school completion rates and household income."
    )

    data2 = school_income[
        (school_income["year"] == year)
        & (school_income["stage"] == stage)
    ].dropna(
        subset=[income_col, "completion"]
    ).copy()

    if len(data2) < 4:

        st.warning(
            "Insufficient valid observations "
            "for the selected filters."
        )

    else:

        # Calculate correlation
        r2 = pearson_r(
            data2,
            income_col,
            "completion"
        )

        # SUMMARY BOXES
        show_summary_cards([
            (
                "Average Completion Rate",
                f"{data2['completion'].mean():.2f}%"
            ),
            (
                "Pearson Correlation (r)",
                f"{r2:+.3f}"
            ),
            (
                "States Compared",
                str(data2["state"].nunique())
            )
        ])

        st.caption(
            f"Year: {year} · "
            f"{stage_choices[stage]} · Both genders · "
            f"{describe_r(r2)} relationship"
        )

        # Create income groups
        income_labels = [
            "Lowest",
            "Lower-middle",
            "Upper-middle",
            "Highest"
        ]

        data2["Income Group"] = pd.qcut(
            data2[income_col].rank(
                method="first"
            ),
            q=4,
            labels=income_labels
        )

        # Calculate average completion rates
        grouped = data2.groupby(
            "Income Group",
            observed=True
        )["completion"].mean().reset_index()

        # ONE VISUALIZATION
        st.subheader(
            "Bar Chart: Average School Completion by Income Group"
        )

        fig2, ax2 = plt.subplots(
            figsize=(10, 5)
        )

        sns.barplot(
            data=grouped,
            x="Income Group",
            y="completion",
            color=GREEN,
            ax=ax2
        )

        # Add values above bars
        for bars in ax2.containers:

            ax2.bar_label(
                bars,
                fmt="%.1f%%",
                padding=4,
                fontsize=10
            )

        ax2.set_xlabel(
            "Relative Household Income Group"
        )

        ax2.set_ylabel(
            "Average School Completion Rate (%)"
        )

        # Zero baseline for bar chart
        ax2.set_ylim(
            0,
            max(
                105,
                float(grouped["completion"].max()) * 1.14
            )
        )

        sns.despine(ax=ax2)

        fig2.tight_layout()

        st.pyplot(
            fig2,
            width="stretch"
        )

        plt.close(fig2)

        st.caption(
            "Income groups are relative quartiles based "
            "on household income across the 16 states "
            "and federal territories. They are not "
            "official B40, M40 or T20 classifications."
        )

        # Find highest and lowest income groups
        highest = grouped.loc[
            grouped["completion"].idxmax()
        ]

        lowest = grouped.loc[
            grouped["completion"].idxmin()
        ]

        # INTERPRETATION
        st.subheader("Interpretation and Discussion")

        st.write(
            f"For **{stage_choices[stage].lower()}** "
            f"education in **{year}**, the "
            f"**{highest['Income Group']}** income group "
            "recorded the highest average school "
            f"completion rate of "
            f"**{highest['completion']:.2f}%**. "
            f"The **{lowest['Income Group']}** group "
            "recorded the lowest average rate of "
            f"**{lowest['completion']:.2f}%**. "
            "The Pearson correlation coefficient "
            f"was **{r2:+.3f}**, indicating a "
            f"**{describe_r(r2)}** state-level relationship. "
            "The bar chart compares group averages "
            "rather than displaying individual "
            "state-level correlations directly."
        )

        # DATA PREVIEW
        show_data_preview(
            data2.sort_values(income_col),
            [
                "state",
                "year",
                "stage",
                income_col,
                "completion",
                "Income Group"
            ],
            "objective2_chart_data.csv"
        )


# ==========================================
# OBJECTIVE 3: INTERACTIVE BOKEH LINE CHART
# ==========================================

with tab3:

    st.header(
        "Objective 3 — Annual Live Births and School Completion"
    )

    st.write(
        "**Research Objective:** To analyse the relationship "
        "between annual live births and school completion rates."
    )

    data3 = births_school[
        births_school["stage"] == stage
    ].dropna(
        subset=["rate", "completion"]
    ).copy()

    # Calculate correlation for each year
    yearly_results = []

    for yr, group in data3.groupby("year"):

        r = pearson_r(
            group,
            "rate",
            "completion"
        )

        if pd.notna(r):

            yearly_results.append({
                "Year": int(yr),
                "Pearson r": r,
                "States": int(
                    group["state"].nunique()
                )
            })

    if not yearly_results:

        st.warning(
            "No valid yearly correlations found."
        )

    else:

        trend = pd.DataFrame(
            yearly_results
        ).sort_values(
            "Year"
        ).reset_index(drop=True)

        latest = trend.iloc[-1]

        # Find strongest absolute correlation
        strongest = trend.loc[
            trend["Pearson r"].abs().idxmax()
        ]

        # SUMMARY BOXES
        show_summary_cards([
            (
                "Years Analysed",
                str(trend["Year"].nunique())
            ),
            (
                f"Strongest Correlation ({int(strongest['Year'])})",
                f"{strongest['Pearson r']:+.3f}"
            ),
            (
                f"Latest Correlation ({int(latest['Year'])})",
                f"{latest['Pearson r']:+.3f}"
            )
        ])

        st.caption(
            f"{stage_choices[stage]} · Both genders · "
            "Yearly correlations across Malaysian states"
        )

        # ONE VISUALIZATION
        st.subheader(
            "Interactive Line Chart: Relationship Changes Over Time"
        )

        source = ColumnDataSource(trend)

        plot = figure(
            height=400,
            sizing_mode="stretch_width",
            x_axis_label="Year",
            y_axis_label="Pearson Correlation Coefficient (r)",
            x_range=(
                int(trend["Year"].min()) - 0.45,
                int(trend["Year"].max()) + 0.45
            ),
            y_range=(-1, 1),
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        # Create line
        plot.line(
            x="Year",
            y="Pearson r",
            source=source,
            line_color=DARK_GREEN,
            line_width=3
        )

        # Create markers
        points = plot.scatter(
            x="Year",
            y="Pearson r",
            source=source,
            size=11,
            color=GREEN
        )

        # Interactive hover tool
        plot.add_tools(
            HoverTool(
                renderers=[points],
                tooltips=[
                    ("Year", "@Year"),
                    (
                        "Pearson r",
                        "@{Pearson r}{+0.000}"
                    ),
                    ("States Compared", "@States")
                ]
            )
        )

        # Horizontal reference at zero
        plot.add_layout(
            Span(
                location=0,
                dimension="width",
                line_color="#85938C",
                line_dash="dashed"
            )
        )

        plot.xaxis.ticker = (
            trend["Year"].astype(int).tolist()
        )

        plot.ygrid.grid_line_alpha = 0.16

        plot.toolbar.logo = None

        # Display interactive Bokeh chart
        st.iframe(
            file_html(
                plot,
                CDN,
                "Birth Rates and School Completion"
            ),
            height=460,
            alt=(
                "Interactive line chart showing "
                "yearly Pearson correlations"
            )
        )

        st.caption(
            "Hover over the green markers to view "
            "the year, Pearson correlation and number "
            "of states included in the analysis."
        )

        # INTERPRETATION
        st.subheader("Interpretation and Discussion")

        st.write(
            f"Between **{int(trend['Year'].min())}** "
            f"and **{int(trend['Year'].max())}**, the "
            "relationship between annual live birth "
            "rates and school completion rates changed "
            "across years. For "
            f"**{stage_choices[stage].lower()}** "
            f"education, the latest coefficient in "
            f"**{int(latest['Year'])}** was "
            f"**{latest['Pearson r']:+.3f}**, representing "
            f"a **{describe_r(latest['Pearson r'])}** "
            "linear relationship. Positive correlations "
            "indicate that states with higher birth rates "
            "tended to have higher school completion "
            "rates, while negative correlations indicate "
            "the opposite. These findings do not "
            "establish causation."
        )

        # DATA PREVIEW
        with st.expander(
            "View Yearly Correlation Data"
        ):

            st.dataframe(
                trend.round(3),
                hide_index=True,
                width="stretch"
            )

            st.download_button(
                "Download Yearly Correlations",
                data=trend.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "objective3_yearly_correlations.csv"
                ),
                mime="text/csv"
            )


# ==========================================
# 8. DATA SOURCES AND METHODOLOGY
# ==========================================

st.divider()

with st.expander(
    "Dataset Sources, Preparation and Limitations"
):

    st.write(
        "**Data Sources:** The datasets were obtained "
        "from the Department of Statistics Malaysia "
        "(OpenDOSM) and relevant contributing agencies."
    )

    st.markdown(
        "[Annual Live Births by State]"
        "(https://open.dosm.gov.my/data-catalogue/births_annual_state)"
    )

    st.markdown(
        "[Household Income by State]"
        "(https://open.dosm.gov.my/data-catalogue/hh_income_state)"
    )

    st.markdown(
        "[School Completion Rates by State]"
        "(https://open.dosm.gov.my/data-catalogue/completion_school_state)"
    )

    st.write(
        "**Data Preparation:** The merged datasets "
        "combine matching records using state and year. "
        "Only combined-gender school completion records "
        "are included. Missing numeric observations "
        "are excluded from the respective analyses."
    )

    st.write(
        "**Limitations:** Household income survey "
        "records are available for 2016, 2019, 2020, "
        "2022 and 2024. Household income is reported "
        "in nominal RM. School completion rates may "
        "exceed 100% under the indicator definition. "
        "All correlations describe state-level "
        "associations and do not demonstrate causation."
    )

st.caption(
    "JIE42403 Scientific Visualisation | "
    "Malaysia Data Insights | OpenDOSM"
)
