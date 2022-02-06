from operator import index
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
import seaborn as sns
import streamlit as st
from typing import List, Tuple
from datetime import datetime as dt
from datetime import timedelta
from prophet import Prophet
from sklearn.metrics import *


pio.templates.default = "presentation"
st.set_page_config(layout="wide")

data_root = os.path.normpath(os.getcwd() + os.sep + "data")


def get_prices_data(data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch csv's from the preprocessed data folder"""
    prices_hourly_df = pd.read_csv(
        f"{data_root}/processed/elspot_hourly.csv", index_col="datetime", parse_dates=["datetime"]
    )
    prices_daily_df = pd.read_csv(
        f"{data_root}/processed/elspot_daily.csv", index_col="datetime", parse_dates=["datetime"]
    )
    prices_monthly_df = pd.read_csv(
        f"{data_root}/processed/elspot_monthly.csv", index_col="datetime", parse_dates=["datetime"]
    )
    return prices_hourly_df, prices_daily_df, prices_monthly_df


prices_hourly_df, prices_daily_df, prices_monthly_df = get_prices_data(data_root=data_root)


def get_denmark_climate_data(data_root: str) -> pd.DataFrame:
    """Fetch processed data of Denmark climate data"""
    denmark_climate_df = pd.read_csv(f"{data_root}/processed/denmark_market_data.csv")
    denmark_climate_df = denmark_climate_df.dropna()
    denmark_climate_df["utc_timestamp"] = pd.to_datetime(denmark_climate_df["utc_timestamp"])
    denmark_climate_df["utc_timestamp"] = denmark_climate_df["utc_timestamp"].dt.tz_convert(None)
    denmark_climate_df = denmark_climate_df.set_index("utc_timestamp", drop=True)
    return denmark_climate_df


denmark_climate_df = get_denmark_climate_data(data_root=data_root)


def run_app():

    areas_list: List[str] = [
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "DK1",
        "SYS",
        "FI",
        "DK2",
        "Oslo",
        "Kr.sand",
        "Bergen",
        "Molde",
        "Tr.heim",
        "TromsÃ¸",
    ]
    years_list: List[int] = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    months_list: List[str] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    def get_prices_barplot(daily_df: pd.DataFrame, year: int, locations: List[str]):
        fig = px.bar(daily_df.loc[str(year)].reset_index(), x="datetime", y=locations)

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=9, label="9m", step="month", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
        )
        fig.update_layout(
            title_text=f"<b><i>Seasonal representations - {year}</b></i>",
            title_x=0.5,
            height=600,
            legend=dict(
                orientation="h",
                xanchor="center",
                x=0.5,
                y=0.01,
                font=dict(
                    family="Courier New",
                    size=12,
                ),
            ),
        )

        return fig

    def get_prices_funnelplot(monthly_df: pd.DataFrame, year: int):
        fig = go.Figure()
        temp_df = monthly_df.loc[monthly_df.loc[str(year)].sum(axis=1).sort_values(ascending=False).index].copy(
            deep=True
        )
        for name in temp_df.columns.tolist():
            fig.add_trace(
                go.Funnel(
                    name=name,
                    y=temp_df.loc[str(year)][name].index.month_name().tolist(),
                    x=temp_df.loc[str(year)][name].to_numpy().astype(int),
                    textinfo="value",
                )
            )
        temp_df = None
        fig.update_layout(
            title_text=f"<b><i>Monthly Averages of NordPool Prices - {year}</b></i>",
            title_x=0.5,
            height=600,
            legend=dict(
                orientation="h",
                xanchor="center",
                x=0.5,
                font=dict(
                    family="Courier New",
                    size=12,
                ),
            ),
        )
        return fig

    def get_prices_boxplot(daily_df: pd.DataFrame, year: int, location: str):
        fig = go.Figure()
        temp_df = daily_df.loc[str(year)][location]
        month_names = months_list

        for month in month_names:
            fig.add_trace(
                go.Box(
                    y=temp_df.loc[temp_df.index.month_name() == month].to_numpy(),
                    name=month,
                    boxpoints="all",
                    showlegend=False,
                )
            )
        temp_df = None
        fig.update_layout(
            title_text=f"<b><i>Prices Deviations for {location} in {year}</b></i>",
            title_x=0.5,
            height=600,
        )
        fig.update_xaxes(tickfont_size=14, tickfont_family="Droid Sans Mono", showgrid=False)
        fig.update_yaxes(showgrid=False)
        return fig

    def get_prices_difference(daily_df: pd.DataFrame, year: int, area_from: str, area_to: str):

        area1 = daily_df.loc[str(year)][area_from]
        area2 = daily_df.loc[str(year)][area_to]
        daily_diff = pd.DataFrame((area1 - area2).groupby("datetime").mean(), columns=["price_difference"])

        fig = go.Figure([go.Bar(x=daily_diff.index, y=daily_diff["price_difference"])])
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=9, label="9m", step="month", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
        )
        fig.update_layout(
            title_text=f"<b><i>Price Differences between {area_from} and {area_to} during {year}</b></i>",
            title_x=0.5,
            height=600,
        )
        return fig

    def get_climate_heat_map(climate_df: pd.DataFrame):
        corr = climate_df.corr()
        fig = plt.figure(figsize=(11, 8))
        sns.heatmap(corr, cmap="Blues", annot=True)
        return fig

    def prophet_regressor(df):

        start_time = max(min(df.index), dt.fromisoformat("2017-01-01"))
        end_time = min(max(df.index), dt.fromisoformat("2019-12-31 23:00:00"))
        num_days_for_test = 30

        df_app = df.loc[start_time:end_time]
        df_app["ds"] = df_app.index
        df_app.rename({"DK_price_day_ahead": "y"}, axis=1, inplace=True)
        test_start_time = end_time - timedelta(days=num_days_for_test)
        df_train = df_app.loc[df_app["ds"] < test_start_time]
        df_test = df_app.loc[df_app["ds"] >= test_start_time]

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        for c in df_app.columns.to_list():
            if c not in ["ds", "y"]:
                m.add_regressor(c)
        m.fit(df_train)

        forecast = m.predict(df_test.drop(columns="y"))

        evaluation_df = pd.DataFrame({"y_pred": forecast["yhat"].round(2).tolist(), "y_true": df_test["y"].tolist()})
        print("MAE: %.3f" % mean_absolute_error(evaluation_df["y_true"], evaluation_df["y_pred"]))

        forecast.index = forecast["ds"]

        fig3 = m.plot(forecast, uncertainty=True, xlabel="Date", ylabel="price_day_ahead")
        ax = fig3.gca()
        ax.set_xlim(pd.to_datetime([test_start_time - timedelta(days=num_days_for_test), end_time]))
        st.pyplot(fig3)

        ax = forecast.plot(x="ds", y="yhat", legend=True, label="predictions", figsize=(12, 8))
        df_test.plot(
            x="ds", y="y", legend=True, label="True Test Data", ax=ax, xlabel="Date", ylabel="price_day_ahead"
        )
        # st.pyplot(ax)

        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)

    st.markdown(
        "<h1 style='text-align: center; color: #00CC96;'>Climate Electricity Interactions!</h1>",
        unsafe_allow_html=True,
    )
    year_funnelplot = st.selectbox("Choose an year", options=years_list, index=7, key="funnel_prices")
    st.plotly_chart(
        get_prices_funnelplot(monthly_df=prices_monthly_df, year=year_funnelplot), use_container_width=True
    )

    year_boxplot = st.selectbox("Choose an year", options=years_list, index=7, key="box_prices")
    location_boxplot = st.selectbox(
        "Choose a Nord area",
        areas_list,
    )
    st.plotly_chart(
        get_prices_boxplot(daily_df=prices_daily_df, year=year_boxplot, location=location_boxplot),
        use_container_width=True,
    )

    year_barplot = st.selectbox("Choose an year", options=years_list, index=7, key="bar_prices")
    location_boxplot = st.multiselect(
        "Nord areas of choice",
        options=areas_list,
        default=["SE1", "SE2", "SE3"],
    )
    st.plotly_chart(
        get_prices_barplot(daily_df=prices_daily_df, year=year_barplot, locations=location_boxplot),
        use_container_width=True,
    )

    year_price_diff = st.selectbox("Choose an year", options=years_list, index=7, key="diff_prices")
    area_from, area_to = st.columns(2)
    with area_from:
        area_from_name = st.selectbox(
            "Choose a Nord area",
            areas_list,
            index=0,
            key="diff_prices_from",
        )
    with area_to:
        area_to_name = st.selectbox(
            "Choose a Nord area",
            areas_list,
            index=8,
            key="diff_prices_to",
        )
    if area_from_name is not area_to_name:
        st.plotly_chart(
            get_prices_difference(
                daily_df=prices_daily_df, year=year_price_diff, area_from=area_from_name, area_to=area_to_name
            ),
            use_container_width=True,
        )
    else:
        st.error("Please select different areas for comparison")

    check_off = ["<select>"]

    check_on = st.selectbox("Check out the climate data!", check_off + ["Denmark"], index=0)

    if check_on != "<select>":
        st.write(denmark_climate_df)

        climate_df = denmark_climate_df[(np.abs(stats.zscore(denmark_climate_df)) < 3).all(axis=1)]
        st.subheader(f"Denmark Climate variables heat map!")
        st.pyplot(get_climate_heat_map(climate_df=climate_df))

    list_of_models = ["Prophet", "Nothing Yet"]

    option = st.selectbox("Which algorithm do you like to run?", check_off + list_of_models)

    if option == "Prophet":
        prophet_regressor(df=climate_df)
    elif option == "Nothing Yet":
        st.error("Error: No model exists")


def main():
    run_app()


if __name__ == "__main__":
    main()
