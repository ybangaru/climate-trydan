import os
import pandas as pd  # type: ignore
import numpy as np

import matplotlib.pyplot as plt  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
import streamlit as st
from typing import List, Tuple
from datetime import datetime as dt
from datetime import timedelta
from prophet import Prophet  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore


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


def run_app_nordpool():

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


def run_app_opsd():
    def get_climate_heat_map(climate_df: pd.DataFrame):
        corr = climate_df.corr()
        fig = plt.figure()
        sns.heatmap(corr, cmap="Blues", annot=True)
        return fig

    def prophet_regressor(df: pd.DataFrame):

        st.header("Time series predictions!")
        list_of_yhats = ["Load", "Price"]
        option_yhat = st.selectbox("Which variable do you want to predict?", list_of_yhats, index=1)

        if option_yhat == "Load":
            pred_var_label = "DK_load_actual_entsoe_transparency"
            pred_var_plot_label = "Load (Watts)"
        elif option_yhat == "Price":
            pred_var_label = "DK_price_day_ahead"
            pred_var_plot_label = "Price Day Ahead (Euros)"

        start_time_value = max(min(df.index), dt.fromisoformat("2018-01-01"))
        end_time_value = min(max(df.index), dt.fromisoformat("2019-12-31 23:00:00"))

        start_time = st.date_input("Start date", start_time_value)
        end_time = st.date_input("End date", end_time_value)

        start_time = max(min(df.index), start_time)
        end_time = min(max(df.index), end_time)

        num_days_for_test = st.number_input("Predict for how many days?", value=14)

        df_app = df.loc[start_time:end_time]
        df_app["ds"] = df_app.index
        df_app.rename({pred_var_label: "y"}, axis=1, inplace=True)
        test_start_time = end_time - timedelta(days=num_days_for_test)

        df_train = df_app.loc[df_app["ds"] < np.datetime64(test_start_time)]
        df_test = df_app.loc[df_app["ds"] >= np.datetime64(test_start_time)]

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        for c in df_app.columns.to_list():
            if c not in ["ds", "y"]:
                m.add_regressor(c)
        m.fit(df_train)

        forecast = m.predict(df_test.drop(columns="y"))

        forecast.index = forecast["ds"]

        fig3 = m.plot(forecast, uncertainty=True, xlabel="Date", ylabel=pred_var_plot_label)
        ax = fig3.gca()
        ax.set_xlim(pd.to_datetime([test_start_time - timedelta(days=num_days_for_test), end_time]))
        st.subheader(f"Forecasted values for {num_days_for_test} days")
        st.pyplot(fig3)

        st.subheader("Model evaluation!")
        evaluation_df = pd.DataFrame({"y_pred": forecast["yhat"].round(2).tolist(), "y_true": df_test["y"].tolist()})
        st.write("MAE: %.3f" % mean_absolute_error(evaluation_df["y_true"], evaluation_df["y_pred"]))

        st.write(pd.merge(df_test[["y"]].head(10), forecast[["yhat"]].head(10), left_index=True, right_index=True))

        y_true = df_test["y"].values
        y_pred = forecast["yhat"].values

        fig1 = plt.figure(figsize=(10, 6))
        plt.plot(df_test.index, y_true, label="Actual")
        plt.plot(df_test.index, y_pred, label="Predicted")
        plt.xticks(rotation=90)
        plt.legend()
        plt.xlabel("Timestamp")
        plt.ylabel(pred_var_plot_label)

        st.subheader(f"Actual vs Predicted {pred_var_plot_label}!")
        st.pyplot(fig1)

        fig2 = m.plot_components(forecast)
        st.subheader("Trend components")
        st.pyplot(fig2)

    st.markdown(
        "<h1 style='text-align: center; color: #00CC96;'>Modelling and visulalizations!</h1>",
        unsafe_allow_html=True,
    )

    check_off = ["<select>"]

    check_on = st.selectbox("Check out the climate data!", check_off + ["Denmark"], index=0)

    if check_on != "<select>":
        st.write(denmark_climate_df)

        climate_df = denmark_climate_df[(np.abs(stats.zscore(denmark_climate_df)) < 3).all(axis=1)]
        st.subheader("Denmark Climate variables heat map!")
        st.pyplot(get_climate_heat_map(climate_df=climate_df))

    list_of_models = ["Prophet", "Nothing Yet"]

    st.subheader("Time series predictions!")
    option = st.selectbox("Which algorithm do you like to run?", check_off + list_of_models)

    if option == "Prophet":
        prophet_regressor(df=climate_df)
    elif option == "Nothing Yet":
        st.error("Error: No model exists")


def main():

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 230px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 500px;
            margin-left: -230px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    mode = st.sidebar.selectbox(
        "",
        [
            "NordPool!",
            "OpenPowerSystems!",
        ],
    )
    if mode == "NordPool!":
        run_app_nordpool()

    elif mode == "OpenPowerSystems!":
        run_app_opsd()


if __name__ == "__main__":
    main()
