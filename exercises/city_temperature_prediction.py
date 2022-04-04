import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    temp_df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    temp_df = temp_df.drop_duplicates().dropna()
    temp_df = temp_df[temp_df["Temp"] >= - 15]
    dates = pd.to_datetime(temp_df["Date"], errors='coerce').dt.to_period('h')
    temp_df["DayOfYear"] = dates.dt.day_of_year
    temp_df["Year"] = temp_df["Year"].astype(str)
    return temp_df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel = df[df["Country"] == "Israel"]
    # df_israel["Year"] = df_israel["Year"].apply(str)
    df_israel_avg = df_israel.groupby(["Year", "DayOfYear"], as_index=False)["Temp"].mean()
    fig = px.scatter(df_israel_avg, x="DayOfYear", y="Temp", color="Year",
                     title="Average daily temperature as a function of the DayOfYear")
    fig.show()

    df_israel_months = df_israel.groupby(["Month"], as_index=False)["Temp"].std()
    fig = px.bar(df_israel_months, x='Month', y='Temp',
                 labels={'Temp': 'std'},
                 title="Standard Deviation Of The Daily Temperatures Over Months")
    fig.show()

    # Question 3 - Exploring differences between countries
    df_3 = df.groupby(["Country", "Month"], as_index=False)["Temp"].agg({
        'avgTemp': 'mean',
        'std': 'std'
    })
    fig = px.line(df_3, x="Month", y="avgTemp", color="Country", error_y="std",
                  labels={'avgTemp': 'Average Temperature'},
                  title="Average monthly temperature as a function of the Month")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df_israel["DayOfYear"], df_israel["Temp"])
    losses = {'k': [], 'test_error': [], 'test_error_rounded': []}
    for k in list(range(1, 11)):
        model = PolynomialFitting(k).fit(train_X, train_y)
        error = model.loss(test_X, test_y)
        losses['k'].append(k)
        losses['test_error'].append(error)
        losses['test_error_rounded'].append(round(error, 2))
        print(f"Degree k={k}, test error: {round(error, 2)}")
    fig = px.bar(losses, x='k', y='test_error', text='test_error_rounded',
                 title="Test Error as a function of Polynomial Degree")
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(k=5).fit(df_israel["DayOfYear"], df_israel["Temp"])
    countries = ["South Africa", "Jordan", "The Netherlands"]
    losses = {'Country': [], 'test_error': []}
    for c in countries:
        df_cur_country = df[df["Country"] == c]
        losses['Country'].append(c)
        losses['test_error'].append(model.loss(df_cur_country["DayOfYear"], df_cur_country["Temp"]))
    fig = px.bar(losses, x='Country', y='test_error', title="Temperature Over Months")
    fig.show()
