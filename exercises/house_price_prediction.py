from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

NOT_KEEP_FEATURES = ["id", "date", "lat", "long"]
MUST_BE_POSITIVE = ["price", "sqft_living", "sqft_lot", "sqft_above",
                    "yr_built", "sqft_living15", "sqft_lot15"]
MUST_BE_NOT_NEGATIVE = ["bathrooms", "floors", "sqft_basement", "yr_renovated"]
MAX_SQFT_LOT15 = 600000
MAX_SQFT_LOT = 1200000
MAX_BEDROOM_NUM = 15
THRESHOLD_RECENT_RENO = 75


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    houses_df = pd.read_csv(filename).drop_duplicates().dropna()

    # drop unnecessary features
    for col in NOT_KEEP_FEATURES:
        houses_df = houses_df.drop(columns=col)
    for col in MUST_BE_NOT_NEGATIVE:
        houses_df = houses_df[houses_df[col] >= 0]
    for col in MUST_BE_POSITIVE:
        houses_df = houses_df[houses_df[col] > 0]

    houses_df.insert(0, 'intercept', 1, True)

    houses_df["was_recently_reno"] = np.where(
        houses_df["yr_renovated"] >=
        np.percentile(houses_df.yr_renovated.unique(), THRESHOLD_RECENT_RENO), 1, 0)
    houses_df = houses_df.drop(columns="yr_renovated")

    min_year = houses_df["yr_built"].min()
    houses_df["rank_built"] = houses_df["yr_built"] - min_year
    houses_df = houses_df.drop(columns="yr_built")

    houses_df["zipcode"] = houses_df["zipcode"].astype(int)
    houses_df = pd.get_dummies(houses_df, prefix='zip_', columns=['zipcode'])

    # There are some rows that are really unusual so we want to remove them
    houses_df = houses_df[houses_df["bedrooms"] < MAX_BEDROOM_NUM]
    houses_df = houses_df[houses_df["sqft_lot"] < MAX_SQFT_LOT]
    houses_df = houses_df[houses_df["sqft_lot15"] < MAX_SQFT_LOT15]

    return houses_df.drop(columns="price"), houses_df["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X_work = X.loc[:, ~(X.columns.str.contains('^zip_', case=False))].\
        drop(columns="intercept")
    for feature in X_work:
        cov = np.cov(X_work[feature], y)[0, 1]
        deviations_mult = (np.std(X_work[feature]) * np.std(y))
        pearson = cov / deviations_mult
        figure = px.scatter(pd.DataFrame({f"{feature} Values": X_work[feature], 'Response Values': y}),
                            x=f"{feature} Values", y="Response Values", trendline="ols",
                            title=f"Correlation Between {feature} Values and Response "
                                  f"<br>Pearson Correlation: {pearson}",
                            labels={"x": f"Values of {feature}", "y": "Responses"})
        figure.write_image(f"{output_path}/correlation_{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "./correlation_graphs")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    results = np.empty((91, 10))
    for i in range(10, 101):
        for j in range(10):
            X = train_X.sample(frac=(i / 100))
            y = train_y.reindex_like(X)
            model = LinearRegression(include_intercept=False).fit(X, y)
            results[i - 10][j] = model.loss(test_X, test_y)

    mean_pred, std_pred = np.mean(results, axis=1), np.std(results, axis=1)

    p = list(range(10, 101))
    fig = go.Figure(go.Scatter(x=p, y=mean_pred, mode="markers+lines", name="Predicted Loss"),
                    layout=go.Layout(title="Model's MSE As A Function Of Portions Of Training data",
                                     xaxis=dict(title="Percentage of Training data"),
                                     yaxis=dict(title="MSE Over Test data"))). \
        add_traces([go.Scatter(x=p, y=mean_pred - 2 * std_pred, fill=None, mode="lines",
                               line=dict(color="lightgrey"), showlegend=False),
                    go.Scatter(x=p, y=mean_pred + 2 * std_pred, fill='tonexty', mode="lines",
                               line=dict(color="lightgrey"), showlegend=False)])
    fig.show()
