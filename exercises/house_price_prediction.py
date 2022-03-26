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
# id : A notation for a house
# date: Date house was sold
# lat: Latitude coordinate
# long: Longitude coordinate

MUST_BE_POSITIVE = ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]
MUST_BE_NOT_NEGATIVE = ["bathrooms", "floors", "sqft_basement", "yr_renovated"]

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

    houses_df["is_renovated"] = np.where(houses_df["yr_renovated"] >=
                                         np.percentile(houses_df.yr_renovated.unique(), 50), 1, 0)
    houses_df = houses_df.drop(columns="yr_renovated")

    # TODO - look on this
    # houses_df["decade_built"] = (houses_df["yr_built"] / 10).astype(int)
    # houses_df = houses_df.drop(columns="yr_built")

    # TODO - decide what about zipcode
    houses_df = pd.get_dummies(houses_df, prefix='zipcode_', columns=['zipcode'])
    # houses_df = pd.get_dummies(houses_df, prefix='decade_built_', columns=['decade_built'])

    # There are some rows that are really unusual so we want to remove them
    houses_df = houses_df[houses_df["bedrooms"] < 15]
    houses_df = houses_df[houses_df["sqft_lot"] < 1200000]
    houses_df = houses_df[houses_df["sqft_lot15"] < 600000]

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

    X_no_intercept = X.drop(columns="intercept")

    for feature in X_no_intercept:
        cov = np.cov(X_no_intercept[feature], y)[0, 1]
        deviations_mult = (np.std(X_no_intercept[feature]) * np.std(y))
        pearson = cov / deviations_mult
        figure = px.scatter(pd.DataFrame({'x_axis': X_no_intercept[feature], 'y_axis': y}),
                            x="x_axis", y="y_axis", trendline="ols",
                            title=f"Correlation Between {feature} Values and Response "
                                  f"<br>Pearson Correlation: {pearson}",
                            labels={"x": f"Values of {feature}", "y": "Responses"})
        figure.write_image(f"{output_path}/correlation_{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y, "./correlation_graphs")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    results = []
    for i in range(10, 101):
        for time in range(10):
            n = round(len(train_y) * (i / 100))
            model = LinearRegression(include_intercept=False).fit(train_X[:n], train_y[:n])
            results.append(model.loss(test_X, test_y))

    fig = go.Figure(go.Scatter(x=sorted(list(range(10, 101)) * 10), y=results, mode="markers"),
                    layout=go.Layout(title="Model's MSE As A Function Of Portions Of Training data",
                                     xaxis=dict(title="Percentage of Training data"),
                                     yaxis=dict(title="MSE Over Test data")))
    fig.show()
