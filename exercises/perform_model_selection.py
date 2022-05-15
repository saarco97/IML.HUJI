from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    epsilon = np.random.normal(0, noise, n_samples)
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(-1.2, 2, n_samples)
    y_ = f(X)
    y = y_ + epsilon

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2.0 / 3)
    train_X = train_X[0].to_numpy()
    test_X = test_X[0].to_numpy()
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    fig = make_subplots(rows=1, cols=1).add_traces([
        go.Scatter(x=X, y=y_, mode='lines', name='true (noiseless) model'),
        go.Scatter(x=train_X, y=train_y, mode='markers', name='train set'),
        go.Scatter(x=test_X, y=test_y, mode='markers', name='test set')
    ]).update_layout(title_text='True Model Vs. train & test sets', height=300)
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = [i for i in range(0, 11)]
    errors = np.array([cross_validate(PolynomialFitting(d), train_X, train_y, mean_square_error) for d in degrees])
    fig = make_subplots(rows=1, cols=1).add_traces([
        go.Scatter(x=degrees, y=errors[:, 0], mode='lines+markers', name='avg training error'),
        go.Scatter(x=degrees, y=errors[:, 1], mode='lines+markers', name='avg validation error')
    ]).update_layout(title_text='Error Values as a function of Polynomial Degree', height=300)
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(errors[:, 1])
    model = PolynomialFitting(k_star)
    model.fit(train_X, train_y)
    test_err = mean_square_error(test_y, model.predict(test_X))
    print(f"k_Star={k_star}, test_Error={round(test_err, 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()  # q.1-3 (part A)
    select_polynomial_degree(noise=0)  # q.4 (part A)
    select_polynomial_degree(n_samples=1500, noise=10)  # q.5 (part A)
    raise NotImplementedError()
