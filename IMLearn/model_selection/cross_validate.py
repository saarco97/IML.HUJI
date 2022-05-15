from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    errors = []
    for i in range(cv):
        X_hat = np.concatenate((X[:i], X[i+1:]))
        y_hat = np.concatenate((y[:i], y[i+1:]))
        h_i = estimator.fit(X_hat, y_hat)
        train_score = scoring(y_hat, h_i.predict(X_hat))
        validation_score = scoring(y[i], h_i.predict(np.array([X[i]])))
        errors.append((train_score, validation_score))
    return tuple(np.mean(np.array(errors), axis=0))

