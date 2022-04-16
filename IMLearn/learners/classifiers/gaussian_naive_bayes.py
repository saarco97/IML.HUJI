from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.array([X[y == i].mean(axis=0) for i in self.classes_]).T

        # TODO
        covs = []
        n_features = X.shape[1]
        for i in self.classes_:
            new_cov = []
            for j in range(n_features):
                mat = X[y == i][:, j] - self.mu_[j][i]
                new_cov.append((mat @ mat.T) / X[y == i].shape[0])
            covs.append(new_cov)

        self.vars_ = np.array(covs)
        self.pi_ = np.array([len(X[y == i]) for i in self.classes_]) / y.size

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.argmax(self._log_likelihood(X), axis=1)

    def _log_likelihood(self, X):
        log_likelihood = []
        for i in range(self.classes_.size):
            temp = -1 * np.sum(np.log(2 * np.pi * self.vars_[i, :]))
            temp -= 0.5 * np.sum(((X - self.mu_.T[i, :]) ** 2), 1)
            log_likelihood.append(np.log(self.pi_[i]) + temp)
        return np.array(log_likelihood).T

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        # TODO
        # gaussian_x = (np.exp(-0.5 * (X - self.mu_).T @ np.linalg.inv(self.vars_) @ (X - self.mu_))) / (
        #         2 * np.pi ** 0.5 * np.linalg.det(self.cov_))
        # return np.prod(gaussian_x * self.pi_)
        return np.exp(self._log_likelihood(X), axis=1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
