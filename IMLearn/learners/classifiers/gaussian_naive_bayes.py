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

        self.vars_ = []
        n_features = X.shape[1]
        for i in self.classes_:
            self.vars_.append(
                [(X[y == i][:, j] - self.mu_[j][i]) @ (X[y == i][:, j] - self.mu_[j][i]).T / X[y == i].shape[0]
                 for j in range(n_features)])

        self.vars_ = np.array(self.vars_)
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
        log_likelihood = np.array([self._log_likelihood_for_class_i(X, i) for i in range(self.classes_.size)]).T
        return np.argmax(log_likelihood, axis=1)

    def _log_likelihood_for_class_i(self, X, i):
        mu_k = self.mu_.T[i, :]
        cov_k = np.diag(self.vars_[i])
        intercept = -0.5 * mu_k @ cov_k @ mu_k + np.log(self.pi_[i])
        elem = X @ cov_k @ mu_k + intercept
        return elem

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

        d = X.shape[1]
        likelihoods = []
        for c in range(self.classes_.size):
            x_mu = X - self.mu_.T[c, :]
            exp = np.exp(-0.5 * x_mu @ self.vars_[c, :] @ x_mu)
            Z = np.sqrt((2 * np.pi) ** d * np.prod(self.vars_[c, :]))
            likelihoods.append(self.pi_[c] * exp / Z)
        return np.array(likelihoods).T

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
