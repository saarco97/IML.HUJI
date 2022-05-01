import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)
    iterations = np.arange(1, n_learners)
    training_loss, test_loss = [], []
    for t in iterations:
        training_loss.append(adaBoost.partial_loss(train_X, train_y, t))
        test_loss.append(adaBoost.partial_loss(test_X, test_y, t))
    fig = make_subplots(rows=1, cols=1).add_traces([
        go.Scatter(x=iterations, y=training_loss, mode='lines', name='train error'),
        go.Scatter(x=iterations, y=test_loss, mode='lines', name='test error')
    ]).update_layout(title_text='AdaBoost Algorithm Loss', height=300)
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"num iterations = {t}" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        # TODO - I need to put here partial predict with given t but..
        model = AdaBoost(DecisionStump, t)
        model.fit(train_X, train_y)
        fig.add_traces(
            [decision_surface(model.predict, lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                    line=dict(color="black", width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
        # fig.add_traces([decision_surface(adaBoost.fit(train_X, train_y).partial_predict, lims[0], lims[1], showscale=False),
        #                 go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
        #                            marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
        #                                        line=dict(color="black", width=1)))],
        #                rows=(i // 3) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of Models}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    test_error = []
    for t in iterations:
        test_error.append(adaBoost.partial_loss(test_X, test_y, t))
    t_min = np.argmin(test_error) + 1
    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"num iterations = {t_min}" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    model = AdaBoost(DecisionStump, t_min)
    model.fit(train_X, train_y)
    fig.add_traces(
        [decision_surface(adaBoost.predict, lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))])
    fig.show()

    # Question 4: Decision surface with weighted samples
    weights = adaBoost.D_ / np.max(adaBoost.D_) * 5
    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"num iterations = {n_learners}"],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces(
        [decision_surface(adaBoost.predict, lims[0], lims[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=train_y, colorscale=[custom[0], custom[-1]], size=weights,
                                line=dict(color="black", width=1)))])
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
