import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

from sklearn.metrics import roc_curve, auc

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, all_weights = [], []

    def callback(solver, weights, val, grad, t, eta, delta):
        values.append(val)
        all_weights.append(weights)

    return callback, values, all_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = {"L1": L1, "L2": L2}
    for module_name, module in modules.items():
        fig = make_subplots(rows=2, cols=2, x_title="iteration", y_title="norm",
                            subplot_titles=[f"$\eta={e}$" for e in etas],
                            horizontal_spacing=.08, vertical_spacing=.15)
        for i, eta in enumerate(etas):
            f = module(np.copy(init))
            callback, values, weights = get_gd_state_recorder_callback()
            GD = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            GD.fit(f=f, X=None, y=None)
            print(f"{module_name} norm, eta={eta}, lowest_loss={np.round(np.min(values), 4)}")
            plt = plot_descent_path(module, np.vstack(weights), title=f"{module_name} with step size of {eta}")
            # plt.show()
            fig.add_traces([go.Scatter(x=np.arange(len(values)), y=values)],
                           rows=i // 2 + 1, cols=(i % 2) + 1)
        fig.update_layout(title_text=f"{module_name} norm")
        fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = make_subplots(rows=2, cols=2, x_title="iteration", y_title="norm",
                        subplot_titles=[f"gamma={g}" for g in gammas],
                        horizontal_spacing=.08, vertical_spacing=.15)
    for i, gamma in enumerate(gammas):
        f = L1(np.copy(init))
        callback, values, weights = get_gd_state_recorder_callback()
        GD = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback)
        GD.fit(f=f, X=None, y=None)
        print(f"gamma={gamma}, lowest_loss={np.round(np.min(values), 9)}")
        fig.add_traces([go.Scatter(x=np.arange(len(values)), y=values)],
                       rows=i // 2 + 1, cols=(i % 2) + 1)

    # Plot algorithm's convergence for the different values of gamma
    fig.show()

    # Plot descent path for gamma=0.95
    callback, values, weights = get_gd_state_recorder_callback()
    GD = GradientDescent(learning_rate=ExponentialLR(eta, 0.95), callback=callback)
    GD.fit(L1(np.copy(init)), None, None)
    fig = plot_descent_path(L1, np.vstack(weights),
                            title=f"$L1\ with\ \eta={eta}\ and\ \gamma={0.95}$")
    fig.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression()
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    y_prob = model.predict_proba(X_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()
    alpha_star_idx = np.argmax(tpr - fpr)
    alpha_star = thresholds[alpha_star_idx]
    print(f"best alpha = {alpha_star}")
    model.alpha_ = alpha_star
    test_err = model.loss(X_test.to_numpy(), y_test.to_numpy())
    print(f"Under that alpha, model's test error is {test_err}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    def questions_10_11(penalty):
        from IMLearn.model_selection import cross_validate
        from IMLearn.metrics import misclassification_error
        lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        errors = np.array(
            [cross_validate(LogisticRegression(penalty=penalty, alpha=0.5, lam=lam), X_train.to_numpy(),
                            y_train.to_numpy(), misclassification_error)
             for lam in lambdas])
        best_lambda = lambdas[np.argmin(errors[:, 1])]
        logistic_model = LogisticRegression(penalty=penalty, alpha=0.5, lam=best_lambda)
        logistic_model.fit(X_train.to_numpy(), y_train.to_numpy())
        print(f"lambda = {best_lambda}")
        error = logistic_model.loss(X_test.to_numpy(), y_test.to_numpy())
        print(f"Under that lambda, model's test error is {error}")

    questions_10_11("l1")
    questions_10_11("l2")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
