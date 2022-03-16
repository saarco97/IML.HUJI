from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"
Q1_HEADER = "$\\text{(Q.1) Histogram of Samples }X_{1},...,X_{1000}" \
            "\\stackrel{iid}\\sim\\mathcal{N}\\left(10,1\\right)$"
Q5_HEADER = "$\\underset{\\text{where }f_{1},f_{3}\\text{ get values from }\\left[-10,10\\right]}" \
            "{\\log likelihood\\text{ as a heatmap of models with }\\mu=\\left[f_{1},0,f_{3},0\\right]^{\\top} " \
            "\\text{ and a given cov-matrix}}$"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    print("Q1:\n" + "-" * 5)
    m, bins = 1000, 100
    mu, sigma = 10, 1
    X = np.random.normal(mu, sigma, m)
    u_var_gauss = UnivariateGaussian().fit(X)
    print(f"({u_var_gauss.mu_}, {u_var_gauss.var_})")
    print("=" * 40 + "\n")

    # Plot the results:
    # fig = make_subplots(rows=5, cols=1, specs=[[{"rowspan": 4, "secondary_y": True}], [None], [None], [None], [{}]]) \
    #     .add_trace(go.Histogram(x=X, opacity=0.75, bingroup=1, nbinsx=bins), secondary_y=False) \
    #     .add_trace(go.Histogram(x=X, opacity=0.75, bingroup=1, nbinsx=bins, histnorm="probability density"),
    #                secondary_y=True)
    # fig.update_layout(title_text=Q1_HEADER) \
    #     .update_yaxes(title_text="Number of samples", secondary_y=False, row=1, col=1) \
    #     .update_yaxes(title_text="Density", secondary_y=True, row=1, col=1) \
    #     .update_xaxes(showgrid=False, title_text="Sample Value") \
    #     .update_layout(showlegend=False)
    # fig.show()

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_mean = []
    for m in ms:
        estimated_mean.append(abs(UnivariateGaussian().fit(X[:m]).mu_ - mu))

    fig = make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=ms, y=estimated_mean, mode='lines', fill='tonexty', marker=dict(color="lightgrey"),
                                showlegend=False)]) \
        .update_layout(title_text=r"$\text{(Q.2) Absolute Distance Between The Estimated And True Value Of The "
                                  r"Expectation As Function Of Sample Size}$", height=350, width=900) \
        .update_yaxes(title_text=r"$\left|\hat{\mu}-\mu\right|$") \
        .update_xaxes(title_text=r"Sample Size")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = u_var_gauss.pdf(X)
    fig = make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=X, y=pdfs, mode='markers', marker=dict(color="black"), showlegend=False)]) \
        .update_layout(title_text=r"$\text{(Q.3) Empirical PDF Function Under The Fitted Model}$", height=350,
                       width=900) \
        .update_yaxes(title_text=r"PDF") \
        .update_xaxes(title_text=r"Sample Value")
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    print("Q2:\n" + "-" * 5)
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    S = np.random.multivariate_normal(mu, sigma, 1000)
    m_var_gauss = MultivariateGaussian().fit(S)
    print(m_var_gauss.mu_)
    print(m_var_gauss.cov_)
    print("=" * 40 + "\n")

    # Plot the results:
    # go.Figure() \
    #     .add_trace(go.Histogram2dContour(x=S[:, 0], y=S[:, 1],
    #                                      colorscale='Blues', reversescale=True, xaxis='x', yaxis='y')) \
    #     .add_trace(go.Scatter(x=S[:, 0], y=S[:, 1], xaxis='x', yaxis='y', mode='markers',
    #                           marker=dict(color='rgba(0,0,0,0.3)', size=3))) \
    #     .add_trace(
    #     go.Histogram(y=S[:, 1], histnorm="probability density", xaxis='x2', marker=dict(color='rgba(0,0,0,1)'))) \
    #     .add_trace(
    #     go.Histogram(x=S[:, 0], histnorm="probability density", yaxis='y2', marker=dict(color='rgba(0,0,0,1)'))) \
    #     .update_layout(
    #     xaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
    #     yaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
    #     xaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
    #     yaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
    #     hovermode='closest', showlegend=False,
    #     title=r"$\text{(Q.4) Fitted multivariate Gaussian}$") \
    #     .show()

    # Question 5 - Likelihood evaluation
    func = np.linspace(-10, 10, 200)
    cartesian = (np.array([np.repeat(func, len(func)), np.tile(func, len(func))])).T

    def calculate_log_likelihood(s):
        return MultivariateGaussian.log_likelihood(np.array([s[0], 0, s[1], 0]), sigma, S)

    log_likelihoods = np.apply_along_axis(calculate_log_likelihood, 1, cartesian)
    go.Figure(go.Heatmap(x=cartesian[:, 1], y=cartesian[:, 0], z=log_likelihoods, colorbar={"title": 'log-likelihood'}),
              layout=go.Layout(title=Q5_HEADER, height=550, width=650)). \
        update_yaxes(title_text=r"$f_{1}\ values$"). \
        update_xaxes(title_text=r"$f_{3}\ values$"). \
        show()

    # Question 6 - Maximum likelihood
    print("Q6:\n" + "-" * 5)
    max_i = np.argmax(log_likelihoods)
    max_model = cartesian[max_i]
    print(f"Max value of log-likelihood is: {round(log_likelihoods[max_i], 3)}")
    print(f"This value is achieved by the model "
          f"(f1={round(max_model[0], 3)}, f2={round(max_model[1], 3)})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
