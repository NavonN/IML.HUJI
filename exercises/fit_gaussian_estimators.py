import matplotlib.pyplot as plt
import numpy.random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = numpy.random.normal(10, 1, 1000)
    res = UnivariateGaussian()
    res.fit(X)
    print(res.mu_, res.var_)

    # Question 2 - Empirically showing sample mean is consistent
    q2_x_axis = np.arange(start=10, stop=1010, step=10).reshape(100,)
    q2_y_axis = np.zeros(q2_x_axis.shape)
    for i in range(100):
        q2_y_axis[i] = res.fit(X[:q2_x_axis[i]]).mu_
    q2_y_axis = np.abs(q2_y_axis-10)
    plot_emp = go.Figure()
    plot_emp.add_trace(go.Scatter(x=q2_x_axis, y=q2_y_axis, mode='lines'))
    plot_emp.update_layout(
        title="Absolute Distance of est. Mean and Real Mean vs Number of Samples",
        xaxis_title="Number of Samples"
        , yaxis_title="Absolute Distance")
    plot_emp.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    q3_y_axis = res.pdf(X)
    plot_scat = go.Figure()
    plot_scat.add_trace(go.Scatter(x=X, y=q3_y_axis, mode='markers'))
    plot_scat.update_layout(title="PDF under Fitted Model ", xaxis_title="Sample"
                            , yaxis_title="PDF per sample")
    plot_scat.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    MU = np.array([0, 0, 4, 0])
    SIGMA = np.array([1,0.2,0,0.5,0.2,2,0,0,0,0,1,0,0.5,0,0,1]).reshape(4,4)

    X = np.random.multivariate_normal(MU, SIGMA, 1000)
    res = MultivariateGaussian().fit(X)
    print(res.mu_)
    print(res.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10,10,200)
    f3 = np.linspace(-10, 10, 200)

    heat_map = np.zeros((200,200))
    for i in range(200):
        for j in range(200):
            heat_map[i][j] = MultivariateGaussian.log_likelihood(np.array([f1[i],0,f3[j],0]),
                                                                 SIGMA,X)
    plot_heat = go.Figure(data=go.Heatmap(x=f1, y=f3,z=heat_map))
    plot_heat.update_layout(title="Log Likelihood as a Function of Expected Value", xaxis_title="F1 Value"
                            , yaxis_title="F3 Value")
    plot_heat.show()

    # Question 6 - Maximum likelihood
    indices = np.where(heat_map == heat_map.max())
    print("Max Value is at F1: ", f1[indices[0]], " F2: ", f3[indices[1]])

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
