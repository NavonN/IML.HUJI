from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


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
    X_dFrame = pd.read_csv(filename)

    # lose duplicate id's
    X_dFrame = X_dFrame.drop_duplicates(subset=['id'])

    # lose ID, zipcode, lat, long
    X_dFrame = X_dFrame.drop(['id', 'zipcode', 'lat', 'long'], axis=1)

    # lose duplicate rows
    X_dFrame = X_dFrame.drop_duplicates()

    # put boundries on all colums
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.price < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.bedroom < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.bathroom < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.sqft_living < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.sqft_lot < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.floors < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.waterfront < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.view < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.view > 4].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.condition < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.grade < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.sqft_above < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.sqft_basement < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.yr_built > 2022].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.yr_renovated < X_dFrame.yr_built].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.yr_renovated < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.sqft_living15 < 0].index)
    X_dFrame = X_dFrame.drop(X_dFrame[X_dFrame.sqft_lot15 < 0].index)
    y = pd.Series(X_dFrame.filter(items=['price']))

    return tuple(X_dFrame, y)


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
    top = X.apply(lambda column: y.cov(column))
    bottom = X.std(axis=0) * y.std()
    pc = top.divide(bottom) #todo: Check the division
        





    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
