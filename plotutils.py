"""Utility functions for plotting data and prediction surfaces"""
import pylab as plt
import numpy as np


def plot_data(data, labels):
    """
    Plot binary-class 2-dimensional data
    
    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param labels: array of length n, where each entry is either 1 or -1
    :type labels: array
    :return: None
    """
    positive_class = labels == +1
    plt.plot(data[0, positive_class], data[1, positive_class], 'rx')
    plt.plot(data[0, ~positive_class], data[1, ~positive_class], 'bo')


def plot_surface(predictor, model, datapoints):
    """
    Plot the decision surface of a predictor for 2-dimensional data.
    
    :param predictor: predictor function that returns a tuple whose second entry is a continuous prediction score.
    :param model: model parameters for the predictor
    :param datapoints: ndarray of shape (2, n), where each column is a data example. Used to set range of inputs.
    :return: 
    """
    res = 100

    xmin = np.min(datapoints[0, :])
    xmax = np.max(datapoints[0, :])
    ymin = np.min(datapoints[1, :])
    ymax = np.max(datapoints[1, :])

    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)

    x_grid, y_grid = np.meshgrid(x, y)

    z_grid = np.zeros(x_grid.shape)

    data = np.zeros((2, res * res))

    # create matrix of all grid points
    c = 0
    for i in range(res):
        for j in range(res):
            data[0, c] = x[i]
            data[1, c] = y[j]
            c += 1

    # score all grid points with the model
    scores = predictor(data, model)
    if isinstance(scores, tuple):
        scores = scores[1]

    z_grid = scores.reshape((res, res)).T

    # plot
    plt.pcolor(x_grid, y_grid, z_grid, cmap='coolwarm', vmin=-0.5, vmax=1.5)
    plt.contour(x_grid, y_grid, z_grid, levels=[0.5])

