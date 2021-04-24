import numpy as np


def computeBilinerWeights(q):
    q[0] = q[0] - np.floor(q[0])
    q[1] = q[1] - np.floor(q[1])
    weights = [(1-q[0])*(1-q[1]), q[0]*(1-q[1]), (1-q[0])*q[1], q[0]*q[1]]
    return weights


def computeGaussianWeights(winsize, sigma):
    x_c, y_c = ((winsize[0] - 1) / 2, (winsize[1] - 1) / 2)
    x_axis = (x_c - np.linspace(0, winsize[0]-1, winsize[0]))/winsize[0]
    y_axis = (y_c - np.linspace(0, winsize[1]-1, winsize[1]))/winsize[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    weights = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return np.array(weights)


def invertMatrix2x2(A):
    return 1/(A[0, 0]*A[1, 1]-A[0, 1]*A[1, 0])*np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])

