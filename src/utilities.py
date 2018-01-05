import numpy as np
from math import ceil
from scipy.special import psi, gammaln


def sample_batches(n, batch_size):
    """

    :param n: total number of things to draw from (e.g. number of documents in corpus)
    :param batch_size: how many documents should appear in each batch
    :return: list of lists of batches.
    """
    indices = np.arange(n)
    np.random.shuffle(indices)
    num_splits = int(ceil(1.0 * n / batch_size))
    rval = [arr.tolist() for arr in np.array_split(indices, num_splits)]
    return rval


def softmax(x, axis):
    """
    Softmax for normalizing a matrix along an axis
    Use max substraction approach for numerical stability
    :param x:
    :return:
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def sum_normalize(mat, axis):
    """
    normalize to a probability distribution, first convert any
    negative numbers if they exist
    :param mat:
    :param axis:
    :return:
    """
    pos_mat = mat - np.min(mat, axis=axis, keepdims=True) + 0.001
    return pos_mat / np.sum(pos_mat, axis=axis, keepdims=True)


def dirichlet_expectation(dirichlet_parameter):
    """
    compute dirichlet expectation
    :param dirichlet_parameter:
    :return:
    """
    if len(dirichlet_parameter.shape) == 1:
        return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter))
    return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter, axis=1, keepdims=True))


def matrix2str(mat, num_digits=2):
    """
    take a matrix (either list of lists of numpy array) and put it in a
    pretty printable format.
    :param mat: matrix to print
    :param num_digits: how many significant digits to show
    :return:
    """
    rval = ''
    for row in mat:
        s = '{:.' + str(num_digits) + '}'
        # rval += '\t'.join([s.format(round(elt, num_digits)) for elt in row]) + '\n'
        fpad = ['' if round(elt, num_digits) < 0 else ' ' for elt in row]
        bpad = [' ' * (7 - len(str(np.abs(round(elt, num_digits))))) for elt in row]
        rval += ''.join([f + s.format(round(elt, num_digits)) + b for elt, f, b in zip(row, fpad, bpad)]) + '\n'
    return rval