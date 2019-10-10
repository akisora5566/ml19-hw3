"""Functions for training kernel support vector machines."""
import numpy as np
from quadprog_wrapper import solve_quadprog


def polynomial_kernel(row_data, col_data, order):
    """
    Compute the Gram matrix between row_data and col_data for the polynomial kernel.

    :param row_data: ndarray of shape (2, m), where each column is a data example
    :type row_data: ndarray
    :param col_data: ndarray of shape (2, n), where each column is a data example
    :type col_data: ndarray
    :param order: scalar quantity is the order of the polynomial (the maximum exponent)
    :type order: float
    :return: a matrix whose (i, j) entry is the kernel value between row_data[:, i] and col_data[:, j]
    :rtype: ndarray
    """
    #############################################
    # TODO: Insert your code below to implement the polynomial kernel. For full
    # credit, and faster code, you should compute the kernel using matrix
    # operations and have no Python loops.
    #
    # This computation should take around 1--3 lines of code.
    #############################################

    poly= (row_data.T.dot(col_data)+1)** order
    return poly

def rbf_kernel(row_data, col_data, sigma):
    """
    Compute the Gram matrix between row_data and col_data for the Gaussian radial-basis function (RBF) kernel.

    :param row_data: ndarray of shape (2, m), where each column is a data example
    :type row_data: ndarray
    :param col_data: ndarray of shape (2, n), where each column is a data example
    :type col_data: ndarray
    :param sigma: scalar quantity that scales the Euclidean distance inside the exponent of the RBF value
    :type sigma: float
    :return: a matrix whose (i, j) entry is the kernel value between row_data[:, i] and col_data[:, j]
    :rtype: ndarray
    """
    #############################################
    # TODO: Insert your code below to implement the RBF kernel. For full
    # credit, and faster code, you should compute the kernel using matrix
    # operations and have no Python loops.
    #
    # One hint on how to accomplish this is the fact that for vectors x, y
    # (x - y).dot(x - y) = x.dot(x) + y.dot(y) - 2 * x.dot(y)
    #############################################
    m = row_data.shape[1]
    n = col_data.shape[1]
    row = row_data.T.dot(row_data)
    col = col_data.T.dot(col_data)
    part_x = np.dot(np.diag(row), np.ones(m).T) + np.dot(np.ones(n), np.diag(col).T) - 2* row_data.T.dot(col_data)
    return np.exp(- part_x / (2*pow(sigma,2)))

def linear_kernel(row_data, col_data):
    """
    Compute the Gram matrix between row_data and col_data for the linear kernel.

    :param row_data: ndarray of shape (2, m), where each column is a data example
    :type row_data: ndarray
    :param col_data: ndarray of shape (2, n), where each column is a data example
    :type col_data: ndarray
    :return: a matrix whose (i, j) entry is the kernel value between row_data[:, i] and col_data[:, j]
    :rtype: ndarray
    """
    return row_data.T.dot(col_data)


def kernel_svm_train(data, labels, params):
    """
    Train a kernel SVM by solving the dual optimization.

    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param labels: array of length n whose entries are all +1 or -1
    :type labels: array
    :param params: dictionary containing model parameters, most importantly 'kernel', which is either 'rbf',
                    'polynomial', or 'linear'
    :type params: dict
    :return: learned SVM model containing 'support_vectors', 'sv_labels', 'alphas', 'params'
    :rtype: dict
    """
    if params['kernel'] == 'rbf':
        gram_matrix = rbf_kernel(data, data, params['sigma'])
    elif params['kernel'] == 'polynomial':
        gram_matrix = polynomial_kernel(data, data, params['order'])
    else:
        # use a linear kernel by default
        gram_matrix = linear_kernel(data, data)

    # symmetrize to help correct minor numerical errors
    gram_matrix = (gram_matrix + gram_matrix.T) / 2

    n = gram_matrix.shape[0]

    ############################################################################
    # TODO: insert your code here to set up the inputs to the quadratic
    # programming solver
    # You must assign a value to the variables:
    # hessian, weights, eq_coeffs, eq_constants, lower_bounds, and upper_bounds
    # which will be passed to the solver that solves
    #
    # minimize      0.5 x^T (hessian) x - (weights)^T x
    # subject to    (eq_coeffs) x = (eq_constants)
    #   and         (lower_bounds) <= x <= (upper_bounds)
    ##########################################################################

    labels=np.reshape(labels,(n,1))
    y_matrix= np.dot(labels,labels.T) #(n,n)
    hessian= np.multiply(gram_matrix,y_matrix) #(n,n)
    weights = np.ones(n)
    eq_coeffs = np.reshape(labels,(1,-1)) #(1,75)
    eq_constants = np.zeros(1)
    lower_bounds = np.zeros(n)
    upper_bounds = np.ones(n)*params['C'] #(n)
    ###########################################################################
    # Call quadratic program with provided inputs.
    ############################################################################
    alphas = solve_quadprog(hessian, weights, eq_coeffs, eq_constants, None,
                            None, lower_bounds, upper_bounds)
    model = dict()

    # process optimized alphas to only store support vectors and alphas that have nonnegligible support
    tolerance = 1e-6
    sv_indices = alphas > tolerance
    model['support_vectors'] = data[:, sv_indices]
    model['alphas'] = alphas[sv_indices]
    model['params'] = params  # store the kernel type and parameters
    model['sv_labels'] = labels[sv_indices]

    # find all alphas that represent points on the decision margin
    margin_alphas = np.logical_and(
        alphas > tolerance, alphas < params['C'] - tolerance)

    # compute the bias value
    if np.any(margin_alphas):
        model['bias'] = np.mean(
            labels[margin_alphas].T - (alphas * labels).T.dot(gram_matrix[:, margin_alphas]))
    else:
        # there were no support vectors on the margin (this should only happen due to numerical errors)
        model['bias'] = 0
    return model


def kernel_svm_predict(data, model):
    """
    Predict binary-class labels for a batch of test points

    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param model: learned model from kernel_svm_train
    :type model: dict
    :return: array of +1 or -1 labels
    :rtype: array
    """
    if model['params']['kernel'] == 'rbf':
        gram_matrix = rbf_kernel(
            data, model['support_vectors'], model['params']['sigma'])
    elif model['params']['kernel'] == 'polynomial':
        gram_matrix = polynomial_kernel(
            data, model['support_vectors'], model['params']['order'])
    else:
        # use a linear kernel by default
        gram_matrix = linear_kernel(data, model['support_vectors'])

    ########################################################################
    # TODO: Insert your code below to compute the prediction score given the
    # Gram matrix, model.alphas, model.svLabels, and model.bias
    # (You should need no for loops. This can be done in 1--3 lines of code.)
    ########################################################################
    n = gram_matrix.shape[1]   #(25,n)
    alpha_y_i= np.multiply(np.reshape(model['alphas'],(n,1)), model['sv_labels']) #(n,1)
    scores = np.sum(alpha_y_i * gram_matrix.T ,axis=0)+model['bias'] #(25,)

    #####################################################################
    # End of score computation
    #####################################################################

    labels = 2 * (scores > 0) - 1  # threshold and map to {-1, 1}

    return labels, scores
