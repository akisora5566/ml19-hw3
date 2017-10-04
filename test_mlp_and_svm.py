"""Test class that trains each model type on one or more dataset and evaluates performance."""
from __future__ import division
import unittest
import numpy as np
import pylab as plt
from mlp import mlp_train, mlp_predict, logistic, nll, mlp_objective
from kernelsvm import kernel_svm_train, kernel_svm_predict
from scipy.io import loadmat
from plotutils import plot_data, plot_surface
from scipy.optimize import check_grad, approx_fprime
import copy


def objective_wrapper(x, data, labels, params):
    """
    Wrapper for mlp_objective that takes in a vector and reshapes it into the MLP weights structure
    :param x: weights squeezed into a single vector
    :type x: array
    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param labels: length-n array of labels in {+1, -1}
    :type labels: array
    :param params: dict containing the MLP options
    :type params: dict
    :return: tuple containing (1) the scalar loss objective value and (2) the gradient vector
    :rtype: tuple
    """
    # test derivative of mlp objective
    num_hidden_units = params['num_hidden_units']
    input_dim = data.shape[0]

    index = 0
    model = dict()
    model['weights'] = list()
    # create input layer
    curr_matrix_size = num_hidden_units[0] * (input_dim + 1)
    model['weights'].append(x[index:index + curr_matrix_size].reshape((num_hidden_units[0], input_dim + 1)))
    index += curr_matrix_size
    # create intermediate layers
    for layer in range(1, len(num_hidden_units)):
        curr_matrix_size = num_hidden_units[layer] * num_hidden_units[layer - 1]
        model['weights'].append(x[index:index + curr_matrix_size].reshape(
            (num_hidden_units[layer], num_hidden_units[layer - 1])))
        index += curr_matrix_size
    # create output layer
    curr_matrix_size = num_hidden_units[-1]
    model['weights'].append(x[index:].reshape((1, num_hidden_units[-1])))

    model['squash_function'] = params['squash_function']

    obj, gradients = mlp_objective(model, data, labels, params['loss_function'])

    grad_vec = np.concatenate([g.ravel() for g in gradients])

    return obj, grad_vec


class MlpSvmTest(unittest.TestCase):
    """Test class that trains each model type on one or more dataset and evaluates performance."""
    def setUp(self):
        """load synthetic binary-class data from MATLAB data file"""

        variables = dict()
        loadmat('syntheticData.mat', variables)

        # use some list comprehensions to clean up MATLAB data conversion
        self.train_labels = [vector[0].ravel() for vector in variables['trainLabels']]
        self.train_data = [matrix[0] for matrix in variables['trainData']]
        self.test_labels = [vector[0].ravel() for vector in variables['testLabels']]
        self.test_data = [matrix[0] for matrix in variables['testData']]

    def test_mlp(self):
        """
        Train MLP on dataset 1, which is not linearly separable. Tests will try multiple random initializations to
        reduce chance of bad initialization.
        """
        i = 1

        num_hidden_units = [4, 5]

        params = {
            'max_iter': 400,
            'squash_function': logistic,
            'loss_function': nll,
            'num_hidden_units': num_hidden_units,
            'lambda': 0.01
        }

        input_dim = self.train_data[i].shape[0] + 1
        total_weight_length = input_dim * num_hidden_units[0]
        for j in range(len(num_hidden_units) - 1):
            total_weight_length += num_hidden_units[j] * num_hidden_units[j + 1]
        total_weight_length += num_hidden_units[-1]

        # try at most 10 random initializations
        for trial in range(10):
            mlp_model = mlp_train(self.train_data[i], self.train_labels[i], params)
            predictions, _, _, _ = mlp_predict(self.test_data[i], mlp_model)
            accuracy = np.mean(predictions == self.test_labels[i])
            print("On trial %d, 3-layer MLP had test accuracy %2.3f "
                  "(should be around 0.95, depending on random initialization)" %
                  (trial, accuracy))

            if accuracy > 0.9:
                return

        assert False, "Accuracy was never above 0.9. Could be a bug, or could be bad luck. " \
                      "Try running again to check."

    def test_linear_svm(self):
        """
        Train linear SVM on dataset 0, which is linearly separable. Then try on dataset 3, which is not separable, but
        close to linearly separable.
        """
        i = 0
        params = {'kernel': 'linear', 'C': 1.0}

        lin_svm_model = kernel_svm_train(self.train_data[i], self.train_labels[i], params)
        predictions, _ = kernel_svm_predict(self.test_data[i], lin_svm_model)
        test_accuracy = np.mean(predictions == self.test_labels[i])

        print("Linear SVM had test accuracy %2.3f (should be around 0.98)" %
              test_accuracy)
        assert test_accuracy > 0.95, "Accuracy was below 0.95."

        i = 3
        params = {'kernel': 'linear', 'C': 1.0}

        lin_svm_model = kernel_svm_train(self.train_data[i], self.train_labels[i], params)
        predictions, _ = kernel_svm_predict(self.test_data[i], lin_svm_model)
        test_accuracy = np.mean(predictions == self.test_labels[i])

        print("Linear SVM had test accuracy %2.3f (should be around 0.93)" %
              test_accuracy)
        assert test_accuracy > 0.9, "Accuracy was below 0.9."

    def test_poly_svm(self):
        """
        Train quadratic polynomial SVM on dataset 1, which is not linearly separable. 
        """
        i = 1
        params = {'kernel': 'polynomial', 'C': 1.0, 'order': 2}

        poly_svm_model = kernel_svm_train(self.train_data[i], self.train_labels[i], params)
        predictions, _ = kernel_svm_predict(self.test_data[i], poly_svm_model)
        test_accuracy = np.mean(predictions == self.test_labels[i])

        print("Polynomial SVM had test accuracy %2.3f (should be around 0.94)" %
              test_accuracy)
        assert test_accuracy > 0.9, "Accuracy was below 0.9."

    def test_rbf_svm(self):
        """
        Train RBF SVM on dataset 1, which is not linearly separable. 
        """
        i = 1
        params = {'kernel': 'rbf', 'C': 1.0, 'sigma': 0.2}

        rbf_svm_model = kernel_svm_train(self.train_data[i], self.train_labels[i], params)
        predictions, _ = kernel_svm_predict(self.test_data[i], rbf_svm_model)
        test_accuracy = np.mean(predictions == self.test_labels[i])

        print("RBF SVM had test accuracy %2.3f (should be around 0.92)" %
              test_accuracy)
        assert test_accuracy > 0.9, "Accuracy was below 0.9."

if __name__ == '__main__':
    unittest.main()
