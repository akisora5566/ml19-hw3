"""Quadratic programming wrapper for quadprog"""
import quadprog
import numpy as np


def solve_quadprog(hessian, weights, eq_coeffs, eq_constants, ineq_coeffs, ineq_constants, lower_bounds, upper_bounds):
    """
    Wrapper for quadprog (https://pypi.python.org/pypi/quadprog/) to make it more compatible with standard
    quadratic program constraint types. This function will solve
        minimize      x^T (hessian) x - (weights)^T x
        subject to  (eq_coeffs) x = (eq_constants)
        and         (ineq_coeffs) x <= (ineq_constants)
        and         (lower_bounds) <= x <= (upper_bounds)

    Note: when given a poorly conditioned problem, this wrapper will catch a failure exception thrown by quadprog,
    print a warning to the console, and return the all-zeros solution.

    :param hessian: Positive definite matrix of shape (n, n) of quadratic coefficeints for the objective
    :type hessian: ndarray
    :param weights: length-n vector of linear coefficents for the objective
    :type weights: array
    :param eq_coeffs: n_eq by n Matrix of linear coefficients for the equality constraints
    :type eq_coeffs: ndarray
    :param eq_constants: length n_eq vector of constants for equality constraints
    :type eq_constants: array
    :param ineq_coeffs: n_ineq by n Matrix of linear coefficients for the inequality constraints
    :type ineq_coeffs: ndarray
    :param ineq_constants: length n_ineq vector of constants for inequality constraints
    :type ineq_constants: array
    :param lower_bounds: length-n vector of lower bounds for variables
    :type lower_bounds: array
    :param upper_bounds: length-n vector of upper bounds for the variables
    :type upper_bounds: array
    :return: length-n solution
    :rtype: array
    """
    # add inequality constraints to capture upper and lower bounds
    n = weights.size

    new_ineq_coeffs = np.vstack((np.eye(n), -np.eye(n)))
    new_ineq_constants = np.zeros(2 * n)
    new_ineq_constants[:n] = lower_bounds
    new_ineq_constants[n:] = -upper_bounds

    # stack up these new inequality constraints with the given inequality constraints
    if ineq_coeffs is not None:
        full_ineq_coeffs = np.vstack((-ineq_coeffs, new_ineq_coeffs))
        full_ineq_constants = np.concatenate((-ineq_constants, new_ineq_constants))
    else:
        full_ineq_coeffs = new_ineq_coeffs
        full_ineq_constants = new_ineq_constants

    # stack up equality constraints with inequality constraints
    if eq_coeffs is not None:
        coeffs = np.vstack((eq_coeffs, full_ineq_coeffs))
        constants = np.concatenate((eq_constants, full_ineq_constants))
        num_eq = eq_constants.size
    else:
        coeffs = full_ineq_coeffs.T
        constants = full_ineq_constants
        num_eq = 0

    small_constant = 1e-8 # add a small constant to the diagonal of the Hessian to help numerical stability

    # solve quadratic program
    try:
        sol = quadprog.solve_qp(hessian + small_constant * np.eye(n), weights, coeffs, constants, num_eq)[0]
    except ValueError:
        print "Warning: Quadratic program solver exited with a numerical error " \
              "(quadprog can be sensitive to very non-separable problems). Guessing all-zeros solution."
        sol = np.zeros(n)

    return sol
