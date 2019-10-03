"""This module includes utilities to run cross-validation on general supervised learning methods."""
import numpy as np


def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """
    Performs cross validation with random splits
    
    :param trainer: function that trains a model from data with the template 
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model), or a tuple that contains the prediction in its 0th entry
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """

    scores = np.zeros(folds)

    d, n = all_data.shape

    indices = np.array(range(n), dtype=int)

    # pad indices to make it divide evenly by folds
    examples_per_fold = int(np.ceil(n / folds))
    ideal_length = int(examples_per_fold * folds)
    # use -1 as an indicator of an invalid index
    indices = np.append(indices, -np.ones(ideal_length - indices.size, dtype=int))
    assert indices.size == ideal_length

    indices = indices.reshape((examples_per_fold, folds))

    models = []

    for i in range(folds):
        train_indices = np.delete(indices, i, 1).ravel()
        # remove invalid padded indices
        train_indices = train_indices[train_indices >= 0]
        val_indices = indices[:, i].ravel()
        # remove invalid padded indices
        val_indices = val_indices[val_indices >= 0]

        val_data = all_data[:, val_indices]
        val_labels = all_labels[val_indices]

        train_data = all_data[:, train_indices]
        train_labels = all_labels[train_indices]

        model = trainer(train_data, train_labels, params)

        predictions = predictor(val_data, model)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        models.append(model)

        scores[i] = np.mean(predictions == val_labels)

    score = np.mean(scores)

    return score, models
