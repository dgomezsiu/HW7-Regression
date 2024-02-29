"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
# (you will probably need to import more things here)
import numpy as np
import math
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import regression.utils
import regression.logreg


def test_prediction():
		X = np.array([[0], [2], [3]])
		y = np.array([0, 1, 1])

        # my logistic regression

		my_model = regression.logreg.LogisticRegressor(num_feats=1)
		my_model.train_model(X, y, X, y)
		my_predictions = my_model.make_prediction(X)

        # Scikit-learn logistic regression

		sklearn_model = SklearnLogisticRegression(solver='liblinear')
		sklearn_model.fit(X, y)
		sklearn_predictions = sklearn_model.predict(X)

        # Compare predictions

		assert np.all(my_predictions == sklearn_predictions)

def test_loss_function():
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([0, 1, 0])

	# my logistic regression

    my_model = regression.logreg.LogisticRegressor(num_feats=2)
    my_model.train_model(X, y, X, y)
    my_predictions = my_model.make_prediction(X)

    # calculate expected loss

    expected_loss = -np.mean(y * np.log(my_predictions) + (1 - y) * np.log(1 - my_predictions))
    calculated_loss = my_model.loss_function(y, my_predictions)
    np.isclose(expected_loss, calculated_loss)

def test_gradient():
	pass

def test_training():
	pass