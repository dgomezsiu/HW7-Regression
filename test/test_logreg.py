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

# set seed for rng

np.random.seed(42)

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
    y_true = np.array([0,1,0,1])
    y_pred = np.array([0.1,0.9,0.1,0.9])

	# my logistic regression

    my_model = regression.logreg.LogisticRegressor(num_feats=2)

    # calculate expected loss

    expected_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    calculated_loss = my_model.loss_function(y_true, y_pred)

	# compare losses

    assert np.isclose(expected_loss, calculated_loss)

def test_gradient():
		X = np.array([[0], [2], [3]])
		y = np.array([0, 1, 1])

        # my logistic regression

		my_model = regression.logreg.LogisticRegressor(num_feats=1)
		my_model.train_model(X, y, X, y)
		my_predictions = my_model.make_prediction(X)
		calculated_gradient = my_model.calculate_gradient(y, X)

		expected_error = y - my_predictions
		expected_gradient = -1 * X.T.dot(expected_error) / len(y)

		assert np.isclose(calculated_gradient, expected_gradient)

def test_training():
		X = np.array([[0], [2], [3]])
		y = np.array([0, 1, 1])

        # my logistic regression. with the same seed and input, chainging iteration number must update values
		
		my_initial_model = regression.logreg.LogisticRegressor(num_feats=1, max_iter=10)

		my_model_1 = regression.logreg.LogisticRegressor(num_feats=1, max_iter=10)
		my_model_1.train_model(X, y, X, y)

		my_model_2 = regression.logreg.LogisticRegressor(num_feats=1, max_iter=20)
		my_model_2.train_model(X, y, X, y)

		assert np.all(my_model_1.W != my_model_2.W)
		assert np.all(my_model_1.W != my_initial_model.W)
		assert np.all(my_model_2.W != my_initial_model.W)