import json

import numpy as np
from sklearn.linear_model import LinearRegression

from doggelganger.models.base import BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        model_params = {
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(model_params, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            model_params = json.load(f)
        model = LinearRegressionModel()
        model.model.coef_ = np.array(model_params["coef"])
        model.model.intercept_ = np.array(model_params["intercept"])
        return model
