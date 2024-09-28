import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import json
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    @abstractmethod
    def load(path):
        pass


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
        with open(path, "r") as f:
            model_params = json.load(f)
        model = LinearRegressionModel()
        model.model.coef_ = np.array(model_params["coef"])
        model.model.intercept_ = np.array(model_params["intercept"])
        return model


class XGBoostModel(BaseModel):
    def __init__(self):
        self.model = XGBRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_model(path)

    @staticmethod
    def load(path):
        model = XGBoostModel()
        model.model.load_model(path)
        return model
