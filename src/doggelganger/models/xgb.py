from xgboost import XGBRegressor

from doggelganger.models.base import BaseModel


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
