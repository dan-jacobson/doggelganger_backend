from doggelganger.models.base import BaseModel as BaseModel
from doggelganger.models.linear import LinearRegressionModel as LinearRegressionModel
from doggelganger.models.resnet import ResNetModel as ResNetModel
from doggelganger.models.xgb import XGBoostModel as XGBoostModel

model_classes = {
    "linear": LinearRegressionModel,
    "xgboost": XGBoostModel,
    "resnet": ResNetModel,
}
