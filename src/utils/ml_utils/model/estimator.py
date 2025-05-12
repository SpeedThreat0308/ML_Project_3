from src.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME

import os
import sys

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def predict(self, X):
        try:
            X_transformed = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_transformed)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e