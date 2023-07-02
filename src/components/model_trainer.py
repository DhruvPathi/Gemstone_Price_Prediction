import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj, load_obj
from src.utils import evaluate_models

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Splitting target and features from train and test array")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }

            models_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            best_model_score = max(models_report.values())
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model found")

            logging.info(f"{best_model_name} is the best model")

            save_obj(self.model_config.model_path, best_model)

            return best_model, best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e,sys)
