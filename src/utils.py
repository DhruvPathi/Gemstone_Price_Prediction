import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_test_predicted = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_predicted)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)