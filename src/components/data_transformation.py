import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.dataTransConfig = DataTransformationConfig()

    def get_transformation_obj(self):

        try:
            num_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cat_columns = ['cut', 'color', 'clarity']

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaling", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoding", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ("scaling", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns)
                ]
            )

            logging.info("Transformation pipeline created")

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("Reading train and test data")

            preprocessor = self.get_transformation_obj()

            X_train = df_train.drop(columns=['id', 'price'], axis=1)
            y_train = df_train['price']

            X_test = df_test.drop(columns=['id', 'price'], axis=1)
            y_test = df_test['price']

            input_train_arr = preprocessor.fit_transform(X_train)
            input_test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[input_train_arr, np.array(y_train)]
            test_arr = np.c_[input_test_arr, np.array(y_test)]
            
            save_obj(self.dataTransConfig.preprocessor_path, preprocessor)

            logging.info("Transformed Data")

            return(
                train_arr,
                test_arr,
                self.dataTransConfig.preprocessor_path
            )

        except Exception as e:
            raise CustomException(e, sys)
