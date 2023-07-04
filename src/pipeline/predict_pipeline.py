import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj


def predict_price(model, preprocessor, input_list):
    try:
        
        df = pd.DataFrame({
            'carat': [input_list[0]],
            'cut': [input_list[1]],
            'color': [input_list[2]],
            'clarity': [input_list[3]],
            'depth': [input_list[4]],
            'table': [input_list[5]],
            'x': [input_list[6]],
            'y': [input_list[7]],
            'z': [input_list[8]]
        })

        arr = preprocessor.transform(df)

        return model.predict(arr[0])

    except Exception as e:
        raise CustomException(e,sys)
