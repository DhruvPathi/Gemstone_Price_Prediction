import os
import sys
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj
from src.utils import InputData
from src.pipeline.predict_pipeline import predict_price

logging.info('getting model')
model = load_obj(os.path.join('artifacts', 'model.pkl'))
logging.info('getting preprocessor')
preprocessor = load_obj(os.path.join('artifacts', 'preprocessor.pkl'))

app = FastAPI()

@app.get('/')
def index():
    return {'message':'Hello World'}

@app.post('/predict')
def predict_gemstone_price(data:InputData):
    data = data.dict()
    carat = data['carat']
    cut = data['cut']
    color = data['color']
    clarity = data['clarity']
    depth = data['depth']
    table = data['table']
    x = data['x']
    y = data['y']
    z = data['z']

    result = predict_price(model,preprocessor,[carat,cut,color,clarity,depth,table,x,y,z])

    return{
        "Price": result
    }

if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
