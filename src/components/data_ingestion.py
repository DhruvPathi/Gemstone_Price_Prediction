import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_path = os.path.join("artifacts", "gemstone.csv")
    train_data = os.path.join("artifacts", "train.csv")
    test_data = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Reading the raw data.")
        try:
            df = pd.read_csv("./notebooks/data/gemstone.csv")
            logging.info("Read the data as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.data_path, index=False, header=True)
            logging.info("Train test split initiated")
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
            df_train.to_csv(self.ingestion_config.train_data, index=False, header=True)
            df_test.to_csv(self.ingestion_config.test_data, index=False, header=True)
            logging.info("Data split complete")
            
            return(
                self.ingestion_config.train_data,
                self.ingestion_config.test_data
            )

        except Exception as e:
            raise CustomException(e,sys)
