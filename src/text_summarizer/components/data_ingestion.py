from datasets import load_dataset

from src.text_summarizer.logger import logger
from src.text_summarizer.utils import read_yaml
from src.text_summarizer.exception import CustomException

import sys

class DataIngestion:

    def __init__(self):
        self.config = read_yaml("config/config.yaml")

    def load_data(self):
        try:
            dataset_name = self.config["dataset"]["dataset_name"]

            logger.info(f"Loading dataset: {dataset_name}")

            dataset = load_dataset(dataset_name)

            logger.info("Dataset loaded successfully.")

            return dataset

        except Exception as e:
            logger.error("Error occurred while loading dataset.")
            raise CustomException(e, sys)
