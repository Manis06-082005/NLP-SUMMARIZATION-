import sys

from src.text_summarizer.components.data_ingestion import DataIngestion
from src.text_summarizer.components.data_transformation import DataTransformation
from src.text_summarizer.components.model_trainer import ModelTrainer

from src.text_summarizer.logger import logger
from src.text_summarizer.exception import CustomException


class TrainingPipeline:

    def __init__(self):
        pass


    def run_training_pipeline(self):

        try:
            logger.info("Training pipeline started")


            # ==========================================
            # 1. DATA INGESTION
            # ==========================================

            logger.info("Starting data ingestion")

            data_ingestion = DataIngestion()

            dataset = data_ingestion.load_data()

            logger.info("Data ingestion completed")


            # ==========================================
            # 2. DATA TRANSFORMATION
            # ==========================================

            logger.info("Starting data transformation")

            data_transformation = DataTransformation()

            tokenized_dataset = (
                data_transformation.transform_data(
                    dataset
                )
            )

            logger.info(
                "Data transformation completed"
            )


            # ==========================================
            # 3. MODEL TRAINING
            # ==========================================

            logger.info("Starting model training")

            model_trainer = ModelTrainer()

            model = model_trainer.train(
                tokenized_dataset
            )

            logger.info(
                "Model training completed"
            )


            logger.info(
                "Training pipeline completed successfully"
            )

            return model


        except Exception as e:
            raise CustomException(e, sys)
