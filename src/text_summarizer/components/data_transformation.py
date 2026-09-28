import sys
from transformers import AutoTokenizer

from src.text_summarizer.logger import logger
from src.text_summarizer.utils import read_yaml
from src.text_summarizer.exception import CustomException


class DataTransformation:

    def __init__(self):
        try:
            # Read configuration
            self.config = read_yaml("config/config.yaml")

            # Get model name from config
            model_name = self.config["model"]["model_name"]

            logger.info(f"Loading tokenizer for: {model_name}")

            # Load Pegasus tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            logger.info("Tokenizer loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)


    def preprocess_function(self, batch):
        """
        Converts dialogues and summaries into token IDs.
        """

        try:
            # Tokenize dialogues
            model_inputs = self.tokenizer(
                batch["dialogue"],
                max_length=self.config["training"]["max_input_length"],
                truncation=True
            )

            # Tokenize summaries
            labels = self.tokenizer(
                text_target=batch["summary"],
                max_length=self.config["training"]["max_target_length"],
                truncation=True
            )

            # Correct summary token IDs become labels
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        except Exception as e:
            raise CustomException(e, sys)


    def transform_data(self, dataset):
        """
        Applies preprocessing to the complete dataset.
        """

        try:
            logger.info("Starting data transformation")

            tokenized_dataset = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )

            logger.info("Data transformation completed successfully")

            return tokenized_dataset

        except Exception as e:
            raise CustomException(e, sys)
