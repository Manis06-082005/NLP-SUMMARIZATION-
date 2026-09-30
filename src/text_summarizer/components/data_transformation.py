 
import sys

from transformers import AutoTokenizer

from src.text_summarizer.exception import CustomException
from src.text_summarizer.logger import logger
from src.text_summarizer.utils import read_yaml


class DataTransformation:
    """Convert SAMSum text examples into Pegasus training features."""

    def __init__(self):
        try:
            self.config = read_yaml("config/config.yaml")
            model_name = self.config["model"]["model_name"]
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise CustomException(exc, sys)

    def transform_data(self, dataset):
        """Tokenize dialogues as inputs and summaries as training labels."""
        try:
            training_config = self.config["training"]
            max_input_length = training_config["max_input_length"]
            max_target_length = training_config["max_target_length"]

            def tokenize_batch(batch):
                model_inputs = self.tokenizer(
                    batch["dialogue"],
                    max_length=max_input_length,
                    truncation=True,
                )
                labels = self.tokenizer(
                    text_target=batch["summary"],
                    max_length=max_target_length,
                    truncation=True,
                )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            logger.info("Tokenizing dataset")
            tokenized_dataset = dataset.map(
                tokenize_batch,
                batched=True,
                remove_columns=dataset["train"].column_names,
                desc="Tokenizing dialogue and summary pairs",
            )
            logger.info("Dataset tokenization completed")
            return tokenized_dataset

        except Exception as exc:
            logger.error("Error occurred during data transformation")
            raise CustomException(exc, sys)
