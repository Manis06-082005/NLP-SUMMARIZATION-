import sys
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from src.text_summarizer.logger import logger
from src.text_summarizer.utils import read_yaml
from src.text_summarizer.exception import CustomException


class PredictionPipeline:

    def __init__(self):

        try:
            # Read configuration
            self.config = read_yaml(
                "config/config.yaml"
            )

            # Path of our fine-tuned model
            model_path = self.config["paths"]["model_dir"]

            logger.info(
                f"Loading trained model from: {model_path}"
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path
            )

            # Load fine-tuned Pegasus model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path
            )

            # Select GPU if available
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            # Move model to device
            self.model.to(self.device)

            # Set model to prediction mode
            self.model.eval()

            logger.info(
                f"Prediction pipeline ready on {self.device}"
            )

        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, dialogue):

        try:
            logger.info(
                "Generating summary"
            )

            # ---------------------------------
            # 1. Tokenize new dialogue
            # ---------------------------------

            inputs = self.tokenizer(
                dialogue,
                return_tensors="pt",
                max_length=self.config["training"][
                    "max_input_length"
                ],
                truncation=True
            )

            # ---------------------------------
            # 2. Move input to GPU / CPU
            # ---------------------------------

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            # ---------------------------------
            # 3. Generate summary
            # ---------------------------------

            with torch.no_grad():

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["training"][
                        "max_target_length"
                    ],
                    num_beams=4,
                    early_stopping=True
                )

            # ---------------------------------
            # 4. Decode token IDs
            # ---------------------------------

            summary = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            logger.info(
                "Summary generated successfully"
            )

            return summary

        except Exception as e:
            raise CustomException(e, sys)
