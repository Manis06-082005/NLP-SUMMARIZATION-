import sys
from pathlib import Path

import evaluate
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from src.text_summarizer.logger import logger
from src.text_summarizer.utils import read_yaml, resolve_path
from src.text_summarizer.exception import CustomException
from src.text_summarizer.components.lora import load_model_for_inference


class ModelEvaluation:

    def __init__(self):

        try:
            # Read config.yaml
            self.config = read_yaml(
                "config/config.yaml"
            )

            # Path where our trained model is saved
            model_path = resolve_path(self.config["paths"]["model_dir"])
            adapter_path = resolve_path(self.config["paths"]["adapter_dir"])
            tokenizer_path = (
                adapter_path
                if (Path(adapter_path) / "tokenizer_config.json").exists()
                else model_path
            )

            logger.info(
                f"Loading trained model from: {model_path}"
            )

            # Load our fine-tuned tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=False,
            )

            # Load our fine-tuned Pegasus model
            self.model = load_model_for_inference(model_path, adapter_path)

            # Choose GPU if available
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            # Move model to GPU/CPU
            self.model.to(self.device)

            # Prediction mode
            self.model.eval()

            # Load ROUGE metric
            self.rouge = evaluate.load("rouge")

            logger.info(
                "Model evaluation initialized successfully"
            )

        except Exception as e:
            raise CustomException(e, sys)


    def generate_summary(self, dialogue):

        try:

            # Convert dialogue into token IDs
            inputs = self.tokenizer(
                dialogue,
                return_tensors="pt",
                max_length=self.config["training"][
                    "max_input_length"
                ],
                truncation=True
            )

            # Move tensors to GPU/CPU
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            # We don't need gradients during prediction
            with torch.no_grad():

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["training"][
                        "max_target_length"
                    ],
                    num_beams=4
                )

            # Token IDs -> readable English
            summary = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            return summary

        except Exception as e:
            raise CustomException(e, sys)


    def evaluate_model(self, dataset):

        try:

            logger.info(
                "Model evaluation started"
            )

            predictions = []
            references = []

            # Use SAMSum test dataset
            test_dataset = dataset["test"]

            for i, example in enumerate(test_dataset):

                dialogue = example["dialogue"]

                actual_summary = example["summary"]

                # Generate summary using Pegasus
                predicted_summary = self.generate_summary(
                    dialogue
                )

                predictions.append(
                    predicted_summary
                )

                references.append(
                    actual_summary
                )

                if (i + 1) % 50 == 0:

                    logger.info(
                        f"Evaluated {i + 1}/"
                        f"{len(test_dataset)} examples"
                    )

            # Calculate ROUGE
            rouge_scores = self.rouge.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )

            logger.info(
                "Model evaluation completed"
            )

            logger.info(
                f"ROUGE Scores: {rouge_scores}"
            )

            return rouge_scores

        except Exception as e:
            raise CustomException(e, sys)
