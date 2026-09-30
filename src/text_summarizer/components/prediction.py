import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.text_summarizer.exception import CustomException
from src.text_summarizer.logger import logger
from src.text_summarizer.utils import read_yaml, resolve_path
from transformers import AutoModelForSeq2SeqLM


class PredictionPipeline:
    def __init__(self):
        try:
            self.config = read_yaml("config/config.yaml")
            model_path = resolve_path(self.config["paths"]["model_dir"])
            tokenizer_path = model_path

            logger.info(f"Loading tokenizer from: {tokenizer_path}")

            # This checkpoint contains a SentencePiece model but no tokenizer.json.
            # Avoid fast-tokenizer conversion, which can misread spiece.model as text.
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=False,
                local_files_only=True,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Prediction pipeline ready on {self.device}")
        except Exception as exc:
            raise CustomException(exc, sys) from exc

    def predict(self, dialogue: str, summary_length: int | None = None) -> str:
        try:
            logger.info("Generating summary")
            inputs = self.tokenizer(
                dialogue,
                return_tensors="pt",
                max_length=self.config["training"]["max_input_length"],
                truncation=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=summary_length or self.config["training"]["max_target_length"],
                    num_beams=4,
                    early_stopping=True,
                )

            summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.info("Summary generated successfully")
            return summary
        except Exception as exc:
            raise CustomException(exc, sys) from exc
