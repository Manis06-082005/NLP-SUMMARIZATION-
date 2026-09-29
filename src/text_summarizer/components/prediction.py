import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.text_summarizer.exception import CustomException
from src.text_summarizer.logger import logger
from src.text_summarizer.utils import read_yaml


class PredictionPipeline:
    def __init__(self):
        try:
            project_root = Path(__file__).resolve().parents[3]
            self.config = read_yaml(project_root / "config" / "config.yaml")
            model_path = project_root / self.config["paths"]["model_dir"]

            logger.info(f"Loading trained model from: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Prediction pipeline ready on {self.device}")
        except Exception as exc:
            raise CustomException(exc, sys) from exc

    def predict(self, dialogue: str) -> str:
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
                    max_new_tokens=self.config["training"]["max_target_length"],
                    num_beams=4,
                    early_stopping=True,
                )

            summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.info("Summary generated successfully")
            return summary
        except Exception as exc:
            raise CustomException(exc, sys) from exc
