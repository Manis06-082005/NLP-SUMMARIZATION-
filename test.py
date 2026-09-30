import sys

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.text_summarizer.exception import CustomException
from src.text_summarizer.utils import read_yaml, resolve_path


def main():
    try:
        config = read_yaml("config/config.yaml")
        model_path = resolve_path(config["paths"]["model_dir"])
        dataset_name = config["dataset"]["dataset_name"]

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            local_files_only=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        test_dataset = load_dataset(dataset_name, split="test")
        predictions = []
        references = []

        for example in test_dataset:
            inputs = tokenizer(
                example["dialogue"],
                return_tensors="pt",
                max_length=config["training"]["max_input_length"],
                truncation=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=config["training"]["max_target_length"],
                    num_beams=4,
                    early_stopping=True,
                )

            predictions.append(
                tokenizer.decode(output_ids[0], skip_special_tokens=True)
            )
            references.append(example["summary"])

        rouge = evaluate.load("rouge")
        scores = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )

        print("\nROUGE test results")
        for name in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
            print(f"{name}: {scores[name]:.4f}")

    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()
