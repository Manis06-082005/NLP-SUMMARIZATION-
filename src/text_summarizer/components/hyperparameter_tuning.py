# import json
# import sys
# from pathlib import Path

# import evaluate
# import numpy as np
# import torch
# from transformers import (
#     AutoModelForSeq2SeqLM,
#     AutoTokenizer,
#     DataCollatorForSeq2Seq,
#     Seq2SeqTrainer,
#     Seq2SeqTrainingArguments,
# )

# from src.text_summarizer.exception import CustomException
# from src.text_summarizer.logger import logger
# from src.text_summarizer.utils import read_yaml
# from src.text_summarizer.components.lora import create_lora_model


# class HyperparameterTuner:
#     def __init__(self):
#         try:
#             self.project_root = Path(__file__).resolve().parents[3]
#             self.config = read_yaml(self.project_root / "config" / "config.yaml")
#             self.model_name = self.config["model"]["model_name"]
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
#             self.rouge = evaluate.load("rouge")
#         except Exception as exc:
#             raise CustomException(exc, sys) from exc

#     def _model_init(self):
#         if self.config["lora"]["enabled"]:
#             return create_lora_model(self.model_name, self.config["lora"])
#         model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
#         model.config.use_cache = False
#         return model

#     def _compute_metrics(self, evaluation_prediction):
#         predictions, labels = evaluation_prediction
#         if isinstance(predictions, tuple):
#             predictions = predictions[0]

#         labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
#         decoded_predictions = self.tokenizer.batch_decode(
#             predictions, skip_special_tokens=True
#         )
#         decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

#         scores = self.rouge.compute(
#             predictions=decoded_predictions,
#             references=decoded_labels,
#             use_stemmer=True,
#         )
#         return {name: round(value, 6) for name, value in scores.items()}

#     @staticmethod
#     def _hp_space(trial):
#         return {
#             "learning_rate": trial.suggest_float(
#                 "learning_rate", 5e-5, 5e-4, log=True
#             ),
#             "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
#             "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
#             "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.15),
#             "label_smoothing_factor": trial.suggest_float(
#                 "label_smoothing_factor", 0.0, 0.15
#             ),
#         }

#     @staticmethod
#     def _objective(metrics):
#         return metrics["eval_rougeLsum"]

#     def tune(self, tokenized_dataset):
#         try:
#             tuning = self.config["tuning"]
#             train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(
#                 range(min(tuning["train_samples"], len(tokenized_dataset["train"])))
#             )
#             eval_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(
#                 range(
#                     min(
#                         tuning["validation_samples"],
#                         len(tokenized_dataset["validation"]),
#                     )
#                 )
#             )

#             output_dir = self.project_root / tuning["output_dir"]
#             output_dir.mkdir(parents=True, exist_ok=True)
#             fp16 = bool(self.config["training"]["fp16"] and torch.cuda.is_available())

#             args = Seq2SeqTrainingArguments(
#                 output_dir=str(output_dir),
#                 per_device_train_batch_size=1,
#                 per_device_eval_batch_size=1,
#                 gradient_accumulation_steps=int(
#                     self.config["training"]["gradient_accumulation_steps"]
#                 ),
#                 gradient_checkpointing=True,
#                 fp16=fp16,
#                 eval_strategy="epoch",
#                 save_strategy="no",
#                 logging_strategy="steps",
#                 logging_steps=25,
#                 report_to="none",
#                 predict_with_generate=True,
#                 generation_max_length=int(
#                     self.config["training"]["max_target_length"]
#                 ),
#                 generation_num_beams=2,
#                 eval_accumulation_steps=1,
#                 optim="adamw_torch",
#                 seed=42,
#             )

#             trainer = Seq2SeqTrainer(
#                 model_init=self._model_init,
#                 args=args,
#                 train_dataset=train_dataset,
#                 eval_dataset=eval_dataset,
#                 data_collator=DataCollatorForSeq2Seq(
#                     tokenizer=self.tokenizer,
#                     label_pad_token_id=-100,
#                 ),
#                 processing_class=self.tokenizer,
#                 compute_metrics=self._compute_metrics,
#             )

#             logger.info(
#                 "Starting %s tuning trials with %s train and %s validation samples",
#                 tuning["n_trials"],
#                 len(train_dataset),
#                 len(eval_dataset),
#             )
#             best_run = trainer.hyperparameter_search(
#                 direction="maximize",
#                 backend="optuna",
#                 hp_space=self._hp_space,
#                 compute_objective=self._objective,
#                 n_trials=int(tuning["n_trials"]),
#             )

#             result = {
#                 "objective": best_run.objective,
#                 "hyperparameters": best_run.hyperparameters,
#                 "train_samples": len(train_dataset),
#                 "validation_samples": len(eval_dataset),
#             }
#             result_path = output_dir / "best_hyperparameters.json"
#             result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
#             logger.info("Best tuning result saved to %s", result_path)
#             return result
#         except Exception as exc:
#             raise CustomException(exc, sys) from exc
