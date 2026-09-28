import sys
import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.text_summarizer.logger import logger
from src.text_summarizer.utils import read_yaml
from src.text_summarizer.exception import CustomException


class ModelTrainer:

    def __init__(self):

        try:
            # ---------------------------------
            # 1. Read configuration
            # ---------------------------------
            self.config = read_yaml("config/config.yaml")

            model_name = self.config["model"]["model_name"]

            logger.info(f"Loading model: {model_name}")

            # ---------------------------------
            # 2. Load tokenizer
            # ---------------------------------
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )

            # ---------------------------------
            # 3. Load pretrained Pegasus model
            # ---------------------------------
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name
            )

            # Required/recommended when using
            # gradient checkpointing
            self.model.config.use_cache = False

            # ---------------------------------
            # 4. Check device
            # ---------------------------------
            if torch.cuda.is_available():
                logger.info(
                    f"GPU available: {torch.cuda.get_device_name(0)}"
                )
            else:
                logger.warning(
                    "GPU not available. Training will use CPU."
                )

            logger.info("Model loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)


    def train(self, tokenized_dataset):

        try:
            logger.info("Model training started")

            # =================================
            # 1. Select datasets
            # =================================

            if self.config["training"]["smoke_test"]:

                train_samples = self.config["training"]["train_samples"]
                validation_samples = self.config["training"]["validation_samples"]

                train_dataset = tokenized_dataset["train"].select(
                    range(
                        min(
                            train_samples,
                            len(tokenized_dataset["train"])
                        )
                    )
                )

                eval_dataset = tokenized_dataset["validation"].select(
                    range(
                        min(
                            validation_samples,
                            len(tokenized_dataset["validation"])
                        )
                    )
                )

                logger.info(
                    f"Smoke test enabled: "
                    f"{len(train_dataset)} train samples, "
                    f"{len(eval_dataset)} validation samples"
                )

            else:

                train_dataset = tokenized_dataset["train"]
                eval_dataset = tokenized_dataset["validation"]

                logger.info(
                    "Full dataset training enabled"
                )


            # =================================
            # 2. Data Collator
            # =================================

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                label_pad_token_id=-100
            )


            # =================================
            # 3. Training Arguments
            # =================================

            training_args = Seq2SeqTrainingArguments(

                # Where checkpoints/model are saved
                output_dir=self.config["paths"]["model_dir"],

                # Training hyperparameters
                num_train_epochs=float(
                    self.config["training"]["epochs"]
                ),

                learning_rate=float(
                    self.config["training"]["learning_rate"]
                ),

                per_device_train_batch_size=int(
                    self.config["training"]["batch_size"]
                ),

                per_device_eval_batch_size=int(
                    self.config["training"]["batch_size"]
                ),

                # Effective batch size becomes:
                # batch_size × accumulation steps
                gradient_accumulation_steps=int(
                    self.config["training"][
                        "gradient_accumulation_steps"
                    ]
                ),

                # Save GPU memory
                gradient_checkpointing=bool(
                    self.config["training"][
                        "gradient_checkpointing"
                    ]
                ),

                # Mixed precision
                fp16=bool(
                    self.config["training"]["fp16"]
                ),

                # Evaluate after every epoch
                eval_strategy="epoch",

                # Save after every epoch
                save_strategy="epoch",

                # Logging
                logging_strategy="steps",
                logging_steps=10,

                # Don't send logs to WandB etc.
                report_to="none",

                # We are not calculating ROUGE yet
                predict_with_generate=False,

                # Keep only a few checkpoints
                save_total_limit=2
            )


            # =================================
            # 4. Create Trainer
            # =================================

            trainer = Seq2SeqTrainer(

                model=self.model,

                args=training_args,

                train_dataset=train_dataset,

                eval_dataset=eval_dataset,

                data_collator=data_collator,

                processing_class=self.tokenizer
            )


            # =================================
            # 5. Start Training
            # =================================

            logger.info("Calling trainer.train()")

            trainer.train()


            # =================================
            # 6. Save Final Model
            # =================================

            logger.info("Saving trained model")

            trainer.save_model(
                self.config["paths"]["model_dir"]
            )

            self.tokenizer.save_pretrained(
                self.config["paths"]["model_dir"]
            )


            logger.info(
                "Model training completed successfully"
            )

            return self.model


        except Exception as e:
            raise CustomException(e, sys)
