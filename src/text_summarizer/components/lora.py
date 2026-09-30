# from pathlib import Path

# from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
# from transformers import AutoModelForSeq2SeqLM


# def create_lora_model(model_name: str, lora_config: dict):
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     model.config.use_cache = False

#     config = LoraConfig(
#         task_type=TaskType.SEQ_2_SEQ_LM,
#         inference_mode=False,
#         r=int(lora_config["rank"]),
#         lora_alpha=int(lora_config["alpha"]),
#         lora_dropout=float(lora_config["dropout"]),
#         target_modules=list(lora_config["target_modules"]),
#         bias="none",
#     )
#     return get_peft_model(model, config)


# def load_model_for_inference(base_model_path: str | Path, adapter_path: str | Path):
#     adapter_path = Path(adapter_path)
#     if (adapter_path / "adapter_config.json").exists():
#         peft_config = PeftConfig.from_pretrained(adapter_path)
#         base_model = AutoModelForSeq2SeqLM.from_pretrained(
#             peft_config.base_model_name_or_path
#         )
#         return PeftModel.from_pretrained(base_model, adapter_path)

#     return AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
