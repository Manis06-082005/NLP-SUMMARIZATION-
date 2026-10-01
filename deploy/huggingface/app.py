"""Gradio entry point for a Hugging Face ZeroGPU Space."""
import os
from pathlib import Path

import spaces  # Import before torch to enable ZeroGPU's CUDA support.
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Upload the saved model files to the Space's final_model/ directory, or set
# MODEL_ID to a Hugging Face model repository containing your trained model.
MODEL_SOURCE = os.environ.get("MODEL_ID") or str(Path(__file__).parent / "final_model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_SOURCE)
model.to("cuda")
model.eval()


@spaces.GPU(duration=60)
def summarize(dialogue: str, summary_length: int) -> str:
    dialogue = dialogue.strip()
    if not dialogue:
        raise gr.Error("Enter a dialogue first.")
    if len(dialogue) > 20_000:
        raise gr.Error("Keep the dialogue under 20,000 characters.")
    inputs = tokenizer(dialogue, return_tensors="pt", max_length=256, truncation=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=int(summary_length), num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Dialogue", lines=10, placeholder="Alex: Are we meeting at 3?\nSam: Yes, I'll send the link."),
        gr.Slider(16, 128, value=64, step=1, label="Maximum summary tokens"),
    ],
    outputs=gr.Textbox(label="Summary"),
    title="Dialogue Summarizer",
    description="Generate a summary with a Pegasus model fine-tuned on SAMSum. Long dialogues are truncated to 256 input tokens. Free GPU use is subject to queues and daily quotas.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=20).launch()
