---
title: Dialogue Summarizer
emoji: 📝
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# Dialogue Summarizer

Paste a conversation to generate a summary with Pegasus fine-tuned on SAMSum.
Choose the maximum summary tokens, then submit. Input is limited to 20,000
characters and truncated to 256 tokens; review summaries for accuracy.

Model: https://huggingface.co/nathmanish/PegasusSamsum

Source and local Docker instructions:
https://github.com/Manis06-082005/NLP-SUMMARIZATION-

This Space uses Gradio and ZeroGPU. Free GPU requests have queues and quotas.

## Deployment settings

Set `MODEL_ID` to your model repository and `MODEL_SUBFOLDER` to the directory
containing its model/tokenizer files. Defaults are `nathmanish/PegasusSamsum`
and `final_model`. Private models need an `HF_TOKEN` secret with read access.
Keep weights in the model repository; upload only `app.py`, `requirements.txt`,
and this README to the Space root.

Select ZeroGPU hardware. This deployment uses Python 3.12 for the ZeroGPU
runtime; the separate local FastAPI Docker deployment uses Python 3.13.
