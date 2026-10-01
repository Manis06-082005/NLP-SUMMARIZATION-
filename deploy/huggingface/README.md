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

Summarizes conversations using a Pegasus model fine-tuned on SAMSum.

Upload the trained model files into a `final_model/` directory in this Space.
Alternatively, set the `MODEL_ID` Space variable to your trained model's
Hugging Face repository ID. Private model repositories also need an `HF_TOKEN`
Space secret with read access.

This deployment uses Gradio and ZeroGPU, not Docker Compose. Python 3.12 is
used for compatibility with the ZeroGPU runtime.
