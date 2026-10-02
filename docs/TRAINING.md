# Training and evaluation

For inference only, download the published model using the [README](../README.md).
Training is optional and writes to the configured model directory. Back up
existing weights before training if you want to preserve them.

## Setup

Use Python 3.13 and install `requirements.txt` as described in the README.
Training downloads the model and dataset. A CUDA-enabled PyTorch installation
and compatible GPU are recommended. CPU training disables fp16 automatically.

## Smoke run

In `config/config.yaml`, set `training.smoke_test: true`. Review
`train_samples` (50) and `validation_samples` (20), leaving other settings intact.

```powershell
python main.py
```

This tests the pipeline on a small sample; it is not a quality benchmark.
The configured model is `google/pegasus-xsum`, dataset `knkarthick/samsum`.
Outputs are saved to `models/pegasus_samsum/final_model/`; logs go to `logs/`.

## Full training

Set `training.smoke_test: false`, review epochs, batch size, learning rate,
gradient accumulation, and token limits, then run `python main.py`.
Duration depends on hardware and dataset size. Keep all saved model and
tokenizer files together for inference.

## ROUGE evaluation

After training or downloading the model:

```powershell
python test.py
```

This evaluates the complete SAMSum test split and prints ROUGE-1, ROUGE-2,
ROUGE-L, and ROUGE-Lsum. It downloads the dataset and metric and can take
considerable time on CPU. It is not a lightweight unit-test suite.

## Deploy updated weights

Upload the complete saved model folder to the model repository and restart
the Space. Retest summary generation. Follow the
[Hugging Face guide](../deploy/huggingface/UPLOAD.md); do not commit large weights
to GitHub.
