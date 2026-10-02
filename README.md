# Dialogue Summarizer

Summarize conversations with Pegasus fine-tuned on SAMSum.

**Live demo:** https://huggingface.co/spaces/nathmanish/NLP

**Saved model:** https://huggingface.co/nathmanish/PegasusSamsum

| Goal | Guide |
| --- | --- |
| Run the Docker app locally | [Docker setup](docs/DOCKER.md) |
| Deploy on free ZeroGPU | [Hugging Face setup](deploy/huggingface/UPLOAD.md) |
| Train or evaluate a model | [Training and evaluation](docs/TRAINING.md) |

Docker runs the FastAPI app locally. The Hugging Face Space runs a separate
Gradio interface and downloads the model from its model repository.

## Get the project

```powershell
git clone https://github.com/Manis06-082005/NLP-SUMMARIZATION-.git
cd NLP-SUMMARIZATION-
```

Run commands from the repository root.

Fine-tune a Pegasus sequence-to-sequence model on the SAMSum dialogue dataset
and serve predictions through a small FastAPI web app.

## Setup

Use Python 3.13 or newer. Create and activate a virtual environment, then install
the dependencies:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the configured dataset and pretrained model. Training
is much faster with a CUDA-enabled PyTorch installation and a compatible GPU.
On CPU, mixed precision is disabled automatically.

## Download the saved model

Model weights are excluded from GitHub. Download them before local inference;
retraining is optional:

```powershell
python -m pip install --upgrade huggingface_hub
hf download nathmanish/PegasusSamsum --include "final_model/*" --local-dir models/pegasus_samsum
```

Check that `models/pegasus_samsum/final_model/` contains `config.json`,
`model.safetensors`, `special_tokens_map.json`, `spiece.model`, and
`tokenizer_config.json`. The weights alone occupy about 2.28 GB.

## Train

From the repository root, run:

```powershell
python main.py
```

Settings live in `config/config.yaml`. Set `training.smoke_test: true` to train
on the configured small sample while checking the pipeline. Training outputs are
written under `models/`.

## Run the web app

After training, start the API from the repository root:

```powershell
uvicorn app.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> to enter a dialogue. The JSON API accepts
`POST /predict` with `{"dialogue":"..."}`; `GET /health` reports service
status. The saved trained model must exist locally before predictions
will work.

## API example

```powershell
$payload = @{
    dialogue = "Alex: Are we meeting at 3 pm? Sam: Yes, I will send the link."
    summary_length = 32
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType 'application/json' -Body $payload -TimeoutSec 300
```

The response contains `summary`. `summary_length` is a maximum token count,
from 16 to 128 (default 64), not a word count. Input is limited to 20,000
characters and truncated to the configured 256 tokens. Review generated
summaries for accuracy.

Open http://localhost:8000/docs for interactive API documentation.
`GET /health` checks that the API is alive, not model readiness. The first
prediction loads the model and may take time on CPU.

## Project layout

```text
app/app.py                  FastAPI routes and web interface
config/config.yaml          Model, training, and path settings
src/text_summarizer/         Training and prediction components
main.py                     Training entry point
test.py                     Full test-set ROUGE evaluation
Dockerfile                  CPU inference image, Python 3.13
docker-compose.yml          Local service and read-only model mount
deploy/huggingface/          Gradio ZeroGPU app, Python 3.12
docs/                       Docker and training guides
```

## Verified behavior

Local prediction, Docker health and prediction requests, and the live Space
have been tested successfully. Full training and test-set evaluation were
not rerun during deployment finalization; no new ROUGE scores are claimed.
GitHub pushes do not update the Space automatically without additional sync.
