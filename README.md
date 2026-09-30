# Dialogue Summarization

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
