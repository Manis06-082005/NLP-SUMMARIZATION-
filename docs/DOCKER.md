# Docker setup

## Requirements

Start Docker Desktop with Linux containers and confirm `docker info` works.
Download the five model files following the [README](../README.md#download-the-saved-model).
They must be in `models/pegasus_samsum/final_model/` before prediction.
Allow enough disk and RAM for the model and PyTorch; start with 8 GB available
to Docker and monitor actual usage. Port 8000 must be available.

## Build and start

From the repository root:

```powershell
docker compose up -d --build
docker compose ps
docker compose logs -f summarizer
```

The first build needs internet access. Wait for `Uvicorn running`, then open
http://localhost:8000 and submit a short conversation. Ctrl+C exits the log
view while the container continues running. A host Python environment is
unnecessary if Docker and the model files are already ready.

## Start an existing image

Compose expects `dialouge-summarizer:production`, matching the existing image
spelling. If you have only the `working` tag:

```powershell
docker images
docker tag dialouge-summarizer:working dialouge-summarizer:production
docker compose up -d --no-build
```

Tagging does not copy the image. The verified working image can differ from a
fresh build because dependency requirements are currently unpinned.

## Verify and manage

```powershell
Invoke-RestMethod http://localhost:8000/health
docker compose logs --tail 50 summarizer
docker stats --no-stream
```

Test a summary through the web interface or the README's `/predict` example.
Health alone does not load the model. The first prediction can be slow on CPU.

```powershell
# Stop and remove the container and Compose network; keep image and weights
docker compose down

# Rebuild after changing app code or dependencies
docker compose up -d --build
```

## How it works

The Dockerfile installs CPU-only PyTorch on Python 3.13 and starts FastAPI
through Uvicorn. Compose maps host port 8000 to container port 8000 and mounts
the local model folder read-only. Newly built images do not include the model.
Runtime inference is offline; missing weights are not downloaded automatically.

## Troubleshooting

| Error | Fix |
| --- | --- |
| Cannot connect to `dockerDesktopLinuxEngine` | Start Docker Desktop and wait for its engine; run `docker info` |
| Image missing / pull access denied | Build the production image or tag your existing image as above |
| Model cannot load | Check all five saved files and the Compose mount path |
| Port 8000 already allocated | Stop the other app, or map `"8001:8000"` and open port 8001 |
| Container exits or stalls | Check logs, `docker stats`, and Docker Desktop memory allocation |

## Remote Docker deployment

The local bind mount cannot be used by a remote host. Copy model files to the
remote server and run Compose there, or adapt model loading to download the
published model. Managed services may also require an image registry and port
configuration. The current offline image does not download weights from the
Hub. Choose memory based on measured inference usage, not image size.
Provider free tiers are not a guarantee this model will fit. The project's
existing free public deployment uses Gradio on Hugging Face ZeroGPU.
