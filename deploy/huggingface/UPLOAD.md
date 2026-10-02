# Deploy on Hugging Face ZeroGPU

Use two repositories: a **model repository** for weights and a **Space** for
the Gradio app. This avoids the Space storage limit encountered when uploading
the 2.28 GB model directly. Check current account eligibility and quotas in
[the ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu).

## 1. Upload the model

The project's model repository is
https://huggingface.co/nathmanish/PegasusSamsum.
For your own account, create a model repository at https://huggingface.co/new.
In **Files and versions > Add file > Upload files**, upload the saved
`final_model` folder. This deployment expects:

```text
final_model/config.json
final_model/model.safetensors
final_model/special_tokens_map.json
final_model/spiece.model
final_model/tokenizer_config.json
```

Commit changes and keep the tab open until the large upload finishes.
Public repositories make weights downloadable. Check applicable model and
dataset licenses before selecting a license for your own uploaded model.

## 2. Create a Space

At https://huggingface.co/new-space choose **Gradio**, **Blank**, **ZeroGPU**,
and Public visibility for a public demo. Leave storage buckets and Dev Mode
disabled. The existing Space is https://huggingface.co/spaces/nathmanish/NLP.

## 3. Set variables

In **Space Settings > Variables and secrets**, add:

| Variable | Value for this project |
| --- | --- |
| `MODEL_ID` | `nathmanish/PegasusSamsum` |
| `MODEL_SUBFOLDER` | `final_model` |

The app defaults to these values. Change `MODEL_ID` for your own model.
If files are at the model repository root, set `MODEL_SUBFOLDER` to an empty
string. Private models also need an `HF_TOKEN` secret with read access.

## 4. Upload app files

Upload only `app.py`, `requirements.txt`, and `README.md` from this folder to
the Space root. Replace the initial Space README with this deployment README.
Its YAML metadata specifies Gradio, Python 3.12, and `app_file: app.py`.
Do not upload the project's root README, training requirements, weights,
`UPLOAD.md`, Docker files, or a virtual environment to the Space.

Commit changes. Watch **Logs** while the Space builds, downloads the model,
packs ZeroGPU tensors, and starts Gradio on port 7860.

## 5. Verify

Wait for **Running**, open App, and submit:

```text
Alex: Are we meeting at 3 pm?
Sam: Yes, I will send the meeting link shortly.
```

Confirm that a summary is returned. Share the public Space URL;
`share=True` is unnecessary. GPU requests may queue and have daily quotas.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Space storage limit reached | Store weights in the model repository and upload only app code to the Space |
| No application file | Commit `app.py` at the root and use the deployment README |
| `NatConfig` error for this model | Check `MODEL_ID` and `MODEL_SUBFOLDER`; model config is inside `final_model` |
| Model files missing | Check all five paths in the model repository |
| Model access denied | Check repository access and the `HF_TOKEN` secret |
| CUDA initialization error | Confirm ZeroGPU hardware and `import spaces` before torch |
| GPU quota exceeded | Wait for the reset or use local Docker inference |

Editing app files and committing triggers deployment. A GitHub push does not
automatically update the Space unless you configure synchronization.
