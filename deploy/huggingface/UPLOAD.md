# Deploy to nathmanish/NLP

1. Open https://huggingface.co/spaces/nathmanish/NLP/tree/main.
2. Choose **Add file → Upload files**. Upload `app.py`, `requirements.txt`, and
   `README.md` from this folder to the Space root. Do not upload `UPLOAD.md`.
   Replace the Space README with this deployment README.
3. From your local `models/pegasus_samsum/final_model/` folder, upload these
   five files into a `final_model/` folder in the Space (not into the root):
   - config.json
   - model.safetensors
   - special_tokens_map.json
   - spiece.model
   - tokenizer_config.json
4. Commit the uploads. The 2.28 GB weights upload may take time. Your model
   files will be public in a public Space.
5. Confirm **Settings → Hardware → ZeroGPU**, then check the build logs.
6. When the Space is running, open **App** and test a short dialogue.

The app cannot start until all model files are uploaded. This route does not
require a storage bucket, Docker image, or Vercel deployment.
