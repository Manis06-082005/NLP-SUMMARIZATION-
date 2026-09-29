from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.text_summarizer.components.prediction import PredictionPipeline


app = FastAPI(title="Dialogue Summarizer", version="1.0.0")


class PredictionRequest(BaseModel):
    dialogue: str = Field(min_length=1, max_length=20_000)


class PredictionResponse(BaseModel):
    summary: str


@lru_cache(maxsize=1)
def get_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dialogue Summarizer</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; color: #18211d; background: #f3f5f2; }
    header { padding: 18px 24px; color: white; background: #173d30; }
    header strong { font-size: 18px; }
    main { width: min(960px, calc(100% - 32px)); margin: 40px auto; }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 5vw, 46px); letter-spacing: 0; }
    .lead { margin: 0 0 28px; color: #54605a; font-size: 17px; }
    label { display: block; margin-bottom: 8px; font-weight: 700; }
    textarea { width: 100%; min-height: 270px; resize: vertical; padding: 16px;
      border: 1px solid #bdc6c0; border-radius: 6px; background: white;
      color: #18211d; font: 15px/1.55 inherit; }
    textarea:focus { outline: 3px solid #b9d9c8; border-color: #276749; }
    .actions { display: flex; align-items: center; gap: 14px; margin-top: 14px; }
    button { border: 0; border-radius: 6px; padding: 11px 18px; background: #276749;
      color: white; font: 700 15px inherit; cursor: pointer; }
    button:hover { background: #1f553b; }
    button:disabled { cursor: wait; opacity: .6; }
    #status { color: #68736d; }
    #result { display: none; margin-top: 30px; padding-top: 24px; border-top: 1px solid #ccd2ce; }
    #result h2 { margin: 0 0 10px; font-size: 18px; }
    #summary { margin: 0; padding: 18px; border-left: 4px solid #d38b32;
      background: white; line-height: 1.65; white-space: pre-wrap; }
    .error { color: #a12622 !important; }
  </style>
</head>
<body>
  <header><strong>Dialogue Summarizer</strong></header>
  <main>
    <h1>Turn conversations into clear summaries.</h1>
    <p class="lead">Paste a dialogue below and generate a concise summary with your trained Pegasus model.</p>
    <form id="form">
      <label for="dialogue">Dialogue</label>
      <textarea id="dialogue" required placeholder="Alex: Are we still meeting at 3?&#10;Sam: Yes, I'll send the link shortly."></textarea>
      <div class="actions">
        <button id="submit" type="submit">Summarize</button>
        <span id="status" role="status"></span>
      </div>
    </form>
    <section id="result" aria-live="polite">
      <h2>Summary</h2>
      <p id="summary"></p>
    </section>
  </main>
  <script>
    const form = document.querySelector('#form');
    const button = document.querySelector('#submit');
    const status = document.querySelector('#status');
    const result = document.querySelector('#result');
    const summary = document.querySelector('#summary');

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      status.className = '';
      status.textContent = 'Generating summary...';
      result.style.display = 'none';

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dialogue: document.querySelector('#dialogue').value })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Prediction failed.');
        summary.textContent = data.summary;
        result.style.display = 'block';
        status.textContent = 'Done';
      } catch (error) {
        status.className = 'error';
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    dialogue = request.dialogue.strip()
    if not dialogue:
        raise HTTPException(status_code=422, detail="Dialogue cannot be blank.")

    try:
        summary = get_pipeline().predict(dialogue)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(summary=summary)
