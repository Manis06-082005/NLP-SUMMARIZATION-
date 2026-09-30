from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.text_summarizer.components.prediction import PredictionPipeline


app = FastAPI(title="Dialogue Summarizer", version="1.0.0")


class PredictionRequest(BaseModel):
    dialogue: str = Field(min_length=1, max_length=20_000)
    summary_length: int = Field(default=64, ge=16, le=128)


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
    :root { color-scheme: light; font-family: "Segoe UI", "Aptos", Arial, sans-serif;
      --ink: #17231e; --muted: #66736c; --line: #d9e1db; --paper: #ffffff;
      --canvas: #f5f7f4; --forest: #123b2c; --leaf: #2e7655; --gold: #d28b35; }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: var(--canvas); }
    header { padding: 18px max(24px, calc((100vw - 1100px) / 2)); color: white;
      background: var(--forest); display: flex; align-items: center; justify-content: space-between; }
    header strong { font-size: 18px; letter-spacing: .01em; }
    header span { color: #b9d9c8; font-size: 13px; }
    main { width: min(1100px, calc(100% - 32px)); margin: 56px auto; }
    .intro { max-width: 720px; margin-bottom: 34px; }
    h1 { margin: 0 0 10px; color: var(--forest); font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(32px, 5vw, 56px); font-weight: 700; line-height: 1.05; letter-spacing: 0; }
    .lead { margin: 0; color: var(--muted); font-size: 18px; line-height: 1.55; }
    .workspace { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(250px, .6fr); gap: 22px; align-items: start; }
    .panel { padding: 24px; border: 1px solid var(--line); border-radius: 8px; background: var(--paper); box-shadow: 0 12px 30px rgba(18, 59, 44, .06); }
    label { display: block; margin-bottom: 8px; color: var(--ink); font-size: 14px; font-weight: 750; }
    textarea { width: 100%; min-height: 270px; resize: vertical; padding: 16px;
      border: 1px solid #bdc6c0; border-radius: 6px; background: white;
      color: var(--ink); font: 15px/1.55 inherit; }
    textarea:focus, select:focus { outline: 3px solid #b9d9c8; border-color: var(--leaf); }
    select { display: block; width: 100%; padding: 11px; border: 1px solid #bdc6c0;
      border-radius: 6px; background: white; color: var(--ink); font: 15px inherit; }
    .hint { margin: 8px 0 0; color: var(--muted); font-size: 13px; line-height: 1.4; }
    .actions { display: flex; align-items: center; gap: 14px; margin-top: 18px; }
    button { border: 0; border-radius: 6px; padding: 12px 20px; background: var(--leaf);
      color: white; font: 750 15px inherit; cursor: pointer; transition: background .2s, transform .2s; }
    button:hover { background: #225d42; transform: translateY(-1px); }
    button:disabled { cursor: wait; opacity: .6; }
    #status { color: #68736d; }
    #result { display: none; margin-top: 24px; }
    #result h2 { margin: 0 0 10px; color: var(--forest); font-size: 18px; }
    #summary { margin: 0; padding: 20px; border-left: 4px solid var(--gold);
      background: #fffdf8; line-height: 1.7; white-space: pre-wrap; }
    .side-title { margin: 0 0 16px; color: var(--forest); font-size: 16px; }
    .side-copy { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.6; }
    .model-note { margin-top: 22px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }
    .error { color: #a12622 !important; }
    @media (max-width: 720px) { main { margin: 36px auto; } .workspace { grid-template-columns: 1fr; }
      .panel { padding: 18px; } header span { display: none; } }
  </style>
</head>
<body>
  <header><strong>Dialogue Summarizer</strong><span>Powered by Pegasus</span></header>
  <main>
    <div class="intro"><h1>Turn conversations into clarity.</h1>
      <p class="lead">Paste a dialogue and generate a focused summary with your trained Pegasus model.</p></div>
    <div class="workspace">
      <section class="panel"><form id="form">
        <label for="dialogue">Dialogue</label>
        <textarea id="dialogue" required placeholder="Alex: Are we still meeting at 3?&#10;Sam: Yes, I'll send the link shortly."></textarea>
        <label for="summary-length">Summary size</label>
        <select id="summary-length"><option value="32">Short</option><option value="64" selected>Medium</option><option value="96">Long</option></select>
        <p class="hint">Choose how much detail to keep in the generated summary.</p>
        <div class="actions"><button id="submit" type="submit">Generate summary</button><span id="status" role="status"></span></div>
      </form><section id="result" aria-live="polite"><h2>Summary</h2><p id="summary"></p></section></section>
      <aside class="panel"><h2 class="side-title">A clearer next step</h2><p class="side-copy">Capture decisions, action items, and the essential thread of a conversation in seconds.</p><p class="model-note">Your saved fine-tuned Pegasus model runs locally through this app.</p></aside>
    </div>
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
          body: JSON.stringify({
            dialogue: document.querySelector('#dialogue').value,
            summary_length: Number(document.querySelector('#summary-length').value)
          })
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
        summary = get_pipeline().predict(dialogue, request.summary_length)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(summary=summary)
