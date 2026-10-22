"""Local web server: serves the browser app and proxies gloss -> sentence LLM calls.

Recognition runs entirely in the browser; the server only holds the LLM API key.

    python -m server            # http://localhost:8000
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from isl.translate import llm_settings, translate

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

app = FastAPI(title="ISL Translator")


class TranslateRequest(BaseModel):
    labels: list[str] = Field(..., max_length=50)
    # per sign, the ranked guesses [[label, probability], ...], best first
    alternatives: list[list[tuple[str, float]]] | None = Field(None, max_length=50)


@app.get("/api/health")
def health():
    settings = llm_settings()
    return {"llm": bool(settings["api_key"]), "model": settings["model"]}


@app.post("/api/translate")
def translate_glosses(request: TranslateRequest):
    return translate(request.labels, request.alternatives)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
