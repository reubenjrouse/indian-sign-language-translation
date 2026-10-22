"""Gloss sequence -> English sentence via an LLM (paper Sec. III-E).

Talks to any OpenAI-compatible chat-completions endpoint (Groq by default).
Configure with environment variables:
    LLM_API_KEY   (falls back to GROQ_API_KEY)
    LLM_BASE_URL  default https://api.groq.com/openai/v1
    LLM_MODEL     default llama-3.3-70b-versatile
"""

import os

import httpx

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
NO_TRANSLATION = "No Translation"

# The paper's prompt, plus a note on ISL word order (listed there as future work).
SYSTEM_PROMPT = f"""You are a specialized translator for Indian Sign Language (ISL) glosses. \
When given gloss sequences, convert them into grammatically correct English sentences that \
preserve the semantic meaning. Follow these rules:
- Create natural, fluent English translations from the input glosses
- Maintain the original meaning and intent of the signs
- Add appropriate articles, conjunctions, and verb forms
- ISL commonly uses Subject-Object-Verb order and omits articles and the verb "to be"; \
reorder into natural English as needed
- Glosses name a sign, not a part of speech: e.g. MEAN is usually the adjective "mean" (unkind)
- Some signs may be listed with alternatives the recogniser was less sure of, as WE (or THEY 7%). Use whichever reading makes the most coherent sentence; prefer the first unless the alternative is clearly better
- Respond only with the translated sentence
- If unable to form a coherent translation, respond with "{NO_TRANSLATION}\""""


def to_gloss(label: str) -> str:
    """'Good Morning' -> 'GOOD-MORNING'."""
    return "-".join(label.strip().upper().split())


def llm_settings() -> dict:
    return {
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
    }


# Runner-up guesses are shown to the LLM only when they are plausible and the top guess is not sure.
ALT_MIN_PROB = 0.05
ALT_IF_TOP_BELOW = 0.9


def describe(labels: list[str], alternatives: list[list] | None = None) -> str:
    """The user message: glosses in order, uncertain ones with their alternatives.

    alternatives[i] is the ranked guesses for sign i: [[label, prob], ...], best first.
    """
    parts = []
    for i, label in enumerate(labels):
        part = to_gloss(label)
        ranked = alternatives[i] if alternatives and i < len(alternatives) else []
        top = ranked[0][1] if ranked else 1.0
        alts = [(l, p) for l, p in ranked[1:] if p >= ALT_MIN_PROB] if top < ALT_IF_TOP_BELOW else []
        if alts:
            part += " (or " + ", ".join(f"{to_gloss(l)} {round(p * 100)}%" for l, p in alts) + ")"
        parts.append(part)
    return " ".join(parts)


def clean_reply(text: str) -> str:
    line = next((l for l in text.strip().splitlines() if l.strip()), "")
    return line.strip().strip('"').strip("'").strip()


def translate(labels: list[str], alternatives: list[list] | None = None,
              temperature: float = 0.3, timeout: float = 20.0) -> dict:
    """Return {"sentence", "glosses", "source", "error", "model"}.

    alternatives: optional ranked guesses per sign (see describe), so the LLM can use
    context to fix a likely recognition error such as I read as YOU.
    source is "llm" on success, otherwise "fallback" (the glosses joined) with
    the reason in "error" so callers can show it instead of failing silently.
    """
    glosses = " ".join(to_gloss(l) for l in labels)
    settings = llm_settings()
    result = {"sentence": "", "glosses": glosses, "source": "fallback", "error": None, "model": settings["model"]}
    if not labels:
        return result
    fallback = " ".join(l.strip() for l in labels)

    if not settings["api_key"]:
        return {**result, "sentence": fallback, "error": "No LLM API key configured"}
    try:
        response = httpx.post(
            f"{settings['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {settings['api_key']}"},
            json={
                "model": settings["model"],
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": describe(labels, alternatives)},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        sentence = clean_reply(response.json()["choices"][0]["message"]["content"])
    except Exception as e:  # network, auth, retired model, malformed reply
        return {**result, "sentence": fallback, "error": f"{type(e).__name__}: {e}"}
    if not sentence or sentence.rstrip(".") == NO_TRANSLATION:
        return {**result, "sentence": fallback, "error": "LLM could not translate this sequence"}
    return {**result, "sentence": sentence, "source": "llm"}
