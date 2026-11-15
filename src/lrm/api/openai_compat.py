"""OpenAI request/response shapes <-> Ollama native API. Pure functions.

Supported in M1: text chat (streaming and not), embeddings with float output.
Not supported yet, rejected with a clear 400: images/multi-part content, tools,
n > 1, base64 embeddings. Non-standard extensions accepted at the top level of the
request: `think` (bool), `keep_alive`, and `options` (raw Ollama options).
"""
from __future__ import annotations

from typing import Any

# OpenAI field -> Ollama option name
_OPTION_MAP = {
    "max_tokens": "num_predict",
    "max_completion_tokens": "num_predict",
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


class BadRequest(ValueError):
    pass


def error_body(message: str, type_: str = "invalid_request_error", code: str | None = None) -> dict:
    return {"error": {"message": message, "type": type_, "code": code}}


def chat_request_to_ollama(req: dict[str, Any]) -> dict[str, Any]:
    model = req.get("model")
    if not model or not isinstance(model, str):
        raise BadRequest("'model' is required")
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest("'messages' must be a non-empty list")
    if req.get("tools") or req.get("functions"):
        raise BadRequest("tools are not supported by this gateway yet")
    if req.get("n", 1) != 1:
        raise BadRequest("only n=1 is supported")
    out_msgs = []
    for i, m in enumerate(msgs):
        content = m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise BadRequest(f"messages[{i}].content must be a string (images/multi-part not supported yet)")
        out_msgs.append({"role": m.get("role", "user"), "content": content})

    options: dict[str, Any] = {}
    for k, ok in _OPTION_MAP.items():
        if req.get(k) is not None:
            options[ok] = req[k]
    stop = req.get("stop")
    if stop is not None:
        options["stop"] = [stop] if isinstance(stop, str) else list(stop)
    if isinstance(req.get("options"), dict):
        options.update(req["options"])

    body: dict[str, Any] = {"model": model, "messages": out_msgs}
    if options:
        body["options"] = options
    for k in ("think", "keep_alive"):
        if k in req:
            body[k] = req[k]
    return body


def _finish_reason(done_reason: str | None) -> str:
    return "length" if done_reason == "length" else "stop"


def usage_from(ollama: dict[str, Any]) -> dict[str, int]:
    p, c = ollama.get("prompt_eval_count") or 0, ollama.get("eval_count") or 0
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def timings_from(ollama: dict[str, Any]) -> dict[str, Any]:
    """Backend-measured timings, nanoseconds -> seconds. Only keys Ollama reported."""
    out = {}
    for k in ("load_duration", "prompt_eval_duration", "eval_duration", "total_duration"):
        if ollama.get(k) is not None:
            out[k.replace("_duration", "_s")] = round(ollama[k] / 1e9, 4)
    for k in ("prompt_eval_count", "eval_count", "done_reason"):
        if ollama.get(k) is not None:
            out[k] = ollama[k]
    return out


def stream_chunk(chunk: dict[str, Any], cid: str, created: int, model: str, first: bool) -> dict[str, Any] | None:
    """One Ollama /api/chat stream line -> one OpenAI chat.completion.chunk (or None)."""
    msg = chunk.get("message") or {}
    delta: dict[str, Any] = {}
    if first:
        delta["role"] = "assistant"
    if msg.get("content"):
        delta["content"] = msg["content"]
    if msg.get("thinking"):
        delta["reasoning_content"] = msg["thinking"]  # same convention as several OpenAI-style servers
    done = bool(chunk.get("done"))
    if not delta and not done:
        return None
    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta,
                         "finish_reason": _finish_reason(chunk.get("done_reason")) if done else None}]}


def chat_response(resp: dict[str, Any], cid: str, created: int, model: str) -> dict[str, Any]:
    msg = resp.get("message") or {}
    out_msg: dict[str, Any] = {"role": "assistant", "content": msg.get("content", "")}
    if msg.get("thinking"):
        out_msg["reasoning_content"] = msg["thinking"]
    return {"id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": out_msg,
                         "finish_reason": _finish_reason(resp.get("done_reason"))}],
            "usage": usage_from(resp)}


def embed_request_to_ollama(req: dict[str, Any]) -> dict[str, Any]:
    model = req.get("model")
    if not model or not isinstance(model, str):
        raise BadRequest("'model' is required")
    inp = req.get("input")
    if isinstance(inp, str):
        inp = [inp]
    if not isinstance(inp, list) or not inp or not all(isinstance(x, str) for x in inp):
        raise BadRequest("'input' must be a string or a non-empty list of strings (token arrays not supported)")
    if req.get("encoding_format", "float") != "float":
        raise BadRequest("only encoding_format='float' is supported")
    body: dict[str, Any] = {"model": model, "input": inp}
    if "keep_alive" in req:
        body["keep_alive"] = req["keep_alive"]
    if isinstance(req.get("options"), dict):
        body["options"] = req["options"]
    return body


def embed_response(resp: dict[str, Any], model: str) -> dict[str, Any]:
    vecs = resp.get("embeddings") or []
    n = resp.get("prompt_eval_count") or 0
    return {"object": "list", "model": model,
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": n, "total_tokens": n}}
