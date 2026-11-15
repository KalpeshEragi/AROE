"""Instant fake of Ollama's /api/chat, for measuring the gateway's own overhead.

python bench/fake_ollama.py 18090
"""
import json
import sys

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

DONE = {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
        "prompt_eval_count": 4, "eval_count": 1}


async def chat(request: Request):
    body = await request.json()
    if not body.get("stream"):
        return JSONResponse({**DONE, "message": {"role": "assistant", "content": "ok"}})

    async def gen():
        yield json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": False}) + "\n"
        yield json.dumps(DONE) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


app = Starlette(routes=[Route("/api/chat", chat, methods=["POST"])])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
