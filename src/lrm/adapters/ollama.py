"""Talks to Ollama's native API (/api/chat, /api/embed, /api/tags) using aiohttp.

Native rather than Ollama's OpenAI-compatible endpoint because only the native API takes
keep_alive and options.num_ctx and returns load/prefill/decode timings.

aiohttp rather than httpx: in the gateway, httpx added ~11.7 ms to streaming time-to-first-
token vs ~1.5 ms with aiohttp (docs/M1_GATEWAY.md, bench/m1_overhead.py).

The gateway sees only BackendError and ChatStream (`lines()`, `aclose()`), never aiohttp
types, so the client library can be swapped again without touching the gateway.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Protocol

import aiohttp

# total=None: a streaming generation may legitimately run for minutes.
# sock_read=300: max silence between bytes. Model loads took up to ~15 s and a crash
# ~25 s to surface (Phase 1, E8), so this must comfortably exceed both.
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5, sock_read=300)
_NET_ERRORS = (aiohttp.ClientError, ConnectionError, TimeoutError)


class BackendError(Exception):
    def __init__(self, status: int, message: str, kind: str = "backend_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


class ChatStream(Protocol):
    def lines(self) -> AsyncIterator[str]: ...
    async def aclose(self) -> None: ...


def error_message(resp_text: str) -> str:
    try:
        return json.loads(resp_text).get("error", resp_text)
    except (ValueError, AttributeError):
        return resp_text


class _AiohttpChatStream:
    def __init__(self, resp: aiohttp.ClientResponse):
        self._resp = resp

    async def lines(self) -> AsyncIterator[str]:
        try:
            async for raw in self._resp.content:  # one line per item, newline included
                yield raw.decode("utf-8", "replace").rstrip("\r\n")
        except _NET_ERRORS as e:
            raise BackendError(502, f"backend connection lost: {e!r}", "backend_connection_lost")

    async def aclose(self) -> None:
        # close(), not release(), when the body was not fully read (client disconnected):
        # the TCP connection must actually close so Ollama notices and stops generating.
        if self._resp.content.at_eof():
            self._resp.release()
        else:
            self._resp.close()


class OllamaAdapter:
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    def _s(self) -> aiohttp.ClientSession:
        # Created lazily: a ClientSession must be created inside the running event loop.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(base_url=self.base_url, timeout=DEFAULT_TIMEOUT)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()

    async def open_chat_stream(self, body: dict[str, Any]) -> ChatStream:
        """Returns an open stream; the caller must aclose() it.

        Errors that Ollama reports before streaming starts (e.g. a failed model load,
        HTTP 500) are raised here, so the gateway can still send a proper HTTP error.
        """
        try:
            resp = await self._s().post("/api/chat", json={**body, "stream": True})
        except _NET_ERRORS as e:
            raise BackendError(502, f"cannot reach Ollama at {self.base_url}: {e!r}", "backend_unreachable")
        if resp.status != 200:
            text = await resp.text()
            resp.release()
            raise BackendError(resp.status, error_message(text))
        return _AiohttpChatStream(resp)

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/api/chat", {**body, "stream": False})

    async def embed(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/api/embed", body)

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            async with self._s().get("/api/tags") as r:
                return (await r.json(content_type=None)).get("models", [])
        except _NET_ERRORS as e:
            raise BackendError(502, f"cannot reach Ollama: {e!r}", "backend_unreachable")

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._s().post(path, json=body) as r:
                if r.status != 200:
                    raise BackendError(r.status, error_message(await r.text()))
                return await r.json(content_type=None)
        except _NET_ERRORS as e:
            raise BackendError(502, f"cannot reach Ollama at {self.base_url}: {e!r}", "backend_unreachable")
