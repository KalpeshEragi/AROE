import pytest

from lrm.api import openai_compat as oa


def test_chat_request_maps_options():
    body = oa.chat_request_to_ollama({
        "model": "qwen3.5:4b", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32, "temperature": 0, "stop": "END", "seed": 1,
        "think": False, "keep_alive": "10m", "options": {"num_ctx": 8192}})
    assert body["options"] == {"num_predict": 32, "temperature": 0, "seed": 1,
                               "stop": ["END"], "num_ctx": 8192}
    assert body["think"] is False and body["keep_alive"] == "10m"
    assert "stream" not in body  # the adapter sets it


@pytest.mark.parametrize("req, msg", [
    ({"messages": [{"role": "user", "content": "x"}]}, "model"),
    ({"model": "m", "messages": []}, "messages"),
    ({"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "string"),
    ({"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": [{}]}, "tools"),
    ({"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 2}, "n=1"),
])
def test_chat_request_rejects_unsupported(req, msg):
    with pytest.raises(oa.BadRequest, match=msg):
        oa.chat_request_to_ollama(req)


def test_stream_chunks():
    first = oa.stream_chunk({"message": {"content": "Hel"}}, "id", 1, "m", first=True)
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "Hel"}
    think = oa.stream_chunk({"message": {"thinking": "hmm"}}, "id", 1, "m", first=False)
    assert think["choices"][0]["delta"] == {"reasoning_content": "hmm"}
    assert oa.stream_chunk({"message": {"content": ""}}, "id", 1, "m", first=False) is None
    last = oa.stream_chunk({"message": {"content": ""}, "done": True, "done_reason": "length"},
                           "id", 1, "m", first=False)
    assert last["choices"][0]["finish_reason"] == "length"


def test_chat_response_and_usage():
    r = oa.chat_response({"message": {"content": "ok"}, "done_reason": "stop",
                          "prompt_eval_count": 5, "eval_count": 2}, "id", 1, "m")
    assert r["choices"][0]["message"]["content"] == "ok"
    assert r["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_timings_convert_ns_to_seconds():
    t = oa.timings_from({"load_duration": 1_500_000_000, "eval_count": 3, "eval_duration": 250_000_000})
    assert t == {"load_s": 1.5, "eval_s": 0.25, "eval_count": 3}


def test_embeddings_roundtrip():
    body = oa.embed_request_to_ollama({"model": "bge-m3", "input": "hello"})
    assert body == {"model": "bge-m3", "input": ["hello"]}
    out = oa.embed_response({"embeddings": [[0.1, 0.2]], "prompt_eval_count": 3}, "bge-m3")
    assert out["data"][0] == {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
    with pytest.raises(oa.BadRequest):
        oa.embed_request_to_ollama({"model": "bge-m3", "input": "x", "encoding_format": "base64"})
    with pytest.raises(oa.BadRequest):
        oa.embed_request_to_ollama({"model": "bge-m3", "input": [[1, 2, 3]]})
