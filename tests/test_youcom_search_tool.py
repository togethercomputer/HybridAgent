"""Tests for the optional You.com search tool. HTTP is mocked; no network or key needed."""
import email.message
import io
import json
import urllib.error

import pytest

from hybrid_agent.tools import executor as executor_mod
from hybrid_agent.tools import youcom_search_tool as ycm
from hybrid_agent.tools.youcom_search_tool import YouComSearchTool

RESPONSE = {
    "results": {
        "web": [
            {"url": "https://a.example", "title": "A", "description": "Paris is the capital of France.", "snippets": ["x"]},
            {"url": "https://b.example", "title": "B", "snippets": ["Second", "page."]},
        ]
    },
    "metadata": {"query": "capital of France"},
}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, headers=None):
    msg = email.message.Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.error.HTTPError(ycm.SEARCH_URL, code, "err", msg, io.BytesIO(b"{}"))


@pytest.fixture(autouse=True)
def _no_disk_cache_or_sleep(monkeypatch):
    monkeypatch.delenv("HYBRID_SEARCH_CACHE_DIR", raising=False)
    sleeps = []
    monkeypatch.setattr(ycm.time, "sleep", sleeps.append)
    return sleeps


def _mock_urlopen(monkeypatch, outcomes):
    """Each outcome is a dict (JSON body) or an exception to raise; records the requests."""
    calls = []
    outcomes = list(outcomes)

    def fake(req, timeout=None):
        calls.append(req)
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return _Resp(json.dumps(out).encode())

    monkeypatch.setattr(ycm.urllib.request, "urlopen", fake)
    return calls


def test_requires_key(monkeypatch):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="YDC_API_KEY"):
        YouComSearchTool()


def test_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("YDC_API_KEY", "env-key")
    calls = _mock_urlopen(monkeypatch, [RESPONSE])
    YouComSearchTool().execute("q")
    assert calls[0].get_header("X-api-key") == "env-key"


def test_request_shape_and_page_format(monkeypatch):
    calls = _mock_urlopen(monkeypatch, [RESPONSE])
    tool = YouComSearchTool(api_key="k", max_results=5)
    out = tool.execute("  capital of France ")
    assert out == "Page 1: Paris is the capital of France.\nPage 2: Second page."
    req = calls[0]
    assert req.full_url == "https://ydc-index.io/v1/search"
    assert req.get_method() == "POST"
    assert json.loads(req.data) == {"query": "capital of France", "count": 5}
    assert req.get_header("X-api-key") == "k"
    assert req.get_header("User-agent") is None
    assert tool.name == "youcom_search" and tool.trigger_tag == "search"


def test_truncates_and_limits(monkeypatch):
    _mock_urlopen(monkeypatch, [RESPONSE])
    out = YouComSearchTool(api_key="k", max_results=1, result_length=5).execute("q")
    assert out == "Page 1: Paris"


def test_in_memory_cache(monkeypatch):
    calls = _mock_urlopen(monkeypatch, [RESPONSE])
    tool = YouComSearchTool(api_key="k")
    assert tool.execute("q") == tool.execute("q")
    assert len(calls) == 1


def test_empty_results_and_query(monkeypatch):
    _mock_urlopen(monkeypatch, [{"results": {"web": []}}])
    tool = YouComSearchTool(api_key="k")
    assert tool.execute("  ") == "Empty search query."
    assert tool.execute("q") == "No search results found."


@pytest.mark.parametrize("code", [401, 402, 403])
def test_auth_and_payment_errors_fail_fast(monkeypatch, code):
    calls = _mock_urlopen(monkeypatch, [_http_error(code)] * 3)
    out = YouComSearchTool(api_key="k").execute("q")
    assert out.startswith("Search failed after")
    assert len(calls) == 1


def test_429_honors_retry_after(monkeypatch, _no_disk_cache_or_sleep):
    _mock_urlopen(monkeypatch, [_http_error(429, {"Retry-After": "7"}), RESPONSE])
    out = YouComSearchTool(api_key="k", retry_delay=2.0).execute("q")
    assert out.startswith("Page 1:")
    assert _no_disk_cache_or_sleep == [7.0]


def test_429_retry_after_is_capped_and_defaults(monkeypatch, _no_disk_cache_or_sleep):
    _mock_urlopen(monkeypatch, [_http_error(429, {"Retry-After": "9999"}), _http_error(429), RESPONSE])
    YouComSearchTool(api_key="k", retry_delay=2.0).execute("q")
    assert _no_disk_cache_or_sleep == [60.0, 2.0]


def test_failure_message_matches_retrying_wrapper(monkeypatch):
    _mock_urlopen(monkeypatch, [_http_error(500)] * 3)
    out = YouComSearchTool(api_key="k", max_retries=3).execute("q")
    assert out == "Search failed after 3 attempts for query: q"


def test_executor_default_is_tavily(monkeypatch):
    monkeypatch.delenv("HYBRID_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("YDC_API_KEY", "k")
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        executor_mod.build_tool_executor()


def test_executor_selects_youcom_by_env(monkeypatch):
    monkeypatch.setenv("HYBRID_SEARCH_PROVIDER", "youcom")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("YDC_API_KEY", "k")
    ex = executor_mod.build_tool_executor()
    assert ex.get_tool("search").name == "youcom_search"


def test_executor_selects_youcom_by_argument(monkeypatch):
    monkeypatch.delenv("HYBRID_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("YDC_API_KEY", "k")
    ex = executor_mod.build_tool_executor(search_provider="youcom")
    assert ex.get_tool("search").name == "youcom_search"


def test_executor_youcom_without_key(monkeypatch):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="YDC_API_KEY"):
        executor_mod.build_tool_executor(search_provider="youcom")
    ex = executor_mod.build_tool_executor(search_provider="youcom", require_search=False)
    assert ex.get_tool("search") is None


def test_executor_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown search provider"):
        executor_mod.build_tool_executor(search_provider="bing")
