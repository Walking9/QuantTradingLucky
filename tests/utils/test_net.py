"""Tests for the per-provider HTTP session factory."""

from __future__ import annotations

import os

import requests

from quant_lucky.utils.net import ProxyPolicy, build_session, bypass_proxy_env


def test_bypass_clears_proxies() -> None:
    s = build_session(ProxyPolicy.BYPASS)
    # Both proxy keys explicitly emptied so requests' env-merge can't fill them in.
    assert s.proxies == {"http": "", "https": ""}
    assert s.trust_env is False


def test_force_sets_proxy_url() -> None:
    s = build_session(ProxyPolicy.FORCE, proxy_url="http://127.0.0.1:7897")
    assert s.proxies == {
        "http": "http://127.0.0.1:7897",
        "https": "http://127.0.0.1:7897",
    }
    assert s.trust_env is False


def test_force_requires_url() -> None:
    try:
        build_session(ProxyPolicy.FORCE)
    except ValueError as e:
        assert "proxy_url" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_use_env_keeps_trust_env() -> None:
    s = build_session(ProxyPolicy.USE_ENV)
    assert s.trust_env is True


def test_retry_adapter_attached() -> None:
    s = build_session(ProxyPolicy.BYPASS, retries=5, backoff=1.0)
    adapter = s.get_adapter("https://example.com")
    # urllib3 Retry is wrapped in HTTPAdapter.max_retries
    assert adapter.max_retries.total == 5
    assert adapter.max_retries.backoff_factor == 1.0
    # We want 429 (rate limit) explicitly retried since it's the most
    # common transient failure on scraping endpoints.
    assert 429 in adapter.max_retries.status_forcelist


def test_session_is_a_requests_session() -> None:
    s = build_session(ProxyPolicy.BYPASS)
    assert isinstance(s, requests.Session)


def test_bypass_proxy_env_clears_then_restores() -> None:
    os.environ["HTTP_PROXY"] = "http://example:1234"
    os.environ["HTTPS_PROXY"] = "http://example:1234"
    try:
        assert os.environ.get("HTTP_PROXY") == "http://example:1234"
        with bypass_proxy_env():
            assert "HTTP_PROXY" not in os.environ
            assert "HTTPS_PROXY" not in os.environ
            assert os.environ.get("NO_PROXY") == "*"
        # Restored after exit
        assert os.environ.get("HTTP_PROXY") == "http://example:1234"
        assert os.environ.get("HTTPS_PROXY") == "http://example:1234"
    finally:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)


def test_bypass_proxy_env_restores_on_exception() -> None:
    def _inner() -> None:
        raise RuntimeError("boom")

    os.environ["HTTP_PROXY"] = "http://example:1234"
    try:
        try:
            with bypass_proxy_env():
                assert "HTTP_PROXY" not in os.environ
                _inner()
        except RuntimeError:
            pass
        assert os.environ.get("HTTP_PROXY") == "http://example:1234"
    finally:
        os.environ.pop("HTTP_PROXY", None)
