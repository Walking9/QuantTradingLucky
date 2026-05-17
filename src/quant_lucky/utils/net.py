"""Per-provider HTTP session factory with explicit proxy policy.

Different data sources have opposite proxy needs:

* Chinese endpoints (Eastmoney/Sina/Tencent used by AkShare; tushare.pro):
  must **bypass** the developer's outbound VPN proxy, otherwise the
  proxy returns ``RemoteDisconnected`` / timeouts because it does not
  route CN traffic.
* US endpoints (Yahoo Finance, Alpha Vantage, Polygon): from mainland
  China usually **require** the proxy.

Process-wide ``HTTP_PROXY`` / ``HTTPS_PROXY`` env vars cannot satisfy
both at once. The fix is to let every provider build its own
``requests.Session`` via :func:`build_session` and explicitly state the
policy. ``Session.proxies`` overrides the environment, so we control
proxy behaviour deterministically regardless of ``.env``.

We also bake in HTTP retries with exponential backoff so transient 502 /
RemoteDisconnected from upstream scraping APIs do not propagate.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from quant_lucky.utils.config import settings
from quant_lucky.utils.logging import logger

_PROXY_ENV_KEYS: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)


class ProxyPolicy(StrEnum):
    """How a provider session should resolve its HTTP proxy.

    * ``USE_ENV``: honour ``HTTP_PROXY`` / ``HTTPS_PROXY`` from settings
      (for endpoints behind a regional firewall).
    * ``BYPASS``: explicitly set ``Session.proxies`` to empty dicts so
      requests cannot pick the env proxy up later.
    * ``FORCE``: use the proxy URL supplied to :func:`build_session`
      regardless of environment — useful for testing or when the user
      configures a per-provider proxy.
    """

    USE_ENV = "use_env"
    BYPASS = "bypass"
    FORCE = "force"


_DEFAULT_RETRY_STATUSES: tuple[int, ...] = (429, 500, 502, 503, 504)


def build_session(
    policy: ProxyPolicy = ProxyPolicy.BYPASS,
    *,
    proxy_url: str | None = None,
    retries: int = 3,
    backoff: float = 0.5,
    retry_statuses: tuple[int, ...] = _DEFAULT_RETRY_STATUSES,
    timeout_marker: float | None = None,
) -> requests.Session:
    """Construct a ``requests.Session`` with the given proxy policy + retries.

    Args:
        policy: How to resolve the proxy. ``BYPASS`` is the safe default
            for CN-hosted endpoints because it guarantees no proxy is
            used even when ``HTTP_PROXY`` is set in the environment.
        proxy_url: Required when ``policy == FORCE``. Ignored otherwise.
        retries: Total retry attempts on connection errors or any of
            ``retry_statuses``.
        backoff: ``backoff_factor`` in ``urllib3.Retry`` — wait
            ``backoff * (2 ** (attempt-1))`` seconds between retries.
        retry_statuses: HTTP status codes that trigger a retry.
        timeout_marker: Optional float stored as ``session.timeout``
            so call sites can read a project-wide default. Not enforced
            by ``requests`` itself.

    Returns:
        A configured ``Session``. The caller owns its lifecycle.
    """
    session = requests.Session()

    if policy is ProxyPolicy.BYPASS:
        # Empty string overrides env-derived proxies in requests' env-merge.
        session.proxies = {"http": "", "https": ""}
        # Make absolutely sure: also set trust_env=False so the urllib3
        # connection pool won't consult NO_PROXY/HTTP_PROXY again.
        session.trust_env = False
    elif policy is ProxyPolicy.USE_ENV:
        # trust_env=True is the default; explicit for readability.
        session.trust_env = True
        if settings.http_proxy or settings.https_proxy:
            # Mirror env values onto the session so they survive even if
            # someone later unsets the process env.
            session.proxies = {
                "http": settings.http_proxy or "",
                "https": settings.https_proxy or settings.http_proxy or "",
            }
    elif policy is ProxyPolicy.FORCE:
        if not proxy_url:
            raise ValueError("ProxyPolicy.FORCE requires proxy_url")
        session.proxies = {"http": proxy_url, "https": proxy_url}
        session.trust_env = False

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=retry_statuses,
        allowed_methods=frozenset(["GET", "HEAD", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if timeout_marker is not None:
        # Stash on the session for callers; requests ignores this attr.
        session.timeout = timeout_marker  # type: ignore[attr-defined]

    logger.debug(
        "build_session policy={pol} retries={r} backoff={b} proxies={p}",
        pol=policy.value,
        r=retries,
        b=backoff,
        p=session.proxies,
    )
    return session


def ccxt_proxies(policy: ProxyPolicy) -> dict[str, str] | None:
    """Return a ``proxies`` dict suitable for ``ccxt.Exchange({'proxies': ...})``.

    ccxt does not take a ``requests.Session``; it accepts a proxy dict
    that mirrors requests' format. We honour the same policy so crypto
    exchanges can share the per-provider proxy model.
    """
    if policy is ProxyPolicy.BYPASS:
        return {"http": "", "https": ""}
    if policy is ProxyPolicy.USE_ENV:
        if not (settings.http_proxy or settings.https_proxy):
            return None
        return {
            "http": settings.http_proxy or "",
            "https": settings.https_proxy or settings.http_proxy or "",
        }
    raise ValueError("ccxt_proxies does not support FORCE policy directly")


@contextmanager
def bypass_proxy_env() -> Iterator[None]:
    """Temporarily strip proxy env vars so libraries that bypass our session honour it.

    Some third-party packages (akshare, tushare) call ``requests.get`` directly
    without giving us a chance to inject a session. They will pick up
    ``HTTP_PROXY`` / ``HTTPS_PROXY`` from the process environment, which is
    wrong for CN-hosted endpoints when the user has a VPN proxy configured.

    This contextmanager saves and clears those env vars for the duration of
    the block, then restores them on exit (even if the block raises).
    """
    saved: dict[str, str] = {}
    for key in _PROXY_ENV_KEYS:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    # Hint to libraries that consult NO_PROXY: bypass all hosts.
    prior_no_proxy = os.environ.get("NO_PROXY")
    os.environ["NO_PROXY"] = "*"
    try:
        yield
    finally:
        for key, value in saved.items():
            os.environ[key] = value
        if prior_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = prior_no_proxy
