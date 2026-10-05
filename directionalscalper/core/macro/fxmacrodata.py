"""FXMacroData release-calendar helpers for event-risk filters."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import requests

FXMACRODATA_BASE_URL = "https://api.fxmacrodata.com/v1"


class FXMacroDataError(RuntimeError):
    """Raised when FXMacroData returns something this helper cannot use."""


def _clean_api_key(api_key: Optional[str]) -> Optional[str]:
    """Strip the key and reject values requests would echo in an error."""

    if not api_key:
        return None
    api_key = api_key.strip()
    if not api_key or any(ch.isspace() or ord(ch) < 32 for ch in api_key):
        raise FXMacroDataError(
            "FXMacroData API key is empty or contains whitespace or control characters"
        )
    return api_key


def fetch_release_calendar(
    currency: str = "usd",
    *,
    limit: int = 50,
    min_tier: Optional[int] = 1,
    api_key: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Fetch official macro release events from FXMacroData."""

    limit_count = max(1, min(int(limit), 100))
    params: dict[str, str] = {"limit": str(limit_count)}
    headers: dict[str, str] = {}
    token = _clean_api_key(api_key or os.getenv("FXMACRODATA_API_KEY"))
    if token:
        headers["X-API-Key"] = token

    # requests strips Authorization on a cross-host redirect but not custom
    # headers, so redirects are refused rather than followed with the key.
    response = requests.get(
        f"{FXMACRODATA_BASE_URL}/calendar/{currency.lower()}",
        params=params,
        headers=headers,
        timeout=20,
        allow_redirects=False,
    )
    if 300 <= response.status_code < 400:
        raise FXMacroDataError(
            f"FXMacroData returned an unexpected redirect (HTTP {response.status_code})"
        )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        raise FXMacroDataError("FXMacroData returned a response that is not JSON") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        detail = payload.get("detail") if isinstance(payload, dict) else None
        message = "FXMacroData returned an unexpected response shape"
        if isinstance(detail, str):
            message += f": {detail}"
        raise FXMacroDataError(message)

    events = [event for event in payload["data"] if isinstance(event, dict)]
    if min_tier is None:
        return events[:limit_count]

    return [
        event
        for event in events
        if _tier(event) is not None and _tier(event) <= min_tier
    ][:limit_count]


def _tier(event: dict[str, Any]) -> Optional[int]:
    tier = event.get("market_tier")
    if isinstance(tier, bool) or not isinstance(tier, int):
        return None
    return tier


def release_date_set(events: list[dict[str, Any]]) -> set[str]:
    """Convert FXMacroData events into ISO release dates (UTC).

    Calendar rows carry the release time in ``announcement_datetime_utc``
    (or as a Unix timestamp in ``announcement_datetime``). The optional
    ``date`` field is the reference period the release covers, not the day
    it is published, so it is not used here.
    """

    dates: set[str] = set()
    for event in events:
        utc = event.get("announcement_datetime_utc")
        if isinstance(utc, str) and len(utc) >= 10:
            dates.add(utc[:10])
            continue
        epoch = event.get("announcement_datetime")
        if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
            dates.add(datetime.fromtimestamp(epoch, tz=timezone.utc).date().isoformat())
    return dates
