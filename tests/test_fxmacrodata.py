"""Offline tests for the FXMacroData release-calendar helpers."""

import pytest

from directionalscalper.core.macro import fxmacrodata
from directionalscalper.core.macro.fxmacrodata import (
    FXMacroDataError,
    fetch_release_calendar,
    release_date_set,
)


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _capture_get(monkeypatch, payload, status_code=200):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None, allow_redirects=True):
        calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "allow_redirects": allow_redirects,
            }
        )
        return _Response(payload, status_code)

    monkeypatch.setattr(fxmacrodata.requests, "get", fake_get)
    return calls


EVENTS = [
    {
        "release": "non_farm_payrolls",
        "announcement_datetime": 1791549000,
        "announcement_datetime_utc": "2026-10-09T12:30:00+00:00",
        "date": "2026-09-30",
        "market_tier": 1,
    },
    {
        "release": "trade_balance",
        "announcement_datetime": 1791289800,
        "announcement_datetime_utc": "2026-10-06T12:30:00+00:00",
        "market_tier": 2,
    },
]


def test_key_is_sent_as_header_not_query(monkeypatch):
    calls = _capture_get(monkeypatch, {"data": EVENTS})
    fetch_release_calendar("USD", api_key="test-key")

    assert calls[0]["url"] == "https://api.fxmacrodata.com/v1/calendar/usd"
    assert calls[0]["headers"] == {"X-API-Key": "test-key"}
    assert "api_key" not in calls[0]["params"]
    assert calls[0]["allow_redirects"] is False


def test_redirect_is_refused(monkeypatch):
    _capture_get(monkeypatch, {}, status_code=302)
    with pytest.raises(FXMacroDataError) as exc:
        fetch_release_calendar("usd", api_key="secret-key")
    assert "secret-key" not in str(exc.value)


@pytest.mark.parametrize("bad", [" secret key", "secret\nkey", "sec ret"])
def test_malformed_key_is_not_echoed(monkeypatch, bad):
    calls = _capture_get(monkeypatch, {"data": EVENTS})
    with pytest.raises(FXMacroDataError) as exc:
        fetch_release_calendar("usd", api_key=bad)
    assert "secret" not in str(exc.value)
    assert calls == []


def test_key_is_stripped(monkeypatch):
    calls = _capture_get(monkeypatch, {"data": EVENTS})
    fetch_release_calendar("usd", api_key="test-key\n")
    assert calls[0]["headers"] == {"X-API-Key": "test-key"}


def test_error_body_with_200_raises(monkeypatch):
    _capture_get(monkeypatch, {"detail": "Unsupported currency"})
    with pytest.raises(FXMacroDataError, match="Unsupported currency"):
        fetch_release_calendar("xyz")


@pytest.mark.parametrize(
    "payload", [["not", "a", "dict"], {"data": "nope"}, ValueError("not json")]
)
def test_malformed_payload_raises(monkeypatch, payload):
    _capture_get(monkeypatch, payload)
    with pytest.raises(FXMacroDataError):
        fetch_release_calendar("usd")


def test_rows_with_bad_tiers_are_dropped(monkeypatch):
    rows = EVENTS + [{"market_tier": "1"}, {"market_tier": None}, "junk"]
    _capture_get(monkeypatch, {"data": rows})
    assert [e["release"] for e in fetch_release_calendar("usd", min_tier=3)] == [
        "non_farm_payrolls",
        "trade_balance",
    ]


def test_no_key_sends_no_auth_header(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    calls = _capture_get(monkeypatch, {"data": EVENTS})
    fetch_release_calendar("usd")

    assert calls[0]["headers"] == {}


def test_min_tier_filters_events(monkeypatch):
    _capture_get(monkeypatch, {"data": EVENTS})
    events = fetch_release_calendar("usd", min_tier=1)

    assert [e["release"] for e in events] == ["non_farm_payrolls"]


def test_release_date_set_uses_release_time_not_reference_period():
    assert release_date_set(EVENTS) == {"2026-10-09", "2026-10-06"}


def test_release_date_set_falls_back_to_epoch():
    events = [{"announcement_datetime": 1791289800}, {"release": "no_time"}]
    assert release_date_set(events) == {"2026-10-06"}
