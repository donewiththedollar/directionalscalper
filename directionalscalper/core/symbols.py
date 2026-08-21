"""Symbol canonicalization, mirroring the conventions used in production by
mantis-mainacct (strategies/ema_cascade_scalper/state.py::canonical_symbol,
core/market_data/blofin_stream.py::normalize_symbol).

Conventions:
- Every venue has an explicit canonical form; there is no single silent
  string-munge shared across exchanges.
- Internal/canonical form is uppercase ``BASEQUOTE`` (e.g. ``BTCUSDT``).
- BloFin instrument IDs are uppercase ``BASE-USDT``.
- Malformed input raises ``ValueError`` instead of being passed through
  silently — fail fast, as in mantis.

Note on case: Bybit linear and BloFin instrument identifiers are uppercase by
definition, so these helpers collapse case deliberately. If support is added
for a venue whose symbols are case-sensitive, give that venue its own explicit
function here rather than relaxing these.
"""

import re

# e.g. BTCUSDT / SUIUSDT / 1000PEPEUSDT — alnum base ending in USDT
_CANONICAL_PATTERN = re.compile(r"^[A-Z0-9]+USDT$")
# e.g. BTC-USDT
_INST_ID_PATTERN = re.compile(r"^[A-Z0-9]+-USDT$")


def canonical_symbol(value) -> str:
    """Return the internal canonical ``BASEQUOTE`` form (uppercase).

    Accepts ``BTC/USDT``, ``BTC/USDT:USDT``, ``btc-usdt``, ``BTC_USDT``,
    ``BTCUSDT`` etc. Raises ``ValueError`` on anything not recognizable as a
    USDT-margined pair.
    """
    raw = str(value or "").strip().upper()
    if "/" in raw:
        base, rest = raw.split("/", 1)
        quote = rest.split(":", 1)[0]
        raw = f"{base}{quote}"
    raw = raw.replace("-", "").replace("_", "")
    if not raw.endswith("USDT") or not raw[:-4].isalnum():
        raise ValueError(f"invalid USDT symbol: {value!r}")
    return raw


def to_bybit_linear(value) -> str:
    """Bybit linear (USDT perpetual) symbol: ``BASEQUOTE``, e.g. ``BTCUSDT``."""
    symbol = canonical_symbol(value)
    if not _CANONICAL_PATTERN.fullmatch(symbol):
        raise ValueError(f"invalid Bybit linear symbol: {value!r}")
    return symbol


def to_blofin_inst_id(value) -> str:
    """BloFin instrument ID: ``BASE-USDT``, e.g. ``BTC-USDT``."""
    raw = str(value or "").strip().upper()
    if "-" in raw and "/" not in raw:
        inst_id = raw  # already BASE-USDT
        base, quote = inst_id.split("-", 1)
    elif "/" in raw:
        # reject anything that is not a USDT pair — no silent re-quoting
        pair = raw.split(":", 1)[0]
        base, quote = pair.split("/", 1)
        if quote != "USDT":
            raise ValueError(f"invalid BloFin instrument ID (quote must be USDT): {value!r}")
        inst_id = f"{base}-USDT"
    elif raw.endswith("USDT"):
        inst_id = f"{raw[:-4]}-USDT"
        base, quote = raw[:-4], "USDT"
    else:
        raise ValueError(f"invalid BloFin instrument ID: {value!r}")
    if not _INST_ID_PATTERN.fullmatch(inst_id):
        raise ValueError(f"invalid BloFin instrument ID: {value!r}")
    return inst_id


def standardize_symbol(symbol) -> str:
    """Back-compat alias used by the bot entry points.

    Equivalent to :func:`canonical_symbol`.
    """
    return canonical_symbol(symbol)
