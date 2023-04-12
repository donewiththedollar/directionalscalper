from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections import OrderedDict
from urllib.parse import urlencode

import requests  # type: ignore

log = logging.getLogger(__name__)


class HTTPRequestError(Exception):
    def __init__(self, url, code, msg=None):
        self.url = url
        self.code = code
        self.msg = msg

    def __str__(self) -> str:
        return f"Request to {self.url!r} failed. Code: {self.code}; Message: {self.msg}"


class BlankResponse:
    def __init__(self):
        self.content = ""


def get_timestamp():
    return int(time.time() * 1000)


def hashing(
    query_string: str,
    exchange: str = "binance",
    timestamp: int = -1,
    keys: dict | None = None,
):
    if keys is None:
        keys = {"key": "", "secret": ""}
    if exchange == "bybit":
        query_string = f"{timestamp}{keys['key']}5000" + query_string
        return hmac.new(
            bytes(keys["secret"].encode("utf-8")),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    return hmac.new(
        bytes(keys["secret"].encode("utf-8")),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def dispatch_request(
    http_method: str,
    key: str = "",
    signature: str = "",
    timestamp: int = -1,
):
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json;charset=utf-8",
            "X-MBX-APIKEY": f"{key}",
            "X-BAPI-API-KEY": f"{key}",
            "X-BAPI-SIGN": f"{signature}",
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": f"{timestamp}",
            "X-BAPI-RECV-WINDOW": "5000",
        }
    )
    return {
        "GET": session.get,
        "DELETE": session.delete,
        "PUT": session.put,
        "POST": session.post,
    }.get(http_method, "GET")


def send_public_request(
    url: str,
    method: str = "GET",
    url_path: str | None = None,
    payload: dict | None = None,
    json_in: dict | None = None,
    json_out: bool = True,
):
    empty_response = BlankResponse().content
    if url_path is not None:
        url += url_path
    if payload is None:
        payload = {}
    query_string = urlencode(payload, True)
    if query_string:
        url = url + "?" + query_string

    log.debug(f"Requesting {url}")

    try:
        response = dispatch_request(method)(
            url=url,
            json=json_in,
            timeout=5,
        )
        headers = response.headers
        if not json_out:
            return headers, response.text
        json_response = response.json()
        if "code" in json_response and "msg" in json_response:
            if len(json_response["msg"]) > 0:
                raise HTTPRequestError(
                    url=url, code=json_response["code"], msg=json_response["msg"]
                )
        if "retCode" in json_response:
            if json_response["retCode"] != 0:
                raise HTTPRequestError(
                    url=url, code=json_response["retCode"], msg=json_response["retMsg"]
                )
        return headers, json_response
    except requests.exceptions.ConnectionError:
        log.warning("Connection error")
    except requests.exceptions.Timeout:
        log.warning("Request timed out")
    except requests.exceptions.TooManyRedirects:
        log.warning("Too many redirects")
    except requests.exceptions.JSONDecodeError as e:
        log.warning(f"JSON decode error: {e}")
    except requests.exceptions.RequestException as e:
        log.warning(f"Request exception: {e}")
    except HTTPRequestError as e:
        log.warning(f"HTTP Request error: {e}")
    return empty_response, empty_response


def send_signed_request(
    http_method: str,
    url_path: str,
    payload: dict | None = None,
    exchange: str = "binance",
    base_url=str,
    keys: dict | None = None,
):
    empty_response = BlankResponse().content
    if keys is None:
        keys = {"key": "", "secret": ""}
    if payload is None:
        payload = {}
    timestamp = get_timestamp()
    if exchange == "binance":
        payload["timestamp"] = timestamp

    query_string = urlencode(OrderedDict(sorted(payload.items())))
    query_string = query_string.replace("%27", "%22")

    url = f"{base_url}{url_path}?{query_string}"
    if exchange == "binance":
        url += f"&signature={hashing(query_string=query_string, exchange=exchange, keys=keys)}"
    params = {"url": url, "params": {}}

    log.debug(f"Requesting {url}")
    try:
        response = dispatch_request(
            http_method=http_method,
            key=keys["key"],
            signature=hashing(
                query_string=query_string,
                exchange=exchange,
                timestamp=timestamp,
                keys=keys,
            ),
            timestamp=timestamp,
        )(**params)
        headers = response.headers
        json_response = response.json()
        if "code" in json_response:
            raise HTTPRequestError(
                url=url, code=json_response["code"], msg=json_response["msg"]
            )
        if "retCode" in json_response:
            if json_response["retCode"] != 0:
                raise HTTPRequestError(
                    url=url, code=json_response["retCode"], msg=json_response["retMsg"]
                )
        return headers, json_response
    except requests.exceptions.ConnectionError:
        log.warning("Connection error")
    except requests.exceptions.Timeout:
        log.warning("Request timed out")
    except requests.exceptions.TooManyRedirects:
        log.warning("Too many redirects")
    except requests.exceptions.RequestException as e:
        log.warning(f"Request exception: {e}")
    except requests.exceptions.JSONDecodeError as e:
        log.warning(f"JSON decode error: {e}")
    except HTTPRequestError as e:
        log.warning(f"HTTP Request error: {e}")
    return empty_response, empty_response
