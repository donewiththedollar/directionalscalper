from __future__ import annotations

import logging

import requests  # type: ignore

log = logging.getLogger(__name__)


class BlankResponse:
    def __init__(self):
        self.content = ""


class Messenger:
    def __init__(self, name, active):
        self.name = name
        self.active = active
        self.empty_response = BlankResponse()

    messenger: str | None = None

    def send_post(self, url, json=None, timeout=5):
        try:
            response = requests.post(url, json=json, timeout=timeout)
            return response
        except requests.exceptions.Timeout:
            log.info("Request timed out")
            return self.empty_response
        except requests.exceptions.TooManyRedirects:
            log.warning("Too many redirects")
            return self.empty_response
        except requests.exceptions.RequestException as e:
            log.warning(f"Request exception: {e}")
            return self.empty_response

    def send_get(self, url, params=None, timeout=5):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            return response
        except requests.exceptions.Timeout:
            log.info("Request timed out")
            return self.empty_response
        except requests.exceptions.TooManyRedirects:
            log.warning("Too many redirects")
            return self.empty_response
        except requests.exceptions.RequestException as e:
            log.warning(f"Request exception: {e}")
            return self.empty_response

    def send_message(self, message):
        log.info(f"Sending message: {message}")
        pass

    def send_embed_message(self, embed_data):
        log.info(f"Sending message: {embed_data}")
        pass

    def format_embed_data(self, data):
        pass
