import logging

from directionalscalper.core.utils import send_public_request
from directionalscalper.messengers.messenger import Messenger

log = logging.getLogger(__name__)


class Telegram(Messenger):
    def __init__(self, name, active, bot_token, chat_id):
        super().__init__(name=name, active=active)
        self.bot_token = bot_token
        self.data = {
            "chat_id": chat_id,
            "parse_mode": "Markdown",
        }

    messenger = "telegram"

    def send_message(self, message):
        if self.active:
            log.info(f"Sending telegram message to {self.name}: {message}")
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            self.data["text"] = message
            header, raw_json = send_public_request(url=url, payload=self.data)
            return raw_json
        log.info(f"{self.name} (telegram messenger) is inactive")
