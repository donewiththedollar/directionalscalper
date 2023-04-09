import logging

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

            return self.send_get(url=url, params=self.data)
        log.info(f"{self.name} (telegram messenger) is inactive")
