import logging
from datetime import datetime

from directionalscalper.core.utils import send_public_request
from directionalscalper.messengers.messenger import Messenger

log = logging.getLogger(__name__)


class Discord(Messenger):
    def __init__(self, name, active, webhook_url):
        super().__init__(name=name, active=active)
        self.webhook_url = webhook_url

    messenger = "discord"
    data = {
        "username": "DirectionalScalper",
        "avatar_url": "https://avatars.githubusercontent.com/u/89611464?v=4",
    }

    def send_message(self, message):
        if self.active:
            log.info(f"Sending discord message via {self.name}: {message}")
            self.data["content"] = message
            header, raw_json = send_public_request(
                url=self.webhook_url, method="POST", json_in=self.data
            )
            return raw_json
        log.info(f"{self.name} (discord messenger) is inactive")

    def send_embed_message(self, embed_data):
        if self.active:
            log.info(f"Sending embedded discord message via {self.name}: {embed_data}")

            options, payload = self.format_embed_data(embed_data)

            self.data["content"] = "Directional Scalper"

            self.data["embeds"] = [
                {
                    "fields": [],
                }
            ]

            for k, v in options.items():
                self.data["embeds"][0][k] = v

            for k, v in payload.items():
                self.data["embeds"][0]["fields"].append(
                    {"name": k, "value": v, "inline": True}
                )
            self.data["embeds"][0]["footer"] = {
                "text": f"DirectionalScalper - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "icon_url": "https://avatars.githubusercontent.com/u/89611464?v=4",
            }
            header, raw_json = send_public_request(
                url=self.webhook_url, method="POST", json_in=self.data
            )
            return raw_json

        log.info(f"{self.name} (discord messenger) is inactive")

    def format_embed_data(self, embed_data):
        options_keys = {
            "Title": "title",
            "Colour": "color",
            "URL": "url",
            "Image": "thumbnail",
        }
        payload_keys = ["Symbol", "Side", "PnL"]
        options = {}
        payload = {}
        for k, v in embed_data.items():
            if k in options_keys:
                if k == "Image":
                    options[options_keys[k]] = {"url": v.replace(" ", "%20")}
                else:
                    options[options_keys[k]] = v

            elif k in payload_keys:
                payload[k] = v

        return options, payload
