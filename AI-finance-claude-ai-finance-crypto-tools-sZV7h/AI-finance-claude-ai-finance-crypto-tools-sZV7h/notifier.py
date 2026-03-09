"""
Telegram notification handler for AI Finance Scheduler.
"""

import logging
import requests

log = logging.getLogger(__name__)

MAX_LENGTH = 4000  # Telegram hard limit is 4096; we leave a safety margin


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._base = f"https://api.telegram.org/bot{token}"

    def send(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a single message (must be < 4096 chars)."""
        try:
            resp = requests.post(
                f"{self._base}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            if not resp.ok:
                # Retry without Markdown if parse error
                if "parse" in resp.text.lower() and parse_mode == "Markdown":
                    return self.send(text, parse_mode=None)
                log.error(f"Telegram error: {resp.text}")
                return False
            return True
        except Exception as e:
            log.error(f"Telegram send failed: {e}")
            return False

    def send_long(self, text: str):
        """Send text that may exceed Telegram's limit by splitting into chunks.

        Tries to split at paragraph boundaries to keep sections readable.
        """
        if len(text) <= MAX_LENGTH:
            self.send(text)
            return

        chunks = _split_smart(text, MAX_LENGTH)
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                chunk = f"_[{i + 1}/{len(chunks)}]_\n\n" + chunk
            self.send(chunk)

    def test_connection(self) -> bool:
        """Verify that the bot token and chat_id are valid."""
        try:
            resp = requests.get(f"{self._base}/getMe", timeout=10)
            if not resp.ok:
                log.error(f"Invalid bot token: {resp.text}")
                return False
            bot_name = resp.json()["result"]["username"]
            log.info(f"Connected to Telegram bot: @{bot_name}")

            # Send a test message
            ok = self.send("🤖 *AI Finance* — Bot connected successfully!")
            if ok:
                log.info(f"Test message sent to chat_id: {self.chat_id}")
            return ok
        except Exception as e:
            log.error(f"Connection test failed: {e}")
            return False


def _split_smart(text: str, max_len: int) -> list[str]:
    """Split text at paragraph boundaries where possible."""
    chunks = []
    while len(text) > max_len:
        # Try to cut at the last double newline before max_len
        cut = text.rfind("\n\n", 0, max_len)
        if cut == -1:
            # Fall back to the last newline
            cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            # Last resort: hard cut
            cut = max_len
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks
