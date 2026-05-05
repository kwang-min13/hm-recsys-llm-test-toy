# src/simulation/ollama_client.py

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout_sec: int = 30
    retries: int = 2
    backoff_sec: float = 0.5

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")

    def _get(self, path: str, timeout: Optional[int] = None) -> requests.Response:
        return requests.get(self.base_url + path, timeout=timeout or self.timeout_sec)

    def _post(self, path: str, json: Dict[str, Any], timeout: Optional[int] = None) -> requests.Response:
        return requests.post(self.base_url + path, json=json, timeout=timeout or self.timeout_sec)

    def check_connection(self) -> bool:
        for path in ("/api/version", "/api/tags"):
            try:
                r = self._get(path, timeout=3)
                if r.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    def _extract_chat_content(self, data: Dict[str, Any]) -> Optional[str]:
        """
        /api/chat 응답에서 assistant content 추출
        (Ollama 표준: {"message": {"role":"assistant","content":"..."}, ...})
        """
        msg = data.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        return None

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        *,
        model: Optional[str] = None,
        num_predict: int = 96,
        stop: Optional[list[str]] = None,
    ) -> Optional[str]:
        payload_generate: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": int(num_predict),
            },
        }
        if stop:
            payload_generate["options"]["stop"] = stop

        # /api/chat 폴백 payload (same options)
        payload_chat: Dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": int(num_predict),
            },
        }
        if stop:
            payload_chat["options"]["stop"] = stop

        last_err: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                r = self._post("/api/generate", json=payload_generate)

                # ✅ 핵심: /api/generate가 404면 /api/chat로 폴백
                if r.status_code == 404:
                    logger.warning(
                        "POST /api/generate returned 404; falling back to /api/chat. "
                        f"base_url={self.base_url}"
                    )
                    rc = self._post("/api/chat", json=payload_chat)
                    rc.raise_for_status()
                    data_c = rc.json()
                    content = self._extract_chat_content(data_c)
                    if not content:
                        logger.warning("Empty response from Ollama /api/chat.")
                        return None
                    return content

                r.raise_for_status()
                data = r.json()

                # /api/generate 표준: {"response": "...", ...}
                resp = data.get("response")
                if not resp:
                    logger.warning("Empty response from Ollama /api/generate.")
                    return None
                return str(resp).strip()

            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    sleep_s = self.backoff_sec * (2 ** attempt)
                    logger.warning(
                        f"Ollama generate failed (attempt {attempt+1}/{self.retries+1}): {e}. "
                        f"Retrying in {sleep_s:.2f}s"
                    )
                    time.sleep(sleep_s)
                else:
                    logger.error(f"Ollama generate failed: {e}")
                    return None

        if last_err:
            logger.error(f"Ollama generate failed after retries: {last_err}")
        return None
