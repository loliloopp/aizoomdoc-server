"""
Local LLM dialog logger.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.config import settings


class LLMDialogLogger:
    """Writes detailed LLM request/response logs to a local file."""

    def __init__(self, chat_id: str) -> None:
        self.enabled = settings.llm_log_enabled
        self.max_chars = settings.llm_log_truncate_chars
        log_dir = Path(settings.llm_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"llm_dialog_{chat_id}.log"

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _truncate(self, text: str) -> str:
        if self.max_chars and len(text) > self.max_chars:
            return f"<truncated {len(text)} chars>\n{text[: self.max_chars]}...\n"
        return text

    def _format(self, content: Any) -> str:
        if isinstance(content, (dict, list)):
            return json.dumps(content, indent=2, ensure_ascii=False)
        if content is None:
            return ""
        return self._truncate(str(content))

    def log_section(self, title: str, content: Any) -> None:
        if not self.enabled:
            return
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(f"\n[{self._timestamp()}] {'=' * 20} {title} {'=' * 20}\n")
                f.write(self._format(content))
                f.write("\n")
        except Exception:
            pass

    def log_line(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(f"[{self._timestamp()}] {text}\n")
        except Exception:
            pass

    def log_request(
        self,
        *,
        phase: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        google_files: Optional[list],
    ) -> None:
        payload = {
            "phase": phase,
            "model": model,
            "system_prompt": self._truncate(system_prompt),
            "user_prompt": self._truncate(user_prompt),
            "google_files": google_files or [],
        }
        self.log_section(f"LLM REQUEST - {phase}", payload)

    def log_response(self, *, phase: str, response_text: str) -> None:
        self.log_section(f"LLM RESPONSE - {phase}", self._truncate(response_text))


