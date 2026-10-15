"""Non-blocking text-to-speech with ``pyttsx3`` (runs in a worker thread)."""

from __future__ import annotations

import logging
import queue
import threading

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:  # optional dependency (requirements-voice.txt)
    pyttsx3 = None
    TTS_AVAILABLE = False

log = logging.getLogger(__name__)


class NonBlockingSpeaker:
    def __init__(self, rate: int = 170) -> None:
        self.available = TTS_AVAILABLE
        self.rate = rate
        if not self.available:
            return
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", 1.0)
        except Exception:
            log.exception("TTS init failed")
            self.available = False
            return
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                log.debug("TTS: %s", text)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                log.exception("TTS error")

    def speak(self, text: str) -> None:
        if self.available:
            self.queue.put(text)
        else:
            log.info("[TTS unavailable] %s", text)

    def stop(self) -> None:
        self.running = False
        if self.available:
            self.queue.put(None)
            if self.thread.is_alive():
                self.thread.join(timeout=2)
