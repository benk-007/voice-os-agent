"""Speech-to-text with the ``SpeechRecognition`` package.

Note: ``recognize_google`` sends the audio to Google's Web Speech API (cloud).
"""

from __future__ import annotations

try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except ImportError:  # optional dependency (requirements-voice.txt)
    sr = None
    SR_AVAILABLE = False


class SpeechListener:
    def __init__(self, language: str = "en-US", energy_threshold: int = 4000) -> None:
        if not SR_AVAILABLE:
            raise RuntimeError("SpeechRecognition is not installed (pip install -r requirements-voice.txt)")
        self.language = language
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = True

    def listen_once(self, timeout: float = 10, phrase_time_limit: float = 10, ambient_duration: float = 1) -> str:
        """Record one phrase and return its transcription.

        Raises ``sr.WaitTimeoutError`` (silence) or ``sr.UnknownValueError`` (not understood).
        """
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=ambient_duration)
            audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        return self.recognizer.recognize_google(audio, language=self.language)
