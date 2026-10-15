"""T5 semantic parser: utterance -> structured intent (+ confidence)."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from voice_os_agent import config
from voice_os_agent.nlu.json_utils import parse_json_from_text

log = logging.getLogger(__name__)


@dataclass
class ParseResult:
    raw: str
    parsed: dict | None
    confidence: float | None
    latency_ms: float


class SemanticParser:
    """Wraps the fine-tuned T5-small model.

    ``torch``/``transformers`` are imported lazily so that the safety layer and
    its tests can run without them.
    """

    def __init__(
        self, model_source: str | None = None, device: str | None = None, num_beams: int = 4, max_new_tokens: int = 64
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        source = model_source or config.model_source()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading model from %s on %s", source, device)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source).to(device).eval()
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

    def generate(self, utterance: str, num_beams: int | None = None) -> tuple[str, float | None]:
        """Decode one utterance; return (raw text, confidence of the returned sequence).

        Confidence = exp(mean token log-probability) of the *chosen* sequence
        (geometric mean of the per-token probabilities), computed from the
        generation scores with ``compute_transition_scores``.
        """
        torch = self._torch
        beams = num_beams or self.num_beams
        enc = self.tokenizer(config.PREFIX + utterance, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                num_beams=beams,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        raw = self.tokenizer.decode(out.sequences[0], skip_special_tokens=True).strip()
        confidence = None
        try:
            beam_indices = getattr(out, "beam_indices", None)
            scores = self.model.compute_transition_scores(
                out.sequences, out.scores, beam_indices, normalize_logits=beam_indices is None
            )[0]
            # Positions after </s> are padding (score 0 / -inf); keep real generated tokens only.
            generated = out.sequences[0, -scores.shape[0] :]
            mask = generated != self.tokenizer.pad_token_id
            token_logprobs = scores[mask]
            token_logprobs = token_logprobs[torch.isfinite(token_logprobs)]
            if token_logprobs.numel():
                confidence = min(1.0, math.exp(token_logprobs.mean().item()))
        except Exception:  # confidence is advisory: never fail the parse because of it
            log.exception("Could not compute confidence")
        return raw, confidence

    def parse(self, utterance: str, num_beams: int | None = None) -> ParseResult:
        start = time.perf_counter()
        raw, confidence = self.generate(utterance, num_beams=num_beams)
        latency_ms = (time.perf_counter() - start) * 1000
        return ParseResult(raw=raw, parsed=parse_json_from_text(raw), confidence=confidence, latency_ms=latency_ms)
