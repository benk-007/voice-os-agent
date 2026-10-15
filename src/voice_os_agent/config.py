"""Runtime configuration, overridable with environment variables (see ``.env.example``)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Prefix used during fine-tuning; must be prepended to every utterance.
PREFIX = "semantic_parse: "

#: Hugging Face Hub repository used when no local model directory is found.
DEFAULT_HUB_ID = "benk-007/t5-small-os-commands"

_DEFAULT_WORKSPACE = r"C:\Users\Public\safe_workspace" if os.name == "nt" else "~/safe_workspace"


def model_source() -> str:
    """Return a local model directory if it exists, otherwise a Hub repository id.

    ``transformers`` downloads (and caches) the weights on first use when given a Hub id.
    """
    configured = os.getenv("VOA_MODEL_DIR")
    if configured:
        return configured
    local = PROJECT_ROOT / "models" / "t5-semantic-parser-final"
    return str(local) if local.exists() else DEFAULT_HUB_ID


def workspace_dir() -> Path:
    """Sandbox directory: every file operation is confined to it."""
    path = Path(os.path.expanduser(os.getenv("VOA_WORKSPACE", _DEFAULT_WORKSPACE)))
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


#: Below this model confidence, the agent asks for confirmation instead of acting.
CONF_THRESHOLD = float(os.getenv("VOA_CONF_THRESHOLD", "0.85"))
