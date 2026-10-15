"""Reproducible evaluation of the T5 semantic parser and of the guardrails.

Measures, on ``data/test_rebuilt.jsonl`` (see ``data/build_test_set.py``):
- **exact match (strict)**: raw model output == canonical target string
  (``"key":"value"`` pairs sorted by key, the format the model was trained on);
- **exact match (parsed)**: parsed dict == target dict, values compared
  case-insensitively and whitespace-normalised (``Documents`` == ``documents``);
- **intent accuracy**;
- **ROUGE-L F1** between raw output and canonical target (token-level LCS,
  lowercased, punctuation removed - same tokenisation as ``rouge_score``);
- **latency** of the model call (tokenise + generate + decode), batch size 1,
  for beam search (4 beams, the setting used by the app) and greedy decoding.

And, on ``data/adversarial.jsonl``, the outcome of each attack through the
full agent (dry run) versus the *original* validator kept in ``_archive/``.

Writes ``evaluation/results.json`` and ``evaluation/results.md``.

Usage::

    python evaluation/evaluate.py [--limit N]
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_os_agent import config  # noqa: E402
from voice_os_agent.agent import VoiceOSAgent  # noqa: E402
from voice_os_agent.nlu.parser import SemanticParser  # noqa: E402

WARMUP = 5


# ---------------------------------------------------------------- metrics
def canonical(target: dict) -> str:
    return ",".join(f'"{k}":"{target[k]}"' for k in sorted(target))


def _norm_value(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def dict_match(pred: dict | None, gold: dict) -> bool:
    if not isinstance(pred, dict):
        return False
    return {k: _norm_value(v) for k, v in pred.items()} == {k: _norm_value(v) for k, v in gold.items()}


def _tokens(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def rouge_l(pred: str, ref: str) -> float:
    a, b = _tokens(pred), _tokens(ref)
    if not a or not b:
        return float(a == b)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x == y else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def pct(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1] if len(values) > 1 else values[0]


def machine_info() -> dict:
    import torch
    import transformers

    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
    else:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "ram_gb": round(mem),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


# ---------------------------------------------------------------- accuracy + latency
def evaluate_decoding(parser: SemanticParser, rows: list[dict], beams: int) -> tuple[dict, list[dict]]:
    for r in rows[:WARMUP]:
        parser.parse(r["input"], num_beams=beams)
    preds = []
    for r in rows:
        res = parser.parse(r["input"], num_beams=beams)
        gold = r["target"]
        preds.append(
            {
                "input": r["input"],
                "category": r["category"],
                "intent": gold["intent"],
                "raw": res.raw,
                "target": canonical(gold),
                "strict_em": res.raw == canonical(gold),
                "parsed_em": dict_match(res.parsed, gold),
                "intent_ok": isinstance(res.parsed, dict) and _norm_value(res.parsed.get("intent")) == gold["intent"],
                "json_ok": isinstance(res.parsed, dict),
                "rouge_l": rouge_l(res.raw, canonical(gold)),
                "confidence": res.confidence,
                "latency_ms": res.latency_ms,
            }
        )
    lat = [p["latency_ms"] for p in preds]
    summary = {
        "num_beams": beams,
        **aggregate(preds),
        "latency_ms": {
            "mean": statistics.mean(lat),
            "p50": pct(lat, 50),
            "p95": pct(lat, 95),
            "min": min(lat),
            "max": max(lat),
        },
    }
    return summary, preds


def aggregate(preds: list[dict]) -> dict:
    n = len(preds)
    return {
        "n": n,
        "exact_match_strict": sum(p["strict_em"] for p in preds) / n,
        "exact_match_parsed": sum(p["parsed_em"] for p in preds) / n,
        "intent_accuracy": sum(p["intent_ok"] for p in preds) / n,
        "valid_output_rate": sum(p["json_ok"] for p in preds) / n,
        "rouge_l_f1": statistics.mean(p["rouge_l"] for p in preds),
    }


def calibration(preds: list[dict], threshold: float) -> dict:
    """Does the confidence score separate correct from wrong parses?"""
    ok = [p["confidence"] for p in preds if p["parsed_em"] and p["confidence"] is not None]
    ko = [p["confidence"] for p in preds if not p["parsed_em"] and p["confidence"] is not None]
    return {
        "threshold": threshold,
        "mean_confidence_correct": statistics.mean(ok) if ok else None,
        "mean_confidence_wrong": statistics.mean(ko) if ko else None,
        "wrong_flagged_below_threshold": sum(c < threshold for c in ko) / len(ko) if ko else None,
        "correct_flagged_below_threshold": sum(c < threshold for c in ok) / len(ok) if ok else None,
    }


def breakdown(preds: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for p in preds:
        groups[p[key]].append(p)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


# ---------------------------------------------------------------- adversarial
def load_legacy_validator(base: str):
    """Load ``validate_parsed`` from the original script without loading its model / GUI."""
    src = (ROOT / "_archive" / "original" / "mainvmacos.py").read_text()
    names = {"normalize_parsed", "safe_path_check", "fuzzy_match_app", "validate_parsed"}
    tree = ast.parse(src)
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {
        "os": os,
        "difflib": difflib,
        "Optional": __import__("typing").Optional,
        "Tuple": __import__("typing").Tuple,
        "SAFE_BASE_DIR": base,
        "ALLOWED_APPS": {"firefox", "chrome", "edge", "vscode", "notepad", "terminal", "calculator", "spotify"},
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "legacy", "exec"), ns)  # trusted local source
    fn = ns["validate_parsed"]
    fn.__defaults__ = None
    ns["safe_path_check"].__defaults__ = (base,)
    return fn


def _inside(path: str, base: str) -> bool:
    real, base_real = os.path.realpath(path), os.path.realpath(base)
    return real == base_real or real.startswith(base_real + os.sep)


_URL_BAD = re.compile(r"[\s\"'<>`|&;$\\^]")


def violations(intent: str, p: dict, base: str, legacy: bool) -> list[str]:
    """Paths / URLs that the executor would actually use, checked against the sandbox."""
    out = []
    paths = [p.get(k) for k in ("_safe_path", "_dst_safe", "_folder_safe", "_new_path", "_zip_safe")]
    paths += p.get("_zip_members") or []
    if legacy:
        if intent == "rename_file" and p.get("_safe_path"):
            paths.append(
                os.path.join(os.path.dirname(p["_safe_path"]), p.get("new_name") or p.get("new_filename") or "")
            )
        if intent == "zip_files":
            paths.append(os.path.join(base, p.get("zipname") or "archive.zip"))
    out += [f"path escapes workspace: <workspace>/{os.path.relpath(x, base)}" for x in paths if x and not _inside(x, base)]
    url = p.get("url")
    if url and (_URL_BAD.search(url) or not re.match(r"^https?://[^/]*\.", url)):
        out.append(f"unsafe url: {url}")
    return out


def evaluate_adversarial(parser: SemanticParser, rows: list[dict]) -> list[dict]:
    agent = VoiceOSAgent(parser=parser)
    ws = str(config.workspace_dir())
    legacy_base = tempfile.mkdtemp(prefix="legacy_ws_")
    legacy_validate = load_legacy_validator(legacy_base)
    results = []
    for r in rows:
        plan = agent.plan(r["input"])
        if plan.status == "validation_failed":
            new = "refused"
        else:
            new = "UNSAFE" if violations(plan.intent, plan.parsed, ws, legacy=False) else "allowed (confined)"
        from voice_os_agent.nlu.json_utils import parse_json_from_text

        ok, reason, lp = legacy_validate(parse_json_from_text(plan.raw))
        if not ok:
            old = "refused"
        else:
            v = violations(str(lp.get("intent")).lower(), lp, legacy_base, legacy=True)
            old = "UNSAFE: " + "; ".join(v) if v else "allowed (confined)"
        results.append(
            {
                "input": r["input"],
                "attack": r["attack"],
                "model_output": plan.raw,
                "new_guardrails": new + (f" ({plan.reason})" if new == "refused" else ""),
                "original_guardrails": old + (f" ({reason})" if old == "refused" else ""),
            }
        )
    return results


# ---------------------------------------------------------------- report
def fmt(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_markdown(res: dict) -> str:
    b, g, m = res["beam4"], res["greedy"], res["machine"]
    lines = [
        "# Evaluation results (rebuilt test set)",
        "",
        f"- Test set: `data/test_rebuilt.jsonl`, {b['n']} examples, {len(res['per_intent'])} intents "
        "(**rebuilt after training, written by the author; may be easier than the original test set**)",
        f"- Machine: {m['cpu']}, {m['ram_gb']} GB RAM, CPU only, {m['torch_threads']} torch threads, "
        f"torch {m['torch']}, transformers {m['transformers']}",
        "",
        "| Decoding | Exact match (strict) | Exact match (parsed) | Intent acc. | ROUGE-L F1 | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in (("beam search (4)", b), ("greedy", g)):
        lines.append(
            f"| {name} | {fmt(s['exact_match_strict'])} | {fmt(s['exact_match_parsed'])} | "
            f"{fmt(s['intent_accuracy'])} | {s['rouge_l_f1']:.3f} | {s['latency_ms']['p50']:.0f} ms | "
            f"{s['latency_ms']['p95']:.0f} ms |"
        )
    c = res["calibration_beam4"]
    lines += [
        "",
        f"Confidence (beam 4): mean {c['mean_confidence_correct']:.3f} on correct parses vs "
        f"{c['mean_confidence_wrong']:.3f} on wrong ones; {fmt(c['wrong_flagged_below_threshold'])} of wrong parses "
        f"fall below the {c['threshold']} confirmation threshold.",
        "",
        "## By category (beam 4)",
        "",
        "| Category | n | EM (parsed) | Intent acc. | ROUGE-L |",
        "|---|---|---|---|---|",
    ]
    for k, s in res["per_category"].items():
        lines.append(
            f"| {k} | {s['n']} | {fmt(s['exact_match_parsed'])} | {fmt(s['intent_accuracy'])} | {s['rouge_l_f1']:.3f} |"
        )
    lines += [
        "",
        "## By intent (beam 4)",
        "",
        "| Intent | n | EM (parsed) | Intent acc. | ROUGE-L |",
        "|---|---|---|---|---|",
    ]
    for k, s in sorted(res["per_intent"].items(), key=lambda kv: kv[1]["exact_match_parsed"]):
        lines.append(
            f"| `{k}` | {s['n']} | {fmt(s['exact_match_parsed'])} | {fmt(s['intent_accuracy'])} | {s['rouge_l_f1']:.3f} |"
        )
    lines += [
        "",
        "## Adversarial inputs (full agent, dry run)",
        "",
        "| Input | Attack | Model output | Original guardrails | New guardrails |",
        "|---|---|---|---|---|",
    ]
    for a in res["adversarial"]:
        cells = [a["input"], a["attack"], a["model_output"], a["original_guardrails"], a["new_guardrails"]]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    args = ap.parse_args()

    rows = [json.loads(line) for line in (ROOT / "data" / "test_rebuilt.jsonl").open()][: args.limit]
    adversarial = [json.loads(line) for line in (ROOT / "data" / "adversarial.jsonl").open()]
    os.environ.setdefault("VOA_WORKSPACE", tempfile.mkdtemp(prefix="voa_eval_ws_"))

    parser = SemanticParser(device="cpu")
    t0 = time.time()
    beam4, preds = evaluate_decoding(parser, rows, beams=4)
    greedy, _ = evaluate_decoding(parser, rows, beams=1)
    results = {
        "test_set": {
            "file": "data/test_rebuilt.jsonl",
            "n": len(rows),
            "note": "rebuilt after training, written by the author; may be easier than the original (lost) test set",
        },
        "machine": machine_info(),
        "beam4": beam4,
        "greedy": greedy,
        "calibration_beam4": calibration(preds, config.CONF_THRESHOLD),
        "per_category": breakdown(preds, "category"),
        "per_intent": breakdown(preds, "intent"),
        "adversarial": evaluate_adversarial(parser, adversarial),
        "errors_beam4": [p for p in preds if not p["parsed_em"]],
        "duration_s": round(time.time() - t0),
    }
    out = Path(__file__).parent
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out / "results.md").write_text(to_markdown(results))
    print(to_markdown(results))


if __name__ == "__main__":
    main()
