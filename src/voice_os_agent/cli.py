"""Command-line entry point.

Examples::

    python -m voice_os_agent "open firefox and go to github.com"   # dry run, prints the plan
    python -m voice_os_agent --execute "create a file called notes.txt"
    python -m voice_os_agent                                        # interactive text REPL
    python -m voice_os_agent --gui                                  # Tkinter + microphone
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _print_plan(plan, as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan.to_dict(), ensure_ascii=False))
        return
    conf = f"{plan.confidence:.1%}" if plan.confidence is not None else "n/a"
    print(f"  model   : {plan.raw}")
    print(f"  intent  : {plan.intent}   confidence: {conf}   latency: {plan.latency_ms:.0f} ms")
    print(f"  status  : {plan.status}" + (f" ({plan.reason})" if plan.status == "validation_failed" else ""))
    if plan.result:
        print(f"  result  : {plan.result}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="voice_os_agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("utterance", nargs="*", help="command in natural language (omit for a REPL)")
    ap.add_argument("--execute", action="store_true", help="really execute (default: dry run / preview)")
    ap.add_argument("--yes", action="store_true", help="confirm destructive or low-confidence actions")
    ap.add_argument("--json", action="store_true", help="print plans as JSON")
    ap.add_argument("--gui", action="store_true", help="start the Tkinter voice interface")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.gui:
        from voice_os_agent.ui.gui import VoiceAssistantGUI

        VoiceAssistantGUI().run()
        return 0

    from voice_os_agent.agent import VoiceOSAgent

    agent = VoiceOSAgent()

    def handle(text: str) -> None:
        plan = agent.plan(text)
        plan = agent.execute(plan, execute=args.execute, confirm=args.yes)
        _print_plan(plan, args.json)

    if args.utterance:
        handle(" ".join(args.utterance))
        return 0

    mode = "EXECUTE" if args.execute else "dry-run"
    print(f"Voice OS agent - text mode ({mode}). Type a command, or 'exit'.")
    while True:
        try:
            text = (input("> ") if sys.stdin.isatty() else sys.stdin.readline() or "exit").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if text:
            handle(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
