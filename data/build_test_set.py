"""Build the *rebuilt* evaluation set (``data/test_rebuilt.jsonl``).

The original 5,760-example test split used in the course report was lost.
This script re-creates an evaluation set **after training**, written by hand
(templates + entity pools + perturbations + a list of hand-written hard cases).

Caveats (also stated in the README):
- The output schema (slot names per intent) was inferred from the inference
  code and from the model's answers to one canonical phrasing per intent.
  Gold *values* always come from the sentence itself, never from the model.
- Phrasings were written independently from the (lost) training templates,
  but by someone who had seen the model's schema, so the set may be easier
  or harder than the original one. Treat the numbers as indicative.

Categories:
- ``clean``   : direct command, synonym verbs
- ``filler``  : politeness / hesitation words ("hey, could you ... please")
- ``typo``    : one character-level typo in a non-entity word (ASR-like noise)
- ``long``    : the command embedded in a long sentence
- ``hard``    : hand-written ambiguous or unusual phrasings

Usage::

    python data/build_test_set.py            # writes data/test_rebuilt.jsonl
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

SEED = 2026
PER_INTENT = 16
OUT = Path(__file__).with_name("test_rebuilt.jsonl")

APPS = ["firefox", "chrome", "edge", "vscode", "notepad", "terminal", "spotify"]
BROWSERS = ["firefox", "chrome", "edge"]
URLS = [
    "github.com",
    "youtube.com",
    "wikipedia.org",
    "stackoverflow.com",
    "lemonde.fr",
    "arxiv.org",
    "huggingface.co",
    "netflix.com",
]
QUERIES = [
    "python decorators",
    "best pizza recipe",
    "transformer architecture",
    "cheap flights to madrid",
    "how to center a div",
    "docker compose tutorial",
    "symptoms of burnout",
    "football results",
]
FILES = [
    "report.txt",
    "notes.md",
    "budget.xlsx",
    "slides.pptx",
    "todo.txt",
    "photo.png",
    "thesis_draft.docx",
    "data.csv",
    "invoice_2025.pdf",
]
FOLDERS = ["projects", "archive", "backup", "invoices", "photos", "music", "work"]
DURATIONS = ["5 minutes", "30 seconds", "1 hour", "15 minutes", "2 minutes", "45 minutes"]
CITIES = ["Paris", "London", "Rabat", "New York", "Tokyo", "Casablanca", "Berlin"]
WORDS = ["serendipity", "ephemeral", "ubiquitous", "resilience", "paradigm"]
LANGS = ["french", "spanish", "german", "arabic"]
TEXTS = ["good morning", "thank you", "where is the station", "see you tomorrow"]
PATTERNS = ["budget", "invoice", "draft", "screenshot", "notes"]

# intent -> (list of templates, slot sampler). Slots in templates are {name}.
TEMPLATES: dict[str, tuple[list[str], dict]] = {
    "open_app": (
        ["open {app}", "launch {app}", "start {app}", "fire up {app}", "bring up {app}", "can you open {app}"],
        {"app": APPS},
    ),
    "open_url": (
        ["go to {url}", "open {url}", "navigate to {url}", "take me to {url}", "visit {url}"],
        {"url": URLS},
    ),
    "open_app_and_go_url": (
        [
            "open {app} and go to {url}",
            "launch {app} and open {url}",
            "use {app} to visit {url}",
            "start {app} then navigate to {url}",
        ],
        {"app": BROWSERS, "url": URLS},
    ),
    "open_app_and_search": (
        ["open {app} and search for {query}", "launch {app} and look up {query}", "start {app} and search {query}"],
        {"app": BROWSERS, "query": QUERIES},
    ),
    "search_web": (
        [
            "search the web for {query}",
            "google {query}",
            "look up {query} online",
            "search for {query}",
            "find information about {query} on the internet",
        ],
        {"query": QUERIES},
    ),
    "create_file": (
        [
            "create a file called {filename}",
            "make a new file named {filename}",
            "create {filename}",
            "new file {filename}",
        ],
        {"filename": FILES},
    ),
    "delete_file": (
        ["delete the file {filename}", "remove {filename}", "delete {filename}", "erase the file {filename}"],
        {"filename": FILES},
    ),
    "open_file": (
        ["open the file {filename}", "show me the file {filename}", "open file {filename}"],
        {"filename": FILES},
    ),
    "open_file_with_app": (
        ["open {filename} with {app}", "open {filename} in {app}", "use {app} to open {filename}"],
        {"filename": FILES, "app": ["vscode", "notepad"]},
    ),
    "rename_file": (
        ["rename {filename} to {newname}", "change the name of {filename} to {newname}"],
        {"filename": FILES, "newname": ["final.txt", "old_notes.md", "v2.csv", "summary.pdf"]},
    ),
    "move_file": (
        [
            "move {filename} to {folder}",
            "move the file {filename} into the {folder} folder",
            "put {filename} in {folder}",
        ],
        {"filename": FILES, "folder": FOLDERS},
    ),
    "copy_file": (
        [
            "copy {filename} to {folder}",
            "duplicate {filename} into {folder}",
            "make a copy of {filename} in the {folder} folder",
        ],
        {"filename": FILES, "folder": FOLDERS},
    ),
    "create_folder": (
        ["create a folder named {folder}", "make a new directory called {folder}", "new folder {folder}"],
        {"folder": FOLDERS},
    ),
    "list_dir": (
        ["list files in {folder}", "show me what is in the {folder} folder", "list the contents of {folder}"],
        {"folder": FOLDERS},
    ),
    "search_files": (
        [
            "search for files named {pattern} in {folder}",
            "find files called {pattern} in {folder}",
            "look for {pattern} files in the {folder} folder",
        ],
        {"pattern": PATTERNS, "folder": FOLDERS},
    ),
    "zip_files": (
        [
            "zip {filenames} into {zipname}",
            "compress {filenames} into {zipname}",
            "create an archive {zipname} with {filenames}",
        ],
        {"filenames": "PAIR", "zipname": ["backup.zip", "docs.zip", "week12.zip", "export.zip"]},
    ),
    "unzip_files": (
        ["unzip {zipname}", "extract {zipname}", "decompress the archive {zipname}"],
        {"zipname": ["archive.zip", "photos.zip", "release.zip", "data.zip"]},
    ),
    "volume_up": (["turn up the volume", "increase the volume", "louder", "volume up", "raise the sound"], {}),
    "volume_down": (["turn down the volume", "decrease the volume", "lower the sound", "volume down", "quieter"], {}),
    "mute": (["mute the sound", "mute", "mute the audio", "silence the computer"], {}),
    "unmute": (["unmute", "unmute the sound", "turn the sound back on"], {}),
    "screenshot": (["take a screenshot", "capture the screen", "screenshot please", "grab a screen capture"], {}),
    "lock_screen": (["lock the screen", "lock my computer", "lock the session"], {}),
    "system_sleep": (["put the computer to sleep", "sleep mode", "make the computer sleep"], {}),
    "shutdown": (["shut down the computer", "turn off the computer", "power off", "shutdown"], {}),
    "restart": (["restart the computer", "reboot", "reboot the machine", "restart my pc"], {}),
    "play_music": (["play music", "play some music", "start the music"], {}),
    "pause_music": (["pause the music", "pause", "pause the song"], {}),
    "set_timer": (
        ["set a timer for {duration}", "start a {duration} timer", "timer {duration}", "remind me in {duration}"],
        {"duration": DURATIONS},
    ),
    "weather": (
        [
            "what is the weather in {location}",
            "weather in {location}",
            "how is the weather in {location} today",
            "is it going to rain in {location}",
        ],
        {"location": CITIES},
    ),
    "dictionary_lookup": (
        ["define {word}", "what does {word} mean", "look up the definition of {word}"],
        {"word": WORDS},
    ),
    "translate_text": (
        ["translate {text} to {lang}", "how do you say {text} in {lang}", "translate {text} into {lang}"],
        {"text": TEXTS, "lang": LANGS},
    ),
}

FILLERS_BEFORE = ["please ", "hey, ", "could you ", "um, ", "can you please ", "hey assistant, "]
FILLERS_AFTER = [" please", " for me", " right now", " thanks", " now"]
LONG_BEFORE = [
    "I'm working on my end of semester project and I need you to ",
    "before my meeting starts in a few minutes, I would like you to ",
    "ok so this is a bit of a weird request but ",
    "I have been trying to do this for a while, could you ",
]
LONG_AFTER = [
    " because I need it for the presentation later today",
    " and let me know when it is done",
    " as soon as possible, thank you very much",
    "",
]

# Hand-written hard / ambiguous cases. Gold = the most reasonable reading.
HARD = [
    ("opne firefx", {"intent": "open_app", "app": "firefox"}),
    ("could you maybe open up google chrome for me", {"intent": "open_app", "app": "chrome"}),
    ("I want to see github.com", {"intent": "open_url", "url": "github.com"}),
    ("open visual studio code", {"intent": "open_app", "app": "vscode"}),
    ("can you open the terminal window", {"intent": "open_app", "app": "terminal"}),
    ("it's too loud", {"intent": "volume_down"}),
    ("I can't hear anything, turn it up", {"intent": "volume_up"}),
    ("get rid of todo.txt", {"intent": "delete_file", "filename": "todo.txt"}),
    ("I need a new empty file, call it ideas.txt", {"intent": "create_file", "filename": "ideas.txt"}),
    ("what's inside the projects folder", {"intent": "list_dir", "folder": "projects"}),
    ("search google for how to cook rice", {"intent": "search_web", "query": "how to cook rice"}),
    ("what's the temperature outside in Paris", {"intent": "weather", "location": "Paris"}),
    ("wake me up in 10 minutes", {"intent": "set_timer", "duration": "10 minutes"}),
    ("I'm leaving, lock everything", {"intent": "lock_screen"}),
    ("turn the computer off now", {"intent": "shutdown"}),
    ("stop the sound completely", {"intent": "mute"}),
    ("I want to listen to some music", {"intent": "play_music"}),
    (
        "put report.txt and notes.md in a zip called bundle.zip",
        {"intent": "zip_files", "filenames": "report.txt, notes.md", "zipname": "bundle.zip"},
    ),
    (
        "take report.txt and move it over to the archive folder",
        {"intent": "move_file", "filename": "report.txt", "folder": "archive"},
    ),
    (
        "open firefox and then go to youtube.com",
        {"intent": "open_app_and_go_url", "app": "firefox", "url": "youtube.com"},
    ),
    ("look for python jobs in paris", {"intent": "search_web", "query": "python jobs in paris"}),
    ("what does the word ubiquitous mean", {"intent": "dictionary_lookup", "word": "ubiquitous"}),
    ("say thank you in spanish", {"intent": "translate_text", "text": "thank you", "lang": "spanish"}),
    ("print screen", {"intent": "screenshot"}),
    ("reboot please, it is frozen", {"intent": "restart"}),
    ("show me the budget.xlsx file", {"intent": "open_file", "filename": "budget.xlsx"}),
    ("make a directory for my photos called holidays", {"intent": "create_folder", "folder": "holidays"}),
    ("hold the music", {"intent": "pause_music"}),
    ("youtube.com in chrome please", {"intent": "open_app_and_go_url", "app": "chrome", "url": "youtube.com"}),
    ("extract everything from release.zip", {"intent": "unzip_files", "zipname": "release.zip"}),
]


def add_typo(text: str, rng: random.Random) -> str:
    """Apply one character-level typo to one template word (never to a {slot})."""
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if "{" not in w and len(w) >= 4 and w.isalpha()]
    if not idx:
        return text
    i = rng.choice(idx)
    w = list(words[i])
    p = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap":
        w[p], w[p + 1 if p + 1 < len(w) else p - 1] = w[p + 1 if p + 1 < len(w) else p - 1], w[p]
    elif op == "drop":
        del w[p]
    else:
        w.insert(p, w[p])
    words[i] = "".join(w)
    return " ".join(words)


def sample_slots(spec: dict, rng: random.Random) -> dict:
    slots = {}
    for name, pool in spec.items():
        if pool == "PAIR":
            a, b = rng.sample(FILES, 2)
            slots[name] = f"{a}, {b}"
        else:
            slots[name] = rng.choice(pool)
    return slots


def render(template: str, slots: dict, category: str, rng: random.Random) -> str:
    if category == "typo":
        template = add_typo(template, rng)
    text = template.format(**slots)
    if category == "filler":
        text = rng.choice(FILLERS_BEFORE) + text + rng.choice(FILLERS_AFTER)
    elif category == "long":
        text = rng.choice(LONG_BEFORE) + text + rng.choice(LONG_AFTER)
    return re.sub(r"\s+", " ", text).strip()


def build() -> list[dict]:
    rng = random.Random(SEED)
    rows, seen = [], set()
    categories = ["clean"] * 6 + ["filler"] * 4 + ["typo"] * 3 + ["long"] * 3
    for intent, (templates, spec) in TEMPLATES.items():
        n, attempts = 0, 0
        while n < PER_INTENT and attempts < 500:
            attempts += 1
            category = categories[n % len(categories)]
            slots = sample_slots(spec, rng)
            text = render(rng.choice(templates), slots, category, rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"input": text, "target": {"intent": intent, **slots}, "category": category})
            n += 1
    for text, target in HARD:
        rows.append({"input": text, "target": target, "category": "hard"})
    return rows


if __name__ == "__main__":
    rows = build()
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {OUT}")
