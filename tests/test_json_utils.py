from voice_os_agent.nlu.json_utils import normalize_quotes_and_commas, parse_json_from_text


def test_bare_kv_list_from_model():
    assert parse_json_from_text('"app":"firefox","intent":"open_app"') == {"app": "firefox", "intent": "open_app"}


def test_full_json_and_trailing_comma():
    assert parse_json_from_text('{"intent": "mute",}') == {"intent": "mute"}


def test_curly_quotes_are_normalized():
    assert normalize_quotes_and_commas("\u201cintent\u201d:\u2018mute\u2019") == "\"intent\":'mute'"
    assert parse_json_from_text("\u201cintent\u201d:\u201cmute\u201d") == {"intent": "mute"}


def test_single_quotes():
    assert parse_json_from_text("{'intent': 'volume_up'}") == {"intent": "volume_up"}


def test_garbage_returns_none():
    assert parse_json_from_text("rm -rf") is None
    assert parse_json_from_text("__import__('os').system('id')") is None
    assert parse_json_from_text('["a", "b"]') is None
