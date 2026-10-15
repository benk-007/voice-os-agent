import pytest


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Every test gets its own sandbox directory."""
    ws = tmp_path / "safe_workspace"
    ws.mkdir()
    monkeypatch.setenv("VOA_WORKSPACE", str(ws))
    return ws.resolve()
