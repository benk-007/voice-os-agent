"""End-to-end agent tests with a fake parser (no model download needed)."""

from voice_os_agent.agent import VoiceOSAgent
from voice_os_agent.nlu.json_utils import parse_json_from_text
from voice_os_agent.nlu.parser import ParseResult


class FakeParser:
    def __init__(self, raw, confidence=0.99):
        self.raw, self.confidence, self.calls = raw, confidence, 0

    def parse(self, utterance):
        self.calls += 1
        return ParseResult(self.raw, parse_json_from_text(self.raw), self.confidence, 1.0)


def agent_for(raw, confidence=0.99):
    return VoiceOSAgent(parser=FakeParser(raw, confidence))


def test_model_called_once_per_command(workspace):
    agent = agent_for('"filename":"a.txt","intent":"create_file"')
    plan = agent.plan("create a.txt")
    agent.execute(plan, execute=False)
    agent.execute(plan, execute=True)
    assert agent.parser.calls == 1


def test_dry_run_touches_nothing(workspace):
    plan = agent_for('"filename":"a.txt","intent":"create_file"').plan("x")
    plan = VoiceOSAgent(parser=FakeParser("")).execute(plan, execute=False)
    assert plan.status == "simulated" and not (workspace / "a.txt").exists()


def test_create_file_inside_workspace(workspace):
    agent = agent_for('"filename":"notes/a.txt","intent":"create_file"')
    plan = agent.execute(agent.plan("x"), execute=True)
    assert plan.status == "executed" and (workspace / "notes" / "a.txt").exists()


def test_rename_uses_model_newname_key(workspace):
    (workspace / "report.txt").write_text("hi")
    agent = agent_for('"filename":"report.txt","intent":"rename_file","newname":"final.txt"')
    plan = agent.execute(agent.plan("x"), execute=True)
    assert plan.status == "executed" and (workspace / "final.txt").exists()


def test_traversal_never_executes(workspace, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    agent = agent_for('"filename":"../victim.txt","intent":"delete_file"')
    plan = agent.execute(agent.plan("delete ../victim.txt"), execute=True, confirm=True)
    assert plan.status == "validation_failed" and victim.exists()


def test_delete_requires_confirmation(workspace):
    (workspace / "a.txt").write_text("x")
    agent = agent_for('"filename":"a.txt","intent":"delete_file"')
    plan = agent.execute(agent.plan("x"), execute=True, confirm=False)
    assert plan.status == "needs_confirmation" and (workspace / "a.txt").exists()
    plan = agent.execute(plan, execute=True, confirm=True)
    assert plan.status == "executed" and not (workspace / "a.txt").exists()


def test_low_confidence_is_not_executed(workspace):
    agent = agent_for('"filename":"a.txt","intent":"create_file"', confidence=0.3)
    plan = agent.execute(agent.plan("x"), execute=True)
    assert plan.status == "needs_confirmation" and not (workspace / "a.txt").exists()


def test_move_to_missing_folder_creates_it(workspace):
    (workspace / "r.txt").write_text("x")
    agent = agent_for('"filename":"r.txt","folder":"archive","intent":"move_file"')
    plan = agent.execute(agent.plan("x"), execute=True)
    assert plan.status == "executed" and (workspace / "archive" / "r.txt").exists()


def test_zip_and_unzip_roundtrip(workspace):
    (workspace / "a.txt").write_text("a")
    (workspace / "b.txt").write_text("b")
    agent = agent_for('"filenames":"a.txt, b.txt","intent":"zip_files","zipname":"out.zip"')
    assert agent.execute(agent.plan("x"), execute=True).status == "executed"
    assert (workspace / "out.zip").exists()
