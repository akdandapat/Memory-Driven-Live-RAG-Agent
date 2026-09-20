"""Memory: selective extraction, cross-session recall, conflict resolution, isolation."""
from __future__ import annotations

import pytest


@pytest.fixture()
def memory(services):
    return services.memory


def test_questions_and_small_talk_are_not_stored(memory, run):
    report = run(memory.observe(user_id="u1", session_id="s1", run_id=None,
                                user_message="What changed in Project Atlas last quarter?"))
    assert report["actions"] == []
    assert any("question" in r["reason"] for r in report["rejected"])
    assert run(memory.list_memories("u1")) == []


def test_durable_preference_is_stored_with_structure(memory, run):
    run(memory.observe(user_id="u1", session_id="s1", run_id=None,
                       user_message="Project Atlas is my highest priority this quarter."))
    records = run(memory.list_memories("u1"))
    assert len(records) == 1
    record = records[0]
    assert record["type"] == "preference"
    assert "Atlas" in record["content"]
    assert 0.0 < record["importance"] <= 1.0
    assert record["status"] == "active"


def test_repeating_the_same_statement_does_not_duplicate(memory, run):
    for _ in range(3):
        run(memory.observe(user_id="u1", session_id="s1", run_id=None,
                           user_message="Project Atlas is my highest priority this quarter."))
    assert len(run(memory.list_memories("u1"))) == 1


def test_contradicting_statement_supersedes_rather_than_appends(memory, run):
    run(memory.observe(user_id="u1", session_id="s1", run_id=None,
                       user_message="Project Atlas is my highest priority."))
    report = run(memory.observe(user_id="u1", session_id="s2", run_id=None,
                                user_message="Actually, Project Apollo has become my highest priority now."))
    assert report["actions"][0]["action"] == "superseded"

    active = run(memory.list_memories("u1"))
    assert len(active) == 1
    assert "Apollo" in active[0]["content"]

    everything = run(memory.list_memories("u1", include_inactive=True))
    superseded = [m for m in everything if m["status"] == "superseded"]
    assert len(superseded) == 1 and "Atlas" in superseded[0]["content"]


def test_recall_is_cross_session_and_scored(memory, run):
    run(memory.observe(user_id="u1", session_id="session-one", run_id=None,
                       user_message="Project Atlas is my highest priority this quarter."))
    recalled = run(memory.recall("u1", "what should I focus on?"))
    assert recalled and "Atlas" in recalled[0].content
    assert recalled[0].score > 0


def test_irrelevant_memory_is_not_injected(memory, run):
    run(memory.observe(user_id="u1", session_id="s1", run_id=None,
                       user_message="I prefer answers that are short and use bullet points."))
    recalled = run(memory.recall("u1", "quantum chromodynamics lattice spacing"))
    assert recalled == []


def test_memory_is_isolated_between_users(memory, run):
    run(memory.observe(user_id="alice", session_id="s1", run_id=None,
                       user_message="Project Atlas is my highest priority."))
    assert run(memory.recall("bob", "what is my priority?")) == []
    assert run(memory.list_memories("bob")) == []


def test_delete_only_affects_the_owner(memory, run):
    run(memory.observe(user_id="alice", session_id="s1", run_id=None,
                       user_message="Project Atlas is my highest priority."))
    memory_id = run(memory.list_memories("alice"))[0]["memory_id"]
    assert run(memory.delete("bob", memory_id)) is False
    assert run(memory.delete("alice", memory_id)) is True
    assert run(memory.list_memories("alice")) == []


def test_short_term_memory_is_session_scoped(services, run):
    short_term = services.short_term
    run(short_term.ensure_session("s1", "u1"))
    run(short_term.add_message("s1", "u1", "user", "first question"))
    run(short_term.add_message("s1", "u1", "assistant", "first answer"))
    context = run(short_term.context_block("s1"))
    assert "first question" in context
    assert run(short_term.context_block("s2")) == ""


def test_session_cannot_be_hijacked_by_another_user(services, run):
    run(services.short_term.ensure_session("shared", "alice"))
    with pytest.raises(PermissionError):
        run(services.short_term.ensure_session("shared", "mallory"))
