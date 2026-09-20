"""The planner must decompose dynamically - different questions, different plans."""
from __future__ import annotations

import pytest

from app.agents.planner import Planner, detect_intents

PROJECTS = [{"key": "ATLAS", "name": "Project Atlas"}, {"key": "APOLLO", "name": "Project Apollo"}]
TOOLS = ["search_projects", "get_project", "search_issues", "get_issue", "get_issue_comments",
         "get_issue_changelog", "get_project_activity", "get_blocked_issues",
         "get_overdue_issues", "search_comments", "semantic_search"]


@pytest.fixture()
def planner():
    return Planner(llm=None, max_sub_questions=6)


def plan_for(planner: Planner, question: str, memory: str = ""):
    return planner.plan_heuristic(question, TOOLS, PROJECTS, memory)


def test_intent_detection_is_multi_label():
    intents = detect_intents("Summarize what changed in Project Atlas in Q2 and identify major risks")
    assert "change" in intents and "risk" in intents


def test_quarterly_change_question_produces_temporal_plan(planner):
    plan = plan_for(planner, "Summarize what changed in Project Atlas during Q2 2026 and identify the major risks")
    assert plan.needs_retrieval is True
    assert plan.entities["project_keys"] == ["ATLAS"]
    assert plan.time_range["label"] == "Q2 2026"
    assert plan.reasoning_strategy in {"temporal", "risk_analysis"}
    assert "get_project_activity" in plan.required_tools
    assert len(plan.sub_questions) >= 3


def test_causal_question_produces_a_different_plan(planner):
    plan = plan_for(planner, "What caused the delay in Project Atlas?")
    assert plan.reasoning_strategy == "causal"
    assert {"get_blocked_issues", "search_comments"} & set(plan.required_tools)


def test_ownership_question_is_narrow(planner):
    plan = plan_for(planner, "Who owns Project Apollo?")
    assert plan.entities["project_keys"] == ["APOLLO"]
    assert "get_project" in plan.required_tools
    assert len(plan.sub_questions) <= 2


def test_plans_differ_across_questions(planner):
    questions = [
        "Summarize what changed in Project Atlas during Q2 2026",
        "What caused the delay in Project Atlas?",
        "Who owns Project Apollo?",
        "What is blocked in Project Atlas?",
        "What was completed in Project Apollo last month?",
    ]
    signatures = {tuple(plan_for(planner, q).required_tools) for q in questions}
    assert len(signatures) >= 4, "planner is producing the same fixed workflow for every question"


def test_issue_specific_question_triggers_multi_hop(planner):
    plan = plan_for(planner, "Why did ATLAS-9 slip? What was discussed on it?")
    assert "ATLAS-9" in plan.entities["issue_keys"]
    assert any(sq.depends_on for sq in plan.sub_questions)


def test_durable_statement_is_routed_to_memory_not_retrieval(planner):
    plan = plan_for(planner, "Project Atlas is my highest priority this quarter.")
    assert plan.needs_retrieval is False
    assert plan.reasoning_strategy == "statement_capture"
    assert plan.sub_questions == []


def test_recall_question_needs_no_tracker_retrieval(planner):
    plan = plan_for(planner, "What did I say my priority was?")
    assert plan.needs_retrieval is False
    assert plan.reasoning_strategy == "memory_only"


def test_general_chat_needs_no_retrieval(planner):
    plan = plan_for(planner, "What can you do?")
    assert plan.needs_retrieval is False


def test_memory_supplies_the_missing_subject(planner):
    plan = plan_for(planner, "Given my priorities, what should I focus on?",
                    memory="- [preference] User's stated top priority is Project Atlas.")
    assert plan.entities["project_keys"] == ["ATLAS"]
    assert {"get_overdue_issues", "get_blocked_issues"} & set(plan.required_tools)


def test_planner_never_plans_a_tool_that_does_not_exist(planner):
    plan = planner.plan_heuristic(
        "What is blocked in Project Atlas?", ["search_issues"], PROJECTS, ""
    )
    assert set(plan.required_tools) <= {"search_issues"}
