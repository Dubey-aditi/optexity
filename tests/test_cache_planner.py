"""The planner may decide what the automation is, never how an element is found."""

from browser_use.cache.views import ElementSignals, LocatorCandidate, TracedAction

from optexity.cache.planner import PlanRejected, build_from_plan, plan_automation
from optexity.inference.agents.automation_planner.automation_planner import (
    AutomationPlan,
    DroppedAction,
    PlanStep,
)
from optexity.schema.token_usage import TokenUsage


def _exact(role: str, value: str) -> LocatorCandidate:
    return LocatorCandidate(
        locator=f'get_by_role("{role}", name="{value}", exact=True)',
        kind="role+name",
        score=72,
        verify_role=role,
        verify_text=value,
        match_count=1,
    )


def _css(selector: str, score: int = 92, count: int = 1) -> LocatorCandidate:
    return LocatorCandidate(
        locator=f'locator("{selector}")',
        kind="id",
        score=score,
        verify_selector=selector,
        match_count=count,
    )


def _run() -> list[TracedAction]:
    """Type a name, click a stale calendar cell by mistake, pick Female, scroll, done."""

    def action(name, element=None, candidates=(), **params):
        return TracedAction(
            step=1,
            action_name=name,
            params=params,
            element=element,
            candidates=list(candidates),
        )

    return [
        action(
            "input",
            ElementSignals(tag_name="input", role="textbox", accessible_name="First"),
            [_css("#firstName")],
            text="Aditi",
        ),
        action(
            "click",
            ElementSignals(tag_name="div", role="gridcell", text="1"),
            [_css("div.day", score=53, count=2)],  # not unique: no locators shown
        ),
        action(
            "click",
            ElementSignals(tag_name="input", role="radio", accessible_name="Female"),
            [_css("#gender-radio-2"), _exact("radio", "Female")],
        ),
        action("scroll"),
        action("done", success=True),
    ]


def _plan(*steps: PlanStep, dropped=(DroppedAction(action=1, reason="stale"),)):
    return AutomationPlan(steps=list(steps), dropped=list(dropped))


GOOD = (
    PlanStep(
        action=0, locator=0, parameter="first_name", prompt_instructions="Type Aditi"
    ),
    PlanStep(
        action=2,
        locator=1,
        parameter="gender",
        label="Female",
        prompt_instructions="Select Female",
    ),
)


def test_a_good_plan_builds_with_recorded_locators_only():
    result = build_from_plan(_plan(*GOOD), _run(), {"first_name": ["Aditi"]})

    first, gender = (n.interaction_action for n in result.nodes)
    assert first.input_text.command == 'locator("#firstName").first'
    assert first.input_text.input_text == "{first_name[0]}"
    assert first.input_text.prompt_instructions == "Type {first_name[0]}"
    assert (
        gender.click_element.command
        == 'get_by_role("radio", name="{gender[0]}", exact=True).first'
    )
    assert result.parameters == {"gender": ["Female"]}


def _problems(plan, parameters=None) -> list[str]:
    try:
        build_from_plan(plan, _run(), parameters or {"first_name": ["Aditi"]})
    except PlanRejected as rejection:
        return rejection.problems
    raise AssertionError("plan was accepted")


def test_a_locator_the_recorder_did_not_verify_is_rejected():
    # Action 1's only candidate matched two elements, so it has no locators at all.
    invented = PlanStep(action=1, locator=0, prompt_instructions="Click day 1")
    problems = _problems(_plan(GOOD[0], invented, GOOD[1], dropped=()))
    assert any("locator 0 does not exist" in p for p in problems)


def test_steps_keep_the_order_the_actions_happened():
    problems = _problems(_plan(GOOD[1], GOOD[0]))
    assert "steps must stay in the order the actions happened" in problems


def test_every_action_is_used_or_dropped_with_a_reason():
    problems = _problems(_plan(*GOOD, dropped=()))
    assert problems == ["action 1 (click) is neither a step nor dropped"]


def test_a_click_parameter_must_be_one_of_the_elements_labels():
    wrong = PlanStep(
        action=2, locator=1, parameter="gender", label="Male", prompt_instructions="x"
    )
    problems = _problems(_plan(GOOD[0], wrong))
    assert "label 'Male' is not one of ['Female']" in problems[0]


def test_a_declared_parameter_must_hold_this_runs_value():
    problems = _problems(_plan(*GOOD), {"first_name": ["Jane"]})
    assert "first_name is 'Jane' for this run" in problems[0]


def test_a_choice_left_to_the_llm_takes_the_verified_locator_that_names_it():
    lazy = PlanStep(
        action=2,
        locator=None,
        parameter="gender",
        label="Female",
        prompt_instructions="x",
    )
    result = build_from_plan(_plan(GOOD[0], lazy), _run(), {"first_name": ["Aditi"]})
    assert result.report.upgraded == [1]
    assert result.report.llm_each_run == []
    assert "{gender[0]}" in result.nodes[1].interaction_action.click_element.command


class ScriptedPlanner:
    """Returns the given plans in order, recording the feedback it was sent."""

    def __init__(self, *plans: AutomationPlan):
        self.plans = list(plans)
        self.feedback: list[list[str] | None] = []

    def plan(self, actions, input_parameters, objective, problems=None, previous=None):
        self.feedback.append(problems)
        return "prompt", self.plans.pop(0), TokenUsage(total_tokens=10)


def test_a_rejected_plan_is_retried_with_its_problems():
    agent = ScriptedPlanner(_plan(*GOOD, dropped=()), _plan(*GOOD))
    result = plan_automation(_run(), "fill it", {"first_name": ["Aditi"]}, agent)

    assert result is not None and result.report.attempts == 2
    assert agent.feedback[1] == ["action 1 (click) is neither a step nor dropped"]
    assert result.report.token_usage.total_tokens == 20


def test_no_plan_after_repeated_rejection_means_the_rules_take_over():
    bad = _plan(*GOOD, dropped=())
    agent = ScriptedPlanner(bad, bad, bad)
    assert plan_automation(_run(), None, {"first_name": ["Aditi"]}, agent) is None
