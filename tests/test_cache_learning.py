"""The learning layer's pure parts: compiling a trace and learning from a run."""

from browser_use.cache.locators import scored_candidates
from browser_use.cache.views import ElementSignals, LocatorCandidate, TracedAction

from optexity.cache.compiler import compile_trace, learn_automation
from optexity.schema.automation import Automation

URL = "https://example.com/form"


def _action(
    step: int,
    name: str,
    tag: str = "input",
    locator: str | None = None,
    node: int | None = 0,
    **params,
) -> TracedAction:
    # Identity comes from the locator, as it would from the DOM path on a real page:
    # two actions on one element share it, which is what the collapse rule keys on.
    element = ElementSignals(
        tag_name=tag,
        attributes={"placeholder": f"field {step}"},
        xpath=f"/html/body/{tag}[{locator}]",
        element_hash=f"{tag}:{locator}",
    )
    candidates = (
        [LocatorCandidate(locator=locator, kind="id", score=92, match_count=1)]
        if locator
        else []
    )
    return TracedAction(
        step=step,
        action_name=name,
        params=params,
        index=step,
        element=None if name == "done" else element,
        candidates=candidates,
        url_before=URL,
        context={} if node is None else {"node": node},
    )


def _agentic_automation() -> Automation:
    return Automation.model_validate(
        {
            "url": URL,
            "parameters": {
                "input_parameters": {"name": ["aditi"]},
                "generated_parameters": {},
            },
            "nodes": [
                {
                    "type": "action_node",
                    "interaction_action": {
                        "agentic_task": {
                            "task": "fill the form",
                            "max_steps": 10,
                            "backend": "browser_use",
                        }
                    },
                }
            ],
        }
    )


def _agent_run() -> list[TracedAction]:
    return [
        _action(1, "screenshot", node=0),
        _action(2, "input", locator='locator("#name")', text="draft"),
        _action(3, "input", locator='locator("#name")', text="aditi"),
        _action(4, "click", tag="select", locator='locator("#state")'),
        _action(
            5, "select_dropdown", tag="select", locator='locator("#state")', text="NY"
        ),
        _action(6, "done", success=True),
    ]


def test_compile_prunes_noise_repeats_and_refused_clicks():
    automation, report = compile_trace(_agent_run(), url=URL)

    kinds = [
        next(
            k
            for k, v in node.interaction_action
            if v is not None and k in ("input_text", "select_option", "click_element")
        )
        for node in automation.nodes
    ]
    assert kinds == ["input_text", "select_option"]
    assert automation.nodes[0].interaction_action.input_text.input_text == "aditi"
    dropped = {reason for _, _, reason in report.dropped}
    assert "overwritten by a later input on the same element" in dropped
    assert "browser-use does not click this element type" in dropped


def test_compile_templates_values_that_came_from_input_parameters():
    automation, _ = compile_trace(
        _agent_run(), url=URL, input_parameters={"name": ["aditi"]}
    )
    step = automation.nodes[0].interaction_action.input_text
    assert step.input_text == "{name[0]}"
    assert "{name[0]}" in step.prompt_instructions


def test_learn_replaces_a_successful_agentic_task():
    learned, report = learn_automation(
        _agentic_automation(), _agent_run(), input_parameters={"name": ["aditi"]}
    )

    assert report.changed
    assert not any(n.interaction_action.agentic_task for n in learned.nodes)
    assert (
        learned.nodes[0].interaction_action.input_text.command
        == 'locator("#name").first'
    )
    # Parameters are the source automation's, so the endpoint's contract is unchanged.
    assert learned.parameters.input_parameters == {"name": ["aditi"]}


def test_learn_keeps_an_agentic_task_that_did_not_succeed():
    unfinished = [a for a in _agent_run() if a.action_name != "done"]
    learned, report = learn_automation(_agentic_automation(), unfinished)

    assert not report.changed
    assert learned.nodes[0].interaction_action.agentic_task is not None


def test_learn_heals_only_the_node_the_llm_rescued():
    compiled, _ = learn_automation(_agentic_automation(), _agent_run())
    # Next run: node 1 (the select) had no working command and the LLM fallback
    # picked an element that has a verified unique locator.
    compiled.nodes[1].interaction_action.select_option.command = None
    rescue = [
        _action(
            1,
            "select_dropdown",
            tag="select",
            locator='locator("#state-v2")',
            node=1,
            text="NY",
        )
    ]

    healed, report = learn_automation(compiled, rescue)

    assert [p for p, _, _ in report.patched] == [1]
    assert (
        healed.nodes[1].interaction_action.select_option.command
        == 'locator("#state-v2").first'
    )
    assert healed.nodes[0] == compiled.nodes[0]


def test_learn_does_not_heal_with_an_unverified_locator():
    compiled, _ = learn_automation(_agentic_automation(), _agent_run())
    rescue = [
        _action(1, "select_dropdown", tag="select", locator=None, node=1, text="NY")
    ]

    healed, report = learn_automation(compiled, rescue)

    assert not report.changed
    assert healed.nodes[1] == compiled.nodes[1]


def test_learn_ignores_actions_without_a_node_tag():
    _, report = learn_automation(
        _agentic_automation(),
        [_action(1, "input", locator='locator("#x")', node=None, text="a")],
    )
    assert not report.changed


def test_compile_drops_a_click_that_the_following_input_repeats():
    run = [
        _action(1, "click", locator='locator("#email")'),
        _action(2, "input", locator='locator("#email")', text="a@b.c"),
        _action(3, "click", locator='locator("#submit")', tag="button"),
    ]
    automation, report = compile_trace(run, url=URL)

    assert automation.nodes[0].interaction_action.input_text is not None
    assert automation.nodes[1].interaction_action.click_element is not None
    assert len(automation.nodes) == 2
    assert "the following input clicks the same field" in {
        r for _, _, r in report.dropped
    }


def test_learning_switches_are_independent():
    compiled, _ = learn_automation(_agentic_automation(), _agent_run())
    rescue = [
        _action(
            1,
            "select_dropdown",
            tag="select",
            locator='locator("#v2")',
            node=1,
            text="NY",
        )
    ]

    # Self-heal off: a rescued step is left alone.
    _, report = learn_automation(compiled, rescue, heal=False)
    assert not report.changed

    # Learning off: an agentic task is left alone, even one that succeeded.
    _, report = learn_automation(
        _agentic_automation(), _agent_run(), compile_agentic=False
    )
    assert not report.changed


def test_role_and_text_locators_are_exact_and_countable():
    # A react-select option: hashed class, counter id, so role and text are all
    # that is left — and they must be exact to be safe to count.
    option = ElementSignals(
        tag_name="div",
        attributes={"class": "css-10wo9uf-option", "id": "react-select-3-option-1"},
        role="option",
        accessible_name="Uttar Pradesh",
        text="Uttar Pradesh",
    )
    best = scored_candidates(option)[0]
    assert best.locator == 'get_by_role("option", name="Uttar Pradesh", exact=True)'
    assert (best.verify_role, best.verify_text) == ("option", "Uttar Pradesh")

    placeholder = ElementSignals(tag_name="div", text="Select State")
    text = next(c for c in scored_candidates(placeholder) if c.kind == "text")
    assert text.locator == 'get_by_text("Select State", exact=True)'
    assert text.verify_role is None and text.verify_text == "Select State"


def _agent_output_type():
    from browser_use.agent.views import AgentOutput
    from browser_use.tools.service import Tools

    return AgentOutput.type_with_custom_actions(Tools().registry.create_action_model())


def test_misshapen_agent_outputs_are_repaired_without_touching_actions():
    from optexity.inference.models.chat_litellm import _repair_agent_output

    output = _agent_output_type()
    # The three shapes Sonnet produced on demoqa, reconstructed around their tails.
    stray_names = '{"memory": "m", "action": [{"click": {"index": 1}}, "evaluation_previous_goal", "memory", "next_goal", {"click": {"index": 18}}]}'
    goal_inside = '{"evaluation_previous_goal": "ok", "memory": "m", "action": [{"next_goal": "Open the date picker.", "screenshot": {}}]}'
    goal_beside_click = '{"memory": "m", "action": [{"next_goal": "Select Lucknow.", "click": {"index": 480}}]}'

    repaired = _repair_agent_output(stray_names, output)
    clicks = [
        a.model_dump(exclude_none=True)["click"]["index"] for a in repaired.action
    ]
    assert clicks == [1, 18]

    repaired = _repair_agent_output(goal_inside, output)
    assert repaired.next_goal == "Open the date picker."
    assert list(repaired.action[0].model_dump(exclude_none=True)) == ["screenshot"]

    repaired = _repair_agent_output(goal_beside_click, output)
    assert repaired.next_goal == "Select Lucknow."
    assert repaired.action[0].model_dump(exclude_none=True)["click"]["index"] == 480


def test_unrepairable_agent_outputs_still_fail():
    from optexity.inference.models.chat_litellm import _repair_agent_output

    output = _agent_output_type()
    assert (
        _repair_agent_output(
            '{"memory": "m", "action": ["memory", "next_goal"]}', output
        )
        is None
    )
    assert (
        _repair_agent_output('{"memory": "m", "action": [{"fly_to_moon": {}}]}', output)
        is None
    )
    assert _repair_agent_output("not json at all", output) is None


def test_a_browser_internal_role_falls_back_to_an_exact_text_locator():
    # Chromium reports a <label> as role "LabelText", which get_by_role rejects.
    label = ElementSignals(
        tag_name="label", role="LabelText", accessible_name="Music", text="Music"
    )
    kinds = {c.kind: c for c in scored_candidates(label)}
    assert "role+name" not in kinds
    assert kinds["text"].locator == 'get_by_text("Music", exact=True)'


def test_an_anonymous_widget_part_is_addressed_by_its_stable_ancestor():
    # react-select: the control has only generated classes; its wrapper is id="state".
    control = ElementSignals(
        tag_name="div", attributes={"class": "css-13cymwt-control"}, anchor_id="state"
    )
    best = scored_candidates(control)[0]
    assert (best.locator, best.verify_selector) == ('locator("#state")', "#state")

    field = ElementSignals(
        tag_name="input", attributes={"id": "react-select-3-input"}, anchor_id="state"
    )
    assert any(c.locator == 'locator("#state input")' for c in scored_candidates(field))

    # An option has text of its own: it must never fall back to the dropdown itself.
    option = ElementSignals(
        tag_name="div",
        role="option",
        accessible_name="NCR",
        text="NCR",
        anchor_id="state",
    )
    assert all(c.kind != "anchor" for c in scored_candidates(option))
