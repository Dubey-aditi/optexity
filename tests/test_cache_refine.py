"""The LLM refinement pass may improve wording and template values, nothing else."""

from optexity.cache.refine import refine_automation
from optexity.inference.agents.automation_refiner.automation_refiner import (
    AutomationRefinement,
    NewParameter,
    NodeEdit,
)
from optexity.schema.automation import Automation
from optexity.schema.token_usage import TokenUsage


class FakeRefiner:
    def __init__(self, refinement: AutomationRefinement | Exception):
        self.refinement = refinement
        self.seen_nodes: list[dict] = []

    def refine(self, nodes, input_parameters, objective):
        self.seen_nodes = nodes
        if isinstance(self.refinement, Exception):
            raise self.refinement
        return "prompt", self.refinement, TokenUsage(total_tokens=42)


def _automation() -> Automation:
    return Automation.model_validate(
        {
            "url": "https://example.com",
            "parameters": {
                "input_parameters": {"first_name": ["Aditi"], "state": ["NY"]},
                "generated_parameters": {},
            },
            "nodes": [
                {
                    "type": "action_node",
                    "interaction_action": {
                        "input_text": {
                            "command": 'locator("#firstName").first',
                            "prompt_instructions": "Enter 'aditi' into the field",
                            "input_text": "aditi",
                        }
                    },
                },
                {
                    "type": "action_node",
                    "interaction_action": {
                        "select_option": {
                            "command": 'locator("#state").first',
                            "prompt_instructions": "Select 'NY' in the 1900 1901 dropdown",
                            "select_values": ["NY"],
                        }
                    },
                },
            ],
        }
    )


def _refine(
    *edits: NodeEdit,
    suggestions: list[str] | None = None,
    new_parameters: list[NewParameter] | None = None,
    automation: Automation | None = None,
):
    agent = FakeRefiner(
        AutomationRefinement(
            edits=list(edits),
            suggestions=suggestions or [],
            new_parameters=new_parameters or [],
        )
    )
    return refine_automation(automation or _automation(), agent, "fill the form")


def test_applies_instructions_and_a_placeholder_that_resolves_to_the_same_value():
    refined, report = _refine(
        NodeEdit(
            node=0,
            prompt_instructions="Type {first_name[0]} into the 'First Name' box",
            value="{first_name[0]}",  # "Aditi" vs typed "aditi": same up to case
        )
    )
    step = refined.nodes[0].interaction_action.input_text
    assert step.input_text == "{first_name[0]}"
    assert step.prompt_instructions == "Type {first_name[0]} into the 'First Name' box"
    assert step.command == 'locator("#firstName").first'
    assert len(report.applied) == 2
    assert report.token_usage.total_tokens == 42


def test_rejects_a_placeholder_that_would_type_something_else():
    refined, report = _refine(NodeEdit(node=1, value="{first_name[0]}"))
    assert refined.nodes[1].interaction_action.select_option.select_values == ["NY"]
    assert "would enter" in report.rejected[0]


def test_rejects_undeclared_parameters_and_unknown_nodes():
    _, report = _refine(
        NodeEdit(node=0, value="{last_name[0]}"),
        NodeEdit(node=0, prompt_instructions="Type {email[0]}"),
        NodeEdit(node=7, prompt_instructions="Click it"),
    )
    assert not report.changed
    assert len(report.rejected) == 3


def test_never_changes_a_command_and_reports_suggestions():
    refined, report = _refine(
        NodeEdit(node=1, prompt_instructions="Select the state {state[0]}"),
        suggestions=["Declare dob as a parameter"],
    )
    assert [
        n.interaction_action.select_option or n.interaction_action.input_text
        for n in refined.nodes
    ][1].command == 'locator("#state").first'
    assert report.suggestions == ["Declare dob as a parameter"]


def test_keeps_the_rule_based_automation_when_the_model_fails():
    original = _automation()
    refined, report = refine_automation(
        original, FakeRefiner(RuntimeError("rate limited")), None
    )
    assert refined == original
    assert "rate limited" in report.render()


def test_only_sends_the_requested_nodes():
    agent = FakeRefiner(AutomationRefinement())
    refine_automation(_automation(), agent, None, positions=[1])
    assert [node["node"] for node in agent.seen_nodes] == [1]


def test_keeps_good_instructions_when_the_value_in_the_same_edit_is_rejected():
    refined, report = _refine(
        NodeEdit(
            node=0,
            prompt_instructions="Type the first name into the 'First Name' box",
            value="{nickname[0]}",
        )
    )
    step = refined.nodes[0].interaction_action.input_text
    assert step.prompt_instructions == "Type the first name into the 'First Name' box"
    assert step.input_text == "aditi"
    assert len(report.applied) == 1
    assert len(report.rejected) == 1


def _literal_automation() -> Automation:
    """A compiled run that declared no inputs, the way an agentic task usually is."""
    return Automation.model_validate(
        {
            "url": "https://example.com",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [
                {
                    "type": "action_node",
                    "interaction_action": {
                        "input_text": {
                            "command": 'locator("#email").first',
                            "prompt_instructions": "Enter it",
                            "input_text": "a@b.com",
                        }
                    },
                },
                {
                    "type": "action_node",
                    "interaction_action": {
                        "select_option": {
                            "command": 'locator("select.year").first',
                            "prompt_instructions": "Select it",
                            "select_values": ["1999"],
                        }
                    },
                },
                {
                    "type": "action_node",
                    "interaction_action": {
                        "click_element": {
                            "command": 'get_by_label("Choose Friday, January 1st, 1999").first',
                            "prompt_instructions": "Click the day",
                        }
                    },
                },
            ],
        }
    )


def test_declares_a_parameter_from_the_literal_it_replaces():
    refined, report = _refine(
        NodeEdit(
            node=0, prompt_instructions="Type the email {email[0]} into the email box"
        ),
        new_parameters=[NewParameter(name="email", nodes=[0])],
        automation=_literal_automation(),
    )
    assert refined.parameters.input_parameters == {"email": ["a@b.com"]}
    step = refined.nodes[0].interaction_action.input_text
    assert step.input_text == "{email[0]}"
    assert step.prompt_instructions == "Type the email {email[0]} into the email box"
    assert report.parameters_added == {"email": "a@b.com"}


def test_refuses_a_parameter_whose_value_is_baked_into_a_locator():
    refined, report = _refine(
        new_parameters=[NewParameter(name="dob_year", nodes=[1])],
        automation=_literal_automation(),
    )
    assert refined.parameters.input_parameters == {}
    assert refined.nodes[1].interaction_action.select_option.select_values == ["1999"]
    assert "part of the locator" in report.rejected[0]


def test_refuses_bad_names_existing_names_and_nodes_without_a_value():
    _, report = _refine(
        new_parameters=[
            NewParameter(name="Email Address", nodes=[0]),
            NewParameter(name="first_name", nodes=[0]),
            NewParameter(name="day", nodes=[2]),
        ]
    )
    assert not report.parameters_added
    assert [r.split(":")[1].strip() for r in report.rejected] == [
        "not a snake_case name",
        "already declared",
        "node 2 is not open to refinement",
    ]


def _form_with_choices():
    """demoqa's shape: a radio, a date picker, a react-select option."""
    from browser_use.cache.views import ElementSignals, LocatorCandidate, TracedAction

    def click(command, prompt):
        return {
            "type": "action_node",
            "interaction_action": {
                "click_element": {"command": command, "prompt_instructions": prompt}
            },
        }

    def select(command, value):
        return {
            "type": "action_node",
            "interaction_action": {
                "select_option": {
                    "command": command,
                    "prompt_instructions": f"Select '{value}'",
                    "select_values": [value],
                }
            },
        }

    day_label = "Choose Friday, January 1st, 1999"
    automation = Automation.model_validate(
        {
            "url": "https://example.com",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [
                click('locator("#gender-radio-2").first', "Click the 'Female' radio"),
                select('locator("select.month").first', "January"),
                select('locator("select.year").first', "1999"),
                click(f'get_by_label("{day_label}").first', "Click the day '1st'"),
                click(
                    'get_by_role("option", name="Uttar Pradesh", exact=True).first',
                    "Click the 'Uttar Pradesh' option",
                ),
            ],
        }
    )

    def traced(role, name, text, candidates):
        return TracedAction(
            step=1,
            action_name="click",
            element=ElementSignals(
                tag_name="div", role=role, accessible_name=name, text=text
            ),
            candidates=[LocatorCandidate(**c) for c in candidates],
        )

    exact = lambda role, value, count: {  # noqa: E731
        "locator": f'get_by_role("{role}", name="{value}", exact=True)',
        "kind": "role+name",
        "score": 72,
        "verify_role": role,
        "verify_text": value,
        "match_count": count,
    }
    evidence = {
        0: traced("radio", "Female", None, [exact("radio", "Female", 1)]),
        # Two cells read "1" (Jan 1 and the next month's 1): no exact locator for it.
        3: traced("gridcell", day_label, "1", [exact("gridcell", "1", 2)]),
        4: traced(
            "option",
            "Uttar Pradesh",
            "Uttar Pradesh",
            [exact("option", "Uttar Pradesh", 1)],
        ),
    }
    return automation, evidence


def _refine_choices(*proposals: NewParameter):
    automation, evidence = _form_with_choices()
    agent = FakeRefiner(AutomationRefinement(new_parameters=list(proposals)))
    return refine_automation(automation, agent, None, None, evidence)


def test_clicked_choices_become_parameters_with_locators_that_follow_them():
    refined, report = _refine_choices(
        NewParameter(name="gender", nodes=[0], value="Female"),
        NewParameter(name="state", nodes=[4], value="Uttar Pradesh"),
    )
    nodes = refined.nodes
    assert refined.parameters.input_parameters == {
        "gender": ["Female"],
        "state": ["Uttar Pradesh"],
    }
    assert (
        nodes[0].interaction_action.click_element.command
        == 'get_by_role("radio", name="{gender[0]}", exact=True).first'
    )
    assert (
        nodes[4].interaction_action.click_element.command
        == 'get_by_role("option", name="{state[0]}", exact=True).first'
    )
    assert (
        "{gender[0]}" in nodes[0].interaction_action.click_element.prompt_instructions
    )
    assert report.llm_each_run == []


def test_a_date_is_parameterised_once_its_day_cell_is_handed_to_the_llm():
    refined, report = _refine_choices(
        NewParameter(name="birth_day", nodes=[3], value="1"),
        NewParameter(name="birth_month", nodes=[1]),
        NewParameter(name="birth_year", nodes=[2]),
    )
    nodes = refined.nodes
    assert set(refined.parameters.input_parameters) == {
        "birth_day",
        "birth_month",
        "birth_year",
    }
    day = nodes[3].interaction_action.click_element
    assert day.command is None  # no locator can follow the day
    assert "{birth_day[0]}" in day.prompt_instructions
    assert report.llm_each_run == [3]
    assert nodes[1].interaction_action.select_option.select_values == [
        "{birth_month[0]}"
    ]


def test_part_of_a_date_is_refused_while_the_day_cell_still_pins_it():
    refined, report = _refine_choices(NewParameter(name="birth_year", nodes=[2]))
    assert refined.parameters.input_parameters == {}
    assert "part of the locator" in report.rejected[0]


def test_a_clicked_choice_takes_only_a_label_the_element_really_has():
    _, report = _refine_choices(NewParameter(name="gender", nodes=[0], value="Male"))
    assert not report.parameters_added
    assert "no recorded label 'Male'" in report.rejected[0]
