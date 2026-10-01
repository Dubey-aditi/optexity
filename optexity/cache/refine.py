"""An optional LLM pass over a compiled automation: better instructions, placeholders.

The rule-based compiler is right about *where* each element is — every command was
verified unique on the live page — and poor at *describing* it: its instructions are
templates, and it only templates a typed value when it equals an input parameter
exactly. Both are language problems, which is what the model is for.

So the model proposes, and this module disposes. The model returns edits, never an
automation, and each edit is checked before it is applied:

* a command is never read back from the model. It changes only when a parameter
  needs it to: swapped for the element's verified exact locator with the value
  templated in, or — when no locator can follow the value — removed, so the LLM
  picks the element on every run (reported per node);
* a new parameter takes its value from the recording, never from the model: a typed
  or selected literal, or a clicked element's own label. It is refused while its
  value still sits in a locator nothing replaced (a calendar cell built around the
  date), which would keep selecting the old value;
* a placeholder must name a declared parameter at a valid index;
* a new value must resolve, for this run's parameters, to the literal the agent
  actually typed (up to case and punctuation) — so the next run with the same
  inputs does exactly what this one did;
* the result must validate as an `Automation`.

Anything that fails is reported and skipped; if the model call fails, the rule-based
automation is kept unchanged. Suggestions the model cannot express as a safe edit are
reported for a human.

A new parameter changes what the endpoint accepts: once synced, the dashboard
requires it from every caller. `RefineReport.parameters_added` lists them, and the
learning log says how to call the endpoint afterwards.
"""

import logging
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from browser_use.cache.views import TracedAction

from optexity.inference.agents.automation_refiner.automation_refiner import (
    AutomationRefinerAgent,
    NewParameter,
)
from optexity.schema.automation import ActionNode, Automation
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\[(\d+)\]\}")
_PARAMETER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_MAX_INSTRUCTION_LEN = 200

# Interactions that carry `prompt_instructions`, and which of them carry a value.
_DESCRIBED = (
    "input_text",
    "select_option",
    "click_element",
    "check",
    "uncheck",
    "hover",
)
_VALUED = ("input_text", "select_option")
_CLICKED = ("click_element", "check", "uncheck")


class RefineReport:
    def __init__(self) -> None:
        self.applied: list[str] = []
        self.rejected: list[str] = []
        self.parameters_added: dict[str, str] = {}  # name -> this run's value
        # Nodes that, to follow a parameter, lost their command and now have the
        # LLM pick the element on every run.
        self.llm_each_run: list[int] = []
        self.suggestions: list[str] = []
        self.token_usage = TokenUsage()
        self.error: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.applied or self.parameters_added)

    def render(self) -> str:
        if self.error:
            return f"llm refinement skipped: {self.error}"
        summary = (
            f"llm refinement: {len(self.applied)} edits applied, "
            f"{len(self.rejected)} rejected, {self.token_usage.total_tokens} tokens"
        )
        lines = [summary]
        lines.extend(
            f"  + parameter {name} = {value!r}"
            for name, value in self.parameters_added.items()
        )
        lines.extend(
            f"  ! node {node} now uses the LLM on every run: no locator can follow "
            "its parameter"
            for node in self.llm_each_run
        )
        lines.extend(f"  + {line}" for line in self.applied)
        lines.extend(f"  - {line}" for line in self.rejected)
        lines.extend(f"  ? {line}" for line in self.suggestions)
        return "\n".join(lines)


def _step(node: object) -> tuple[str, object] | None:
    interaction = getattr(node, "interaction_action", None)
    if not isinstance(node, ActionNode) or interaction is None:
        return None
    for field in _DESCRIBED:
        step = getattr(interaction, field, None)
        if step is not None:
            return field, step
    return None


def _value_of(field: str, step: object) -> str | None:
    if field == "input_text":
        return step.input_text
    if field == "select_option" and step.select_values and len(step.select_values) == 1:
        return step.select_values[0]
    return None


def _resolve(text: str, parameters: dict) -> str | None:
    """`text` with every placeholder replaced by this run's value, or None if one
    names an undeclared parameter or an index past the end."""
    failed = False

    def substitute(match: re.Match) -> str:
        nonlocal failed
        values = parameters.get(match.group(1))
        index = int(match.group(2))
        if not values or index >= len(values):
            failed = True
            return match.group(0)
        return str(values[index])

    resolved = _PLACEHOLDER_RE.sub(substitute, text)
    return None if failed else resolved


def _normalized(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def _clean(value: str | None) -> str | None:
    collapsed = " ".join((value or "").split())
    return collapsed or None


def _labels(action: TracedAction | None) -> list[str]:
    """What a clicked element is called: the values a parameter for it may hold."""
    if action is None or action.element is None:
        return []
    found = [_clean(action.element.accessible_name), _clean(action.element.text)]
    return list(dict.fromkeys(label for label in found if label))


def _describe_for_model(
    automation: Automation,
    positions: list[int],
    evidence: dict[int, TracedAction],
) -> list[dict]:
    described = []
    for position in positions:
        found = _step(automation.nodes[position])
        if found is None:
            continue
        field, step = found
        entry = {
            "node": position,
            "action": field,
            "command": step.command,
            "prompt_instructions": step.prompt_instructions,
        }
        value = _value_of(field, step)
        if value is not None:
            entry["value"] = value
        if field in _CLICKED and (labels := _labels(evidence.get(position))):
            entry["labels"] = labels
        described.append(entry)
    return described


def _check_instructions(instructions: str, parameters: dict) -> str | None:
    """Why these instructions must not be applied, or None if they are safe."""
    if not instructions or len(instructions) > _MAX_INSTRUCTION_LEN:
        return "instructions empty or too long"
    if _resolve(instructions, parameters) is None:
        return "instructions reference an undeclared parameter"
    return None


def _check_value(field: str, step: object, value: str, parameters: dict) -> str | None:
    """Why this value must not be applied, or None if it enters the same thing."""
    current = _value_of(field, step)
    if current is None:
        return f"{field} has no single value to change"
    resolved_new = _resolve(value, parameters)
    resolved_current = _resolve(current, parameters)
    if resolved_new is None:
        return "value references an undeclared parameter"
    if resolved_current is None or _normalized(resolved_new) != _normalized(
        resolved_current
    ):
        return f"value {value!r} would enter {resolved_new!r}, not {resolved_current!r}"
    return None


@dataclass
class _ParameterPlan:
    """What declaring one parameter would change, before anything is changed."""

    name: str
    value: str
    values: dict[int, str] = dataclass_field(
        default_factory=dict
    )  # node -> placeholder value
    commands: dict[int, str | None] = dataclass_field(
        default_factory=dict
    )  # node -> command

    @property
    def placeholder(self) -> str:
        return f"{{{self.name}[0]}}"


def _templated_locator(
    action: TracedAction, value: str, placeholder: str
) -> str | None:
    """The element's verified exact locator for `value`, with the value swapped for
    the placeholder, so the one step selects whatever the caller passes."""
    quoted = f'"{value}"'
    for candidate in sorted(action.candidates, key=lambda c: -c.score):
        if (
            candidate.match_count == 1
            and candidate.verify_text == value
            and candidate.locator.count(quoted) == 1
        ):
            return candidate.locator.replace(quoted, f'"{placeholder}"') + ".first"
    return None


def _plan(
    proposal: NewParameter,
    automation: Automation,
    positions: set[int],
    evidence: dict[int, TracedAction],
) -> tuple[_ParameterPlan | None, str | None]:
    """How `proposal` would be applied, or why it cannot be."""
    if not _PARAMETER_NAME_RE.match(proposal.name):
        return None, "not a snake_case name"
    if proposal.name in automation.parameters.input_parameters:
        return None, "already declared"
    if not proposal.nodes:
        return None, "no nodes"

    plan_values: dict[int, str] = {}
    for position in proposal.nodes:
        found = _step(automation.nodes[position]) if position in positions else None
        if found is None:
            return None, f"node {position} is not open to refinement"
        node_field, step = found
        if node_field in _VALUED:
            value = _value_of(node_field, step)
            if value is None or _PLACEHOLDER_RE.search(value):
                return None, f"node {position} has no literal value"
        elif node_field in _CLICKED:
            # The value comes from the recording, not the model: the model only
            # says which of the element's own labels it means.
            labels = _labels(evidence.get(position))
            value = _clean(proposal.value) if proposal.value else None
            if value is None and len(labels) == 1:
                value = labels[0]
            if value not in labels:
                return None, f"node {position} has no recorded label {proposal.value!r}"
        else:
            return None, f"node {position} cannot take a parameter"
        plan_values[position] = value
    if len(set(plan_values.values())) != 1:
        return None, f"nodes hold different values: {sorted(set(plan_values.values()))}"

    plan = _ParameterPlan(name=proposal.name, value=next(iter(plan_values.values())))
    for position in proposal.nodes:
        node_field, step = _step(automation.nodes[position])
        if node_field in _VALUED:
            plan.values[position] = plan.placeholder
        else:
            action = evidence.get(position)
            plan.commands[position] = (
                _templated_locator(action, plan.value, plan.placeholder)
                if action
                else None
            )
    return plan, None


def _conflicts(plan: _ParameterPlan, commands: dict[int, str | None]) -> str | None:
    """A locator the plan leaves alone that still contains its value — a calendar
    cell "Choose ... 1999" would keep picking 1999 whatever the year is set to."""
    needle = plan.value.casefold()
    for position, command in commands.items():
        if position not in plan.commands and command and needle in command.casefold():
            return f"{plan.value!r} is part of the locator {command}"
    return None


def _with_placeholder(instructions: str, plan: _ParameterPlan) -> str:
    """Instructions that name the parameter instead of this run's value, so a step
    resolved by the LLM looks for the caller's value."""
    pattern = re.compile(
        rf"(?<![\w{{]){re.escape(plan.value)}(?![\w}}])", re.IGNORECASE
    )
    replaced = pattern.sub(plan.placeholder, instructions)
    return replaced if plan.placeholder in replaced else instructions


def refine_automation(
    automation: Automation,
    agent: AutomationRefinerAgent,
    objective: str | None = None,
    positions: list[int] | None = None,
    evidence: dict[int, TracedAction] | None = None,
) -> tuple[Automation, RefineReport]:
    """Apply the model's safe edits to the nodes at `positions` (default: all).

    `evidence` is the recorded action behind each node. Without it a clicked
    choice cannot become a parameter, since its label and alternative locators
    exist only in the recording."""
    evidence = evidence or {}
    report = RefineReport()
    parameters = automation.parameters.input_parameters or {}
    if positions is None:
        positions = list(range(len(automation.nodes)))
    open_positions = [p for p in positions if _step(automation.nodes[p]) is not None]
    if not open_positions:
        return automation, report

    try:
        _, refinement, token_usage = agent.refine(
            _describe_for_model(automation, open_positions, evidence),
            parameters,
            objective,
        )
    except Exception as error:
        report.error = str(error)
        logger.warning(f"refine: model call failed, keeping rule-based output: {error}")
        return automation, report
    report.token_usage = token_usage
    report.suggestions = [s.strip() for s in refinement.suggestions if s.strip()]

    refined = automation.model_copy(deep=True)
    allowed = set(open_positions)
    parameters = refined.parameters.input_parameters

    plans: list[_ParameterPlan] = []
    for proposal in refinement.new_parameters:
        if any(plan.name == proposal.name for plan in plans):
            report.rejected.append(f"parameter {proposal.name}: proposed twice")
            continue
        plan, reason = _plan(proposal, refined, allowed, evidence)
        if plan is None:
            report.rejected.append(f"parameter {proposal.name}: {reason}")
            continue
        plans.append(plan)

    # Plans depend on each other: the year may only follow a parameter once the day
    # cell built around it has been handed to the LLM by the day's own plan. Drop
    # conflicting plans until the rest are consistent.
    while True:
        commands = {i: _command(node) for i, node in enumerate(refined.nodes)}
        for plan in plans:
            commands.update(plan.commands)
        conflicting = [(p, r) for p in plans if (r := _conflicts(p, commands))]
        if not conflicting:
            break
        for plan, reason in conflicting:
            plans.remove(plan)
            report.rejected.append(f"parameter {plan.name}: {reason}")

    expected_commands = [_command(node) for node in refined.nodes]
    for plan in plans:
        parameters[plan.name] = [plan.value]
        for position, placeholder in plan.values.items():
            node_field, step = _step(refined.nodes[position])
            if node_field == "input_text":
                step.input_text = placeholder
            else:
                step.select_values = [placeholder]
        for position, command in plan.commands.items():
            _, step = _step(refined.nodes[position])
            step.command = command
            step.xpath = None
            expected_commands[position] = command
            if command is None:
                report.llm_each_run.append(position)
        for position in allowed:
            found = _step(refined.nodes[position])
            if found and found[1].prompt_instructions:
                found[1].prompt_instructions = _with_placeholder(
                    found[1].prompt_instructions, plan
                )
        report.parameters_added[plan.name] = plan.value

    # The two halves of an edit are judged separately: a placeholder that cannot be
    # used is no reason to throw away a better description of the same step.
    for edit in refinement.edits:
        found = _step(refined.nodes[edit.node]) if edit.node in allowed else None
        if found is None:
            report.rejected.append(f"node {edit.node}: not a node open to refinement")
            continue
        field, step = found

        if edit.prompt_instructions is not None:
            instructions = edit.prompt_instructions.strip()
            reason = _check_instructions(instructions, parameters)
            if reason is None:
                step.prompt_instructions = instructions
                report.applied.append(f"node {edit.node}: {instructions!r}")
            else:
                report.rejected.append(f"node {edit.node}: {reason}")

        if edit.value is not None:
            reason = _check_value(field, step, edit.value, parameters)
            if reason is None:
                if field == "input_text":
                    step.input_text = edit.value
                else:
                    step.select_values = [edit.value]
                report.applied.append(f"node {edit.node}: value -> {edit.value!r}")
            else:
                report.rejected.append(f"node {edit.node}: {reason}")

    # A node that follows a parameter must say so in its instructions: the LLM
    # resolves those on a fallback, and for a node with no command on every run.
    for plan in plans:
        for position in plan.commands:
            _, step = _step(refined.nodes[position])
            if plan.placeholder not in (step.prompt_instructions or ""):
                step.prompt_instructions = (
                    f"{step.prompt_instructions} (the value to pick is {plan.placeholder})"
                ).strip()

    # Commands change only as the parameter plans said: templated from a verified
    # locator, or removed. Never from model output. Guard it anyway.
    if [_command(node) for node in refined.nodes] != expected_commands:
        report.error = "a command changed outside a parameter plan"
        return automation, report

    try:
        refined = Automation.model_validate_json(
            refined.model_dump_json(exclude_none=True)
        )
    except ValueError as error:
        report.error = f"refined automation failed validation: {error}"
        return automation, report
    return refined, report


def _command(node: object) -> str | None:
    found = _step(node)
    return getattr(found[1], "command", None) if found else None
