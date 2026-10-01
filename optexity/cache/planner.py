"""Have an LLM plan the deterministic automation for a recorded agent run.

The rule-based compiler (`compiler.py`) translates every action it can and drops a
fixed list of noise. It cannot tell an exploratory mistake from a real step, merge or
reorder, or decide which values a caller will want to vary. A model can — but a model
asked to write the automation would also write locators, and an invented locator is
exactly the failure this whole layer exists to remove.

So the model plans and this module builds. The plan refers to recorded actions and
to their verified locators by index; it never contains a locator. Building checks
everything the plan claims:

* every recorded action is either a step or dropped with a reason;
* a step uses only a locator verified unique on the live page for that action, or
  none (the LLM picks the element on every run, and the report says so);
* a parameter's value comes from the recording — the typed text, the picked option,
  or one of the clicked element's labels — and an existing parameter must hold this
  run's value;
* a clicked step bound to a parameter keeps a command only if a verified exact
  locator names that value, which is then templated;
* no locator still contains a parameter's value (a calendar cell built around a date
  would keep selecting it);
* the result validates as an Automation.

A plan that fails is sent back with the list of problems, a bounded number of times.
If no plan holds up, the caller falls back to the rule-based compiler.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from browser_use.cache.views import TracedAction

from optexity.cache.compiler import NOISE_ACTIONS, key_press_for
from optexity.cache.refine import (
    _PARAMETER_NAME_RE,
    _clean,
    _labels,
    _normalized,
    _resolve,
)
from optexity.inference.agents.automation_planner.automation_planner import (
    AutomationPlan,
    AutomationPlannerAgent,
)
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoToUrlAction,
    InputTextAction,
    InteractionAction,
    SelectOptionAction,
)
from optexity.schema.automation import ActionNode
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_MAX_INSTRUCTION_LEN = 200
_STEP_ACTIONS = frozenset(
    {"input", "click", "select_dropdown", "send_keys", "navigate"}
)
_SHOWN_ATTRIBUTES = ("id", "name", "type", "placeholder", "aria-label", "role", "for")


class PlanRejected(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class PlanReport:
    """What the planner produced and what it cost. Shaped like CompileReport where
    the learning pass reads it: `render()` and `sources`."""

    attempts: int = 0
    steps: int = 0
    dropped: list[tuple[int, str]] = dataclass_field(default_factory=list)
    parameters_added: dict[str, str] = dataclass_field(default_factory=dict)
    llm_each_run: list[int] = dataclass_field(default_factory=list)
    upgraded: list[int] = dataclass_field(default_factory=list)
    suggestions: list[str] = dataclass_field(default_factory=list)
    rejections: list[list[str]] = dataclass_field(default_factory=list)
    token_usage: TokenUsage = dataclass_field(default_factory=TokenUsage)
    sources: list[TracedAction] = dataclass_field(default_factory=list)
    planned: bool = True

    def render(self) -> str:
        lines = [
            f"planned by the LLM: {self.steps} steps from attempt {self.attempts}, "
            f"{self.token_usage.total_tokens} tokens"
        ]
        for attempt, problems in enumerate(self.rejections, start=1):
            lines.append(f"  attempt {attempt} rejected:")
            lines.extend(f"    - {problem}" for problem in problems)
        lines.extend(
            f"  + parameter {name} = {value!r}"
            for name, value in self.parameters_added.items()
        )
        lines.extend(
            f"  ! step {node} uses the LLM on every run: no verified locator follows it"
            for node in self.llm_each_run
        )
        lines.extend(
            f"  ^ step {node}: the plan left it to the LLM, but a verified locator "
            "names its value; using that"
            for node in self.upgraded
        )
        lines.extend(f"  - dropped action {a}: {reason}" for a, reason in self.dropped)
        lines.extend(f"  ? {line}" for line in self.suggestions)
        return "\n".join(lines)


@dataclass
class PlanResult:
    nodes: list[ActionNode]
    parameters: dict[str, list[str]]  # parameters the plan added
    report: PlanReport


def _verified(action: TracedAction) -> list:
    """The action's locators that matched exactly one element, best first. The
    order is the one shown to the model, so indexes mean the same thing to both."""
    unique = [c for c in action.candidates if c.match_count == 1]
    return sorted(unique, key=lambda c: (-c.score, c.locator))


def describe_actions(actions: list[TracedAction]) -> list[dict]:
    """The recorded run as the model sees it: compact, and nothing it can misuse."""
    described = []
    for index, action in enumerate(actions):
        entry: dict = {"action": index, "type": action.action_name}
        params = {
            key: action.params[key]
            for key in ("text", "url", "keys", "success")
            if key in action.params
        }
        if params:
            entry["params"] = params
        if action.error:
            entry["error"] = action.error[:160]
        if (
            action.url_after
            and action.url_before
            and action.url_after != action.url_before
        ):
            entry["navigated_to"] = action.url_after
        if action.element is not None:
            element = action.element
            entry["element"] = {
                key: value
                for key, value in {
                    "tag": element.tag_name,
                    "role": element.role,
                    "name": _clean(element.accessible_name),
                    "text": (_clean(element.text) or "")[:80] or None,
                    **{
                        attribute: element.attributes.get(attribute)
                        for attribute in _SHOWN_ATTRIBUTES
                    },
                }.items()
                if value
            }
            if action.action_name == "click" and (labels := _labels(action)):
                entry["labels"] = labels
        if action.action_name in _STEP_ACTIONS:
            entry["locators"] = [candidate.locator for candidate in _verified(action)]
        described.append(entry)
    return described


def _literal(action: TracedAction) -> str | None:
    if action.action_name in ("input", "select_dropdown"):
        value = action.params.get("text")
        return str(value) if value is not None else None
    return None


def build_from_plan(
    plan: AutomationPlan,
    actions: list[TracedAction],
    input_parameters: dict,
) -> PlanResult:
    """The automation the plan describes, or `PlanRejected` listing every problem."""
    problems: list[str] = []
    report = PlanReport(steps=len(plan.steps))
    declared = {name: list(values) for name, values in (input_parameters or {}).items()}
    added: dict[str, str] = {}

    # Coverage: nothing the agent did may disappear silently.
    used = [step.action for step in plan.steps]
    dropped = {d.action: d.reason for d in plan.dropped}
    for index in used:
        if not 0 <= index < len(actions):
            problems.append(f"step uses action {index}, which does not exist")
    if len(set(used)) != len(used):
        problems.append("an action is used by more than one step")
    if used != sorted(used):
        problems.append("steps must stay in the order the actions happened")
    for index, action in enumerate(actions):
        if action.action_name in NOISE_ACTIONS or index in used or index in dropped:
            continue
        problems.append(
            f"action {index} ({action.action_name}) is neither a step nor dropped"
        )
    if problems:
        raise PlanRejected(problems)

    nodes: list[ActionNode] = []
    templated: dict[int, str] = {}  # node -> parameter its command follows
    for node_index, step in enumerate(plan.steps):
        action = actions[step.action]
        where = f"step {node_index} (action {step.action})"
        if action.action_name not in _STEP_ACTIONS:
            problems.append(f"{where}: a {action.action_name} cannot be a step")
            continue
        if action.error:
            problems.append(f"{where}: the action failed; drop it")
            continue

        verified = _verified(action)
        candidate = None
        if step.locator is not None:
            if not 0 <= step.locator < len(verified):
                problems.append(
                    f"{where}: locator {step.locator} does not exist; it has "
                    f"{len(verified)} verified locators"
                )
                continue
            candidate = verified[step.locator]
        command = f"{candidate.locator}.first" if candidate else None

        value = _literal(action)
        name = step.parameter
        if name is None and value is not None:
            # A value that is a declared parameter's value is that parameter.
            for declared_name, values in declared.items():
                if values and str(values[0]) == value and len(value) > 1:
                    name = declared_name
                    break
        if name is not None:
            if not _PARAMETER_NAME_RE.match(name):
                problems.append(f"{where}: {name!r} is not a snake_case name")
                continue
            if action.action_name == "click":
                labels = _labels(action)
                value = _clean(step.label) if step.label else None
                if value is None and len(labels) == 1:
                    value = labels[0]
                if value not in labels:
                    problems.append(
                        f"{where}: label {step.label!r} is not one of {labels}"
                    )
                    continue
            if value is None:
                problems.append(f"{where}: a {action.action_name} carries no value")
                continue
            known = declared.get(name) or ([added[name]] if name in added else None)
            if known and _normalized(str(known[0])) != _normalized(value):
                problems.append(
                    f"{where}: {name} is {known[0]!r} for this run, but this step "
                    f"used {value!r}"
                )
                continue
            if name not in declared:
                added.setdefault(name, value)
            placeholder = f"{{{name}[0]}}"
            if action.action_name == "click":
                quoted = f'"{value}"'

                def names_value(c, quoted=quoted, value=value) -> bool:
                    return c.verify_text == value and c.locator.count(quoted) == 1

                if candidate is None:
                    # The model may leave a choice to the LLM when a verified locator
                    # naming exactly this value exists; that costs a call on every
                    # run for nothing, so take the locator.
                    candidate = next((c for c in verified if names_value(c)), None)
                    if candidate is not None:
                        report.upgraded.append(node_index)
                if candidate is None:
                    command = None
                    report.llm_each_run.append(node_index)
                elif not names_value(candidate):
                    problems.append(
                        f"{where}: locator {step.locator} does not name {value!r}; "
                        "choose the locator whose quoted name is the label, or null"
                    )
                    continue
                else:
                    templated_locator = candidate.locator.replace(
                        quoted, f'"{placeholder}"'
                    )
                    command = f"{templated_locator}.first"
                    templated[node_index] = name
            literal, value = value, placeholder
        elif action.action_name == "click" and candidate is None:
            report.llm_each_run.append(node_index)

        instructions = (step.prompt_instructions or "").strip()
        if not instructions or len(instructions) > _MAX_INSTRUCTION_LEN:
            problems.append(
                f"{where}: prompt_instructions empty or over 200 characters"
            )
            continue
        if name is not None:
            placeholder = f"{{{name}[0]}}"
            # Name the parameter, not this run's value, so a fallback looks for the
            # caller's value.
            instructions = re.sub(
                rf"'?(?<![\w{{]){re.escape(literal)}(?![\w}}])'?",
                placeholder,
                instructions,
                flags=re.IGNORECASE,
            )
            if placeholder not in instructions:
                instructions = f"{instructions} (the value is {placeholder})"

        interaction = _interaction(action, command, instructions, value)
        if interaction is None:
            problems.append(
                f"{where}: Optexity cannot replay this {action.action_name}; drop it"
            )
            continue
        nodes.append(ActionNode(type="action_node", interaction_action=interaction))
        report.sources.append(action)

    parameters = {**declared, **{name: [value] for name, value in added.items()}}
    for node_index, node in enumerate(nodes):
        step = _step_of(node)
        if step is None:
            continue
        for text in (step.prompt_instructions, step.command):
            if text and _resolve(text, parameters) is None:
                problems.append(
                    f"step {node_index}: references an undeclared parameter in {text!r}"
                )
    # A value still sitting in a locator nothing templated ties that step to this run.
    for name, value in {**added}.items():
        needle = value.casefold()
        for node_index, node in enumerate(nodes):
            step = _step_of(node)
            command = getattr(step, "command", None)
            if (
                command
                and needle in command.casefold()
                and templated.get(node_index) != name
            ):
                problems.append(
                    f"parameter {name}: its value {value!r} is part of step "
                    f"{node_index}'s locator {command}; give that step locator null "
                    "and a parameter of its own, or drop this parameter"
                )
    if problems:
        raise PlanRejected(problems)

    report.dropped = sorted(dropped.items())
    report.parameters_added = added
    report.suggestions = [s.strip() for s in plan.suggestions if s.strip()]
    return PlanResult(
        nodes=nodes,
        parameters={name: [value] for name, value in added.items()},
        report=report,
    )


def _step_of(node: ActionNode):
    interaction = node.interaction_action
    if interaction is None:
        return None
    for name in ("input_text", "click_element", "select_option", "key_press"):
        if (step := getattr(interaction, name, None)) is not None:
            return step
    return None


def _interaction(
    action: TracedAction, command: str | None, instructions: str, value: str | None
) -> InteractionAction | None:
    name = action.action_name
    if name == "input":
        return InteractionAction(
            input_text=InputTextAction(
                command=command,
                prompt_instructions=instructions,
                input_text=value or "",
            )
        )
    if name == "click":
        return InteractionAction(
            click_element=ClickElementAction(
                command=command, prompt_instructions=instructions
            )
        )
    if name == "select_dropdown":
        return InteractionAction(
            select_option=SelectOptionAction(
                command=command,
                prompt_instructions=instructions,
                select_values=[value or ""],
            )
        )
    if name == "send_keys":
        key_press = key_press_for(str(action.params.get("keys", "")), command)
        return InteractionAction(key_press=key_press) if key_press else None
    if name == "navigate":
        url = action.params.get("url") or action.url_after
        return InteractionAction(go_to_url=GoToUrlAction(url=str(url))) if url else None
    return None


def plan_automation(
    actions: list[TracedAction],
    objective: str | None,
    input_parameters: dict,
    agent: AutomationPlannerAgent,
) -> PlanResult | None:
    """Ask for a plan, check it, and on rejection ask again with the reasons.

    Returns None when no attempt holds up; the caller then compiles by rules."""
    described = describe_actions(actions)
    report_rejections: list[list[str]] = []
    usage = TokenUsage()
    problems: list[str] | None = None
    previous: AutomationPlan | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            _, plan, token_usage = agent.plan(
                described, input_parameters, objective, problems, previous
            )
        except Exception as error:
            logger.warning(f"planner: model call failed: {error}")
            return None
        usage += token_usage
        try:
            result = build_from_plan(plan, actions, input_parameters)
        except PlanRejected as rejection:
            problems, previous = rejection.problems, plan
            report_rejections.append(rejection.problems)
            logger.info(
                f"planner: attempt {attempt} rejected: {'; '.join(rejection.problems)}"
            )
            continue
        result.report.attempts = attempt
        result.report.rejections = report_rejections
        result.report.token_usage = usage
        return result
    logger.warning(f"planner: no plan held up after {_MAX_ATTEMPTS} attempts")
    return None


def planner_for(
    model,
) -> Callable[[list[TracedAction], str | None, dict], PlanResult | None]:
    """A planning function bound to a model, in the shape `learn_automation` takes."""
    agent = AutomationPlannerAgent(model)

    def plan(actions, objective, input_parameters):
        return plan_automation(actions, objective, input_parameters, agent)

    return plan
