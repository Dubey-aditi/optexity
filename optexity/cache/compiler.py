"""Turn a browser-use action trace into a deterministic Automation.

The agent solved the task once, paying for an LLM call per step. Everything it
decided is in the trace; this rewrites those decisions from DOM indices — which are
positions in one run's page listing and mean nothing on the next — into Playwright
locators, which do not move.

Deliberately rule-based. The LLM already made every decision here; this only
translates an unstable address into a stable one, which is a translation problem,
not a judgement one. An LLM in this path would also make the compiler
non-deterministic about determinism: compile twice, get two different automations.

Steps whose locator could not be verified unique are not dropped. They are emitted
with `prompt_instructions` and no `command`, which is Optexity's own LLM-fallback
shape — so an automation degrades to "agentic for this one step" rather than to
"silently acts on the wrong element".
"""

import json
import logging
import re
from pathlib import Path

from browser_use.cache.views import TracedAction

from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoToUrlAction,
    InputTextAction,
    InteractionAction,
    KeyPressAction,
    SelectOptionAction,
    UploadFileAction,
)
from optexity.schema.automation import ActionNode, Automation, Parameters

logger = logging.getLogger(__name__)

# Actions that change nothing in the browser, or that the replay path already handles.
# `scroll` is here because command_based_action_with_retry calls
# scroll_into_view_if_needed before acting, so a recorded scroll is redundant.
NOISE_ACTIONS = frozenset(
    {
        "done",
        "wait",
        "screenshot",
        "scroll",
        "extract",
        "read_file",
        "write_file",
        "replace_file",
        "find_text",
        "dropdown_options",
    }
)


class CompileReport:
    """What the compiler did, so a human can see what was dropped and why."""

    def __init__(self) -> None:
        self.total = 0
        self.dropped: list[tuple[int, str, str]] = []  # (step, action, reason)
        self.deterministic = 0
        self.agentic_fallback: list[tuple[int, str]] = []  # (step, reason)
        # The recorded action behind each emitted node, in node order: its element
        # and every verified locator, which a later pass may need to re-template.
        self.sources: list[TracedAction] = []
        self.planned = False

    def drop(self, action: TracedAction, reason: str) -> None:
        self.dropped.append((action.step, action.action_name, reason))

    def render(self) -> str:
        lines = [
            f"{self.total} recorded actions -> {self.deterministic + len(self.agentic_fallback)} nodes",
            f"  {self.deterministic} deterministic, {len(self.agentic_fallback)} needing an LLM fallback",
            f"  {len(self.dropped)} dropped",
        ]
        for step, action, reason in self.dropped:
            lines.append(f"    - step {step} {action}: {reason}")
        for step, reason in self.agentic_fallback:
            lines.append(f"    ! step {step}: {reason}")
        return "\n".join(lines)


def load_trace(path: Path) -> list[TracedAction]:
    """Read a JSONL trace. A malformed line is skipped rather than failing the run —
    a trace is a log, and one bad line should not cost the other nineteen."""
    actions: list[TracedAction] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            actions.append(TracedAction.model_validate_json(line))
        except ValueError as error:
            logger.warning(
                f"{path.name}:{number} skipped, not a valid traced action: {error}"
            )
    return actions


def _prune(
    actions: list[TracedAction], start_url: str | None, report: CompileReport
) -> list[TracedAction]:
    """Drop everything that does not need to happen again.

    Runs in reverse for the collapse rule: when the agent typed into the same field
    twice, the *last* value is the one that survived, so keeping the last occurrence
    of each element is what reproduces the end state.
    """
    report.total = len(actions)
    kept: list[TracedAction] = []
    seen_input_elements: set[tuple[str | None, str | None, str]] = set()

    for action in reversed(actions):
        if action.action_name in NOISE_ACTIONS:
            report.drop(action, "action does not mutate the page")
            continue

        if action.error:
            report.drop(action, f"failed during exploration: {action.error[:60]}")
            continue

        # Optexity's input_text clicks the field before typing (click_before_input),
        # so an agent's click straight before typing into the same field is replayed
        # twice. `kept[-1]` is the next surviving action: the loop runs in reverse.
        if (
            action.action_name == "click"
            and kept
            and kept[-1].action_name == "input"
            and _same_element(action, kept[-1])
        ):
            report.drop(action, "the following input clicks the same field")
            continue

        # browser-use refuses to click a <select> or a file input and answers with
        # the dropdown options / an upload hint instead, without setting `error`.
        # Nothing on the page changed, so there is nothing to replay.
        if action.action_name == "click" and _is_unclickable(action):
            report.drop(action, "browser-use does not click this element type")
            continue

        # The automation's `url` already opens the start page; navigating to it again
        # costs a page load and can undo work done before it.
        if action.action_name in ("navigate", "search") and start_url:
            if str(action.params.get("url", "")).rstrip("/") == start_url.rstrip("/"):
                report.drop(action, "navigates to the automation's own start url")
                continue

        # Two inputs are the same field only if they would replay to the same command
        # *and* carry the same element identity. Keying on identity alone would be
        # enough if the hash were always right, but this rule deletes steps silently,
        # so it requires two independent signals to agree before doing so.
        if action.action_name == "input" and action.element:
            identity = (
                action.element.element_hash,
                _locator_for(action),
                action.element.xpath,
            )
            if None in identity[:2] and not action.element.xpath:
                kept.append(action)
                continue
            if identity in seen_input_elements:
                report.drop(action, "overwritten by a later input on the same element")
                continue
            seen_input_elements.add(identity)

        kept.append(action)

    kept.reverse()
    return kept


def _locator_for(action: TracedAction) -> str | None:
    """The command to replay this action with, or None when nothing is trustworthy.

    Only a candidate verified to match exactly one element is used. A locator that
    matches several is worse than none: it turns an ambiguous match into a confident
    action on whichever element happens to be first.
    """
    best = action.best_verified
    if best is None:
        return None
    # Optexity evaluates `page.<command>`, so the command is the locator expression
    # without the `page.` prefix. `.first` is safe here precisely because uniqueness
    # was confirmed — it guards against a page that gains a duplicate later, rather
    # than papering over one that already has duplicates.
    return f"{best.locator}.first"


def _same_element(first: TracedAction, second: TracedAction) -> bool:
    if first.element is None or second.element is None:
        return False
    return (
        first.element.element_hash is not None
        and first.element.element_hash == second.element.element_hash
    )


# browser-use names keys the way Playwright does ("Enter", "Control+a"); Optexity's
# KeyPressAction takes its own lowercase names ("enter", ["ctrl", "a"]).
_KEY_ALIASES = {
    "control": "ctrl",
    "meta": "command",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "escape": "esc",
}


def key_press_for(keys: str, command: str | None = None) -> KeyPressAction | None:
    """The Optexity key press for a browser-use `send_keys`, or None if Optexity has
    no name for one of its keys."""
    names = [
        _KEY_ALIASES.get(part.lower(), part.lower()) for part in keys.split("+") if part
    ]
    if not names:
        return None
    try:
        return KeyPressAction(
            type=names[0] if len(names) == 1 else names, command=command
        )
    except ValueError:
        return None


def _is_unclickable(action: TracedAction) -> bool:
    element = action.element
    if element is None:
        return False
    if element.tag_name == "select":
        return True
    return (
        element.tag_name == "input"
        and element.attributes.get("type", "").lower() == "file"
    )


_FORM_TAGS = frozenset({"input", "select", "textarea"})
_LABELLED_INPUT_TYPES = frozenset({"checkbox", "radio", "button", "submit"})


def _clean_label(value: str | None) -> str | None:
    """A short, single-line label, or None when the value is not usable as one.

    A <select>'s text is every option joined by newlines ("1900\\n1901\\n..."), which
    reads as noise in an instruction, so multi-line values are rejected outright.
    """
    if not value:
        return None
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines or len(lines) > 2:
        return None
    label = " ".join(" ".join(lines).split())
    return label if len(label) <= 60 else label[:57] + "..."


def _humanize(identifier: str | None) -> str | None:
    """`dateOfBirthInput` -> "date of birth input"; `react-datepicker__year-select`
    -> "year select" (the BEM element, not the block)."""
    if not identifier:
        return None
    tail = identifier.split()[0].split("__")[-1]
    words = (
        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", tail).replace("-", " ").replace("_", " ")
    )
    return _clean_label(words.lower())


def _label(action: TracedAction) -> str:
    element = action.element
    if element is None:
        return "target"
    attributes = element.attributes
    ordered = [
        attributes.get("aria-label"),
        element.accessible_name,
        attributes.get("placeholder"),
        attributes.get("name"),
        element.text,
    ]
    # A text field's accessible name is often its current *value* ("01 Oct 2026"),
    # which changes between runs, so attributes describing the field come first.
    # Checkboxes and radios are the exception: their name is the visible label.
    field_type = attributes.get("type", "").lower()
    if element.tag_name in _FORM_TAGS and field_type not in _LABELLED_INPUT_TYPES:
        ordered = [
            attributes.get("aria-label"),
            attributes.get("placeholder"),
            attributes.get("name"),
            _humanize(attributes.get("id")),
            element.accessible_name,
            element.text,
        ]
    for value in ordered:
        label = _clean_label(value)
        if label:
            return label
    return (
        _humanize(attributes.get("id"))
        or _humanize(attributes.get("class"))
        or element.tag_name
    )


def _describe(action: TracedAction, value: str = "") -> str:
    """A human-readable instruction, used as the LLM fallback when the command fails."""
    label = _label(action)
    if action.action_name == "input":
        return f"Enter {value!r} into the {label} field"
    if action.action_name == "click":
        return f"Click the {label} element"
    if action.action_name == "select_dropdown":
        return f"Select {value!r} in the {label} dropdown"
    if action.action_name == "upload_file":
        return f"Upload a file using the {label} input"
    return f"Perform {action.action_name} on {label}"


def _parameterize(value: str, input_parameters: dict | None) -> str:
    """Replace a literal the agent typed with the input parameter it came from.

    Exact matches only, and never for one-character values: a placeholder that
    captures the wrong literal replays a different value than the agent chose.
    """
    if not input_parameters or len(value) < 2:
        return value
    for key, values in input_parameters.items():
        for position, candidate in enumerate(values or []):
            if str(candidate) == value:
                return f"{{{key}[{position}]}}"
    return value


def _to_interaction(
    action: TracedAction,
    report: CompileReport,
    input_parameters: dict | None = None,
) -> InteractionAction | None:
    """Map one traced action onto an Optexity interaction, or None if it has no
    equivalent worth replaying."""
    command = _locator_for(action)
    value = _parameterize(str(action.params.get("text", "")), input_parameters)
    instructions = _describe(action, value)

    if command is None and action.action_name in (
        "input",
        "click",
        "select_dropdown",
        "upload_file",
    ):
        counts = (
            ", ".join(f"{c.kind}={c.match_count}" for c in action.candidates)
            or "no candidates"
        )
        report.agentic_fallback.append(
            (action.step, f"no uniquely-matching locator ({counts})")
        )

    if action.action_name == "input":
        return InteractionAction(
            input_text=InputTextAction(
                command=command,
                prompt_instructions=instructions,
                input_text=value,
            )
        )

    if action.action_name == "click":
        return InteractionAction(
            click_element=ClickElementAction(
                command=command, prompt_instructions=instructions
            )
        )

    if action.action_name == "select_dropdown":
        return InteractionAction(
            select_option=SelectOptionAction(
                command=command,
                prompt_instructions=instructions,
                select_values=[value],
            )
        )

    if action.action_name == "upload_file":
        return InteractionAction(
            upload_file=UploadFileAction(file_path=str(action.params.get("path", "")))
        )

    if action.action_name in ("navigate", "search"):
        url = action.params.get("url") or action.url_after
        if not url:
            return None
        return InteractionAction(go_to_url=GoToUrlAction(url=str(url)))

    if action.action_name == "send_keys":
        key_press = key_press_for(str(action.params.get("keys", "")), command)
        if key_press is None:
            report.drop(
                action, f"Optexity has no key named {action.params.get('keys')!r}"
            )
            return None
        return InteractionAction(key_press=key_press)

    report.drop(action, f"no Optexity equivalent for '{action.action_name}'")
    return None


def compile_trace(
    actions: list[TracedAction],
    url: str | None = None,
    input_parameters: dict | None = None,
) -> tuple[Automation, CompileReport]:
    """Build an Automation from a trace.

    `url` defaults to the page the first action happened on, which is where a replay
    has to start for the locators to resolve.
    """
    report = CompileReport()
    start_url = url or next((a.url_before for a in actions if a.url_before), None)

    nodes: list[ActionNode] = []
    for action in _prune(actions, start_url, report):
        interaction = _to_interaction(action, report, input_parameters)
        if interaction is None:
            continue
        nodes.append(ActionNode(type="action_node", interaction_action=interaction))
        report.sources.append(action)
        if _locator_for(action) is not None:
            report.deterministic += 1

    automation = Automation(
        url=start_url or "",
        parameters=Parameters(
            input_parameters=input_parameters or {},
            generated_parameters={},
        ),
        nodes=nodes,
    )
    return automation, report


class LearnReport:
    """What one learning pass changed, node by node."""

    def __init__(self) -> None:
        # (source node, first new node, new node count, objective, compile report)
        self.replaced: list[tuple[int, int, int, str, CompileReport]] = []
        self.patched: list[tuple[int, str | None, str]] = []  # (node, old, new command)
        self.skipped: list[tuple[int, str]] = []  # (node, reason)

    @property
    def changed(self) -> bool:
        return bool(self.replaced or self.patched)

    @property
    def compiled_positions(self) -> list[int]:
        """Where the nodes compiled this pass sit in the learned automation."""
        return [
            index
            for _, start, count, _, _ in self.replaced
            for index in range(start, start + count)
        ]

    @property
    def rule_compiled_positions(self) -> list[int]:
        """Positions compiled by rules rather than planned: the ones a refinement
        pass can still improve. A planned node already went through the model."""
        return [
            index
            for _, start, count, _, report in self.replaced
            if not getattr(report, "planned", False)
            for index in range(start, start + count)
        ]

    @property
    def evidence(self) -> dict[int, TracedAction]:
        """The recorded action behind each node compiled this pass, by position in
        the learned automation."""
        return {
            start + offset: action
            for _, start, _, _, report in self.replaced
            for offset, action in enumerate(report.sources)
        }

    @property
    def objectives(self) -> list[str]:
        """The agentic tasks that were replaced, in the words they were given."""
        return [objective for _, _, _, objective, _ in self.replaced]

    def render(self) -> str:
        if not (self.replaced or self.patched or self.skipped):
            return "nothing to learn: no node needed the LLM"
        lines = []
        for position, _, count, _, report in self.replaced:
            lines.append(f"node {position}: agentic_task -> {count} nodes")
            lines.extend(f"  {line}" for line in report.render().splitlines())
        for position, old, new in self.patched:
            lines.append(f"node {position}: command {old or '(none)'} -> {new}")
        for position, reason in self.skipped:
            lines.append(f"node {position}: kept as is, {reason}")
        return "\n".join(lines)


# Which recorded browser-use action can stand in for each Optexity interaction when
# the command failed and the LLM fallback acted instead.
_HEALABLE = {
    "input_text": frozenset({"input"}),
    "click_element": frozenset({"click"}),
    "select_option": frozenset({"select_dropdown"}),
    "check": frozenset({"click"}),
    "uncheck": frozenset({"click"}),
}


def _heal(
    interaction: InteractionAction, actions: list[TracedAction]
) -> tuple[InteractionAction | None, str | None, str]:
    """A copy of `interaction` with its command replaced by the locator the LLM
    fallback resolved, or None plus the reason it cannot be repaired."""
    field = next((f for f in _HEALABLE if getattr(interaction, f, None)), None)
    if field is None:
        return None, None, "not a repairable interaction"
    step = getattr(interaction, field)

    usable = [
        a
        for a in actions
        if a.action_name in _HEALABLE[field] and a.succeeded and a.element is not None
    ]
    if not usable:
        return None, step.command, "the LLM fallback did not act on an element"
    # The last successful attempt is the one that stuck.
    command = _locator_for(usable[-1])
    if command is None:
        return (
            None,
            step.command,
            "no uniquely-matching locator for the element the LLM chose",
        )
    if command == step.command:
        return None, step.command, "command already matches"

    healed = interaction.model_copy(deep=True)
    healed_step = getattr(healed, field)
    healed_step.command = command
    # A step carries exactly one of xpath / command.
    healed_step.xpath = None
    healed_step.skip_command = False
    return healed, step.command, command


def learn_automation(
    automation: Automation,
    actions: list[TracedAction],
    input_parameters: dict | None = None,
    compile_agentic: bool = True,
    heal: bool = True,
    planner=None,
) -> tuple[Automation, LearnReport]:
    """Rewrite the nodes that needed an LLM this run so the next run does not.

    Two cases, told apart by the node the action was recorded under:

    * an `agentic_task` the agent finished successfully is replaced by the
      deterministic nodes compiled from what it did;
    * any other interaction the LLM fallback had to rescue keeps its shape and gets
      the verified locator of the element the LLM chose.

    Every other node is left untouched, so a run that needed no LLM changes nothing.
    `compile_agentic` and `heal` switch the two cases off independently.

    `planner(actions, objective, input_parameters)`, when given, is tried first for
    an agentic task and may return a `PlanResult`; on None the rules compile it.
    """
    report = LearnReport()
    by_node: dict[int, list[TracedAction]] = {}
    for action in actions:
        position = action.context.get("node")
        if isinstance(position, int):
            by_node.setdefault(position, []).append(action)

    nodes = []
    added_parameters: dict[str, list] = {}
    for position, node in enumerate(automation.nodes):
        recorded = by_node.get(position)
        interaction = getattr(node, "interaction_action", None)
        if not recorded or not isinstance(node, ActionNode) or interaction is None:
            nodes.append(node)
            continue

        if interaction.agentic_task:
            if not compile_agentic:
                nodes.append(node)
                continue
            done = [a for a in recorded if a.action_name == "done"]
            if not done or done[-1].params.get("success") is not True:
                report.skipped.append((position, "the agent did not report success"))
                nodes.append(node)
                continue
            objective = interaction.agentic_task.task
            planned = (
                planner(recorded, objective, input_parameters or {})
                if planner
                else None
            )
            if planned is not None and planned.nodes:
                report.replaced.append(
                    (
                        position,
                        len(nodes),
                        len(planned.nodes),
                        objective,
                        planned.report,
                    )
                )
                nodes.extend(planned.nodes)
                added_parameters.update(planned.parameters)
                continue
            compiled, compile_report = compile_trace(
                recorded, url=automation.url, input_parameters=input_parameters
            )
            if not compiled.nodes:
                report.skipped.append(
                    (position, "the agent's actions compiled to nothing")
                )
                nodes.append(node)
                continue
            report.replaced.append(
                (
                    position,
                    len(nodes),
                    len(compiled.nodes),
                    interaction.agentic_task.task,
                    compile_report,
                )
            )
            nodes.extend(compiled.nodes)
            continue

        if not heal:
            nodes.append(node)
            continue
        healed, old, outcome = _heal(interaction, recorded)
        if healed is None:
            report.skipped.append((position, outcome))
            nodes.append(node)
            continue
        nodes.append(node.model_copy(update={"interaction_action": healed}))
        report.patched.append((position, old, outcome))

    parameters = automation.parameters.model_copy(deep=True)
    for name, values in added_parameters.items():
        parameters.input_parameters.setdefault(name, values)
    learned = automation.model_copy(update={"nodes": nodes, "parameters": parameters})
    # Round-trip through the schema: a learned automation that does not validate
    # must never replace one that does.
    learned = Automation.model_validate_json(learned.model_dump_json(exclude_none=True))
    return learned, report


def compile_file(
    trace_path: Path,
    output_path: Path,
    url: str | None = None,
    input_parameters: dict | None = None,
) -> CompileReport:
    """Compile a trace file to an automation file, validating before it is written."""
    actions = load_trace(trace_path)
    automation, report = compile_trace(
        actions, url=url, input_parameters=input_parameters
    )

    # Round-trip through the schema: an automation that does not validate is not a
    # deliverable, and failing here beats failing inside a browser run.
    payload = automation.model_dump_json(indent=2, exclude_none=True)
    Automation.model_validate_json(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n")
    return report


def _parse_input_parameters(raw: str | None) -> dict:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("input parameters must be a JSON object")
    return value
