"""The learning layer: remember what the LLM worked out, so the next run does not.

The dashboard is the memory. A run learns, and the learned automation is written back
to the recording it came from, so the next call — from anywhere — is served the
improved version. Nothing is kept on disk: actions are recorded in the worker's
memory and learned from before it exits.

Configured by the environment, read in the worker:

* `OPTEXITY_LEARN=1` — an `agentic_task` the agent finished is replaced by the
  deterministic nodes compiled from what it did.
* `OPTEXITY_SELF_HEAL=1` — a step whose command failed and that the LLM fallback
  rescued gets the verified locator of the element the LLM chose.
* `OPTEXITY_LEARN_REFINE=1` — an LLM plans the automation from the recorded run:
  which steps to keep, which values become inputs, how each step is described
  (see `planner.py`). Every locator still comes from the recording, verified. If no
  plan holds up, the rules compile it and an LLM pass refines that (`refine.py`).

Only a run that succeeded is learned from, and only a change is synced. Each pass is
logged in full, so the task log on the dashboard shows what was learned and why.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from browser_use.cache import TRACE_MEMORY_ENV, recorded_actions

from optexity.cache.compiler import learn_automation
from optexity.cache.planner import planner_for
from optexity.cache.refine import refine_automation
from optexity.inference.agents.automation_refiner.automation_refiner import (
    AutomationRefinerAgent,
)
from optexity.inference.models import get_llm_model_with_fallback
from optexity.schema.automation import Automation
from optexity.schema.task import Task
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)

LEARN_ENV = "OPTEXITY_LEARN"
SELF_HEAL_ENV = "OPTEXITY_SELF_HEAL"
REFINE_ENV = "OPTEXITY_LEARN_REFINE"
UPDATE_RECORDING_ENDPOINT = "api/v1/update_recording/{recording_id}"


def _flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class LearningSettings:
    learn: bool
    heal: bool
    refine: bool

    @classmethod
    def from_env(cls) -> "LearningSettings":
        return cls(
            learn=_flag(LEARN_ENV), heal=_flag(SELF_HEAL_ENV), refine=_flag(REFINE_ENV)
        )

    @property
    def enabled(self) -> bool:
        return self.learn or self.heal

    def start_recording(self) -> None:
        """Have browser-use record every action in this process. Must run before the
        browser starts: `Tools` decides whether to record when it is built."""
        if self.enabled:
            os.environ[TRACE_MEMORY_ENV] = "1"


async def learn_from_run(task: Task, source: Automation, learning: LearningSettings):
    """Fold this run's recorded actions into `source` and sync any change.

    `source` must be the automation as it was before the run: running substitutes
    input values into the nodes in place, and the learned automation must keep the
    placeholders, not this run's values.
    """
    # Only an agentic task is planned or refined; a pure replay or a heal needs no
    # model, so do not build one (its creation is logged and reads like an LLM call).
    has_agentic_task = any(
        getattr(getattr(node, "interaction_action", None), "agentic_task", None)
        for node in source.nodes
    )
    model = (
        get_llm_model_with_fallback(task.llm_provider, task.llm_model_name, True)
        if learning.refine and has_agentic_task
        else None
    )
    # Planning calls the model synchronously; keep it off the event loop.
    learned, report = await asyncio.to_thread(
        learn_automation,
        source,
        recorded_actions(),
        input_parameters=task.input_parameters,
        compile_agentic=learning.learn,
        heal=learning.heal,
        planner=planner_for(model) if model else None,
    )
    rendered = report.render()

    # Nodes the planner could not handle were compiled by rules; refine those.
    if model is not None and report.rule_compiled_positions:
        agent = AutomationRefinerAgent(model)
        # The model client is synchronous; keep it off the event loop.
        learned, refine_report = await asyncio.to_thread(
            refine_automation,
            learned,
            agent,
            "\n".join(report.objectives),
            report.rule_compiled_positions,
            report.evidence,
        )
        rendered += "\n" + refine_report.render()

    # Planned or refined, new inputs change what the endpoint accepts once synced.
    declared = set(source.parameters.input_parameters)
    learned_inputs = learned.parameters.input_parameters
    if added := [name for name in learned_inputs if name not in declared]:
        rendered += (
            f"\nthe endpoint now requires inputs {', '.join(added)}; call it with "
            f"input_parameters {json.dumps({**task.input_parameters, **{n: learned_inputs[n] for n in added}})}"
        )

    logger.info(f"learning: {task.endpoint_name}\n{rendered}")
    # Refinement only runs on nodes this pass compiled, so no compile, no change.
    if not report.changed:
        return

    if os.getenv("OPTEXITY_LOCAL_AUTOMATION"):
        # Learned from a local file, not from what the dashboard holds: show it
        # rather than overwrite the recording with something it never contained.
        logger.info(
            "learning: local automation in use, not syncing; learned automation:\n"
            + learned.model_dump_json(indent=2, exclude_none=True)
        )
        return
    await sync_to_dashboard(task, learned)


async def sync_to_dashboard(task: Task, automation: Automation) -> None:
    """Replace the recording's automation with `automation`."""
    url = urljoin(
        settings.SERVER_URL,
        UPDATE_RECORDING_ENDPOINT.format(recording_id=task.recording_id),
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.patch(
                url,
                json={
                    "automation": automation.model_dump(mode="json", exclude_none=True)
                },
                headers={"x-api-key": settings.OPTEXITY_API_KEY},
            )
            response.raise_for_status()
        logger.info(
            f"learning: synced to the dashboard, recording {task.recording_id} "
            f"({task.endpoint_name}) now has {len(automation.nodes)} nodes"
        )
    except Exception as error:
        logger.error(f"learning: sync to the dashboard failed: {error}")
