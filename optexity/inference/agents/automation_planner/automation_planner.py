import json
import logging

from pydantic import BaseModel, Field

from optexity.inference.agents.automation_planner.prompt import system_prompt
from optexity.inference.models.llm_model import LLMModel
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


class PlanStep(BaseModel):
    action: int
    locator: int | None = None
    parameter: str | None = None
    label: str | None = None
    prompt_instructions: str = ""


class DroppedAction(BaseModel):
    action: int
    reason: str = ""


class AutomationPlan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    dropped: list[DroppedAction] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class AutomationPlannerAgent:
    """Plans a deterministic automation from a recorded agent run. Plans only: every
    locator it chooses is an index into what the recorder verified, and the caller
    rejects any plan that does not hold up."""

    def __init__(self, model: LLMModel):
        self.model = model

    def plan(
        self,
        actions: list[dict],
        input_parameters: dict,
        objective: str | None,
        problems: list[str] | None = None,
        previous: AutomationPlan | None = None,
    ) -> tuple[str, AutomationPlan, TokenUsage]:
        feedback = ""
        if problems and previous is not None:
            feedback = f"""
        YOUR PREVIOUS PLAN:
        {previous.model_dump_json()}

        IT WAS REJECTED FOR THESE REASONS:
        {json.dumps(problems, indent=2)}
        """

        final_prompt = f"""
        [INPUT]
        OBJECTIVE:
        {objective or "(not recorded)"}

        DECLARED PARAMETERS (name -> this run's values):
        {json.dumps(input_parameters, ensure_ascii=False)}

        ACTIONS:
        {json.dumps(actions, indent=1, ensure_ascii=False)}
        {feedback}
        [/INPUT]
        """

        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt=final_prompt,
            response_schema=AutomationPlan,
            system_instruction=system_prompt,
        )
        return final_prompt, response, token_usage
