import json
import logging

from pydantic import BaseModel, Field

from optexity.inference.agents.automation_refiner.prompt import system_prompt
from optexity.inference.models.llm_model import LLMModel
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


class NodeEdit(BaseModel):
    node: int
    prompt_instructions: str | None = None
    value: str | None = None


class NewParameter(BaseModel):
    name: str
    nodes: list[int]
    # For a click node: which of its labels the parameter stands for.
    value: str | None = None


class AutomationRefinement(BaseModel):
    new_parameters: list[NewParameter] = Field(default_factory=list)
    edits: list[NodeEdit] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class AutomationRefinerAgent:
    """Proposes readable instructions and parameter placeholders for a compiled
    automation. Proposes only: the caller validates every edit before applying it."""

    def __init__(self, model: LLMModel):
        self.model = model

    def refine(
        self,
        nodes: list[dict],
        input_parameters: dict,
        objective: str | None,
    ) -> tuple[str, AutomationRefinement, TokenUsage]:
        final_prompt = f"""
        [INPUT]
        OBJECTIVE THE AGENT WAS GIVEN:
        {objective or "(not recorded)"}

        DECLARED PARAMETERS (name -> this run's values):
        {json.dumps(input_parameters, ensure_ascii=False)}

        NODES:
        {json.dumps(nodes, indent=2, ensure_ascii=False)}
        [/INPUT]
        """

        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt=final_prompt,
            response_schema=AutomationRefinement,
            system_instruction=system_prompt,
        )
        return final_prompt, response, token_usage
