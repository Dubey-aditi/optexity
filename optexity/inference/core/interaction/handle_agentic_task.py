import logging

from browser_use import Agent, BrowserSession, Tools

from optexity.inference.infra.browser import Browser
from optexity.inference.models import normalize_model
from optexity.inference.models.chat_litellm import build_agent_llm
from optexity.schema.actions.interaction_action import (
    AgenticTask,
    CloseOverlayPopupAction,
)
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_agentic_task(
    agentic_task_action: AgenticTask | CloseOverlayPopupAction,
    task: Task,
    memory: Memory,
    browser: Browser,
):

    if agentic_task_action.backend == "browser_use":
        if isinstance(agentic_task_action, CloseOverlayPopupAction):
            tools = Tools(
                exclude_actions=[
                    "search",
                    "navigate",
                    "go_back",
                    "upload_file",
                    "scroll",
                    "find_text",
                    "send_keys",
                    "evaluate",
                    "switch",
                    "close",
                    "extract",
                    "dropdown_options",
                    "select_dropdown",
                    "write_file",
                    "read_file",
                    "replace_file",
                ]
            )
        else:
            # browser-use drops its screenshot tool when vision is off, but only for
            # the Tools it builds itself. Passing our own skipped that, so a model
            # without vision could call screenshot, never receive the image, and ask
            # again until max_steps ran out.
            tools = Tools(
                exclude_actions=(
                    [] if agentic_task_action.use_vision else ["screenshot"]
                )
            )
        llm = build_agent_llm(normalize_model(task.llm_provider, task.llm_model_name))
        browser_session = BrowserSession(
            cdp_url=browser.cdp_url, keep_alive=agentic_task_action.keep_alive
        )

        step_directory = (
            task.logs_directory / f"step_{str(memory.automation_state.step_index)}"
        )
        step_directory.mkdir(parents=True, exist_ok=True)

        agent = Agent(
            task=agentic_task_action.task,
            llm=llm,
            browser_session=browser_session,
            use_vision=agentic_task_action.use_vision,
            tools=tools,
            calculate_cost=True,
            save_conversation_path=step_directory,
            # The judge is an extra LLM call that grades the finished run. Success is
            # taken from the agent's own done(success=...), so it adds cost and
            # nothing else — and it fails outright on Anthropic models, which reject
            # the JPEG screenshots it sends labelled as PNG.
            use_judge=False,
        )
        logger.debug(f"Starting browser session for agentic task {browser.cdp_url} ")
        await agent.browser_session.start()
        logger.debug(f"Finally running agentic task on browser_use {browser.cdp_url} ")
        history = await agent.run(max_steps=agentic_task_action.max_steps)
        logger.debug(f"Agentic task completed on browser_use {browser.cdp_url} ")

        agent.stop()
        if agent.browser_session:
            await agent.browser_session.stop()
            await agent.browser_session.reset()

        return history

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")

    return None
