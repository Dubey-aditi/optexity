import argparse
import logging
import os
import subprocess
import sys

from dotenv import load_dotenv
from uvicorn import run

logger = logging.getLogger(__name__)

env_path = os.getenv("ENV_PATH")
if not env_path:
    logger.warning("ENV_PATH is not set, using default values")
else:
    load_dotenv(env_path)


def install_browsers() -> None:
    """Install Playwright + Patchright browsers."""
    try:
        subprocess.run(
            ["playwright", "install", "--with-deps", "chromium", "chrome"],
            check=True,
        )
        subprocess.run(
            ["patchright", "install", "chromium", "chrome"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install browsers", file=sys.stderr)
        sys.exit(e.returncode)


def compile_cache(args: argparse.Namespace) -> None:
    """Compile a recorded action trace into a deterministic automation.

    With --llm an LLM plans it from the trace, every locator still taken from the
    recording; if no plan holds up, the rules compile it and an LLM pass refines it.
    """
    import pathlib

    from optexity.cache.compiler import (
        _parse_input_parameters,
        compile_file,
        load_trace,
    )
    from optexity.schema.automation import Automation, Parameters

    trace = pathlib.Path(args.trace).expanduser()
    out = pathlib.Path(args.out).expanduser()
    input_parameters = _parse_input_parameters(args.input_parameters)

    if args.llm:
        from optexity.cache.planner import planner_for
        from optexity.inference.models import get_llm_model_with_fallback

        actions = load_trace(trace)
        plan = planner_for(get_llm_model_with_fallback(None, None, True))
        result = plan(actions, args.objective, input_parameters)
        if result is not None:
            automation = Automation(
                url=args.url
                or next((a.url_before for a in actions if a.url_before), ""),
                parameters=Parameters(
                    input_parameters={**input_parameters, **result.parameters},
                    generated_parameters={},
                ),
                nodes=result.nodes,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                automation.model_dump_json(indent=2, exclude_none=True) + "\n"
            )
            print(result.report.render())
            print(f"\nwrote {args.out}")
            return
        print("no plan held up; compiling by rules instead\n")

    report = compile_file(
        trace_path=trace,
        output_path=out,
        url=args.url,
        input_parameters=input_parameters,
    )
    print(report.render())
    if args.llm:
        refined = _refine(
            Automation.model_validate_json(out.read_text()),
            args.objective,
            None,
            dict(enumerate(report.sources)),
        )
        out.write_text(refined.model_dump_json(indent=2, exclude_none=True) + "\n")
    print(f"\nwrote {args.out}")


def _refine(
    automation,
    objective: str | None,
    positions: list[int] | None,
    evidence: dict | None = None,
):
    """One LLM refinement pass, on the model configured by LLM_MODEL."""
    from optexity.cache.refine import refine_automation
    from optexity.inference.agents.automation_refiner.automation_refiner import (
        AutomationRefinerAgent,
    )
    from optexity.inference.models import get_llm_model_with_fallback

    agent = AutomationRefinerAgent(get_llm_model_with_fallback(None, None, True))
    refined, report = refine_automation(
        automation, agent, objective, positions, evidence
    )
    print(report.render())
    return refined


def learn(args: argparse.Namespace) -> None:
    """Apply one learning pass to an automation from a node-tagged trace."""
    import pathlib

    from optexity.cache.compiler import (
        _parse_input_parameters,
        learn_automation,
        load_trace,
    )
    from optexity.schema.automation import Automation

    automation = Automation.model_validate_json(
        pathlib.Path(args.automation).expanduser().read_text()
    )
    planner = None
    if args.llm:
        from optexity.cache.planner import planner_for
        from optexity.inference.models import get_llm_model_with_fallback

        planner = planner_for(get_llm_model_with_fallback(None, None, True))
    learned, report = learn_automation(
        automation,
        load_trace(pathlib.Path(args.trace).expanduser()),
        input_parameters=_parse_input_parameters(args.input_parameters),
        planner=planner,
    )
    print(report.render())
    if args.llm and report.rule_compiled_positions:
        learned = _refine(
            learned,
            "\n".join(report.objectives),
            report.rule_compiled_positions,
            report.evidence,
        )
    if report.changed:
        out = pathlib.Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(learned.model_dump_json(indent=2, exclude_none=True) + "\n")
        print(f"\nwrote {out}")


def run_inference(args: argparse.Namespace) -> None:
    from optexity.inference.child_process import get_app_with_endpoints

    app = get_app_with_endpoints(
        is_aws=args.is_aws, child_id=args.child_process_id, port=args.port
    )
    run(
        app,
        host=args.host,
        port=args.port,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="optexity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------------------------
    # install-browsers
    # ---------------------------
    install_cmd = subparsers.add_parser(
        "install_browsers",
        help="Install required browsers for Optexity",
        aliases=["install-browsers"],
    )
    install_cmd.set_defaults(func=lambda _: install_browsers())

    # ---------------------------
    # inference
    # ---------------------------
    inference_cmd = subparsers.add_parser(
        "inference", help="Run Optexity inference server"
    )
    inference_cmd.add_argument("--host", default="0.0.0.0")
    inference_cmd.add_argument("--port", type=int, default=9000)
    inference_cmd.add_argument(
        "--child_process_id", "--child-process-id", type=int, default=0
    )
    inference_cmd.add_argument(
        "--is_aws", "--is-aws", action="store_true", default=False
    )

    inference_cmd.set_defaults(func=run_inference)

    # ---------------------------
    # compile-cache
    # ---------------------------
    compile_cmd = subparsers.add_parser(
        "compile_cache",
        help="Compile a browser-use action trace into a deterministic automation",
        aliases=["compile-cache"],
    )
    compile_cmd.add_argument(
        "--trace", required=True, help="JSONL action trace to read"
    )
    compile_cmd.add_argument("--out", required=True, help="automation JSON to write")
    compile_cmd.add_argument(
        "--url",
        default=None,
        help="start url (default: the page the first recorded action ran on)",
    )
    compile_cmd.add_argument(
        "--input-parameters",
        "--input_parameters",
        default=None,
        dest="input_parameters",
        help='JSON object of input parameters, e.g. \'{"stock_ticker": ["NVDA"]}\'',
    )
    compile_cmd.add_argument(
        "--llm",
        action="store_true",
        help="have an LLM plan the automation from the trace (locators still come "
        "only from the recording; every plan is validated), falling back to rules",
    )
    compile_cmd.add_argument(
        "--objective",
        default=None,
        help="the task the agent was given, as context for --llm",
    )
    compile_cmd.set_defaults(func=compile_cache)

    # ---------------------------
    # learn
    # ---------------------------
    learn_cmd = subparsers.add_parser(
        "learn",
        help="Rewrite the nodes of an automation that needed an LLM, from a run's trace",
    )
    learn_cmd.add_argument(
        "--automation", required=True, help="automation JSON that was run"
    )
    learn_cmd.add_argument(
        "--trace", required=True, help="that run's node-tagged action trace"
    )
    learn_cmd.add_argument("--out", required=True, help="learned automation to write")
    learn_cmd.add_argument(
        "--input-parameters",
        "--input_parameters",
        default=None,
        dest="input_parameters",
        help="JSON object of the run's input parameters, used to template typed values",
    )
    learn_cmd.add_argument(
        "--llm",
        action="store_true",
        help="have an LLM plan the automation from the trace (locators still come "
        "only from the recording; every plan is validated), falling back to rules",
    )
    learn_cmd.set_defaults(func=learn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
