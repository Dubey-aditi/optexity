"""Compile a browser-use action trace into a deterministic Automation.

A task the agent solved once with an LLM becomes a node list that replays without
one. See `optexity compile-cache`.
"""

from optexity.cache.compiler import (
    CompileReport,
    LearnReport,
    compile_file,
    compile_trace,
    learn_automation,
    load_trace,
)

__all__ = [
    "CompileReport",
    "LearnReport",
    "compile_file",
    "compile_trace",
    "learn_automation",
    "load_trace",
]
