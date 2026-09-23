"""Clone-friendly command line entry points for the research toolkit."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage

from .comparison import compare_experiment, configured_prices
from .console import ConsoleRenderer
from .configuration import ConfigurationError, load_configuration
from .configured_systems import build_configured_systems
from .contracts import RunIdentity
from .dataset import DatasetError, load_configured_dataset
from .experiment import run_configured_experiment
from .grading import grade_case, require_grader
from .preview import inspect_configuration
from .registry import ComponentRegistry, register_builtin_components
from .tracing import trace_case_execution, trace_experiment


EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_INFRASTRUCTURE = 3
EXIT_INTERRUPTED = 130


class _ScriptedModel:
    """One fresh, deterministic response sequence for an offline demo role."""

    def __init__(self, responses: Sequence[AIMessage]) -> None:
        self._responses = iter(responses)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise RuntimeError("offline fixture response sequence exhausted") from exc


def _load_fixture(path: Path | None) -> tuple[Mapping[str, str] | None, Any]:
    if path is None:
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"offline fixture {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("environment"), dict) or not isinstance(data.get("roles"), dict):
        raise ValueError("offline fixture requires environment and roles objects")
    environment = data["environment"]
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
        raise ValueError("offline fixture environment must contain string names and values")
    roles = data["roles"]
    scripted: dict[str, tuple[AIMessage, ...]] = {}
    for role, rows in roles.items():
        if not isinstance(role, str) or not isinstance(rows, list) or not rows:
            raise ValueError("offline fixture roles must map names to nonempty response lists")
        messages: list[AIMessage] = []
        for row in rows:
            if not isinstance(row, dict) or set(row) - {"content", "tool_calls"}:
                raise ValueError(f"offline fixture role {role}: response has unknown fields")
            content = row.get("content", "")
            tool_calls = row.get("tool_calls", [])
            if not isinstance(content, str) or not isinstance(tool_calls, list):
                raise ValueError(f"offline fixture role {role}: invalid response")
            messages.append(AIMessage(content=content, tool_calls=tool_calls))
        scripted[role] = tuple(messages)

    def model_factory(model: Any, role: str) -> _ScriptedModel:
        if role not in scripted:
            raise ValueError(f"offline fixture lacks role {role!r}")
        return _ScriptedModel(scripted[role])

    return environment, model_factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mas-slm", description="Run and inspect registered SAS/MAS experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "inspect", "run", "compare"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path, help="Experiment YAML path")
        command.add_argument("--fixture", type=Path, help="Explicit scripted offline model fixture JSON")
        if name == "inspect":
            command.add_argument("--details", action="store_true", help="Include assembled prompts and tool schemas")
        if name == "run":
            command.add_argument("--system", choices=("single", "multi"), required=True)
            command.add_argument("--case", required=True, help="Case ID from the configured dataset")
        if name in {"run", "compare"}:
            command.add_argument("--console", choices=("none", "summary", "events", "full"),
                                 help="Human progress on stderr; overrides reporting.console")
            command.add_argument("--color", choices=("auto", "always", "never"),
                                 help="ANSI color policy; NO_COLOR always wins")
            trace = command.add_mutually_exclusive_group()
            trace.add_argument("--trace", action="store_true", help="Enable configured span recording/export")
            trace.add_argument("--no-trace", action="store_true", help="Disable configured span recording/export")
    summary = commands.add_parser("summarize")
    summary.add_argument("report", type=Path, help="JSON report captured from compare stdout")
    return parser


async def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "summarize":
        try:
            report = json.loads(args.report.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"report {args.report}: {exc}") from exc
        if not isinstance(report, dict) or not isinstance(report.get("systems"), dict) or not isinstance(report.get("pairs"), list):
            raise ValueError("report must be JSON emitted by compare")
        return {key: report[key] for key in (
            "experiment_id", "experiment_status", "systems", "single_wins", "multi_wins", "ties", "uncomparable_pairs"
        ) if key in report}

    fixture_env, model_factory = _load_fixture(args.fixture)
    if args.fixture is not None:
        # Scripted demonstrations must stay offline even when tiktoken's
        # encoding cache is empty (the general fix is tracked as KI-06).
        from .telemetry import token_estimator

        token_estimator.tiktoken = None
    registry = ComponentRegistry()
    register_builtin_components(registry)
    loaded = load_configuration(args.config, registry=registry, environment=fixture_env or os.environ)
    renderer = None
    if args.command in {"run", "compare"}:
        renderer = ConsoleRenderer(
            mode=args.console or loaded.experiment.reporting.console,
            color=args.color or loaded.experiment.reporting.color,
        )
        trace_config = loaded.experiment.telemetry.model_copy(update={
            "enabled": True if args.trace else False if args.no_trace else loaded.experiment.telemetry.enabled,
        })
    if args.command == "validate":
        dataset = load_configured_dataset(loaded)
        return {"valid": True, "experiment": loaded.experiment.name,
                "cases": len(dataset.cases), "case_ids": list(dataset.case_ids),
                "configuration": loaded.safe_snapshot()}
    if args.command == "inspect":
        return inspect_configuration(loaded).details if args.details else inspect_configuration(loaded).concise
    if args.command == "run":
        dataset = load_configured_dataset(loaded, case_ids=(args.case,))
        case = dataset.cases[0]
        systems = build_configured_systems(loaded, model_factory=model_factory)
        grader = require_grader(loaded.registry.resolve("graders", loaded.experiment.grader))
        identity = RunIdentity(experiment_id=str(uuid.uuid4()), system_id=args.system,
                               case_id=case.case_id, repetition=1, run_id=str(uuid.uuid4()))
        if args.system == "single":
            execution = await systems.sas_runner.run_case(identity=identity, payload=case.agent_input())
        else:
            execution = await systems.mas_runner.run_case(identity=identity, case_info=case.agent_input())
        grade = grade_case(grader, expected=case.expected_label(), result=execution.result)
        assert renderer is not None
        try:
            renderer.render_case(system_id=args.system, case_id=case.case_id,
                                 repetition=1, execution=execution, grade=grade)
        except Exception as exc:
            print(f"reporting warning: {type(exc).__name__}: {exc}", file=sys.stderr)
        trace = trace_case_execution(
            execution, grade=grade, config=trace_config,
            environment=fixture_env or os.environ,
        )
        if trace.enabled:
            print(f"tracing: {len(trace.spans)} spans captured", file=sys.stderr)
        for warning in trace.warnings:
            print(f"tracing warning: {warning}", file=sys.stderr)
        return {"result": execution.result.to_dict(), "grade": grade.to_dict()}
    if args.command == "compare":
        assert renderer is not None
        run = await run_configured_experiment(
            loaded, model_factory=model_factory, on_attempt=renderer.render_attempt,
        )
        for warning in run.reporting_errors:
            print(f"reporting warning: {warning}", file=sys.stderr)
        trace = trace_experiment(run, config=trace_config, environment=fixture_env or os.environ)
        if trace.enabled:
            print(f"tracing: {len(trace.spans)} spans captured", file=sys.stderr)
        for warning in trace.warnings:
            print(f"tracing warning: {warning}", file=sys.stderr)
        grader = require_grader(loaded.registry.resolve("graders", loaded.experiment.grader))
        return compare_experiment(
            run, grader=grader, prices_by_role=configured_prices(loaded),
        ).to_dict()
    raise AssertionError(f"unhandled command {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = asyncio.run(_execute(args))
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
        return EXIT_OK
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (ConfigurationError, DatasetError, ValueError, KeyError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except Exception as exc:
        print(f"infrastructure failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE


if __name__ == "__main__":
    raise SystemExit(main())
