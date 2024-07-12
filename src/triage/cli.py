"""Command-line entry point.

``triage ask`` runs the LLM agent; ``triage check`` runs the offline
playbook. Both stream every tool call as it happens, because an agent you
can't watch is an agent you can't trust: the trace is how a human checks
that the conclusion actually follows from the evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import playbook, tools
from .agent import Agent, Run, Step
from .llm import PRESETS, OpenAICompatible
from .policy import Policy

console = Console()


def show_step(step: Step) -> None:
    mark = "[green]✓[/]" if step.result.get("ok") else "[red]✗[/]"
    args = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items())
    console.print(f"{mark} [bold]{step.tool}[/]({args}) [dim]{step.seconds}s[/]")
    console.print(f"   [dim]{step.result.get('summary', '')}[/]", highlight=False)


def show_run(run: Run) -> None:
    console.print()
    console.print(Panel(Markdown(run.answer or "_no answer_"), title="Triage", border_style="cyan"))
    meta = f"{len(run.steps)} tool calls" + (f" · {run.turns} turns" if run.turns else "")
    if run.tokens:
        meta += f" · {run.tokens} tokens"
    if run.stopped_reason != "answered":
        meta += f" · stopped: {run.stopped_reason}"
    console.print(f"[dim]{meta}[/]")


def build_policy(args) -> Policy:
    return Policy(allowed_networks=args.allow_net or [], allowed_domains=args.allow_domain or [],
                  max_tool_calls=args.budget)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="triage", description="AI agent for network troubleshooting")
    sub = p.add_subparsers(dest="cmd", required=True)

    def scope(sp):
        sp.add_argument("--allow-net", action="append", metavar="CIDR", help="restrict targets to this network")
        sp.add_argument("--allow-domain", action="append", metavar="DOMAIN", help="restrict targets to this domain")
        sp.add_argument("--budget", type=int, default=25, help="maximum tool calls per run")
        sp.add_argument("--json", action="store_true", help="print the full run as JSON")

    ask = sub.add_parser("ask", help="investigate a problem with an LLM agent")
    ask.add_argument("question", help='e.g. "users on guest Wi-Fi cannot open intranet.corp.lan"')
    ask.add_argument("--provider", choices=list(PRESETS), default="openai")
    ask.add_argument("--model")
    ask.add_argument("--base-url", help="any OpenAI-compatible endpoint")
    ask.add_argument("--context", default="", help="extra facts: topology, recent changes, error text")
    ask.add_argument("--max-turns", type=int, default=10)
    scope(ask)

    chk = sub.add_parser("check", help="run the offline layered playbook against a host, host:port or URL")
    chk.add_argument("target")
    scope(chk)

    sub.add_parser("tools", help="list tools and their JSON schemas")
    args = p.parse_args(argv)

    if args.cmd == "tools":
        console.print_json(json.dumps(tools.registry.specs()))
        return

    policy = build_policy(args)
    tools.set_policy(policy)

    if args.cmd == "check":
        run = asyncio.run(playbook.diagnose(args.target, on_step=None if args.json else show_step))
    else:
        try:
            model = OpenAICompatible.from_preset(args.provider, args.model, args.base_url)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/] — or run [bold]triage check <target>[/] for the offline playbook")
            sys.exit(2)

        def thought(text: str) -> None:
            if not args.json:
                console.print(f"[italic dim]» {text.strip()}[/]")

        agent = Agent(model, tools.registry, max_turns=args.max_turns,
                      on_step=None if args.json else show_step, on_thought=thought)
        run = asyncio.run(agent.run(args.question, args.context, charge=policy.charge))

    if args.json:
        print(json.dumps(run, default=lambda o: o.__dict__, ensure_ascii=False, indent=2))
    else:
        show_run(run)


if __name__ == "__main__":
    main()
