import asyncio
import json
import time

from triage.agent import Agent
from triage.llm import Reply, ToolCall, parse_reply
from triage.policy import Policy
from triage.registry import Registry


class ScriptedModel:
    """Replays a fixed sequence of replies and records what it was shown."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    async def complete(self, messages, tools):
        self.seen.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.replies.pop(0)


def make_registry(delay=0.0):
    reg = Registry()

    @reg.tool
    async def lookup(name: str) -> dict:
        """Look a name up for testing purposes in the agent loop tests.

        Args:
            name: Name.
        """
        await asyncio.sleep(delay)
        return {"ok": True, "summary": f"{name} -> 10.0.0.5"}

    @reg.tool
    async def explode() -> dict:
        """Always raises, to check tool bugs become observations not crashes."""
        raise RuntimeError("kaboom")

    return reg


async def test_tool_result_is_fed_back_and_answer_returned():
    model = ScriptedModel([
        Reply("Checking DNS first.", [ToolCall("c1", "lookup", {"name": "intranet"})]),
        Reply("## Diagnosis\nDNS is fine."),
    ])
    run = await Agent(model, make_registry()).run("intranet is down")
    assert run.answer.startswith("## Diagnosis")
    assert [s.tool for s in run.steps] == ["lookup"]
    tool_msg = model.seen[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert "10.0.0.5" in tool_msg["content"]


async def test_independent_calls_run_in_parallel():
    calls = [ToolCall(f"c{i}", "lookup", {"name": f"h{i}"}) for i in range(5)]
    model = ScriptedModel([Reply("", calls), Reply("done")])
    start = time.perf_counter()
    await Agent(model, make_registry(delay=0.2)).run("q")
    assert time.perf_counter() - start < 0.6  # five 0.2 s calls, concurrently


async def test_errors_become_observations_the_model_can_react_to():
    model = ScriptedModel([
        Reply("", [ToolCall("a", "explode", {}),
                   ToolCall("b", "lookup", {"name": "x", "bogus": 1}),
                   ToolCall("c", "nonexistent", {}),
                   ToolCall("d", "lookup", {}, "{bad json", "invalid JSON arguments: Expecting value")]),
        Reply("answer"),
    ])
    run = await Agent(model, make_registry()).run("q")
    summaries = [s.result["summary"] for s in run.steps]
    assert "tool error: RuntimeError: kaboom" in summaries[0]
    assert "unknown argument" in summaries[1]
    assert "no such tool" in summaries[2]
    assert "invalid JSON" in summaries[3]
    assert run.answer == "answer"


async def test_last_turn_withholds_tools_to_force_an_answer():
    loop_forever = [Reply("", [ToolCall(str(i), "lookup", {"name": "x"})]) for i in range(2)]
    model = ScriptedModel(loop_forever + [Reply("forced answer")])
    run = await Agent(model, make_registry(), max_turns=3).run("q")
    assert model.seen[-1]["tools"] == []
    assert run.answer == "forced answer" and run.stopped_reason == "turn limit"


async def test_budget_exhaustion_stops_tools_and_asks_for_answer():
    policy = Policy(max_tool_calls=1)
    model = ScriptedModel([
        Reply("", [ToolCall("a", "lookup", {"name": "x"}), ToolCall("b", "lookup", {"name": "y"})]),
        Reply("answer with partial evidence"),
    ])
    run = await Agent(model, make_registry()).run("q", charge=policy.charge)
    assert run.stopped_reason == "tool budget"
    assert any(s.result.get("budget_exhausted") for s in run.steps)
    assert run.answer == "answer with partial evidence"


async def test_large_results_are_truncated_before_reaching_the_model():
    reg = Registry()

    @reg.tool
    async def dump() -> dict:
        """Return a very large result to verify truncation of tool output."""
        return {"ok": True, "summary": "x" * 10_000}

    model = ScriptedModel([Reply("", [ToolCall("a", "dump", {})]), Reply("ok")])
    run = await Agent(model, reg).run("q")
    sent = model.seen[1]["messages"][-1]["content"]
    assert len(sent) < 2000 and "[truncated]" in sent
    assert len(run.steps[0].result["summary"]) == 10_000  # trace keeps everything


def test_parse_reply_tolerates_malformed_arguments():
    data = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "lookup", "arguments": "{name: x,}"}},
        {"id": "2", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"name": "y"})}},
    ]}}], "usage": {"total_tokens": 42}}
    reply = parse_reply(data)
    assert reply.tool_calls[0].parse_error and reply.tool_calls[1].arguments == {"name": "y"}
    assert reply.usage["total_tokens"] == 42


async def test_openai_compatible_client_wire_format():
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from triage.llm import OpenAICompatible

    seen = {}

    async def completions(request):
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        if len(seen.setdefault("n", [])) == 0:
            seen["n"].append(1)
            return web.json_response({"error": "slow down"}, status=429, headers={"Retry-After": "0"})
        return web.json_response({"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": '{"name": "a"}'}}]}}],
            "usage": {"total_tokens": 7}})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    srv = TestServer(app)
    await srv.start_server()
    try:
        model = OpenAICompatible(str(srv.make_url("/v1")), "test-model", api_key="sk-test")
        reply = await model.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])
    finally:
        await srv.close()
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "test-model" and seen["body"]["tool_choice"] == "auto"
    assert reply.tool_calls[0].arguments == {"name": "a"}
