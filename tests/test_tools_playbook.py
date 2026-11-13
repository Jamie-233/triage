import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from triage import playbook, tools
from triage.policy import Policy


@pytest.fixture(autouse=True)
def open_policy():
    tools.set_policy(Policy())
    yield
    tools.set_policy(Policy())


async def test_tcp_connect_open_and_refused():
    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        ok = await tools.tcp_connect("127.0.0.1", port)
    refused = await tools.tcp_connect("127.0.0.1", port)
    assert ok["ok"] and "open" in ok["summary"]
    assert not refused["ok"] and "refused" in refused["summary"]


async def test_subnet_info_flags_gateway_outside_subnet():
    r = await tools.subnet_info("192.168.10.37/26", "192.168.10.1")
    assert r["network"] == "192.168.10.0/26"
    assert r["usable_hosts"] == 62
    assert r["contains"] is True
    r2 = await tools.subnet_info("192.168.10.70/26", "192.168.10.1")
    assert r2["contains"] is False and "OUTSIDE" in r2["summary"]


async def _serve(routes):
    app = web.Application()
    for path, handler in routes.items():
        app.router.add_get(path, handler)
    srv = TestServer(app, host="127.0.0.1")
    await srv.start_server()
    return srv


async def test_http_redirects_are_followed_and_reported():
    async def start(_):
        raise web.HTTPFound("/final")

    async def final(_):
        return web.Response(text="hello", headers={"Server": "test"})

    srv = await _serve({"/": start, "/final": final})
    try:
        r = await tools.http_request(str(srv.make_url("/")))
    finally:
        await srv.close()
    assert r["ok"] and r["status"] == 200
    assert len(r["redirects"]) == 2 and r["redirects"][-1].endswith("/final")


async def test_redirect_out_of_scope_is_blocked():
    async def start(_):
        raise web.HTTPFound("http://10.99.99.99/admin")

    srv = await _serve({"/": start})
    tools.set_policy(Policy(allowed_networks=["127.0.0.0/8"]))
    try:
        with pytest.raises(tools.PolicyError, match="outside the allowed networks"):
            await tools.http_request(str(srv.make_url("/")))
    finally:
        await srv.close()


async def test_redirect_loop_is_detected():
    async def loop(_):
        raise web.HTTPFound("/")

    srv = await _serve({"/": loop})
    try:
        r = await tools.http_request(str(srv.make_url("/")))
    finally:
        await srv.close()
    assert not r["ok"] and "redirect loop" in r["summary"]


@pytest.mark.parametrize("target, host, port", [
    ("https://intranet.corp.lan/login", "intranet.corp.lan", 443),
    ("http://10.0.0.5:8080/", "10.0.0.5", 8080),
    ("files.corp.lan:22", "files.corp.lan", 22),
    ("files.corp.lan", "files.corp.lan", 443),
])
def test_parse_target(target, host, port):
    t = playbook.parse_target(target)
    assert (t.host, t.port) == (host, port)


async def test_playbook_stops_at_closed_port():
    server = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()
    run = await playbook.diagnose(f"127.0.0.1:{port}")
    assert [s.tool for s in run.steps] == ["tcp_connect"]
    assert "nothing is listening" in run.answer


async def test_playbook_reports_application_error():
    async def broken(_):
        return web.Response(status=502, text="bad gateway")

    srv = await _serve({"/": broken})
    try:
        run = await playbook.diagnose(str(srv.make_url("/")))
    finally:
        await srv.close()
    assert [s.tool for s in run.steps] == ["tcp_connect", "http_request"]
    assert "proxy cannot reach its backend" in run.answer
    assert "TCP port" in run.answer


async def test_http_request_dials_the_approved_address_not_a_second_resolution(monkeypatch):
    # The hostname deliberately does not resolve on this machine. Only the
    # pinned resolver can turn it into the address the policy approved, so an
    # implementation that re-resolves at connect time (the TOCTOU window)
    # fails here while the pinned one succeeds.
    async def fake_resolve(host):
        return ["127.0.0.1"]

    monkeypatch.setattr(tools, "_resolve", fake_resolve)

    async def ok(_):
        return web.Response(text="pinned", headers={"Server": "test"})

    srv = await _serve({"/": ok})
    try:
        url = str(srv.make_url("/")).replace("127.0.0.1", "rebind.invalid")
        r = await tools.http_request(url)
    finally:
        await srv.close()
    assert r["ok"] and r["status"] == 200


async def test_pinned_resolver_only_answers_for_its_host():
    resolver = tools._PinnedResolver("approved.test", "127.0.0.1")
    with pytest.raises(OSError):
        await resolver.resolve("other.test", 80)
    [entry] = await resolver.resolve("approved.test", 80)
    assert entry["host"] == "127.0.0.1" and entry["port"] == 80


# --- the same guarantee for the tools that are not HTTP -------------------------
#
# Each of these resolves the name, checks the answer against the policy, and
# must then connect to *that* address. The hostname used below does not resolve
# on this machine, so a tool that resolves a second time cannot produce the
# approved address by accident.

def _approve(monkeypatch, address="127.0.0.1"):
    async def fake_resolve(host):
        return [address]

    monkeypatch.setattr(tools, "_resolve", fake_resolve)


async def test_tls_certificate_connects_to_the_approved_address(monkeypatch):
    _approve(monkeypatch)
    seen = {}

    async def fake_open_connection(host, port, **kw):
        seen.update(host=host, port=port, **kw)
        raise OSError("stop here")

    monkeypatch.setattr(tools.asyncio, "open_connection", fake_open_connection)
    r = await tools.tls_certificate("rebind.invalid", 8443)
    assert not r["ok"]
    assert seen["host"] == "127.0.0.1", "connected by name, which resolves a second time"
    assert seen["server_hostname"] == "rebind.invalid", "SNI and verification must still use the name"


async def test_ping_targets_the_approved_address(monkeypatch):
    import icmplib
    from icmplib.exceptions import ICMPLibError

    _approve(monkeypatch)
    seen = {}

    async def fake_ping(address, **kw):
        seen["address"] = address
        raise ICMPLibError("stop here")

    monkeypatch.setattr(icmplib, "async_ping", fake_ping)
    await tools.ping("rebind.invalid")
    assert seen["address"] == "127.0.0.1"


async def test_traceroute_targets_the_approved_address(monkeypatch):
    _approve(monkeypatch)
    seen = {}

    async def fake_exec(*argv, **kw):
        seen["argv"] = argv
        raise OSError("stop here")

    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/traceroute" if name == "traceroute" else None)
    monkeypatch.setattr(tools.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(OSError):
        await tools.traceroute("rebind.invalid")
    assert seen["argv"][-1] == "127.0.0.1" and "rebind.invalid" not in seen["argv"]


@pytest.mark.parametrize("call", [
    lambda: tools.ping("files.corp.lan"),
    lambda: tools.traceroute("files.corp.lan"),
    lambda: tools.tls_certificate("files.corp.lan"),
])
async def test_name_resolving_outside_scope_is_refused_before_any_packet(monkeypatch, call):
    from triage.policy import PolicyError

    tools.set_policy(Policy(allowed_networks=["10.0.0.0/8"], allowed_domains=["corp.lan"]))
    _approve(monkeypatch, "203.0.113.5")

    async def must_not_run(*a, **kw):
        raise AssertionError("a packet would have been sent")

    monkeypatch.setattr(tools.asyncio, "open_connection", must_not_run)
    monkeypatch.setattr(tools.asyncio, "create_subprocess_exec", must_not_run)
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/traceroute")
    import icmplib
    monkeypatch.setattr(icmplib, "async_ping", must_not_run)
    with pytest.raises(PolicyError, match="resolves to"):
        await call()
