"""Offline mode: a deterministic decision tree, no model required.

This is the same layered method the agent's prompt describes, written as
code. It exists for three reasons: Triage stays useful on an isolated
network with no model access; it gives a baseline to compare the agent
against (if the LLM can't beat a fixed tree, it isn't adding anything); and
it documents the method in a form that can't be misread.

The tree short-circuits at the first broken layer, because everything above
a failed layer is noise: if DNS fails, an HTTP timeout tells you nothing new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

from . import tools
from .agent import Run, Step


@dataclass
class Target:
    host: str
    port: int
    url: str | None


def parse_target(target: str) -> Target:
    if "://" in target:
        parts = urlsplit(target)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return Target(parts.hostname or "", port, target)
    host, _, port = target.rpartition(":") if target.count(":") == 1 else (target, "", "")
    return Target(host or target, int(port) if port else 443, None)


async def diagnose(target: str, on_step: Callable[[Step], None] | None = None) -> Run:
    t = parse_target(target)
    run = Run(f"offline playbook for {target}")
    findings: list[str] = []
    ok_layers: list[str] = []

    async def step(name: str, **kwargs) -> dict:
        import time

        start = time.perf_counter()
        try:
            result = await tools.registry.tools[name].fn(**kwargs)
        except tools.PolicyError as exc:
            result = {"ok": False, "summary": f"blocked by policy: {exc}"}
        s = Step(name, kwargs, result, round(time.perf_counter() - start, 2))
        run.steps.append(s)
        if on_step:
            on_step(s)
        return result

    def finish(diagnosis: str, next_steps: list[str], confidence: str) -> Run:
        run.answer = "\n".join([
            "## Diagnosis", diagnosis, "",
            "## Evidence", *[f"- {f}" for f in findings], "",
            "## Ruled out", *([f"- {o}" for o in ok_layers] or ["- nothing yet"]), "",
            "## Next steps", *[f"- {n}" for n in next_steps], "",
            f"Confidence: {confidence}",
        ])
        return run

    is_ip = t.host.replace(".", "").isdigit() or ":" in t.host
    if not is_ip:
        sys_dns = await step("dns_lookup", name=t.host)
        findings.append(f"system resolver: {sys_dns['summary']}")
        if not sys_dns["ok"]:
            pub = await step("dns_lookup", name=t.host, server="1.1.1.1")
            findings.append(f"1.1.1.1: {pub['summary']}")
            if pub["ok"]:
                return finish("Name resolution fails on the local resolver but works publicly: "
                              "the internal DNS server is down, unreachable or missing the record.",
                              ["Check the DNS server handed out by DHCP and whether it answers on port 53",
                               "Look for a missing or mistyped record / split-horizon zone"], "high")
            return finish(f"{t.host} does not resolve anywhere.",
                          ["Verify the hostname spelling", "Check the zone's NS delegation and records"], "high")
        ok_layers.append("DNS resolution")

    port = await step("tcp_connect", host=t.host, port=t.port)
    findings.append(f"TCP: {port['summary']}")
    if not port["ok"]:
        if "refused" in port["summary"]:
            return finish(f"The host is reachable but nothing is listening on port {t.port}.",
                          ["Check the service is running and bound to the right interface",
                           "Check a local host firewall rejecting (not dropping) the port"], "high")
        icmp = await step("ping", host=t.host, count=3)
        findings.append(f"ICMP: {icmp['summary']}")
        if icmp["ok"]:
            return finish(f"The host answers ping but port {t.port} times out: traffic to that port is filtered.",
                          ["Check firewall / security-group rules for the port",
                           "Confirm the service listens on the address being tested"], "medium")
        trace = await step("traceroute", host=t.host, max_hops=20)
        findings.append(f"traceroute: {trace['summary']}; last hops {trace.get('hops', [])[-2:]}")
        return finish("The host is unreachable at the network layer.",
                      ["Use the traceroute output to find the last responding hop",
                       "Check routing and ACLs at and after that hop",
                       "Confirm the host is powered on and has a correct gateway"], "medium")
    ok_layers.append(f"TCP port {t.port}")

    if t.port in (443, 8443) or (t.url or "").startswith("https"):
        cert = await step("tls_certificate", host=t.host, port=t.port)
        findings.append(f"TLS: {cert['summary']}")
        if not cert["ok"]:
            return finish("TLS fails: the certificate is invalid, expired or doesn't match the name.",
                          ["Renew or replace the certificate", "Check the full chain is served",
                           "Make sure the requested name is in the certificate's SANs"], "high")
        ok_layers.append("TLS certificate")

    url = t.url or f"{'https' if t.port in (443, 8443) else 'http'}://{t.host}:{t.port}/"
    http = await step("http_request", url=url)
    findings.append(f"HTTP: {http['summary']}")
    if not http["ok"]:
        status = http.get("status")
        hint = {502: "the proxy cannot reach its backend", 503: "the backend is overloaded or in maintenance",
                504: "the backend is too slow for the proxy's timeout"}.get(status, "the application returned an error")
        lower = "Network and TLS are fine" if "TLS certificate" in ok_layers else "The network path is fine"
        return finish(f"{lower}; the application layer fails ({hint}).",
                      ["Check application and proxy logs at the time of the request",
                       "Check backend health from the proxy host"], "medium")
    ok_layers.append("HTTP response")

    return finish("Every layer checked is healthy from this machine.",
                  ["If users still report problems, test from their network segment — the fault may be "
                   "local to them (Wi-Fi, VLAN, client DNS, proxy settings)"], "medium")
