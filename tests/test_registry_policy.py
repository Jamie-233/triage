from typing import Literal

import pytest

from triage.policy import BudgetExhausted, Policy, PolicyError
from triage.registry import Registry
from triage.tools import registry


def test_schema_is_derived_from_signature_and_docstring():
    reg = Registry()

    @reg.tool
    async def probe(host: str, port: int = 443, mode: Literal["fast", "slow"] = "fast") -> dict:
        """Check a thing.

        Args:
            host: Target host.
            port: Target port.
        """
        return {}

    params = reg.tools["probe"].parameters
    assert reg.tools["probe"].description == "Check a thing."
    assert params["required"] == ["host"]
    assert params["properties"]["port"] == {"type": "integer", "description": "Target port.", "default": 443}
    assert params["properties"]["mode"]["enum"] == ["fast", "slow"]
    assert params["additionalProperties"] is False


def test_every_builtin_tool_has_a_description_and_documented_args():
    for tool in registry.tools.values():
        assert len(tool.description) > 40, tool.name
        for name, prop in tool.parameters["properties"].items():
            assert "description" in prop, f"{tool.name}.{name}"


def test_coerce_fixes_scalars_and_rejects_hallucinated_args():
    assert registry.coerce("tcp_connect", {"host": "a.example", "port": "443"})["port"] == 443
    with pytest.raises(ValueError, match="unknown argument"):
        registry.coerce("tcp_connect", {"host": "a", "port": 1, "verbose": True})
    with pytest.raises(ValueError, match="missing required"):
        registry.coerce("tcp_connect", {"host": "a"})
    with pytest.raises(ValueError, match="one of"):
        registry.coerce("dns_lookup", {"name": "a", "record": "SRV"})


@pytest.mark.parametrize("bad", ["a b", "a;rm -rf /", "-oProxyCommand=x", "$(id)", "a..b", ""])
def test_hosts_that_could_be_shell_or_flags_are_rejected(bad):
    with pytest.raises(PolicyError):
        Policy().check_host(bad)


def test_metadata_and_link_local_are_always_blocked():
    with pytest.raises(PolicyError, match="link-local"):
        Policy().check_host("169.254.169.254")


def test_network_scope():
    p = Policy(allowed_networks=["10.10.0.0/16"])
    assert p.check_host("10.10.20.1") == "10.10.20.1"
    with pytest.raises(PolicyError, match="outside the allowed networks"):
        p.check_host("8.8.8.8")


def test_domain_scope_matches_subdomains_only():
    p = Policy(allowed_domains=["corp.lan"])
    assert p.check_host("files.corp.lan")
    with pytest.raises(PolicyError):
        p.check_host("corp.lan.evil.com")


def test_allowed_name_resolving_outside_scope_is_blocked():
    p = Policy(allowed_networks=["10.0.0.0/8"], allowed_domains=["corp.lan"])
    with pytest.raises(PolicyError, match="resolves to"):
        p.check_resolved("files.corp.lan", ["203.0.113.5"])


def test_budget():
    p = Policy(max_tool_calls=2)
    p.charge()
    p.charge()
    with pytest.raises(BudgetExhausted):
        p.charge()
