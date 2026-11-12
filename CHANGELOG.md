# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-12

### Fixed

- `http_request` resolved a hostname, checked the addresses against the policy,
  and then let aiohttp resolve it again to open the socket. A record that
  changed between the two lookups sent the request somewhere the allowlist
  never approved (DNS rebinding / TOCTOU). Connections are now pinned to the
  approved address; SNI and the `Host` header still come from the URL, so
  certificate validation is unchanged.

## [0.1.0] - 2026-07-08

- Initial release: layered diagnosis over DNS, routing, TCP, TLS and HTTP
  tools, with a tool budget, target scoping and an offline deterministic
  playbook when no model is available.
