# AGENTS.md - Lumilake

Routing doc for coding agents (Claude Code, Codex, Cursor, etc.) working in this repository. The full project rules live in dedicated docs; read the ones relevant to your task before editing.

Lumilake is a data analytics engine for agentic workflows. It accepts workflow specs, optimizes the runtime graph with HALO, dispatches work through FlowMesh, and stores progress, results, traces, and artifacts through the archive layer.

## Where to read what

- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** - control-plane topology, package map, runtime flow, storage model, FlowMesh integration, and hook placement. Read before any cross-component change.
- **[`docs/CODE_STYLE.md`](docs/CODE_STYLE.md)** - Python rules, import and typing expectations, deploy constraints, testing expectations, and commit hygiene. Read before writing source code.
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** - setup, pre-commit hooks, validation commands, CI workflows, DCO sign-off, and PR title conventions. Read before committing or opening a PR.
- **[`docs/API.md`](docs/API.md)** - server route overview, response shape, job lifecycle endpoints, workers, traces, and health/status surfaces. Read before calling the server directly or changing a router.
- **[`docs/SDK.md`](docs/SDK.md)** - sync and async client usage, install extras, configuration, resource surfaces, and deploy helper behavior.
- **[`docs/CLI.md`](docs/CLI.md)** - `lumilake ...` command groups, local deploy lifecycle, job operations, workers, and traces.
- **[`docs/ENV.md`](docs/ENV.md)** - deploy-time environment variables, archive settings, direct data mode, lumid.data mode, model settings, and FlowMesh settings.
- **[`docs/PLUGINS.md`](docs/PLUGINS.md)** - shared hook interfaces, Lumilake resource kinds, plugin loading, and sample plugin layout.

Concrete examples and runnable workflows live in `examples/templates/`. Example plugins live in `examples/plugins/`. When code, APIs, CLI commands, SDK methods, env vars, workflow formats, hook behavior, or runtime behavior change, update the corresponding docs in the same PR.

## Reminders

The full justification lives in the linked docs; these are the rules that come up most often.

- **PR title type**: one of `feat | fix | refactor | chore | test | perf | build | ci | docs`. Prefix with `[BREAKING]` for breaking changes.
- **No source-history breadcrumbs**: do not write comments or docstrings that say what code used to replace, wrap, or migrate from.
- **No back-compat shims**: replace old paths outright; do not keep aliases, re-exports, no-op stubs, or dual contracts for unreleased behavior.
- **Imports stay at the top** unless a real circular import or optional heavy dependency forces otherwise.
- **Prefer typed models and explicit fields** over untyped dicts, dynamic `setattr`, or `getattr` for known attributes.
- **Use `uv run ...`** for local commands. Run the relevant checks from `CONTRIBUTING.md` before pushing.
- **Cluster management goes through Lumilake CLI/SDK/deploy helpers**. Raw Docker commands are for unsupported diagnostics only.

## Dev workflow

Use `CONTRIBUTING.md` for setup, hooks, tests, dependency pins, and commit rules. Use `docs/CLI.md` for local stack and workflow commands.

## Auto-loaded references

@docs/ARCHITECTURE.md
@docs/CODE_STYLE.md
@CONTRIBUTING.md
