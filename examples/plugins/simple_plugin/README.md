# `simple_plugin`

A self-contained example Lumilake plugin that exercises every shared hook protocol Lumilake wires through `lumid-hooks`, plus Lumilake's repo-owned optimizer registration surface, against an in-memory store.

> **Not for production.** Tokens are plaintext in source, all state is dropped on restart, and every hook is permissive by design. Use this only for poking at the contract.

## What each plugin file does here

| File | Surface | Behavior |
|------|---------|----------|
| `simple_identity.py` | `IdentityProvider` | Looks the bearer token up in `state.TOKENS`. Returns the matching `PrincipalContext`, or `None` to defer to the next provider. |
| `simple_submission.py` | `SubmissionGuard` | Rejects with HTTP 403 if `principal_id` is in `state.BLOCKED_PRINCIPALS`. |
| `simple_usage.py` | `UsageSink` | Appends each `UsageRow` to `state.USAGE_LEDGER`. |
| `simple_permissions.py` | `PermissionChecker` | Admin scope bypasses every check. Otherwise `accessible_ids` returns resources the principal owns; type-level actions require at least one scope; table/object-prefix access requires data scope; concrete-id actions require ownership. |
| `simple_registrar.py` | `ResourceRegistrar` | Records `(resource_kind, resource_id) -> principal_id` in `state.OWNERSHIP` on `register`; drops the row on `deregister`. |
| `simple_optimizer.py` | Lumilake optimizer plugin | Registers `SimpleRoundRobinOptimizer` as the `simple` optimizer type. This is Lumilake-specific and intentionally not part of `lumid-hooks`. |

`state.py` holds every shared dict / set / list. `__init__.py` wires the hook classes into a `BaseBindings`, registers the optimizer, and exposes `install()`.

## Demo principals

| Token | `principal_id` | `org_id` | `scopes` |
|-------|----------------|----------|----------|
| `demo-admin` | `alice` | `demo` | `["admin", "data"]` |
| `demo-user` | `bob` | `demo` | `["user", "data"]` |

`alice` bypasses `PermissionChecker`; `bob` only sees jobs / traces / artifacts he submitted himself.

## Enabling it on a deployed stack

Point `PYTHONPATH` at `examples/plugins`, then list the plugin:

```bash
PYTHONPATH=./examples/plugins
LUMILAKE_PLUGINS=simple_plugin
```

To make the example optimizer the cluster-wide default:

```bash
LUMILAKE_DEFAULT_OPTIMIZER=simple
```

Per-job submissions can also pass `"optimizer": "simple"` to opt in
without changing the server default.

Authenticate requests with one of the demo tokens:

```bash
DEMO_AUTH_VALUE=demo-user
curl -H "Authorization: Bearer $DEMO_AUTH_VALUE" http://localhost:9000/api/v1/jobs
```

## Inspecting what fired

Every hook logs through the injected server logger. Tail the server log and filter for `simple_plugin` to watch `resolve`, `check`, `accessible_ids`, `require`, `register`, `deregister`, and `emit` calls.

## Caveats

- Every store is in-process Python state. A server restart wipes it.
- Demo bearer values are committed plaintext. Real plugins should not ship secrets.
- `PermissionChecker` here is intentionally simple. Real plugins should map specific scopes to specific `(resource_kind, action)` pairs.
- The optimizer plugin is a Lumilake extension point, not a shared `lumid-hooks` protocol.
- Real plugins ship the subset they need; absent hooks fall through to the runtime's documented default.
