# Hooks and Plugins

Lumilake uses `lumid-hooks` for shared extension points that can also be reused by other projects. Lumilake-specific extension points stay in this repository.

## Shared Hook Surfaces

| Hook | Purpose |
|------|---------|
| `IdentityProvider` | Resolve a request into a `PrincipalContext`. |
| `SubmissionGuard` | Allow or deny job submission before work is queued. |
| `PermissionChecker` | Authorize actions on resource kinds and resource IDs. |
| `ResourceRegistrar` | Register or deregister resource ownership/lifecycle records. |
| `UsageSink` | Receive usage rows emitted by the server. |

Lumilake resource kinds and actions live under `lumilake_hook` so hook implementations can avoid hard-coded strings.

## Runtime Credentials

The bearer that `IdentityProvider.resolve` accepts is forwarded to FlowMesh as the per-request API key. Two consequences for plugin authors:

- A token an `IdentityProvider` plugin chooses to accept must also authenticate against the configured FlowMesh deployment, since the same string is what FlowMesh sees on the runtime side.
- Local deployments with no `IdentityProvider` plugin installed dispatch to FlowMesh with no `Authorization` header, matching FlowMesh's local-mode behavior. Set `LUMILAKE_REQUIRE_IDENTITY_PROVIDER=1` in cloud deploys to refuse this fallback.

Workflows that share a `PrincipalContext.principal_id` may coalesce into one FlowMesh submission; workflows from different principals always dispatch separately so each carries the originating principal's bearer.

`principal_id` is a **security boundary**. Two requests resolving to the same `principal_id` are treated as one ownership scope — their workflows can be merged into one FlowMesh submission with a single bearer, and FlowMesh attribution falls on that principal. An `IdentityProvider` plugin must therefore guarantee one `principal_id` per ownership scope. Collisions (two distinct users mapped to one id) enable cross-tenant data flow; collisions are the plugin's responsibility to prevent, not Lumilake's.

Lumilake holds two distinct FlowMesh credentials and never crosses them:

- **Per-request bearer.** HTTP routes (`GET /workers`, `GET /trace/{id}`, job submission, cancel) and the runtime dispatch of a user's tasks all carry the user's own bearer to FlowMesh. The user's FlowMesh scope governs what they can do.
- **`LUMILAKE_RUNTIME_TOKEN`.** Scheduler-internal credential. Only the runtime manager's `get_workers` / `get_worker_profile` reach it, and only to plan dispatch. Never reachable from any HTTP route handler. With `lumid.flowmesh-plugin`'s permission model, this token's FlowMesh principal needs `flowmesh:workers:read`; user PATs need only their task-related scopes.

## Lumilake-Owned Plugin Surface

Optimizer registration is Lumilake-specific. A plugin may register an optimizer implementation and select it with `LUMILAKE_OPTIMIZER_TYPE`.

The example plugin registers:

- shared hook implementations for identity, permissions, submission guards, registrar, and usage;
- a simple Lumilake optimizer plugin.

See `examples/plugins/simple_plugin/`.

## Enabling Plugins

Plugins are loaded inside the running server process (the Lumilake server image). The optimizer-registration and shared-hook surfaces — `lumilake_hook`, `lumilake_server.runtime.optimizer.*`, `lumilake_server.runtime.runtime_graph.*` — are reachable only from inside the image; they are intentionally not part of any published PyPI wheel. Develop plugins against a checked-out repository or against a custom Docker image that layers your plugin code on top of `ghcr.io/mlsys-io/lumilake_server`.

Point `PYTHONPATH` at the plugin package and set `LUMILAKE_PLUGINS`:

```bash
PYTHONPATH=./examples/plugins
LUMILAKE_PLUGINS=simple_plugin
```

Multiple plugins are comma-separated. Plugins should expose an `install()` function that wires their bindings or registrations.

`install()` may return `HookBindings` directly, return an awaitable that resolves to `HookBindings`, or be an async context manager that yields `HookBindings`. Context managers are entered during server startup and unwound during server shutdown.

## Plugin Dev Workflow

### Layout

A plugin is an ordinary Python package whose top-level module exposes an
`install()` callable. The example layout under
`examples/plugins/simple_plugin/` is:

```text
simple_plugin/
├── __init__.py          # defines install()
├── simple_identity.py   # IdentityProvider impl
├── simple_permissions.py
├── simple_registrar.py
├── simple_submission.py
├── simple_usage.py
├── simple_optimizer.py  # optimizer registration helper
└── state.py             # shared in-memory state for the demo hooks
```

`__init__.py` returns the bundle:

```python
from lumilake_hook import BaseBindings

from .simple_identity import SimpleIdentityProvider
from .simple_optimizer import install_optimizer
from .simple_permissions import SimplePermissionChecker
from .simple_registrar import SimpleResourceRegistrar
from .simple_submission import SimpleSubmissionGuard
from .simple_usage import SimpleUsageSink


def install() -> BaseBindings:
    install_optimizer()
    return BaseBindings(
        identity_providers=(SimpleIdentityProvider(),),
        submission_guards=(SimpleSubmissionGuard(),),
        usage_sinks=(SimpleUsageSink(),),
        permission_checkers=(SimplePermissionChecker(),),
        resource_registrars=(SimpleResourceRegistrar(),),
    )
```

`BaseBindings` (re-exported from `lumilake_hook`) is the default
`HookBindings` implementation. A plugin only needs to populate the
fields it actually overrides.

### `install()` shapes

The server accepts three shapes; pick whichever matches the
implementation:

- **Plain function** returning `HookBindings`:

  ```python
  def install() -> HookBindings:
      return BaseBindings(...)
  ```

- **Async function** returning `HookBindings`. Use when setup needs to
  await I/O at boot:

  ```python
  async def install() -> HookBindings:
      client = await connect_to_my_backend()
      return BaseBindings(usage_sinks=(MyUsageSink(client),))
  ```

- **Async context manager** yielding `HookBindings`. Use when the
  plugin owns resources that need clean shutdown (the server enters the
  context at startup and exits it on shutdown):

  ```python
  from contextlib import asynccontextmanager

  @asynccontextmanager
  async def install():
      client = await connect_to_my_backend()
      try:
          yield BaseBindings(usage_sinks=(MyUsageSink(client),))
      finally:
          await client.aclose()
  ```

### Local testing

Plugins are loaded inside the server process, so test against a source
checkout. Set `PYTHONPATH` to the directory that contains the plugin
package and list the module name in `LUMILAKE_PLUGINS`:

```bash
export PYTHONPATH=$PWD/examples/plugins
export LUMILAKE_PLUGINS=simple_plugin
uv run lumilake deploy -C ~/lumilake-deploy up
uv run lumilake deploy -C ~/lumilake-deploy logs server --tail 100
```

The server logs `Plugin '<name>' registered.` on success. A plugin that
imports, defines `install()`, but returns something other than
`HookBindings` (or whose `install()` raises) is logged with
`Plugin '<name>' install() failed validation; skipping` — the server
boots without it instead of crashing. Watch the log for this line when a
new plugin appears not to take effect.

### Unit tests

Tests can call `install()` directly and inspect the returned bindings,
or feed the bindings into `lumilake_server.hooks.register()` and drive
the resulting providers from `lumilake_server.hooks.*`. See
`tests/server/test_simple_plugin_e2e.py` for a worked example covering
identity, submission, permissions, registrar, usage, and optimizer
behavior end-to-end.

## Schedule Protocol Contract

`RemoteOptimizer` (`lumilake_server.runtime.optimizer.RemoteOptimizer`) implements `BaseOptimizer` by POSTing a `ScheduleRequest` to `{LUMILAKE_REMOTE_OPTIMIZER_URL}/api/v1/optimizer/schedule` and deserializing the `ScheduleResponse`. `ScheduleRequest` carries the serialized `RuntimeGraph` (under `graph`), `worker_names`, `worker_profiles`, `data_profile_results`, and an `optimizer_type` selector. `ScheduleResponse` carries `worker_assignment`. Both models are importable from `lumilake_server.runtime.optimizer.schemas` and are the canonical contract that any compatible schedule-protocol server must satisfy.

`RemoteOptimizer` requires an explicit `optimizer_type` kwarg at construction time (e.g. `RemoteOptimizer(optimizer_type="halo-greedy")`). The typical integration path is via an `OptimizerProvider` plugin (see below).

## OptimizerProvider Hook

Plugins can contribute optimizer types beyond the built-in `halo` and `topological-sort` by implementing `lumilake_hook.OptimizerProvider` and including instances in `BaseBindings.optimizer_providers`. Lumilake's `create_optimizer(type)` falls through to registered providers when `type` is absent from the local `OPTIMIZER_TYPES` dict.

```python
from lumilake_hook import BaseBindings, OptimizerProvider

class MyProvider:
    def list_optimizers(self) -> list[str]:
        return ["halo-greedy"]

    def create_optimizer(self, optimizer_type: str, **kwargs):
        from lumilake_server.runtime.optimizer.remote import RemoteOptimizer
        return RemoteOptimizer(optimizer_type=optimizer_type, **kwargs)

def install() -> BaseBindings:
    return BaseBindings(optimizer_providers=(MyProvider(),))
```

Setting `LUMILAKE_OPTIMIZER_TYPE=halo-greedy` then routes schedule generation through the provider. The server also exposes `GET /api/v1/optimizer` which returns all locally registered and provider-advertised types.

## Design Rule

Shared hooks should contain only project-neutral contracts. Resource names, optimizer registration, and workflow/runtime-specific behavior should remain in Lumilake.
