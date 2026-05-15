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

## Design Rule

Shared hooks should contain only project-neutral contracts. Resource names, optimizer registration, and workflow/runtime-specific behavior should remain in Lumilake.
