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

Point `PYTHONPATH` at the plugin package and set `LUMILAKE_PLUGINS`:

```bash
PYTHONPATH=./examples/plugins
LUMILAKE_PLUGINS=simple_plugin
```

Multiple plugins are comma-separated. Plugins should expose an `install()` function that wires their bindings or registrations.

`install()` may return `HookBindings` directly, return an awaitable that resolves to `HookBindings`, or be an async context manager that yields `HookBindings`. Context managers are entered during server startup and unwound during server shutdown.

## Design Rule

Shared hooks should contain only project-neutral contracts. Resource names, optimizer registration, and workflow/runtime-specific behavior should remain in Lumilake.
