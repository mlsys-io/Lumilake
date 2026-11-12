import re
import subprocess
import sys
from pathlib import Path

from lumilake import envs
from lumilake_deploy import doctor

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_env_examples.py"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

_ASSIGNMENT_RE = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=")


def test_env_trusted_origins_key_is_registered() -> None:
    """LUMILAKE_API_TRUSTED_ORIGINS must be registered in
    doctor._OPTIONAL_KEYS, or scripts/dev/check_env_examples.py flags it as
    an unknown key."""
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_flowmesh_stack_paths_are_not_lumilake_env_settings() -> None:
    """The FlowMesh stack reads these from ``.env.flowmesh``, so listing them
    in Lumilake's ``.env`` or its example promises an effect they do not
    have."""
    keys = {"REDIS_TLS_DIR", "SERVER_TLS_DIR", "SERVER_WORKER_CONFIG"}
    example_keys = {
        match.group(1)
        for line in ENV_EXAMPLE.read_text().splitlines()
        if (match := _ASSIGNMENT_RE.match(line))
    }
    assert keys.isdisjoint(doctor._OPTIONAL_KEYS)
    assert keys.isdisjoint(vars(envs))
    assert keys.isdisjoint(example_keys)
