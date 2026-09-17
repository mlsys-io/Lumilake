import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_env_examples.py"


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
