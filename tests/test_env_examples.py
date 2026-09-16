import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_env_examples.py"


def test_env_trusted_origins_key_is_registered() -> None:
    """LUMILAKE_API_TRUSTED_ORIGINS was added to .env.example (documenting the
    LLMChatOp API-mode trust boundary) and must be registered in
    doctor._OPTIONAL_KEYS, or scripts/dev/check_env_examples.py flags it as
    an unknown key. Pins that registration in
    packages/deploy/src/lumilake_deploy/doctor.py's ``_OPTIONAL_KEYS`` tuple:
    removing the entry there (while keeping the .env.example line) makes
    this fail with "Unknown env example keys"."""
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
