import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TESTS_DIR.parent

for path in (str(_TESTS_DIR), str(_ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
