from unittest.mock import MagicMock

import pytest

from lumilake.runtime.job_manager import create_job_manager
from lumilake.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake.runtime.optimizer.base import BaseOptimizer


def test_create_job_manager_selects_priority_manager() -> None:
    manager = create_job_manager(
        "priority",
        optimizer=MagicMock(spec=BaseOptimizer),
    )
    assert isinstance(manager, PriorityJobManager)


def test_create_job_manager_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown job manager type"):
        create_job_manager(
            "unknown",
            optimizer=MagicMock(spec=BaseOptimizer),
        )
