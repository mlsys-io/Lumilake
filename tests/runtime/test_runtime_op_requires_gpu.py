"""``_runtime_op_requires_gpu`` decides whether the scheduler claims a GPU group."""

from types import SimpleNamespace

import pytest

from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager


def _op(task_type: str, backend: str = "", gpu_request=None) -> SimpleNamespace:
    return SimpleNamespace(
        task_type=task_type,
        backend=backend,
        data_spec=SimpleNamespace(gpu_request=gpu_request),
    )


@pytest.mark.parametrize(
    "task_type", ["inference", "embedding", "diffusion", "omni_text2image"]
)
def test_gpu_task_types_claim_a_gpu(task_type: str) -> None:
    assert FlowmeshRuntimeManager._runtime_op_requires_gpu(_op(task_type))


def test_image_generation_lowering_claims_a_gpu() -> None:
    # ImageGenerationOp lowers to omni_text2image; without it the batch went to a
    # CPU worker and HALO failed with "requires at least one GPU worker".
    assert FlowmeshRuntimeManager._runtime_op_requires_gpu(_op("OMNI_TEXT2IMAGE "))


@pytest.mark.parametrize("task_type", ["python", "data_retrieval", "api", ""])
def test_cpu_task_types_do_not(task_type: str) -> None:
    assert not FlowmeshRuntimeManager._runtime_op_requires_gpu(_op(task_type))
