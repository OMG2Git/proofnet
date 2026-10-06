"""Kernel registry: 'task_type@version' -> kernel (ARCHITECTURE 3.4)."""

from dataclasses import dataclass
from types import ModuleType

from ..core import gaussian_nb as core_gnb
from ..core import linear_ridge as core_ridge
from . import gaussian_nb, linear_ridge


@dataclass(frozen=True)
class Kernel:
    task_type: str
    version: str
    core: ModuleType  # worker side: map, merge
    server: ModuleType  # backend side: validate, prepare, validate_partial, finalize, ...

    @property
    def key(self) -> str:
        return f"{self.task_type}@{self.version}"


REGISTRY: dict[str, Kernel] = {
    k.key: k
    for k in (
        Kernel("gaussian_nb_train", "1", core_gnb, gaussian_nb),
        Kernel("linear_ridge_train", "1", core_ridge, linear_ridge),
    )
}


def get_kernel(key: str) -> Kernel:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown kernel '{key}'") from None
