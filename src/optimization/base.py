from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseOptimizer(ABC):

    def __init__(self, bounds: np.ndarray, dim: int | None = None):
        """"""

    @abstractmethod
    def ask(self) -> np.ndarray:
        """"""

    @abstractmethod
    def tell(self):
        """"""

    def _store_observations(self):
        """"""

    def best(self):
        """"""