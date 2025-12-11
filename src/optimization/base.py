from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseOptimizer(ABC):

    _X: list[np.ndarray]
    _y: list[float]

    def __init__(self, bounds: np.ndarray, seed: int | None = None):
        """
        sets up the bounds and dimensions for further computation after ensuring that
        bounds are stored as an array of type float. Initializes the dimensional
        information, internal storage for data (`_X` and `_y`) and an optional seed
        """

        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[1]

        self._X = []
        self._y = []

        self.seed = seed

        pass

    @abstractmethod
    def ask(self, n_points: int = 1) -> np.ndarray:

        """
        Returns n_points points within the bounds
        """

        pass

    @abstractmethod
    def tell(self, X: np.ndarray, y: np.ndarray, meta: dict[str, Any] | None = None):

        """
        passes observations back to the optimizer
        """

        pass

    def _store_observations(self, X: np.ndarray, y: np.ndarray) -> None:

        """
        ensures that the dimensions of X and y are correct and appends them to the internal lists
        """

        X = np.atleast_2d(X)

        y = np.asarray(y, dtype=float).reshape(-1)

        for xi, yi in zip(X, y):
            self._X.append(np.asarray(xi, dtype=float).copy())
            self._y.append(float(yi))

        pass

    def best(self) -> tuple[np.ndarray, float] | None:
        """
        returns the best sample found so far, or None if samples are yet to exist
        """

        if not self._y:
            return None

        y_array = np.array(self._y, dtype=float)

        best_index = int(np.argmin(y_array))

        x_best = self._X[best_index]
        y_best = float(y_array[best_index])

        return x_best, y_best