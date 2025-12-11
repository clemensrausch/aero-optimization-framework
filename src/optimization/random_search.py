from __future__ import annotations

import numpy as np
from typing import Any
from .base import BaseOptimizer

"""
This Optimizer can be used to perform random searches. It completely ignores previous results and samples randomly 
within the bounds given by the user. 
"""

class RandomSearchOptimizer(BaseOptimizer):

    """
    bounds: 2-dim array of lower and upper bounds for each dimension
    seed: (optional) random seed for reproducibility
    """

    def __init__(self, bounds: np.ndarray, seed: int | None = None):

        super().__init__(bounds=bounds)
        self.rng = np.random.default_rng(seed)

    def ask(self, n_points: int = 1) -> np.ndarray:

        lb = self.bounds[0] #lower bounds
        ub = self.bounds[1] #upper bounds

        return self.rng.uniform(lb, ub, size=(n_points, len(lb)))

    def tell(self, X: np.ndarray, y: np.ndarray, meta: dict[str, Any] | None = None):
        self._store_observations(X, y)


    pass