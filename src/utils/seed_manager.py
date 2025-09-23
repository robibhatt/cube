import random
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["seed_all", "SeedManager"]


def seed_all(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs with ``seed``."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SeedManager:
    """Utility for spawning deterministic seeds and :class:`torch.Generator`s."""

    seed: int
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def spawn_seed(self) -> int:
        """Return a fresh random integer seed."""
        return self.rng.randint(0, 2**63 - 1)

    def spawn_generator(self) -> torch.Generator:
        """Return a :class:`torch.Generator` seeded from :meth:`spawn_seed`."""
        g = torch.Generator()
        g.manual_seed(self.spawn_seed())
        return g
