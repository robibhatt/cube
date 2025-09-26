"""Bootstrap registrations for joint distributions and their configs."""

# Import the Cube distribution so that its registration decorator executes at
# import time.
from . import cube_distribution  # noqa: F401

# Import the corresponding config module for the same reason.
from .configs import cube_distribution as cube_distribution_config  # noqa: F401
