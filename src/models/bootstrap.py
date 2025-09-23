# Import all concrete model architectures and their configs so that
# registration decorators execute at import time.

# Concrete model architectures
from .architectures import mlp  # noqa: F401

# Corresponding config modules
from .architectures.configs import mlp as mlp_config  # noqa: F401

