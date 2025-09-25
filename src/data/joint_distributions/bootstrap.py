# Import all concrete joint distributions and their configs so that
# registration decorators execute at import time.

# Concrete distributions
from . import gaussian  # noqa: F401
from . import hypercube  # noqa: F401
from . import mapped_joint_distribution  # noqa: F401
from . import noisy_distribution  # noqa: F401
from . import staircase  # noqa: F401

# Corresponding config modules
from .configs import gaussian as gaussian_config  # noqa: F401
from .configs import hypercube as hypercube_config  # noqa: F401
from .configs import mapped_joint_distribution as mapped_joint_distribution_config  # noqa: F401
from .configs import noisy_distribution as noisy_distribution_config  # noqa: F401
from .configs import staircase as staircase_config  # noqa: F401
