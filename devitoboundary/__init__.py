from .symbolics import *  # noqa: F401
from .geometry import *  # noqa: F401
from .distance import *  # noqa: F401
from .stencils import *  # noqa: F401
# from .topography import *  # noqa: F401

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
