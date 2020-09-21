from .structures import *
from .geometry import *
from .stencils import *
from .topography import *

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
