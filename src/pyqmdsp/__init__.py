"""Direct NumPy interfaces to qm-dsp algorithms.

Reads: domain modules backed by the pinned native extension.
"""

from . import rhythm, spectral, statistics, utilities
from .__about__ import __version__
from ._native import engine_info
from .rhythm import beats

__all__ = [
    "__version__",
    "beats",
    "engine_info",
    "rhythm",
    "spectral",
    "statistics",
    "utilities",
]
