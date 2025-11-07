"""Top-level package for Docs2Synth.

Auto-initializes global configuration on import. The configuration will be
loaded from the following locations in order of precedence:
1) Path specified by the DOCS2SYNTH_CONFIG environment variable
2) ./config.yml in the current working directory
3) Built-in defaults
"""

from importlib import metadata

from .utils.config import get_config  # Auto-loads config on first call
from .utils.logging import get_logger

try:
    __version__: str = metadata.version(__name__)
except metadata.PackageNotFoundError:  # pragma: no cover
    # Package is not installed, default to dev version
    __version__ = "0.0.0.dev0"

logger = get_logger(__name__)

try:
    # Trigger config auto-load (safe: falls back to defaults on failure)
    get_config()
except Exception as _e:  # pragma: no cover
    logger.warning(f"Config auto-initialization failed: {_e}")

__all__ = ["__version__"]
