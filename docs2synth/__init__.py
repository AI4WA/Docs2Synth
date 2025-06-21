"""Top-level package for Docs2Synth."""

from importlib import metadata

try:
    __version__: str = metadata.version(__name__)
except metadata.PackageNotFoundError:  # pragma: no cover
    # Package is not installed, default to dev version
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"] 