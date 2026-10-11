from pathlib import Path

try:
    # Written at build time by setuptools_scm (see [tool.setuptools_scm]).
    # Importing it costs ~0.1 ms, whereas resolving the version through
    # importlib.metadata costs ~39 ms per interpreter start -- a cost paid on
    # every single ORCA external-tool invocation.
    from oet._version import __version__
except ImportError:  # source tree that was never built
    __version__ = "unknown"

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent

# Assets (model/parameter files, etc.)
ASSETS_DIR = ROOT_DIR / "assets"
