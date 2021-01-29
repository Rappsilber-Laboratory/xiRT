"""Init module for xiRT."""
import os


files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
