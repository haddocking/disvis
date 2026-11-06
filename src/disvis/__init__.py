"""Visualise and quantify the accessible interaction space of complexes."""

from importlib.metadata import PackageNotFoundError, version

from .disvis import DisVis
from .pdb import PDB
from .volume import Volume

try:
    __version__ = version('disvis')
except PackageNotFoundError:
    # Not installed, e.g. extensions built in place in a source checkout.
    __version__ = '0+unknown'

__all__ = ['DisVis', 'PDB', 'Volume']
