"""
pyClimAir - A Python package for easily study your climatological data.

This package helps to:
- Perform a quality control of your data: identify and remove suspicious data.
- Calculate climate normal values given a certain period.
- Perform a large variety of publication-quality data plotting and compare with the climatology normal.

All the tools included in pyClimAir are located within three subpackages:
- "clim": This subpackage includes all the functions specific for climatological data.
- "air": This subpackage includes all the functions specific for air quality data.
- "common": This subpackage includes all the functions common for all kinds of data.

"""

# ruff: noqa


from .pyclimair import *
import importlib as _importlib

_submodules = [
    "clim",
    "air",
    "common"
    ]


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"pyclimair.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'pyclimair' has no attribute '{name}'")
