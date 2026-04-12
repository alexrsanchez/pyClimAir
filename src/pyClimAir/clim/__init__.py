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


from .pyclimair.clim._clim import compare_with_globaldataset

