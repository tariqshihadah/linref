from __future__ import annotations
import functools
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from linref.errors import GeometrySyncWarning, GeometrySyncError


def _method_deprecates_geometry(func) -> callable:
    """
    Decorator for LRS_Accessor methods that may de-synchronize the geometry
    of the underlying GeoDataFrame. Deals with this by raising an error,
    raising a warning, or dropping the geometry column, depending on the
    user's preference.

    Parameters
    ----------
    func : callable
        The LRS_Accessor method to decorate.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if the dataframe is spatial
        if self.is_spatial or self.is_spatial_m:
            # Check for geometry synchronization preference
            geometry_sync = kwargs.pop('geometry_sync', self.geometry_sync)
            # Identify action based on preference
            if geometry_sync == 'error':
                raise GeometrySyncError(
                    f"The `{func.__name__}` method may de-synchronize the "
                    "geometry of the underlying GeoDataFrame with event data. "
                    "To proceed anyway, set the `LRS.geometry_sync` attribute "
                    "to 'warn' or 'drop'."
                )
            elif geometry_sync == 'warn':
                warnings.warn(
                    f"The `{func.__name__}` method may de-synchronize the "
                    "geometry of the underlying GeoDataFrame with event data.",
                    GeometrySyncWarning
                )
            elif geometry_sync in ('none', 'ignore'):
                pass
            elif geometry_sync == 'drop':
                # Drop geometry column
                if self.is_spatial:
                    self.df = self.df.drop(columns=self.lrs.geom_col)
                if self.is_spatial_m:
                    self.df = self.df.drop(columns=self.lrs.geom_m_col)
        return func(self, *args, **kwargs)
    return wrapper