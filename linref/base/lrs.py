from __future__ import annotations
import numpy as np
import pandas as pd
import geopandas as gpd
from rangel import RangeCollection


class LRS(object):
    """
    Linear Referencing System (LRS) class for managing core linear referencing 
    features and operations.
    """

    def __init__(self, df, keys, beg=None, end=None, geom=None, closed=None):
        """
        """
        self.df = df
        self.keys = keys
        self.beg = beg
        self.end = end
        self.geom = geom
        self.closed = closed

    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    @df.setter
    def df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame or subclass.")
        # Register the lrs accessor on the dataframe
        df.lrs = property(lambda self: self)
        self._df = df

    def get_group(self, key) -> pd.DataFrame:
        """
        """


class LRS_NumericalOperations(object):
    """
    Linear Referencing System (LRS) class for managing numerical event 
    operations within a linear referencing system.
    """

    def round(self, decimals=0):
        """
        Round the event measures to the specified number of decimal places.
        """
        pass

    def shift(self):
        """
        Shift the event measures by a specified offset.
        """
        pass

    def scale(self):
        """
        Scale the event measures by a specified factor.
        """
        pass


class LRS_TabularOperations(object):
    """
    Linear Referencing System (LRS) class for managing tabular event operations 
    within a linear referencing system.
    """

    def join(self):
        """
        Join the event measures to another table based on linear referencing 
        keys and measures.
        """
        pass

    def dissolve(self):
        """
        Dissolve adjacent events with the same linear referencing keys.
        """
        pass

    def union(self):
        """
        Unify overlapping events with the same linear referencing keys.
        """
        pass


class LRS_SpatialOperations(object):
    """
    Linear Referencing System (LRS) class for managing spatial event operations 
    within a linear referencing system.
    """

    @
    def project(self):
        """
        Project point geometries to the linear referencing system.
        """
        pass
