from __future__ import annotations
import pandas as pd
from linref.base.lrs import LRS


@pd.api.extensions.register_dataframe_accessor("lrs")
class LRSAccessor(object):
    """
    Linear Referencing System (LRS) accessor class for managing linear
    referencing features and operations.
    """

    def __init__(self, df):
        self._validate(df)
        self._df = df

    def __call__(self, keys, beg=None, end=None, closed=None):
        """
        Register a Linear Referencing System (LRS) on the root DataFrame.
        """
        return self.register(keys, beg, end, closed)

    @staticmethod
    def _validate(obj):
        """
        Validate the input dataframe.
        """
        pass
    
    @property
    def lrs(self):
        """
        Returns the base linear referencing system (LRS) object.
        """
        return self._lrs
    
    def register(self, keys, beg=None, end=None, closed=None):
        """
        Register a Linear Referencing System (LRS) on the root DataFrame.
        """
        # Instantiate the LRS object and log it to the accessor
        lrs = LRS(self._obj, keys, beg, end, closed)
        self._lrs = lrs
        return lrs
