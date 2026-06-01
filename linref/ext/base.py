from __future__ import annotations
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import copy, hashlib
from shapely.errors import GeometryTypeError
from pandas.api.extensions import register_dataframe_accessor
from scipy import sparse as sp
from linref.utility.utility import label_list_or_none
from linref.events.base import EventsData
from linref.events.utility import _method_require
from linref.events import modify, relate, integration
from linref import geometry
from linref.errors import LRSConfigurationError, LRSCompatibilityError, GeometryTopologyError, GeometryMeasureError
from linref.ext.validation import _method_deprecates_geometry
from linref.ext.lrs import LRS
from linref.options import options, set_default_lrs

# Keys for storing accessor state in DataFrame.attrs, ensuring persistence
# across accessor re-instantiation (required for pandas >= 3.0 compatibility)
_LINREF_LRS_KEY = '_linref_lrs'
_LINREF_GEOMETRY_SYNC_KEY = '_linref_geometry_sync'


@register_dataframe_accessor("lr")
class LRS_Accessor(object):
    """
    Accessor for working with linear referencing systems (LRS) in GeoDataFrames.
    """

    def __init__(self, df) -> None:
        # Log dataframe
        self.df = df
        # Initialize LRS in attrs only if not already present (preserves
        # state across accessor re-instantiation in pandas >= 3.0)
        if _LINREF_LRS_KEY not in self._df.attrs:
            self.lrs = options.default_lrs.copy()
        if _LINREF_GEOMETRY_SYNC_KEY not in self._df.attrs:
            self.geometry_sync = options.default_geometry_sync

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self.is_lrs_set:
            # Prepare LRS information
            study = self.lrs.study(self._df)
            tags = []
            tags.append('GR' if study['keys']      ['valid'] else 'gr' if study['keys']      ['defined'] else '--')
            tags.append('LC' if study['location']  ['valid'] else 'lc' if study['location']  ['defined'] else '--')
            tags.append('LN' if study['linear']    ['valid'] else 'ln' if study['linear']    ['defined'] else '--')
            tags.append('SP' if study['geometry']  ['valid'] else 'sp' if study['geometry']  ['defined'] else '--')
            tags.append('SM' if study['geometry_m']['valid'] else 'sm' if study['geometry_m']['defined'] else '--')
            tags = ' '.join(tags)
            lrs_info = f"[{tags}] {str(self.lrs)}"
        else:
            lrs_info = "- No LRS set"
        return "LRS_Accessor with linear referencing system (LRS):\n" + lrs_info

    def __getitem__(self, obj) -> LRS_Accessor:
        """
        Activate an LRS by index or by setting it directly.
        """
        if isinstance(obj, LRS):
            obj = self.set_lrs(obj, activate=True, append=False, inplace=False)
            return obj
        else:
            raise TypeError("Invalid input type. Must provide an LRS object.")

    @property
    def df(self) -> pd.DataFrame:
        """
        Return the underlying DataFrame.
        """
        return self._df
    
    @df.setter
    def df(self, df) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input DataFrame must be of type `pandas.DataFrame`.")
        if isinstance(df.index, pd.MultiIndex):
            raise ValueError("MultiIndex DataFrames are not currently supported.")
        self._df = df

    @property
    def lrs_cols(self) -> list[str]:
        """
        Return a list of all LRS event-related columns in the dataframe, 
        including key, location, begin, and end columns as defined in the LRS.
        """
        cols = []
        if self.is_grouped:
            cols.extend(self.key_col)
        if self.is_located:
            cols.append(self.loc_col)
        if self.is_linear:
            cols.extend([self.beg_col, self.end_col])
        return cols
    
    @property
    def target_cols(self) -> list[str]:
        """
        Return a list of all LRS event and geometry-related columns in the
        dataframe, including key, location, begin, end, and geometry columns
        as defined in the LRS.
        """
        cols = []
        if self.is_grouped:
            cols.extend(self.key_col)
        if self.is_located:
            cols.append(self.loc_col)
        if self.is_linear:
            cols.extend([self.beg_col, self.end_col])
        if self.is_spatial:
            cols.append(self.geom_col)
        if self.is_spatial_m:
            cols.append(self.geom_m_col)
        return cols
    
    @property
    def other_cols(self) -> list[str]:
        """
        Return a list of all non-LRS event and geometry-related columns in the
        dataframe.
        """
        # Get all restricted columns
        remove = self.target_cols
        cols = self._df.columns.to_list()
        for col in remove:
            cols.remove(col)
        return cols

    @property
    def geometry_sync(self) -> str:
        """
        The geometry synchronization behavior for this DataFrame.
        State is stored in df.attrs for pandas >= 3.0 compatibility.
        """
        return self._df.attrs.get(_LINREF_GEOMETRY_SYNC_KEY, options.default_geometry_sync)

    @geometry_sync.setter
    def geometry_sync(self, value: str) -> None:
        valid_behaviors = ['none', 'warn', 'error', 'remove']
        if value not in valid_behaviors:
            raise ValueError(
                f"Invalid geometry synchronization behavior '{value}'. "
                f"Must be one of {valid_behaviors}."
            )
        self._df.attrs[_LINREF_GEOMETRY_SYNC_KEY] = value

    @property
    def lrs(self) -> LRS:
        """
        The LRS object currently set for the DataFrame.
        State is stored in df.attrs for pandas >= 3.0 compatibility.
        """
        return self._df.attrs.get(_LINREF_LRS_KEY)
    
    @lrs.setter
    def lrs(self, lrs) -> None:
        if lrs is not None and not isinstance(lrs, LRS):
            raise ValueError("Input LRS object must be of type `LRS`.")
        self._df.attrs[_LINREF_LRS_KEY] = lrs

    @property
    def is_lrs_set(self) -> bool:
        """
        Whether an LRS object is currently set for the DataFrame.
        """
        return self.lrs is not None
    
    @property
    def is_lrs_empty(self) -> bool:
        """
        Whether the currently set LRS object is empty (i.e., has no key, 
        location, begin, end, or geometry columns defined).
        """
        if not self.is_lrs_set:
            return True
        return self.lrs == LRS()

    @property
    def key_col(self) -> list[str]:
        """
        Return a list of all event key columns in the set LRS if defined,
        including the chain column when it exists in the DataFrame.
        """
        try:
            cols = list(self.lrs.key_col)
            # Dynamically include chain_col when it exists in the DataFrame
            if (self.lrs.chain_col is not None 
                    and self.lrs.chain_col not in cols
                    and self.lrs.chain_col in self._df.columns):
                cols.append(self.lrs.chain_col)
            return cols
        except AttributeError:
            return None

    @property
    def base_key_col(self) -> list[str]:
        """
        Return the base key columns from the LRS definition, excluding 
        the chain column. Used for operations that need to group by route 
        identifiers without chaining (e.g., computing chain indices).
        """
        try:
            return self.lrs.key_col
        except AttributeError:
            return None

    @property
    def loc_col(self) -> str:
        """
        Return the location column in the set LRS if defined.
        """
        try:
            return self.lrs.loc_col
        except AttributeError:
            return None

    @property
    def beg_col(self) -> str:
        """
        Return the begin column in the set LRS if defined.
        """
        try:
            return self.lrs.beg_col
        except AttributeError:
            return None

    @property
    def end_col(self) -> str:
        """
        Return the end column in the set LRS if defined.
        """
        try:
            return self.lrs.end_col
        except AttributeError:
            return None
    
    @property
    def geom_col(self) -> str:
        """
        Return the geometry column in the set LRS if defined.
        """
        try:
            return self.lrs.geom_col
        except AttributeError:
            return None

    @property
    def geom_m_col(self) -> str:
        """
        Return the M-enabled geometry column in the set LRS if defined.
        """
        try:
            return self.lrs.geom_m_col
        except AttributeError:
            return None

    @property
    def closed(self) -> str:
        """
        Return the closure type in the set LRS if defined.
        """
        try:
            return self.lrs.closed
        except AttributeError:
            return None

    @property
    def index(self) -> np.ndarray:
        """
        Return the index values of the dataframe. Equivalent to 
        self.df.index.values.
        """
        return self._df.index.values
    
    @property
    def keys(self) -> np.ndarray:
        """
        Return the event keys from the dataframe as an array.
        """
        return self.get_keys(require=False)
        
    @property
    def locs(self) -> np.ndarray:
        """
        Return the location values from the dataframe as an array. If no
        location column is defined in the LRS, returns None.
        """
        # Select data from the dataframe if locations are present
        col = self.loc_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @locs.setter
    def locs(self, values) -> None:
        if self.loc_col is None:
            raise ValueError("No locations column in the LRS.")
        # Set location values in the DataFrame
        col = self.loc_col
        self._df[col] = values
                
    @property
    def begs(self) -> np.ndarray:
        """
        Return the begin location values from the dataframe as an array. If no
        begin column is defined in the LRS, returns None.
        """
        # Select data from the dataframe if begin locations are present
        col = self.beg_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @begs.setter
    def begs(self, values) -> None:
        if self.beg_col is None:
            raise ValueError("No begins column in the LRS.")
        # Set begin location values in the DataFrame
        col = self.beg_col
        self._df[col] = values
        
    @property
    def ends(self) -> np.ndarray:
        """
        Return the end location values from the dataframe as an array. If no
        end column is defined in the LRS, returns None.
        """
        # Select data from the dataframe if end locations are present
        col = self.end_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @ends.setter
    def ends(self, values) -> None:
        if self.end_col is None:
            raise ValueError("No ends column in the LRS.")
        # Set end location values in the DataFrame
        col = self.end_col
        self._df[col] = values

    @property
    def event_lengths(self) -> np.ndarray:
        """
        Return the event lengths from the dataframe as an array. If no
        begin and end columns are defined in the LRS, returns None.
        """
        if not self.is_linear:
            return None
        return self.ends - self.begs

    @property
    def geoms(self) -> np.ndarray:
        """
        Return the geometry values from the dataframe as an array. If no
        geometry column is defined in the LRS, returns None.
        """
        # Select data from the dataframe if geometry is present
        col = self.geom_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @property
    def geoms_m(self) -> np.ndarray:
        """
        Return the M-enabled geometry values from the dataframe as an array. If
        no M-enabled geometry column is defined in the LRS, returns None.
        """
        # Select data from the dataframe if geometry is present
        col = self.geom_m_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @property
    def geoms_m_reduced(self) -> np.ndarray:
        """
        Extract and return the shapely geometries from the M-enabled geometry
        column in the dataframe.
        """
        # Select M-enabled geometries and extract shapely geometries
        geoms_m = self.geoms_m
        if geoms_m is None:
            return None
        return np.array([geom_m.geom for geom_m in geoms_m])

    @property
    def is_grouped(self) -> bool:
        """
        Return whether the active LRS is grouped and the key columns are 
        present in the dataframe.
        """
        if len(self.key_col) == 0:
            return False
        else:
            # Check for presence of key columns in the dataframe
            return all([key in self._df.columns for key in self.key_col])
    
    @property
    def missing_key_cols(self) -> list[str]:
        """
        Return a list of key columns defined in the LRS but missing from the
        DataFrame. Returns an empty list if no keys are defined or all keys
        are present.
        """
        if not self.is_lrs_grouped:
            return []
        return [k for k in self.key_col if k not in self._df.columns]

    @property
    def is_linear(self) -> bool:
        """
        Return whether the active LRS is linear and the begin and end columns
        are present in the dataframe.
        """
        if not self.lrs.is_linear:
            return False
        else:
            # Check for presence of begin and end columns in the dataframe
            return all([col in self._df.columns for col in [self.beg_col, self.end_col]])
    
    @property
    def is_point(self) -> bool:
        """
        Return whether the active LRS is point-based and the location column is
        present in the dataframe.
        """
        if self.is_linear or not self.lrs.is_located:
            return False
        # Check for presence of location column in the dataframe
        return self.loc_col in self._df.columns
    
    @property
    def is_located(self) -> bool:
        """
        Return whether the active LRS is located and the location column is 
        present in the dataframe.
        """
        if not self.lrs.is_located:
            return False
        else:
            # Check for presence of location column in the dataframe
            return self.loc_col in self._df.columns
    
    @property
    def is_spatial(self) -> bool:
        """
        Return whether the active LRS is spatial and the geometry column is 
        present in the dataframe.
        """
        if not self.lrs.is_spatial:
            return False
        else:
            # Check for presence of geometry column in the dataframe
            return self.geom_col in self._df.columns
        
    @property
    def is_spatial_m(self) -> bool:
        """
        Return whether the active LRS is spatial and the geometry column is 
        present in the dataframe.
        """
        if not self.lrs.is_spatial_m:
            return False
        else:
            # Check for presence of geometry column in the dataframe
            return self.geom_m_col in self._df.columns
        
    @property
    def is_lrs_grouped(self) -> bool:
        """
        Return whether the active LRS is grouped, regardless of presence of key
        columns in the dataframe.
        """
        return self.lrs.is_grouped
    
    @property
    def is_lrs_linear(self) -> bool:
        """
        Return whether the active LRS is linear, regardless of presence of 
        begin and end columns in the dataframe.
        """
        return self.lrs.is_linear
    
    @property
    def is_lrs_point(self) -> bool:
        """
        Return whether the active LRS is point-based, regardless of presence of 
        location column in the dataframe.
        """
        return self.lrs.is_point

    @property
    def is_lrs_located(self) -> bool:
        """
        Return whether the active LRS is located, regardless of presence of 
        location column in the dataframe.
        """
        return self.lrs.is_located
    
    @property
    def is_lrs_spatial(self) -> bool:
        """
        Return whether the active LRS is spatial, regardless of presence of 
        geometry column in the dataframe.
        """
        return self.lrs.is_spatial
    
    @property
    def is_lrs_spatial_m(self) -> bool:
        """
        Return whether the active LRS is spatial with M values, regardless of 
        presence of geometry column in the dataframe.
        """
        return self.lrs.is_spatial_m

    @property
    def is_lrs_chained(self) -> bool:
        """
        Return whether the active LRS has a chain column defined, regardless 
        of presence of the chain column in the dataframe.
        """
        return self.lrs.is_chained

    @property
    def is_chained(self) -> bool:
        """
        Return whether the active LRS has a chain column defined and the 
        chain column is present in the dataframe.
        """
        if not self.lrs.is_chained:
            return False
        return self.lrs.chain_col in self._df.columns

    @property
    def is_geometrically_contiguous(self) -> bool:
        """
        Return whether linear geometries in the dataframe when grouped by LRS 
        keys are geometrically contiguous, producing single continuous 
        geometries without gaps when dissolved and merged.
        """
        if not self.is_linear or not self.is_spatial:
            return False
        # Check for contiguity by dissolving and counting geometries
        counts = self.df.dissolve(by=self.key_col).line_merge().count_geometries().values
        return all(counts == 1)
    
    @property
    def is_contiguous(self) -> bool:
        """
        Return whether linear geometries in the dataframe when grouped by LRS 
        keys are contiguous, producing continuous geometries without disjoints 
        such as overlaps or gaps.
        """
        if not self.is_linear or not self.is_spatial:
            return False
        # Check for contiguity by generating chains and checking for multiples
        chains = self.get_chains(include_chain=True)
        return all(chains == 0)

    @property
    def is_disjointed(self) -> bool:
        """
        Return whether linear geometries in the dataframe when grouped by LRS 
        keys contain disjointed segments (i.e., are not contiguous). Equivalent 
        to not self.is_contiguous.
        """
        return not self.is_contiguous
    
    @property
    def valid_events(self) -> pd.Series:
        """
        Return a boolean Series indicating which events in the dataframe are
        valid according to the active LRS, e.g., have populated key columns 
        and event bounds.
        """
        # Identify valid events
        valid = pd.Series(np.ones(self._df.shape[0], dtype=bool), index=self.index)
        # Check key columns
        for col in self.key_col:
            valid &= self._df[col].notna()
        # Check event bounds
        if self.is_located:
            valid &= self._df[self.loc_col].notna()
        if self.is_linear:
            valid &= self._df[self.beg_col].notna()
            valid &= self._df[self.end_col].notna()
        return valid
    
    @property
    def invalid_events(self) -> pd.Series:
        """
        Return a boolean Series indicating which events in the dataframe are
        invalid according to the active LRS, e.g., have missing key columns 
        or event bounds.
        """
        return ~self.valid_events
    
    def drop_invalid_events(self) -> pd.DataFrame | None:
        """
        Drop invalid events from the dataframe according to the active LRS.
        Invalid events are those with missing data in key columns or event 
        bounds.

        Returns
        -------
        df : DataFrame
            A copy of the DataFrame with invalid events removed.
        """
        # Identify valid events
        valid = self.valid_events
        return self._df[valid].copy()

    def study(self) -> dict:
        """
        Evaluate which LRS properties are satisfied by the dataframe.
        """
        return self.lrs.study(self._df)

    @property
    def events(self) -> EventsData:
        """
        Return the events object for the active LRS.
        """
        # Create the events object
        return self.get_events(allow_undefined_events=True)
    
    def _validate_other_dataframe_lrs(self, other) -> None:
        """
        Validate that another dataframe object is compatible with the 
        current object for relational operations. Can also take input of 
        LRS Accessor objects, returning the underlying dataframe.
        """
        if isinstance(other, LRS_Accessor):
            other = other.df
        if not isinstance(other, pd.DataFrame):
            raise TypeError("Input object must be of type `pd.DataFrame`.")
        if not other.lr.is_lrs_set:
            raise LRSCompatibilityError("Input DataFrame has no LRS set.")
        if not self.is_lrs_set:
            raise LRSConfigurationError("Current DataFrame has no LRS set.")
        if self.is_lrs_grouped:
            missing = self.missing_key_cols
            if missing:
                raise LRSConfigurationError(
                    f"Key columns {missing} are defined in the LRS but not "
                    f"found in the primary DataFrame."
                )
            missing_other = other.lr.missing_key_cols
            if missing_other:
                raise LRSConfigurationError(
                    f"Key columns {missing_other} are defined in the LRS of "
                    f"the other DataFrame but not found in it."
                )
            if not other.lr.is_grouped:
                raise LRSCompatibilityError("LRS of other DataFrame is not grouped.")
            if len(self.lrs.key_col) != len(other.lr.lrs.key_col):
                raise LRSCompatibilityError(
                    "LRS of other DataFrame has a different number of key "
                    f"columns. Received {len(other.lr.lrs.key_col)} "
                    f"columns, but expected {len(self.lrs.key_col)}."
                )
            
            if self.events.groups.dtype != other.lr.events.groups.dtype:
                raise LRSCompatibilityError("LRS of other DataFrame has different key column data types.")
        return other
    
    def check_exact_geoms(self, if_missing: bool=True) -> np.ndarray:
        """
        Check if geometries in both the geometry and geometry_m columns are
        exactly the same, returning an array of boolean values.

        Parameters
        ----------
        if_missing : bool, default True
            The value to return for rows where either geometry column is missing.
        """
        # Check for presence of geometry columns
        if not self.is_spatial or not self.is_spatial_m:
            return np.array([if_missing] * len(self._df))
        # Compare geometries
        return np.array([shapely.equals_exact(geom, geom_m.geom, tolerance=0)
                         for geom, geom_m in zip(self.geoms, self.geoms_m)])
    
    def get_keys(self, col=None, require=True) -> np.ndarray:
        """
        Return the event keys from the dataframe as an array.

        Parameters
        ----------
        col : str, optional
            The column name to use for extracting keys. If None, uses the
            default key columns in the LRS.
        require : bool, default True
            Whether to raise an error if the key column is not found.

        Returns
        -------
        np.ndarray
            An array of event keys.
        """
        # Resolve columns from LRS if not provided
        from_lrs = col is None
        if from_lrs:
            col = self.key_col
        if not col:
            return None
        # Check for missing columns
        col_list = [col] if isinstance(col, str) else col
        missing = [c for c in col_list if c not in self._df.columns]
        if missing:
            if not require:
                return None
            if from_lrs:
                raise LRSConfigurationError(
                    f"LRS key columns {missing} are not found in the "
                    f"DataFrame."
                )
            raise LRSConfigurationError(
                f"Requested key columns {missing} are not found in the "
                f"DataFrame."
            )
        return self._df[col].to_records(index=False)
        
    def get_events(self, key_col=None, require=True, allow_undefined_events=False) -> EventsData:
        """
        Return the events object for the active LRS.

        Parameters
        ----------
        key_col : str, optional
            The column name to use for extracting keys. If None, uses the
            default key columns in the LRS.
        require : bool, default True
            Whether to raise an error if the key column is not found.
        allow_undefined_events : bool, default False
            Whether to allow undefined events when creating the EventsData 
            object. If False, will raise an error if undefined events are 
            present. Undefined events are those with no defined location, 
            begin, or end values.
        """
        # Get keys if needed
        if key_col is None:
            keys = self.keys
        else:
            keys = self.get_keys(col=key_col, require=require)
        # Create the events object
        return EventsData(
            index=self.index,
            groups=keys,
            locs=self.locs if self.loc_col else None,
            begs=self.begs if self.beg_col else None,
            ends=self.ends if self.end_col else None,
            closed=self.closed if self.is_linear else None,
            force_monotonic=False,
            allow_undefined_events=allow_undefined_events
        )
    
    def copy(self, deep: bool = False) -> LRS_Accessor:
        """
        Create an exact copy of the object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    
    def copy_df(self) -> pd.DataFrame:
        """
        Create a copy of the underlying DataFrame, preserving LRS settings.
        """
        df_copy = self.df.copy()
        df_copy.lr.set_lrs(self.lrs, inplace=True)
        return df_copy
    
    def lrs_like(self, other: pd.DataFrame | LRS_Accessor, inplace: bool = False) -> pd.DataFrame | None:
        """
        Assign the LRS settings of another DataFrame to the current DataFrame.

        Parameters
        ----------
        other : DataFrame or LRS_Accessor
            The DataFrame or LRS_Accessor to copy LRS settings from.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.

        Returns
        -------
        df : DataFrame
            A copy of the current DataFrame with LRS settings copied from the
            input DataFrame.
        """
        # Copy DataFrame
        df = self.df if inplace else self.df.copy()
        # Copy LRS settings
        if isinstance(other, pd.DataFrame):
            other = other.lr
        df.lr.set_lrs(other.lrs, inplace=True)
        return None if inplace else df

    def set_lrs(self, lrs: LRS | None = None, inplace: bool = False, **kwargs) -> pd.DataFrame | None:
        """
        Set the LRS object for the DataFrame.

        Parameters
        ----------
        lrs : LRS , default None
            The LRS object to set for the DataFrame.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        **kwargs
            Additional parameters to set for the LRS object if `lrs` is None.
        """
        # Validate LRS object type
        if lrs is None:
            lrs = LRS(**kwargs)
        elif not isinstance(lrs, LRS):
            raise ValueError("Input LRS object must be of type `LRS`.")

        # Append or replace LRS objects
        df = self.df if inplace else self.df.copy()
        df.lr.lrs = lrs
        return None if inplace else df

    def modify_lrs(self, inplace: bool = False, **kwargs) -> pd.DataFrame | None:
        """
        Modify the parameters of the existing LRS object in the DataFrame.

        Parameters
        ----------
        **kwargs
            The LRS parameters from an LRS object to modify.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place. Note, this
            does not modify the existing LRS object in place, but creates a
            modified copy of the existing LRS object and sets it to the 
            DataFrame. To modify the LRS object itself, access the LRS object
            directly via `df.lr.lrs`.
        """
        # Retrieve and modify LRS
        df = self.df if inplace else self.df.copy()
        lrs = self.lrs.copy(deep=True)
        lrs.set_params(inplace=True, **kwargs)
        # Set the modified LRS
        df.lr.lrs = lrs
        return None if inplace else df

    def add_key(self, key_col: str | list[str], inplace: bool = False) -> pd.DataFrame | None:
        """
        Add one or more key columns to an existing LRS object in the DataFrame.

        Parameters
        ----------
        key_col : label or array-like
            The key column or array-like of key columns to add.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        """
        # Add key columns
        df = self.df if inplace else self.df.copy()
        lrs = self.lrs.copy(deep=True)
        lrs.add_key(key_col, inplace=True)
        # Set the modified LRS
        df.lr.lrs = lrs
        return None if inplace else df

    def remove_key(self, key_col: str | list[str], errors: str = 'raise', inplace: bool = False) -> None:
        """
        Remove one or more key columns from an existing LRS object in the DataFrame.

        Parameters
        ----------
        key_col : label or array-like
            The key column or array-like of key columns to remove.
        errors : {'ignore', 'raise'}, default 'raise'
            Whether to raise an error or ignore missing keys.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        """
        # Add key columns
        df = self.df if inplace else self.df.copy()
        lrs = self.lrs.copy(deep=True)
        lrs.remove_key(key_col, errors=errors, inplace=True)
        # Set the modified LRS
        df.lr.lrs = lrs
        return None if inplace else df

    def clear_lrs(self, inplace: bool = False) -> None:
        """
        Clear the LRS object from the DataFrame. Applies a new empty LRS object
        equivalent to LRS().

        Parameters
        ----------
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        """
        # Clear LRS object
        df = self.df if inplace else self.df.copy()
        df.lr.lrs = LRS()
        return None if inplace else df
    
    @_method_require(is_linear=True)
    def set_monotonic(self, reverse_geom: bool = True, inplace: bool = False) -> pd.DataFrame | None:
        """
        Ensure that linear events in the DataFrame are monotonic according to
        the set LRS, reordering begin and end values as needed. Optionally 
        reverses associated geometries for events whose bounds are swapped.

        Parameters
        ----------
        reverse_geom : bool, default True
            Whether to reverse geometries for events whose begin and end 
            values are swapped to enforce monotonicity. When True, shapely 
            geometries in the geometry column will be reversed for affected 
            events, and M-enabled geometries will be reversed if supported. 
            When False, only begin and end values are swapped and geometries 
            are left unchanged.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        """
        # Identify non-monotonic events before swapping
        mask = self.begs > self.ends
        # Create events object and enforce monotonicity
        events = self.get_events(allow_undefined_events=True)
        events.set_monotonic(inplace=True)
        # Apply changes to the DataFrame
        df = self.df if inplace else self.copy_df()
        if self.is_located:
            df[self.loc_col] = events.locs
        if self.is_linear:
            df[self.beg_col] = events.begs
            df[self.end_col] = events.ends
        # Reverse geometries for non-monotonic events if requested
        if reverse_geom and np.any(mask):
            # Reverse shapely geometries
            if self.is_spatial:
                geoms = df[self.geom_col].values.copy()
                geoms[mask] = shapely.reverse(geoms[mask])
                df[self.geom_col] = geoms
            # Reverse M-enabled geometries
            if self.is_spatial_m:
                # NOTE: Reversing M-enabled geometries is currently not 
                # supported as the LineStringM class does not allow the 
                # creation of reversed geometries. This is a known limitation 
                # and will be addressed in a future release or with improved
                # support for M-enabled geometries in shapely.
                geoms_m = df[self.geom_m_col].values
                for i in np.where(mask)[0]:
                    try:
                        geoms_m[i].reverse(inplace=True)
                    except NotImplementedError:
                        raise NotImplementedError(
                            "Cannot reverse M-enabled geometries during "
                            "monotonic enforcement because LineStringM does "
                            "not currently support reversing. Remove the "
                            "M-enabled geometry column before calling "
                            "set_monotonic, or set reverse_geom=False to "
                            "skip geometry reversal."
                        )
        return None if inplace else df

    def build_geom_m(self) -> np.ndarray:
        """
        Build a list of geometry_m objects based on the begin and end 
        locations of the LRS.

        Returns
        -------
        geoms_m : np.ndarray
            An array of geometry_m objects.
        """
        # Check for presence of geometries
        if not self.lrs.is_spatial:
            raise ValueError("No geometry column in the LRS.")
        if not self.is_spatial:
            raise ValueError("LRS geometry column not present in the DataFrame.")
        def _upgrade_geom(geom, beg, end):
            geom_m = geometry.LineStringM(geom)
            geom_m.set_m_from_bounds(beg=beg, end=end, inplace=True)
            return geom_m
        # Cast linear geometries to LineStringM
        geoms_m = list(map(_upgrade_geom, self.geoms, self.begs, self.ends))
        return np.array(geoms_m)

    def add_geom_m(self, name: str = 'geometry_m', inplace: bool = False) -> pd.DataFrame | None:
        """
        Add an M-enabled geometry column to the DataFrame based on the begin 
        and end values of the LRS.

        Parameters
        ----------
        name : str, default 'geometry_m'
            The name of the geometry column to return.
        inplace : bool, default False
            Whether to apply changes to the dataframe in place.
        """
        # Cast linear geometries to LineStringM
        geoms_m = self.build_geom_m()
        # Apply changes to the DataFrame
        df = self.df if inplace else self.copy_df()
        df[name] = geoms_m
        # Update LRS if needed
        if self.lrs.geom_m_col != name:
            new_lrs = self.lrs.copy(deep=True)
            new_lrs.geom_m_col = name
            df.lr.set_lrs(new_lrs, inplace=True)
        return None if inplace else df

    @_method_require(is_grouped=True)
    def iter_groups(self) -> Iterator[tuple]:
        """
        Iterate over unique event groups in the dataframe based on the 
        LRS key columns.

        Yields
        ------
        tuple
            A tuple containing the group value and a DataFrame of events
            corresponding to that group.
        """
        for group, index in self.events.iter_group_indices():
            yield group, self.df.loc[index]

    @_method_require(is_grouped=True)
    def group_counts(self) -> pd.Series:
        """
        Return a Series of counts of events per group in the dataframe based
        on the LRS key columns.
        """
        groups, counts = self.events.group_counts()
        return pd.Series(data=counts, index=groups)

    def sort_standard(
        self,
        return_index: bool = False,
        inplace: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray] | None:
        """
        Sort the dataframe by the LRS columns in the standard order
        of 'groups', 'begs', 'ends' for linear events and 'groups', 'locs' 
        for point events.

        Parameters
        ----------
        return_index : bool, default False
            Whether to return an array of the indices which represent the 
            performed sort in addition to the sorted events.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        # Get sorter
        sorter = self.events.sort_standard(return_index=True)[1]
        # Apply changes to the DataFrame
        df = self.df.iloc[sorter]
        if inplace:
            self._df = df
            return
        else:
            return (df, sorter) if return_index else df
        
    @_method_require(is_grouped=True)
    def get_group(self, group: str | list) -> pd.DataFrame:
        """
        Retrieve a subset of the dataframe corresponding to a specific group
        based on the LRS key columns.

        Parameters
        ----------
        group : group value or array-like of group values
            The group value or array-like of group values to retrieve.

        Returns
        -------
        group_df : pd.DataFrame
            A subset of the dataframe corresponding to the specified group.
        """
        # Get group indices
        try:
            index = self.events.select_group(group).index
        except:
            raise KeyError(f"Group {group} not found in the DataFrame.")
        # Return subset of dataframe
        return self.df.loc[index]
        
    @_method_require(is_linear=True, is_spatial=True)
    def get_chains(self, name: str | None = None, enforce_m: bool = True, include_chain: bool = False) -> pd.Series:
        """
        Identify the chain indices for each event in the dataframe based on 
        contiguous linear geometries within each group.

        Parameters
        ----------
        name : str, optional
            The name of the chain index column to return. If None, uses the
            chain column name from the LRS if defined, or 'chain' otherwise.
        enforce_m : bool, default True
            Whether to require the use of M-enabled geometries for chaining.
            If True, an error will be raised if M-enabled geometries are not 
            present in the dataframe.
        include_chain : bool, default False
            Whether to include the existing chain column in the grouping keys
            when computing chains. When True and the chain column is defined 
            in the LRS and present in the DataFrame, groups by full keys 
            (including chain) to detect sub-chains within existing chain 
            groups. When False, groups by base keys only (excluding the chain 
            column), computing chains from scratch.

        Returns
        -------
        chains : pd.Series
            A series of chain indices for each event in the dataframe.
        """
        # Resolve chain column name
        if name is None:
            name = self.lrs.chain_col or 'chain'
        # Validate presence of M-enabled geometries
        if enforce_m and not self.is_spatial_m:
            raise ValueError(
                "M-enabled geometries are required for chaining but are not "
                "present in the DataFrame. Use `add_geom_m` to add M-enabled "
                "geometries or set `enforce_m` to False."
            )
        # Determine grouping keys
        if (include_chain 
                and self.lrs.chain_col is not None 
                and self.lrs.chain_col in self._df.columns):
            base_keys = list(self.key_col)
        else:
            base_keys = list(self.base_key_col)
        # Iterate over groups
        index = []
        chains = []
        if base_keys:
            # Get events using keys, excluding chain column if in keys
            events = self.get_events(key_col=base_keys)
            for group, idx in events.iter_group_indices():
                df = self.df.loc[idx]
                if enforce_m:
                    chains_i = geometry.get_linestring_chains(df[self.geom_m_col].values)
                else:
                    chains_i = geometry.get_linestring_chains(df[self.geom_col].values)
                chains.append(chains_i)
                index.append(df.index.values)
        else:
            # No base keys — treat entire dataframe as one group
            if enforce_m:
                chains_i = geometry.get_linestring_chains(self.df[self.geom_m_col].values)
            else:
                chains_i = geometry.get_linestring_chains(self.df[self.geom_col].values)
            chains.append(chains_i)
            index.append(self.df.index.values)
        # Return series
        chains = pd.Series(
            np.concatenate(chains),
            index=np.concatenate(index),
            name=name,
            dtype=float
        )
        return chains.reindex_like(self.df)
    
    @_method_require(is_linear=True, is_spatial=True)
    def add_chaining(self, name: str | None = None, inplace: bool = False, replace: bool = False, enforce_m: bool = True) -> pd.DataFrame | None:
        """
        Add chain indices to the dataframe based on contiguous linear 
        geometries within each group, adding a new column to the dataframe
        and setting the chain column on the LRS.

        Parameters
        ----------
        name : str, optional
            The name of the chain index column to add. If None, uses the
            chain column name from the LRS if defined, or 'chain' otherwise.
            If this column already exists, it will be replaced if ``replace``
            is True.
        inplace : bool, default False
            Whether to apply changes to the dataframe in place.
        replace : bool, default False
            Whether to replace an existing column with the same name in the
            dataframe. If False, an error will be raised if the column already
            exists.
        enforce_m : bool, default True
            Whether to require the use of M-enabled geometries for chaining.
            If True, an error will be raised if M-enabled geometries are not 
            present in the dataframe.
        """
        # Resolve chain column name
        if name is None:
            name = self.lrs.chain_col or 'chain'
        # Validate column name
        if name in self.df and not replace:
            raise ValueError(
                f"Column name '{name}' is already in use in the DataFrame."
            )
        # Prepare chain data using base keys (without chain column)
        df = self.df if inplace else self.copy_df()
        chains = df.lr.get_chains(name=name, enforce_m=enforce_m)
        
        # Prepare updated LRS with chain_col set
        new_lrs = df.lr.lrs.copy(deep=True)
        new_lrs.chain_col = name
        if name in new_lrs.key_col:
            new_lrs.remove_key(name, inplace=True)
        
        # Apply changes to the DataFrame
        df[name] = chains
        if new_lrs != df.lr.lrs:
            df.lr.set_lrs(new_lrs, inplace=True)
        return None if inplace else df
    
    @_method_require(is_point=True)
    def point_to_linear(
        self,
        beg_col: str | None = None,
        end_col: str | None = None,
        replace: bool = False,
        drop_loc: bool = False,
        inplace: bool = False
    ) -> pd.DataFrame | None:
        """
        Convert point-based locations in the dataframe to linear begin and 
        end locations, adding new begin and end columns to the dataframe.

        Parameters
        ----------
        beg_col : str, optional
            The name of the begin location column to add. If None, uses the
            existing begin column name in the LRS, or 'beg' if no begin column
            is defined.
        end_col : str, optional
            The name of the end location column to add. If None, uses the
            existing end column name in the LRS, or 'end' if no end column is
            defined.
        replace : bool, default False
            Whether to replace existing begin or end columns in the dataframe. 
            If False, an error will be raised if the columns already exist.
        drop_loc : bool, default False
            Whether to drop the original location column from the dataframe
            after conversion.
        inplace : bool, default False
            Whether to apply changes to the dataframe in place.

        Returns
        -------
        df : DataFrame
            A copy of the current DataFrame with begin and end location columns 
            added.
        """
        # Validate column names
        if beg_col is None:
            if self.beg_col is not None:
                beg_col = self.beg_col
            else:
                beg_col = 'beg'
        if end_col is None:
            if self.end_col is not None:
                end_col = self.end_col
            else:
                end_col = 'end'
        for col_name in [beg_col, end_col]:
            if col_name in self.df and not replace:
                raise ValueError(
                    f"Column name '{col_name}' is already in use in the "
                    "DataFrame."
                )
        # Apply changes to the DataFrame
        df = self.df if inplace else self.copy_df()
        locs = df[self.loc_col].values
        df[beg_col] = locs
        df[end_col] = locs
        if drop_loc:
            df = df.drop(columns=[self.loc_col])
        # Update LRS if needed
        if beg_col != self.beg_col or end_col != self.end_col:
            new_lrs = df.lr.lrs.copy(deep=True)
            new_lrs.beg_col = beg_col
            new_lrs.end_col = end_col
            df.lr.set_lrs(new_lrs, inplace=True)
        return None if inplace else df
    
    @_method_require(is_spatial=True)
    def generate_linear_events(
        self,
        beg_col: str | None = None,
        end_col: str | None = None,
        chain_col: str | None = None,
        geom_m_col: str | None = None,
        add_chain: bool = True,
        add_geom_m: bool = True,
        scale: float = 1.0,
        decimals: int | None = None,
        allow_disjoint: bool = True,
        inplace: bool = False,
        replace: bool = False
    ) -> pd.DataFrame | None:
        """
        Add begin and end location columns to the dataframe based on the 
        lengths of contiguous linear geometries within each group, adding new
        columns to the dataframe for begin and end locations.

        Parameters
        ----------
        beg_col : str, optional
            The name of the begin location column to add. If None, uses the
            existing begin column name in the LRS, or 'beg' if no begin column
            is defined.
        end_col : str, optional
            The name of the end location column to add. If None, uses the
            existing end column name in the LRS, or 'end' if no end column is
            defined.
        chain_col : str, optional
            The name of the chain index column to add. If None, uses the
            chain column name from the LRS if defined, or 'chain' otherwise.
            Only used if `add_chain` is True.
        geom_m_col : str, optional
            The name of the M-enabled geometry column to add. If None, uses the
            existing geometry_m column name in the LRS, or 'geometry_m' if no
            geometry_m column is defined. Only used if `add_geom_m` is True.
        add_chain : bool, default True
            Whether to add a chain index column to the dataframe.
        add_geom_m : bool, default True
            Whether to add an M-enabled geometry column to the dataframe.
        scale : float, default 1.0
            A scale factor to apply to computed lengths when generating begin
            and end locations. This can be used to convert units from the 
            dataframe's CRS units to desired linear units, e.g., from feet to
            miles using a scale factor of 1/5280.
        decimals : int, optional
            The number of decimal places to round computed lengths to when
            generating begin and end locations. If None, no rounding is applied.
        allow_disjoint : bool, default True
            Whether to allow disjoint geometries within each group when
            generating begin and end locations. If False, an error will be
            raised if geometries within a group are not contiguous. If True,
            disjoint geometries will be treated as separate chains.
        inplace : bool, default False
            Whether to apply changes to the dataframe in place.
        """
        # Validate column names
        if beg_col is None:
            if self.beg_col is not None:
                beg_col = self.beg_col
            else:
                beg_col = 'beg'
        if end_col is None:
            if self.end_col is not None:
                end_col = self.end_col
            else:
                end_col = 'end'
        if add_chain:
            if chain_col is None:
                chain_col = self.lrs.chain_col or 'chain'
        else:
            if not chain_col is None:
                raise ValueError(
                    "Cannot specify `chain_col` when `add_chain` is False."
                )
        if add_geom_m:
            if geom_m_col is None:
                if self.geom_m_col is not None:
                    geom_m_col = self.geom_m_col
                else:
                    geom_m_col = 'geometry_m'
        else:
            if not geom_m_col is None:
                raise ValueError(
                    "Cannot specify `geom_m_col` when `add_geom_m` is False."
                )
        for col_name in [beg_col, end_col, geom_m_col]:
            if col_name in self.df and not replace:
                raise ValueError(
                    f"Column name '{col_name}' is already in use in the "
                    "DataFrame."
                )
        
        # Iterate over groups using base keys (without chain column)
        index  = []
        chains = []
        begs   = []
        ends   = []
        for group, df in self.remove_key(chain_col, errors='ignore').lr.iter_groups():
            # Get group geometries
            geoms = df[self.geom_col].values
            # Get chain indices
            try:
                _, orders_i, chains_i = geometry.line_merge_m(
                    geoms,
                    allow_multiple=allow_disjoint,
                    allow_mismatch=False,
                    return_orders=True,
                    return_chains=True,
                    squeeze=False,
                    cast_geom=True
                )
            except GeometryTopologyError:
                raise GeometryTopologyError(
                    f"Unable to merge geometries in group {group}. Ensure "
                    "geometries are contiguous and non-overlapping within "
                    "each group. If non-contiguous geometries are expected, "
                    "add chaining first using `add_chaining`, or set "
                    "`allow_disjoint=True`."
                )
            # Compute lengths of geometries
            try:
                lengths = np.array([geom.length for geom in geoms]) * scale
                if decimals is not None:
                    lengths = np.round(lengths, decimals=decimals)
            except:
                raise ValueError(
                    f"Unable to compute lengths of geometries in group "
                    f"{group}."
                )
            # Sort lengths by order
            sorter = np.argsort(orders_i)
            # Compute cumulative lengths
            cum_lengths = np.cumsum(lengths[orders_i])
            # Compute begin and end locations based on cumulative lengths
            begs_i = np.append(0, cum_lengths[:-1])
            ends_i = cum_lengths
            # Store results in original order
            begs.append(begs_i[sorter])
            ends.append(ends_i[sorter])
            chains.append(chains_i)
            index.append(df.index.values)
        
        # Concatenate results and apply to DataFrame
        index = np.concatenate(index)
        df = self.df if inplace else self.copy_df()
        df[beg_col] = pd.Series(np.concatenate(begs), index=index)
        df[end_col] = pd.Series(np.concatenate(ends), index=index)
        if add_chain:
            df[chain_col] = pd.Series(
                np.concatenate(chains), index=index, dtype=float
            )
        # Update LRS
        new_lrs = df.lr.lrs.copy(deep=True)
        new_lrs.beg_col = beg_col
        new_lrs.end_col = end_col
        if add_chain:
            new_lrs.chain_col = chain_col
            if chain_col in new_lrs.key_col:
                new_lrs.remove_key(chain_col, inplace=True)
        if new_lrs != df.lr.lrs:
            df.lr.set_lrs(new_lrs, inplace=True)
        # Add M-enabled geometries if needed
        if add_geom_m:
            df.lr.add_geom_m(name=geom_m_col, inplace=True)
        
        # Return updated DataFrame
        return None if inplace else df

    @_method_deprecates_geometry
    def extend(
        self,
        extend_begs: float = 0,
        extend_ends: float = 0,
        inplace: bool = False
    ) -> pd.DataFrame | None:
        """
        Extend the begin and end locations of the LRS by the specified 
        amounts.

        Parameters
        ----------
        extend_begs : float, default 0
            The amount to extend the begin locations to the left. Negative
            values shift the begin locations to the right.
        extend_ends : float, default 0
            The amount to extend the end locations to the right. Negative
            values shift the end locations to the left.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.

        Returns
        -------
        df : DataFrame
            A copy of the current DataFrame with extended begin and end 
            locations.
        """
        # Apply changes to the DataFrame
        obj = self if inplace else self.copy(deep=True)
        # Upgrade point to linear events if needed
        if self.is_point:
            obj.point_to_linear(inplace=True)
        # Extend events
        events = self.events.extend(
            extend_begs=extend_begs,
            extend_ends=extend_ends,
            inplace=False
        )
        # Apply changes to the DataFrame
        obj.begs = events.begs
        obj.ends = events.ends
        return None if inplace else obj.df
    
    @_method_deprecates_geometry
    def shift(
        self,
        shift: float,
        inplace: bool = False
    ) -> pd.DataFrame | None:
        """
        Shift the events of the LRS by the specified amount.

        Parameters
        ----------
        shift : float
            The amount to shift the events by.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.

        Returns
        -------
        df : DataFrame
            A copy of the current DataFrame with shifted events.
        """
        # Shift events
        events = self.events.shift(shift, inplace=False)
        # Apply changes to the DataFrame
        obj = self if inplace else self.copy(deep=True)
        if self.is_located:
            obj.locs = events.locs
        if self.is_linear:
            obj.begs = events.begs
            obj.ends = events.ends
        return None if inplace else obj.df
    
    @_method_deprecates_geometry
    def round(
        self,
        decimals: int = 0,
        inplace: bool = False
    ) -> pd.DataFrame | None:
        """
        Round the events of the LRS to the specified number of 
        decimals.

        Parameters
        ----------
        decimals : int, default 0
            The number of decimal places to round the events to.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.

        Returns
        -------
        df : DataFrame
            A copy of the current DataFrame with rounded events.
        """
        # Round events
        events = self.events.round(decimals=decimals, inplace=False)
        # Apply changes to the DataFrame
        obj = self if inplace else self.copy(deep=True)
        if self.is_located:
            obj.locs = events.locs
        if self.is_linear:
            obj.begs = events.begs
            obj.ends = events.ends
        return None if inplace else obj.df
    
    @_method_require(is_grouped=True)
    def impute_keys(
        self,
        other: pd.DataFrame,
        keys: list | None = None,
        func: str = 'first',
        fillna: float | str | dict | None = None,
        missing: str = 'warn'
    ) -> pd.DataFrame:
        """
        Impute missing key values from this dataframe onto another dataframe 
        based on matches between other keys and LRS locations.

        Parameters
        ----------
        other : DataFrame
            The DataFrame to impute key values onto.
        keys : list, optional
            A list of key column labels to impute. If None, all key columns
            from the active LRS not present on the other will be imputed.
        func : str, default 'first'
            EventsRelation aggregation function to use when multiple matches
            are found. See the EventsRelation class for available options.
        fillna : scalar or dict, optional
            Value or dictionary of values to use to fill NaN values after
            imputation. If None, no filling is performed. If provided as a
            dictionary, keys should be column labels and values should be
            the fill values for each column.
        missing : {'warn', 'ignore', 'raise'}, default 'warn'
            How to handle data that cannot be imputed because no matching
            LRS locations were found.

        Returns
        -------
        df : DataFrame
            A copy of the other DataFrame with imputed key values.
        """
        # Validate other dataframe
        if not isinstance(other, pd.DataFrame):
            raise TypeError("Input object must be of type `pd.DataFrame`.")
        if not other.lr.is_lrs_set:
            raise LRSCompatibilityError("Input DataFrame has no LRS set.")
        if not other.lr.is_located and not other.lr.is_linear:
            raise LRSCompatibilityError(
                "Other dataframe contains no valid event bounds."
            )
        # Define keys to impute
        if keys is None:
            keys = [key for key in self.key_col if key not in other.columns]
        else:
            for key in keys:
                if key not in self.key_col:
                    raise KeyError(f"Key column '{key}' not found in the active LRS.")
                if key not in self.df.columns:
                    raise KeyError(f"Key column '{key}' not found in the current DataFrame.")
        # Define LRS to use for relation
        lrs = self.lrs.remove_key(keys)
        # Relate the dataframes
        relation = self.set_lrs(lrs).lr.relate(other.lr.set_lrs(lrs))[keys]
        data = getattr(relation, func)(squeeze=False, axis=0)
        # Apply imputed keys to other dataframe
        df = other.copy()
        df[keys] = data
        # Fill NaN values if needed
        if fillna is not None:
            # If fillna is a dict, confirm that all columns are within the 
            # imputed keys
            if isinstance(fillna, dict):
                for col in fillna.keys():
                    if col not in keys:
                        raise KeyError(
                            f"Fill value provided for column '{col}' which "
                            f"was not imputed."
                        )
            df.fillna(value=fillna, inplace=True)
        # Handle missing data
        missing_data = df[keys].isna().any(axis=1)
        if missing_data.any():
            n_missing = missing_data.sum()
            msg = f"{n_missing} rows contain missing key values after imputation."
            if missing == 'warn':
                warnings.warn(msg, UserWarning)
            elif missing == 'raise':
                raise ValueError(msg)
        # Update LRS of other dataframe
        df.lr.set_lrs(self.lrs, inplace=True)
        return df
    
    @_method_require(is_linear=True)
    def resegment(
        self,
        length: float = 1,
        fill: str = 'cut',
        inverse_col: str | None = None,
        return_relation: bool = False,
        cut_geom: bool = True
    ) -> pd.DataFrame | None:
        """
        Resegment the events of the LRS to the specified length.

        Parameters
        ----------
        length : float, default 1
            The length to resegment the events to.
        fill : {'none','cut','left','right','extend','balance'}, default 'cut'
            How to fill a gap at the end of the input range.

            Options
            -------
            none : no range will be generated to fill the gap at the end of the 
                input range.
            cut : a truncated range will be created to fill the gap with a 
                length less than the full range length.
            left : the final range will be anchored on the end value and will 
                extend the full range length to the left. 
            right : the final range will be anchored on the grid defined by the 
                step value, extending the full range length to the right, 
                beyond the defined end value.
            extend : the final range will be anchored on the grid defined by 
                the step value, extending beyond the step length to the right
                bound of the range.
            balance : if the final range is greater than or equal to half the 
                target range length, perform the cut method; if it is less, 
                perform the extend method.

            Schematics
            ----------
            bounds :    [------------------------]
            none :   
                        [---------|              ]
                        [         |---------|    ]
            cut : 
                        [---------|              ]
                        [         |---------|    ]
                        [                   |----]
            left :   
                        [---------|              ]
                        [         |---------|    ]
                        [              |---------]
            right :  
                        [---------|              ]
                        [         |---------|    ]
                        [                   |----]----|
            extend :
                        [---------|              ]
                        [         |--------------]

        inverse_col : str, default 'segment_index'
            The label for the inverse index column that maps resegmented
            events to the original events. If not provided and the index of 
            the dataframe is unnamed, defaults to 'segment_index'. If the index
            is named, uses the name of the index.
        return_relation : bool, default False
            Whether to return an EventsRelation object between the resegmented
            events and the input events to allow for easy aggregation of data.
        cut_geom : bool, default True
            Whether to cut new geometries for the resegmented events based on
            the existing geometries in the dataframe.

        Returns
        -------
        df : DataFrame
            A copy of the current DataFrame with the events resegmented.
        """
        # Resegment events
        events, relation = self.events.resegment(
            length=length,
            fill=fill,
            return_relation=True
        )
        # Apply changes to the DataFrame
        df_left = events.to_frame(
            index_name=self._df.index.name,
            group_name=self.key_col,
            loc_name=self.loc_col,
            beg_name=self.beg_col,
            end_name=self.end_col,
        )
        df_right = self.df[self.other_cols]
        df = pd.merge(df_left, df_right, left_index=True, right_index=True)
        # Reset index and add inverse index
        if inverse_col is None:
            if self._df.index.name is None:
                inverse_col = 'segment_index'
            else:
                inverse_col = self._df.index.name
        df = df.reset_index(drop=False, names=inverse_col).lr.lrs_like(self)
        # Prepare relation object as needed
        if return_relation or cut_geom:
            # relation = df.lr.relate(self)
            relation.left_df = df
            relation.right_df = self.df
        # Cut new geometries if needed
        if cut_geom and self.is_spatial_m:
            try:
                df[self.geom_m_col] = relation.cut()
                df[self.geom_col] = np.array([
                    geom_m.geom if geom_m is not None else None \
                    for geom_m in df[self.geom_m_col]
                ])
                df = gpd.GeoDataFrame(
                    df, geometry=self.geom_col, crs=getattr(self.df, 'crs', None)
                ).lr.lrs_like(self)
            except:
                raise ValueError(
                    f"Unable to cut new geometries for resegmented events."
                )
        elif cut_geom and self.is_spatial:
            raise ValueError(
                f"Cannot cut new geometries: no geometry_m column in the "
                "dataframe. This can be resolved by adding a geometry_m using "
                "the `add_geom_m` method."
            )
        # Return results
        return (df, relation) if return_relation else df
    
    @_method_require(is_linear=True)
    def separate(
        self,
        anchor: str = 'centers',
        method: str = 'balanced',
        drop_short: bool = False,
        inplace: bool = False
    ) -> pd.DataFrame | None:
        """
        Address overlapping ranges by distributing overlaps between adjacent
        events. Eclipsed ranges (fully contained within another range) are
        eliminated. Identical ranges are deduplicated (first kept).

        Parameters
        ----------
        anchor : str {'centers', 'begs', 'ends'}, default 'centers'
            The anchor point of each event to be used when sorting events and
            distributing overlaps.
        method : str {'balanced', 'center', 'left', 'right'}, default 'balanced'
            The strategy for splitting overlapping regions between adjacent
            events.
        drop_short : bool, default False
            Whether to drop events that have been reduced to zero length.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        """
        # Separate events
        events = self.events.separate(
            anchor=anchor,
            method=method,
            drop_short=drop_short,
            inplace=False
        )
        # Apply changes to the DataFrame
        if drop_short and events.num_events < len(self.df):
            if inplace:
                raise ValueError(
                    "Cannot use inplace=True with drop_short=True because "
                    "dropping rows requires creating a new DataFrame.")
            df = self.df.loc[events.index].lr.lrs_like(self)
        else:
            df = self.df if inplace else self.copy_df()
        df[self.beg_col] = events.begs
        df[self.end_col] = events.ends
        return None if inplace else df

    @_method_require(is_linear=True)
    def dissolve(
        self, 
        retain: list[str] = [], 
        sort: bool = True, 
        inverse_index: bool = True, 
        inverse_col: str = 'dissolved_index', 
        merge_geom: bool | None = None,
        return_relation: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, relate.EventsRelation] | None:
        """
        Merge consecutive ranges. For best results, input events should be sorted.

        Parameters
        ----------
        retain : list, default []
            A list of column labels to retain during the dissolve operation.
        sort : bool, default True
            Whether to sort the events before dissolving. If True, results 
            will still be aligned to the original events. Unsorted events
            may produce unexpected results.
        inverse_index : bool, default True
            Whether to append an inverse index to the dissolved events dataframe.
        inverse_col : str, default 'dissolved_index'
            The label for the inverse index column.
        merge_geom : bool | None, default None
            Whether to merge the geometries of the dissolved events if present
            in the dataframe. If None, geometries will be merged if present in
            the dataframe. If True, an error will be raised if no geometry 
            column is present in the dataframe.
        return_relation : bool, default False
            Whether to return an EventsRelation object which describes the
            relationship between the dissolved (left) and original (right) 
            events.
        """
        # Validate input parameters
        if not isinstance(retain, list):
            raise ValueError("Input `retain` must be a list of valid dataframe column labels.")
        if merge_geom is None:
            merge_geom = self.is_spatial or self.is_spatial_m
        # Validate retain columns
        if not isinstance(retain, list):
            raise ValueError("Input `retain` must be a list of valid dataframe column labels.")
        else:
            missing_cols = set(retain) - set(self.df.columns)
            if len(missing_cols) > 0:
                raise KeyError(f"Retain columns not found in the dataframe: {missing_cols}")
        # Define key values to retain during dissolve
        if self.is_grouped:
            key_col = self.key_col + retain
        else:
            if self.is_lrs_grouped:
                raise LRSConfigurationError(
                    f"LRS key columns {self.missing_key_cols} not found "
                    f"in the DataFrame. Either add the missing columns or "
                    f"update the LRS configuration with set_lrs() or "
                    f"remove_key()."
                )
            key_col = retain
        # Dissolve events
        events = self.get_events(key_col=key_col, require=True)
        data, index, relation = events.dissolve(sort=sort, return_index=True, return_relation=True)
        # Convert events to dataframe
        df = data.to_frame(
            index_name=self._df.index.name,
            group_name=key_col,
            loc_name=self.loc_col,
            beg_name=self.beg_col,
            end_name=self.end_col,
        )
        relation.left_df = df
        relation.right_df = self.df

        # Append inverse index
        if inverse_index:
            df[inverse_col] = index
        # Merge geometries
        # - Merge from existing geometry_m column
        if merge_geom and self.is_spatial_m:
            try:
                merged_m = relation[self.geom_m_col].line_merge_m()
            except GeometryTopologyError:
                raise GeometryTopologyError(
                    "Linear geometries of adjacent events are disjointed and "
                    "cannot be merged into a single geometry. To resolve this, "
                    "ensure that geometries are contiguous or add chaining to "
                    "the dataframe using the `lr.add_chaining` method."
                )
            merged = np.array([i.geom for i in merged_m])
            # Assign merged geometries to the dataframe
            if self.is_spatial:
                df[self.geom_col] = merged
            df[self.geom_m_col] = merged_m

        # - Merge from ad-hoc geometry_m data
        elif merge_geom and self.is_spatial:
            try:
                merged_m = relation.line_merge_m(data=self.build_geom_m())
            except GeometryTopologyError:
                raise GeometryTopologyError(
                    "Linear geometries of adjacent events are disjointed and "
                    "cannot be merged into a single geometry. To resolve this, "
                    "first add M-enabled geometries to the dataframe using the "
                    "`add_geom_m` method, then ensure that geometries are "
                    "contiguous or add chaining to the dataframe using the "
                    "`lr.add_chaining` method."
                )
            merged = np.array([i.geom for i in merged_m])
            # Assign merged geometries to the dataframe
            df[self.geom_col] = merged
            df[self.geom_m_col] = merged_m

        # - No valid geometry column
        elif merge_geom:
            raise ValueError(
                "Cannot merge geometries: no geometry column in the dataframe."
            )

        # Upgrade to geodataframe
        if merge_geom:
            try:
                df = gpd.GeoDataFrame(
                    df, geometry=self.geom_col, crs=getattr(self.df, 'crs', None)
                ).lr.lrs_like(self)
            except:
                raise ValueError(
                    "Failed to convert dissolved DataFrame to GeoDataFrame."
                )
        else:
            df = df.lr.lrs_like(self)

        # Return results
        return (df, relation) if return_relation else df
    
    @_method_require(is_linear=True)
    def integrate(
        self,
        objs: pd.DataFrame | list[pd.DataFrame],
        fill_gaps: bool = False,
        split_at_locs: bool = False,
        inverse_col: str | list[str] | None = None
    ):
        """
        Combine one or more linearly referenced dataframes with the current 
        dataframe, creating new linear events based on least common intervals 
        among all input events. The input dataframes must have equivalent linear
        referencing systems.

        Parameters
        ----------
        objs : pd.DataFrame | list[pd.DataFrame]
            A DataFrame or a list of DataFrames with equivalent linear 
            referencing systems to integrate.
        fill_gaps : bool, default False
            Whether to fill gaps in merged events between the maximum beginning
            and minimum ending points of all input events within a group. These
            gaps will be represented as events with no associated input events.
        split_at_locs : bool, default False
            Whether to split events at location points within the input events.
            This allows for breaking events at point events as well as linear
            events.
        inverse_col : str or list of str, optional
            The label or list of labels for the inverse index columns that map
            integrated events to the original events from each input dataframe.
            If a single string is provided, all inverse index columns will use
            the same label with an appended suffix of '_0', '_1', etc. If None,
            default labels of 'integrated_index_0', 'integrated_index_1', etc.
            will be used.
        """
        # Perform integration
        return integrate(
            ([self.df] + objs) if isinstance(objs, list) else [self.df, objs],
            fill_gaps=fill_gaps,
            split_at_locs=split_at_locs,
            inverse_col=inverse_col,
            index_adjustment=1
        )
    
    @_method_require(is_linear=True)
    def cut_from(
        self,
        other: gpd.GeoDataFrame,
        geom_col: str | None = None,
        geom_m_col: str | None = None,
        multiple: str = 'first',
        inplace: bool = False
    ) -> gpd.GeoDataFrame | None:
        """
        Cut new geometries for the events in the dataframe from the geometries
        of another dataframe based on the LRS locations.
        
        Parameters
        ----------
        other : DataFrame
            The other DataFrame to cut geometries from. Must have an equivalent 
            linear referencing system with populated M-enabled linear 
            geometries.
        geom_col : str, optional
            The name of the geometry column to create in the dataframe. If None,
            use the geometry column name from the LRS if present, otherwise
            use the geometry column name from the other dataframe LRS if present,
            otherwise 'geometry'. This will also update the LRS geometry 
            column name.
        geom_m_col : str, optional
            The name of the geometry_m column to create in the dataframe. If None,
            use the geometry_m column name from the LRS if present, otherwise
            use the geometry_m column name from the other dataframe LRS if present,
            otherwise 'geometry_m'. This will also update the LRS geometry_m 
            column name.
        multiple : {'first', 'last', 'merge', 'list', 'raise'}, default 'first'
            The strategy to use when multiple geometries intersect.
            - 'first' : Use the first intersecting geometry only.
            - 'last' : Use the last intersecting geometry only.
            - 'merge' : Attempt to merge all intersecting geometries into a 
                        single M-enabled geometry.
            - 'raise' : Raise an error if multiple geometries intersect.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.

        Returns
        -------
        df : GeoDataFrame
            A copy of the current DataFrame with new cut geometries.
        """
        # Validate other dataframe
        if not isinstance(other, gpd.GeoDataFrame):
            raise TypeError("Input object must be of type `gpd.GeoDataFrame`.")
        if not other.lr.is_lrs_set:
            raise LRSCompatibilityError("Input DataFrame has no LRS set.")
        if not other.lr.lrs.is_spatial_m:
            raise LRSCompatibilityError(
                "Input DataFrame LRS has no geometry_m column set."
            )
        
        # Define geometry column names
        if geom_col is None:
            geom_col = \
                self.geom_col if self.geom_col is not None else \
                other.lr.geom_col if other.lr.geom_col is not None else \
                'geometry'
        if geom_m_col is None:
            geom_m_col = \
                self.geom_m_col if self.geom_m_col is not None else \
                other.lr.geom_m_col if other.lr.geom_m_col is not None else \
                'geometry_m'

        # Relate dataframes
        relation = self.relate(other)
        # Cut geometries
        cut_geoms_m = relation.cut(axis=1, multiple=multiple)
        cut_geoms = np.array([geom_m.geom for geom_m in cut_geoms_m])
        # Apply changes to the DataFrame
        df = self.df if inplace else self.copy_df()
        df[geom_col] = cut_geoms
        df[geom_m_col] = cut_geoms_m
        # Update LRS if needed
        if self.lrs.geom_col != geom_col or self.lrs.geom_m_col != geom_m_col:
            lrs = df.lr.lrs.copy(deep=True)
            lrs = lrs.set_params(geom_col=geom_col, geom_m_col=geom_m_col)
            update_lrs = True
        else:
            update_lrs = False
        # Upgrade to GeoDataFrame
        df = gpd.GeoDataFrame(df, geometry=geom_col, crs=getattr(other, 'crs', None))
        if update_lrs:
            df.lr.set_lrs(lrs, inplace=True)
        return None if inplace else df
    
    @_method_require(is_located=True)
    def interpolate_from(
        self,
        other: pd.DataFrame,
        geom_col: str | None = None,
        multiple: str = 'first',
        inplace: bool = False
    ) -> gpd.GeoDataFrame | None:
        """
        Interpolate new point geometries for the located events in the 
        dataframe from the geometries of another dataframe based on the LRS 
        locations.
        
        Parameters
        ----------
        other : DataFrame
            The other DataFrame to interpolate geometries from. Must have an 
            equivalent linear referencing system with populated M-enabled 
            linear geometries.
        geom_col : str, optional
            The name of the geometry column to create in the dataframe. If None,
            use the geometry column name from the LRS if present, otherwise
            use the geometry column name from the other dataframe LRS if present,
            otherwise 'geometry'. This will also update the LRS geometry 
            column name.
        multiple : {'first', 'last', 'raise'}, default 'first'
            The strategy to use when multiple geometries intersect.
            - 'first' : Use the first intersecting geometry only.
            - 'last' : Use the last intersecting geometry only.
            - 'raise' : Raise an error if multiple geometries intersect.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.

        Returns
        -------
        df : GeoDataFrame
            A copy of the current DataFrame with new interpolated geometries.
        """
        # Validate other dataframe
        if not isinstance(other, pd.DataFrame):
            raise TypeError("Input object must be of type `pd.DataFrame`.")
        if not other.lr.is_lrs_set:
            raise LRSCompatibilityError("Input DataFrame has no LRS set.")
        if not other.lr.lrs.is_spatial_m:
            raise LRSCompatibilityError(
                "Input DataFrame LRS has no geometry_m column set."
            )
        
        # Define geometry column name
        if geom_col is None:
            geom_col = \
                self.geom_col if self.geom_col is not None else \
                other.lr.geom_col if other.lr.geom_col is not None else \
                'geometry'
            
        # Relate dataframes
        relation = self.relate(other)
        # Interpolate geometries
        interp_geoms = relation.interpolate(axis=1, multiple=multiple)
        # Apply changes to the DataFrame
        df = self.df if inplace else self.copy_df()
        df[geom_col] = interp_geoms
        # Update LRS if needed
        if self.lrs.geom_col != geom_col:
            lrs = df.lr.lrs.copy(deep=True)
            lrs = lrs.set_params(geom_col=geom_col)
            update_lrs = True
        else:
            update_lrs = False
        # Upgrade to GeoDataFrame
        df = gpd.GeoDataFrame(df, geometry=geom_col, crs=getattr(other, 'crs', None))
        if update_lrs:
            df.lr.set_lrs(lrs, inplace=True)
        return None if inplace else df
    
    def distribute_from(
        self,
        other: pd.DataFrame,
        columns: list | str | None = None,
        inplace: bool = False,
        replace: bool = False,
        **params
    ) -> pd.DataFrame | None:
        """
        Distribute attribute data from another dataframe onto the current 
        dataframe based on linear referencing relationships. This is a shortcut
        for ``lr.relate(other)[columns].distribute(**params)``.

        Parameters
        ----------
        other : DataFrame
            The other DataFrame to distribute attributes from. Must have an 
            equivalent linear referencing system.
        columns : str or list, optional
            The column label or list of column labels to distribute from the
            other dataframe. If None, all columns except key and geometry
            columns will be distributed.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        replace : bool, default False
            Whether to replace existing columns in the DataFrame with the
            same names as those being distributed. If False, an error will be
            raised if any column names conflict.
        **params
            Additional parameters to pass to the EventsRelation.distribute
            method.

        Returns
        -------
        df : DataFrame
            A copy of the current DataFrame with distributed attributes or 
            None if inplace=True.
        """
        # Validate other dataframe
        if not isinstance(other, pd.DataFrame):
            raise TypeError("Input object must be of type `pd.DataFrame`.")
        if not other.lr.is_lrs_set:
            raise LRSCompatibilityError("Input DataFrame has no LRS set.")
        # Validate columns to distribute
        columns = label_list_or_none(columns)
        # Check for column name conflicts
        if not replace:
            if columns is None:
                if 'distributed' in self.df.columns:
                    raise ValueError(
                        "Default column name 'distributed' is already in use "
                        "in the DataFrame. Set `replace=True` to overwrite "
                        "existing columns."
                    )
            else:
                conflict_cols = list(set(columns) & set(self.df.columns))
                if len(conflict_cols) > 0:
                    raise ValueError(
                        f"Column name conflict(s) detected: {conflict_cols}. "
                        "Set `replace=True` to overwrite existing columns."
                    )
        
        # Relate dataframes
        relation = self.relate(other)
        relation._set_selector(columns, inplace=True)

        # Distribute attributes
        params['squeeze'] = False
        distributed = relation.distribute(**params)
        # Apply changes to the DataFrame
        df = self.df if inplace else self.copy_df()
        if columns is None:
            df['distributed'] = distributed
        else:
            df[columns] = distributed
        return None if inplace else df
    
    def parse_geom_m_wkt(
        self,
        geom_m_col: str | None = None,
        inplace: bool = False
    ) -> pd.DataFrame | None:
        """
        Parse the WKT representation of the M-enabled geometry data into a
        LineStringM object.

        Parameters
        ----------
        geom_m_col : str, optional
            The name of the geometry_m column to parse. If None, use the
            geometry_m column name from the LRS if present.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        """
        # Define geometry column name
        if geom_m_col is None:
            if not self.lrs.is_spatial_m:
                raise ValueError(
                    "No geometry_m column defined in the LRS. "
                    "Please provide a geometry_m_col parameter."
                )
            else:
                geom_m_col = self.lrs.geom_m_col

        # Parse WKT
        df = self.df if inplace else self.copy_df()
        df[geom_m_col] = df[geom_m_col].apply(geometry.parse_linestring_m_wkt)
        # Update LRS if needed
        if self.lrs.geom_m_col != geom_m_col:
            new_lrs = self.lrs.copy(deep=True)
            new_lrs.geom_m_col = geom_m_col
            df.lr.set_lrs(new_lrs, inplace=True)
        return None if inplace else df
    
    def geom_m_to_wkt(
        self,
        geom_m_col: str | None = None,
        inplace: bool = False
    ) -> pd.DataFrame | None:
        """
        Convert LineStringM objects in the M-enabled geometry column to WKT 
        strings. Typically used for export compatibility while using the 
        linref LineStringM implementation. Will become unnecessary once M-
        enabled geometries are more widely supported in shapely.

        Parameters
        ----------
        geom_m_col : str, optional
            The name of the geometry_m column to convert. If None, use the
            geometry_m column name from the LRS if present.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        """
        # Define geometry column name
        if geom_m_col is None:
            if not self.lrs.is_spatial_m:
                raise ValueError(
                    "No geometry_m column defined in the LRS. "
                    "Please provide a geom_m_col parameter."
                )
            else:
                geom_m_col = self.lrs.geom_m_col

        # Convert LineStringM to WKT
        df = self.df if inplace else self.copy_df()
        df[geom_m_col] = df[geom_m_col].apply(
            lambda x: x.wkt if isinstance(x, geometry.LineStringM) else x
        )
        return None if inplace else df

    def extract_m_values(
        self,
        beg_col: str | None = None,
        end_col: str | None = None,
        replace: bool = False,
        inplace: bool = False
    ) -> pd.DataFrame | None:
        """
        Extract the M values from the geometry_m column, applying them to the
        begin and end location columns.

        Parameters
        ----------
        beg_col : str, optional
            The name of the begin location column to add. If None, uses the
            existing begin column name in the LRS, or 'beg' if no begin column
            is defined.
        end_col : str, optional
            The name of the end location column to add. If None, uses the
            existing end column name in the LRS, or 'end' if no end column is
            defined.
        replace : bool, default False
            Whether to replace existing begin or end columns in the dataframe. 
            If False, an error will be raised if the columns already exist.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.
        """
        # Validate column names
        if beg_col is None:
            if self.beg_col is not None:
                beg_col = self.beg_col
            else:
                beg_col = 'beg'
        if end_col is None:
            if self.end_col is not None:
                end_col = self.end_col
            else:
                end_col = 'end'
        for col_name in [beg_col, end_col]:
            if col_name in self.df and not replace:
                raise ValueError(
                    f"Column name '{col_name}' is already in use in the "
                    "DataFrame."
                )
        # Apply changes to the DataFrame
        df = self.df if inplace else self.copy_df()
        df[beg_col] = [geom.beg_m for geom in self.geoms_m]
        df[end_col] = [geom.end_m for geom in self.geoms_m]

        # Update LRS if needed
        if beg_col != self.beg_col or end_col != self.end_col:
            new_lrs = df.lr.lrs.copy(deep=True)
            new_lrs.beg_col = beg_col
            new_lrs.end_col = end_col
            df.lr.set_lrs(new_lrs, inplace=True)
        return None if inplace else df

    def relate(
        self,
        other: pd.DataFrame,
        cache: bool = True
    ) -> relate.EventsRelation:
        """
        Create an events data relationship between two linearly referenced
        datasets.

        Parameters
        ----------
        other : DataFrame
            The other DataFrame to relate with. Must have an equivalent 
            linear referencing system.
        cache : bool, default True
            Whether to cache computed relationship operations, such as 
            intersections and overlays, for faster subsequent operations. For 
            one-time operations or to save on memory use for large datasets, 
            set cache=False.
        """
        other = self._validate_other_dataframe_lrs(other)
        # Create relationship
        return self.events.relate(
            other=other.lr.events,
            cache=cache,
            left_df=self.df,
            right_df=other,
        )
    
    def overlay(
        self,
        other: pd.DataFrame,
        normalize: bool = False,
        norm_by: str = 'right',
        chunksize: int = 1000,
        grouped: bool = True
    ) -> sp.csr_array:
        """
        Overlay two sets of linearly referenced datasets, computing the 
        length or proportion of overlap between each pair of events.

        Parameters
        ----------
        other : DataFrame
            The other DataFrame to overlay with. Must have an equivalent 
            linear referencing system.
        normalize : bool, default False
            Whether overlapping lengths should be normalized to give a 
            proportional result with a float value between 0 and 1.
        norm_by : str, default 'right'
            How overlapping lengths should be normalized. Only applied if
            `normalize` is True.
            - 'right' : Normalize by the length of the right events.
            - 'left' : Normalize by the length of the left events.
        chunksize : int or None, default 1000
            The maximum number of events to process in a single chunk.
            Input chunksize will affect the memory usage and performance of
            the function. This does not affect actual results, only 
            computation.
        grouped : bool, default True
            Whether to process the overlay operation for each group separately.
            This will affect the memory usage and performance of the function. 
            This does not affect actual results, only computation.
        """
        other = self._validate_other_dataframe_lrs(other)
        # Perform overlay
        return self.events.overlay(
            other=other.lr.events,
            normalize=normalize,
            norm_by=norm_by,
            chunksize=chunksize,
            grouped=grouped
        )

    def intersect(
        self,
        other: pd.DataFrame,
        enforce_edges: bool = True,
        chunksize: int = 1000,
        grouped: bool = True
    ) -> sp.csr_array:
        """
        Identify intersections between two sets of linearly referenced datasets.

        Parameters
        ----------
        other : DataFrame
            The other DataFrame to intersect with. Must have an equivalent 
            linear referencing system.
        enforce_edges : bool, default True
            Whether to consider cases of coincident begin and end points, 
            according to each collection's closed state. For instances where 
            these cases are not relevant, set enforce_edges=False for improved 
            performance. Ignored for point to point intersections.
        chunksize : int or None, default 1000
            The maximum number of events to process in a single chunk.
            Input chunksize will affect the memory usage and performance of
            the function. This does not affect actual results, only 
            computation.
        grouped : bool, default True
            Whether to process the overlay operation for each group separately.
            This will affect the memory usage and performance of the function. 
            This does not affect actual results, only computation.
        """
        other = self._validate_other_dataframe_lrs(other)
        # Perform intersect
        return self.events.intersect(
            other=other.lr.events,
            enforce_edges=enforce_edges,
            chunksize=chunksize,
            grouped=grouped
        )

    def cluster(
        self,
        max_gap: float,
        name: str = 'cluster',
        enforce_edges: bool | None = None,
        inplace: bool = False
    ) -> pd.DataFrame | None:
        """
        Cluster events based on linear referencing proximity within each group.
        Events whose locations are within the specified max gap of each other
        (directly or transitively) are assigned to the same cluster.

        This method buffers each event location by the max gap to create 
        small linear ranges, then identifies overlapping ranges within each 
        group using a self-intersection, and finally extracts connected 
        components from the resulting adjacency graph to form clusters.

        For point events, two points are proximal if their locations are within 
        the max gap distance. For linear events, two events are proximal if 
        their ranges (after extension by the max gap) overlap.

        Parameters
        ----------
        max_gap : float
            The maximum linear distance between event locations for them to 
            be considered proximal. Events within this distance (directly or 
            transitively through intermediate events) will be assigned to the 
            same cluster.
        name : str, default 'cluster'
            The name of the cluster column to add to the DataFrame.
        enforce_edges : bool or None, default None
            Whether to treat shared edges (touching boundaries) as 
            intersections. If None, defaults to False for linear events and 
            is disregarded for point events.
        inplace : bool, default False
            Whether to apply changes to the DataFrame in place.

        Returns
        -------
        df : DataFrame or None
            A copy of the DataFrame with a new cluster column, or None if 
            inplace=True.

        Raises
        ------
        LRSConfigurationError
            If the LRS has no location or linear data defined, or if event 
            locations or bounds contain undefined (NaN) values.
        """
        # Get events, requiring fully defined positional data
        try:
            events = self.get_events(allow_undefined_events=False)
        except LRSConfigurationError as e:
            raise LRSConfigurationError(
                "Cannot cluster events with undefined locations or bounds. "
                "Use `drop_invalid_events()` to remove events with missing "
                "data before clustering."
            ) from e

        # Buffer event locations into linear ranges (skip if max_gap is zero)
        if max_gap > 0:
            buffered = events.extend(
                extend_begs=max_gap,
                extend_ends=max_gap,
                inplace=False
            )
        elif max_gap < 0:
            raise ValueError("max_gap must be non-negative.")
        else:
            buffered = events

        # Self-relate buffered events and extract connected components
        relation = buffered.relate(buffered)
        if buffered.is_point:
            if enforce_edges is not None:
                raise ValueError(
                    "The enforce_edges parameter is not applicable to point "
                    "events."
                )
            n_clusters, labels = relation.connected_components()
        else:
            if enforce_edges is None:
                enforce_edges = False
            n_clusters, labels = relation.connected_components(
                enforce_edges=enforce_edges
            )

        # Apply cluster labels to the DataFrame
        df = self.df if inplace else self.copy_df()
        df[name] = labels
        return None if inplace else df

    @_method_require(is_spatial=True)
    def generate_intersections(
        self,
        exclude_groups: bool | str | list[str] = True,
        touches: bool = True,
        crosses: bool = True,
        project: bool = True,
        expand: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Find unique intersection locations between line geometries in the 
        DataFrame, returning one row per unique intersection point with all
        participating source geometry indices.

        Optionally projects intersection points onto the LRS to obtain linear
        referencing locations.

        Parameters
        ----------
        exclude_groups : bool, str, or list of str, default True
            Controls which column(s) are used for group exclusion:
            - True : Use the LRS key columns for group exclusion, preventing
              intersections between segments of the same route.
            - str or list of str : Use the specified column name(s).
            - False : No group exclusion; all intersecting pairs are included.
        touches : bool, default True
            If True, include pairs that share boundary points (endpoints) only.
        crosses : bool, default True
            If True, include pairs whose interiors intersect (interior 
            crossings).
        project : bool, default True
            Whether to project intersection points onto the LRS to obtain 
            linear referencing locations. Requires a fully configured linear 
            spatial LRS with M-enabled geometries.
        expand : bool, default True
            When projecting, whether to expand results to one row per 
            intersecting line per node. If True, each intersection node 
            produces one row for each coincident line, giving the LRS 
            location on every line. If False, each node produces a single 
            row projected onto an arbitrary coincident line. Only used when 
            ``project=True``.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame with one row per unique intersection location 
            (or per line per location when ``expand=True``):
            - geometry : The intersection Point geometry.
            - indices : A list of source geometry index labels participating 
              in the intersection.
            
            When ``project=True``, additional LRS columns are included from
            the projection step.
        """
        from linref.ext.spatial import generate_intersection_nodes
        # Resolve exclude_groups from LRS keys
        if exclude_groups is True:
            cols = self.key_col if self.is_grouped else None
        elif exclude_groups is False:
            cols = None
        else:
            cols = exclude_groups
        # Generate intersection nodes (tuple of arrays)
        geoms, indices = generate_intersection_nodes(
            self.df, exclude_groups=cols, touches=touches, crosses=crosses
        )
        # Construct GeoDataFrame from arrays
        result = gpd.GeoDataFrame(
            {'geometry': geoms, 'indices': indices},
            geometry='geometry',
            crs=self.df.crs,
        )
        # Optionally project onto LRS
        if project:
            result = self.dissolve().lr.project(
                result, buffer=1e-10, nearest=not expand, replace=True,
            )
        return result

    @_method_require(is_spatial=True, is_spatial_m=True, is_linear=True)
    def project(
        self,
        other: gpd.GeoDataFrame,
        buffer: float = 100,
        nearest: bool = True,
        distance_col: str = 'project_distance',
        replace: bool = False,
        dropna: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Project the input DataFrame of point events onto the active DataFrame
        of linear events.

        Parameters
        ----------
        other : GeoDataFrame
            The other GeoDataFrame to project. Must be point-based and spatially
            referenced.
        buffer : float, default 100
            The buffer distance to use when searching for nearest linear events.
            In units of the spatial reference system.
        nearest : bool, default True
            Whether to choose only the nearest match within the defined buffer. 
            If False, all matches will be returned. If True, when multiple 
            equidistant points exist, choose the first result that appears.
        dist_label : str, default 'project_distance'
            The label for the distance column in the returned DataFrame.
        replace : bool, default False
            Whether to replace the existing key and location columns in the 
            dataframe. If False, an error will be raised if the columns 
            already exist.
        dropna : bool, default True
            Whether to drop rows from the returned DataFrame where no matching
            linear event was found within the defined buffer. Events with no
            match will have NaN values for LRS columns which may produce 
            unexpected results in subsequent operations.

        Returns
        -------
        df : GeoDataFrame
            A copy of the other GeoDataFrame with projected LRS locations and
            distance column added.
        """
        # Ensure that the LRS has a location column
        if self.loc_col is None:
            raise ValueError(
                "LRS must contain a location column to be applied to "
                "the projected points."
            )
        # Validate input geodataframe
        if not isinstance(other, gpd.GeoDataFrame):
            raise TypeError("Other object must be gpd.GeoDataFrame instance.")
        else:
            try:
                other_geometry_name = other.geometry.name
            except AttributeError:
                raise AttributeError(
                    "No geometry data set in other geodataframe.")
        # Log dataframe index name
        index_right_name = (
            self.df.index.name if self.df.index.name is not None else 'index_right'
        )
        # Check for presence of LRS columns already in the other dataframe
        protected_cols = set(self.key_col + [self.loc_col, distance_col, index_right_name])
        if not replace:
            overlapping_cols = protected_cols.intersection(set(other.columns))
            if len(overlapping_cols) > 0:
                raise ValueError(
                    f"Other geodataframe contains protected column names: "
                    f"{', '.join(overlapping_cols)}"
                )

        # Spatial join points to lines
        select_cols = self.key_col + [self.geom_col, self.geom_m_col]
        if nearest:
            joined = other.drop(columns=protected_cols, errors='ignore') \
                .sjoin_nearest(
                self.df[select_cols],
                how='left',
                max_distance=buffer,
                distance_col=distance_col,
            )
            # Drop duplicates for cases of equidistant matches
            joined = joined[~joined.index.duplicated(keep='first')]
        else:
            joined = other.sjoin(
                self.df[select_cols],
                how='left',
                predicate='dwithin',
                distance=buffer,
            )
            # Get distances for all matches
            left_geoms = joined.geometry
            right_geoms = gpd.GeoSeries(
                joined[index_right_name].map(self.df[self.geom_col]).values,
                index=joined.index,
                crs=getattr(self.df, 'crs', None),
            )
            joined[distance_col] = left_geoms.distance(right_geoms)
        
        # Project input points onto event geometries (vectorized)
        locs = geometry.line_locate_point_m(
            joined[self.geom_m_col].values,
            joined[other_geometry_name].values,
            m=True,
        )
        joined[self.loc_col] = locs
        # Drop rows with no match if needed
        if dropna:
            joined = joined.dropna(
                subset=self.key_col + [self.loc_col],
                how='any',
            )
        # Return projected dataframe
        return joined.drop(columns=[self.geom_m_col]).lr.lrs_like(self)
    

# Helper functions for event operations


def check_compatibility(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Validate that all dataframes have compatible LRS objects assigned, 
    raising various errors for incompatible LRS configurations.

    Parameters
    ----------
    dfs : list of DataFrame
        A list of DataFrames to validate.
    """
    # Validate input list
    if not isinstance(dfs, list):
        raise TypeError("Input `dfs` must be a list of DataFrames.")
    if len(dfs) < 1:
        raise ValueError(
            "Input `dfs` must contain at least one DataFrame."
        )
    
    # Validate basic LRS settings
    for i, df in enumerate(dfs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "All input objects must be of type `pd.DataFrame`. Object "
                f"at index {i} is not a DataFrame."
            )
        if not df.lr.is_lrs_set:
            raise LRSCompatibilityError(
                f"Input dataframe at index {i} has no LRS set."
            )
        
    # Validate LRS compatibility
    primary_df = dfs[0]
    for i, df in enumerate(dfs[1:]):
        # Is grouped
        if primary_df.lr.is_grouped != df.lr.is_grouped:
            raise LRSCompatibilityError(
                f"LRS of dataframe at index {i + 1} has a different "
                "grouped state than the dataframe at index 0."
            )
        # Number of key columns
        if primary_df.lr.is_grouped:
            if len(primary_df.lr.key_col) != len(df.lr.key_col):
                raise LRSCompatibilityError(
                    f"LRS of dataframe at index {i + 1} has a different number "
                    "of key columns than the dataframe at index 0. Received "
                    f"{len(df.lr.key_col)} columns, but expected "
                    f"{len(primary_df.lr.key_col)}."
                )
            # Key column data types
            if primary_df.lr.events.groups.dtype != df.lr.events.groups.dtype:
                raise LRSCompatibilityError(
                    f"LRS of dataframe at index {i + 1} has different key column "
                    "data types than the dataframe at index 0."
                )
    return dfs

def integrate(
        dfs: list[pd.DataFrame],
        fill_gaps: bool = False,
        split_at_locs: bool = False,
        inverse_col: str | list[str] | None = None,
        **kwargs
    ) -> pd.DataFrame:
    """
    Combine multiple linearly referenced dataframes into a single dataframe,
    creating new linear events based on least common intervals among all input 
    events.

    Parameters
    ----------
    dfs : list of DataFrame
        A list of DataFrames with equivalent linear referencing systems to
        integrate.
    fill_gaps : bool, default False
        Whether to fill gaps in merged events between the maximum beginning
        and minimum ending points of all input events within a group. These
        gaps will be represented as events with no associated input events.
    split_at_locs : bool, default False
        Whether to split events at location points within the input events.
        This allows for breaking events at point events as well as linear
        events.
    inverse_col : str or list of str, optional
        The label or list of labels for the inverse index columns that map
        integrated events to the original events from each input dataframe.
        If a single string is provided, all inverse index columns will use
        the same label with an appended suffix of '_0', '_1', etc. If None,
        default labels of 'integrated_index_0', 'integrated_index_1', etc.
        will be used.
    """
    # Check for dataframe index adjustment
    adj = kwargs.pop('index_adjustment', 0)
    if len(kwargs) > 0:
        raise TypeError(
            f"integrate() got unexpected keyword arguments: "
            f"{', '.join(kwargs.keys())}"
        )
    # Validate input dataframes
    if not isinstance(dfs, list):
        raise TypeError("Input `dfs` must be a list of DataFrames.")
    if len(dfs) < 1:
        raise ValueError(
            "Input `dfs` must contain at least one DataFrame."
        )
    for i, df in enumerate(dfs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Input object at index {i - adj} is not a DataFrame."
            )
        if not df.lr.is_lrs_set:
            raise LRSCompatibilityError(
                f"Input DataFrame at index {i - adj} has no LRS set."
            )
        if not df.lr.lrs.is_linear:
            raise LRSCompatibilityError(
                f"Input DataFrame at index {i - adj} LRS has no linear events."
            )
        # Validate that all LRS have compatible linear referencing params
        # (geometry columns are not used by integrate and are ignored)
        if df.lr.lrs.set_params(geom_col=None, geom_m_col=None) != \
                dfs[0].lr.lrs.set_params(geom_col=None, geom_m_col=None):
            raise LRSCompatibilityError(
                f"Input DataFrame at index {i - adj} has an incompatible "
                "linear referencing system."
            )
    # Validate inverse column names
    if inverse_col is None:
        inverse_col = [f'integrated_index_{i}' for i in range(len(dfs))]
    elif isinstance(inverse_col, str):
        inverse_col = [f'{inverse_col}_{i}' for i in range(len(dfs))]
    elif isinstance(inverse_col, list):
        if len(inverse_col) != len(dfs):
            raise ValueError(
                "Input `inverse_col` list length must match number of input "
                "dataframes."
            )
    else:
        raise TypeError(
            "Input `inverse_col` must be a string, list of strings, or None."
        )
        
    # Perform integration
    events, indices = integration.integrate(
        [df.lr.events for df in dfs],
        fill_gaps=fill_gaps,
        split_at_locs=split_at_locs,
        return_index=True
    )
    # Convert events to dataframe
    integrated = events.to_frame(
        index_name=None,
        group_name=dfs[0].lr.key_col,
        beg_name=dfs[0].lr.beg_col,
        end_name=dfs[0].lr.end_col,
    )
    # Convert appendices from generic to dataframe-specific indices
    for i, df in enumerate(dfs):
        selection = indices[:, i]
        index = np.where(
            selection != -1,
            df.index.values[selection],
            np.nan
        )
        # Append indices from each input dataframe
        integrated[inverse_col[i]] = index

    # Set LRS on integrated output from first dataframe and return
    return integrated.lr.lrs_like(dfs[0])

def parse_geoms_m_wkt(geoms: pd.Series) -> pd.Series:
    """
    Parse the WKT representations of M-enabled geometries into a series of
    LineStringM object.

    Parameters
    ----------
    geoms : pd.Series
        A pandas Series containing WKT representations of M-enabled geometries.
    """
    # Validate input
    if not isinstance(geoms, pd.Series):
        raise TypeError("Input `geoms` must be a pandas Series.")
    # Parse WKT geometries
    return geoms.apply(geometry.parse_linestring_m_wkt)

def parse_geoms_m_shapely(geoms: pd.Series, reverse: bool = False) -> pd.Series:
    """
    Parse the Shapely geometry representations of M-enabled geometries
    into a series of LineStringM object.

    NOTE: This method requires Shapely version 2.1 or higher. This process
    will become deprecated in future versions in favor of direct support
    for M-enabled geometries in Shapely.

    Parameters
    ----------
    geoms : pd.Series
        A pandas Series containing Shapely geometry representations of
        M-enabled geometries.
    reverse : bool, default False
        Whether to attempt reversing the geometries in case of decreasing
        M values. If False, an error will be raised if decreasing M values
        are detected.
    """
    # Validate input
    if not isinstance(geoms, pd.Series):
        raise TypeError("Input `geoms` must be a pandas Series.")
    # Parse Shapely geometries
    def parse_geom(geom):
        try:
            parsed = geometry.LineStringM.from_shapely(geom)
        except GeometryMeasureError:
            if not reverse:
                raise GeometryMeasureError(
                    "Failed to parse M-enabled geometry from Shapely object. "
                    "Ensure that the geometry contains valid, monotonic, "
                    "increasing M values. Set `reverse=True` to attempt reversing "
                    "the geometry in case of decreasing M values."
                )
            try:
                # Try reversing the geometries in case of decreasing M values
                parsed = geometry.LineStringM.from_shapely(shapely.reverse(geom))
            except GeometryMeasureError:
                raise GeometryMeasureError(
                    "Failed to parse M-enabled geometry from Shapely object. "
                    "Ensure that the geometry contains valid, monotonic, "
                    "increasing M values."
                )
        return parsed
    return geoms.apply(parse_geom)


# Declare the accessor for static type checkers (Pylance, Pyright, mypy)
if TYPE_CHECKING:
    pd.DataFrame.lr: LRS_Accessor  # type: ignore[attr-defined]