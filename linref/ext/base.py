from __future__ import annotations
import numpy as np
import pandas as pd
import copy, hashlib
from pandas.api.extensions import register_dataframe_accessor
from scipy import sparse as sp
from linref.ext.utility import label_list_or_none, label_or_none
from linref.events.common import closed_all
from linref.events.base import EventsData
from linref.events.utility import _method_require
from linref.events import relate, geometry


class LRS(object):

    def __init__(self, key_col=None, loc_col=None, beg_col=None, end_col=None, geom_col=None, geom_m_col=None, closed=None) -> None:
        # Validate LRS column labels
        self.key_col = label_list_or_none(key_col)
        self.loc_col = label_or_none(loc_col)
        self.beg_col = label_or_none(beg_col)
        self.end_col = label_or_none(end_col)
        self.geom_col = label_or_none(geom_col)
        self.geom_m_col = label_or_none(geom_m_col)
        # Validate LRS closure
        if closed not in closed_all:
            raise ValueError(
                f"Invalid LRS closure: {closed}. Must be one of: {closed_all}.")
        else:
            self.closed = closed

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            "LRS("
            f"key_col={self.key_col}, "
            f"loc_col={self.loc_col}, "
            f"beg_col={self.beg_col}, "
            f"end_col={self.end_col}, "
            f"geom_col={self.geom_col}, "
            f"geom_m_col={self.geom_m_col}, "
            f"closed={self.closed})"
        )
    
    @property
    def is_linear(self) -> bool:
        return (self.beg_col is not None) and (self.end_col is not None)
    
    @property
    def is_point(self) -> bool:
        return (self.loc_col is not None) and (self.beg_col is None) and (self.end_col is None)
    
    @property
    def is_located(self) -> bool:
        return self.loc_col is not None
    
    @property
    def is_grouped(self) -> bool:
        return self.key_col is not None
    
    @property
    def is_spatial(self) -> bool:
        return self.geom_col is not None
    
    @property
    def is_spatial_m(self) -> bool:
        return self.geom_m_col is not None
    
    def copy(self, deep=False) -> LRS:
        """
        Create an exact copy of the object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    
    def add_key(self, key_col) -> None:
        self.key_col.extend(label_list_or_none(key_col))

    def study(self, df) -> dict:
        """
        Validate the dataframe for LRS compatibility.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to validate.
        how : {'raise', 'verbose', 'bool'}, default 'raise'
            Whether to raise an error or return a boolean for validation failures.
        """
        # Check for presence of LRS columns in the dataframe
        result = {}
        if self.is_grouped:
            missing_keys = [key for key in self.key_col if key not in df.columns]
            result['keys'] = {'valid': len(missing_keys) == 0, 'missing': missing_keys}
        if self.is_linear:
            missing_linear = [col for col in [self.beg_col, self.end_col] if col not in df.columns]
            result['linear'] = {'valid': len(missing_linear) == 0, 'missing': missing_linear}
        if self.is_located:
            valid = self.loc_col in df.columns
            result['location'] = {'valid': valid, 'missing': self.loc_col if not valid else None}
        if self.is_spatial:
            valid = self.geom_col in df.columns
            result['geometry'] = {'valid': valid, 'missing': self.geom_col if not valid else None}
        if self.is_spatial_m:
            valid = self.geom_m_col in df.columns
            result['geometry_m'] = {'valid': valid, 'missing': self.geom_m_col if not valid else None}
        return result


@register_dataframe_accessor("lr")
class LRS_Accessor(object):

    # Initialize default LRS list
    _default_lrs = []

    def __init__(self, df) -> None:
        # Log dataframe
        self._df = df
        # Initialize LRS
        self._lrs = []
        self._active_index = 0

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self.is_lrs_set:
            lrs_lines = '\n'.join(['- ' + str(o) for o in self.lrs])
        else:
            lrs_lines = "- No LRS set"
        return "LRS_Accessor with linear referencing system (LRS) objects:\n" + lrs_lines

    def __getitem__(self, index) -> LRS_Accessor:
        """
        Activate an LRS by index.
        """
        self.activate_lrs(index)
        return self
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    @df.setter
    def df(self, df) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input DataFrame must be of type `pandas.DataFrame`.")
        self._df = df
    
    @property
    def lrs(self) -> list[LRS]:
        if len(self._lrs) > 0:
            return self._lrs
        return self._default_lrs
        
    @property
    def is_lrs_set(self) -> bool:
        return len(self.lrs) > 0

    @property
    def active_index(self) -> int:
        return self._active_index
    
    @property
    def active_lrs(self) -> LRS:
        if not self.is_lrs_set:
            raise ValueError("No LRS set for the DataFrame.")
        return self.lrs[self.active_index]
    
    @property
    def key_col(self) -> list[str]:
        return self.active_lrs.key_col
    
    @property
    def loc_col(self) -> str:
        return self.active_lrs.loc_col
    
    @property
    def beg_col(self) -> str:
        return self.active_lrs.beg_col
    
    @property
    def end_col(self) -> str:
        return self.active_lrs.end_col
    
    @property
    def geom_col(self) -> str:
        return self.active_lrs.geom_col
    
    @property
    def geom_m_col(self) -> str:
        return self.active_lrs.geom_m_col
    
    @property
    def closed(self) -> str:
        return self.active_lrs.closed
    
    @property
    def index(self) -> np.ndarray:
        return self._df.index.values
    
    @property
    def keys(self) -> np.ndarray:
        return self.get_keys(require=False)
        
    @property
    def locs(self) -> np.ndarray:
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
    def geoms(self) -> np.ndarray:
        # Select data from the dataframe if geometry is present
        col = self.geom_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @property
    def geoms_m(self) -> np.ndarray:
        # Select data from the dataframe if geometry is present
        col = self.geom_m_col
        try:
            return self._df[col].values
        except KeyError:
            return None
    
    @property
    def is_grouped(self) -> bool:
        """
        Return whether the active LRS is grouped and the key columns are 
        present in the dataframe.
        """
        if self.key_col is None:
            return False
        else:
            # Check for presence of key columns in the dataframe
            return all([key in self._df.columns for key in self.key_col])
    
    @property
    def is_linear(self) -> bool:
        """
        Return whether the active LRS is linear and the begin and end columns
        are present in the dataframe.
        """
        if self.beg_col is None or self.end_col is None:
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
        if self.loc_col is None:
            return False
        elif self.beg_col is not None or self.end_col is not None:
            return False
        else:
            # Check for presence of location column in the dataframe
            return self.loc_col in self._df.columns
    
    @property
    def is_located(self) -> bool:
        """
        Return whether the active LRS is located and the location column is 
        present in the dataframe.
        """
        if self.loc_col is None:
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
        if self.geom_col is None:
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
        if self.geom_m_col is None:
            return False
        else:
            # Check for presence of geometry column in the dataframe
            return self.geom_m_col in self._df.columns

    @property
    def events(self) -> EventsData:
        """
        Return the events object for the active LRS.
        """
        # Create the events object
        return self.get_events()
    
    def get_keys(self, col=None, require=True) -> np.ndarray:
        # Select data from the dataframe if keys are present
        if col is None:
            col = self.key_col
        if col is None:
            return None
        try:
            # Convert data to list of tuples
            return self._df[col].to_records(index=False)
        except KeyError as e:
            if require:
                raise e
            return None
        
    def get_events(self, key_col=None, require=True) -> EventsData:
        """
        Return the events object for the active LRS.
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
            closed=self.closed,
            force_monotonic=False
        )
    
    def copy(self, deep=False) -> LRS_Accessor:
        """
        Create an exact copy of the object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    
    def lrs_like(self, other, inplace=False) -> pd.DataFrame:
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
        df.lr.set_lrs(other.lrs, append=False)
        df.lr.activate_lrs(other.active_index)
        return df
    
    def activate_lrs(self, index) -> None:
        """
        Activate a specific LRS for the DataFrame by selecting the index from 
        the list of LRS objects.

        Parameters
        ----------
        index : int
            The index of the LRS object to activate. Used to index the list of
            LRS objects stored in the DataFrame.
        """
        if index >= len(self.lrs):
            raise ValueError(
                f"Invalid LRS index: {index}. Must be less than {len(self.lrs)}.")
        self._active_index = index

    def set_lrs(self, lrs=None, append=False, activate=False, **kwargs) -> None:
        """
        Set the LRS object for the DataFrame.

        Parameters
        ----------
        lrs : LRS or list[LRS], default None
            The LRS object or list of LRS objects to set for the DataFrame.
        append : bool, default False
            Whether to append the input LRS objects to the existing LRS objects
            or replace them.
        activate : bool, default False
            Whether to activate the added LRS object. If multiple LRS objects
            are added, the last object will be activated.
        """
        # Validate LRS object type
        if lrs is None:
            lrs = [LRS(**kwargs)]
        elif isinstance(lrs, LRS):
            lrs = [lrs]
        elif not all([isinstance(lrs, LRS) for lrs in lrs]):
            raise ValueError("Input LRS objects must be of type `LRS`.")
        
        # Append or replace LRS objects
        if append:
            self._lrs.extend(lrs)
        else:
            self._lrs = lrs

        # Activate LRS object if requested
        if activate:
            self.activate_lrs(len(self.lrs) - 1)
        
    def add_lrs(self, lrs=None, activate=False, **kwargs) -> None:
        """
        Add LRS objects to the DataFrame. Equivalent to 
        `set_lrs(..., append=True)`.

        Parameters
        ----------
        lrs : LRS or list[LRS], default None
            The LRS object or list of LRS objects to add to the DataFrame.
        activate : bool, default False
            Whether to activate the added LRS object. If multiple LRS objects
            are added, the last object will be activated.
        """
        self.set_lrs(lrs=lrs, append=True, activate=activate, **kwargs)

    def clear_lrs(self) -> None:
        """
        Clear the LRS objects from the DataFrame.
        """
        self._lrs = []

    def add_geom_m(self, name='geometry_m', inplace=False) -> pd.DataFrame | None:
        """
        Add a geometry column to the DataFrame based on the begin and end 
        locations of the active LRS.

        Parameters
        ----------
        name : str, default 'geometry_m'
            The name of the geometry column to return.
        inplace : bool, default False
            Whether to apply changes to the dataframe in place.
        """
        def _upgrade_geom(geom, beg, end):
            geom_m = geometry.LineStringM(geom)
            geom_m.set_m_from_bounds(beg=beg, end=end, inplace=True)
            return geom_m
        # Cast linear geometries to LineStringM
        geoms_m = list(map(_upgrade_geom, self.geoms, self.begs, self.ends))
        # Apply changes to the DataFrame
        df = self.df if inplace else self.df.copy()
        df[name] = geoms_m
        # Update LRS if needed
        if not self.is_spatial:
            new_lrs = self.active_lrs.copy(deep=True)
            new_lrs.geom_m_col = name
            df.lr.add_lrs(new_lrs, activate=True)
        elif self.active_lrs.geom_m_col != name:
            new_lrs = self.active_lrs.copy(deep=True)
            new_lrs.geom_m_col = name
            df.lr.add_lrs(new_lrs, activate=True)
        return None if inplace else df

    @_method_require(is_grouped=True)
    def iter_groups(self):
        """
        Iterate over unique event groups in the dataframe based on the 
        active LRS key columns.
        """
        for group, index in self.events.iter_group_indices():
            yield group, self.df.loc[index]

    @classmethod
    def set_default_lrs(cls, lrs=None, append=False, **kwargs) -> None:
        """
        Set the default LRS objects for the LRS_Accessor class. Default LRS
        objects are used when no LRS objects are set for a specific DataFrame.

        Parameters
        ----------
        lrs : LRS or list[LRS], default None
            The LRS object or list of LRS objects to set as the default for the
            LRS_Accessor class.
        append : bool, default False
            Whether to append the input LRS objects to the existing default LRS
            objects or replace them.
        """
        # Validate LRS object type
        if lrs is None:
            lrs = [LRS(**kwargs)]
        elif isinstance(lrs, LRS):
            lrs = [lrs]
        elif not all([isinstance(lrs, LRS) for lrs in lrs]):
            raise ValueError("Input LRS objects must be of type `LRS`.")
        
        # Append or replace LRS objects
        if append:
            cls._default_lrs.extend(lrs)
        else:
            cls._default_lrs = lrs

    @classmethod
    def add_default_lrs(cls, lrs=None, **kwargs) -> None:
        """
        Add default LRS objects to the LRS_Accessor class. Equivalent to
        `set_default_lrs(..., append=True)`.

        Parameters
        ----------
        lrs : LRS or list[LRS], default None
            The LRS object or list of LRS objects to add as default for the
            LRS_Accessor class.
        """
        cls.set_default_lrs(lrs=lrs, append=True, **kwargs)

    @classmethod
    def clear_default_lrs(cls) -> None:
        """
        Clear the default LRS objects from the LRS_Accessor class.
        """
        cls._default_lrs = []

    def sort_standard(self, return_inverse=False, inplace=False) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray] | None:
        """
        Sort the DataFrame in standard order based on the active LRS columns.
        """
        # Get sorter
        sorter = self.events.sort_standard(return_inverse=True)[1]
        # Apply changes to the DataFrame
        df = self.df.iloc[sorter]
        if inplace:
            self._df = df
            return
        else:
            return (df, sorter) if return_inverse else df
        
    @_method_require(is_linear=True, is_spatial=True)
    def get_chains(self, name='chain') -> pd.Series:
        """
        Identify the chain indices for each event in the dataframe based on 
        contiguous linear geometries within each group.

        Parameters
        ----------
        name : str, default 'chain'
            The name of the chain index column to return.

        Returns
        -------
        chains : pd.Series
            A series of chain indices for each event in the dataframe.
        """
        # Iterate over groups
        index = []
        chains = []
        for group, df in self.iter_groups():
            # Get chain indices
            chains.append(geometry.get_linestring_chains(df[self.geom_col]))
            index.append(df.index.values)
        # Return series
        chains = pd.Series(
            np.concatenate(chains),
            index=np.concatenate(index),
            name=name
        )
        return chains.reindex_like(self.df)
    
    @_method_require(is_linear=True, is_spatial=True)
    def add_chaining(self, name='chain', inplace=False) -> pd.DataFrame | None:
        """
        Add chain indices to the dataframe based on contiguous linear 
        geometries within each group, adding a new column to the dataframe
        and adding the chain column to the active LRS.

        Parameters
        ----------
        name : str, default 'chain'
            The name of the chain index column to return.
        inplace : bool, default False
            Whether to apply changes to the dataframe in place.
        """
        # Validate chain column name
        if name in self.active_lrs.key_col:
            raise ValueError(
                f"Column name '{name}' is already in use as a key column in "
                "the active LRS.")
        # Get chain indices
        chains = self.get_chains(name=name)
        # Apply changes to the DataFrame
        df = self.df if inplace else self.df.copy()
        df[name] = chains
        # Update LRS
        new_lrs = self.active_lrs.copy(deep=True)
        new_lrs.add_key(name)
        df.lr.add_lrs(new_lrs, activate=True)
        return None if inplace else df

    @_method_require(is_linear=True)
    def extend(self, extend_begs=0, extend_ends=0, inplace=False) -> pd.DataFrame | None:
        """
        Extend the begin and end locations of the active LRS by the specified 
        amounts.
        """
        # Extend events
        events = self.events.extend(extend_begs=extend_begs, extend_ends=extend_ends, inplace=False)
        # Apply changes to the DataFrame
        obj = self if inplace else self.copy()
        obj.begs = events.begs
        obj.ends = events.ends
        return None if inplace else obj.df
    
    def shift(self, shift, inplace=False) -> pd.DataFrame | None:
        """
        Shift the events of the active LRS by the specified amount.
        """
        # Shift events
        events = self.events.shift(shift, inplace=False)
        # Apply changes to the DataFrame
        obj = self if inplace else self.copy()
        if self.is_located:
            obj.locs = events.locs
        if self.is_linear:
            obj.begs = events.begs
            obj.ends = events.ends
        return None if inplace else obj.df
    
    @_method_require(is_linear=True)
    def dissolve(
        self, 
        retain=[], 
        sort=False, 
        inverse_index=True, 
        inverse_label='dissolved_index', 
        return_relation=False,
        ) -> pd.DataFrame | tuple[pd.DataFrame, relate.EventsRelation] | None:
        """
        Merge consecutive ranges. For best results, input events should be sorted.

        Parameters
        ----------
        retain : list, default []
            A list of column labels to retain during the dissolve operation.
        sort : bool, default False
            Whether to sort the events before dissolving. If True, results 
            will still be aligned to the original events. Unsorted events
            may produce unexpected results.
        inverse_index : bool, default True
            Whether to append an inverse index to the dissolved events dataframe.
        inverse_label : str, default 'dissolved_index'
            The label for the inverse index column.
        return_relation : bool, default False
            Whether to return an EventsRelation object which describes the
            relationship between the dissolved (left) and original (right) 
            events.
        """
        # Validate input parameters
        if not isinstance(retain, list):
            raise ValueError("Input `retain` must be a list of valid dataframe column labels.")
        # Define key values to retain during dissolve
        if self.is_grouped:
            key_col = self.key_col + retain
        else:
            key_col = retain
        # Dissolve events
        events = self.get_events(key_col=key_col, require=True)
        output = events.dissolve(sort=sort, return_index=True, return_relation=return_relation)
        # Convert events to dataframe
        df = output[0].to_frame(
            index_name=self._df.index.name,
            group_name=key_col,
            loc_name=self.loc_col,
            beg_name=self.beg_col,
            end_name=self.end_col,
        ).lr.lrs_like(self)
        # Append inverse index
        if inverse_index:
            df[inverse_label] = output[1]
        if return_relation:
            output[-1].left_df = df
            output[-1].right_df = self.df
            return (df, output[-1])
        else:
            return df
    
    def relate(self, other, cache=True) -> relate.EventsRelation:
        """
        Create an events data relationship between two linearly referenced
        datasets.

        Parameters
        ----------
        other : DataFrame
            The other DataFrame to relate with. Must be linearly referenced.
        cache : bool, default True
            Whether to cache computed relationship operations, such as 
            intersections and overlays, for faster subsequent operations. For 
            one-time operations or to save on memory use for large datasets, 
            set cache=False.
        """
        # Create relationship
        return self.events.relate(
            other=other.lr.events,
            cache=cache
        )
    
    def overlay(self, other, normalize=False, norm_by='right', chunksize=1000, grouped=True) -> sp.csr_array:
        """
        Overlay two sets of linearly referenced datasets, computing the 
        length or proportion of overlap between each pair of events.

        Parameters
        ----------
        other : DataFrame
            The other DataFrame to overlay with. Must be linearly referenced.
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
        # Perform overlay
        return self.events.overlay(
            other=other.lr.events,
            normalize=normalize,
            norm_by=norm_by,
            chunksize=chunksize,
            grouped=grouped
        )

    def intersect(self, other, enforce_edges=True, chunksize=1000, grouped=True) -> sp.csr_array:
        """
        Identify intersections between two sets of linearly referenced datasets.

        Parameters
        ----------
        other : DataFrame
            The other DataFrame to intersect with. Must be linearly referenced.
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
        # Perform intersect
        return self.events.intersect(
            other=other.lr.events,
            enforce_edges=enforce_edges,
            chunksize=chunksize,
            grouped=grouped
        )
    