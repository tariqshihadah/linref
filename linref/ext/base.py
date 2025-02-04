import numpy as np
import pandas as pd
import copy, hashlib
from pandas.api.extensions import register_dataframe_accessor
from linref.ext.utility import label_list_or_none, label_or_none
from linref.events.common import closed_all
from linref.events.base import EventsData
from linref.events.utility import _method_require
from linref.events import relate


class LRS(object):

    def __init__(self, key_col=None, loc_col=None, beg_col=None, end_col=None, geom_col=None, closed=None):
        # Validate LRS column labels
        self.key_col = label_list_or_none(key_col)
        self.loc_col = label_or_none(loc_col)
        self.beg_col = label_or_none(beg_col)
        self.end_col = label_or_none(end_col)
        self.geom_col = label_or_none(geom_col)
        # Validate LRS closure
        if closed not in closed_all:
            raise ValueError(
                f"Invalid LRS closure: {closed}. Must be one of: {closed_all}.")
        else:
            self.closed = closed

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"LRS(key_col={self.key_col}, loc_col={self.loc_col}, beg_col={self.beg_col}, end_col={self.end_col}, geom_col={self.geom_col}, closed={self.closed})"
    
    @property
    def is_linear(self):
        return (self.beg_col is not None) and (self.end_col is not None)
    
    @property
    def is_point(self):
        return (self.loc_col is not None) and (self.beg_col is None) and (self.end_col is None)
    
    @property
    def is_located(self):
        return self.loc_col is not None
    
    @property
    def is_grouped(self):
        return self.key_col is not None
    
    @property
    def is_spatial(self):
        return self.geom_col is not None

    def study(self, df):
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
        return result


@register_dataframe_accessor("lr")
class LRS_Accessor(object):

    # Initialize default LRS list
    _default_lrs = []

    def __init__(self, df):
        # Log dataframe
        self._df = df
        # Initialize LRS
        self._lrs = []
        self._active_index = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.is_lrs_set:
            lrs_lines = '\n'.join(['- ' + str(o) for o in self.lrs])
        else:
            lrs_lines = "- No LRS set"
        return "LRS_Accessor with linear referencing system (LRS) objects:\n" + lrs_lines

    def __getitem__(self, index):
        """
        Activate an LRS by index.
        """
        self.activate_lrs(index)
        return self
    
    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input DataFrame must be of type `pandas.DataFrame`.")
        self._df = df
    
    @property
    def lrs(self):
        if len(self._lrs) > 0:
            return self._lrs
        return self._default_lrs
        
    @property
    def is_lrs_set(self):
        return len(self.lrs) > 0

    @property
    def active_index(self):
        return self._active_index
    
    @property
    def active_lrs(self):
        if not self.is_lrs_set:
            raise ValueError("No LRS set for the DataFrame.")
        return self.lrs[self.active_index]
    
    @property
    def key_col(self):
        return self.active_lrs.key_col
    
    @property
    def loc_col(self):
        return self.active_lrs.loc_col
    
    @property
    def beg_col(self):
        return self.active_lrs.beg_col
    
    @property
    def end_col(self):
        return self.active_lrs.end_col
    
    @property
    def geom_col(self):
        return self.active_lrs.geom_col
    
    @property
    def closed(self):
        return self.active_lrs.closed
    
    @property
    def index(self):
        return self._df.index.values
    
    @property
    def keys(self):
        return self.get_keys(require=False)
        
    @property
    def locs(self):
        # Select data from the dataframe if locations are present
        col = self.loc_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @locs.setter
    def locs(self, values):
        if self.loc_col is None:
            raise ValueError("No locations column in the LRS.")
        # Set location values in the DataFrame
        col = self.loc_col
        self._df[col] = values
                
    @property
    def begs(self):
        # Select data from the dataframe if begin locations are present
        col = self.beg_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @begs.setter
    def begs(self, values):
        if self.beg_col is None:
            raise ValueError("No begins column in the LRS.")
        # Set begin location values in the DataFrame
        col = self.beg_col
        self._df[col] = values
        
    @property
    def ends(self):
        # Select data from the dataframe if end locations are present
        col = self.end_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @ends.setter
    def ends(self, values):
        if self.end_col is None:
            raise ValueError("No ends column in the LRS.")
        # Set end location values in the DataFrame
        col = self.end_col
        self._df[col] = values
    
    @property
    def is_grouped(self):
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
    def is_linear(self):
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
    def is_point(self):
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
    def is_located(self):
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
    def is_spatial(self):
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
    def events(self):
        """
        Return the events object for the active LRS.
        """
        # Create the events object
        return self.get_events()
    
    def get_keys(self, col=None, require=True):
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
        
    def get_events(self, key_col=None, require=True):
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
    
    def copy(self, deep=False):
        """
        Create an exact copy of the object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    
    def activate_lrs(self, index):
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

    def set_lrs(self, lrs=None, append=False, **kwargs):
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
        
    def add_lrs(self, lrs=None, **kwargs):
        self.set_lrs(lrs=lrs, append=True, **kwargs)

    def clear_lrs(self):
        self._lrs = []

    @classmethod
    def set_default_lrs(cls, lrs=None, append=False, **kwargs):
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
    def add_default_lrs(cls, lrs=None, **kwargs):
        cls.set_default_lrs(lrs=lrs, append=True, **kwargs)

    @classmethod
    def clear_default_lrs(cls):
        cls._default_lrs = []

    def sort_standard(self, return_inverse=False, inplace=False):
        """
        Sort the DataFrame in standard order based on the active LRS columns.
        """
        # Get sorter
        sorter = self.events.sort_standard(return_inverse=True)[1]
        # Apply changes to the DataFrame
        df = self.df
        if inplace:
            df = df.iloc[sorter]
            return
        else:
            df = df.iloc[sorter]
            return (df, sorter) if return_inverse else df

    @_method_require(is_linear=True)
    def extend(self, extend_begs=0, extend_ends=0, inplace=False):
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
    
    def shift(self, shift, inplace=False):
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
    def dissolve(self, retain=[], sort=False, inverse_index=True, inverse_label='dissolved_index', return_relation=False):
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
        )
        # Append inverse index
        if inverse_index:
            df[inverse_label] = output[1]
        return df, output[-1] if return_relation else df
    
    def relate(self, other, cache=True):
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
    
    def overlay(self, other, normalize=True, norm_by='right', chunksize=1000, grouped=True):
        """
        Overlay two sets of linearly referenced datasets, computing the 
        length or proportion of overlap between each pair of events.

        Parameters
        ----------
        other : DataFrame
            The other DataFrame to overlay with. Must be linearly referenced.
        normalize : bool, default True
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

    def intersect(self, other, enforce_edges=True, chunksize=1000, grouped=True):
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
    