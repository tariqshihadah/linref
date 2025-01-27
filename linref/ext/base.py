import numpy as np
import copy, hashlib
from pandas.api.extensions import register_dataframe_accessor
from linref.ext.utility import label_list_or_none, label_or_none
from linref.events.common import closed_all
from linref.events.base import Rangel
from linref.events.utility import _method_require

class LRS(object):

    def __init__(self, keys_col=None, locs_col=None, begs_col=None, ends_col=None, geom_col=None, closed=None):
        # Validate LRS column labels
        self.keys_col = label_list_or_none(keys_col)
        self.locs_col = label_or_none(locs_col)
        self.begs_col = label_or_none(begs_col)
        self.ends_col = label_or_none(ends_col)
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
        return f"LRS(keys_col={self.keys_col}, locs_col={self.locs_col}, begs_col={self.begs_col}, ends_col={self.ends_col}, geom_col={self.geom_col}, closed={self.closed})"
    
    @property
    def is_linear(self):
        return (self.begs_col is not None) and (self.ends_col is not None)
    
    @property
    def is_point(self):
        return (self.locs_col is not None) and (self.begs_col is None) and (self.ends_col is None)
    
    @property
    def is_located(self):
        return self.locs_col is not None
    
    @property
    def is_grouped(self):
        return self.keys_col is not None
    
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
            missing_keys = [key for key in self.keys_col if key not in df.columns]
            result['keys'] = {'valid': len(missing_keys) == 0, 'missing': missing_keys}
        if self.is_linear:
            missing_linear = [col for col in [self.begs_col, self.ends_col] if col not in df.columns]
            result['linear'] = {'valid': len(missing_linear) == 0, 'missing': missing_linear}
        if self.is_located:
            valid = self.locs_col in df.columns
            result['location'] = {'valid': valid, 'missing': self.locs_col if not valid else None}
        if self.is_spatial:
            valid = self.geom_col in df.columns
            result['geometry'] = {'valid': valid, 'missing': self.geom_col if not valid else None}
        return result


@register_dataframe_accessor("lr")
class LRS_Accessor(object):

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
            lrs_lines = '\n'.join(['- ' + str(o) for o in self._lrs])
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
    
    @property
    def is_lrs_set(self):
        return len(self._lrs) > 0

    @property
    def active_index(self):
        return self._active_index
    
    @property
    def active_lrs(self):
        if not self.is_lrs_set:
            raise ValueError("No LRS set for the DataFrame.")
        return self._lrs[self.active_index]
    
    @property
    def keys_col(self):
        return self.active_lrs.keys_col
    
    @property
    def locs_col(self):
        return self.active_lrs.locs_col
    
    @property
    def begs_col(self):
        return self.active_lrs.begs_col
    
    @property
    def ends_col(self):
        return self.active_lrs.ends_col
    
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
        col = self.locs_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @locs.setter
    def locs(self, values):
        if self.locs_col is None:
            raise ValueError("No locations column in the LRS.")
        # Set location values in the DataFrame
        col = self.locs_col
        self._df[col] = values
                
    @property
    def begs(self):
        # Select data from the dataframe if begin locations are present
        col = self.begs_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @begs.setter
    def begs(self, values):
        if self.begs_col is None:
            raise ValueError("No begins column in the LRS.")
        # Set begin location values in the DataFrame
        col = self.begs_col
        self._df[col] = values
        
    @property
    def ends(self):
        # Select data from the dataframe if end locations are present
        col = self.ends_col
        try:
            return self._df[col].values
        except KeyError:
            return None
        
    @ends.setter
    def ends(self, values):
        if self.ends_col is None:
            raise ValueError("No ends column in the LRS.")
        # Set end location values in the DataFrame
        col = self.ends_col
        self._df[col] = values
    
    @property
    def is_grouped(self):
        """
        Return whether the active LRS is grouped and the key columns are 
        present in the dataframe.
        """
        if self.keys_col is None:
            return False
        else:
            # Check for presence of key columns in the dataframe
            return all([key in self._df.columns for key in self.keys_col])
    
    @property
    def is_linear(self):
        """
        Return whether the active LRS is linear and the begin and end columns
        are present in the dataframe.
        """
        if self.begs_col is None or self.ends_col is None:
            return False
        else:
            # Check for presence of begin and end columns in the dataframe
            return all([col in self._df.columns for col in [self.begs_col, self.ends_col]])
    
    @property
    def is_point(self):
        """
        Return whether the active LRS is point-based and the location column is
        present in the dataframe.
        """
        if self.locs_col is None:
            return False
        elif self.begs_col is not None or self.ends_col is not None:
            return False
        else:
            # Check for presence of location column in the dataframe
            return self.locs_col in self._df.columns
    
    @property
    def is_located(self):
        """
        Return whether the active LRS is located and the location column is 
        present in the dataframe.
        """
        if self.locs_col is None:
            return False
        else:
            # Check for presence of location column in the dataframe
            return self.locs_col in self._df.columns
    
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
            col = self.keys_col
        if col is None:
            return None
        try:
            # Convert data to list of tuples
            return self._df[col].to_records(index=False)
        except KeyError as e:
            if require:
                raise e
            return None
        
    def get_events(self, keys_col=None, require=True):
        """
        Return the events object for the active LRS.
        """
        # Get keys if needed
        if keys_col is None:
            keys = self.keys
        else:
            keys = self.get_keys(col=keys_col, require=require)
        # Create the events object
        return Rangel(
            index=self.index,
            groups=keys,
            locs=self.locs if self.locs_col else None,
            begs=self.begs if self.begs_col else None,
            ends=self.ends if self.ends_col else None,
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
        """
        if index >= len(self._lrs):
            raise ValueError(
                f"Invalid LRS index: {index}. Must be less than {len(self._lrs)}.")
        self._active_index = index

    def set_lrs(self, lrs, append=False):
        # Validate LRS object type
        if isinstance(lrs, LRS):
            lrs = [lrs]
        elif not all([isinstance(lrs, LRS) for lrs in lrs]):
            raise ValueError("Input LRS objects must be of type `LRS`.")
        
        # Append or replace LRS objects
        if append:
            self._lrs.extend(lrs)
        else:
            self._lrs = lrs
        
    def add_lrs(self, lrs):
        self.set_lrs(lrs, append=True)

    def clear_lrs(self):
        self._lrs = []

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
    def dissolve(self, retain=[], inverse_index=True, inverse_label='dissolved_index'):
        """
        Merge consecutive ranges. For best results, input events should be sorted.

        Parameters
        ----------
        retain : list, default []
            A list of column labels to retain during the dissolve operation.
        inverse_index : bool, default True
            Whether to append an inverse index to the dissolved events dataframe.
        inverse_label : str, default 'dissolved_index'
            The label for the inverse index column.
        """
        # Validate input parameters
        if not isinstance(retain, list):
            raise ValueError("Input `retain` must be a list of valid dataframe column labels.")
        # Define key values to retain during dissolve
        if self.is_grouped:
            keys_col = self.keys_col + retain
        else:
            keys_col = retain
        # Dissolve events
        events = self.get_events(keys_col=keys_col, require=True)
        dissolved, index = events.dissolve(return_index=True)
        # Convert events to dataframe
        df = dissolved.to_frame(
            index_name=self._df.index.name,
            group_name=keys_col,
            loc_name=self.locs_col,
            beg_name=self.begs_col,
            end_name=self.ends_col,
        )
        # Append inverse index
        if inverse_index:
            df[inverse_label] = index
        return df

def _only_if_hashing(m):
    def wrapper(*args, **kwargs):
        if args[0].hashing:
            return m(*args, **kwargs)
    return wrapper


class LRS_Manager:
    """
    Class for managing linear referencing system data in `pandas` DataFrames.
    """

    def __init__(self, df, lrs, hashing=True):
        # Log DataFrame and LRS objects
        self.df = df
        self.lrs = lrs
        self.hashing = hashing

        # Set column indices for LRS keys and location information to improve performance
        self._set_column_indices()

        # Create data hashes to log current dataframe state
        self._hash_columns(save=True, compare=False)
        self._hash_lrs_data(save=True, compare=False)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input DataFrame must be of type `pandas.DataFrame`.")
        self._df = df
        
    def _set_column_indices(self):
        """
        Set the indices of the columns in the DataFrame that correspond to the LRS keys.
        """
        # Get the indices of the LRS keys in the DataFrame
        self._key_indices = [self.df.columns.get_loc(key) for key in self.lrs.keys]
        self._beg_index = self.df.columns.get_loc(self.lrs.beg) if self.lrs.beg else None
        self._end_index = self.df.columns.get_loc(self.lrs.end) if self.lrs.end else None
        self._loc_index = self.df.columns.get_loc(self.lrs.loc) if self.lrs.loc else None
        self._geom_index = self.df.columns.get_loc(self.lrs.geom) if self.lrs.geom else None

    @_only_if_hashing
    def _hash_columns(self, save=False, compare=False):
        """ 
        Hash the columns of the DataFrame to ensure that the LRS is applied to the correct columns.
        """
        # Hash the columns of the DataFrame
        columns_hash = hashlib.sha256(self.df.columns.to_numpy()).hexdigest()

        # Save hash information
        if compare:
            # Log comparisons with previous hashes
            self._columns_hash_status = self._columns_hash == columns_hash
        else:
            # Set comparison to successful
            self._columns_hash_status = True
        if save:
            # Save latest hashes
            self._columns_hash = columns_hash

    @_only_if_hashing
    def _hash_lrs_data(self, save=False, compare=False):
        """
        Hash the LRS data to easily check for changes to LRS locational information.
        """
        # Hash key columns
        key_hash = hashlib.sha256(self.df.iloc[:, self._key_indices].to_numpy()).hexdigest()
        # Hash location columns
        if self.lrs.is_linear:
            beg_hash = hashlib.sha256(self.df.iloc[:, self._beg_index].to_numpy()).hexdigest()
            end_hash = hashlib.sha256(self.df.iloc[:, self._end_index].to_numpy()).hexdigest()
        else:
            beg_hash = None
            end_hash = None
        if self.lrs.is_point or self.lrs.is_locational:
            loc_hash = hashlib.sha256(self.df.iloc[:, self._loc_index].to_numpy()).hexdigest()
        else:
            loc_hash = None
        # Hash geometry columns
        if self.lrs.geom:
            geom_hash = hashlib.sha256(self.df.iloc[:, self._geom_index].to_numpy()).hexdigest()
        else:
            geom_hash = None

        # Save hash information
        if compare:
            # Log comparisons with previous hashes
            self._key_hash_status = self._key_hash == key_hash
            self._beg_hash_status = self._beg_hash == beg_hash
            self._end_hash_status = self._end_hash == end_hash
            self._loc_hash_status = self._loc_hash == loc_hash
            self._geom_hash_status = self._geom_hash == geom_hash
        else:
            # Set comparison to successful
            self._key_hash_status = True
            self._beg_hash_status = True
            self._end_hash_status = True
            self._loc_hash_status = True
            self._geom_hash_status = True
        if save:
            # Save latest hashes
            self._key_hash = key_hash
            self._beg_hash = beg_hash
            self._end_hash = end_hash
            self._loc_hash = loc_hash
            self._geom_hash = geom_hash



#@pd.api.extensions.register_dataframe_accessor("lrs")
class LRS_Accessor:

    def __init__(self, obj):
        # Log extended DataFrame
        self._obj = obj
        # Set null LRS
        self._lrs = []
        self._lrs_data = []

    def __repr__(self):
        if self.is_lrs_set:
            lrs_lines = '\n'.join(['- ' + str(o) for o in self._lrs])
        else:
            lrs_lines = "- No LRS set"
        return "LRS_Accessor with linear referencing system (LRS) objects:\n" + lrs_lines

    def __str__(self):
        if self.is_lrs_set:
            lrs_lines = '\n'.join(['- ' + str(o) for o in self._lrs])
        else:
            lrs_lines = "- No LRS set"
        return "LRS_Accessor with linear referencing system (LRS) objects:\n" + lrs_lines

    @property
    def lrs(self):
        return self._lrs

    @property
    def lrs_managers(self):
        return self._lrs_managers

    @property
    def managers(self):
        return self._lrs_managers

    @lrs.setter
    def lrs(self, lrs):
        # Check for valid LRS objects and create managers
        lrs_managers = [LRS_Manager(self._obj, lrs) for lrs in lrs]
        # Set LRS objects
        self._lrs = lrs
        self._lrs_managers = lrs_managers

    @property
    def is_lrs_set(self):
        return len(self._lrs) > 0

    def set_lrs(self, lrs=None, **kwargs):
        """
        Set one or more linear referencing systems (LRS) for the DataFrame. The LRS objects can be provided as a 
        single `LRS` object or a list of `LRS` objects using the `lrs` keyword argument, or as a set of keyword 
        arguments to create a new `LRS` object.
        """
        # Check for valid LRS objects
        if lrs is not None:
            if isinstance(lrs, LRS):
                lrs = [lrs]
            elif isinstance(lrs, list):
                if not all([isinstance(lrs, LRS) for lrs in lrs]):
                    raise ValueError("Input LRS objects must be of type `LRS`.")
            else:
                raise ValueError("Input LRS objects must be of type `LRS` or a list of `LRS` objects.")
        else:
            # Create LRS objects from keyword arguments
            lrs = [LRS(**kwargs)]
        # Set LRS objects
        self.lrs = lrs

    def clear_lrs(self):
        """
        Clear all linear referencing systems (LRS) from the DataFrame.
        """
        self._lrs = []