from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import pandas as pd
import copy, hashlib
from scipy import sparse as sp
import warnings

# Import helper modules
from linref.events import common, utility, relate, modify, selection, analyze
from linref.errors import LRSConfigurationError


class EventsData:
    """
    Class for managing collections of events with linear or point data.
    """

    def __init__(
            self, 
            index: np.ndarray = None, 
            groups: np.ndarray = None, 
            locs: np.ndarray = None, 
            begs: np.ndarray = None, 
            ends: np.ndarray = None, 
            closed: np.ndarray = None, 
            dtype: np.dtype = float, 
            copy: bool = None,
            force_monotonic: bool = True,
            **kwargs
        ):
        # Validate inputs
        self._validate_data(index, groups, locs, begs, ends, dtype=dtype, copy=copy, **kwargs)
        self.set_closed(closed, inplace=True)
        self._dtype = dtype
        # Prepare data
        if force_monotonic and self.is_linear:
            self.set_monotonic(inplace=True)

    def __str__(self):
        return utility._stringify_instance(self)
    
    def __repr__(self):
        return utility._represent_records(self)
    
    def __getitem__(self, index):
        return self.select(index, ignore=False, inplace=False)
    
    def __len__(self):
        return self.num_events

    @property
    def index(self) -> np.ndarray:
        """
        Event index.
        """
        return self._index
    
    @property
    def generic_index(self) -> np.ndarray:
        """
        Generic 0-based integer index.
        """
        return np.arange(self.num_events, dtype=int)
    
    @property
    def index_data(self) -> np.ndarray:
        """
        Event index data.
        """
        if self.index is not None:
            return self.index
        else:
            return self.generic_index

    @property
    def groups(self) -> np.ndarray | None:
        """
        Event reference groups.
        """
        return self._groups
    
    @property
    def groups_hashed(self) -> np.ndarray | None:
        """
        Event reference groups hashed.
        """
        if self.is_grouped:
            return np.array([hashlib.sha256(x.encode()).hexdigest() for x in self.groups])
        else:
            return None
        
    @property
    def groups_data(self) -> np.ndarray:
        """
        Event reference groups data.
        """
        if self.is_grouped:
            return self.groups
        else:
            return np.zeros(self.num_events, dtype=object)

    @property
    def locs(self) -> np.ndarray | None:
        """
        Event reference positions.
        """
        return self._locs
    
    @property
    def begs(self) -> np.ndarray | None:
        """
        Event begin positions.
        """
        return self._begs
    
    @property
    def ends(self) -> np.ndarray | None:
        """
        Event end positions.
        """
        return self._ends
    
    @property
    def lengths(self) -> np.ndarray | None:
        """
        Event lengths. If the events are points, lengths are zero.
        """
        if self.is_point:
            return None
        else:
            return self._ends - self._begs
        
    @property
    def centers(self) -> np.ndarray | None:
        """
        Event centers. If the events are points, centers are the locations.
        """
        if self.is_point:
            return self._locs
        else:
            return (self._begs + self._ends) / 2
        
    @property
    def num_events(self) -> int:
        """
        Number of events.
        """
        if self.is_located:
            return len(self._locs)
        elif self.is_linear:
            return len(self._begs)
        elif self.is_grouped:
            return len(self._groups)
        else:
            try:
                return len(self._index)
            except:
                raise ValueError("Cannot determine number of events.")
        
    @property
    def closed(self) -> str:
        """
        Whether the ranges are closed on left, right, both, or neither side.
        """
        return self._closed
    
    @property
    def closed_base(self) -> str:
        """
        Base closed parameter without the 'mod' suffix.
        """
        return self._closed.replace('_mod','')
    
    @property
    def closed_mod(self) -> bool:
        """
        Whether the closed parameter has modified edges.
        """
        return self._closed in ['left_mod','right_mod']
    
    @property
    def arr(self) -> np.ndarray:
        if self.is_point:
            return self._locs.reshape(-1, 1)
        elif self.is_linear and not self.is_located:
            return np.stack((self.begs, self.ends), axis=1)
        elif self.is_linear and self.is_located:
            return np.stack((self.begs, self.ends, self.locs), axis=1)

    @property
    def is_linear(self) -> bool:
        """
        Whether the events are linear.
        """
        # If begs and ends are defined, the events are linear.
        return self._begs is not None and self._ends is not None
    
    @property
    def is_point(self) -> bool:
        """
        Whether the events are points.
        """
        # If begs and ends are not defined but location is defined, the events are points.
        return self._begs is None and self._ends is None and self._locs is not None
    
    @property
    def is_located(self) -> bool:
        """
        Whether the events are located.
        """
        # If locs are defined, the events are located.
        return self._locs is not None
    
    @property
    def is_grouped(self) -> bool:
        """
        Whether the events are grouped.
        """
        # If groups are defined, the events are grouped.
        return self._groups is not None
    
    @property
    def is_monotonic(self) -> bool:
        """
        Whether the events are all monotonic, increasing from begin to end position. 
        If the events are points, they are always monotonic.
        """
        # If the events are linear, check if they are monotonic.
        if self.is_linear:
            return np.all(self._begs <= self._ends)
        # If the events are points, they are always monotonic.
        else:
            return True
        
    @property
    def is_empty(self) -> bool:
        """
        Whether the collection is empty.
        """
        return self.num_events == 0
    
    @property
    def anchors(self) -> list[str]:
        """
        Get the anchor references for the events.
        """
        anchors = []
        if self.is_located:
            anchors.append('locs')
        else:
            if self.is_linear:
                anchors.extend(['begs', 'ends'])
        return anchors
        
    @property
    def modified_edges(self) -> np.ndarray:
        """
        Get indexes of ranges with modified edges. Only applicable when 
        self.closed in {'left_mod','right_mod'}.
        """
        # Check for event type
        if self.is_point:
            edges = np.zeros(self.locs.shape, dtype=bool)
        else:
            # Require minimum ranges
            if self.num_events == 0:
                edges = np.zeros(self.begs.shape, dtype=bool)
            else:
                # Modify test for specific closed cases
                when_one = np.array([], dtype=bool)
                if self.closed in ['left_mod']:
                    # Identify ends of group ranges which will be modified
                    edges = self.next_overlapping(
                        all_=False, when_one=when_one, enforce_edges=True)
                    edges = np.append(~edges, True)
                elif self.closed in ['right_mod']:
                    # Identify ends of group ranges which will be modified
                    edges = self.next_overlapping(
                        all_=False, when_one=when_one, enforce_edges=True)
                    edges = np.append(True, ~edges)
                else:
                    edges = np.zeros(self.begs.shape, dtype=bool)
        return edges

    @property
    def unique_groups(self) -> np.ndarray | None:
        """
        Get unique group values.
        """
        if self.is_grouped:
            return np.unique(self.groups)
        else:
            return None

    def _validate_index(self, index, allow_duplicate_index=False):
        """
        Validate input index as a 1D scalar np.array.
        """
        if index is None:
            # Create a generic zero-based integer index
            index = np.arange(self.num_events, dtype=int)
        else:
            index = utility._prepare_data_array(index, 'index')
            # Check that all indices are unique
            if (len(np.unique(index)) < len(index)) and not allow_duplicate_index:
                warnings.warn(
                    "Input indices are not unique. This may cause unexpected "
                    "behavior when selecting and modifying events.")
        return index
    
    def _validate_groups(self, groups):
        """
        Validate input groups as a 1D scalar np.array.
        """
        if groups is None:
            pass
        else:
            groups = utility._prepare_data_array(groups, 'groups')
        return groups
    
    def _validate_data(
        self,
        index: np.ndarray | None,
        groups: np.ndarray | None,
        locs: np.ndarray | None,
        begs: np.ndarray | None,
        ends: np.ndarray | None,
        dtype: type | None = None,
        copy: bool | None = None,
        allow_duplicate_index: bool = False,
        allow_undefined_events: bool = False,
    ) -> None:
        """
        Validate input data based on the requirements of the class.
        """
        # Check possible valid cases of data input combinations
        data_input_case = (locs is not None, begs is not None, ends is not None)
        data_arrays = {}

        # - Located point events
        if data_input_case == (True, False, False):
            # Check that locs are not passed as an anchor reference
            if isinstance(locs, str):
                raise ValueError(
                    "For located point events, `locs` must be a 1D scalar array-like object."
                )
            # Convert locs to a numpy array
            locs = utility._prepare_data_array(locs, 'locs')
            data_arrays['locs'] = locs

        # - Located linear events
        elif data_input_case == (True, True, True):
            # If locs are passed as an anchor reference, validate
            if isinstance(locs, str):
                if not locs in common.anchors_locs:
                    raise LRSConfigurationError(
                        f"Invalid anchor reference for `locs`. Must be one of: {common.anchors_locs}."
                    )
            # Convert data to numpy arrays
            else:
                locs = utility._prepare_data_array(locs, 'locs', dtype=dtype, copy=copy)
                data_arrays['locs'] = locs
            begs = utility._prepare_data_array(begs, 'begs', dtype=dtype, copy=copy)
            ends = utility._prepare_data_array(ends, 'ends', dtype=dtype, copy=copy)
            data_arrays['begs'] = begs; data_arrays['ends'] = ends

        # - Unlocated linear events
        elif data_input_case == (False, True, True):
            begs = utility._prepare_data_array(begs, 'begs', dtype=dtype, copy=copy)
            ends = utility._prepare_data_array(ends, 'ends', dtype=dtype, copy=copy)
            data_arrays['begs'] = begs; data_arrays['ends'] = ends
        
        # - Invalid input data case
        else:
            if not allow_undefined_events:
                raise LRSConfigurationError(
                    "Invalid input data. Must provide either `locs`, `begs` and `ends`, or both. "
                    f"Received: locs={locs is not None}, begs={begs is not None}, ends={ends is not None}."
                )
            
        # Validate index and groups
        if not index is None:
            index = self._validate_index(index, allow_duplicate_index=allow_duplicate_index)
            data_arrays['index'] = index
        if not groups is None:
            groups = self._validate_groups(groups)
            data_arrays['groups'] = groups

        # Validate equal lengths of data arrays
        data_array_lengths = {k: len(v) for k, v in data_arrays.items()}
        if len(set(data_array_lengths.values())) > 1:
            raise ValueError(
                "Input data arrays must have the same length. "
                f"Data array lengths: {data_array_lengths}."
            )
        
        # Assign validated data to class attributes
        self._groups = groups
        self._locs = locs
        self._begs = begs
        self._ends = ends
        if not index is None:
            self._index = index
        else:
            self.reset_index(inplace=True)
        return

    def to_frame(
        self,
        index_name: str | None = None,
        group_name: str | list[str] | None = None,
        loc_name: str = 'loc',
        beg_name: str = 'beg',
        end_name: str = 'end'
        ) -> pd.DataFrame:
        """
        Convert the collection to a pandas DataFrame.

        Parameters
        ----------
        index_name : str, optional
            Name for the index column in the DataFrame.
        group_name : str, optional
            Name for the group column in the DataFrame.
        loc_name : str, default 'loc'
            Name for the location column in the DataFrame.
        beg_name : str, default 'beg'
            Name for the begin position column in the DataFrame.
        end_name : str, default 'end'
            Name for the end position column in the DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame representation of the events.
        """
        # Define frame data
        data = []
        index = self.index if self.index is not None else self.generic_index
        if self.is_grouped:
            groups = pd.DataFrame(index=index, data=self.groups, columns=group_name)
            data.append(groups)
        if self.is_located:
            locs = pd.DataFrame(index=index, data=self.locs, columns=[loc_name])
            data.append(locs)
        if self.is_linear:
            begs = pd.DataFrame(index=index, data=self.begs, columns=[beg_name])
            ends = pd.DataFrame(index=index, data=self.ends, columns=[end_name])
            data.extend([begs, ends])
        # Concatenate data
        frame = pd.concat(data, axis=1)
        # Set index name
        if not index_name is None:
            frame.index.name = index_name
        return frame
    
    def from_similar(
        self,
        index: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        locs: np.ndarray | None = None,
        begs: np.ndarray | None = None,
        ends: np.ndarray | None = None,
        **kwargs
        ) -> EventsData:
        """
        Create a new instance of the class with similar properties to the 
        current instance.

        Parameters
        ----------
        index : np.ndarray, optional
            Event index for the new instance.
        groups : np.ndarray, optional
            Event groups for the new instance.
        locs : np.ndarray, optional
            Event locations for the new instance.
        begs : np.ndarray, optional
            Event begin positions for the new instance.
        ends : np.ndarray, optional
            Event end positions for the new instance.
        **kwargs : 
            Additional keyword arguments to pass to the new instance.
        """
        # Populate kwargs
        kwargs = {
            'closed': self.closed,
            'dtype': self._dtype,
            **kwargs
        }
        # Create new instance
        return self.__class__(
            index=index, groups=groups, locs=locs, begs=begs, ends=ends, **kwargs)
    
    @utility._method_require(is_grouped=True)
    def group_counts(self) -> dict:
        """
        Return a pair of lists containing unique group labels and their 
        corresponding counts.
        """
        return np.unique(self.groups, return_counts=True)

    def copy(self, deep: bool = False) -> EventsData:
        """
        Create an exact copy of the object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    
    def reset_index(self, inplace: bool = False) -> EventsData | None:
        """
        Reset the index to a generic 0-based index.
        """
        # Create new index
        index = np.arange(self.num_events, dtype=int)
        # Apply changes
        rc = self if inplace else self.copy()
        rc._index = index
        return None if inplace else rc
    
    @utility._method_require(is_linear=True)
    def reset_locs(self, inplace: bool = False) -> EventsData | None:
        """
        Reset the locs to None.
        """
        # Apply changes
        rc = self if inplace else self.copy()
        rc._locs = None
        return None if inplace else rc
    
    def select(self, selector: np.ndarray | slice, ignore: bool = False, inplace: bool = False) -> EventsData | None:
        """
        Select events by index, slice, or boolean mask. Use ignore=True to use 
        a generic, 0-based index, ignoring the current index values.

        Parameters
        ----------
        selector : array-like or slice
            Array-like of event indices, a boolean mask aligned to the events, 
            or a slice object to select events.
        ignore : bool, default False
            Whether to use a generic 0-based index, ignoring the current index 
            values.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        return selection.select(self, selector, ignore=ignore, inplace=inplace)
    
    def select_slice(self, slice_: slice, inplace: bool = False) -> EventsData | None:
        """
        Select events by slice.

        Parameters
        ----------
        slice_ : slice
            Slice object to select events.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        return selection.select_slice(self, slice_, inplace=inplace)

    def select_group(self, group, ungroup: bool | None = None, ignore_missing: bool = True, inplace: bool = False) -> EventsData | None:
        """
        Select events by group.

        Parameters
        ----------
        group : label or array-like
            The label of the group to select or array-like of the same.
        ungroup : bool, default None
            Whether to ungroup the selection, returning the selected events 
            without their group labels. If None and a single group is selected,
            the result will be ungrouped otherwise the group labels will be
            retained.
        ignore_missing : bool, default True
            Whether to ignore missing groups in the selection. If False, an error
            will be raised if any groups are not found in the collection. If True,
            missing groups will be ignored; in cases where no groups are found,
            an empty collection will be returned.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        return selection.select_group(
            self, group, ungroup=ungroup, ignore_missing=ignore_missing, inplace=inplace)
    
    def drop(self, mask: np.ndarray, inplace: bool = False) -> EventsData | None:
        """
        Drop events by boolean mask.

        Parameters
        ----------
        mask : array-like
            Boolean mask aligned to the events.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        return selection.select_mask(self, ~mask, inplace=inplace)
    
    def drop_group(self, group, inplace: bool = False) -> EventsData | None:
        """
        Drop events by group.

        Parameters
        ----------
        group : label or array-like
            The label of the group to drop or array-like of the same.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        return selection.drop_group(self, group, inplace=inplace)
    
    def set_closed(self, closed: str | None = None, inplace: bool = False) -> EventsData | None:
        """
        Change whether ranges are closed on left, right, both, or neither side. 
        
        Parameters
        ----------
        closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
                'neither'}, default 'right'
            Whether collection intervals are closed on the left-side, 
            right-side, both or neither.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        # Check for events type
        if not self.is_linear and not closed is None:
            raise LRSConfigurationError(
                f"Only linear events can have closed parameters. Provided: "
                f"closed={closed}"
            )
        # Validate input closed parameter
        if closed is None:
            closed = common.default_closed
        elif not closed in common.closed_all:
            raise LRSConfigurationError(
                "Collection's closed parameter must be one of "
                f"{common.closed_all}.")
        # Apply changes
        rc = self if inplace else self.copy()
        rc._closed = closed
        rc._closed_base = closed.replace('_mod','')
        return None if inplace else rc

    def ungroup(self, inplace: bool = False) -> EventsData | None:
        """
        Remove group labels from the collection.
        """
        # Apply changes
        rc = self if inplace else self.copy()
        rc._groups = None
        return None if inplace else rc
    
    @utility._method_require(is_linear=True)
    def set_monotonic(self, inplace: bool = False, **kwargs) -> EventsData | None:
        """
        Arrange begin and end positions so that all ranges are increasing.

        Parameters
        ----------
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        # Sort ranges to enforce monotony
        begs, ends = np.sort(np.stack((self.begs, self.ends), axis=0), axis=0)

        # Apply changes
        rc = self if inplace else self.copy()
        rc._begs, rc._ends = begs, ends
        return None if inplace else rc
    
    def argsort(self, by: str | list[str], ascending: bool | list[bool] = True) -> np.ndarray:
        f"""
        Get the indices which would sort the events by a selected event data
        anchor.

        Parameters
        ----------
        by : {common.keys_all}
            The event data property or list of properties by which all events 
            should be sorted.
        ascending : bool or list of bool, default True
            Whether to sort in ascending order. If a single bool, applied to 
            all keys. If a list, must be the same length as `by`.
        """
        # Determine sorting parameters
        if not type(by) is list:
            by = [by]
        if not set(by).issubset(common.keys_all):
            raise ValueError(
                "Input 'by' parameter must be one or more of "
                f"{common.keys_all}.")
        if self.is_point and ('begs' in by or 'ends' in by):
            raise ValueError(
                "Sorting by 'begs' or 'ends' is not available for point events.")
        if not self.is_located and 'locs' in by:
            raise ValueError(
                "Sorting by 'locs' is not available for unlocated events.")
        
        # Validate ascending parameter
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        elif len(ascending) != len(by):
            raise ValueError(
                "'ascending' must be a single bool or a list of the same "
                "length as 'by'.")
        
        # Get the arrays for lexsort (reverse order per numpy lexsort)
        # For descending: negate numeric arrays, rank-invert non-numeric
        keys = []
        for key_name, asc in zip(by[::-1], ascending[::-1]):
            arr = getattr(self, key_name)
            if not asc:
                if np.issubdtype(arr.dtype, np.number):
                    arr = -arr
                else:
                    # For non-numeric (e.g. groups): rank and negate
                    _, inv = np.unique(arr, return_inverse=True)
                    arr = -inv
            keys.append(arr)
        
        # Apply sorting
        index = np.lexsort(keys)
        return index

    def sort(self, by: str | list[str], ascending: bool | list[bool] = True, return_index: bool = False, inplace: bool = False) -> EventsData | tuple[EventsData, np.ndarray] | None:
        f"""
        Sort the events by a selected event data anchor.
        
        Parameters
        ----------
        by : {common.keys_all}
            The event data property or list of properties by which all events 
            should be sorted.
        ascending : bool or list of bool, default True
            Whether to sort in ascending order. If a single bool, applied to 
            all keys. If a list, must be the same length as `by`.
        return_index : bool, default False
            Whether to return an array of the indices which represent the 
            performed sort in addition to the sorted events.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        # Get argsort index
        index = self.argsort(by, ascending=ascending)
        
        # Apply changes
        res = self if inplace else self.copy()
        res = res.select(index, ignore=True, inplace=False)
        res = res if not return_index else (res, index)
        return None if inplace else res
    
    def sort_standard(self, return_index: bool = False, inplace: bool = False) -> EventsData | tuple[EventsData, np.ndarray] | None:
        """
        Sort the events by their positional information in the standard order
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
        # Determine sorting parameters
        by = []
        if self.is_grouped:
            by.append('groups')
        if self.is_linear:
            by.extend(['begs', 'ends'])
        if self.is_located:
            by.append('locs')
        
        # Apply sorting
        return self.sort(by, return_index=return_index, inplace=inplace)
    
    def duplicated(self, subset: list[str] | None = None, keep: str = 'first') -> np.ndarray:
        """
        Return a boolean mask of duplicated events in terms of all or a 
        selection of event anchors.

        Parameters
        ----------
        subset : array-like, default None
            Array-like of event anchors to use for duplicated comparison. If 
            None, all event anchors are used.
        keep : {'first', 'last', 'none'}, default 'first'
            Whether to keep the first, last, or none of the duplicated events.
        """
        return analyze.duplicated(self, subset=subset, keep=keep)
    
    @utility._method_require(is_linear=True)
    def find_same(self, keep: str = 'first') -> np.ndarray:
        """
        Return a boolean mask of events which have the same begin and end 
        points as at least one other event in the collection.

        Parameters
        ----------
        keep : {'first', 'last', 'none'}, default 'first'
            Which of the duplicate events to keep (mark as False).
        """
        return analyze.find_same(self, keep=keep)
    
    @utility._method_require(is_linear=True)
    def find_inside(self, enforce_edges: bool = False) -> np.ndarray:
        """
        Return a boolean mask of events which fall entirely inside at least 
        one other event in the collection.

        Parameters
        ----------
        enforce_edges : bool, default False
            Whether to consider events touching at a vertex as being inside.
        """
        return analyze.find_inside(self, enforce_edges=enforce_edges)
    
    def next_same_group(self, all_: bool = True, when_one: bool = True) -> bool | np.ndarray:
        """
        Whether all or any ranges have the same group as the next range in the
        collection.
        """
        # Validate input
        if self.num_events == 1:
            return when_one
        elif self.num_events == 0:
            raise ValueError("No ranges in collection.")
        
        # Check for same group
        res = self.groups[1:] == self.groups[:-1]
        if all_:
            return res.all()
        else:
            return res
            
    def next_overlapping(self, all_: bool = True, when_one: bool = True, enforce_edges: bool = False) -> bool | np.ndarray:
        """
        Whether all or any ranges are overlapping the next range in the 
        collection.

        Parameters
        ----------
        all_ : bool, default True
            Whether to aggregate all tests of overlapping ranges, returning a 
            single boolean value. If True, will return True if all ranges are 
            overlapping, False if any adjacent ranges are not overlapping. If 
            False, will return an array of shape num_events - 1 of boolean 
            values indicating whether each range is overlapping the next.
        when_one : bool, default True
            The default boolean value to return when only one range is included 
            in the collection.
        enforce_edges : bool, default False
            Whether to consider ranges which have a common vertex as 
            overlapping. This is independent of the collection's closed state.
        """
        # Validate input
        if self.num_events == 1:
            return when_one
        elif self.num_events == 0:
            raise ValueError("No ranges in collection.")

        # Check for overlapping
        cond1 = self.groups[1:] == self.groups[:-1] if self.groups is not None else True
        if enforce_edges:
            cond2 = self.begs[1:] <= self.ends[:-1]
        else:
            cond2 = self.begs[1:] < self.ends[:-1]
        res = cond1 & cond2
        if all_:
            return res.all()
        else:
            return res

    @utility._method_require(is_linear=True, is_monotonic=True, is_empty=False)
    def next_consecutive(self, all_: bool = True, when_one: bool = True) -> bool | np.ndarray:
        """
        Whether all or any ranges are consecutive with the next range in the 
        collection, i.e. the end of one range is the beginning of the next.

        Parameters
        ----------
        all_ : bool, default True
            Whether to aggregate all tests of consecutive ranges, returning a 
            single boolean value. If True, will return True if all ranges are 
            consecutive, False if any adjacent ranges are not consecutive. If 
            False, will return an array of shape num_events - 1 of boolean 
            values indicating whether each range is consecutive to the next.
        when_one : bool, default True
            The default boolean value to return when only one range is included 
            in the collection.
        """
        # Validate input
        if self.num_events == 1:
            return np.array([when_one])
        elif self.num_events == 0:
            raise ValueError("No ranges in collection.")
        
        # Check for consecutive ranges
        if self.is_grouped:
            res = \
                (self.begs[1:] == self.ends[:-1]) & \
                (self.groups[1:] == self.groups[:-1])
        else:
            res = self.begs[1:] == self.ends[:-1]
        if all_:
            return res.all()
        else:
            return res
        
    def consecutive_strings(self) -> np.ndarray:
        """
        Identify strings of consecutive events in the collection, returning 
        an array of integers indicating which consecutive string each event 
        belongs to.
        """
        # Validate input
        if self.num_events == 0:
            raise ValueError("No ranges in collection.")
        elif self.num_events == 1:
            return np.array([0])
        
        # Identify consecutive strings
        res = np.zeros(self.num_events, dtype=int)
        res[1:] = np.cumsum(~self.next_consecutive(all_=False))
        return res

    @utility._method_require(is_grouped=True)
    def iter_groups(self, ungroup: bool = True) -> Iterator[tuple]:
        """
        Iterate over the groups in the collection, yielding the group label
        and the corresponding EventsData for each group.

        Parameters
        ----------
        ungroup : bool, default True
            Whether to ungroup the selected events for each group, returning
            an ungrouped EventsData instance.

        Yields
        ------
        tuple
            A tuple of (group label, EventsData) for each group in the 
            collection.
        """
        # Get group indices with groupby routine
        sorted_events = self.sort_standard(inplace=False)
        unique_groups, splitter_i = np.unique(sorted_events.groups, return_index=True)
        splitter_j = np.append(splitter_i[1:], len(sorted_events.groups))
        
        # Iterate over groups
        for group, i, j in zip(unique_groups, splitter_i, splitter_j):
            events = sorted_events.select(slice(i, j), inplace=False)
            # Ungroup if necessary
            if ungroup:
                events = events.ungroup()
            yield group, events

    @utility._method_require(is_grouped=True)
    def iter_group_indices(self) -> Iterator[tuple]:
        """
        Iterate over the group indices in the collection.
        """
        # Get group indices with groupby routine
        sorted_events = self.sort_standard(inplace=False)
        unique_groups, splitter_i = np.unique(sorted_events.groups, return_index=True)
        splitter_j = np.append(splitter_i[1:], len(sorted_events.groups))
        
        # Iterate over group indices
        for group, i, j in zip(unique_groups, splitter_i, splitter_j):
            indices = sorted_events.index[i:j]
            yield group, indices
        
    @utility._method_require(is_linear=True, is_monotonic=True, is_empty=False)
    def separate(self):
        pass

    @utility._method_require(is_linear=True, is_monotonic=True, is_empty=False)
    def dissolve(
            self,
            sort: bool = False,
            return_index: bool = False,
            return_relation: bool = False
        ) -> EventsData | tuple[EventsData, np.ndarray] | tuple[EventsData, relate.EventsRelation] | tuple[EventsData, np.ndarray, relate.EventsRelation]:
        """
        Dissolve consecutive linear events into single events. For best 
        results, input events should be sorted.

        Parameters
        ----------
        sort : bool, default False
            Whether to sort the events before dissolving. If True, results 
            will still be aligned to the original events. Unsorted events
            may produce unexpected results.
        return_index : bool, default False
            Whether to return a list of arrays indicating the indices of the 
            original events which were dissolved into each new event.
        return_relation : bool, default False
            Whether to return an EventsRelation object which describes the
            relationship between the dissolved (left) and original (right) 
            events.
        """
        return modify.dissolve(
            self,
            sort=sort,
            return_index=return_index,
            return_relation=return_relation
        )

    @utility._method_require(is_linear=True, is_monotonic=True, is_empty=False)
    def resegment(self, length: float = 1, fill: str = 'cut', return_relation: bool = False):
        """
        Resegment events into smaller segments of equal length, addressing 
        edge cases in a variety of ways using the fill parameter.

        Parameters
        ----------
        length : float, optional
            Length of each segment. Default is 1.
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
        
        return_relation : bool, default False
            Whether to return a list of arrays indicating the indices of the 
            original events which were resegmented into each new event.

        Returns
        -------
        linref.events.EventsData
        """
        return modify.resegment(self, length=length, fill=fill, return_relation=return_relation)

    @utility._method_require(is_empty=False)
    def relate(self, other: EventsData, cache: bool = True, **kwargs) -> relate.EventsRelation:
        """
        Create an events data relationship between two collections of events.

        Parameters
        ----------
        other : EventsData
            The other collection of events to relate.
        cache : bool, default True
            Whether to cache computed relationship operations, such as 
            intersections and overlays, for faster subsequent operations. For 
            one-time operations or to save on memory use for large datasets, 
            set cache=False.

        Returns
        -------
        linref.events.relate.EventsRelation
        """
        return relate.EventsRelation(self, other, cache=cache, **kwargs)

    @utility._method_require(is_linear=True, is_monotonic=True, is_empty=False)
    def overlay(self, other: EventsData, normalize: bool = False, norm_by: str = 'right', chunksize: int = 1000, grouped: bool = True) -> sp.csr_matrix:
        """
        Compute the overlay of two collections of events.

        Parameters
        ----------
        left, right : EventsData
            Input EventsData instances to overlay.
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
        # Create relationship
        relation = relate.EventsRelation(self, other, cache=False)
        
        # Perform overlay
        return relation.overlay(
            normalize=normalize,
            norm_by=norm_by,
            chunksize=chunksize
        )

    @utility._method_require(is_empty=False)
    def intersect(self, other: EventsData, enforce_edges: bool = True, chunksize: int = 1000, grouped: bool = True) -> sp.csr_matrix:
        """
        Identify intersections between two collections of events.

        Parameters
        ----------
        other : EventsData
            The other collection of events to test for intersections.
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
        # Create relationship
        relation = relate.EventsRelation(self, other, cache=False)

        # Perform intersection
        return relation.intersect(
            enforce_edges=enforce_edges,
            chunksize=chunksize
        )
    
    def extend(self, extend_begs: float | np.ndarray = 0, extend_ends: float | np.ndarray = 0, inplace: bool = False) -> EventsData | None:
        """
        Extend the range of events by a specified amount in either or both 
        directions.

        Parameters
        ----------
        extend_begs : float or array-like, optional
            Amount to extend the beginning and end of each event range. If an array-like
            is provided, it must be the same length as the number of events in the 
            collection. Positive values extend ranges to the left, negative values to
            the right. Default is 0.
        extend_ends : float or array-like, optional
            Amount to extend the end of each event range. If an array-like is provided,
            it must be the same length as the number of events in the collection. Positive
            values extend ranges to the right, negative values to the left. Default is 0.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        return modify.extend(
            self,
            extend_begs=extend_begs,
            extend_ends=extend_ends,
            inplace=inplace
        )
    
    def shift(self, shift: float | np.ndarray, inplace: bool = False) -> EventsData | None:
        """
        Shift the range of events by a specified amount.

        Parameters
        ----------
        shift : float or array-like
            Amount to shift all events. If an array-like is provided, it must
            be the same length as the number of events in the collection. Positive
            values shift events to the right, negative values to the left.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        return modify.shift(
            self,
            shift=shift,
            inplace=inplace
        )
    
    def round(self, decimals: int | None = None, factor: float | None = None, inplace: bool = False) -> EventsData | None:
        """
        Round the begin and end positions of the events to a specified number 
        of decimals or to the nearest multiple of a specified factor.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to. If not provided, the values 
            will be rounded to the nearest integer.
        factor : float, optional
            Round to the nearest multiple of this factor. If not provided, the 
            values will be rounded to the nearest integer.
        inplace : bool, default False
            Whether to perform the operation in place, returning None.
        """
        return modify.round(
            self,
            decimals=decimals,
            factor=factor,
            inplace=inplace
        )