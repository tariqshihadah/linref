import numpy as np
from linref.events import base, utility, relate
from scipy import sparse as sp

def dissolve(events, sort=False, return_index=False, return_relation=False):
    """
    Merge consecutive ranges. For best results, input events should be sorted.

    Parameters
    ----------
    events : EventsData
        Input range of events to dissolve.
    sort : bool, default False
        Whether to sort the input events before dissolving. If True, results 
        will still be aligned to the original input events. Unsorted events
        may produce unexpected results.
    return_index : bool, default False
        Whether to return the inverse index of the dissolved events as a list
        of arrays of indices to the original events.
    return_relation : bool, default False
        Whether to return an EventsRelation object between the dissolved 
        events and the input events to allow for easy aggregation of data.
    """
    # Validate input
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    if not events.is_linear:
        raise ValueError("Input object must be a linear EventsData instance.")
    if events.is_empty:
        return base.EventsData()
    
    # Sort events if requested
    if sort:
        events_original = events.copy()
        events, events_sorter = \
            events.sort_standard(return_inverse=True, inplace=False)
    
    # Define indices to align inverse index and relation index to the input 
    # events
    INDEX_GENERIC = events_sorter if sort else events.generic_index
    INDEX_INVERSE = events.index
    
    # Identify edges of dissolvable events
    consecutive_strings = events.consecutive_strings()
    string_number, string_start = \
        np.unique(consecutive_strings, return_index=True)
    
    # Initialize new event edges
    indices_generic = []
    indices_inverse = []
    groups = [events.groups[0] if events.groups is not None else None]
    begs = [events.begs[0]]
    ends = [] 
    
    # Get min and max bounds of dissolved events
    if len(string_start) > 1:
        for i, j in zip(string_start[:-1], string_start[1:]):
            ends.append(events.ends[j - 1])
            begs.append(events.begs[j])
            indices_generic.append(INDEX_GENERIC[i:j])
            indices_inverse.append(INDEX_INVERSE[i:j])
            groups.append(events.groups[j] if events.groups is not None else None)
    
    # Add final end point
    ends.append(events.ends[-1])
    indices_generic.append(INDEX_GENERIC[string_start[-1]:])
    indices_inverse.append(INDEX_INVERSE[string_start[-1]:])
    
    # Prepare output class instance
    dissolved = events.from_similar(
        index=None,
        groups=groups if events.groups is not None else None,
        begs=begs, 
        ends=ends, 
        closed=events.closed
    )

    # Return results
    outputs = [dissolved]
    if return_index:
        outputs.append(indices_inverse)
    if return_relation:
        # Prepare relation array
        row_index = np.repeat(
            np.arange(len(indices_generic)),
            np.array(list(map(len, indices_generic)))
        )
        col_index = np.concatenate(indices_generic)
        arr = sp.csr_array(
            (np.ones(len(row_index)), (row_index, col_index)), 
            shape=(len(indices_generic), events.num_events)
        )
        # Prepare relation object
        relation = relate.EventsRelation(dissolved, events, cache=True)
        relation._intersect_data = arr
        relation._intersect_kwargs = {}
        outputs.append(relation)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]

def concatenate(objs, ignore_index=False, closed=None):
    """
    Concatenate multiple ranges of events, returning a single collection.

    Parameters
    ----------
    objs : list
        List of EventsData instances to concatenate.
    ignore_index : bool, default False
        Whether to ignore the index values of the input objects, returning a
        new collection with a new generic 0-based index.
    closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
            'neither'}, optional
        Whether collection intervals are closed on the left-side, 
        right-side, both or neither. If provided, the setting will be applied 
        to the concatenated results. If not provided, the setting will be 
        inferred from the first object in the list for linear events.
    """
    # Validate input
    if not isinstance(objs, list):
        raise TypeError("Input must be a list of EventsData class instances.")
    if len(objs) == 0:
        raise ValueError("No events to concatenate.")
    if not all(isinstance(obj, base.EventsData) for obj in objs):
        raise TypeError("All input objects must be EventsData class instances.")
    # Ensure all objects have the same characteristics
    test_linear = [obj.is_linear for obj in objs]
    test_located = [obj.is_located for obj in objs]
    test_grouped = [obj.is_grouped for obj in objs]
    if not len(set(test_linear)) == 1:
        raise ValueError(
            "All input events must have the same structure. Mix of linear and "
            "non-linear events detected.")
    if not len(set(test_located)) == 1:
        raise ValueError(
            "All input events must have the same structure. Mix of located and "
            "non-located events detected.")
    if not len(set(test_grouped)) == 1:
        raise ValueError(
            "All input events must have the same structure. Mix of grouped and "
            "ungrouped events detected.")
    if (not all(obj.is_linear for obj in objs)) and closed is not None:
        raise ValueError(
            "The 'closed' parameter is only applicable to linear events.")
    
    # Identify event structure
    is_linear = test_linear[0]
    is_located = test_located[0]
    is_grouped = test_grouped[0]

    # Concatenate events
    if is_located:
        locs = np.concatenate([obj.locs for obj in objs])
    else:
        locs = None
    if is_linear:
        begs = np.concatenate([obj.begs for obj in objs])
        ends = np.concatenate([obj.ends for obj in objs])
    else:
        begs = None
        ends = None
    if is_grouped:
        groups = np.concatenate([obj.groups for obj in objs])
    else:
        groups = None
    if ignore_index:
        index = None
    else:
        index = np.concatenate([obj.index for obj in objs])

    # Return concatenated events
    return objs[0].from_similar(
        index=index,
        groups=groups,
        locs=locs,
        begs=begs,
        ends=ends,
        closed=closed if closed is not None else objs[0].closed
    )

def extend(events, extend_begs=0, extend_ends=0, inplace=False):
    """
    Extend the range of events by a specified amount in either or both directions.

    Parameters
    ----------
    events : EventsData
        Input range of events.
    extend_begs : float or array-like, optional
        Amount to extend the beginning and end of each event range. If an array-like
        is provided, it must be the same length as the number of events in the 
        collection. Positive values extend ranges to the left, negative values to
        the right. Default is 0.
    extend_ends : float or array-like, optional
        Amount to extend the end of each event range. If an array-like is provided,
        it must be the same length as the number of events in the collection. Positive
        values extend ranges to the right, negative values to the left. Default is 0.
    inplace : bool, optional
        If True, modify the input object in place. Default is False.
    """
    # Validate input
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    extend_begs = utility._validate_scalar_or_array_input(events, extend_begs, 'extend_begs')
    extend_ends = utility._validate_scalar_or_array_input(events, extend_ends, 'extend_ends')

    # Select object to modify
    events = events if inplace else events.copy()

    # Select methodology
    if events.is_point:
        events._begs = events.locs - extend_begs
        events._ends = events.locs + extend_ends
    else:
        events._begs = events._begs - extend_begs
        events._ends = events._ends + extend_ends
    
    # Return results
    return None if inplace else events

def shift(events, shift, inplace=False):
    """
    Shift the range of events by a specified amount.

    Parameters
    ----------
    events : EventsData
        Input range of events.
    shift : float or array-like
        Amount to shift all events. If an array-like is provided, it must
        be the same length as the number of events in the collection. Positive
        values shift events to the right, negative values to the left.
    inplace : bool, optional
        If True, modify the input object in place. Default is False.
    """
    # Validate input
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    shift = utility._validate_scalar_or_array_input(events, shift, 'shift')

    # Select object to modify
    events = events if inplace else events.copy()

    # Select methodology
    if events.is_located:
        events._locs = events._locs + shift
    if events.is_linear:
        events._begs = events._begs + shift
        events._ends = events._ends + shift
    
    # Return results
    return None if inplace else events

def round(events, decimals=None, factor=None, inplace=False):
    """
    Round the bounds and locations of events to a specified number of decimal 
    places or using a specified rounding factor.

    Parameters
    ----------
    events : EventsData
        Input range of events.
    decimals : int, optional
        Number of decimal places to round to. If an array-like is provided, it must
        be the same length as the number of events in the collection. Default 
        is None.
    factor : float, optional
        Rounding factor. If provided, the bounds and locations of events will be
        rounded to the nearest multiple of this factor. Default is None.
    inplace : bool, optional
        If True, modify the input object in place. Default is False.
    """
    # Validate input
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    if decimals is not None:
        if not isinstance(decimals, int):
            raise TypeError("'decimals' must be an integer.")
        _rounder = lambda x: np.round(x, decimals=decimals)
    elif factor is not None:
        factor = utility._validate_scalar_or_array_input(events, factor, 'factor', nonzero=True)
        _rounder = lambda x: np.round(x / factor, decimals=0) * factor
    else:
        raise ValueError("Either 'decimals' or 'factor' must be provided.")
    
    # Select object to modify
    events = events if inplace else events.copy()

    # Select methodology
    
    if events.is_located:
        events._locs = _rounder(events._locs)
    if events.is_linear:
        events._begs = _rounder(events._begs)
        events._ends = _rounder(events._ends)

    # Return results
    return None if inplace else events

def separate(events, by='centers', inplace=False):
    """
    Address overlapping ranges by distributing overlaps between adjacent
    events. Distributions are made equally and are based on a specified 
    event anchor point of the location, begin, end, or center of each 
    event.

    Parameters
    ----------
    events : EventsData
        Input range of events.
    by : str {'locs', 'begs', 'ends', 'centers'}, default 'centers'
        The anchor point of each event to be used when distributing 
        overlaps between events.
    inplace : bool, optional
        If True, modify the input object in place. Default is False.
    """
    # Validate input
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    if not events.is_linear:
        raise ValueError("Input object must be a linear EventsData instance.")
    if events.is_empty:
        raise ValueError("No events to separate.")
    if not by in [None, 'locs', 'begs', 'ends', 'centers']:
        raise ValueError("Separate 'by' must be either 'locs', 'begs', "
            "'ends', 'centers' or None.")

    # Select object to modify
    modified = events if inplace else events.copy()

    # Prepare sorted events for processing
    modified, inv = modified.sort(
        by=[by, 'lengths'],
        ascending=[True, False],
        inplace=False,
        return_inverse=True
    )

#    # Eliminate concentric and same ranges
#
#
#
#
#    
#    # Eliminate concentric, same, and inside ranges
#    rc = rc.eliminate_concentric(**kwargs).eliminate_same(**kwargs)
#    if eliminate_inside:
#        rc = rc.eliminate_inside(**kwargs)
#    index = np.where(rc.lengths > 0)[0]
#    
#    #---------------#
#    # MODIFY RANGES #
#    #---------------#
#    # Identify the new begin and end points based on computed
#    # midpoints and existing begin and end points
#    rights    = rc.ends[index[:-1]].copy()
#    lefts     = rc.begs[index[1:]].copy()
#    centers_l = rc.centers[index[:-1]].copy()
#    centers_r = rc.centers[index[1:]].copy()
#    
#    # Compute midpoints between consecutive centers
#    center_mids = (centers_l + centers_r) / 2
#    center_mids_valid = (rights >= center_mids) & (lefts <= center_mids)
#    
#    # Compute midpoints between consecutive termini
#    termini_mids = (rights + lefts)/2
#    termini_mids = np.min([np.max([termini_mids, centers_l], axis=0),
#                            centers_r], axis=0)
#    termini_mids_valid = (
#        (rights >= termini_mids) &
#        (lefts <= termini_mids) &
#        (termini_mids >= centers_l)
#    )
#    
#    # Apply termini mids
#    rights[termini_mids_valid] = termini_mids[termini_mids_valid]
#    lefts[termini_mids_valid]  = termini_mids[termini_mids_valid]
#    
#    # Apply center mids
#    rights[center_mids_valid] = center_mids[center_mids_valid]
#    lefts[center_mids_valid]  = center_mids[center_mids_valid]
#
#    # Assign the new begin and end points to the processed ranges
#    rc.reset_centers(inplace=True)
#    rc._ends[index[:-1]] = rights
#    rc._begs[index[1:]]  = lefts
#    rc = rc[inv]
#
#    if inplace:
#        self._begs = rc._begs
#        self._ends = rc._ends
#        self.reset_centers(inplace=True)
#        # Drop short if requested
#        if drop_short:
#            self.drop_short(length=0, inplace=True)
#        return
#    else:
#        # Drop short if requested
#        if drop_short:
#            rc.drop_short(length=0, inplace=True)
#        return rc