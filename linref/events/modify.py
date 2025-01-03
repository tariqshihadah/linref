import numpy as np
from linref.events import base, utility

def dissolve(rng, return_index=False):
    """
    Merge consecutive ranges. For best results, input events should be sorted.
    """
    # Validate input
    if not isinstance(rng, base.Rangel):
        raise TypeError("Input object must be a Rangel class instance.")
    if not rng.is_linear:
        raise ValueError("Input object must be a linear Rangel instance.")
    if rng.is_empty:
        return base.Rangel()
    
    # Identify edges of dissolvable events
    consecutive_strings = rng.consecutive_strings()
    string_number, string_start = \
        np.unique(consecutive_strings, return_index=True)
    
    # Initialize new event edges
    index = []
    groups = [rng.groups[0] if rng.groups is not None else None]
    begs = [rng.begs[0]]
    ends = []
    
    # Get min and max bounds of dissolved events
    if len(string_start) > 1:
        for i, j in zip(string_start[:-1], string_start[1:]):
            ends.append(rng.ends[j - 1])
            begs.append(rng.begs[j])
            index.append(rng.index[i:j])
            groups.append(rng.groups[j] if rng.groups is not None else None)
    
    # Add final end point
    ends.append(rng.ends[-1])
    index.append(rng.index[string_start[-1]:])
    
    # Prepare output class instance
    res = rng.from_similar(
        index=None,
        groups=groups if rng.groups is not None else None,
        begs=begs, 
        ends=ends, 
        closed=rng.closed
    )

    # Return results
    if return_index:
        return res, index
    return res

def concatenate(rngs, ignore_index=False, closed=None):
    """
    Concatenate multiple ranges of events, returning a single collection.

    Parameters
    ----------
    rngs : list
        List of Rangel instances to concatenate.
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
    if not isinstance(rngs, list):
        raise TypeError("Input must be a list of Rangel class instances.")
    if len(rngs) == 0:
        raise ValueError("No events to concatenate.")
    if not all(isinstance(rng, base.Rangel) for rng in rngs):
        raise TypeError("All input objects must be Rangel class instances.")
    # Ensure all objects have the same characteristics
    test_linear = [rng.is_linear for rng in rngs]
    test_located = [rng.is_located for rng in rngs]
    test_grouped = [rng.is_grouped for rng in rngs]
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
    if (not all(rng.is_linear for rng in rngs)) and closed is not None:
        raise ValueError(
            "The 'closed' parameter is only applicable to linear events.")
    
    # Identify event structure
    is_linear = test_linear[0]
    is_located = test_located[0]
    is_grouped = test_grouped[0]

    # Concatenate events
    if is_located:
        locs = np.concatenate([rng.locs for rng in rngs])
    else:
        locs = None
    if is_linear:
        begs = np.concatenate([rng.begs for rng in rngs])
        ends = np.concatenate([rng.ends for rng in rngs])
    else:
        begs = None
        ends = None
    if is_grouped:
        groups = np.concatenate([rng.groups for rng in rngs])
    else:
        groups = None
    if ignore_index:
        index = None
    else:
        index = np.concatenate([rng.index for rng in rngs])

    # Return concatenated events
    return rngs[0].from_similar(
        index=index,
        groups=groups,
        locs=locs,
        begs=begs,
        ends=ends,
        closed=closed if closed is not None else rngs[0].closed
    )

def extend(rng, extend_begs=0, extend_ends=0, inplace=False):
    """
    Extend the range of events by a specified amount in either or both directions.

    Parameters
    ----------
    rng : Rangel
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
    if not isinstance(rng, base.Rangel):
        raise TypeError("Input object must be a Rangel class instance.")
    extend_begs = utility._validate_scalar_or_array_input(rng, extend_begs, 'extend_begs')
    extend_ends = utility._validate_scalar_or_array_input(rng, extend_ends, 'extend_ends')

    # Select object to modify
    rng = rng if inplace else rng.copy()

    # Select methodology
    if rng.is_point:
        rng._begs = rng.locs - extend_begs
        rng._ends = rng.locs + extend_ends
    else:
        rng._begs = rng._begs - extend_begs
        rng._ends = rng._ends + extend_ends
    
    # Return results
    return None if inplace else rng

def shift(rng, shift, inplace=False):
    """
    Shift the range of events by a specified amount.

    Parameters
    ----------
    rng : Rangel
        Input range of events.
    shift : float or array-like
        Amount to shift all events. If an array-like is provided, it must
        be the same length as the number of events in the collection. Positive
        values shift events to the right, negative values to the left.
    inplace : bool, optional
        If True, modify the input object in place. Default is False.
    """
    # Validate input
    if not isinstance(rng, base.Rangel):
        raise TypeError("Input object must be a Rangel class instance.")
    shift = utility._validate_scalar_or_array_input(rng, shift, 'shift')

    # Select object to modify
    rng = rng if inplace else rng.copy()

    # Select methodology
    if rng.is_located:
        rng._locs = rng._locs + shift
    if rng.is_linear:
        rng._begs = rng._begs + shift
        rng._ends = rng._ends + shift
    
    # Return results
    return None if inplace else rng

def round(rng, decimals=None, factor=None, inplace=False):
    """
    Round the bounds and locations of events to a specified number of decimal 
    places or using a specified rounding factor.

    Parameters
    ----------
    rng : Rangel
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
    if not isinstance(rng, base.Rangel):
        raise TypeError("Input object must be a Rangel class instance.")
    if decimals is not None:
        if not isinstance(decimals, int):
            raise TypeError("'decimals' must be an integer.")
        _rounder = lambda x: np.round(x, decimals=decimals)
    elif factor is not None:
        factor = utility._validate_scalar_or_array_input(rng, factor, 'factor', nonzero=True)
        _rounder = lambda x: np.round(x / factor, decimals=0) * factor
    else:
        raise ValueError("Either 'decimals' or 'factor' must be provided.")
    
    # Select object to modify
    rng = rng if inplace else rng.copy()

    # Select methodology
    
    if rng.is_located:
        rng._locs = _rounder(rng._locs)
    if rng.is_linear:
        rng._begs = _rounder(rng._begs)
        rng._ends = _rounder(rng._ends)

    # Return results
    return None if inplace else rng

def separate(rng, by='centers', inplace=False):
    """
    Address overlapping ranges by distributing overlaps between adjacent
    events. Distributions are made equally and are based on a specified 
    event anchor point of the location, begin, end, or center of each 
    event.

    Parameters
    ----------
    rng : Rangel
        Input range of events.
    by : str {'locs', 'begs', 'ends', 'centers'}, default 'centers'
        The anchor point of each event to be used when distributing 
        overlaps between events.
    inplace : bool, optional
        If True, modify the input object in place. Default is False.
    """
    # Validate input
    if not isinstance(rng, base.Rangel):
        raise TypeError("Input object must be a Rangel class instance.")
    if not rng.is_linear:
        raise ValueError("Input object must be a linear Rangel instance.")
    if rng.is_empty:
        raise ValueError("No events to separate.")
    if not by in [None, 'locs', 'begs', 'ends', 'centers']:
        raise ValueError("Separate 'by' must be either 'locs', 'begs', "
            "'ends', 'centers' or None.")

    # Select object to modify
    modified = rng if inplace else rng.copy()

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