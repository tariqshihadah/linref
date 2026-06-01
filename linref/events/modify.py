from __future__ import annotations
import numpy as np
from linref.events import base, utility, relate, common, selection
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
            events.sort_standard(return_index=True, inplace=False)
    
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
            (np.ones(len(row_index), dtype=bool), (row_index, col_index)), 
            shape=(len(indices_generic), events.num_events)
        )
        # Prepare relation object using original events if sorted
        events_for_relation = events_original if sort else events
        relation = relate.EventsRelation(dissolved, events_for_relation, cache=True)
        relation._intersect_data = arr
        relation._intersect_kwargs = {'enforce_edges': False}
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

def resegment(events, length=1, fill='cut', return_relation=False):
    """
    Resegment events into smaller segments of equal length.

    Parameters
    ----------
    events : EventsData
        Input range of events.
    length : float, optional
        Length of each segment. Default is 1.
    fill : {'none','cut','left','right','extend','balance'}, default 'cut'
        How to fill a gap at the end of the input range.

        - ``none`` : no range will be generated to fill the gap at the end of
          the input range.
        - ``cut`` : a truncated range will be created to fill the gap with a
          length less than the full range length.
        - ``left`` : the final range will be anchored on the end value and
          will extend the full range length to the left.
        - ``right`` : the final range will be anchored on the grid defined by
          the step value, extending the full range length to the right,
          beyond the defined end value.
        - ``extend`` : the final range will be anchored on the grid defined
          by the step value, extending beyond the step length to the right
          bound of the range.
        - ``balance`` : if the final range is greater than or equal to half
          the target range length, perform the cut method; if it is less,
          perform the extend method.

        Schematics::

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
        Whether to return an EventsRelation object between the resegmented
        events and the input events to allow for easy aggregation of data.

    Returns
    -------
    linref.events.EventsData
    """
    # Validate input
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    if not events.is_linear:
        raise ValueError("Input object must be a linear EventsData instance.")
    if events.is_empty:
        raise ValueError("No events to resegment.")
    if not isinstance(length, (int, float)):
        raise TypeError("'length' must be a numeric value.")
    if not fill in common.segment_fill_all:
        raise ValueError(f"'fill' must be one of {common.segment_fill_all}.")
    
    # Determine number of segments to create from existing events
    orig_lengths = events.lengths
    if fill in ['none', 'extend']:
        num_segments = np.floor(orig_lengths / length).astype(int)
    else:
        num_segments = np.ceil(orig_lengths / length).astype(int)
    
    # Compute new segment bounds
    reverse_index = np.repeat(np.arange(events.num_events), num_segments)
    begs, ends, index, groups = [], [], [], []
    for orig_beg, orig_end, orig_index, orig_group, num_segment in \
        zip(events.begs, events.ends, events.index_data, events.groups_data, num_segments):
        # Compute new default segment bounds
        new_begs = np.arange(0, max(num_segment, 1)) * length + orig_beg
        new_ends = np.append(new_begs[1:], new_begs[-1] + length)
        # Adjust bounds based on fill method
        if fill == 'balance':
            if ((orig_end - new_begs[-1]) < (length / 2)) and (num_segment > 1):
                new_begs = new_begs[:-1]
                new_ends = new_ends[:-1]
                num_segment -= 1
                fill_i = 'extend'
            else:
                fill_i = 'cut'
        else:
            fill_i = fill
        if (fill_i in ['cut', 'left', 'extend']) or \
            (num_segment == 0 and not fill_i == 'right'):
            new_ends[-1] = orig_end
        if fill_i == 'left':
            new_begs[-1] = orig_end - length
        # Append to lists
        repeats = max(num_segment, 1)
        begs.extend(new_begs)
        ends.extend(new_ends)
        index.extend(np.repeat(orig_index, repeats))
        groups.extend(np.repeat(orig_group, repeats))
    # Create new events object
    new_events = events.from_similar(
        index=index,
        groups=groups,
        locs=None,
        begs=begs,
        ends=ends,
        allow_duplicate_index=True,
    )

    # Return results
    outputs = [new_events]
    if return_relation:
        relation = relate.EventsRelation(new_events, events, cache=True)
        relation._intersect_data = _indices_to_sparse_many_to_one(
            index, events.index_data
        )
        relation._intersect_kwargs = {'enforce_edges': False}
        outputs.append(relation)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]

def separate(events, anchor='centers', method='balanced', drop_short=False, inplace=False):
    """
    Address overlapping ranges by distributing overlaps between adjacent
    events. Eclipsed ranges (fully contained within another range) are
    eliminated. Identical ranges are deduplicated (first kept). Distributions
    are made based on a specified event anchor point and split method.

    Parameters
    ----------
    events : EventsData
        Input range of events.
    anchor : str {'centers', 'begs', 'ends'}, default 'centers'
        The anchor point of each event to be used when sorting events and 
        distributing overlaps. Events are sorted by this anchor before 
        processing, which determines overlap resolution priority.
    method : str {'balanced', 'center', 'left', 'right'}, default 'balanced'
        The strategy for splitting overlapping regions between adjacent events.

        - ``balanced`` : Use the midpoint of the overlapping termini (ends[i],
          begs[j]), clamped between the two event centers. Where the overlap is
          large enough that the center midpoint falls within the overlap, use
          the center midpoint instead. This provides conservative splits for
          small overlaps and fair splits for large symmetric overlaps.
        - ``center`` : Always split at the midpoint of the two event centers.
          Provides a fair, geometry-independent split regardless of overlap
          magnitude.
        - ``left`` : Assign the full overlap to the left (earlier) event. The
          right event's beg is set to the left event's end.
        - ``right`` : Assign the full overlap to the right (later) event. The
          left event's end is set to the right event's beg.

    drop_short : bool, default False
        Whether to drop events that have been reduced to zero length 
        (eclipsed or fully squeezed out by adjacent events).
    inplace : bool, optional
        If True, modify the input object in place. Default is False.

    Returns
    -------
    EventsData or None
        The resulting events with all overlapping ranges separated. Returns
        None if inplace is True.
    """
    # Validate input
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    if not events.is_linear:
        raise ValueError("Input object must be a linear EventsData instance.")
    if events.is_empty:
        return None if inplace else events.copy()
    if anchor not in ['centers', 'begs', 'ends']:
        raise ValueError("'anchor' must be one of 'centers', 'begs', or 'ends'.")
    if method not in ['balanced', 'center', 'left', 'right']:
        raise ValueError(
            "'method' must be one of 'balanced', 'center', 'left', or 'right'.")

    # Select object to modify
    modified = events if inplace else events.copy()
    n = modified.num_events
    if n == 1:
        return None if inplace else modified

    # Sort by group, anchor ascending, length descending
    sort_by = [anchor, 'lengths']
    sort_ascending = [True, False]
    if modified.is_grouped:
        sort_by.insert(0, 'groups')
        sort_ascending.insert(0, True)
    sorted_events, sort_idx = modified.sort(
        by=sort_by, ascending=sort_ascending, return_index=True)
    inv_sort_idx = np.argsort(sort_idx)

    # Extract sorted data
    begs = sorted_events.begs.copy()
    ends = sorted_events.ends.copy()
    centers = (begs + ends) / 2

    # --- Eliminate identical ranges (keep first occurrence) ---
    same_mask = sorted_events.find_same(keep='first')

    # --- Eliminate eclipsed ranges (fully contained within another) ---
    eclipsed = sorted_events.find_inside(enforce_edges=True)

    # Combine masks: zero out both same and eclipsed events
    eliminate = same_mask | eclipsed
    begs[eliminate] = centers[eliminate]
    ends[eliminate] = centers[eliminate]
    valid_idx = np.where(~eliminate)[0]

    # --- Separate overlapping valid events ---
    if len(valid_idx) > 1:
        # Same-group check for adjacent valid pairs
        if modified.is_grouped:
            pair_same_group = (
                sorted_events.groups[valid_idx[:-1]] == 
                sorted_events.groups[valid_idx[1:]]
            )
        else:
            pair_same_group = np.ones(len(valid_idx) - 1, dtype=bool)

        # Adjacent valid pair bounds and centers
        rights = ends[valid_idx[:-1]]
        lefts = begs[valid_idx[1:]]
        centers_l = centers[valid_idx[:-1]]
        centers_r = centers[valid_idx[1:]]

        # Determine split points based on method
        overlapping = pair_same_group & (rights > lefts)
        if method == 'center':
            # Fair split: always use midpoint between event centers
            mids = (centers_l + centers_r) / 2
            split_mask = overlapping
        elif method == 'left':
            # Assign overlap to left event: split at right's original beg
            # (i.e., left keeps its end, right's beg stays unchanged)
            mids = rights
            split_mask = overlapping
        elif method == 'right':
            # Assign overlap to right event: split at left's original end
            # (i.e., right keeps its beg, left's end stays unchanged)
            mids = lefts
            split_mask = overlapping
        else:
            # Balanced (default): two-level approach.
            # Level 1 (termini): midpoint of the overlapping bounds, clamped
            # between centers. Handles small overlaps conservatively by
            # splitting near the actual overlap region.
            # Level 2 (center): midpoint of centers. Overrides termini when
            # the overlap is large enough for this fairer split to apply.
            termini_mids = np.clip((rights + lefts) / 2, centers_l, centers_r)
            center_mids = (centers_l + centers_r) / 2

            termini_valid = (
                overlapping & 
                (rights >= termini_mids) & (lefts <= termini_mids)
            )
            center_valid = (
                overlapping & 
                (rights >= center_mids) & (lefts <= center_mids)
            )

            # Apply termini first, then center overrides where applicable
            mids = np.where(termini_valid, termini_mids, rights)
            mids = np.where(center_valid, center_mids, mids)
            split_mask = termini_valid | center_valid

        # Apply split points to ends of left events and begs of right events
        ends[valid_idx[:-1]] = np.where(split_mask, mids, rights)
        begs[valid_idx[1:]] = np.where(split_mask, mids, lefts)

    # Unsort and apply
    modified._begs = begs[inv_sort_idx]
    modified._ends = ends[inv_sort_idx]

    # Drop zero-length events if requested
    if drop_short:
        mask = modified.lengths > 0
        if not mask.all():
            return selection.select_mask(modified, mask, inplace=inplace)

    return None if inplace else modified

def _indices_to_sparse_many_to_one(idx_left, idx_right):
    """
    Create a sparse matrix which represents the many to one relationship between two 
    sets of indices. The matrix will have shape (len(idx_left), len(idx_right)), where
    each entry (i, j) is 1 if idx_left[i] == idx_right[j] and is empty otherwise. This
    assumes that each entry in idx_left maps to exactly one entry in idx_right, but
    that multiple entries in idx_left may map to the same entry in idx_right.

    Parameters
    ----------
    idx_left : array-like
        1D array of indices for the left set.
    idx_right : array-like
        1D array of indices for the right set.
    """
    # Analyze unique values
    left_unique, left_inverse = np.unique(idx_left, return_inverse=True)
    right_unique, right_indices = np.unique(idx_right, return_index=True)
    # Validate that all left unique values are in right unique values
    if not np.array_equal(left_unique, right_unique):
        raise ValueError(
            "All values must be consistent between idx_left and idx_right."
        )
    # Create data for intersections
    data = np.ones(len(idx_left))
    # Create row and column parameters for COO instantiation
    row = np.arange(len(idx_left))
    col = right_indices[left_inverse]
    # Create sparse matrix
    arr = sp.csr_array(
        (data, (row, col)),
        shape=(len(idx_left), len(idx_right))
    )
    return arr