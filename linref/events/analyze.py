import numpy as np
from linref.events import common, base, utility

def duplicated(events, subset=None, keep='first'):
    """
    Return a boolean mask of duplicated events in terms of all or a selection of 
    event anchors.

    Parameters
    ----------
    events : EventsData
        The events object to analyze.
    subset : array-like, default None
        Array-like of event anchors to use for duplicated comparison. If None,
        all event anchors are used.
    keep : {'first', 'last', 'none'}, default 'first'
        Whether to keep the first, last, or none of the duplicated events.
    """
    # Validate input
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    if subset is None:
        subset = events.anchors
    else:
        if isinstance(subset, str):
            subset = [subset]
        elif not isinstance(subset, (list, np.ndarray)):
            raise TypeError("Input 'subset' must be an array-like.")
        if not all([anchor in events.anchors for anchor in subset]):
            raise ValueError(
                "Input 'subset' contains invalid anchor values. Must be one or "
                f"more of: {events.anchors}")
    
    # Create array to study for uniqueness based on selected anchors
    if events.is_grouped:
        subset.append('groups')
    study = np.array([getattr(events, anchor) for anchor in subset]).T
    if keep == 'last':
        study = study[::-1]

    # Find unique indices
    unique, uindex, ucounts = np.unique(
        study, axis=0, return_index=True, return_counts=True)
    
    # Determine which indices to keep
    if keep in ('first', 'last'):
        index = uindex
    elif keep == 'none':
        index = uindex[ucounts == 1]
    else:
        raise ValueError("Input 'keep' must be one of: 'first', 'last', 'none'")

    # Convert to boolean mask
    mask = np.ones(len(study), dtype=bool)
    mask[index] = False
    return mask if keep != 'last' else mask[::-1]

def find_same(events, keep='first'):
    """
    Return a boolean mask of events which have the same begin and end points
    as at least one other event in the collection. Group-aware: only compares
    events within the same group.

    Parameters
    ----------
    events : EventsData
        The events object to analyze.
    keep : {'first', 'last', 'none'}, default 'first'
        Which of the duplicate events to keep (mark as False in the mask).
        - 'first': keep the first occurrence, mark later ones as True.
        - 'last': keep the last occurrence, mark earlier ones as True.
        - 'none': mark all duplicates as True (none are kept).
    
    Returns
    -------
    np.ndarray
        Boolean mask where True indicates a "same" (duplicate) event.
    """
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    if not events.is_linear:
        raise ValueError("Input object must be a linear EventsData instance.")
    return duplicated(events, subset=['begs', 'ends'], keep=keep)

def find_inside(events, enforce_edges=False):
    """
    Return a boolean mask of events which fall entirely inside at least one 
    other event in the collection. Group-aware: only compares events within 
    the same group. Note that events that are the same (same beg/end) are not 
    considered inside each other, even with enforce_edges=True.

    Parameters
    ----------
    events : EventsData
        The events object to analyze.
    enforce_edges : bool, default False
        Whether to consider events touching at a vertex as being inside.
        - False (strict): beg > other_beg AND end < other_end
        - True (inclusive): (beg >= other_beg AND end < other_end) OR
          (beg > other_beg AND end <= other_end)

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates an event inside another.
    """
    if not isinstance(events, base.EventsData):
        raise TypeError("Input object must be a EventsData class instance.")
    if not events.is_linear:
        raise ValueError("Input object must be a linear EventsData instance.")
    if events.num_events <= 1:
        return np.zeros(events.num_events, dtype=bool)

    result = np.zeros(events.num_events, dtype=bool)

    if events.is_grouped:
        for group in events.unique_groups:
            group_mask = events.groups == group
            group_result = _find_inside_ungrouped(
                events.begs[group_mask], events.ends[group_mask], enforce_edges
            )
            result[group_mask] = group_result
    else:
        result = _find_inside_ungrouped(events.begs, events.ends, enforce_edges)

    return result


def _find_inside_ungrouped(begs, ends, enforce_edges=False):
    """
    Find inside ranges for an ungrouped set of events. Follows the original
    rangel algorithm: sort by begs ascending / ends descending, then use
    cumulative max of ends and a cummin lookup to identify containment.
    """
    n = len(begs)
    if n <= 1:
        return np.zeros(n, dtype=bool)

    # Sort by begs ascending, ends descending (longest first for same beg)
    sort_idx = np.lexsort([-(ends - begs), begs])
    inv = np.argsort(sort_idx)

    begs = begs[sort_idx]
    ends = ends[sort_idx]

    # Find dominating range extents
    cummax = np.maximum.accumulate(ends)
    _, uindex, uinv = np.unique(cummax, return_index=True, return_inverse=True)
    cummin = begs[uindex[uinv]]

    # Identify inside ranges
    if enforce_edges:
        inside = (
            ((begs >= cummin) & (ends < cummax)) |
            ((begs > cummin) & (ends <= cummax))
        )
    else:
        inside = (begs > cummin) & (ends < cummax)

    # Unsort back to original order
    return inside[inv]