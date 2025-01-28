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