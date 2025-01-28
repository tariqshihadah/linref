import numpy as np
from linref.events import base, utility

def _validate_any_selector(events, selector, ignore=False):
    """
    Function for validating input selector as a slice, boolean array, or array
    of indices aligned with the input range's actual or generic index values.
    """
    # Identify the selector type
    if isinstance(selector, slice):
        pass
    elif isinstance(selector, (list, tuple, np.ndarray)):
        selector = np.asarray(selector)
        if selector.dtype == bool:
            selector = _validate_boolean_selector(events, selector)
        else:
            selector = _validate_index_selector(events, selector, ignore)
    else:
        raise ValueError(
            "Input selector must be a slice object, boolean array, or an "
            "array of event indices.")
    return selector

def _validate_slice_selector(events, selector):
    """
    Function for validating input selector as a slice object.
    """
    # Validate input
    if not isinstance(selector, slice):
        raise ValueError(
            "Input selector must be a slice object.")
    return selector

def _validate_boolean_selector(events, selector):
    """
    Function for validating input selector as a boolean array.
    """
    # Validate input
    try:
        selector = np.asarray(selector)
    except:
        raise ValueError(
            "Input selector must be an array-like object.")
    if not selector.ndim == 1:
        raise ValueError(
            "Input selector must be a 1D array-like object.")
    if not len(selector) == events.num_events:
        raise ValueError(
            "Input selector must be the same length as the number of events. "
            f"Expected {events.num_events}, received {len(selector)}.")
    if not selector.dtype == bool:
        raise ValueError(
            "Input selector must be a boolean array.")
    return selector

def _validate_index_selector(events, selector, ignore=False):
    """
    Function for validating input selector as an array of indices.
    """
    # Validate input
    try:
        selector = np.asarray(selector)
    except:
        raise ValueError(
            "Input selector must be an array-like object.")
    if not selector.ndim == 1:
        raise ValueError(
            "Input selector must be a 1D array-like object.")
    if ignore and not np.issubdtype(selector.dtype, np.integer):
        raise ValueError(
            "When ignoring the set index, input selector must be an array of "
            "integers.")
    
    # Apply to index values
    if not ignore:
        # Ensure that all values are present in the index
        selector_test = np.in1d(selector, events._index)
        if not np.all(selector_test):
            missing_values = selector[~selector_test]
            raise ValueError(
                f"Index values not found: {missing_values}")
        # Sort the event index values
        sorter = np.argsort(events._index)
        # Apply the selector to the sorted index values
        selector = np.searchsorted(events._index[sorter], selector)
    else:
        # Ensure that all values are within the range of the number of events
        selector_test = (selector >= 0) & (selector < events.num_events)
        if not np.all(selector_test):
            missing_values = selector[~selector_test]
            raise ValueError(
                f"Index values out of range: {missing_values}")
    return selector

def _validate_group_selector(events, group, ignore_missing=True):
    # Validate input group
    if not events.is_grouped:
        raise ValueError("Collection is not grouped.")
    
    # Convert input to array
    try:
        arr = np.asarray(group, dtype=events.groups.dtype)
    except:
        raise ValueError(
            "Unable to convert input group to array-like object.")
    # Check dimensions to determine if multiple groups are selected
    if arr.ndim > 1:
        raise ValueError(
            "Input group selection must be a scalar or 1D array-like object.")
    
    # Check for missing groups
    if not ignore_missing:
        # Ensure that all groups are present
        group_test = np.isin(arr, events.unique_groups)
        if not np.all(group_test):
            missing_groups = arr[~group_test]
            raise KeyError(
                f"Groups not found in collection: {missing_groups}")
        
    # Identify group indices
    index = np.isin(events.groups, arr)
    return index

def _apply_selector(events, selector, inplace=False):
    """
    Apply a selector to the input events.
    """
    # Apply selection
    selected = events if inplace else events.copy()
    try:
        selected._index = events._index[selector]
        selected._groups = events._groups[selector] if events._groups is not None else None
        selected._locs = events._locs[selector] if events._locs is not None else None
        selected._begs = events._begs[selector] if events._begs is not None else None
        selected._ends = events._ends[selector] if events._ends is not None else None
    except:
        raise ValueError(
            f"Invalid selection: {selector}")
    return None if inplace else selected

def select(events, selector, ignore=False, inplace=False):
    """
    Select events by index, slice, or boolean mask. Use ignore=True to use 
    a generic, 0-based index, ignoring the current index values.

    Parameters
    ----------
    events : EventsData
        The events object to select from.
    selector : array-like or slice
        Array-like of event indices, a boolean mask aligned to the events, 
        or a slice object to select events.
    ignore : bool, default False
        Whether to use a generic 0-based index, ignoring the current index 
        values.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    selector = _validate_any_selector(events, selector, ignore=ignore)
    return _apply_selector(events, selector, inplace=inplace)

def select_slice(events, slice_, inplace=False):
    """
    Select events by slice.

    Parameters
    ----------
    events : EventsData
        The events object to select from.
    slice_ : slice
        Slice object to select events.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    selector = _validate_slice_selector(events, slice_)
    return _apply_selector(events, selector, inplace=inplace)

def select_mask(events, mask, inplace=False):
    """
    Select events by boolean mask.

    Parameters
    ----------
    events : EventsData
        The events object to select from.
    mask : array-like
        Boolean mask aligned to the events.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    selector = _validate_boolean_selector(events, mask)
    return _apply_selector(events, selector, inplace=inplace)

def select_index(events, index, ignore=False, inplace=False):
    """
    Select events by index values.

    Parameters
    ----------
    events : EventsData
        The events object to select from.
    index : array-like
        Array-like of event indices to select.
    ignore : bool, default False
        Whether to use a generic 0-based index, ignoring the current index 
        values.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    selector = _validate_index_selector(events, index, ignore)
    return _apply_selector(events, selector, inplace=inplace)

def select_group(events, group, ungroup=None, ignore_missing=True, inplace=False):
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
    # Validate group input
    mask = _validate_group_selector(events, group, ignore_missing=ignore_missing)

    # Determine ungrouping
    if ungroup is None:
        ungroup = not isinstance(group, (list, np.ndarray))
    else:
        ungroup = bool(ungroup)

    # Apply selection
    events = events if inplace else events.copy()
    select_mask(events, mask, inplace=True)
    if ungroup:
        events.ungroup(inplace=True)

    return None if inplace else events

def drop(events, selector, inplace=False):
    """
    Drop events by boolean mask.

    Parameters
    ----------
    events : EventsData
        The events object to select from.
    mask : array-like
        Boolean mask aligned to the events.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    # Validate boolean mask input and invert
    selector = _validate_boolean_selector(events, selector)
    np.logical_not(selector, out=selector)
    return _apply_selector(events, selector, inplace=inplace)

def drop_group(events, group, inplace=False):
    """
    Drop events by group.
    
    Parameters
    ----------
    events : EventsData
        The events object to select from.
    group : label or array-like
        The label of the group to drop or array-like of the same.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    # Validate group input
    mask = _validate_group_selector(events, group)

    # Apply selection
    events = events if inplace else events.copy()
    drop(events, mask, inplace=True)

    return None if inplace else events