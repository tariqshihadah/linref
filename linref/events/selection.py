import numpy as np
from linref.events import base, utility

def _validate_any_selector(rng, selector, ignore=False):
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
            selector = _validate_boolean_selector(rng, selector)
        else:
            selector = _validate_index_selector(rng, selector, ignore)
    else:
        raise ValueError(
            "Input selector must be a slice object, boolean array, or an "
            "array of event indices.")
    return selector

def _validate_slice_selector(rng, selector):
    """
    Function for validating input selector as a slice object.
    """
    # Validate input
    if not isinstance(selector, slice):
        raise ValueError(
            "Input selector must be a slice object.")
    return selector

def _validate_boolean_selector(rng, selector):
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
    if not len(selector) == rng.num_events:
        raise ValueError(
            "Input selector must be the same length as the number of events. "
            f"Expected {rng.num_events}, received {len(selector)}.")
    if not selector.dtype == bool:
        raise ValueError(
            "Input selector must be a boolean array.")
    return selector

def _validate_index_selector(rng, selector, ignore=False):
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
        selector_test = np.in1d(selector, rng._index)
        if not np.all(selector_test):
            missing_values = selector[~selector_test]
            raise ValueError(
                f"Index values not found: {missing_values}")
        # Sort the event index values
        sorter = np.argsort(rng._index)
        # Apply the selector to the sorted index values
        selector = np.searchsorted(rng._index[sorter], selector)
    else:
        # Ensure that all values are within the range of the number of events
        selector_test = (selector >= 0) & (selector < rng.num_events)
        if not np.all(selector_test):
            missing_values = selector[~selector_test]
            raise ValueError(
                f"Index values out of range: {missing_values}")
    return selector

def _validate_group_selector(rng, group):
    # Validate input group
    if not rng.is_grouped:
        raise ValueError("Collection is not grouped.")
    if isinstance(group, (list, np.ndarray)):
        # Multiple group selection
        select_multiple = True
        # Ensure that all groups are present
        group_test = np.in1d(group, rng.unique_groups)
        if not np.all(group_test):
            missing_groups = group[~group_test]
            raise KeyError(
                f"Groups not found in collection: {missing_groups}")
    else:
        # Single group selection
        select_multiple = False
        if not group in rng.unique_groups:
            raise KeyError(
                f"Group not found in collection: {group}")
    
    # Identify group indices
    if select_multiple:
        index = np.isin(rng.groups, group)
    else:
        index = np.equal(rng.groups, group)

    # Return results
    return index

def _apply_selector(rng, selector, inplace=False):
    """
    Apply a selector to the input events.
    """
    # Apply selection
    rc = rng if inplace else rng.copy()
    try:
        rc._index = rng._index[selector]
        rc._groups = rng._groups[selector] if rng._groups is not None else None
        rc._locs = rng._locs[selector] if rng._locs is not None else None
        rc._begs = rng._begs[selector] if rng._begs is not None else None
        rc._ends = rng._ends[selector] if rng._ends is not None else None
    except:
        raise ValueError(
            f"Invalid selection: {selector}")
    return None if inplace else rc

def select(rng, selector, ignore=False, inplace=False):
    """
    Select events by index, slice, or boolean mask. Use ignore=True to use 
    a generic, 0-based index, ignoring the current index values.

    Parameters
    ----------
    rng : Rangel
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
    selector = _validate_any_selector(rng, selector, ignore=ignore)
    return _apply_selector(rng, selector, inplace=inplace)

def select_slice(rng, slice_, inplace=False):
    """
    Select events by slice.

    Parameters
    ----------
    rng : Rangel
        The events object to select from.
    slice_ : slice
        Slice object to select events.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    selector = _validate_slice_selector(rng, slice_)
    return _apply_selector(rng, selector, inplace=inplace)

def select_mask(rng, mask, inplace=False):
    """
    Select events by boolean mask.

    Parameters
    ----------
    rng : Rangel
        The events object to select from.
    mask : array-like
        Boolean mask aligned to the events.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    selector = _validate_boolean_selector(rng, mask)
    return _apply_selector(rng, selector, inplace=inplace)

def select_index(rng, index, ignore=False, inplace=False):
    """
    Select events by index values.

    Parameters
    ----------
    rng : Rangel
        The events object to select from.
    index : array-like
        Array-like of event indices to select.
    ignore : bool, default False
        Whether to use a generic 0-based index, ignoring the current index 
        values.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    selector = _validate_index_selector(rng, index, ignore)
    return _apply_selector(rng, selector, inplace=inplace)

def select_group(rng, group, ungroup=None, inplace=False):
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
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    # Validate group input
    mask = _validate_group_selector(rng, group)

    # Determine ungrouping
    if ungroup is None:
        ungroup = not isinstance(group, (list, np.ndarray))
    else:
        ungroup = bool(ungroup)

    # Apply selection
    rng = rng if inplace else rng.copy()
    select_mask(rng, mask, inplace=True)
    if ungroup:
        rng.ungroup(inplace=True)

    return None if inplace else rng

def drop(rng, selector, inplace=False):
    """
    Drop events by boolean mask.

    Parameters
    ----------
    rng : Rangel
        The events object to select from.
    mask : array-like
        Boolean mask aligned to the events.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    # Validate boolean mask input and invert
    selector = _validate_boolean_selector(rng, selector)
    np.logical_not(selector, out=selector)
    return _apply_selector(rng, selector, inplace=inplace)

def drop_group(rng, group, inplace=False):
    """
    Drop events by group.
    
    Parameters
    ----------
    rng : Rangel
        The events object to select from.
    group : label or array-like
        The label of the group to drop or array-like of the same.
    inplace : bool, default False
        Whether to perform the operation in place, returning None.
    """
    # Validate group input
    mask = _validate_group_selector(rng, group)

    # Apply selection
    rng = rng if inplace else rng.copy()
    drop(rng, mask, inplace=True)

    return None if inplace else rng