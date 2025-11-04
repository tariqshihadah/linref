from __future__ import annotations
import numpy as np
from linref.events import base, utility, relate, common

def integrate(objs, fill_gaps=False, split_at_locs=False, return_index=False) -> base.EventsData:
    """
    Combine multiple sets of events into a single collection, creating new 
    linear events based on least common intervals among all input events. If 
    requested, an array of index arrays for each range can be provided to 
    indicate which input events contributed to each output event.

    Parameters
    ----------
    objs : list
        List of EventsData instances to integrate.
    fill_gaps : bool, default False
        Whether to fill gaps in merged events between the maximum beginning
        and minimum ending points of all input events within a group. These
        gaps will be represented as events with no associated input events.
    split_at_locs : bool, default False
        Whether to split events at location points within the input events.
        This allows for breaking events at point events as well as linear 
        events.
    return_index : bool, default False
        Whether to return the inverse index of the integrated events as a list
        of arrays of indices to the original events.

    Returns
    -------
    linref.events.EventsData
    """
    # Validate input objects
    if not isinstance(objs, list):
        raise TypeError("Input must be a list of EventsData class instances.")
    if len(objs) == 0:
        raise ValueError("Must provide at least one EventsData instance.")
    for obj in objs:
        if not isinstance(obj, base.EventsData):
            raise TypeError("All input objects must be EventsData class instances.")
        
    # Iterate over all unique groups and integrate events within each group
    unique_groups = np.unique(np.concatenate([obj.groups for obj in objs]))
    new_groups = []
    new_begs = []
    new_ends = []
    for group in unique_groups:
        # Get all unique event edges within each group in each collection
        edges = []
        for obj in objs:
            data = obj.select_group(group)
            if data.is_linear:
                edges.append(data.begs)
                edges.append(data.ends)
            if data.is_located and split_at_locs:
                edges.append(data.locs)

        # Find unique edges and their corresponding groups
        unique_edges = np.unique(np.concatenate(edges))
        new_groups.append(np.full(len(unique_edges) - 1, group))
        new_begs.append(unique_edges[:-1])
        new_ends.append(unique_edges[1:])

    # Prepare new events data object
    integrated = objs[0].from_similar(
        index=None,
        groups=np.concatenate(new_groups),
        begs=np.concatenate(new_begs),
        ends=np.concatenate(new_ends),
    )

    # Relate new events to original events
    if return_index or not fill_gaps:
        indices = []
        masks = []
        for obj in objs:
            if not obj.is_linear:
                # If not linear, skip intersection
                index = np.full((integrated.num_events,), np.nan)
            else:
                # Get index of original events that intersect with new events
                arr = relate.intersect_linear_linear(
                    integrated,
                    obj,
                    enforce_edges=False,
                )
                mask = arr.sum(axis=1) > 0
                index = np.where(
                    mask,
                    obj.generic_index[arr.argmax(axis=1)],
                    -1
                )
            indices.append(index)
            masks.append(mask)
        # Combine indices from all input events
        indices = np.array(indices, dtype=int).T
        masks = np.any(masks, axis=0)
    
    # Remove gaps if needed
    if not fill_gaps:
        integrated = integrated[masks]
        indices = indices[masks]
    
    # Return results
    return (integrated, indices) if return_index else integrated
