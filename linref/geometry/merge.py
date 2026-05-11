from __future__ import annotations
import numpy as np
from collections import deque
from shapely.errors import GeometryTypeError
from linref.errors import GeometryTopologyError
from linref.geometry.linestring_m import LineStringM


def line_merge_m(
    lines: list[LineStringM],
    allow_multiple: bool = False,
    allow_mismatch: bool = False,
    squeeze: bool = False,
    return_orders: bool = False,
    return_chains: bool = False,
    cast_geom: bool = False
    ):
    """
    Merge a list of LineStringM geometries into a single LineStringM geometry 
    or a list of LineStringM geometries if they are not contiguous.

    Parameters
    ----------
    lines : list[LineStringM | shapely.LineString]
        An array-like of LineStringM geometries to be merged. If cast_geom is 
        True, can also include shapely LineString geometries which will be 
        cast to LineStringM.
    allow_multiple : bool, default False
        If True, allows the function to return multiple merged geometries if 
        the input lines are not all contiguous. If False, raises an error if 
        multiple merged geometries would be returned.
    allow_mismatch : bool, default False
        If True, allows M values to be mismatched at the termini of contiguous 
        lines. This will retain the first M value of each merged line segment. 
        If False, will not merge lines if M values are mismatched at the 
        termini.
    squeeze : bool, default False
        If True and only one merged geometry is produced, returns the geometry 
        directly instead of a list containing the single geometry.
    return_orders : bool, default False
        If True, also returns a list of indices indicating the order of the 
        input lines in the merged geometries.
    return_chains : bool, default False
        If True, also returns a list indicating the chain index of each input 
        line in the merged geometries.
    cast_geom : bool, default False
        If True, attempts to cast any non-LineStringM geometries in the input 
        list to LineStringM. If False, raises an error if any non-LineStringM
        geometries are found.
    """
    # Validate input geometries
    if not isinstance(lines, (list, np.ndarray)):
        try:
            lines = list(lines)
        except:
            raise TypeError(
                "Input must be an array-like of LineStringM geometries. "
                f"Provided type: {type(lines)}"
            )
    lines_prepared = []
    for line in lines:
        if not isinstance(line, LineStringM):
            if cast_geom:
                try:
                    line = LineStringM(line)
                    lines_prepared.append(line)
                except Exception:
                    raise GeometryTypeError(
                        "All geometries must be LineStringM or castable to "
                        "LineStringM."
                    )
            else:
                raise GeometryTypeError("All geometries must be LineStringM.")
        else:
            lines_prepared.append(line)
    # Initialize mapping of geometries
    merged_geoms = []
    orders = deque()
    chains = [0] * len(lines_prepared)
    indices = list(range(len(lines_prepared)))
    # Iterate through indices of potential merged lines
    for merged_index in range(len(lines_prepared)):
        # Initialize list of indices for the current merged line
        orders_current = deque()
        coords_current = deque()
        # Initialize coordinates for the indexed merged line
        beg_coords = None
        end_coords = None
        # Initialize a repeated cycle to find contiguous lines
        while True:
            # Initialize a counter for successes within the cycle
            success_count = 0
            # Iterate through all unassigned lines to find contiguous lines
            for line_index in indices:
                # Get coordinates of the current line
                coords = lines_prepared[line_index].coords
                if beg_coords is None:
                    # If starting a new merged geometry, set beg and end 
                    # coordinates to the current line's termini
                    beg_coords, end_coords = coords[0], coords[-1]
                    # Record the order and chain
                    orders_current.append(line_index)
                    coords_current.append(coords)
                else:
                    # Check if the current line is contiguous with the merged 
                    # geometry
                    if np.array_equal(
                        coords[0, :2] if allow_mismatch else coords[0],
                        end_coords[:2] if allow_mismatch else end_coords
                    ):
                        # If contiguous at the end, extend the merged geometry
                        end_coords = coords[-1]
                        # Record the order and chain
                        orders_current.append(line_index)
                        coords_current.append(coords)
                    elif np.array_equal(
                        coords[-1, :2] if allow_mismatch else coords[-1],
                        beg_coords[:2] if allow_mismatch else beg_coords
                    ):
                        # If contiguous at the beginning, extend the merged 
                        # geometry
                        beg_coords = coords[0]
                        # Record the order and chain
                        orders_current.appendleft(line_index)
                        coords_current.appendleft(coords)
                    else:
                        # If not contiguous, skip to the next line
                        continue
                # If successfully merged, record the chain index and remove 
                # from indices
                chains[line_index] = merged_index
                indices.remove(line_index)
                success_count += 1
            # Check if another cycle is needed
            if success_count == 0 or len(indices) == 0:
                break
        # Construct the merged geometry from the collected coordinates
        if len(coords_current) == 1:
            merged_geom = LineStringM.from_coords(coords_current[0])
        else:
            # Prepare coordinates for merging by removing duplicate termini
            coords_current = list(coords_current)
            coords_prepared = \
                [c[:-1, :] for c in coords_current[:-1]] + \
                [coords_current[-1]]
            merged_geom = LineStringM.from_coords(np.vstack(coords_prepared))
        # Append the orders of the current merged geometry
        orders.extend(orders_current)
        merged_geoms.append(merged_geom)
        # Break if all lines have been assigned
        if len(indices) == 0:
            break

    # Check if multiple merged geometries are allowed
    if not allow_multiple and len(merged_geoms) > 1:
        raise GeometryTopologyError(
            "Multiple merged geometries were produced. "
            "Set allow_multiple=True to permit this."
        )
    # Squeeze output if only one merged geometry and requested
    if squeeze and len(merged_geoms) == 1:
        merged_geoms = merged_geoms[0]
    # Return merged geometries and order and chain mappings as lists as 
    # requested
    returns = [merged_geoms]
    if return_orders:
        returns.append(list(orders))
    if return_chains:
        returns.append(list(chains))
    if len(returns) == 1:
        return returns[0]
    else:
        return tuple(returns)    


def get_linestring_chains(objs):
    """
    Return the chain indices for each LineStringM object in the list.

    Parameters
    ----------
    objs : list of LineStringM
        The LineStringM objects to get the chain indices for.
    """
    chains = line_merge_m(
        objs,
        allow_multiple=True,
        allow_mismatch=False,
        return_orders=False,
        return_chains=True,
        squeeze=False,
        cast_geom=True
    )[1]
    return chains
