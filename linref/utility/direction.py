from __future__ import annotations
import math
import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString

def extract_direction(line: LineString | MultiLineString, labels: list[str] = ['E', 'N', 'W', 'S']) -> str:
    """
    Approximate the cardinal direction of the line, based on the first and 
    last points in the geometry.
    
    Parameters
    ----------
    line : shapely.geometry.LineString or .MultiLineString
        Linear geometry being analyzed.
    """
    # Get bearing angle
    bearing = extract_bearing(line, positive=True, invert=False)

    # Convert to direction
    range = 360 / len(labels)
    bins = np.arange(0, 360, range) + range / 2
    select = np.digitize(bearing, bins)
    return (labels + [labels[0]])[select]

def extract_bearing(line: LineString | MultiLineString, positive: bool = True, invert: bool = False, first: int = 0, last: int = -1) -> float:
    """
    Approximate the bearing angle of the line, based on the first and 
    last points in the geometry.
    
    Parameters
    ----------
    line : shapely.geometry.LineString or .MultiLineString
        Linear geometry being analyzed.
    positive : bool, default True
        Whether to enforce a positive range on the computed bearing angle. 
        If True, the bearing angle will fall on the range [0,360). If 
        False, the bearing angle will fall on the range (-180,180].
    invert : bool, default False
        Whether to invert the computed bearing angle, effectively 
        reversing the direction of the line.
    first : int, default 0
        Index of the point to use as the beginning of the line.
    last : int, default -1
        Index of the point to use as the end of the line.
    """
    # Get first/last coordinates
    try:
        pt_beg = line.coords[first]
        pt_end = line.coords[last]
    except:
        try:
            pt_beg = line.geoms[0].coords[first]
            pt_end = line.geoms[-1].coords[last]
        except:
            raise ValueError(
                "Unable to extract begin/end points from the input geometry."
            )
    # Get X/Y distances
    x_diff = pt_end[0] - pt_beg[0]
    y_diff = pt_end[1] - pt_beg[1]
    
    # Compute bearing angle
    bearing = math.degrees(math.atan2(y_diff, x_diff))
    
    # Invert if requested
    if invert:
        bearing += 180
    
    # Enforce range
    if positive and bearing < 0:
        bearing += 360
    elif not positive and bearing > 180:
        bearing -= 360
        
    return bearing

