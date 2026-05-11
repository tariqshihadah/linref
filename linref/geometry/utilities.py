from __future__ import annotations
import numpy as np
import shapely


def get_chord_lengths(ls, normalized=False):
    """
    Return an array of the chord lengths for each segment in the LineString.

    Parameters
    ----------
    ls : LineString
        The LineString to compute chord lengths for.
    normalized : bool, default False
        Whether to return the chord lengths as a proportion of the total 
        length of the LineString.
    """
    # Compute the chord lengths
    lengths = np.sqrt(
        np.sum(np.power(np.diff(np.array(ls.coords), axis=0), 2), axis=1))
    if normalized:
        lengths = lengths / lengths.sum()
    return lengths


def parse_linestring_wkt(wkt):
    """
    Parse a WKT representation of one or many LineStrings, returning one or 
    an array of LineString objects.

    Parameters
    ----------
    wkt : str or array-like of str
        The WKT representation of the LineString.
    """
    if isinstance(wkt, (list, np.ndarray)):
        return np.array([shapely.from_wkt(wkt_i) for wkt_i in wkt])
    else:
        return shapely.from_wkt(wkt)


def parse_linestring_m_wkt(wkt):
    """
    Parse a WKT representation of one or many LineStringMs, returning one or 
    an array of LineStringM objects.

    Parameters
    ----------
    wkt : str or array-like of str
        The WKT representation of the LineStringM.
    """
    from linref.geometry.linestring_m import LineStringM
    if isinstance(wkt, (list, np.ndarray)):
        return np.array([LineStringM.from_wkt(wkt_i) for wkt_i in wkt])
    else:
        return LineStringM.from_wkt(wkt)


def substring_m_coords(coords, m, start, end, normalized=False, tolerance=1e-10):
    """
    Extract a substring of set of coordinates given start and end fractions.
    Intended to provide similar functionality to shapely's substring, but working
    on raw coordinate lists.

    Parameters
    ----------
    coords : np.ndarray
        An NxM array of coordinates representing the linestring, where N indicates
        the number of points and M indicates the dimensionality (e.g., 2 for 2D, 
        3 for 3D).
    m : np.ndarray
        An array of M values corresponding to each coordinate in the coords array.
    start : float
        The starting distance along the linestring where the substring should begin.
    end : float
        The ending distance along the linestring where the substring should end.
    normalized : bool, optional
        If True, the start and end values are treated as fractions of the total
        length of the linestring (ranging from 0 to 1). Default is False.
    tolerance : float, optional
        Tolerance for detecting duplicate coordinates and handling floating point
        precision errors. Default is 1e-10.
    """
    
    def _interpolate_point(coords, m, cumdist, distance):
        """
        Helper function to interpolate coordinates and M values at a specific distance.
        Uses consistent interpolation logic to ensure reproducible results for the
        same distance value (critical for adjacent substrings sharing boundaries).
        
        Returns (index, coord, m_value) where index is the segment index before the point.
        """
        if distance <= 0:
            return 0, coords[0].copy(), m[0]
        elif distance >= cumdist[-1]:
            return len(cumdist) - 1, coords[-1].copy(), m[-1]
        else:
            # Find the segment containing this distance
            idx = np.argmax(cumdist >= distance)
            # Use consistent interpolation formula: lerp(a, b, t) = a + t * (b - a)
            # This is more numerically stable than: a * (1 - t) + b * t
            segment_start = cumdist[idx - 1]
            segment_end = cumdist[idx]
            t = (distance - segment_start) / (segment_end - segment_start)
            
            coord = coords[idx - 1] + t * (coords[idx] - coords[idx - 1])
            m_val = m[idx - 1] + t * (m[idx] - m[idx - 1])
            
            return idx, coord, m_val
    
    # Validate input parameters
    if start > end:
        raise ValueError("Start value must be less than or equal to end value.")
    
    # Calculate cumulative distances along the coordinate list
    # Use vectorized operations for better performance
    diff = np.diff(coords, axis=0)
    segment_lengths = np.sqrt(np.sum(diff * diff, axis=1))
    cumdist = np.empty(len(coords), dtype=np.float64)
    cumdist[0] = 0.0
    np.cumsum(segment_lengths, out=cumdist[1:])
    
    # Normalize cumulative distances if required
    if normalized:
        cumdist = cumdist / cumdist[-1]
    
    # Compute start and end coordinates using consistent interpolation
    start_index, start_coord, start_m = _interpolate_point(coords, m, cumdist, start)
    end_index, end_coord, end_m = _interpolate_point(coords, m, cumdist, end)
    
    # Construct the substring coordinate sequence directly as numpy arrays for better performance
    # Calculate the size needed
    n_intermediate = max(0, end_index - start_index)
    n_total = 2 + n_intermediate  # start + intermediate + end
    
    # Pre-allocate arrays
    substring_coords = np.empty((n_total, coords.shape[1]), dtype=coords.dtype)
    substring_m = np.empty(n_total, dtype=m.dtype)
    
    # Fill in the values
    substring_coords[0] = start_coord
    substring_m[0] = start_m
    
    if n_intermediate > 0:
        substring_coords[1:1+n_intermediate] = coords[start_index:end_index]
        substring_m[1:1+n_intermediate] = m[start_index:end_index]
    
    substring_coords[-1] = end_coord
    substring_m[-1] = end_m
    
    # Check for and remove duplicate coordinates at the ENDS only within floating point tolerance
    # The interpolated start/end points may coincide with existing vertices
    # Intermediate coordinates from the original line should be preserved exactly
    # Use squared distance to avoid expensive sqrt operation
    tolerance_sq = tolerance * tolerance
    
    # Check if the interpolated start point duplicates the first intermediate point
    if len(substring_coords) > 2:
        coord_diff = substring_coords[0] - substring_coords[1]
        coord_dist_sq = np.dot(coord_diff, coord_diff)
        m_diff = abs(substring_m[0] - substring_m[1])
        if coord_dist_sq <= tolerance_sq and m_diff <= tolerance:
            # Remove the interpolated start point, keep the original vertex
            substring_coords = substring_coords[1:]
            substring_m = substring_m[1:]
    
    # Check if the interpolated end point duplicates the last intermediate point
    if len(substring_coords) > 2:
        coord_diff = substring_coords[-1] - substring_coords[-2]
        coord_dist_sq = np.dot(coord_diff, coord_diff)
        m_diff = abs(substring_m[-1] - substring_m[-2])
        if coord_dist_sq <= tolerance_sq and m_diff <= tolerance:
            # Remove the interpolated end point, keep the original vertex
            substring_coords = substring_coords[:-1]
            substring_m = substring_m[:-1]
    
    # Special case: if start equals end (zero-length substring), ensure we have 2 points
    if len(substring_coords) < 2:
        # Duplicate the single point to create a valid LineString
        substring_coords = np.array([substring_coords[0], substring_coords[0]])
        substring_m = np.array([substring_m[0], substring_m[0]])
    
    return substring_coords, substring_m
