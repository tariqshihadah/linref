"""
Module-level vectorized geometry operations for LineStringM objects.

Follows the same pattern as shapely's module-level functions
(e.g., shapely.line_locate_point) to provide both scalar and
array-based interfaces.
"""
from __future__ import annotations
import numpy as np
import shapely
from linref.geometry.linestring_m import LineStringM


def line_locate_point_m(
    line: LineStringM | np.ndarray,
    point,
    normalized: bool = False,
    m: bool = False,
) -> float | np.ndarray:
    """
    Return the distance along the linear geometry to the nearest point on the
    geometry from the specified point. Supports both scalar and vectorized
    (array) inputs.

    Analogous to ``shapely.line_locate_point``, extended with M-value support.

    Parameters
    ----------
    line : LineStringM or array of LineStringM
        The M-enabled line geometry (or geometries) to project onto.
    point : shapely.Point or array of shapely.Point
        The point (or points) to project.
    normalized : bool, default False
        Whether to return the distance as normalized (0-1) or absolute.
    m : bool, default False
        Whether to return the result as an M value instead of a distance.

    Returns
    -------
    float or np.ndarray
        The projected distance (or M value) for each input pair.
    """
    # Scalar path: single LineStringM + single point
    if isinstance(line, LineStringM):
        dist = shapely.line_locate_point(line.geom, point, normalized=normalized)
        if m:
            dist = distance_to_m(line, dist, normalized=normalized)
        return dist

    # Array path: arrays of LineStringM + points
    return _line_locate_point_m_array(line, point, normalized=normalized, m=m)


def _line_locate_point_m_array(
    line_m_objects: np.ndarray,
    points: np.ndarray,
    normalized: bool = False,
    m: bool = False,
) -> np.ndarray:
    """
    Vectorized projection of point geometries onto LineStringM objects.

    Parameters
    ----------
    line_m_objects : array of LineStringM
        The LineStringM objects to project onto.
    points : array of shapely.Point
        The point geometries to project.
    normalized : bool, default False
        Whether to return distances as normalized (0-1) or absolute.
    m : bool, default False
        Whether to convert distances to M values.

    Returns
    -------
    np.ndarray
        Array of distances (or M-values). NaN where projection is not
        possible (e.g., None geometry).
    """
    n = len(line_m_objects)
    result = np.full(n, np.nan, dtype=np.float64)

    # Build mask of valid (non-None) LineStringM entries
    valid_mask = np.array(
        [obj is not None and hasattr(obj, 'geom') for obj in line_m_objects],
        dtype=bool,
    )
    if not valid_mask.any():
        return result

    # Extract raw shapely LineStrings and points for valid rows
    valid_indices = np.where(valid_mask)[0]
    valid_line_m = line_m_objects[valid_indices]
    line_geoms = np.array([obj.geom for obj in valid_line_m], dtype=object)
    point_geoms = points[valid_indices]

    # Vectorized shapely.line_locate_point: compute distance along each line
    distances = shapely.line_locate_point(line_geoms, point_geoms, normalized=normalized)

    if not m:
        result[valid_indices] = distances
        return result

    # Vectorized distance-to-M conversion
    m_values = _distance_to_m_impl(valid_line_m, distances, normalized=normalized)
    result[valid_indices] = m_values
    return result


def distance_to_m(
    line_m: LineStringM | np.ndarray,
    distance: float | np.ndarray,
    normalized: bool = False,
) -> float | np.ndarray:
    """
    Convert distance(s) along LineStringM geometry(ies) to M value(s).
    Supports both scalar and vectorized (array) inputs.

    Parameters
    ----------
    line_m : LineStringM or array of LineStringM
        The M-enabled line geometry (or geometries).
    distance : float or array of float
        The distance(s) along each geometry.
    normalized : bool, default False
        Whether the distance(s) are normalized (0-1) or absolute.

    Returns
    -------
    float or np.ndarray
        The M value(s). NaN where conversion is not possible.
    """
    # Scalar path: wrap to 1-element arrays, compute, unwrap
    if isinstance(line_m, LineStringM):
        result = _distance_to_m_impl(
            np.array([line_m]),
            np.array([distance], dtype=np.float64),
            normalized=normalized,
        )
        return float(result[0])

    # Array path
    return _distance_to_m_impl(line_m, distance, normalized=normalized)


def _distance_to_m_impl(
    lines_m: np.ndarray,
    distances: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    """
    Core implementation: convert distances along LineStringM geometries to
    M values. Always operates on arrays.

    Groups inputs by unique LineStringM object so that the searchsorted +
    interpolation can be batched with numpy rather than looped per-element.
    """
    # Initialize output with NaN (returned for lines with no M values)
    m_values = np.full(len(lines_m), np.nan, dtype=np.float64)

    # Group by unique LineStringM object to batch the interpolation.
    # Multiple rows may share the same line geometry, so grouping lets us
    # run vectorized numpy operations per unique line instead of per row.
    obj_ids = np.array([id(obj) for obj in lines_m])
    unique_ids, inverse = np.unique(obj_ids, return_inverse=True)

    # Process each unique line group
    for group_idx in range(len(unique_ids)):
        # Find the representative LineStringM for this group
        group_mask = inverse == group_idx
        line_m = lines_m[np.argmax(group_mask)]
        m_arr = line_m.m

        # Skip lines without M values (result stays NaN)
        if m_arr is None:
            continue

        cumdist = line_m._cumulative_distances
        group_distances = distances[group_mask]

        # Convert normalized distances to absolute if needed
        if normalized:
            group_distances = group_distances * line_m.geom.length

        # Find which segment each distance falls in via binary search
        indices = np.searchsorted(cumdist, group_distances)
        # Clip to valid range [1, n-1] so indices-1 and indices are both valid
        indices = np.clip(indices, 1, len(cumdist) - 1)

        # Compute the proportion along each segment
        seg_start_dist = cumdist[indices - 1]
        seg_end_dist = cumdist[indices]
        seg_len = seg_end_dist - seg_start_dist

        # Safe division: zero-length segments produce prop=0
        prop = np.divide(group_distances - seg_start_dist, seg_len, out=np.zeros_like(seg_len), where=seg_len > 0)
        prop = np.clip(prop, 0.0, 1.0)

        # Linearly interpolate M values within each segment
        m_values[group_mask] = m_arr[indices - 1] + (m_arr[indices] - m_arr[indices - 1]) * prop

    return m_values


def m_to_distance(
    line_m: LineStringM | np.ndarray,
    m_value: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert M value(s) to distance(s) along LineStringM geometry(ies).
    Supports both scalar and vectorized (array) inputs.

    Parameters
    ----------
    line_m : LineStringM or array of LineStringM
        The M-enabled line geometry (or geometries).
    m_value : float or array of float
        The M value(s) to convert.

    Returns
    -------
    float or np.ndarray
        The distance(s). NaN where conversion is not possible.
    """
    # Scalar path: wrap to 1-element arrays, compute, unwrap
    if isinstance(line_m, LineStringM):
        result = _m_to_distance_impl(
            np.array([line_m]),
            np.array([m_value], dtype=np.float64),
        )
        return float(result[0])

    # Array path
    return _m_to_distance_impl(line_m, m_value)


def _m_to_distance_impl(
    lines_m: np.ndarray,
    m_values: np.ndarray,
) -> np.ndarray:
    """
    Core implementation: convert M values to distances along LineStringM
    geometries. Always operates on arrays.

    Groups inputs by unique LineStringM object so that the searchsorted +
    interpolation can be batched with numpy rather than looped per-element.
    """
    # Initialize output with NaN (returned for lines with no M values)
    distances = np.full(len(lines_m), np.nan, dtype=np.float64)

    # Group by unique LineStringM object to batch the interpolation.
    # Multiple rows may share the same line geometry, so grouping lets us
    # run vectorized numpy operations per unique line instead of per row.
    obj_ids = np.array([id(obj) for obj in lines_m])
    unique_ids, inverse = np.unique(obj_ids, return_inverse=True)

    # Process each unique line group
    for group_idx in range(len(unique_ids)):
        # Find the representative LineStringM for this group
        group_mask = inverse == group_idx
        line_m = lines_m[np.argmax(group_mask)]
        m_arr = line_m.m

        # Skip lines without M values (result stays NaN)
        if m_arr is None:
            continue

        cumdist = line_m._cumulative_distances
        group_m = m_values[group_mask]

        # Find which segment each M value falls in via binary search
        indices = np.searchsorted(m_arr, group_m)
        # Clip to valid range [1, n-1] so indices-1 and indices are both valid
        indices = np.clip(indices, 1, len(m_arr) - 1)

        # Compute the proportion along each segment
        m_start = m_arr[indices - 1]
        m_end = m_arr[indices]
        m_span = m_end - m_start

        # Safe division: zero-span segments produce prop=0
        prop = np.divide(group_m - m_start, m_span, out=np.zeros_like(m_span), where=m_span > 0)
        prop = np.clip(prop, 0.0, 1.0)

        # Linearly interpolate distances within each segment
        dist_start = cumdist[indices - 1]
        dist_end = cumdist[indices]
        distances[group_mask] = dist_start + (dist_end - dist_start) * prop

    return distances
