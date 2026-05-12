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

    # Array path: extract raw geometries (None stays None) and let shapely
    # handle null entries natively (returns NaN for None geometries)
    line_geoms = np.array(
        [obj.geom if isinstance(obj, LineStringM) else None for obj in line],
        dtype=object,
    )
    distances = shapely.line_locate_point(line_geoms, point, normalized=normalized)

    if m:
        distances = _distance_to_m_impl(line, distances, normalized=normalized)

    return distances


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

        # Skip non-LineStringM entries or lines without M values
        if not isinstance(line_m, LineStringM) or line_m.m is None:
            continue

        m_arr = line_m.m
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


def line_interpolate_point_m(
    line: LineStringM | np.ndarray,
    distance,
    normalized: bool = False,
    m: bool = False,
):
    """
    Return a point at the specified distance along the linear geometry.
    Supports both scalar and vectorized (array) inputs.

    Analogous to ``shapely.line_interpolate_point``, extended with M-value
    support.

    Parameters
    ----------
    line : LineStringM or array of LineStringM
        The M-enabled line geometry (or geometries) to interpolate along.
    distance : float or array of float
        The distance(s) along each geometry at which to interpolate.
    normalized : bool, default False
        Whether the distance is normalized (0-1) or absolute.
    m : bool, default False
        Whether the distance should be interpreted as an M value.

    Returns
    -------
    shapely.Point or np.ndarray of shapely.Point
        The interpolated point(s). None where interpolation is not possible.
    """
    # Scalar path: single LineStringM + single distance
    if isinstance(line, LineStringM):
        if m:
            distance = m_to_distance(line, distance)
        return shapely.line_interpolate_point(line.geom, distance, normalized=normalized)

    # Array path: convert M values to distances if needed, then extract raw
    # geometries (None stays None) and let shapely handle null natively
    distances = np.asarray(distance, dtype=np.float64)
    if m:
        distances = _m_to_distance_impl(line, distances)

    line_geoms = np.array(
        [obj.geom if isinstance(obj, LineStringM) else None for obj in line],
        dtype=object,
    )
    return shapely.line_interpolate_point(line_geoms, distances, normalized=normalized)


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

        # Skip non-LineStringM entries or lines without M values
        if not isinstance(line_m, LineStringM) or line_m.m is None:
            continue

        m_arr = line_m.m
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
