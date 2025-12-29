from __future__ import annotations
import numpy as np
import copy
import warnings
import shapely
from collections import deque
from shapely.geometry import LineString, Point
from shapely.errors import GeometryTypeError
from linref.errors import *


class LineStringM:

    def __init__(self, geom, m=None):
        self.geom = geom
        # Check that M values haven't already been set from geometry
        if not self.has_m:
            self.m = m

    def __str__(self):
        return self.wkt
    
    def __repr__(self):
        return self.wkt + r' # linref compatibility approximation'
    
    def __eq__(self, other):
        if not isinstance(other, LineStringM):
            return False
        if not self.geom == other.geom:
            return False
        if self.m is None and other.m is None:
            return True
        if (self.m is None) != (other.m is None):
            return False
        return np.array_equal(self.m, other.m)

    def _reset_cache(self):
        """Reset all cached properties when geometry or M values change."""
        if hasattr(self, '_cached_cumdist'):
            delattr(self, '_cached_cumdist')

    @property
    def geom(self):
        return self._geom
    
    @geom.setter
    def geom(self, geom):
        if not isinstance(geom, shapely.geometry.LineString):
            raise ValueError('LineStringM geom must be a shapely LineString')
        
        # Invalidate cached properties when geometry changes
        self._reset_cache()
        
        try:
            if geom.has_z:
                raise ValueError('LineStringM geom must not have z values')
            if geom.has_m:
                # If the geometry has M values, extract them
                # FUTURE: Future versions of shapely may provide better support
                # for M values; update this section accordingly when that happens
                m = np.array([coord[2] for coord in geom.coords])
                self._geom = shapely.force_2d(geom)
                self.m = m
            else:
                self._geom = shapely.force_2d(geom)
        except AttributeError:
            self._geom = shapely.force_2d(geom)

    @property
    def m(self):
        return self._m
    
    @m.setter
    def m(self, m):
        if m is not None:
            # Enforce input type
            if not isinstance(m, np.ndarray):
                try:
                    m = np.array(m)
                except:
                    raise ValueError("LineStringM m must be a numpy array.")
            # Enforce input shape
            if len(m) != len(self.geom.coords):
                raise ValueError(
                    f"LineStringM m must be same length as coords. Received {len(m)} "
                    f"M values for coords of length {len(self.geom.coords)}."
                )
            # Enforce monotonic increase
            if not np.all(np.diff(m) >= 0):
                raise GeometryMeasureError(
                    "LineStringM m must be monotonic and increasing."
                )
        # Invalidate cached properties when M values change
        self._reset_cache()
        self._m = m

    @property
    def beg_m(self):
        """
        Return the beginning M value of the LineStringM object.
        """
        if self.m is None:
            return None
        return self.m[0]
    
    @property
    def end_m(self):
        """
        Return the ending M value of the LineStringM object.
        """
        if self.m is None:
            return None
        return self.m[-1]

    @property
    def coords(self):
        """
        Return the coordinates of the LineString as an array, interwoven
        with the M values if defined. Similar to shapely.get_coordinates, but 
        with M values appended.
        """
        # Get the geometry coordinates
        coords = shapely.get_coordinates(self.geom)
        # Append M values if they exist
        if self.has_m:
            coords = np.append(coords, self.m.reshape(-1, 1), axis=1)
        return coords
    
    @property
    def chord_lengths(self):
        """
        Return the chord lengths for each segment in the LineString.
        """
        return get_chord_lengths(self.geom, normalized=False)
    
    @property
    def chord_proportions(self):
        """
        Return the chord lengths for each segment in the LineString as a 
        proportion of the total length of the LineString.
        """
        return get_chord_lengths(self.geom, normalized=True)
    
    @property
    def _cumulative_distances(self):
        """
        Return the cumulative distances along the LineString at each vertex.
        Cached for performance optimization.
        """
        if not hasattr(self, '_cached_cumdist'):
            coords = shapely.get_coordinates(self.geom)
            diff = np.diff(coords, axis=0)
            segment_lengths = np.sqrt(np.sum(diff * diff, axis=1))
            cumdist = np.empty(len(coords), dtype=np.float64)
            cumdist[0] = 0.0
            np.cumsum(segment_lengths, out=cumdist[1:])
            self._cached_cumdist = cumdist
        return self._cached_cumdist
    
    @property
    def wkt(self):
        """
        Return the WKT representation of the object's LineString, interwoven 
        with the M values.
        """
        if self.m is None:
            return self.geom.wkt
        # Add M values to the WKT representation
        points = [f'{x[0]} {x[1]} {m}' for x, m in zip(self.geom.coords, self.m)]
        return 'LINESTRING M (' + ', '.join(points) + ')'
    
    @property
    def has_m(self):
        try:
            return self.m is not None
        except AttributeError:
            return False
    
    @classmethod
    def from_coords(cls, coords):
        """
        Create a LineStringM object from an array of coordinates with shape 
        (n, 2) or (n, 3), where n is the number of vertices. If shape (n, 3),
        the third column is interpreted as the M values.
        """
        # Validate input type
        if not isinstance(coords, np.ndarray):
            try:
                coords = np.array(coords)
            except:
                raise ValueError("Coordinates must be array-like.")
        # Validate input shape
        if coords.ndim != 2 or coords.shape[1] not in [2, 3]:
            raise ValueError("Coordinates must have shape (n, 2) or (n, 3).")
        # Extract geometry and M values
        if coords.shape[1] == 2:
            geom = LineString(coords)
            m = None
        else:
            geom = LineString(coords[:, :2])
            m = coords[:, 2]
        return cls(geom, m)
    
    @classmethod
    def from_shapely(cls, geom, m=None):
        """
        Create a LineStringM object from a shapely LineString object which
        may or may not have M values.

        Parameters
        ----------
        geom : shapely.LineString
            The shapely LineString object to convert.
        m : array-like, optional
            The M values associated with the LineString, if any. If the input
            shapely LineString already has M values, this parameter is ignored.
        """
        return cls(geom, m=m)
    
    @classmethod
    def from_wkt(cls, wkt):
        """
        Create a LineStringM object from a WKT representation.

        Parameters
        ----------
        wkt : str
            The WKT representation of the LineStringM.
        """
        # Validate input type
        if not isinstance(wkt, str):
            raise ValueError("WKT must be a string")
        # Remove comment if needed
        wkt = wkt.split(" # ")[0]
        # Parse the WKT string
        if not wkt.startswith("LINESTRING M"):
            raise ValueError("WKT must be of type LINESTRING M")
        coord_str = wkt[len("LINESTRING M ("):-1]
        coord_tuples = [tuple(map(float, point.split())) for point in coord_str.split(", ")]
        coords = [(x[0], x[1]) for x in coord_tuples]
        m = np.array([x[2] for x in coord_tuples])
        geom = LineString(coords)
        return cls(geom, m)
    
    def copy(self, deep=False):
        """
        Create an exact copy of the object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    
    def _check_snapping(self, distance, normalized=False, m=False, snap=False):
        """
        Check if snapping is required for the distance input and type.

        Parameters
        ----------
        distance : float
            The distance along the geometry to check.
        normalized : bool, default False
            Whether the distance is normalized (0-1) or absolute.
        m : bool, default False
            Whether the distance should be interpreted as an M value.
        snap : bool, default False
            Whether to snap the distance to the nearest vertex if it is out of 
            range. If False, a ValueError will be raised if the distance is out
            of range.

        Returns
        -------
        tuple
            A tuple containing the (possibly snapped) distance and a snap
            indicator (0 = no snap, 1 = snapped to beginning, 2 = snapped to 
            end).
        """
        # Ensure only one distance type is specified
        if normalized and m:
            raise ValueError('normalized and m cannot both be True')
        # Check normalized distance
        if normalized:
            if distance < 0 or distance > 1:
                if not snap:
                    raise ValueError(
                        f'Normalized distance {distance} is out of range; '
                        'must be between 0 and 1')
                return (0, 1) if distance < 0 else (1, 2)
            return (distance, 0)
        # Check M value
        elif m:
            if self.m is None:
                raise ValueError('M values are not defined')
            if distance < self.m[0] or distance > self.m[-1]:
                if not snap:
                    raise ValueError(
                        f'M value {distance} is out of range; '
                        f'must be between {self.m[0]} and {self.m[-1]}')
                return (self.m[0], 1) if distance < self.m[0] else (self.m[-1], 2)
            return (distance, 0)
        # Check absolute distance
        else:
            if distance < 0 or distance > self.geom.length:
                if not snap:
                    raise ValueError(
                        f'Distance {distance} is out of range; '
                        f'must be between 0 and {self.geom.length}')
                return (0, 1) if distance < 0 else (self.geom.length, 2)
            return (distance, 0)
        
    def set_m(self, array=None, beg=None, end=None, inplace=True):
        """
        Set the M values of the LineStringM object to the specified values
        using an array containing values for each vertex or using begin and 
        end values, imputing the M values for each vertex based on the 
        proportional length along the LineString.
        
        Parameters
        ----------
        m : array-like
            The M values to set for each vertex.
        inplace : bool, default True
            Whether to modify the object in place or return a new object.
        """
        # Check input type
        if array is not None:
            return self.set_m_from_array(array, inplace=inplace)
        elif (beg is not None) or (end is not None):
            beg = 0 if beg is None else beg
            return self.set_m_from_bounds(beg=beg, end=end, inplace=inplace)
    
    def set_m_from_array(self, m, inplace=True):
        """
        Set the M values of the LineStringM object to the specified values.
        
        Parameters
        ----------
        m : array-like
            The M values to set for each vertex.
        inplace : bool, default True
            Whether to modify the object in place or return a new object.
        """
        obj = self if inplace else self.copy()
        obj.m = m
        return obj if not inplace else None    
        
    def set_m_from_bounds(self, beg=0, end=None, inplace=True):
        """
        Set the M values of the LineStringM object to a range from beg to end,
        imputing the M values for each vertex based on the proportional length 
        along the LineString.
        
        Parameters
        ----------
        beg : float, default 0
            The starting M value.
        end : float, default None
            The ending M value. If None, the ending M value will be set to be
            beg plus the length of the LineString.
        inplace : bool, default True
            Whether to modify the object in place or return a new object.
        """
        # Set the ending M value
        if end is None:
            end = beg + self.geom.length
        # Compute the M values for each vertex
        m_span = end - beg
        m = np.append(beg, self.chord_proportions.cumsum() * m_span + beg)
        m[-1] = end  # Ensure exact match at the end
        # Update the M values
        return self.set_m_from_array(m, inplace=inplace)
    
    def m_to_distance(self, m, snap=False, _skip_validation=False):
        """
        Return the distance along the LineString for the specified M value.

        Parameters
        ----------
        m : float
            The M value to find the distance for.
        snap : bool, default False
            Whether to snap the M value to the nearest vertex if it is out of 
            range. If False, a ValueError will be raised if the M value is out
            of range.
        _skip_validation : bool, default False
            Internal parameter to skip validation when already performed by caller.
            Not intended for public use.
        """
        # Check input snapping
        if not _skip_validation:
            m, snapped = self._check_snapping(m, m=True, snap=snap)
            if snapped != 0:
                return 0 if snapped == 1 else self.geom.length
        # Find the nearest M value
        index = np.searchsorted(self.m, m)
        if index == 0:
            return 0
        elif index == len(self.m):
            return self.geom.length
        # Compute the proportional distance between the two nearest M values
        prop = (m - self.m[index - 1]) / (self.m[index] - self.m[index - 1])
        # Use cached cumulative distances for better performance
        cumdist = self._cumulative_distances
        distance = cumdist[index - 1]
        distance += (cumdist[index] - cumdist[index - 1]) * prop
        return distance
    
    def m_to_norm_distance(self, m, snap=False):
        """
        Return the normalized distance along the LineString for the specified M 
        value.

        Parameters
        ----------
        m : float
            The M value to find the normalized distance for.
        snap : bool, default False
            Whether to snap the M value to the nearest vertex if it is out of 
            range. If False, a ValueError will be raised if the M value is out
            of range.
        """
        return self.m_to_distance(m, snap) / self.geom.length
    
    def distance_to_m(self, distance, normalized=False, snap=False, _skip_validation=False):
        """
        Return the M value for the specified distance along the LineString.
        
        Parameters
        ----------
        distance : float
            The distance along the geometry to find the M value for.
        normalized : bool, default False
            Whether the distance is normalized (0-1) or absolute.
        snap : bool, default False
            Whether to snap the distance to the nearest vertex if it is out of
            range. If False, a ValueError will be raised if the distance is out
            of range.
        _skip_validation : bool, default False
            Internal parameter to skip validation when already performed by caller.
            Not intended for public use.
        """
        # Check if M values are defined
        if self.m is None:
            raise ValueError("Cannot convert distance to M value: M values are not defined")
        # Check input snapping
        if not _skip_validation:
            distance, snapped = self._check_snapping(distance, normalized=normalized, m=False, snap=snap)
            if snapped != 0:
                return self.m[0] if snapped == 1 else self.m[-1]
        
        # Convert normalized distance to absolute if needed
        if normalized:
            distance = distance * self.geom.length
        
        # Use cached cumulative distances for efficient lookup
        cumdist = self._cumulative_distances
        
        # Find the segment containing this distance using binary search
        index = np.searchsorted(cumdist, distance)
        
        if index == 0:
            return self.m[0]
        elif index >= len(cumdist):
            return self.m[-1]
        
        # Check if distance exactly matches a vertex
        if np.isclose(cumdist[index], distance):
            return self.m[index]
        
        # Interpolate M value within the segment
        segment_start_dist = cumdist[index - 1]
        segment_end_dist = cumdist[index]
        prop = (distance - segment_start_dist) / (segment_end_dist - segment_start_dist)
        
        return self.m[index - 1] + (self.m[index] - self.m[index - 1]) * prop
    
    def project(self, point, normalized=False, m=False):
        """
        Return the distance along the linear geometry to the nearest point
        on the geometry from the specified point.

        Parameters
        ----------
        point : shapely.Point
            The point to project onto the geometry.
        normalized : bool, default False
            Whether to return the distance as normalized (0-1) or absolute.
        m : bool, default False
            Whether to return the distance as an M value.
        """
        # Compute the projected distance
        distance = self.geom.project(point, normalized=normalized)
        if m:
            distance = self.distance_to_m(distance, normalized=normalized, snap=False)
        return distance

    def interpolate(self, distance, normalized=False, m=False, snap=False):
        """
        Return a point at the specified distance along the linear geometry.

        Parameters
        ----------
        distance : float
            The distance along the geometry to interpolate.
        normalized : bool, default False
            Whether the distance is normalized (0-1) or absolute.
        m : bool, default False
            Whether the distance should be interpreted as an M value.
        snap : bool, default False
            Whether to snap the distance to the nearest vertex if it is out of 
            range. If False, a ValueError will be raised if the distance is out
            of range.
        """
        # Check input snapping
        distance = self._check_snapping(distance, normalized=normalized, m=m, snap=snap)[0]
        # Compute the interpolated point
        if normalized:
            return self.geom.interpolate(distance, normalized=True)
        if m:
            distance = self.m_to_distance(distance, snap=False)
        return self.geom.interpolate(distance, normalized=False)

    def cut(self, beg, end, normalized=False, m=False, snap=False):
        """
        Return a LineStringM object with the geometry cut between the specified
        distances along the linear geometry.

        Parameters
        ----------
        beg : float
            The distance along the geometry to start the cut geometry.
        end : float
            The distance along the geometry to end the cut geometry.
        normalized : bool, default False
            Whether the distances are normalized (0-1) or absolute.
        m : bool, default False
            Whether the distances should be interpreted as M values.
        snap : bool, default False
            Whether to snap the distances to the nearest vertex if they are out
            of range. If False, a ValueError will be raised if the distances are
            out of range.
        """
        # Validate input parameters
        if normalized and m:
            raise ValueError('normalized and m cannot both be True')
        
        # Validate and snap inputs once, then convert efficiently
        # Key optimization: validate once with _check_snapping, then skip validation in conversion methods
        if m:
            # Input is in M values - validate once, then convert to distance
            beg_m = self._check_snapping(beg, normalized=False, m=True, snap=snap)[0]
            end_m = self._check_snapping(end, normalized=False, m=True, snap=snap)[0]
            # Convert M to distance (skip validation since already done)
            beg = self.m_to_distance(beg_m, _skip_validation=True)
            end = self.m_to_distance(end_m, _skip_validation=True)
        else:
            # Input is in distance - validate once, then convert to M
            beg = self._check_snapping(beg, normalized=normalized, m=False, snap=snap)[0]
            end = self._check_snapping(end, normalized=normalized, m=False, snap=snap)[0]
            # Convert distance to M (skip validation since already done)
            # For normalized distances, convert to absolute first
            if normalized:
                beg_dist = beg * self.geom.length
                end_dist = end * self.geom.length
            else:
                beg_dist = beg
                end_dist = end
            beg_m = self.distance_to_m(beg_dist, normalized=False, _skip_validation=True)
            end_m = self.distance_to_m(end_dist, normalized=False, _skip_validation=True)

        # Compute the substring of the LineString
        new_geom_coords, new_geom_m = substring_m_coords(
            shapely.get_coordinates(self.geom),
            self.m,
            beg,
            end,
            normalized=normalized
        )
        new_geom = LineString(new_geom_coords)

        # Replaced with custom substring function to handle M values with 
        # improved performance
        # new_geom = shapely.ops.substring(self.geom, beg, end, normalized=normalized)
        # new_geom_coords = shapely.get_coordinates(new_geom)

        # Address zero-length geometries
        if new_geom.length == 0:
            # Warn for now
            warnings.warn('Zero-length geometry created', RuntimeWarning)
            new_geom = LineString([new_geom_coords[0], new_geom_coords[0]])
        
        # # Compute the M values for the substring
        # if self.m is not None:
        #     # NOTE: Because conversions between distance and M are based on 
        #     # the index of the first match, we need to account for cases of 
        #     # duplicate M values at the start of the cut geometry. This 
        #     # should not be needed for the end of the cut geometry.
        #     # middle = self.m[np.logical_and(self.m > beg_m, self.m < end_m)]
        #     # front = np.repeat(beg_m, max(1, np.sum(self.m == beg_m)))
        #     # print(f'Front: {front}; based on {np.sum(self.m == beg_m)}')
        #     # back  = np.array([end_m])
        #     # new_geom_m = np.concatenate([front, middle, back])
        #     new_geom_m = np.concatenate([beg_m, self.m[np.logical_and(self.m >= beg_m, self.m <= end_m)], end_m])

        # else:
        #     new_geom_m = None
        # Identify and address cases where number of M values does not match
        # number of vertices (due to the substring operation producing zero-
        # length chords)
        if new_geom_m is not None:
            if len(new_geom_m) < len(new_geom_coords):
                # Warn for now
                warnings.warn(
                    "M values length does not match number of vertices in cut "
                    "geometry; adjusting to match", RuntimeWarning)
                # Recompute M values based on proportional lengths
                # NOTE: Removing the first vertex may not be appropriate given 
                # the assembly of M values above, which accounts for duplicate M
                # values at the start of the cut geometry. This may still be 
                # needed for ends. If an error brings this up again, revisit.
                chord_lengths = get_chord_lengths(new_geom, normalized=False)
                if chord_lengths[0] == 0:
                    new_geom = LineString(new_geom_coords[1:])
                elif chord_lengths[-1] == 0:
                    new_geom = LineString(new_geom_coords[:-1])
                else:
                    raise ValueError(
                        f"M values length of {len(new_geom_m)} does not match number "
                        f"of vertices in cut geometry of {len(new_geom_coords)}."
                    )
        return LineStringM(new_geom, m=new_geom_m)


#
# Helper functions
#

def _linemerge_m_geometry(objs):
    """
    Merge multiple LineStringM objects into a single shapely LineString
    or MultiLineString geometry.
    """
    # Merge geometries
    return shapely.ops.linemerge([obj.geom for obj in objs], directed=True)

def _linemerge_m_mapping(objs, allow_multiple=False, allow_mismatch=False, cast_geom=False):
    """
    Merge multiple LineStringM objects into a single LineStringM object 
    or list of non-contiguous LineStringM objects, returning the merged 
    geometries, the indices of the original geometries within the merged 
    geometries, and the indices of which merged geometry each original
    geometry belongs to, i.e., the geometry's chain index.

    This is done by first merging the geometries using an extension on 
    shapely's linemerge function, enforcing directionality. Then, instances 
    of multipart geometries are analyzed, stepping through each to identify 
    the order of the original geometries that compose them. During this
    process, the indices of the original geometries are recorded to allow
    for proper mapping back to the merged geometries.

    Parameters
    ----------
    objs : list of LineStringM
        The LineStringM objects to merge.
    allow_multiple : bool, default False
        Whether to allow multiple merged geometries to be returned. Note, 
        unless squeeze is True, results will always be returned as a list, 
        even if allow_multiple is False.
    allow_mismatch : bool, default False
        Whether to allow the M values of the merged geometries to be 
        mismatched.
    cast_geom : bool, default False
        Whether to cast input LineString objects to LineStringM objects.
    """
    # Validate input objects
    sml_geoms = []
    for obj in objs:
        if not isinstance(obj, LineStringM):
            if cast_geom:
                try:
                    obj = LineStringM(obj)
                except:
                    raise ValueError(
                        'All objects must be LineStringM or shapely.LineString '
                        'instances')
            else:
                raise ValueError(
                    'All objects must be LineStringM instances')
        sml_geoms.append(obj)

    # Merge geometries
    merged = _linemerge_m_geometry(sml_geoms)

    # Check if the merged geometry is a single or multiple geometries
    try:
        # Multipart geometry
        geom_iter = merged.geoms
    except AttributeError:
        # Single geometry
        geom_iter = [merged]
    # Raise error if multiple geometries are not allowed
    if len(geom_iter) > 1 and not allow_multiple:
        raise GeometryTypeError(
            'Multiple merged geometries detected. Set allow_multiple=True '
            'to perform merge anyways.')

    # Determine the order of merged geometries
    orders = []
    chains = np.zeros(len(sml_geoms), dtype=int)
    indices = list(range(len(sml_geoms)))
    # Iterate over multipart geometries
    for i, big_geom in enumerate(geom_iter):
        order = []
        # Identify the first node in the current merged geometry
        big_coords = shapely.get_coordinates(big_geom)
        # Extract coordinates for improved performance
        big_index = 0
        node = big_coords[0]
        node_m = None
        # Find the original geometry that starts at the first node
        cycle = 0
        while cycle >= 0:
            cycle += 1
            for j in indices:
                # Check if the node is present in the indexed single geometry
                sml_geom = sml_geoms[j] # Type: LineStringM
                sml_coords = shapely.get_coordinates(sml_geom.geom)
                if np.array_equal(sml_coords[0], node):
                    if np.array_equal(
                        big_coords[big_index:big_index + sml_coords.shape[0]],
                        sml_coords
                    ):
                        big_index += sml_coords.shape[0] - 1
                    else:
                        continue
                    # Check if the M values are consistent
                    if node_m is not None:
                        if node_m != sml_geom.m[0]:
                            msg = (
                                "Inconsistent M values detected at the "
                                "termini of adjacent chained geometries: "
                                f"M values {node_m} and {sml_geom.m[0]}."
                            )
                            if not allow_mismatch:
                                raise GeometryTopologyError(msg)
                            else:
                                warnings.warn(msg)
                    # Log geometry order and chain index
                    order.append(j)
                    chains[j] = i
                    # Remove spent original geometry
                    indices.remove(j)
                    # Check if we have reached the end of the merged geometry
                    if big_index >= big_coords.shape[0] - 1:
                        cycle = -1
                    # Progress to the next node
                    else:
                        node = sml_coords[-1]
                        node_m = sml_geom.m[-1] if sml_geom.m is not None else None
                    # Break out of the for loop and reset the index search
                    break
                else:
                    # Node is not present in the indexed single geometry, try 
                    # next
                    continue
            else:
                # No matching geometry found for the current node
                raise GeometryTopologyError(
                    "Unable to definitively determine chained geometry "
                    "topology. Geometries may contain branching, overlaps, or "
                    "other complexities not currently supported."
                )
        orders.append(order)

    # Merge LineStringM objects
    return geom_iter, orders, chains

def line_merge_m_old(objs, allow_multiple=False, allow_mismatch=False, squeeze=False, return_index=False, cast_geom=False):
    """
    Merge multiple LineStringM objects into a single LineStringM object 
    or list of non-contiguous LineStringM objects.

    Parameters
    ----------
    objs : list of LineStringM
        The LineStringM objects to merge.
    allow_multiple : bool, default False
        Whether to allow multiple merged geometries to be returned. Note, 
        unless squeeze is True, results will always be returned as a list, 
        even if allow_multiple is False.
    allow_mismatch : bool, default False
        Whether to allow the M values of the merged geometries to be mismatched.
    squeeze : bool, default False
        Whether to return a single LineStringM object when possible.
    return_index : bool, default False
        Whether to return the indices of where the original geometries fall
        within the merged geometry.
    cast_geom : bool, default False
        Whether to cast input LineString objects to LineStringM objects.
    """
    # Get merged geometries and orders
    geom_iter, orders = _linemerge_m_mapping(
        objs,
        allow_multiple=allow_multiple,
        allow_mismatch=allow_mismatch,
        cast_geom=cast_geom,
    )[:2]

    # Merge LineStringM objects
    new_objs = []
    for merged_geom_i, order in zip(geom_iter, orders):
        # Initialize the m data from the first ordered object
        m = objs[order[0]].m
        for j in order[1:]:
            m = np.append(m, objs[j].m[1:])
        new_objs.append(LineStringM(merged_geom_i, m))

    # Return the merged geometries
    if squeeze and len(new_objs) == 1:
        new_objs = new_objs[0]
    return (new_objs, orders) if return_index else new_objs

def line_merge_m(
    lines: list[LineStringM | shapely.LineString],
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
                    # coordinates
                    # to the current line's termini
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

def get_linestring_chains_old(objs):
    """
    Return the chain indices for each LineStringM object in the list.

    Parameters
    ----------
    objs : list of LineStringM
        The LineStringM objects to get the chain indices for.
    """
    # Get merged geometries and orders
    chains = _linemerge_m_mapping(objs, allow_multiple=True, allow_mismatch=False, cast_geom=True)[2]
    return chains

def get_linestring_chains(objs):
    """
    Return the chain indices for each LineStringM object in the list.

    Parameters
    ----------
    objs : list of LineStringM
        The LineStringM objects to get the chain indices for.
    """
    # Get merged geometries and orders
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
