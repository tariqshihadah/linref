from __future__ import annotations
import numpy as np
import copy
import warnings
import shapely
from shapely.geometry import LineString
from linref.errors import *
from linref.geometry.utilities import get_chord_lengths, substring_m_coords


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

        Alias of ``m_to_distance`` for single-geometry calls.

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
        from linref.geometry.operations import m_to_distance as _m_to_distance
        return _m_to_distance(self, m)
    
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

        Alias of ``distance_to_m`` for single-geometry calls.
        
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
        from linref.geometry.operations import distance_to_m as _distance_to_m
        return _distance_to_m(self, distance, normalized=normalized)
    
    def project(self, point, normalized=False, m=False):
        """
        Return the distance along the linear geometry to the nearest point
        on the geometry from the specified point.

        Alias of `line_locate_point_m` for single-geometry calls.

        Parameters
        ----------
        point : shapely.Point
            The point to project onto the geometry.
        normalized : bool, default False
            Whether to return the distance as normalized (0-1) or absolute.
        m : bool, default False
            Whether to return the distance as an M value.
        """
        from linref.geometry.operations import line_locate_point_m
        return line_locate_point_m(self, point, normalized=normalized, m=m)

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

    def reverse(self, inplace: bool = False):
        """
        Reverse the direction of the geometry and its M values.

        Parameters
        ----------
        inplace : bool, default False
            Whether to modify the object in place or return a new object.

        Raises
        ------
        NotImplementedError
            LineStringM does not currently support reversing. This method is 
            a placeholder for future support of non-monotonic M-enabled 
            geometries.
        """
        raise NotImplementedError(
            "LineStringM does not currently support reversing. Reversing "
            "would produce non-monotonic M values, which are not supported "
            "by LineStringM at this time."
        )

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

        # Address zero-length geometries
        if new_geom.length == 0:
            # Warn for now
            warnings.warn('Zero-length geometry created', RuntimeWarning)
            new_geom = LineString([new_geom_coords[0], new_geom_coords[0]])
        
        # Identify and address cases where number of M values does not match
        # number of vertices (due to the substring operation producing zero-
        # length chords)
        if new_geom_m is not None:
            if len(new_geom_m) < len(new_geom_coords):
                # Warn for now
                warnings.warn(
                    "M values length does not match number of vertices in cut "
                    "geometry; adjusting to match", RuntimeWarning)
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
