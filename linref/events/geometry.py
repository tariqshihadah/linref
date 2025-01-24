from __future__ import annotations
import numpy as np
import copy
import warnings
import shapely
from shapely.geometry import LineString, Point


class LineStringM:

    def __init__(self, geom, m=None):
        self.geom = geom
        self.m = m

    @property
    def geom(self):
        return self._geom
    
    @geom.setter
    def geom(self, geom):
        if not isinstance(geom, shapely.geometry.LineString):
            raise ValueError('LineStringM geom must be a shapely LineString')
        try:
            if geom.has_z:
                raise ValueError('LineStringM geom must not have z values')
            if geom.has_m:
                raise ValueError(
                    "LineStringM geom must not have m values. Future versions "
                    "will transition to native LineString with M support.")
        except AttributeError:
            pass
        self._geom = geom

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
                    raise ValueError('LineStringM m must be a numpy array')
            # Enforce input shape
            if len(m) != len(self.geom.coords):
                raise ValueError('LineStringM m must be same length as coords')
            # Enforce monotonic increase
            if not np.all(np.diff(m) >= 0):
                raise ValueError('LineStringM m must be monotonic increasing')
        self._m = m

    @property
    def coords(self):
        """
        Return the coordinates of the LineString as a list, interwoven
        with the M values.
        """
        if self.m is None:
            return self.geom.coords
        return [(x[0], x[1], m) for x, m in zip(self.geom.coords, self.m)]
    
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
    
    def m_to_distance(self, m, snap=False):
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
        """
        # Check input snapping
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
        # Get length up to the indexed vertice
        if index == 1:
            distance = 0
        else:
            distance = LineString(self.geom.coords[:index]).length
        distance += LineString(self.geom.coords[index - 1: index + 1]).length * prop
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
    
    def distance_to_m(self, distance, normalized=False, snap=False):
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
        """
        # Check if M values are defined
        if self.m is None:
            raise ValueError('M values are not defined')
        # Check input snapping
        distance, snapped = self._check_snapping(distance, normalized=normalized, m=False, snap=snap)
        if snapped != 0:
            return self.m[0] if snapped == 1 else self.m[-1]
        
        # Get the nearest vertice index to the left of the specified distance
        substring = shapely.ops.substring(
            self.geom, 0, distance, normalized=normalized)
        # Determine which endpoint to use
        if substring.coords[-1] in self.geom.coords:
            endpoint = substring.coords[-1]
            index = list(self.geom.coords).index(endpoint)
            return self.m[index]
        else:
            endpoint = substring.coords[-2]
            index = list(self.geom.coords).index(endpoint)

        # Compute the M value for the substring and remaining distance
        if index == 0:
            distance_to_vertice = 0
        else:
            distance_to_vertice = LineString(self.geom.coords[: index + 1]).length
        prop = (distance - distance_to_vertice) / \
            LineString(self.geom.coords[index: index + 2]).length
        return self.m[index] + (self.m[index + 1] - self.m[index]) * prop

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
        # Check input snapping and transform if needed
        if m:
            beg = self.m_to_distance(beg, snap=snap)
            end = self.m_to_distance(end, snap=snap)
        else:
            beg = self._check_snapping(beg, normalized=normalized, m=m, snap=snap)[0]
            end = self._check_snapping(end, normalized=normalized, m=m, snap=snap)[0]

        # Compute the substring of the LineString
        new_geom = shapely.ops.substring(self.geom, beg, end, normalized=normalized)

        # Address zero-length geometries
        if new_geom.length == 0:
            # Warn for now
            warnings.warn('Zero-length geometry created', RuntimeWarning)
            new_geom = LineString([new_geom.coords[0], new_geom.coords[0]])
        
        # Compute the M values for the substring
        if self.m is not None:
            beg_m = self.distance_to_m(beg, normalized=normalized, snap=snap)
            end_m = self.distance_to_m(end, normalized=normalized, snap=snap)
            m = self.m[np.logical_and(self.m > beg_m, self.m < end_m)]
            m = np.insert(m, 0, beg_m)
            m = np.append(m, end_m)
            display(m, beg_m, end_m, new_geom.coords[:])
        else:
            m = None
        return LineStringM(new_geom, m=m)
        