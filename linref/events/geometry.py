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

    def __str__(self):
        return self.wkt + r' # linref compatibility approximation'
    
    def __repr__(self):
        return self.__str__()

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
        # Update the M values
        return self.set_m_from_array(m, inplace=inplace)
    
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
        else:
            m = None
        return LineStringM(new_geom, m=m)
        

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
    objs_prepared = []
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
        objs_prepared.append(obj)

    # Merge geometries
    merged_geom = _linemerge_m_geometry(objs_prepared)

    # Check if the merged geometry is a single or multiple geometries
    try:
        geom_iter = merged_geom.geoms
    except AttributeError:
        geom_iter = [merged_geom]
    if len(geom_iter) > 1 and not allow_multiple:
        raise ValueError(
            'Multiple merged geometries detected. Set allow_multiple=True '
            'to perform merge anyways.')

    # Determine the order of merged geometries
    orders = []
    chains = np.zeros(len(objs_prepared), dtype=int)
    indices = list(range(len(objs_prepared)))
    # Iterate over multipart geometries
    for i, merged_geom_i in enumerate(geom_iter):
        order = []
        # Identify the first node in the current merged geometry
        node = merged_geom_i.coords[0]
        node_m = None
        # Find the original geometry that starts at the first node
        recurse = True
        while recurse:
            for j in indices:
                # Check if the node is present in the indexed geometry
                obj = objs_prepared[j]
                if node in obj.geom.coords:
                    # Check if the M values are consistent
                    if node_m is not None:
                        if node_m != obj.m[0]:
                            msg = 'Inconsistent m values detected in merged geometry'
                            if not allow_mismatch:
                                raise ValueError(msg)
                            else:
                                warnings.warn(msg)
                    # Log geometry order and chain index
                    order.append(j)
                    chains[j] = i
                    # Remove spent original geometry
                    indices.remove(j)
                    # Progress to the next node
                    node = obj.geom.coords[-1]
                    node_m = obj.m[-1] if obj.m is not None else None
                    if node == merged_geom_i.coords[-1]:
                        recurse = False
                    break
        orders.append(order)

    # Merge LineStringM objects
    return geom_iter, orders, chains

def linemerge_m(objs, allow_multiple=False, allow_mismatch=False, squeeze=False, return_index=False, cast_geom=False):
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

def get_linestring_chains(objs):
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

def prepare_chained_linestring_m(objs):
    """
    Generate M values for a list of LineStringM objects that are chained 
    together, ensuring that the M values are continuous across the chain 
    boundaries.

    Parameters
    ----------
    objs : list of shapely.LineString or LineStringM
        The LineStringM objects to generate chained M values for.
    """
    # Validate input objects
    if not all(isinstance(obj, LineStringM) for obj in objs):
        try:
            objs = [LineStringM(obj, m=None) for obj in objs]
        except:
            raise ValueError(
                'All objects must be LineStringM or shapely.LineString '
                'instances')
        
    # Get merged geometries and orders
    orders, chains = _linemerge_m_mapping(objs, allow_multiple=True)[1:]
    # Generate chained M values
    for order in orders:
        # Initialize the m data from the first ordered object
        m_begin = 0
        for j in order:
            objs[j].set_m_from_bounds(beg=m_begin, inplace=True)
            m_begin = objs[j].m[-1]
    return objs, chains
    
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
        lengths = lengths / ls.length
    return lengths