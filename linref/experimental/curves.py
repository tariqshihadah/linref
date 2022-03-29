from shapely.geometry import LineString, MultiLineString
import numpy as np
import math


class CurveDetector(object):
    """
    Class for detecting curves along polylines based on geometric features of 
    rays, arcs, and complex segments.

    To-do
    -----
    - Deal with adjacent curves/reverse curves
    - Minimum points/length for curve definition
    - Add point/length buffer
    - Full curve radius estimate
    """

    def __init__(self, line):
        # Log parameters
        self.line = line

        # Initialize fitting properties
        self._segment_mask = None

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, line):
        if not isinstance(line, (LineString, MultiLineString)):
            raise TypeError("Input line must be valid shapely linear geometry.")
        else:
            self._line = line

    @property
    def segment_mask(self):
        """
        A boolean array mask indicating which 4-point segments have been 
        detected to be part of a fitted curve.
        """
        return self._segment_mask

    @property
    def point_mask(self):
        """
        A boolean array mask indicating which points have been detected to be 
        part of a fitted curve
        """
        size = self.size
        mask = np.zeros(size, dtype=bool)
        for i in range(4):
            mask[i:size+i-3] = mask[i:size+i-3] | self.segment_mask
        return mask

    @property
    def point_map(self):
        """
        Array of values indicating which points are associated with which 
        curves. Points with a value of 0 are not associated with any unique 
        fitted curve.
        """
        mask = self.point_mask
        res = (np.append([True], np.diff(mask * 1) > 0) * mask).cumsum() * mask
        return res

    @property
    def curves(self):
        """
        A list of shapely LineStrings for fitted curves.
        """
        # Iterate through unique curve numbers
        point_map = self.point_map
        lines = []
        for num in np.unique(point_map)[1:]:
            # Get mask for curve number
            mask = point_map == num
            # Create curve linestring
            lines.append(LineString(list(zip(self.xs[mask], self.ys[mask]))))
        return lines
            
    @property
    def size(self):
        return len(self.xs)

    @property
    def xs(self):
        # Compute x values
        xs = np.array(self.line.xy[0], dtype=float)
        return xs

    @property
    def ys(self):
        # Compute x values
        ys = np.array(self.line.xy[1], dtype=float)
        return ys

    @property
    def dx(self):
        """
        X-dimension distance between adjacent points.
        Size = n - 1
        """
        dx = np.diff(self.xs)
        return dx

    @property
    def dy(self):
        """
        Y-dimension distance between adjacent points.
        Size = n - 1
        """
        dy = np.diff(self.ys)
        return dy

    @property
    def bearing(self):
        """
        Bearing of the ray defined by two adjacent points.
        Size = n - 1
        """
        bearing = np.arctan2(self.dy, self.dx)
        return bearing

    @property
    def ray_length(self):
        """
        Length of the ray defined by two adjacent points.
        Size = n - 1
        """
        ray_length = (self.dx ** 2 + self.dy ** 2) ** 0.5
        return ray_length

    @property
    def relangle(self):
        """
        Relative angle between two adjacent rays.
        Size = n - 2
        """
        relangle = np.diff(self.bearing)
        return relangle

    @property
    def direction(self):
        """
        The direction of the relative angle between two adjacent rays, 
        indicating a left-hand angle (1) and a right-hand angle (0).
        Size = n - 2
        """
        direction = (self.relangle > 0) * 1
        return direction

    @property
    def span(self):
        """
        Span length of each 3-point arc, measuring between the begin and end 
        points of each arc defined by two adjacent rays.
        Size = n - 2
        """
        dx, dy = self.dx, self.dy
        span = ((dx[:-1]+dx[1:])**2+(dy[:-1]+dy[1:])**2)**0.5
        return span
    
    @property
    def span_ratio(self):
        """
        Ratio of the smaller ray to the larger ray of each arc defined by 
        two adjacent rays.
        Size = n - 2
        """
        ray_length = self.ray_length
        span_ratio = ray_length[:-1]/ray_length[1:]
        span_ratio = np.where(span_ratio>1, 1/span_ratio, span_ratio)
        return span_ratio

    def span_index(self, span_ratio_sensitivity=0.2):
        """
        Abstract quantification of the influence of each arc's span ratio on 
        curve detection given an input sensitivity value between 0 and 1.
        Size = n - 2
        
        Parameters
        ----------
        span_ratio_sensitivity : number [0, 1], default 0.35
            A measure of the sensitivity of the detector to a given arc's span 
            ratio where 0 means no sensitivity and 1 means full sensitivity.
        """
        span_index = self.span_ratio*span_ratio_sensitivity+(1-span_ratio_sensitivity)
        return span_index

    @property
    def radius(self):
        """
        The radius of each arc defined by two adjacent rays.
        Size = n - 2
        """
        # Compute radius based on arc span and relative angle
        radius = self.span/(2*np.sin(math.pi-self.relangle))
        # Address effective tangents
        radius = np.where(self.relangle<math.pi, radius, np.inf)
        return radius

    @property
    def central_angle(self):
        """
        The central angle of each arc defined by two adjacent rays.
        Size = n - 2
        """
        central_angle = 2*np.arcsin(self.span/(2*self.radius))
        return central_angle

    @property
    def arc_length(self):
        """
        The actual outer arc length of each arc defined by two adjacent rays.
        Size = n - 2
        """
        arc_length = self.central_angle*self.radius
        return arc_length

    @property
    def radius_max(self):
        """
        The maximum radius between all pairs of adjacent arcs (i.e., 4-point 
        segments).
        Size = n - 3
        """
        radius_max = np.max([self.radius[:-1],self.radius[1:]], axis=0)
        return radius_max

    @property
    def radius_dif(self):
        """
        The mathematical difference in radius between all pairs of adjacent 
        arcs (i.e., 4-point segments).
        Size = n - 3
        """
        radius_dif = (self.radius[1:]-self.radius[:-1])
        return radius_dif

    @property
    def radius_scale(self):
        """
        The mathematical difference in radius between all pairs of adjacent 
        arcs (i.e., 4-point segments), normalized by the maximum radius between 
        each pair.
        Size = n - 3
        """
        radius_scale = self.radius_dif/self.radius_max
        return radius_scale

    def fit(self, max_radius=10000, max_radius_scale=0.65, span_ratio_sensitivity=0.35):
        """
        A test of whether or not adjacent arcs have similar radii based on 
        input detection parameters for the maximum radius scale and the 
        maximum radius which can be detected within a curve.
        Size = n - 3
        
        Parameters
        ----------
        max_radius : number, default 10000
            The maximum radius to be considered within a detected curve.
        max_radius_scale : float [0, 1], default 0.65
            The maximum span ratio-adjusted radius scale value to be considered 
            within a detected curve. If span_ratio_sensitivity == 0, no 
            adjustements to the radius scale are made.
        span_ratio_sensitivity : number [0, 1], default 0.35
            A measure of the sensitivity of the detector to a given arc's span 
            ratio where 0 means no sensitivity and 1 means full sensitivity.
        """
        # Compute span index
        # - Span index is used to adjust sensitivity to differing radii which 
        #   may result from inconsistently spaced line points (i.e., large 
        #   span index values)
        span_index = self.span_index(span_ratio_sensitivity)
        # Compute radius match boolean mask
        self._segment_mask = \
            (np.abs(self.radius_scale*span_index[:-1]*span_index[1:]) < max_radius_scale) & \
            (np.abs(self.radius_max) < max_radius)
