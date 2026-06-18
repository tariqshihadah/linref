from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import hausdorff_distance
from itertools import starmap
from shapely.ops import nearest_points, substring
from shapely.errors import GeometryTypeError
from pandas.api.extensions import register_dataframe_accessor
from linref.errors import LRSConfigurationError, LRSCompatibilityError, GeometryTopologyError
from linref.utility.utility import label_list_or_none


def parallel_project_hausdorff(
    target: gpd.GeoDataFrame,
    projected: gpd.GeoDataFrame,
    buffer: float = 0,
    max_distance: float | None = None,
    match: int = 1,
    densify: float | None = None,
    replace: bool = False,
) -> gpd.GeoDataFrame:
    """
    Experimental class for performing projections of linear geometries onto
    a primary linearly referenced layer using a series of tests based on the
    Hausdorff distance metric.

    The methodology involves the following steps:

    1. Identify candidate target geometries by joining the bounds of each 
    projected geometry to the target data using a specified buffer distance.
    Only target geometries within the buffer distance of both ends of the 
    projected geometry are retained as candidates.

    2. For each candidate target geometry, compute the Hausdorff distance 
    between the projected geometry and the target geometry. This distance
    quantifies the maximum distance of a point on one geometry to the nearest
    point on the other geometry.

    3. Select the target geometry with the minimum Hausdorff distance as the
    best match for the projected geometry, or specify multiple matches if 
    desired.

    4. Project the endpoints of the projected geometry onto the matched target
    geometry to obtain linear referencing information, producing a new 
    dataframe with the projected geometries referenced to the target's linear
    referencing system.

    Parameters
    ----------
    target : gpd.GeoDataFrame
        The target GeoDataFrame with m-enabled geometries to project onto.
    projected : gpd.GeoDataFrame
        The GeoDataFrame containing geometries to be projected. These
        geometries must be singlepart.
    buffer : float, default 0
        The buffer distance (in the units of the CRS) to use when identifying
        candidate target geometries.
    max_distance : float, optional
        The maximum allowable Hausdorff distance (in the units of the CRS)
        for a candidate target geometry to be considered a valid match. 
        Geometries with a Hausdorff distance exceeding this value will not be
        matched. If not specified, will default to be equal to the buffer 
        value.
    match : int, default 1
        The number of closest matching target geometries to return for each
        projected geometry based on the Hausdorff distance metric. If set to
        1, only the closest match is returned. If set to a value greater than 
        1, the specified number of closest matches are returned. If set to
        0, all candidate matches within the max_distance are returned.
    densify : float, optional
        A value between 0 and 1 indicating the fraction of each geometry's
        length to use for densification when computing the Hausdorff distance.
        Densification adds additional vertices along the geometry to improve
        the accuracy of the distance calculation. If not specified, no
        densification is applied.
    replace : bool, default False
        If True, will replace any existing linear referencing columns in the
        projected GeoDataFrame with the newly computed values. If False, will
        raise an error if the projected GeoDataFrame already contains linear
        referencing columns from the target's system.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame containing the projected geometries with linear
        referencing information based on the target's system.
    """
    # Validate input dataframes
    if not isinstance(target, gpd.GeoDataFrame):
        raise TypeError(
            "Input projection target must be valid gpd.GeoDataFrame "
            "instance."
        )
    if not isinstance(projected, gpd.GeoDataFrame):
        raise TypeError(
            "Input projected data must be valid gpd.GeoDataFrame instance."
        )
    # Ensure that target data has a valid LRS configuration
    if not target.lr.is_linear:
        raise LRSConfigurationError(
            "Input projection target GeoDataFrame must have a valid "
            "linear referencing configuration."
        )
    # Ensure that target data has m-enabled geometries
    if not target.lr.is_spatial_m:
        raise LRSConfigurationError(
            "Input projection target GeoDataFrame must have "
            "m-enabled geometries."
        )
    # Ensure that the target data has singlepart geometries
    if not all(target.geometry.count_geometries() == 1):
        raise GeometryTopologyError(
            "Input target GeoDataFrame must have singlepart geometries."
        )
    # Ensure that the projected data has singlepart geometries
    if not all(projected.geometry.count_geometries() == 1):
        raise GeometryTopologyError(
            "Input projected GeoDataFrame must have singlepart "
            "geometries."
        )
    # Check for existing linear referencing columns in projected data
    target_cols = target.lr.key_col + [target.lr.beg_col, target.lr.end_col]
    if not replace:
        if len(set(target_cols) & set(projected.columns)) > 0:
            raise LRSCompatibilityError(
                "Input projected GeoDataFrame already contains linear "
                "referencing columns from the target's system. To "
                "overwrite these columns, set replace=True."
            )
    # Validate additional parameters
    try:
        buffer = float(buffer)
        assert buffer >= 0
    except:
        raise ValueError(
            "Buffer parameter must be a non-negative numeric value."
        )
    if max_distance is None:
        max_distance = buffer
    else:
        try:
            max_distance = float(max_distance)
            assert max_distance >= 0
        except:
            raise ValueError(
                "Max distance parameter must be a non-negative numeric value."
            )
    try:
        match = int(match)
        assert match >= 0
    except:
        raise ValueError(
            "Match parameter must be a non-negative integer value."
        )
    if densify is not None:
        try:
            densify = float(densify)
            assert 0 <= densify <= 1
        except:
            raise ValueError(
                "Densify parameter must be a float value between 0 and 1."
            )
        
    # Create buffered target geometries for candidate selection
    buffered = target.geometry.buffer(buffer).to_frame(name='__target_buffered__')
    buffered = buffered.rename_axis('__target_index__')

    # Create multipoint geometries for projected geometry endpoints
    boundaries = projected.geometry.boundary.to_frame(name='__projected_boundary__')
    boundaries = boundaries.rename_axis('__projected_index__')

    # Identify candidate target geometries within buffer distance
    joined = gpd.sjoin(boundaries, buffered, how='inner', predicate='within').reset_index(drop=False)

    # Merge original geometries back into joined dataframe
    # FUTURE: Optimize to avoid multiple merges
    joined = pd.merge(
        joined,
        target.geometry.to_frame(name='__target_geometry__'),
        left_on='__target_index__',
        right_index=True
    )
    joined = pd.merge(
        joined,
        target[target.lr.geom_m_col].to_frame(name='__target_geometry_m__'),
        left_on='__target_index__',
        right_index=True
    )    
    joined = pd.merge(
        joined,
        projected.geometry.to_frame(name='__projected_geometry__'),
        left_on='__projected_index__',
        right_index=True
    )

    # Create substrings of the target geometries for accurate distance 
    # computation
    def _create_substring(target, boundary):
        try:
            # Project boundary points onto target geometry
            beg_dist, end_dist = (
                target.project(boundary.geoms[0], normalized=False),
                target.project(boundary.geoms[-1], normalized=False)
            )
            # Create substring geometry
            return substring(
                target, beg_dist, end_dist, normalized=False
            )
        except (AttributeError, IndexError, GeometryTypeError):
            return None
    joined['__target_substring__'] = list(starmap(
        _create_substring,
        zip(joined['__target_geometry__'], joined['__projected_boundary__'])
    ))

    # Compute Hausdorff distances for candidate matches
    joined['__hausdorff_distance__'] = list(starmap(
        lambda a, b: hausdorff_distance(a, b, densify=densify),
        zip(joined['__target_substring__'], joined['__projected_geometry__'])
    ))

    # Filter candidates based on max distance
    joined = joined[joined['__hausdorff_distance__'] <= max_distance]
    # Select best matches based on Hausdorff distance if needed
    if match == 1:
        idx = joined.groupby('__projected_index__')['__hausdorff_distance__'].idxmin()
        joined = joined.loc[idx]
    elif match > 1:
        joined = joined.sort_values(
            by=['__projected_index__', '__hausdorff_distance__'],
            ascending=[True, True]
        )
        joined = joined.groupby('__projected_index__').head(match)

    # Retrieve linear referencing keys from target data
    joined = pd.merge(
        joined,
        target[target.lr.key_col],
        left_on='__target_index__',
        right_index=True
    )

    # Project projected geometry endpoints onto matched target geometries
    def _project_onto_target(linestring_m, boundary):
        try:
            # Project bounds onto target linestring_m
            beg_loc, end_loc = (
                linestring_m.project(boundary.geoms[0], m=True),
                linestring_m.project(boundary.geoms[-1], m=True)
            )
            return beg_loc, end_loc
        except (AttributeError, IndexError):
            return np.nan, np.nan
    joined[[target.lr.beg_col, target.lr.end_col]] = pd.DataFrame(list(starmap(
        _project_onto_target,
        zip(joined['__target_geometry_m__'], joined['__projected_boundary__'])
    )), index=joined.index, columns=[target.lr.beg_col, target.lr.end_col])

    # Construct final projected dataframe
    joined = pd.merge(
        projected.drop(columns=target_cols, errors='ignore'),
        joined[target_cols + ['__projected_index__']],
        left_index=True,
        right_on='__projected_index__',
        how='outer',

    )
    joined.index = joined['__projected_index__'].values
    joined = joined.drop(columns=['__projected_index__'])
    return joined


class ParallelProjector(object):
    """
    Experimental class for performing projections of linear geometries onto 
    linear events collections.

    The methodology used by this class involves the following steps:

    1. Create sample points along the projected geometries using a fixed 
    number of samples per geometry.

    2. Spatially join these sample points to the target EventsCollection's 
    geometry using the provided buffer distance to identify candidate matches.

    3. Process all possible matches using the .match() method to produce 
    linear referencing information for the projected geometries based on that 
    of the target EventsCollection.
    """

    def __init__(
        self,
        target: gpd.GeoDataFrame,
        projected: gpd.GeoDataFrame,
        samples=3,
        buffer=100
    ) -> None:
        self.target = target
        self.projected = projected
        self.samples = samples
        self.buffer = buffer

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, df):
        # Validate object type
        if not isinstance(df, gpd.GeoDataFrame):
            raise TypeError(
                "Input projection target must be valid gpd.GeoDataFrame "
                "instance."
            )
        # Ensure that data has m-enabled geometries
        if not df.lr.is_spatial_m:
            raise LRSConfigurationError(
                "Input projection target GeoDataFrame must have "
                "m-enabled geometries."
            )
        self._target = df

    @property
    def target_buffered(self):
        return self._target.set_geometry(
            self._target.geometry.buffer(self.buffer)
        )

    @property
    def projected(self):
        return self._projected

    @projected.setter
    def projected(self, df):
        # Validate object type
        if not isinstance(df, gpd.GeoDataFrame):
            raise TypeError(
                "Input projected data must be valid gpd.GeoDataFrame instance."
            )
        self._projected = df
    
    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        # Validate
        if not isinstance(samples, int):
            raise ValueError("Samples parameter must be an integer.")
        self._samples = samples

        # Build sampling points
        self._build_sample_points()

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, buffer):
        # Validate
        try:
            buffer = float(buffer)
            assert buffer >= 0
            self._buffer = buffer
        except:
            raise ValueError(
                "Buffer parameter must be a non-negative numeric value.")
        
        # Perform spatial join
        self._buffer_join()

    @property
    def sample_locs(self):
        return np.linspace(0, 1, num=self.samples)

    @property
    def sample_points(self):
        return self._sample_points

    @property
    def projectors(self):
        return self.projected.geometry.values

    def _build_sample_points(self):
        """
        Build sample points along each projector geometry for matching.
        """
        # Generate sampling points
        sample_locs = self.sample_locs
        points = []
        for projector in self.projectors:
            # Interpolate sample points
            points.extend([projector.interpolate(loc) for loc in \
                sample_locs * projector.length])
        self._sample_points = gpd.GeoDataFrame({
            '__projector__': np.repeat(self.projected.index.values, self.samples),
            'geometry': points}, geometry='geometry', crs=self.projected.crs
        )

    def _buffer_join(self):
        """
        Join projector sample points to target geometry within buffer for 
        matching.
        """
        # Join sampling points with target data
        joined = gpd.sjoin(
            self.target_buffered,
            self.sample_points,
            predicate='intersects',
            how='left'
        )
        # Define distance computation function
        def _get_distance(o1, o2):
            # If geometries are valid, compute distance
            try:
                p1, p2 = nearest_points(o1, o2)
                return p1.distance(p2)
            except (AttributeError, ValueError) as e:
                return -1
            
        # Compute distances between matched geometries
        joined.geometry = self.target.geometry
        geometries = joined[[joined.geometry.name, 'index_right']].merge(
            self.sample_points.geometry,
            left_on='index_right',
            right_index=True,
            how='left'
        )
        geometries = zip(
            geometries.iloc[:,0], # target geometry
            geometries.iloc[:,2], # sample point geometry
        )
        joined['__distance__'] = \
            list(map(lambda x: _get_distance(*x), geometries))

        # Process, clean results
        joined.index.rename('__target__', inplace=True)
        joined = joined \
            .reset_index(drop=False)[['__projector__','__target__','__distance__']]
        joined = joined \
            .sort_values(by=['__projector__','__target__'], ascending=True)
        joined = joined.dropna(how='any')
        self._joined = joined

    def match(self, match='all', choose=1, sort_locs=True):
        """
        Perform the actual matching of nearby geometries to one another based 
        on input analysis parameters, producing a dataframe which has been 
        applied to the target EventsCollection's linear referencing system.
        """
        # Validate matching parameters
        if match=='all':
            match = self.samples
        elif not isinstance(match, int):
            raise ValueError("Match parameter must be 'all' or an integer <= "
                             "samples.")
        # Validate choose parameter
        if not isinstance(choose, int):
            if not choose=='all':
                raise ValueError("Choose parameter must be 'all' or an "
                                "integer >= 1")
        elif choose < 1:
            raise ValueError("Integer choose parameter must be >= 1")
        
        # Get target event bound labels
        labels = [self.target.lr.beg_col, self.target.lr.end_col]

        # Group unique pairs of targets and projectors
        pair_unique, pair_index, pair_counts = np.unique(
            self._joined.values[:,:2].astype(int),
            axis=0,
            return_index=True,
            return_counts=True
        )
        # Test all unique pairs for minimum match count
        match_mask = pair_counts >= match
        # Compute mean distances for all unique matched pairs
        all_distances = np.split(self._joined.values[:,2], pair_index)[1:]
        split = np.array(all_distances, dtype=object)[match_mask]
        mean_distances = np.array([np.mean(i) for i in split])
        # Group matched targets for all matched projectors
        proj_unique, proj_index = np.unique(
            pair_unique[match_mask,0],
            axis=0,
            return_index=True
        )
        pair_distances = np.array(
            np.split(mean_distances, proj_index)[1:], dtype=object)
        
        # Identify the index of the target(s) with the lowest mean distance for 
        # each projector if requested
        pair_groups = np.split(pair_unique[match_mask,1], proj_index)[1:]
        if choose == 'all':
            pair_select = [slice(None) for i in pair_distances]
        elif choose == 1:
            pair_select = [np.argmin(i) for i in pair_distances]
        else:
            pair_select = [np.argpartition(i, min(choose, i.size)-1) \
                        [:min(choose, i.size)] \
                        for i in pair_distances]
        # - Produce a projector-target map
        zipped = zip(pair_groups, pair_select)
        targets = np.array(
            [group[index] for group, index in zipped], dtype=object)
        # - Flatten map if multiple choices
        if choose != 1:
            projectors = np.repeat(proj_unique, [len(i) for i in targets])
            targets = np.concatenate(targets, axis=0)
        else:
            projectors = proj_unique

        # Select the matched pairs
        matched_pairs = pd.DataFrame({
            '__projector__': projectors, '__target__': targets})
        
        # Merge matched records
        proj_lines = self.projected.geometry.rename('__proj_lines__')
        select = matched_pairs \
            .merge(proj_lines, left_on='__projector__', right_index=True)
        target_data = self.target[self.target.lr.key_col + [self.target.lr.geom_m_col]]
        select = select \
            .merge(target_data, left_on='__target__', right_index=True)
        select = select.reset_index(drop=True)

        # Project ends onto matched linestring_m
        def _project(linestring_m, line):
            try:
                # Project bounds onto target linestring_m
                boundary = line.boundary
                beg, end = boundary.geoms[0], boundary.geoms[-1]
                beg_loc, end_loc = linestring_m.project(beg, m=True), linestring_m.project(end, m=True)
                return beg_loc, end_loc
            except (AttributeError, IndexError):
                return np.nan, np.nan
        proj_bounds = np.asarray(list(map(
            _project,
            select[self.target.lr.geom_m_col].values,
            select['__proj_lines__'].values
        )))
            
        # Merge with input and target data
        if sort_locs:
            select[labels[0]] = proj_bounds.min(axis=1)
            select[labels[1]] = proj_bounds.max(axis=1)
        else:
            select[labels[0]] = proj_bounds[:, 0]
            select[labels[1]] = proj_bounds[:, 1]
        clean = self.projected.drop(
            columns=labels + [self.target.lr.geom_col, self.target.lr.geom_m_col], errors='ignore')
        select = select \
            .merge(clean, how='left', left_on='__projector__', right_index=True) \
            .drop(columns=['__projector__', '__target__', '__proj_lines__', self.target.lr.geom_m_col],
                  errors='ignore')
        return select


def generate_intersection_pairs(
    gdf: gpd.GeoDataFrame,
    exclude_groups: str | list[str] | None = None,
    touches: bool = True,
    crosses: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find pairwise intersection geometries between linestring geometries within 
    a GeoDataFrame. Uses spatial predicates to control which types of 
    intersections are returned.

    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame containing linestring geometries to find 
        intersections within.
    exclude_groups : str or list of str, optional
        Column name(s) used for group exclusion. Pairs where both geometries
        share the same value in these columns are excluded from results. 
        Useful for excluding intersections between segments of the same route.
    touches : bool, default True
        If True, include pairs that share boundary points (endpoints) only.
    crosses : bool, default True
        If True, include pairs whose interiors intersect (interior crossings).

    Returns
    -------
    tuple of (ndarray, ndarray, ndarray)
        A tuple of three arrays:

        - intersections : Array of intersection geometries (shapely objects).
        - index_left : Array of index labels of the first geometry in each 
          pair.
        - index_right : Array of index labels of the second geometry in each 
          pair.

    Raises
    ------
    ValueError
        If both `touches` and `crosses` are False.
    """
    # Validate inputs
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame.")
    if not touches and not crosses:
        raise ValueError("At least one of 'touches' or 'crosses' must be True.")
    
    # Normalize exclude_groups to list or None
    exclude_groups = label_list_or_none(exclude_groups)
    if exclude_groups is not None:
        missing = [c for c in exclude_groups if c not in gdf.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} specified in exclude_groups are not "
                f"found in the GeoDataFrame."
            )

    # Extract geometry array and index
    geoms = gdf.geometry.values
    idx = gdf.index

    # Empty result
    empty = (np.array([], dtype=object), np.array([]), np.array([]))
    if len(geoms) < 2:
        return empty

    # Build spatial index and query candidate pairs
    tree = shapely.STRtree(geoms)
    left_idx, right_idx = tree.query(geoms, predicate='intersects')

    # Enforce i < j to deduplicate pairs and remove self-intersections
    mask = left_idx < right_idx
    left_idx, right_idx = left_idx[mask], right_idx[mask]

    if len(left_idx) == 0:
        return empty

    # Refine to only enabled predicates
    checks = []
    if touches:
        checks.append(shapely.touches(geoms[left_idx], geoms[right_idx]))
    if crosses:
        checks.append(shapely.crosses(geoms[left_idx], geoms[right_idx]))
    keep = np.any(checks, axis=0)
    left_idx, right_idx = left_idx[keep], right_idx[keep]

    if len(left_idx) == 0:
        return empty

    # Filter out same-group pairs
    if exclude_groups is not None:
        group_values = gdf[exclude_groups].values
        left_groups = group_values[left_idx]
        right_groups = group_values[right_idx]
        if left_groups.ndim == 1:
            different_group = left_groups != right_groups
        else:
            different_group = np.any(left_groups != right_groups, axis=1)
        left_idx = left_idx[different_group]
        right_idx = right_idx[different_group]

    if len(left_idx) == 0:
        return empty

    # Compute intersection geometries
    intersections = shapely.intersection(geoms[left_idx], geoms[right_idx])
    index_left = idx[left_idx]
    index_right = idx[right_idx]
    return intersections, np.asarray(index_left), np.asarray(index_right)


def generate_intersection_nodes(
    gdf: gpd.GeoDataFrame,
    exclude_groups: str | list[str] | None = None,
    touches: bool = True,
    crosses: bool = True,
) -> tuple[np.ndarray, list[list]]:
    """
    Find unique intersection nodes between geometries in a GeoDataFrame,
    returning one entry per unique intersection location with the list of all
    participating source geometry indices.

    Calls :func:`generate_intersection_pairs` to obtain pairwise results,
    then explodes multipart geometries and groups by unique location.

    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame containing geometries to find intersections within.
    exclude_groups : str or list of str, optional
        Column name(s) used for group exclusion. Pairs where both geometries
        share the same value in these columns are excluded from results.
        Useful for excluding intersections between segments of the same route.
    touches : bool, default True
        If True, include pairs that share boundary points (endpoints) only.
    crosses : bool, default True
        If True, include pairs whose interiors intersect (interior crossings).

    Returns
    -------
    tuple of (ndarray, list of list)
        A tuple of two elements:

        - geometries : Array of unique intersection geometries.
        - indices : List of lists, where each inner list contains the sorted
          source geometry index labels participating at that location.
    """
    # Get pairwise intersection arrays
    intersections, index_left, index_right = generate_intersection_pairs(
        gdf, exclude_groups=exclude_groups, touches=touches, crosses=crosses
    )

    # Empty result
    empty = (np.array([], dtype=object), [])
    if len(intersections) == 0:
        return empty

    # Explode multipart geometries to individual parts
    parts, part_pair_idx = shapely.get_parts(intersections, return_index=True)

    if len(parts) == 0:
        return empty

    # Map each part back to the source index labels
    part_left = index_left[part_pair_idx]
    part_right = index_right[part_pair_idx]

    # Group by unique geometry using WKB representation
    wkb = shapely.to_wkb(parts)
    unique_wkb, inverse = np.unique(wkb, return_inverse=True)

    # Collect all participating source indices per unique geometry
    n_unique = len(unique_wkb)
    indices_sets = [set() for _ in range(n_unique)]
    for i, group_id in enumerate(inverse):
        indices_sets[group_id].add(part_left[i])
        indices_sets[group_id].add(part_right[i])

    # Build result arrays
    unique_geoms = shapely.from_wkb(unique_wkb)
    indices = [sorted(s) for s in indices_sets]
    return unique_geoms, indices
