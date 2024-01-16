"""
===============================================================================

Module featuring classes and functionality for spatial analysis of events data 
including parallel projection.


Classes
-------
ParallelProjector


Dependencies
------------
pandas, numpy, copy, warnings, functools


Development
-----------
Developed by:
Tariq Shihadah, tariq.shihadah@gmail.com

Created:
3/3/2022

Modified:
3/3/2022

===============================================================================
"""


################
# DEPENDENCIES #
################

import pandas as pd
import geopandas as gpd
import numpy as np
import copy, warnings
from functools import wraps
from linref.various.geospatial import join_nearby


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

    def __init__(self, target, projected, samples=3, buffer=100) -> None:
        self.target = target
        self.projected = projected
        self.samples = samples
        self.buffer = buffer

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, obj):
        # Validate object type
        if not isinstance(obj, EventsCollection):
            raise TypeError(
                "Input target must be valid lr.EventsCollection instance.")
        # Ensure that geometries and routes are available
        if obj.geom is None:
            raise ValueError(
                "No geometry found in the target events dataframe.")
        elif obj.route is None:
            raise ValueError(
                "No routes found in events dataframe.")
        self._target = obj

    @property
    def projected(self):
        return self._projected

    @projected.setter
    def projected(self, obj):
        # Validate object type
        if not isinstance(obj, gpd.GeoDataFrame):
            raise TypeError(
                "Input projected object must be valid gpd.GeoDataFrame "
                "instance.")
        self._projected = obj
    
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
            '__projector': np.repeat(self.projected.index.values, self.samples),
            'geometry': points}, geometry='geometry', crs=self.projected.crs
        )

    def _buffer_join(self):
        """
        Join projector sample points to target geometry within buffer for 
        matching.
        """
        # Join sampling points with target data
        joined = join_nearby(
            self.target.df, self.sample_points, buffer=self.buffer, choose='all'
        )
        # Process, clean results
        joined.index.rename('__target', inplace=True)
        joined = joined \
            .reset_index(drop=False)[['__projector','__target','DISTANCE']]
        joined = joined \
            .sort_values(by=['__projector','__target'], ascending=True)
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
        labels = [self.target.beg, self.target.end]

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
            '__projector': projectors, '__target': targets})
        
        # Merge matched records
        proj_lines = self.projected.geometry.rename('__proj_lines')
        select = matched_pairs \
            .merge(proj_lines, left_on='__projector', right_index=True)
        target_data = self.target.df[self.target.keys + [self.target.route]]
        select = select \
            .merge(target_data, left_on='__target', right_index=True)
        select = select.reset_index(drop=True)

        # Project ends onto matched routes
        def _project(route, line):
            try:
                # Project bounds onto target route
                boundary = line.boundary
                beg, end = boundary.geoms[0], boundary.geoms[-1]
                beg_loc, end_loc = route.project(beg), route.project(end)
                return beg_loc, end_loc
            except (AttributeError, IndexError):
                return np.nan, np.nan
        proj_bounds = np.asarray(list(map(
            _project,
            select[self.target.route].values,
            select['__proj_lines'].values
        )))
            
        # Merge with input and target data
        if sort_locs:
            select[labels[0]] = proj_bounds.min(axis=1)
            select[labels[1]] = proj_bounds.max(axis=1)
        else:
            select[labels[0]] = proj_bounds[:, 0]
            select[labels[1]] = proj_bounds[:, 1]
        clean = self.projected.drop(
            columns=labels + ['geometry','route'], errors='ignore')
        select = select \
            .merge(clean, how='left', left_on='__projector', right_index=True) \
            .drop(columns=['__projector','__target','__proj_lines','route'],
                  errors='ignore')
        return select
    

#####################
# LATE DEPENDENCIES #
#####################

from linref.events.collection import EventsCollection
