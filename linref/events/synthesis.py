"""
===============================================================================

Module featuring classes and functionality for synthesizing linear referencing 
information for existing data that is not LRS-enabled and other manipulations 
which support linear referencing data engineering and analysis.


Classes
-------
None


Dependencies
------------
geopandas, shapely, pandas, numpy, rangel, copy, warnings, functools


Development
-----------
Developed by:
Tariq Shihadah, tariq.shihadah@gmail.com

Created:
4/1/2023

Modified:
5/19/2023

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
from rangel import RangeCollection
from shapely.geometry import Point
from shapely import unary_union


#######################
# SYNTHESIS FUNCTIONS #
#######################

def generate_linear_events(
    df, 
    keys=None, 
    beg_label='BEG', 
    end_label='END', 
    chain_label='CHAIN', 
    buffer=None, 
    scale=1,
    decimals=None, 
    breaks='continue', 
    **kwargs
):
    """
    Function for generating events information for existing chains of linear 
    geospatial data based on the geographic lengths of chain members. This 
    function is intended to synthesize a new linear referencing system for 
    chains of adjacent linear data with matching key values, which are 
    oriented in the same direction, and whose end and begin points are 
    coincident or fall within a defined buffer distance of one another.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        Valid geopandas GeoDataFrame with linear geometries.
    keys : list or tuple, optional
        A list or tuple of dataframe column labels which define the unique 
        groups of events within the events dataframe. Common examples include 
        year or route ID columns which distinguish unrelated sets of events 
        within the events dataframe. If not provided, it will be assumed that 
        the whole dataframe represents a single group of events. As a result, 
        the only key included in the resulting EventsCollection will be the
        provided `chain_label`.
    beg_label, end_label : str or label
        Column labels to be created within the events dataframe which 
        represent the linearly referenced location of each event.
    chain_label : str or label
        Column label to be created within the events dataframe which indicates 
        the index of the linear chain each given event is associated with.
    buffer : scalar, optional
        A spatial buffer to use when determining if end and begin points 
        are coincident. If not provided, only points which are exactly 
        identical will be determined to be coincident.
    scale : scalar, default 1
        A value to multiply all linearly referenced location values by to 
        transform geographic distance values to a preferred scale. For 
        example, using a value of 5280 to convert from miles to feet.
    decimals : scalar, optional
        If provided, linearly referenced location values will be rounded to 
        the provided number of decimals.
    breaks : {'extend'}
        How to address breaks in linear chains which fall within the same 
        group based on provided keys.

        Options
        -------
        continue : Continue the event bound definitions as if the chains 
            were continuous.
        TBD
    **kwargs
        Keyword arguments to be passed to the EventsCollection constructor 
        before returning.
    """
    # Validate parameters
    if not isinstance(df, gpd.GeoDataFrame):
        raise ValueError(
            "Input dataframe must be gpd.GeoDataFrame instance.")
    if not decimals is None:
        try:
            assert isinstance(decimals, int)
            assert decimals >= 0
        except:
            raise ValueError(
                "If provided, input decimals parameter must be an integer "
                "that is greater than or equal to zero.")
    if not buffer is None:
        try:
            buffer = float(buffer)
            assert buffer > 0
        except:
            raise TypeError(
                "If provided, buffer must be a positive, non-zero "
                "numeric value.")
    if not keys is None:
        try:
            assert len(set(keys) - set(list(df))) == 0
        except:
            raise ValueError(
                "Input keys must all be present within the target dataframe.")
    if beg_label in list(df):
        raise ValueError(
            f"Begin location label `{beg_label}` is already present in the "
            "input dataframe.")
    if end_label in list(df):
        raise ValueError(
            f"End location label `{end_label}` is already present in the "
            "input dataframe.")
    if chain_label in list(df):
        raise ValueError(
            f"Chain index label `{chain_label}` is already present in the "
            "input dataframe.")

    # Group the dataframe
    if not keys is None:
        groups = df.groupby(by=keys)
    else:
        groups = {np.nan: df}.items()

    # Define function for retrieving boundary points
    def _get_boundary_point(line, index=None):
        # Use coords approach for multi-part geometries
        try:
            return Point(line.geoms[index].coords[index])
        # Use coords approach for single-part geometries
        except AttributeError:
            return Point(line.coords[index])

    # Iterate over all groups and perform analysis
    geom_column = df.geometry.name
    record_indexes = []
    record_begs = []
    record_ends = []
    record_chains = []
    for key, group in groups:

        # Get boundaries points of all lines for finding matches
        begs = group[geom_column].apply(_get_boundary_point, index=0)
        ends = group[geom_column].apply(_get_boundary_point, index=-1)
        # Buffer the end points if requested
        if not buffer is None:
            begs = begs.apply(lambda x: x.buffer(buffer))
            ends = ends.apply(lambda x: x.buffer(buffer))
        # Create geodataframes for intersecting
        begs = gpd.GeoDataFrame(begs, geometry=geom_column)
        ends = gpd.GeoDataFrame(ends, geometry=geom_column)
        
        # Intersect boundary geometries
        intersection = gpd.sjoin(ends, begs)
        intersection['index_left'] = intersection.index
        # Get all unique matches between the line end points and other line 
        # begin points
        pairs = intersection[['index_left','index_right']].values
        
        # Remove instances of multiple matches on left or right; only the 
        # first unique value on the left and right are kept based on original 
        # sorting of data
        pairs = pairs[np.unique(pairs[:,0], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:,1], return_index=True)[1]]
        
        # Identify single-segment chains with no matches to other segments as 
        # well as downstream terminals, holding their places with pairs of 
        # these segments' indices matched with null values
        missing = set(group.index) - set(pairs[:,0])
        missing = np.array([sorted(missing), np.full(len(missing), np.nan)]).T
        pairs = np.concatenate([pairs, missing])

        # Iterate through chaining process until all pairs are eliminated
        chains = []
        chain_index = 0
        while pairs.shape[0] > 0:
            # Initialize pairs filter, retaining all pairs
            pairs_filter = np.ones(pairs.shape[0], dtype=bool)
            
            # Identify upstream terminals which will be used to begin segment 
            # chains
            terminals = sorted(set(pairs[:,0]) - set(pairs[:,1]))

            # Address complete loops by selecting a terminal from remaining 
            # data if none remain
            try:
                assert len(terminals) > 0
            except AssertionError:
                terminals = [pairs[0,0]]

            # Iterate over upstream terminals, chaining from them to their 
            # downstream matches
            for terminal in terminals:
        
                # Iterate through all subsequent matches to the terminal
                chains.append([])
                member = terminal
                while True:
                    try:
                        # Update the chain by adding the current member
                        chains[chain_index].append(member)
                        # Identify pairs matching the selected chain member
                        member_loc = (pairs[:,0] == member) & pairs_filter
                        # Check for no matches; if there are none, end the 
                        # chain and move to the next one
                        assert member_loc.sum() > 0
                        # Update pairs filter to remove matches to the member
                        pairs_filter[member_loc] = False
                        # Update indexed chain member by selecting the next 
                        # match
                        member = pairs[member_loc.argmax()][1]
                        # Check for the end of the chain if the new selected 
                        # member is null, or if looping is detected; end the 
                        # chain in either case
                        assert ~np.isnan(member) # Check for end of chain
                        assert member != terminal # Check for looping
                    except AssertionError:
                        break
        
                # Update the index of the chain
                chain_index += 1
        
            # Filter pairs to remove addressed items
            pairs = pairs[pairs_filter]

        # Get lengths of all geometries
        lengths_all = group.length
        # Iterate over chains and create events information
        for chain_index, chain in enumerate(chains):
            # Compute cumulative sum of chain geometry lengths
            lengths = lengths_all.loc[chain].cumsum().to_list()
            # Generate LRS data
            record_indexes.extend(chain)
            record_begs.extend([0] + lengths[:-1])
            record_ends.extend(lengths)
            record_chains.extend([chain_index] * len(chain))

    # Scale location data
    record_begs = np.array(record_begs) * scale
    record_ends = np.array(record_ends) * scale

    # Round location data if requested
    if not decimals is None:
        record_begs = np.round(record_begs, decimals=decimals)
        record_ends = np.round(record_ends, decimals=decimals)
        
    # Synthesize events data
    events = pd.DataFrame(data={
        beg_label: record_begs,
        end_label: record_ends,
        chain_label: record_chains,
    }, index=record_indexes)
    events = df.merge(events, left_index=True, right_index=True)

    # Merge with parent table and create EventsCollection
    ec = EventsCollection(
        events, 
        keys=keys + [chain_label] if not keys is None else [chain_label], 
        beg=beg_label, 
        end=end_label, 
        geom=geom_column, 
        **kwargs
    )
    return ec


def find_intersections(df):
    """
    Generate intersection points for an input geodataframe of linear 
    geometries. Output will be a geodataframe with a single point 
    geometry at each location where a line intersects with another 
    and will have the same CRS.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        Valid geopandas geodataframe containing linear geometries.
    """
    # Validate parameters
    if not isinstance(df, gpd.GeoDataFrame):
        raise ValueError(
            "Input data must be of gpd.GeoDataFrame type.")
        
    # Iterate through slices of the dataframe
    geoms = df.geometry.values
    records = []
    for i in range(geoms.shape[0] - 1):
        # Select the row and unary union of lower remainder
        target = geoms[i]
        sub = unary_union(geoms[i+1:])
        # Find intersection points
        intersection = sub.intersection(target)
        # Check if none found
        if intersection.is_empty:
            continue
        # Add to records
        try:
            records.extend(list(intersection.geoms))
        except:
            records.append(intersection)

    # Cast to dataframe
    res = gpd.GeoDataFrame(geometry=records, crs=df.crs)
    return res


#####################
# LATE DEPENDENCIES #
#####################

from linref.events.collection import EventsCollection
