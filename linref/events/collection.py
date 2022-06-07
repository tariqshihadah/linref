"""
===============================================================================

Module featuring EventsCollection and EventsGroup object classes for the 
management of linear referencing events data and optimized performance of 
various events operations including dissolves, automated intersections 
and attribute retrievals, linear overlays, and more.

EventsCollection class instances represent complex events data sets with 
multiple groups of events which are distinguished by at least one set of keys 
(e.g., years of data or inventory categories). These collections can be used 
for a variety of linear referencing operations and events manipulations, such 
as dissolves based on a subset of events columns, returning a simplified 
data set with a selection of columns aggregated. Additionally, these 
collections can be used to perform automated intersections with another 
EventsCollection class instance using the retrieve() method, retrieving 
column data from another collection and relating it to the original 
collection's events data.

EventsGroup class instances represent simple events data sets with a single 
group of contiguous events. These groups can be used for a variety of linear 
referencing operations such as overlays to determine portions of events 
overlapped by an input range, intersections to determine which events intersect 
with an input range, length-weighted averages of event column values based on 
an input range, and more.

EventsCollection class instances can be queried using square bracket indexing 
or the get_subset() and get_group() methods, returning a pared down 
EventsCollection or a specific EventsGroup, respectively. Similarly, this can 
be done using object indexing, passing a mixture of unique values and valid 
slices of unique key values to return a subset of the collection as an 
EventsCollection instance, or just unique key values to return a unique group 
as an EventsGroup instance.


Classes
-------
EventsCollection, EventsGroup


Dependencies
------------
pandas, geopandas, numpy, shapely, copy, warnings, rangel


Examples
--------
Create an events collection for a sample roadway events dataframe with unique  
route identifier represented by the 'Route' column and data for multiple years, 
represented by the 'Year' column. The begin and end mile points are defined by 
the 'Begin' and 'End' columns.
>>> ec = EventsCollection(df, keys=['Route','Year'], beg='Begin', end='End')

To select events from a specific route and a specific year, indexing for all 
keys can be used, producing an EventsGroup.
>>> eg = ec['Route 50', 2018]

To select events on all routes but only those from a specific year, indexing 
for only some keys can be used.
>>> ec_2018 = ec[:, 2018]

To get all events which intersect with a numeric range, the intersecting() 
method can be used on an EventsGroup instance.
>>> df_intersecting = eg.intersecting(0.5, 1.5, closed='left_mod')

The intersecting() method can also be used for point locations by ommitting the 
second location attribute.
>>> df_intersecting = eg.intersecting(0.75, closed='both')

The linearly weighted average of one or more attributes can be obtained using 
the overlay_average() method.
>>> df_overlay = eg.overlay_average(0.5, 1.5, cols=['Speed_Limit','Volume'])

If the events include information on the roadway speed limit and number of 
lanes, they can be dissolved on these attributes. During the dissolve, other 
attributes can be aggregated, providing a list of associated values or 
performing an aggregation function over these values.
>>> ec_dissolved = ec.dissolve(attr=['Speed_Limit','Lanes'], aggs=['County'])


Development
-----------
Developed by:
Tariq Shihadah, tariq.shihadah@gmail.com

Created:
10/22/2019

Modified:
3/3/2021

===============================================================================
"""


################
# DEPENDENCIES #
################

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge
from linref.route import MLSRoute, combine_mpgs
from rangel import RangeCollection
import copy, warnings

# Temporary dependencies
from linref.various.geospatial import join_nearby


##################
# EVENTS CLASSES #
##################

class EventsFrame(object):
    """
    High-level class for managing linear events data. Users should instead use 
    the EventsCollection class for complex data sets with multiple groups of 
    events, grouped by at least one key column (e.g., route ID), or the 
    EventsGroup class for simple data sets with only a single group of events.
    """

    # Default standard column values
    default_keys = ['RID', 'YEAR', 'KEY']
    default_beg =  ['BMP', 'BEG', 'FROM']
    default_end =  ['EMP', 'END', 'TO']
    default_geom = ['geometry']

    def __init__(
        self, df, keys=None, beg=None, end=None, geom=None, route=None, 
        closed='left_mod', sort=False, **kwargs):
        # Log input values
        super(EventsFrame, self).__init__()
        self._df = df
        self.keys = keys
        self.beg = beg
        self.end = end
        self.geom = geom
        self.route = route
        self.closed = closed
        self._sort = sort
        self.df = df

    def __repr__(self):
        # Define representation components
        nm = f"{self.__class__.__name__}"
        kwargs = ['df']
        if self.num_keys > 0:
            kwargs.append(f"""keys=['{"','".join(self.keys)}']""")
        kwargs.append(f"beg='{self.beg}'")
        kwargs.append(f"end='{self.end}'")
        if not self.geom is None:
            kwargs.append(f"geom='{self.geom}'")
        kwargs.append(f"closed='{self.closed}'")
        # Return text
        text = f"{nm}({', '.join(kwargs)})"
        return text

    @property
    def df(self):
        """
        The collection's events dataframe.
        """
        return self._df

    @df.setter
    def df(self, df):
        # Validate input
        if isinstance(df, pd.DataFrame) or isinstance(df, gpd.GeoDataFrame):
            # Sort dataframe
            self._df = df
            if self._sort:
                self._df = self._sort_df(df)

            # Define the key groups
            if self.num_keys > 0:
                self._groups = self._df.groupby(by=self.keys)
            else:
                self._groups = None

            # Retrieve default geometry column if df is geodataframe
            if isinstance(df, gpd.GeoDataFrame) and self.geom is None:
                self.geom = df.geometry.name
        else:
            raise TypeError("Input dataframe must be pandas DataFrame class "
                "instance.")

    def _sort_df(self, df):
        """
        Sort the given dataframe by the collection's keys and begin/end 
        columns, returning the sorted dataframe.
        """
        return df.sort_values(
            by=self.keys + [self.beg, self.end], ascending=True)

    def df_exportable(self):
        """
        Return a dataframe which is optimized for exporting.
        """
        # Create a copy of the events dataframe
        df = self.df.copy()
        # Convert route data to wkt
        try:
            df[self.route] = \
                df[self.route].apply(lambda x: x.wkt)
        except:
            # Remove route column
            df = df.drop(columns=[self.route], errors='ignore')
        return df

    @property
    def keys(self):
        """
        The list of column names within the events dataframe which are queried 
        to define specific events groups (e.g., events on a specific route).
        """
        return self._keys

    @property
    def key_locs(self):
        return self._key_locs
    
    @keys.setter
    def keys(self, keys):
        # Address null input
        if keys is None:
            keys = []
        # Validate input type
        elif isinstance(keys, str):
            # If string, assume single column reference
            keys = [keys]
        else:
            try:
                # Validate list-like
                keys = list(keys)
            except TypeError:
                raise TypeError("Input key column name(s) must be a string or "
                    "list-like of strings which refer to valid columns within "
                    "the collection's events dataframe.")
        # Validate presence within events dataframe
        for key in keys:
            if not key in self.df.columns:
                raise ValueError(f"Key column value '{key}' is not present "
                    "within the collection's events dataframe.")
        # Log validated keys
        self._keys = keys
        self._key_locs = [self.columns.index(key) for key in keys]

    @property
    def num_keys(self):
        """
        The number of key columns within self.keys.
        """
        return len(self.keys)
    
    @property
    def key_values(self):
        """
        A dictionary of valid values for each key column.
        """
        # Identify all unique values for each key
        values = {col:self.df[col].unique() for col in self.keys}
        return values

    @property
    def columns(self):
        """
        A list of all columns within the events dataframe.
        """
        return list(self._df.columns)

    @property
    def targets(self):
        """
        A list of begin, end, and key columns within the events dataframe.
        """
        # Define target columns
        targets = [self.beg, self.end] + self.keys
        return targets        

    @property
    def others(self):
        """
        A list of columns within the events dataframe which are not the begin, 
        end, or key columns.
        """
        # Define other columns
        others = [col for col in self.df.columns if not col in self.targets]
        return others

    @property
    def groups(self):
        """
        The pandas GroupBy of the events dataframe, grouped by the collection's 
        key columns. This defines the basis for key queries.
        """
        return self._groups

    @property
    def group_keys(self):
        return list(map(tuple, self.df.values[:, self.key_locs]))
    
    @property
    def group_keys_unique(self):
        return list(set(map(tuple, self.df.values[:, self.key_locs])))
    
    @property
    def beg(self):
        return self._beg
    
    @property
    def beg_loc(self):
        return self._beg_loc

    @property
    def begs(self):
        return self.df.values[:, self.beg_loc]
    
    @beg.setter
    def beg(self, beg):
        # Address null input
        if beg is None:
            raise ValueError("Begin location column cannot be None.")
        # Validate presence within events dataframe
        elif not beg in self.df.columns:
            raise ValueError(f"Begin location column name '{beg}' is not "
                "present within the collection's events dataframe.")
        # Log validated keys
        self._beg = beg
        self._beg_loc = self.columns.index(beg)

    @property
    def end(self):
        return self._end
    
    @property
    def end_loc(self):
        return self._end_loc
    
    @property
    def ends(self):
        return self.df.values[:, self.end_loc]
    
    @end.setter
    def end(self, end):
        # Address null input
        if end is None:
            end = self.beg
        # Validate presence within events dataframe
        elif not end in self.df.columns:
            raise ValueError(f"End location column name '{end}' is not "
                "present within the collection's events dataframe.")
        # Log validated keys
        self._end = end
        self._end_loc = self.columns.index(end)

    @property
    def geom(self):
        return self._geom
    
    @property
    def geom_loc(self):
        return self._geom_loc
    
    @geom.setter
    def geom(self, geom):
        # Address null input
        if geom is None:
            pass
        # Validate presence within events dataframe
        elif not geom in self.df.columns:
            raise ValueError(f"Geometry column name '{geom}' is not "
                "present within the collection's events dataframe.")
        # Log validated keys
        self._geom = geom
        self._geom_loc = self.columns.index(geom) if not geom is None else None

    @property
    def route(self):
        return self._route
    
    @property
    def route_loc(self):
        return self._route_loc
    
    @route.setter
    def route(self, route):
        # Address null input
        if route is None:
            pass
        # Validate presence within events dataframe
        elif not route in self.df.columns:
            raise ValueError(f"Geometry column name '{route}' is not "
                "present within the collection's events dataframe.")
        # Log validated keys
        self._route = route
        self._route_loc = self.columns.index(route) \
                if not route is None else None

    def parse_routes(self, col=None, inplace=False, errors='raise'):
        """
        Parse MLSRoutes data in the provided column, which contains either 
        MLSRoute objects, WKT data for MULTILINESTRINGs or LINESTRINGs with 
        M-values, or a mixture of both.

        Parameters
        ----------
        col : label, optional
            A valid column label within the events dataframe which contains the 
            target MLSRoute data. If not provided, will attempt to retrieve a 
            previously assigned column label from the self.route property.
        inplace : boolean, default False
            Whether to perform the operation in place. If False, will return a 
            modified copy of the events object.
        errors : {'raise','ignore'}
            How to address errors which arise when coercing MLSRoute data 
            during processing. If ignored, errors will result in null values 
            in the events dataframe where errors occurred.
        """
        # Check column
        if col is None:
            try:
                col = self._route
            except:
                raise ValueError("No route column label provided.")
        # Coerce data
        def _to_routes(x):
            if isinstance(x, MLSRoute):
                return x
            elif isinstance(x, str):
                try:
                    return MLSRoute.from_wkt(x)
                except Exception as e:
                    if errors=='raise':
                        raise e
                    else:
                        return
            else:
                if errors=='raise':
                    raise TypeError(
                        "Route data must be MLSRoute object or WKT valid "
                        "string.")
                else:
                    return
        routes = self.df[col].apply(_to_routes)
        # Apply update
        if inplace:
            self.df[col] = routes
            self.route = col
            return
        else:
            ec = self.copy(deep=True)
            ec.df[col] = routes
            ec.route = col
            return ec

    @property
    def closed(self):
        """
        Collection parameter for whether event intervals are closed on the 
        left-side, right-side, both or neither.
        """
        return self._closed

    @closed.setter
    def closed(self, closed):
        self.set_closed(closed, inplace=True)
    
    @property
    def shape(self):
        return self.df.shape
    
    def _validate_cols(self, cols=None, require=False):
        """
        Process input columns as list, string, or None, converting to list.
        """
        # Validate column inputs and coerce list type
        if cols is None:
            if require:
                raise ValueError("Must provide at least one column label.")
            else:
                cols = []
        elif isinstance(cols, tuple) or isinstance(cols, list):
            cols = list(cols)
        else:
            cols = [cols]
        # Check presence in dataframe
        try:
            # Check for presence in events dataframe
            for col in cols:
                assert col in self.df.columns
        except ValueError:
            raise ValueError("Provided column labels must exist within the " 
                "events dataframe.")
        except AssertionError:
            raise ValueError(f"Column '{col}' does not exist within the "
                "events dataframe.")
        # Return validated columns
        return cols

    def build_routes(self, label='route', errors='raise'):
        """
        Build MLSRoute instances for each event based on available geometry 
        and begin and end locations.

        Parameters
        ----------
        label : valid pandas column label
            Column label to use for newly generated column populated with 
            routes data.
        errors : {'raise','ignore'}
            How to address errors if they arise when producing routes. If 
            errors are not raised, inviable records in the new column will 
            be filled with np.nan.
        """
        # Validate
        if self.geom is None:
            raise ValueError("No geometry column label defined.")
        # Build routes
        locs = (self.beg_loc, self.end_loc, self.geom_loc)
        routes = []
        for beg, end, geom in self.df.values[:, locs]:
            try:
                routes.append(MLSRoute.from_lines(geom, beg, end))
            except Exception as e:
                if errors=='ignore':
                    routes.append(np.nan)
                else:
                    raise e
        self.df[label] = routes
        self._route = label

    def copy(self, deep=False):
        """
        Create an exact copy of the events class instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
    
    def set_closed(self, closed, inplace=False):
        """
        Change whether ranges are closed on left, right, both, or neither side. 
        
        Parameters
        ----------
        closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
                'neither'}
            Whether intervals are closed on the left-side, right-side, both or 
            neither.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        if closed in RangeCollection._ops_closed:
            if inplace:
                self._closed = closed
            else:
                rc = self.copy()
                rc._closed = closed
                return rc
        else:
            raise ValueError(f"Closed parameter must be one of "
                f"{RangeCollection._ops_closed}.")

    def geometry_from_xy(self, x, y, col_name='geometry', crs=None, 
            inplace=False):
        """
        Use X and Y coordinates in the events dataframe to generate point 
        geometry.
        """
        # Validate columns
        x, y = self._validate_cols(cols=[x, y])
        # Generate geometry
        geometry = self.df.apply(lambda r: Point(r[x], r[y]), axis=1)
        # Apply geometry
        if inplace:
            ef = self
        else:
            ef = self.copy()
        ef.df[col_name] = geometry
        ef.df = gpd.GeoDataFrame(ef.df, geometry=col_name, crs=crs)
        ef.geom = col_name
        return None if inplace else ef
    
    def dissolve(self, attr=None, aggs=None, agg_func=None, agg_suffix='_agg', 
        agg_geometry=False, agg_routes=False, dropna=False, fillna=None, 
        reorder=True, merge_lines=True):
        """
        Dissolve the events dataframe on a selection of event attributes.

        Note: Data will be sorted by keys and begin/end columns prior to 
        performing the dissolve.
        
        Note: Missing data in selected attribute fields may cause problems with 
        dissolving; please use df.fillna(...) or df.dropna(...) to avoid this 
        problem.
        
        Parameters
        ----------
        attr : str or list
            Which event attribute(s) within the events dataframe to dissolve 
            on.
        aggs : str or list, default None
            Which event attribute(s) within the events dataframe to aggregate
            during the dissolve. Attributes will be aggregated into a list
            and returned under the same attribute name.
        agg_func : callable function or list of callable functions, default None
            A function or list of functions corresponding to the list of 
            aggregation attributes which will be called on the list-aggregated
            contents of those attributes.
        agg_suffix : str or list, default '_agg'
            A suffix to be added to the name of aggregated columns. If provided 
            as a list, must correspond to provided lost of aggregation 
            attributes.
        agg_geometry : bool, default False
            Whether to create an aggregated geometries field, populated with 
            aggregated shapely geometries based on those contained in the 
            collection's geometry field.
        agg_routes : bool, default False
            Whether to create an aggregated routes field, populated with 
            MLSRoute object class instances, created based on aggregated 
            segment geometries and begin and end mile posts.
        dropna : bool, default False
            Whether to drop records with empty values in the attribute fields. 
            This parameter is passed to the df.groupby call.
        fillna : optional
            A value or dictionary used to fill instances of np.nan in the 
            target dataframe. Consistent with the DataFrame.fillna() method.
        reorder : bool, default True
            Whether to reorder the resulting dataframe columns to match the 
            order of the collection's events dataframe.
        merge_lines : bool, default True
            Whether to use shapely's ops.linemerge function to combine 
            contiguous linestrings when aggregating linear geometries. Only 
            applicable when agg_geometry=True.
        """
        # Validate inputs
        # - Create, sort dummy dataframe
        df = self._sort_df(self.df.copy())
        col_order = list(df.columns)
        df['__DUMMY__'] = True # Dummy data guarantees >0 groupby keys
        # - Dissolve attributes
        attr = ['__DUMMY__'] + self.keys + self._validate_cols(attr)
        aggs = self._validate_cols(aggs)

        # - Aggregation functions
        if agg_func is None:
            agg_func = [None for i in aggs]
        elif callable(agg_func):
            agg_func = [agg_func for i in aggs]
        elif type(agg_func) is list:
            if not len(agg_func) == len(aggs):
                raise ValueError("Aggregation functions must be "
                    "provided as a single callable function or a list of "
                    "functions the same length as the list of aggregation "
                    "attributes.")
        else:
            raise ValueError("Aggregation functions must be provided as "
                "a single callable function or a list of functions the "
                "same length as the list of aggregation attributes.")
        
        # - Aggregation suffixes
        if agg_suffix is None:
            agg_suffix = ['' for i in aggs]
        elif type(agg_suffix) is str:
            agg_suffix = [agg_suffix for i in aggs]
        elif type(agg_suffix) is list:
            if not len(agg_suffix) == len(aggs):
                raise ValueError("Aggregation suffixes must be provided as a "
                    "single string or a list of strings the same length as "
                    "the list of aggregation attributes.")
        else:
            raise ValueError("Aggregation suffixes must be provided as a "
                "single string or a list of strings the same length as the "
                "list of aggregation attributes.")
        
        # Additional aggregation requests
        # - Prepare geometry dissolve if requested
        if agg_geometry:
            # Confirm valid geometry field
            if self.geom is None:
                raise ValueError("Collection does not include an identified "
                    "geometry field. This must be provided at initialization "
                    "of the collection or by setting it directly.")
            # Create geometry aggregation function
            if merge_lines:
                func = lambda x: linemerge(combine_mpgs(x, cls=MultiLineString))
            else:
                func = lambda x: combine_mpgs(x, cls=MultiLineString)
            # Append routes field name to aggregation list
            aggs.append(self.geom)
            agg_func.append(func)
            agg_suffix.append('')
        
        # - Prepare route dissolve if requested
        if agg_routes:
            # Confirm valid geometry field
            if self.geom is None:
                raise ValueError("Collection does not include an identified "
                    "geometry field. This must be provided at initialization "
                    "of the collection or by setting it directly.")
            # Create route information feed
            route_feed_col = 'route'
            build_feed = lambda r: \
                (r[self.geom], r[self.beg], r[self.end])
            df[route_feed_col] = df.apply(build_feed, axis=1)
            # Create route aggregation function
            func = lambda x: MLSRoute.from_lines(*list(zip(*x)))
            # Append routes field name to aggregation list
            aggs.append(route_feed_col)
            agg_func.append(func)
            agg_suffix.append('')

        # Prepare for dissolve
        # - Process selected columns for valid groupby
        select_cols  = [self.beg, self.end]
        select_cols += [x for x in attr if not x in select_cols]
        select_cols += [x for x in aggs if not x in select_cols]
        df = df[select_cols]
        df = df.fillna(fillna) if not fillna is None else df
        # - Group events data
        grouped = df.groupby(by=attr, dropna=dropna) \
                [[self.beg, self.end] + aggs].agg(list)
        beg_groups = grouped[self.beg].to_list()
        end_groups = grouped[self.end].to_list()
        agg_groups = grouped[aggs] if len(aggs) > 0 else None

        # Iterate through groups of data and define new dataframe records
        records = []
        for index, begs_i, ends_i in zip(grouped.index, beg_groups, end_groups):
            
            # Identify breaks between consecutive events
            rc = RangeCollection(begs=begs_i, ends=ends_i, centers=None,
                                 copy=False, sort=False)
            consecutive = rc.are_consecutive(all_=False, when_one=True)
            splitter    = (np.where(np.invert(consecutive))[0] + 1).tolist()
            
            # Get aggregation data
            lin_ranges = np.split(np.stack([begs_i, ends_i]), splitter, axis=1)
            if not agg_groups is None:
                try:
                    agg_data = agg_groups.loc[index, :].to_list()
                    agg_ranges = [[agg[i:j] for agg in agg_data] for i,j in \
                                zip([None]+splitter, splitter+[None])]
                except KeyError:
                    raise KeyError(
                        f"Unable to retrieve data group with index {index}. "
                        "This may be due to nan data in one or more of the "
                        "dissolving attributes.")
            else:
                agg_ranges = iter(list, 1)
            
            # Enforce grouped index as a list
            index = list(index) if len(attr) > 1 else [index]

            # Iterate over ranges and store data in records
            for lin_range, agg_range in zip(lin_ranges, agg_ranges):
                records.append([lin_range[0].min(), lin_range[1].max()] \
                            + index[1:] + agg_range) # Remove dummy column data

        # Create new dataframe with dissolved results
        aggs = [agg + suf for agg, suf in zip(aggs, agg_suffix)]
        res_cols = [self.beg, self.end] + attr[1:] + aggs # Remove dummy column
        res = pd.DataFrame.from_records(data=records, columns=res_cols)
        
        # Apply aggregation functions if requested
        if not agg_func is None:
            for col, func in zip(aggs, agg_func):
                if not func is None:
                    res.loc[:,col] = res.loc[:,col].apply(func)
        
        # Reorder columns and records
        if reorder:
            col_order = [c for c in col_order if c in res.columns] + \
                        [c for c in res.columns if not c in col_order]
            res = res[col_order]
        res = res.sort_values(by=self.keys+[self.beg,self.end], 
                              axis=0, ascending=True)

        # Convert to geodataframe if geometry is aggregated
        if agg_geometry:
            res = gpd.GeoDataFrame(res, geometry=self.geom, crs=self.df.crs)
        
        # Generate events collection
        ec = EventsCollection(res, keys=self.keys, beg=self.beg, end=self.end, 
            geom=self.geom if agg_geometry else None,
            route='route' if agg_routes else None, 
            closed=self.closed)
        return ec

    def project(self, other, buffer=100, nearest=True, loc_label='LOC', 
            dist_label='DISTANCE', **kwargs):
        """
        Project an input geodataframe onto the events dataframe, producing 
        linearly referenced point locations relative to events for all input 
        geometries within a buffered search area.

        Parameters
        ----------
        other : gpd.GeoDataFrame
            Geodataframe containing geometry which will be projected onto the 
            events dataframe.
        buffer : float, default 100
            The max distance to search for input geometries to project against 
            the events' geometries. Measured in terms of the geometries' 
            coordinate reference system.
        nearest : bool, default True
            Whether to choose only the nearest match within the defined buffer. 
            If False, all matches will be returned.
        loc_label, dist_label : label
            Labels to be used for created columns for projected locations on 
            target events groups and nearest point distances between target 
            geometries and events geometries.
        **kwargs
            Keyword arguments to be passed to the EventsFrame constructor 
            upon completion of the projection.
        """
        # Validate input geodataframe
        if not isinstance(other, gpd.GeoDataFrame):
            raise TypeError("Other object must be gpd.GeoDataFrame instance.")
        other = other.copy()

        # Check for invalid column names
        if self.route in other.columns:
            raise ValueError(f"Invalid column name '{self.route}' found in "
                "target geodataframe.")

        # Ensure that geometries and routes are available
        if self.geom is None:
            raise ValueError("No geometry found in events dataframe. If "
                "valid shapely geometries are available in the dataframe, "
                f"set this with the {self.__class__.__name__}'s geom "
                "property.")
        elif self.route is None:
            raise ValueError("No routes found in events dataframe. If valid "
                "shapely geometries are available in the dataframe, create "
                "routes by calling the build_routes() method on the "
                f"{self.__class__.__name__} class instance.")
        
        # Join the other geodataframe to this one
        select_cols = self.keys + [self.route, self.geom]
        try:
            if nearest:
                joined = other.sjoin_nearest(
                    self.df[select_cols],
                    max_distance=buffer,
                    how='left'
                )
            else:
                warnings.warn(
                    "Performance when nearest=False is currently limited and "
                    "will be improved in future versions.")
                joined = join_nearby(
                    other, 
                    self.df[select_cols], 
                    buffer=buffer, 
                    choose='all',
                    dist_label=dist_label
                )
        except AttributeError:
            # Optional dependency warning for improved performance
            warnings.warn(
                "Performance will be reduced for this operation when using "
                "the current geopandas version. Upgrade to geopandas>=v0.10.0 "
                "for improved performance.")
            joined = join_nearby(
                other, 
                self.df[select_cols], 
                buffer=buffer, 
                choose='min' if nearest else 'all',
                dist_label=dist_label
            )

        # Project input geometries onto event geometries
        def _project(r):
            try:
                return r[self.route].project(r.geometry)
            except AttributeError:
                return
        locs = joined.apply(_project, axis=1)
        joined[loc_label] = locs

        # Prepare and return data
        return self.__class__(
            joined.drop(columns=[self.route]),
            keys=self.keys,
            beg=loc_label,
            closed=self.closed,
            **kwargs
        )

    def project_old(self, other, buffer=100, choose='min', loc_label='LOC', 
            dist_label='DISTANCE', **kwargs):
        """
        Project an input geodataframe onto the events dataframe, producing 
        linearly referenced point locations relative to events for all input 
        geometries within a buffered search area.

        Parameters
        ----------
        other : gpd.GeoDataFrame
            Geodataframe containing geometry which will be projected onto the 
            events dataframe.
        buffer : float, default 100
            The max distance to search for input geometries to project against 
            the events' geometries. Measured in terms of the geometries' 
            coordinate reference system.
        choose : {'min', 'max', 'all'}, default 'min'
            Which target geometry to choose when more than one falls within the 
            buffer distance.

            Options
            -------
            min : choose the geometry with the shortest distance from the 
                events data
            max : choose the geometry with the longest distance from the 
                events data
            all : return all geometries which fall within the buffer area
        loc_label, dist_label : label
            Labels to be used for created columns for projected locations on 
            target events groups and nearest point distances between target 
            geometries and events geometries.
        **kwargs
            Keyword arguments to be passed to the EventsFrame constructor 
            upon completion of the projection.
        """
        # Validate input geodataframe
        if not isinstance(other, gpd.GeoDataFrame):
            raise TypeError("Other object must be gpd.GeoDataFrame instance.")
        other = other.copy()

        # Check for invalid column names
        if self.route in other.columns:
            raise ValueError(f"Invalid column name '{self.route}' found in "
                "target geodataframe.")

        # Ensure that geometries and routes are available
        if self.geom is None:
            raise ValueError("No geometry found in events dataframe. If "
                "valid shapely geometries are available in the dataframe, "
                f"set this with the {self.__class__.__name__}'s geom "
                "property.")
        elif self.route is None:
            raise ValueError("No routes found in events dataframe. If valid "
                "shapely geometries are available in the dataframe, create "
                "routes by calling the build_routes() method on the "
                f"{self.__class__.__name__} class instance.")
        
        # Join the other geodataframe to this one
        select_cols = self.keys + [self.route, self.geom]
        joined = join_nearby(
            other, 
            self.df[select_cols], 
            buffer=buffer, 
            choose=choose,
            dist_label=dist_label
        )

        # Project input geometries onto event geometries
        def _project(r):
            try:
                return r[self.route].project(r.geometry)
            except AttributeError:
                return
        locs = joined.apply(_project, axis=1)
        joined[loc_label] = locs

        # Prepare and return data
        return self.__class__(
            joined.drop(columns=[self.route]),
            keys=self.keys,
            beg=loc_label,
            closed=self.closed,
            **kwargs
        )

    def to_windows(self, dissolve=False, **kwargs):
        """
        Use the events dataframe to create sliding window events of a fixed 
        length and a fixed number of steps, and which fill the bounds of each 
        event.

        Parameters
        ----------
        length : numerical, default 1.0
            A fixed length for all windows being defined.
        steps : int, default 10
            A number of steps per window length. The resulting step length will 
            be equal to length / steps. For non-overlapped windows, use a steps 
            value of 1.
        fill : {'none','cut','left','right'}, default 'cut'
            How to fill a gap at the end of an event's range.

            Options
            -------
            none : no window will be generated to fill the gap at the end of 
                the input range.
            cut : a truncated window will be created to fill the gap with a 
                length less than the full window length.
            left : the final window will be anchored on the end of the event  
                and will extend the full window length to the left. 
            right : the final window will be anchored on the grid defined by 
                the step value, extending the full window length to the right, 
                beyond the event's end value.
            extend : the final window will be anchored on the grid defined by 
                the step value, extending beyond the window length to the right
                bound of the event.
        
        dissolve : bool, default False
            Whether to dissolve the events dataframe before performing the 
            transformation.
        """
        # Dissolve events
        if dissolve:
            events = self.dissolve().df
        else:
            events = self.df
        # Iterate over roads and create sliding window segments
        gen = zip(
            events[self.keys + [self.beg, self.end]].values,
            events.index.values
        )
        windows = []
        for (*keys, beg, end), index in gen:
            # Build sliding window ranges
            rng = RangeCollection.from_steps(beg, end, **kwargs).cut(beg, end)
            # Assemble sliding window data
            windows.append(
                np.concatenate(
                    [
                        [keys]*rng.num_ranges,   # Event keys
                        rng.rng.T,               # Window bounds
                        [[index]]*rng.num_ranges # Parent index value
                    ],
                    axis=1
                )
            )

        # Merge and prepare data, return
        windows = np.concatenate(windows, axis=0)
        df = pd.DataFrame(
            data=windows,
            columns=self.keys + [self.beg, self.end, 'index_parent'],
            index=None,
        )
        # Enforce data types
        dtypes = {
            **events.dtypes,
            'index_parent': events.index.dtype
        }
        dtypes = {col: dtypes[col] for col in df.columns}
        df = df.astype(dtypes, copy=False)
        res = self.__class__(df, keys=self.keys, beg=self.beg, end=self.end)
        return res


class EventsLog(object):
    """
    High-level class for logging and managing child EventsGroups created within 
    the context of a parent EventsCollection class instance.
    """

    def __init__(self, **kwargs):
        # Log input values
        super(EventsLog, self).__init__(**kwargs)
        self.reset()

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError as e:
            raise e

    def __setitem__(self, key, obj):
        self.log(key, obj, overwrite=True)

    @property
    def data(self):
        return self._data

    @property
    def keys(self):
        return list(self._data.keys())

    def reset(self):
        self._data = {}

    def log(self, key, obj, overwrite=True):
        """
        Store the input events class instance within the log's data under the 
        provided key.
        """
        if overwrite:
            self.data[key] = obj
        else:
            if key in self.data.keys():
                raise ValueError(f"Provided key '{key}' already exists within "
                    "the log.")
            else:
                self.data[key] = obj


class EventsGroup(EventsFrame):
    """
    User-level class for managing linear and points events data. This class is 
    used for simple data sets with only a single group of events. Data is 
    managed using both the pandas tabular data package as well as the ranges 
    range data package.

    EventsGroup class isntances can be used for a variety of linear referencing 
    operations such as overlays to determine portions of events overlapped by 
    an input range, intersections to determine which events intersect with an 
    input range, length-weighted averages of event column values based on an 
    input range, and more.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe which contains linear or point events data.
    beg, end : str or label
        Column labels within the events dataframe which represent the linearly 
        referenced location of each event. For linear events both are required,
        defining the begin and end location of each event. For point events, 
        only 'beg' is required, defining the exact location of each event (the 
        'end' property will automatically be set to be equal to the 'beg' 
        property).
    geom : str or label, optional
        Column label within the events dataframe which represents the shapely 
        geometry associated with each event if available. If provided, 
        certain additional class functionalities will be made available.
    closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
            'neither'}, default 'left_mod'
        Whether intervals are closed on the left-side, right-side, both or 
        neither.

        Options
        -------
        left : ranges are always closed on the left and never closed on the 
            right.
        left_mod : ranges are always closed on the left and only closed on the 
            right when the next range is not consecutive.
        right : ranges are always closed on the right and never closed on the 
            right.
        right_mod : ranges are always closed on the right and only closed on 
            the left when the previous range is not consecutive.
        both : ranges are always closed on both sides
        neither : ranges are never closed on either side
    """

    def __init__(self, df, beg=None, end=None, geom=None, closed='left_mod', 
        **kwargs):
        # Initialize EventsFrame superclass
        super(EventsGroup, self).__init__(
            df=df, keys=None, beg=beg, end=end, geom=geom, **kwargs)
        # Build data
        self._build_rng()
        # Log input values
        self.closed = closed

    def __getitem__(self, keys):
        """
        Select from the EventsGroup instance with numerical index values or a 
        boolean mask. Note: selection keys must be compatible with 
        np.ndarray[], rangel.RangeCollection[], and pd.DataFrame.loc[] 
        functionality.
        """
        # Retrieve selection and return new EventsGroup
        
    @property
    def rng(self):
        return self._rng
    
    @rng.setter
    def rng(self, rng):
        # Validate input
        if isinstance(rng, RangeCollection):
            self._rng = rng
        else:
            raise TypeError("Input rng must be RangeCollection class "
                "instance.")

    @property
    def lengths(self):
        """
        Lengths of all events ranges.
        """
        return self.rng.lengths
            
    @property
    def shape(self):
        return self.df.shape
    
    def _build_rng(self):
        # Build range collection
        rng = RangeCollection.from_array(self.df[[self.beg,self.end]].values.T,
                                         closed=self.closed, sort=False)
        self.rng = rng

    def set_closed(self, closed, inplace=False):
        """
        Change whether ranges are closed on left, right, both, or neither side. 
        
        Parameters
        ----------
        closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
                'neither'}, default 'left'
            Whether intervals are closed on the left-side, right-side, both or 
            neither.

            Options
            -------
            left : ranges are always closed on the left and never closed on the 
                right.
            left_mod : ranges are always closed on the left and only closed on 
                the right when the next range is not consecutive.
            right : ranges are always closed on the right and never closed on 
                the right.
            right_mod : ranges are always closed on the right and only closed 
                on the left when the previous range is not consecutive.
            both : ranges are always closed on both sides
            neither : ranges are never closed on either side
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        # Call super method
        super(EventsGroup, self).set_closed(closed=closed, inplace=inplace)
        try:
            self.rng.set_closed(closed=closed, inplace=inplace)
        except AttributeError:
            pass

    def intersecting(self, beg, end=None, closed=None, mask=False, **kwargs):
        """
        Retrieve a selection of records from the group of events based 
        on provided begin and end locations.

        Parameters
        ----------
        beg : float
            Beginning milepost of the overlaid segment.
        end : float, optional
            Ending milepost of the overlaid segment. If not provided, a point 
            overlay is assumed.
        closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
                'neither'}, default 'left'
            Whether intervals are closed on the left-side, right-side, both or 
            neither.

            Options
            -------
            left : ranges are always closed on the left and never closed on the 
                right.
            left_mod : ranges are always closed on the left and only closed on 
                the right when the next range is not consecutive.
            right : ranges are always closed on the right and never closed on 
                the right.
            right_mod : ranges are always closed on the right and only closed 
                on the left when the previous range is not consecutive.
            both : ranges are always closed on both sides
            neither : ranges are never closed on either side

        mask : bool, default False
            Whther to return a boolean mask for selecting from the events 
            dataframe instead of the selection from the dataframe itself.
        """
        # Intersect range
        mask_ = self.rng.intersecting(beg=beg, end=end, closed=closed, **kwargs)
        if mask:
            return mask_
        else:
            df = self.df.loc[mask_, :]
            return df

    def overlay(self, beg=None, end=None, arr=False, **kwargs):
        """
        Compute overlap of the input bounds with respect to the 
        events group.
        
        Parameters
        ----------
        beg : float
            Beginning milepost of the overlaid segment.
        end : float
            Ending milepost of the overlaid segment.
        normalize : boolean, default True
            Whether overlapping lengths should be normalized range length to 
            give a proportional result.
        how : {'right','left','sum'}, default 'right'
            How overlapping lengths should be normalized. Only applied when 
            normalize=True.

            right : Normalize overlaps by the length of each provided overlay 
                range.
            left : Normalize overlaps by the length of each of the collection's 
                ranges being overlaid.
            sum : Normalize overlaps by the sum of the lengths of all overlaps 
                for each provided overlay range. If there are gaps in the 
                collection's ranges or overlaps between the collection's 
                ranges, this will allow the sum of the overlaps to still equal 
                1.0, except where no overlaps occur.
        norm_zero : float, optional
            A number to substitute for instances where the normalizing factor 
            (denominator) is equal to zero, e.g., when the overlay range has a 
            length of zero and how='right'. If not provided, all instances of 
            zero division will return float value 0.0.
        arr : bool, default False
            Whther to return an array of resulting overlay values instead of 
            a pandas Series with an index matching the events dataframe.
        """
        # Compute range overlaps
        weights = self.rng.overlay(beg=beg, end=end, **kwargs)
        if arr:
            return weights
        else:
            sr = pd.Series(data=weights, index=self.df.index)
            return sr
    
    def overlay_average(self, beg=None, end=None, cols=None, weighted=True, 
                        zeroweight=None, how='right', weights=None, 
                        suffix='_average', **kwargs):
        """
        Compute the weighted average of a selection of events columns based on 
        the overlap of the input bounds with respect to linear events.
        
        Parameters
        ----------
        beg : float
            Beginning milepost of the overlaid segment.
        end : float
            Ending milepost of the overlaid segment.
        cols : list
            List of column labels to aggregate.
        weighted : boolean, default True
            Whether the computed average should be weighted. If False, an
            un-weighted average will be computed, giving all intersecting 
            values an equal weight.
        zeroweight : default None
            If weights sum to zero, how to compute average. If None, an
            un-weighted average will be computed. Else, no average will be 
            computed and the input value will be returned instead.
        how : {'right','left','sum'}, default 'right'
            How overlapping lengths should be normalized. Only applied when 
            normalize=True.

            Options
            -------
            right : Normalize overlaps by the length of each provided overlay 
                range.
            left : Normalize overlaps by the length of each of the collection's 
                event ranges.
            sum : Normalize overlaps by the sum of the lengths of all overlaps 
                for each provided overlay range. If there are gaps in the 
                collection's event ranges or overlaps between the collection's 
                ranges, this will allow the sum of the overlaps to still equal 
                1.0, except where no overlaps occur.

        weights : np.ndarray
            An array of length-normalized overlay weights; if excluded, 
            weights will be computed based on given mileposts and parameters; 
            if multiple overlay computations are being conducted, computing 
            the weights separately and then inputting them directly into the 
            aggregation functions will produce time savings.
        """
        # Validate inputs
        cols = self._validate_cols(cols=cols, require=True)
        
        # Compute weights
        if weights is None and weighted:
            weights = self.overlay(beg, end, normalize=True, 
                                   how=how, **kwargs).values
        elif weights is None and not weighted:
            weights = self.is_intersecting(beg, end, any_=False) * 1
                    
        # Aggregate selected columns
        res = []
        for col in cols:
            vals = self.df[col].values
            if len(vals) == 0:
                avg = np.nan
            # If weights are available, calculate weighted average
            elif len(weights) > 0 and weights.sum() > 0:
                avg = (vals * weights).sum()
            # If weights are not available, use substitute
            else:
                if zeroweight is None:
                    avg = vals.sum() / len(vals)
                else:
                    avg = zeroweight
            # Log computed averages
            res.append(avg)

        # Process results
        if len(cols) == 1:
            return res[0]
        else:
            return pd.Series(data=res, index=[str(col)+suffix for col in cols])
                             
    def overlay_sum(self, beg=None, end=None, cols=None, weighted=True, 
                    weights=None, suffix='_sum', **kwargs):
        """
        Compute the weighted average of a selection of events columns based on 
        the overlap of the input bounds with respect to route events.
        
        Parameters
        ----------
        beg : float
            Beginning milepost of the overlaid segment.
        end : float
            Ending milepost of the overlaid segment.
        cols : list
            List of column labels to aggregate.
        weighted : boolean, default True
            Whether the computed sum should be weighted. If False, an
            un-weighted sum will be computed, giving all intersecting values an 
            equal weight.
        weights : np.ndarray
            An array of length-normalized overlay weights; if excluded, 
            weights will be computed based on given mileposts and parameters; 
            if multiple overlay computations are being conducted, computing 
            the weights separately and then inputting them directly into the 
            aggregation functions will produce time savings.
        """
        # Validate inputs
        cols = self._validate_cols(cols=cols, require=True)
        
        # Compute weights
        if weights is None and weighted:
            weights = self.overlay(beg, end, normalize=False, **kwargs).values
            weights = np.divide(weights, self.lengths)
        elif weights is None and not weighted:
            weights = self.is_intersecting(beg, end) * 1
                    
        # Aggregate selected columns
        res = []
        for col in cols:
            vals = self._df[col].values
            if len(vals) == 0:
                sum_ = np.nan
            # If weights are available, calculate sum
            elif len(weights) > 0 and weights.sum() > 0:
                sum_ = (vals * weights).sum()
            # If weights are not available, assume zero
            else:
                sum_ = 0
            # Log computed sums
            res.append(sum_)
        
        # Process results
        if len(cols) == 1:
            return res[0]
        else:
            return pd.Series(data=res, index=[str(col)+suffix for col in cols])
    
    def overlay_most(self, beg=None, end=None, cols=None, weights=None,
                     suffix='_most', **kwargs):
        """
        Compute the most represented values of a selection of events columns 
        based on the overlap of the input bounds with respect to route events.
        
        Parameters
        ----------
        beg : float
            Beginning milepost of the overlaid segment.
        end : float
            Ending milepost of the overlaid segment.
        cols : list
            List of column labels to aggregate.
        weights : pd.Series
            A series of length-normalized overlay weights; if excluded, 
            weights will be computed based on given mileposts and parameters; 
            if multiple overlay computations are being conducted, computing 
            the weights separately and then inputting them directly into the 
            aggregation functions will produce time savings.
        """
        # Validate inputs
        cols = self._validate_cols(cols=cols, require=True)
        
        # Validate group shape
        if self.shape[0] == 0:
            if len(cols) == 1:
                return np.nan
            else:
                return pd.Series(data=np.nan, 
                    index=[str(col) + suffix for col in cols])

        # Compute weights
        if weights is None:
            weights = self.overlay(beg, end, normalize=True, how='right')
        
        # Aggregate selected columns
        res = []
        for col in cols:
            vals = self.df[col].values
            unique = np.unique(vals)
            scores = []
            # Score each unique value based on associated weights
            for val in unique:
                scores.append(np.where(vals==val, weights, 0).sum())
            res.append(unique[np.argmax(scores)])
        
        # Process results
        if len(cols) == 1:
            return res[0]
        else:
            return pd.Series(data=res, index=[str(col)+suffix for col in cols])
    

class EventsCollection(EventsFrame):
    """
    User-level class for managing linear and points events data. This class is 
    used for complex data sets with multiple groups of events, grouped by at 
    least one key column (e.g., route ID). Data is managed using both the 
    pandas tabular data package as well as the ranges range data package.

    EventsCollection class instances represent complex events data sets with 
    multiple groups of events which are distinguished by at least one set of 
    keys (e.g., years of data or inventory categories). These collections can 
    be used for a variety of linear referencing operations and events 
    manipulations, such as dissolves based on a subset of events columns, 
    returning a simplified data set with a selection of columns aggregated. 
    Additionally, these collections can be used to perform automated 
    intersections with another EventsCollection class instance using the 
    retrieve() method, retrieving column data from another collection and 
    relating it to the original collection's events data.

    EventsCollection class instances can be queried using the get_subset() and 
    get_group() methods, returning a pared down EventsCollection or a specific 
    EventsGroup, respectively. Similarly, this can be done using object 
    indexing, passing a mixture of unique values and valid slices of unique 
    key values to return a subset of the collection as an EventsCollection 
    instance, or just unique key values to return a unique group as an 
    EventsGroup instance.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe which contains linear or point events data.
    keys : list or tuple
        A list or tuple of dataframe column labels which define the unique 
        groups of events within the events dataframe. Common examples include 
        year or route ID columns which distinguish unrelated sets of events 
        within the events dataframe.
    beg, end : str or label
        Column labels within the events dataframe which represent the linearly 
        referenced location of each event. For linear events both are required,
        defining the begin and end location of each event. For point events, 
        only 'beg' is required, defining the exact location of each event (the 
        'end' property will automatically be set to be equal to the 'beg' 
        property).
    geom : str or label, optional
        Column label within the events dataframe which represents the shapely 
        geometry associated with each event if available. If provided, 
        certain additional class functionalities will be made available.
    closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
            'neither'}, default 'left_mod'
        Whether intervals are closed on the left-side, right-side, both or 
        neither.

        Options
        -------
        left : ranges are always closed on the left and never closed on the 
            right.
        left_mod : ranges are always closed on the left and only closed on the 
            right when the next range is not consecutive.
        right : ranges are always closed on the right and never closed on the 
            right.
        right_mod : ranges are always closed on the right and only closed on 
            the left when the previous range is not consecutive.
        both : ranges are always closed on both sides
        neither : ranges are never closed on either side

    sort : bool, default False
        Whether to sort the events dataframe by its keys and begin and end 
        values upon its creation.
    """

    def __init__(self, df, keys=None, beg=None, end=None, 
        geom=None, closed='left_mod', sort=False, **kwargs):
        # Validate keys option
        if keys is None:
            raise Exception("If no keys are required to define unique groups "
                "of events, please use the EventsGroup class instead of the "
                "EventsCollection class.")
        # Initialize EventsFrame superclass
        super(EventsCollection, self).__init__(
            df=df, keys=keys, beg=beg, end=end, geom=geom, sort=sort, **kwargs)
        # Log input values
        self.closed = closed
        
        # Create events log
        self.log = EventsLog()
        
    def __getitem__(self, keys):
        # Determine type of retrieval - single group or filter slice
        if isinstance(keys, tuple):
            if any(isinstance(key, slice) for key in keys):
                # Partial slice
                return self.get_subset(keys)
            else:
                # Single group
                return self.get_group(keys, empty=False)
        else:
            if isinstance(keys, slice):
                # Partial slice
                return self.get_subset(keys)
            else:
                # Single group
                return self.get_group(keys, empty=False)

    def from_similar(self, df, **kwargs):
        """
        Create an EventsCollection from the input dataframe, assuming the same 
        column labels and closed parameter as the calling collection. 
        Additional constructor keyword arguments can be passed through 
        **kwargs.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe which contains linear or point events data, 
            formatted with standard labels. If multiple keys are detected, they 
            will be assigned in the order in which they appear within the 
            target dataframe. Only one of each begin and end option may be 
            used. The geometry label is optional.
        **kwargs
            Additional keyword arguments to be passed to the EventsCollection 
            constructor.
        """
        # Build the events collection
        kwargs = {**dict(
            keys=self.keys,
            beg=self.beg,
            end=self.end,
            geom=self.geom,
            closed=self.closed,
        ), **kwargs}
        ec = self.__class__(df, **kwargs)
        return ec

    @classmethod
    def from_standard(cls, df, require_end=False, **kwargs):
        """
        Create an EventsCollection from the input dataframe assuming standard 
        column labels. These standard labels can be modified on the class 
        directly be modifying the associated class attributes:
        - default_keys
        - default_beg
        - default_end
        - default_geom

        Standard labels include:
        keys : 'RID', 'YEAR', 'KEY'
        beg : 'BMP', 'BEG', 'FROM'
        end : 'EMP', 'END', 'TO'
        geom : 'geometry'

        Additional constructor keyword arguments can be passed through 
        **kwargs.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe which contains linear or point events data, 
            formatted with standard labels. If multiple keys are detected, they 
            will be assigned in the order in which they appear within the 
            target dataframe. Only one of each begin and end option may be 
            used. The geometry label is optional.
        require_end : bool, default False
            Whether to raise an error if no valid unique end column label is 
            found. If False, no end label will be used when generating the 
            collection.
        **kwargs
            Additional keyword arguments to be passed to the EventsCollection 
            constructor.
        """
        # Check for standard label assignments
        keys, beg, end, geom = [], None, None, None
        for col in df.columns:
            # Check for key labels
            if col in cls.default_keys:
                keys.append(col)
            # Check for other labels
            if col in cls.default_beg:
                if not beg is None:
                    raise ValueError("There is more than one standard label "
                        "in the provided dataframe for the 'beg' parameter."
                        f"Standard labels include {cls.default_beg}.")
                beg = col
            if col in cls.default_end:
                if not end is None:
                    raise ValueError("There is more than one standard label "
                        "in the provided dataframe for the 'end' parameter."
                        f"Standard labels include {cls.default_end}.")
                end = col
            if col in cls.default_geom:
                if not geom is None:
                    raise ValueError("There is more than one standard label "
                        "in the provided dataframe for the 'geom' parameter."
                        f"Standard labels include {cls.default_geom}.")
                geom = col
        # Check for missing labels
        if beg is None:
            raise ValueError("No standard label provided for the 'beg' "
                f"parameter. Standard labels include {cls.default_beg}.")
        if end is None and require_end:
            raise ValueError("No standard label provided for the 'end' "
                f"parameter. Standard labels include {cls.default_end}.")
        # Build the events collection
        ec = cls(df, keys=keys, beg=beg, end=end, geom=geom, **kwargs)
        return ec
        
    @property
    def log(self):
        return self._log
        
    @log.setter
    def log(self, obj):
        if not isinstance(obj, EventsLog):
            raise TypeError("Log must be EventsLog class instance.")
        self._log = obj
    
    def _build_empty(self):
        return pd.DataFrame(columns=self.df.columns)
    
    def reset_log(self):
        """
        Reset the log of built events groups.
        """
        # Reset log
        self._log = {}

    def _validate_keys(self, keys):
        """
        Validate the input list or tuple of keys to determine if it is a valid 
        query for the collection's collection dictionary.
        """
        # Validate input keys
        if self.num_keys == 0:
            if not keys is None:
                raise ValueError("No keys defined in the collection to be "
                    "queried.")
        elif self.num_keys == 1:
            if isinstance(keys, list) or isinstance(keys, tuple):
                keys = keys[0]
        elif self.num_keys > 1:
            if not isinstance(keys, list) and not isinstance(keys, tuple):
                raise TypeError("Input keys information must be provided as a "
                    "list or tuple with a length equal to self.num_keys.")
            elif len(keys) != self.num_keys:
                raise ValueError("Must provide a number of keys which is "
                    "equal to the number of keys defined in the collection "
                    f"({self.num_keys} required, {len(keys)} provided).")
            keys = tuple(keys)
        # Return validated keys
        return keys

    def overlay_average(self, other, cols=None, **kwargs):
        """
        """
        # Validate input
        # - Input events
        if not isinstance(other, self.__class__):
            raise TypeError(f"Input 'other' must be {self.__class__.__name__} "
                "type.")
        # - Same number of keys
        if not self.num_keys == other.num_keys:
            raise ValueError("Other collection must have the same number of "
                "keys as the target collection.")
        # - Input retrieval columns
        cols = other._validate_cols(cols)
        if len(cols) == 0:
            raise ValueError("At least one retrieve column must be provided.")

        # Prepare for retrieval
        def _apply_retrieve(r):
            try:
                # Retrieve corresponding events group
                group_key = tuple(r[loc] for loc in self.key_locs)
                eg = other.get_group(group_key, empty=False)
                # Overlay with record bounds
                res = eg.overlay_average(r[self.beg_loc], r[self.end_loc], 
                    cols=cols, **kwargs)
                # Retrieve requested column data
                if not res is list:
                    res = [res]
            except KeyError:
                res = [np.nan for loc in locs]
            return res

        # Get positional indexes of requested columns
        locs = [other.columns.index(col) for col in cols]
        
        # Perform overlays
        res = [_apply_retrieve(r) for r in self.df.values]
        res = pd.DataFrame(res, columns=cols, index=self.df.index)

        # Return retrieved column data
        return res

    def merge(self, other):
        """
        Create an EventsMerge instance with this collection as the left and the 
        other collection as the right. This can then be used to retrieve 
        attributes from the other collection to be appended to this 
        collection's dataframe.
        
        Parameters
        ----------
        other : EventsCollection
            Another events collection with similar keys which will be merged 
            with this events collection, producing an EventsMerge instance 
            which can be used to perform various overlay operations to retrieve 
            attributes and more from the target collection.
        """
        # Create merge
        em = EventsMerge(self, other)
        return em

    def project_parallel(self, other, samples=3, buffer=100, match='all', 
            choose=1, sort_locs=True, **kwargs):
        """
        Project an input geodataframe of linear geometries onto parallel events 
        in the events dataframe, producing linearly referenced locations for all 
        input geometries which are found to be parallel based on buffer and 
        sampling parameters.
        
        Parameters
        ----------
        other : gpd.GeoDataFrame
            Geodataframe containing linear geometry which will be projected onto 
            the events dataframe.
        samples : int, default 3
            The number of equidistant sample points to take along each geometry 
            being projected to check for nearby geometry.
        buffer : float, default 100
            The max distance to search for input geometries to project against 
            the events' geometries. Measured in terms of the geometries' 
            coordinate reference system.
        match : {'all', int}, default 'all'
            How many sample points must find a nearby target event to produce a 
            positive match to that event, resulting in a projection.
        choose : {int, 'all'}, default 1
            How many target geometries to choose when more than one match 
            occurs.
        sort_locs : bool, default True
            Whether begin and end location values should be sorted, ensuring 
            that all events are increasing and monotonic.
        **kwargs
            Keyword arguments to be passed to the EventsCollection constructor 
            upon completion of the projection.
        """
        # Create projector
        pp = ParallelProjector(self, other, samples=samples, buffer=buffer)
        # Perform match and return results in new events collection
        return EventsCollection(
            pp.match(match=match, choose=choose, sort_locs=sort_locs),
            keys=self.keys,
            beg=self.beg,
            end=self.end,
            closed=self.closed,
            **kwargs
        )
    
    def get_group(self, keys, empty=True, **kwargs) -> EventsGroup:
        """
        Retrieve a unique group of events based on provided key values.
 
        Parameters
        ----------
        keys : key value, tuple of key values, or list of the same
            If only one key column is defined within the collection, a single 
            column value may be provided. Otherwise, a tuple of column values 
            must be provided in the same order as they appear in self.keys. To 
            get multiple groups, a list of key values or tuples may be 
            provided.
        empty : bool, default True
            Whether to allow for empty events groups to be returned when the 
            provided keys are valid but are not associated with any actual 
            events. If False, these cases will return a KeyError.
        """
        # Enforce multiple keys method
        if not isinstance(keys, list):
            keys = [keys]
            ndim = 1
        else:
            ndim = len(keys)

        # Iterate over keys
        dfs = []
        for keys_i in keys:
            # Attempt to retrieve from log
            keys_i = self._validate_keys(keys_i)
            try:
                dfs.append(self.log[keys_i].df)
            except KeyError:
                # Attempt to retrieve group
                try:
                    df = self.groups.get_group(keys_i)
                    dfs.append(df)
                    # Add group to log
                    self.log[keys_i] = self._build_group(df)
                # - Collection is None (i.e., no defined keys)
                except AttributeError:
                    dfs.append(self.df)
                    break
                # - Invalid group keys (i.e., empty group)
                except KeyError as e:
                    # Deal with empty group
                    if empty:
                        dfs.append(self._build_empty())
                    else:
                        raise KeyError("Invalid EventsCollection keys: "
                                       f"{keys_i}")
        # Log and return retrieved group
        if ndim == 1:
            return self._build_group(dfs[0])
        else:
            df = pd.concat(dfs)
            try:
                df = gpd.GeoDataFrame(
                    df, geometry=self.geom, crs=self.df.crs)
            except:
                pass
            return EventsCollection(
                df=df, keys=self.keys, beg=self.beg, end=self.end, 
                geom=self.geom, closed=self.closed)

    def get_subset(self, keys, reduce=True, **kwargs):
        """
        Retrieve a subset of the events collection based on the provided key 
        values or slices. Returned events must satisfy all keys.

        Parameters
        ----------
        keys : list or tuple of slice, list, or other
            A list of either (1) slices which can be used to slice the key 
            values present in self.key_values for the associated key, (2) a 
            list of values which reflect those in self.key_values, or (3) a 
            single value which is present in self.key_values. Inputs must be 
            provided in the same order as they appear in self.keys.
        reduce : bool, default True
            Whether to simplify the resulting EventsCollection by removing any 
            keys which are queried for a single value and become obsolete.
            
            For example, if one key represents years of data and a single year 
            is provided, that key will be removed from the resulting collection 
            as it can no longer be queried further.
        """
        # Determine filtering
        keys = self._validate_keys(keys)
        key_values = self.key_values
        mask = pd.Series(data=True, index=self.df.index)
        new_keys = []
        for key, val in zip(self.keys, keys):
            # Determine input type and perform filter
            try:
                if isinstance(val, slice):
                    new_keys.append(key)
                    mask &= self.df[key].isin(key_values[key][val])
                elif isinstance(val, (list, np.ndarray)):
                    new_keys.append(key)
                    mask &= self.df[key].isin(val)
                else:
                    if not reduce:
                        new_keys.append(key)
                    mask &= self.df[key] == val
            except:
                raise ValueError(f"Unable to filter key '{key}' based on "
                    f"provided input value {val}.")
        
        # Produce filtered collection
        df = self.df.loc[mask, :]
        ec = EventsCollection(df, keys=new_keys, beg=self.beg, end=self.end, 
                              geom=self.geom, closed=self.closed)
        return ec

    def get_matching(self, other, **kwargs):
        """
        Retrieve a subset of the events collection based on the unique group 
        values present in another provided events collection.

        Parameters
        ----------
        other : EventsCollection
            Another events collection with matching keys which will be used to 
            select a subset of this events collection based on its key values.
        """
        # Get subset of groups
        return self.get_group(other.group_keys_unique, empty=True)

    def _build_group(self, df):
        """
        Build a group based on the input dataframe which should be a subset of 
        the events collection's dataframe.
        """
        # Build and return events group
        return EventsGroup(df=df, beg=self.beg, end=self.end, 
                           geom=self.geom, closed=self.closed)


###########
# HELPERS #
###########

def get_most(arr, weights):
    """
    Select the item from the input array which is associated with the highest 
    total weight from the weights array. Scores are computed by summing the 
    weights for each unique array value.
    
    Parameters
    ----------
    arr, weights : array-like
        Arrays of equal length of target values and weights associated with 
        each value.
    """
    # Enforce numpy arrays
    arr = np.asarray(arr)
    weights = np.asarray(weights)
    # Group and split sorted target array
    sorter = np.argsort(arr)
    unique, splitter = np.unique(arr[sorter], return_index=True)
    splitter = splitter[1:]
    # Split weights and aggregate
    splits = np.split(weights[sorter], splitter)
    scores = [x.sum() for x in splits]
    # Return the highest scoring item
    return unique[np.argmax(scores)]

def get_mode(arr):
    """
    Select the item from the input array which appears most frequently.
    
    Parameters
    ----------
    arr : array-like
        Array with target values
    """
    # Enforce numpy array
    arr = np.asarray(arr)
    # Find most frequent unique value and return
    unique, counts = np.unique(arr, return_counts=True)
    return unique[np.argmax(counts)]


####################
# COMMON FUNCTIONS #
####################

def check_compatibility(objs, errors='raise', **kwargs):
    """
    Check if the input list of EventsCollections are all compatible for 
    merging, unifying, or similar relational processes. Errors will be raised 
    if objects are not found to be compatible with information about why they 
    are not compatible. If requested, errors can be ignored, returning False 
    instead. If all objects are compatible, the function will return True.

    Parameters
    ----------
    objs : list-like of EventsCollections
        List of EventsCollection objects to be tested against each other.
    errors : {'raise','ignore'}
        How to respond to errors when they arise.
    """
    # Ensure minimum objects provided
    try:
        assert len(objs) > 0
    except AssertionError:
        raise ValueError("Must provide at least one object for testing.")
    try:
        # Ensure type
        try:
            assert all(isinstance(obj, EventsCollection) for obj in objs)
        except AssertionError:
            raise TypeError("All input objects must be EventsCollections.")
        # Ensure matching keys
        try:
            num_keys = objs[0].num_keys
            for obj in objs[1:]:
                assert obj.num_keys == num_keys
        except AssertionError:
            raise ValueError(
                "All input objects must have the same number of keys.")
    except Exception as e:
        if errors == 'raise':
            raise e
        else:
            return False
    return True


#####################
# LATE DEPENDENCIES #
#####################

from linref.events.merge import EventsMerge, EventsMergeAttribute
from linref.events.spatial import ParallelProjector