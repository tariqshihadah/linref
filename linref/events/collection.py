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
collections can be used to perform automated merges and intersections with 
other EventsCollection class instances using the .merge() method, retrieving 
column data from another collection and relating it to the original 
collection's events data.

EventsGroup class instances represent simple events data sets with a single 
group of contiguous events. These groups can be used for a variety of linear 
referencing operations such as overlays to determine portions of events 
overlapped by an input range, intersections to determine which events intersect 
with an input range, length-weighted averages of event column values based on 
an input range, and more.

EventsCollection class instances can be queried using square bracket indexing 
or the .get_subset() and .get_group() methods, returning a pared down 
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

To retrieve information from one events collection and apply it to the events 
of the other.
>>> ec.merge()

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
    default_beg =  ['BMP', 'BEG', 'FROM', 'LOC']
    default_end =  ['EMP', 'END', 'TO']
    default_geom = ['geometry']

    def __init__(
        self, df, keys=None, beg=None, end=None, geom=None, route=None, 
        closed=None, sort=False, **kwargs):
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

    def __iter__(self):
        return (self.get_group(key) for key in self.group_keys_unique)

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

            # Reset logs
            try:
                self.log.reset()
            except:
                pass
            self._initialize_df()
        else:
            raise TypeError(
                "Input dataframe must be pandas DataFrame class instance.")

    @property
    def size(self):
        """
        Return the size of the events dataframe.
        """
        return self._df.size

    @property
    def shape(self):
        """
        Return the shape of the events dataframe.
        """
        return self._df.shape

    def _initialize_df(self):
        """
        Class-specific dataframe initialization processes.
        """
        pass

    def _sort_df(self, df):
        """
        Sort the given dataframe by the collection's keys and begin/end 
        columns, returning the sorted dataframe.
        """
        return df.sort_values(
            by=self.keys + [self.beg, self.end], ascending=True)

    def set_df(self, obj, inplace=False):
        """
        Set a new events dataframe.
        """
        # Define target, copy if needed
        ef = self if inplace else self.copy()
        # Assign dataframe
        ef.df = obj
        # Return if needed
        if not inplace:
            return ef

    def sort(self, inplace=False):
        """
        Sort the events dataframe based on target columns.
        """
        # Create a copy if requested
        ef = self if inplace else self.copy()
        # Log sorting
        ef._sort = True
        ef.df = self.df
        if not inplace:
            return ef

    def cast_gdf(self, inplace=False, **kwargs):
        """
        Convert the events dataframe to a geodataframe, passing the input 
        keyword arguments, such as crs and geometry, to the gpd.GeoDataFrame 
        constructor. See documentation for this constructor for more 
        information.
        """
        # Attempt to convert to geodataframe
        gdf = gpd.GeoDataFrame(self.df, **kwargs)
        # Log new geodataframe
        if inplace:
            self.df = gdf
            return
        else:
            ef = self.copy()
            ef.df = gdf
            return ef

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
        return self._df.columns.values.tolist()

    @property
    def targets(self):
        """
        A list of begin, end, and key columns within the events dataframe.
        """
        # Define target columns
        targets = [self.beg, self.end] + self.keys
        return targets

    @property
    def spatials(self):
        """
        A list of geometry and route columns within the events dataframe, if 
        defined.
        """
        # List defined spatial columns
        spatials = []
        if not self.geom is None:
            spatials.append(self.geom)
        if not self.route is None:
            spatials.append(self.route)
        return spatials

    @property
    def others(self):
        """
        A list of columns within the events dataframe which are not the begin, 
        end, or key columns.
        """
        # Define other columns
        exclude = self.targets + self.spatials
        others = [col for col in self.df.columns if not col in exclude]
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
    def is_point(self):
        """
        Returns True if the collection's beg and end columns are the same, 
        implying that it is a collection of point events.
        """
        return self._beg == self._end

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

    def iter_groups(self):
        """
        Return an iterator which will iterate through all groups in the 
        collection, yielding each group's key as well as the associated 
        EventsGroup.
        """
        return ((key, self.get_group(key)) for key in self.group_keys_unique)

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
    
    def set_closed(self, closed=None, inplace=False):
        """
        Change whether ranges are closed on left, right, both, or neither side. 
        
        Parameters
        ----------
        closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
                'neither'}, optional
            Whether intervals are closed on the left-side, right-side, both or 
            neither. If None, will default to 'left_mod' for linear events and 
            'both' for point events.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        # Ensure valid option selected
        if closed is None:
            if self.is_point:
                closed = 'both'
            else:
                closed = 'left_mod'
        elif not closed in RangeCollection._ops_closed:
            raise ValueError(
                "Closed parameter must be one of "
                f"{RangeCollection._ops_closed}.")
        # Apply parameter
        if inplace:
            self._closed = closed
        else:
            ec = self.copy()
            ec._closed = closed
            return ec

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
        agg_geometry=True, agg_routes=True, dropna=False, fillna=None, 
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
        agg_geometry : bool, default True
            Whether to create an aggregated geometries field, populated with 
            aggregated shapely geometries based on those contained in the 
            collection's geometry field. If not needed, set to False to reduce 
            processing time.
        agg_routes : bool, default True
            Whether to create an aggregated routes field, populated with 
            MLSRoute object class instances, created based on aggregated 
            segment geometries and begin and end mile posts. If not needed, 
            set to False to reduce processing time.
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
            closed=self.closed, missing_data='ignore')
        return ec

    def project(self, other, buffer=100, nearest=True, loc_label='LOC', 
            dist_label='DISTANCE', build_routes=True, **kwargs):
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
            If False, all matches will be returned. If True, when multiple 
            equidistant points exist, choose the first result that appears.
        loc_label, dist_label : label
            Labels to be used for created columns for projected locations on 
            target events groups and nearest point distances between target 
            geometries and events geometries.
        build_routes : bool, default True
            Whether to automatically build routes using the build_routes() 
            method if routes are not already available.
        **kwargs
            Keyword arguments to be passed to the EventsFrame constructor 
            upon completion of the projection.
        """
        # Validate input geodataframe
        if not isinstance(other, gpd.GeoDataFrame):
            raise TypeError("Other object must be gpd.GeoDataFrame instance.")
        else:
            try:
                other_geometry = other.geometry.name
            except AttributeError:
                raise AttributeError(
                    "No geometry data set in other geodataframe.")
        other = other.copy()

        # Check for invalid column names
        if (self.route in other.columns):
            raise ValueError(
                f"Invalid column name '{self.route}' found in target "
                "geodataframe.")
        if len(set(self.keys) & set(other.columns)) > 0:
            invalid = set(self.keys) & set(other.columns)
            raise ValueError(
                f"Target geodataframe contains at least one events collection "
                f"key column name {invalid}.")

        # Ensure that geometries and routes are available
        if self.geom is None:
            raise ValueError(
                "No geometry found in events dataframe. If valid shapely "
                "geometries are available in the dataframe, set this with the "
                f"{self.__class__.__name__}'s geom property.")
        elif self.route is None:
            if build_routes:
                self.build_routes()
            else:
                raise ValueError(
                    "No routes found in events dataframe. If valid shapely "
                    "geometries are available in the dataframe, create routes "
                    "by calling the build_routes() method on the "
                    f"{self.__class__.__name__} class instance.")
        
        # Join the other geodataframe to this one
        select_cols = self.keys + [self.route, self.geom]
        if nearest:
            joined = other.sjoin_nearest(
                self.df[select_cols],
                max_distance=buffer,
                how='left'
            )
            # Drop duplicates (required for equidistant ties)
            joined = joined[~joined.index.duplicated(keep='first')]
        else:
            # Buffer geometry for spatial join
            buffered_geoms = self.df.geometry.buffer(buffer)
            joined = other.sjoin(
                self.df[select_cols].set_geometry(buffered_geoms),
                how='left'
            )

        # Project input geometries onto event geometries
        def _project(r):
            try:
                return r[self.route].project(r[other_geometry])
            except AttributeError:
                return
        locs = joined.apply(_project, axis=1)
        joined[loc_label] = locs
        # return joined # modified to return EC 7/27/2022
        # Prepare and return data
        return self.__class__(
            joined.drop(columns=[self.route]),
            keys=self.keys,
            beg=loc_label,
            closed=self.closed,
            missing_data='ignore',
            **kwargs
        )

    def to_grid(self, dissolve=False, **kwargs):
        """
        Use the events dataframe to create a grid of zero-length, equidistant 
        point events which span the bounds of each event.

        Parameters
        ----------
        length : numerical, default 1.0
            A fixed distance between each point on the grid.
        fill : {'none','cut','extend','right','balance'}, default 'cut'
            How to fill a gap at the end of an event's range.

            Options
            -------
            none : no point will be generated at the end of the input range 
                unless it falls directly on the defined grid distance.
            cut : a point will be generated at the very end of the input range, 
                at a distance less than or equal to the defined grid distance.
            right : the final point will be generated at a distance equal to 
                the defined grid distance, even if this extends beyond the full 
                input range.
            extend : a point will be generated at the very end of the input 
                range, at a distance greater than or equal to the defined grid 
                distance.
            balance : if the final range is greater than or equal to half the 
                target range length, perform the cut method; if it is less, 
                perform the extend method.
        
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
        grid = []
        for (*keys, beg, end), index in gen:
            # Build grid points
            rng = RangeCollection.from_steps(beg, end, **kwargs).cut(beg, end)
            locs = np.append(rng.begs, rng.ends[-1])
            num_locs = len(locs)
            # Assemble sliding window data
            grid.append(
                np.concatenate(
                    [
                        [keys]*num_locs,        # Event keys
                        np.tile(locs, (2,1)).T, # Point locations
                        [[index]]*num_locs      # Parent index value
                    ],
                    axis=1
                )
            )

        # Merge and prepare data, return
        grid = np.concatenate(grid, axis=0)
        df = pd.DataFrame(
            data=grid,
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
        res = self.__class__(
            df,
            keys=self.keys,
            beg=self.beg,
            end=self.end,
            missing_data='ignore'
        )
        return res

    def to_windows(self, dissolve=False, retain=True, endpoint=False, **kwargs):
        """
        Use the events dataframe to create sliding window events of a fixed 
        length and a fixed number of steps, and which fill the bounds of each 
        event.

        Parameters
        ----------
        length : numerical, default 1.0
            A fixed length for all windows being defined.
        steps : int, default 1
            A number of steps per window length. The resulting step length will 
            be equal to length / steps. For non-overlapped windows, use a steps 
            value of 1.
        fill : {'none','cut','extend','left','right','balance'}, default 'cut'
            How to fill a gap at the end of an event's range.

            Options
            -------
            none : no window will be generated to fill the gap at the end of 
                the input range.
            cut : a truncated window will be created to fill the gap with a 
                length less than the full window length.
            extend : the final window will be anchored on the grid defined by 
                the step value, extending beyond the window length to the right
                bound of the event.
            left : the final window will be anchored on the end of the input 
                range and will extend the full window length to the left. 
            right : the final window will be anchored on the grid defined by 
                the step value, extending the full window length to the right, 
                beyond the event's end value.
            balance : if the final range is greater than or equal to half the 
                target range length, perform the cut method; if it is less, 
                perform the extend method.
        
        dissolve : bool, default False
            Whether to dissolve the events dataframe before performing the 
            transformation.
        retain : bool, default True
            Whether to retain all fields from the original dataframe in the 
            newly generated dataframe.
        endpoint : bool, default False
            Add a point event at the end of each event range.
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
            if endpoint:
                rng = rng.append(end, end)
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
        # Retain original fields if requested
        if retain:
            df = df.merge(
                self.df[self.others], left_on='index_parent', 
                right_index=True, how='left'
            )
        # Prepare collection and return
        res = self.__class__(
            df,
            keys=self.keys,
            beg=self.beg,
            end=self.end,
            missing_data='ignore'
        )
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
            'neither'}, optional
        Whether intervals are closed on the left-side, right-side, both or 
        neither. If None, will default to 'left_mod' for linear events and 
        'both' for point events.

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

    def __init__(self, df, beg=None, end=None, geom=None, closed=None, 
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
        Lengths of all event ranges.
        """
        return self.rng.lengths

    @property
    def centers(self):
        """
        Centers of all event ranges.
        """
        return self.rng.centers
            
    @property
    def shape(self):
        return self.df.shape
    
    def _build_rng(self):
        # Build range collection
        rng = RangeCollection.from_array(
            self.df[[self.beg,self.end]].values, closed=self.closed, sort=False)
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

    def intersecting(self, beg=None, end=None, other=None, closed='both', 
        get_mask=False, **kwargs):
        """
        Retrieve a selection of records from the group of events based 
        on provided begin and end locations.

        Parameters
        ----------
        beg, end : numerical or array-like, optional
            The begin and end locations of the range or ranges to be tested. If 
            a single range is to be tested, provide a numeric value. If 
            multiple, provide an array-like with a single begin and end value 
            for each range. If no end parameter provided, point locations will 
            be assumed and end will be set equal to beg. Not required if other 
            parameter is used.
        other : EventsGroup, optional
            Other EventsGroup instance to be intersected with this one. Can 
            be provided instead of beg, end, and closed parameters and will 
            take precedence over other input.
        closed : str {'left', 'right', 'both', 'neither'}, default 'both'
            Whether input interval is closed on the left-side, right-side, both 
            or neither.

            Options
            -------
            left : ranges are always closed on the left and never closed on the 
                right.
            right : ranges are always closed on the right and never closed on 
                the right.
            both : ranges are always closed on both sides
            neither : ranges are never closed on either side

        get_mask : bool, default False
            Whether to return a boolean mask for selecting from the events 
            dataframe instead of the selection from the dataframe itself.
        """
        # Deprecation
        get_mask = kwargs.get('mask', get_mask)
        # Check for other input
        if not other is None:
            if not isinstance(other, EventsGroup):
                raise TypeError(
                    "If provided, input other parameter must be valid "
                    "EventsGroup instance.")
            other = other.rng
        # Intersect range
        mask = self.rng.intersecting(
            beg=beg, end=end, other=other, closed=closed, **kwargs)
        if get_mask:
            return mask
        else:
            if mask.ndim > 1:
                mask = mask.any(axis=1)
            df = self.df.loc[mask, :]
            return df

    def overlay(self, beg=None, end=None, other=None, **kwargs):
        """
        Compute overlap of the input bounds with respect to the 
        events group.
        
        Parameters
        ----------
        beg, end : scalar or array of scalars
            Begin and end locations of the overlaid range(s).
        other : EventsGroup, optional
            Other EventsGroup instance to be overlaid with this one. Can be 
            provided instead of beg and end parameters and will take precedence 
            over other input.
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
        """
        # Check for other input
        if not other is None:
            if not isinstance(other, EventsGroup):
                raise TypeError(
                    "If provided, input other parameter must be valid "
                    "EventsGroup instance.")
            beg = other.rng.begs
            end = other.rng.ends
        # Compute range overlaps
        weights = self.rng.overlay(beg=beg, end=end, **kwargs)
        return weights
    
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
            'neither'}, optional
        Whether intervals are closed on the left-side, right-side, both or 
        neither. If None, will default to 'left_mod' for linear events and 
        'both' for point events.

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
    missing_data : {'ignore','drop','warn','raise'}, default 'warn'
        What to do when the input dataframe contains missing values in the 
        target key, beg, and end columns.

        Options
        -------
        ignore : do nothing.
        drop : drop all records which contain any missing data in the target 
            columns.
        warn : log a warning when records are missing data.
        raise : raise a ValueError when records are missing data.
    """

    def __init__(self, df, keys=None, beg=None, end=None, 
        geom=None, closed=None, sort=False, missing_data='warn', **kwargs):
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
        
        # Additional processing
        self._check_missing_data(missing_data=missing_data)

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

    def _initialize_df(self):
        """
        Class-specific dataframe initialization processes.
        """
        self._empty_df = pd.DataFrame(columns=self.columns)
        self._empty_group = self._build_group(self._empty_df.copy())

    def _check_missing_data(self, missing_data='warn'):
        """
        Check for missing data in keys, beg, end, and geometry fields. Warn 
        user when target fields contain null data.
        """
        # If ignore
        if missing_data=='ignore':
            return
        elif missing_data in ['warn','raise','drop']:
            # Find, count missing data records
            mask = self.df[self.targets].isna().any(axis=1)
            count = mask.sum()
            # Address if more than one records contain missing data
            if count > 0:
                # Drop records
                if missing_data=='drop':
                    self.df = self.df[~mask].copy()
                    return
                # Warn or raise error
                else:
                    # Prepare message
                    message = (
                        f"Input events dataframe has {count:,.0f} records "
                        "with missing data in target columns. This may cause "
                        "unexpected behaviors.")
                    if missing_data=='raise':
                        raise ValueError(message)
                    else:
                        warnings.warn(message)
                        return
        else:
            raise ValueError(
                "Invalid input missing_data parameter. Must be one of "
                "('ignore','drop','warn','raise').")

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
        return self._empty_group.copy(deep=True)
    
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
                raise ValueError(
                    "No keys defined in the collection to be queried.")
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
    
    def round(self, decimals=0, factor=1, inplace=False):
        """
        Round the bounds of all events to the specified number of decimals 
        or to a specified rounding factor.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimals to round event bound values to.
        factor : scalar, default 1
            Rounding factor to apply to the event bound values. For example, 
            use `factor=0.5` (and `decimals=0`) to round each value to the 
            nearest 0.5.
        """
        # Copy data
        df = self.df.copy()
        # Perform rounding
        df[self.beg] = \
            np.round(df[self.beg] / factor, decimals=decimals) * factor
        df[self.end] = \
            np.round(df[self.end] / factor, decimals=decimals) * factor
    
        # Apply update
        if inplace:
            self.df = df
            return
        else:
            ec = self.from_similar(df)
            return ec

    def shift(self, distance=0, inplace=False):
        """
        Shift the bounds of all events by the specified value.

        Parameters
        ----------
        distance : scalar, default 0
            The amount to shift each event bound by.
        """
        # Copy data
        df = self.df.copy()
        # Perform shifting
        df[self.beg] = df[self.beg] + distance
        df[self.end] = df[self.end] + distance
    
        # Apply update
        if inplace:
            self.df = df
            return
        else:
            ec = self.from_similar(df)
            return ec

    def separate(self, eliminate_inside=False, inplace=False, **kwargs):
        """
        Separate the bounds of all events so that none directly overlap. 
        This is done using the rangel.RangeCollection.separate() method 
        on each EventsGroup.

        Parameters
        ----------
        eliminate_inside : boolean, default False
            Whether to automatically eliminate ranges which are entirely 
            overlapped by other ranges, producing zero-length ranges (e.g., 
            point events).
        inplace : boolean, default False
            Whether to perform the operation in place. If False, will return a 
            modified copy of the events object.
        """
        # Iterate through events groups
        records = []
        for key, group in self.iter_groups():
            # Separate group ranges
            separated = group.rng.separate(
                drop_short=False, eliminate_inside=eliminate_inside)
            # Apply separated ranges to a copy of the group
            updated = group.df.copy()
            updated[self.beg] = separated.begs
            updated[self.end] = separated.ends
            records.append(updated)

        # Prepare new dataframe
        df = pd.concat(records)
        df = df.loc[self.df.index] # Retain original sorting

        # Apply update
        if inplace:
            self.df = df
            return
        else:
            ec = self.from_similar(df)
            return ec

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
            choose=1, sort_locs=True, build_routes=True, **kwargs):
        """
        Project an input geodataframe of linear geometries onto parallel events 
        in the events dataframe, producing linearly referenced locations for all 
        input geometries which are found to be parallel based on buffer and 
        sampling parameters.
        
        Parameters
        ----------
        other : gpd.GeoDataFrame
            Geodataframe containing linear geometry which will be projected 
            onto the events dataframe.
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
        build_routes : bool, default True
            Whether to automatically build routes using the build_routes() 
            method if routes are not already available.
        **kwargs
            Keyword arguments to be passed to the EventsCollection constructor 
            upon completion of the projection.
        """
        # Ensure that geometries and routes are available
        if self.geom is None:
            raise ValueError(
                "No geometry found in events dataframe. If valid shapely "
                "geometries are available in the dataframe, set this with the "
                f"{self.__class__.__name__}'s geom property.")
        elif self.route is None:
            if build_routes:
                self.build_routes()
            else:
                raise ValueError(
                    "No routes found in events dataframe. If valid shapely "
                    "geometries are available in the dataframe, create routes "
                    "by calling the build_routes() method on the "
                    f"{self.__class__.__name__} class instance.")
            
        # Create projector
        pp = ParallelProjector(self, other, samples=samples, buffer=buffer)
        # Perform match and return results in new events collection
        return EventsCollection(
            pp.match(match=match, choose=choose, sort_locs=sort_locs),
            keys=self.keys,
            beg=self.beg,
            end=self.end,
            closed=self.closed,
            missing_data='ignore',
            **kwargs
        )
    
    def get_group(self, keys, empty=True, log_empty=True, 
            **kwargs) -> EventsGroup:
        """
        Retrieve a unique group of events based on provided key values.
 
        Parameters
        ----------
        keys : key value, tuple of key values, or list of the same
            If only one key column is defined within the collection, a single 
            column value may be provided. Otherwise, a tuple of column values 
            must be provided in the same order as they appear in self.keys.
        empty : bool, default True
            Whether to allow for empty events groups to be returned when the 
            provided keys are valid but are not associated with any actual 
            events. If False, these cases will return a KeyError.
        log_empty : bool, default True
            Whether created empty events should be logged and stored within 
            the collection to allow for quicker access. More memory intensive 
            but may produce moderate performance improvements if empty keys 
            will be accessed repeatedly.
        """
        # Attempt to retrieve from log
        keys = self._validate_keys(keys)
        try:
            # Retrieve from log
            group = self.log[keys]
        except KeyError:
            # Attempt to retrieve dataframe to create new group
            try:
                # Build and add group to log
                group = self._build_group(self._groups.get_group(keys))
                self.log[keys] = group
            # Invalid group keys (i.e., empty group)
            except KeyError:
                # Deal with empty group
                if empty:
                    group = self._build_empty()
                    if log_empty:
                        self.log[keys] = group
                else:
                    raise KeyError(
                        f"Invalid EventsCollection keys: {keys}")
            # Collection is None (i.e., no defined keys)
            except AttributeError:
                raise ValueError("No defined group keys.")
        return group

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
        try:
            ec = EventsCollection(
                df, keys=new_keys, beg=self.beg, end=self.end, 
                geom=self.geom, closed=self.closed, missing_data='ignore')
        except:
            raise ValueError(
                "Unable to produce EventsCollection subset due to unknown "
                "error.")
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
        try:
            return EventsGroup(
                df=df, beg=self.beg, end=self.end, geom=self.geom, 
                closed=self.closed)
        except Exception as e:
            display(df)
            raise e


####################
# COMMON FUNCTIONS #
####################

def from_standard(df, require_end=False, **kwargs):
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
    ec = EventsCollection.from_standard(df, require_end=require_end, **kwargs)
    return ec


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