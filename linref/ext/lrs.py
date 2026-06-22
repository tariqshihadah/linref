from __future__ import annotations
import copy
import pandas as pd
from linref.utility.utility import label_list_or_none, label_or_none
from linref.events.common import closed_all, default_closed
from linref.errors import LRSConfigurationError


class LRS(object):

    def __init__(
        self,
        key_col: str | list[str] | None = None,
        chain_col: str | None = None,
        loc_col: str | None = None,
        beg_col: str | None = None,
        end_col: str | None = None,
        geom_col: str | None = None,
        geom_m_col: str | None = None,
        closed: str | None = None,
    ) -> None:
        """
        Define a linear referencing system (LRS) configuration.

        Parameters
        ----------
        key_col : str or list of str, optional
            Column name(s) identifying one or more unique route identifier 
            or grouping columns (e.g., 'Route', 'County', etc.).
        chain_col : str, optional
            Column name for chain indices identifying contiguous geometry 
            groups within each route. Unlike key columns, this column may 
            not exist in the DataFrame until ``add_chaining()`` is called.
            When present in the DataFrame, it is automatically included in
            grouping operations alongside the key columns and functionally
            operates as an additional key level.
        loc_col : str, optional
            Column name for the location measure of point events or located
            linear events.
        beg_col : str, optional
            Column name for the begin measure of linear events.
        end_col : str, optional
            Column name for the end measure of linear events.
        geom_col : str, optional
            Column name for the geometry geometry objects.
        geom_m_col : str, optional
            Column name for M-enabled geometry objects.
        closed : {'left', 'right', 'left_mod', 'right_mod', 'both', 'neither'}, optional
            Interval closure type for linear events. Defaults to 'left_mod'.
        """
        # Set LRS parameters
        self.set_params(
            key_col=key_col,
            chain_col=chain_col,
            loc_col=loc_col,
            beg_col=beg_col,
            end_col=end_col,
            geom_col=geom_col,
            geom_m_col=geom_m_col,
            closed=closed,
            inplace=True
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        def _fmt(val):
            return repr(val) if isinstance(val, str) else val
        return (
            "LRS("
            f"key_col={self.key_col}, "
            f"chain_col={_fmt(self.chain_col)}, "
            f"loc_col={_fmt(self.loc_col)}, "
            f"beg_col={_fmt(self.beg_col)}, "
            f"end_col={_fmt(self.end_col)}, "
            f"geom_col={_fmt(self.geom_col)}, "
            f"geom_m_col={_fmt(self.geom_m_col)}, "
            f"closed={_fmt(self.closed)})"
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, LRS):
            return False
        return self.params == other.params
    
    @property
    def is_linear(self) -> bool:
        """
        Return whether the LRS is linear (i.e., has begin and end columns 
        defined). This does not check for presence of the columns in the 
        DataFrame.
        """
        return (self.beg_col is not None) and (self.end_col is not None)
    
    @property
    def is_point(self) -> bool:
        """
        Return whether the LRS is point-based (i.e., has a location column but
        no begin or end columns defined). This does not check for presence of 
        the columns in the DataFrame.
        """
        return (self.loc_col is not None) and (self.beg_col is None) and (self.end_col is None)
    
    @property
    def is_located(self) -> bool:
        """
        Return whether the LRS is located (i.e., has a location column 
        defined). This does not check for presence of the column in the 
        DataFrame.
        """
        return self.loc_col is not None
    
    @property
    def is_grouped(self) -> bool:
        """
        Return whether the LRS is grouped (i.e., has one or more key columns
        defined). This does not check for presence of the columns in the 
        DataFrame.
        """
        return len(self.key_col) > 0
    
    @property
    def is_spatial(self) -> bool:
        """
        Return whether the LRS is spatial (i.e., has a geometry column
        defined). This does not check for presence of the column in the
        DataFrame.
        """
        return self.geom_col is not None
    
    @property
    def is_spatial_m(self) -> bool:
        """
        Return whether the LRS is spatial with M-enabled geometries (i.e., has
        a geometry_m column defined). This does not check for presence of the
        column in the DataFrame.
        """
        return self.geom_m_col is not None
    
    @property
    def is_chained(self) -> bool:
        """
        Return whether the LRS has a chain column defined. This does not 
        check for presence of the column in the DataFrame.
        """
        return self.chain_col is not None
    
    @property
    def params(self) -> dict:
        return {
            'key_col': self.key_col,
            'chain_col': self.chain_col,
            'loc_col': self.loc_col,
            'beg_col': self.beg_col,
            'end_col': self.end_col,
            'geom_col': self.geom_col,
            'geom_m_col': self.geom_m_col,
            'closed': self.closed
        }
    
    def copy(self, deep: bool = False) -> LRS:
        """
        Create an exact copy of the object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    def set_closed(
            self,
            closed: str | None = None,
            inplace: bool = False
        ) -> LRS | None:
        """
        Set LRS closure type.

        Parameters
        ----------
        closed : {'left', 'right', 'left_mod', 'right_mod', 'both', 'neither'}, optional
            The closure type to set. If None, sets to default closure type.
        inplace : bool, default False
            Whether to apply changes to the LRS in place.
        """
        if self.is_linear:
            if closed is None:
                closed = default_closed
            elif closed not in closed_all:
                raise ValueError(
                    f"Invalid LRS closure: {closed}. Must be one of: {closed_all}.")
        else:
            if closed is not None:
                raise LRSConfigurationError(
                    "Cannot set closure type for non-linear LRS.")
        obj = self if inplace else self.copy(deep=True)
        obj.closed = closed
        return None if inplace else obj

    
    def set_params(self, inplace: bool = False, **kwargs) -> LRS | None:
        """
        Set LRS parameters.

        Parameters
        ----------
        inplace : bool, default False
            Whether to apply changes to the LRS in place.
        key_col : label or array-like, optional
            The key column or array-like of key columns to set.
        loc_col : label, optional
            The location measure column to set.
        beg_col : label, optional
            The begin measure column to set.
        end_col : label, optional
            The end measure column to set.
        geom_col : label, optional
            The geometry column to set.
        geom_m_col : label, optional
            The geometry_m column to set.
        closed : {'left', 'right', 'left_mod', 'right_mod', 'both', 'neither'}, optional
            The closure type to set.
        """
        obj = self if inplace else self.copy(deep=True)
        for key, value in kwargs.items():
            if key == 'key_col':
                obj.key_col = label_list_or_none(value, if_none=[])
            elif key == 'chain_col':
                obj.chain_col = label_or_none(value)
            elif key == 'loc_col':
                obj.loc_col = label_or_none(value)
            elif key == 'beg_col':
                obj.beg_col = label_or_none(value)
            elif key == 'end_col':
                obj.end_col = label_or_none(value)
            elif key == 'geom_col':
                obj.geom_col = label_or_none(value)
            elif key == 'geom_m_col':
                obj.geom_m_col = label_or_none(value)
            elif key == 'closed':
                obj.set_closed(closed=value, inplace=True)

        return None if inplace else obj
    
    def add_key(self, key_col: str | list[str], inplace: bool = False) -> LRS | None:
        """
        Add one or more key columns to the LRS.
        
        Parameters
        ----------
        key_col : label or array-like
            The key column or array-like of key columns to add.
        inplace : bool, default False
            Whether to apply changes to the LRS in place.
        """
        obj = self if inplace else self.copy(deep=True)
        obj.key_col.extend(label_list_or_none(key_col))
        return None if inplace else obj

    def remove_key(self, key_col: str | list[str], errors: str = 'raise', inplace: bool = False) -> LRS | None:
        """
        Remove one or more key columns from the LRS.

        Parameters
        ----------
        key_col : label or array-like
            The key column or array-like of key columns to remove.
        errors : {'ignore', 'raise'}, default 'raise'
            Whether to raise an error or ignore missing keys.
        inplace : bool, default False
            Whether to apply changes to the LRS in place.
        """
        obj = self if inplace else self.copy(deep=True)
        for key in label_list_or_none(key_col, if_none=[]):
            try:
                obj.key_col.remove(key)
            except ValueError:
                if errors == 'raise':
                    raise KeyError(f"Key column '{key}' not found in current list: {obj.key_col}.")
                continue
        return None if inplace else obj

    def study(self, df: pd.DataFrame) -> dict:
        """
        Validate the dataframe for LRS compatibility.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to validate.
        how : {'raise', 'verbose', 'bool'}, default 'raise'
            Whether to raise an error or return a boolean for validation failures.
        """
        # Check for presence of LRS columns in the dataframe
        result = {}
        if self.is_grouped:
            missing_keys = [key for key in self.key_col if key not in df.columns]
            result['keys'] = {'defined': True, 'valid': len(missing_keys) == 0, 'missing': missing_keys}
        else:
            result['keys'] = {'defined': False, 'valid': False, 'missing': None}
        if self.is_linear:
            missing_linear = [col for col in [self.beg_col, self.end_col] if col not in df.columns]
            result['linear'] = {'defined': True, 'valid': len(missing_linear) == 0, 'missing': missing_linear}
        else:
            result['linear'] = {'defined': False, 'valid': False, 'missing': None}
        if self.is_located:
            valid = self.loc_col in df.columns
            result['location'] = {'defined': True, 'valid': valid, 'missing': self.loc_col if not valid else None}
        else:
            result['location'] = {'defined': False, 'valid': False, 'missing': None}
        if self.is_spatial:
            valid = self.geom_col in df.columns
            result['geometry'] = {'defined': True, 'valid': valid, 'missing': self.geom_col if not valid else None}
        else:
            result['geometry'] = {'defined': False, 'valid': False, 'missing': None}
        if self.is_spatial_m:
            valid = self.geom_m_col in df.columns
            result['geometry_m'] = {'defined': True, 'valid': valid, 'missing': self.geom_m_col if not valid else None}
        else:
            result['geometry_m'] = {'defined': False, 'valid': False, 'missing': None}
        if self.is_chained:
            valid = self.chain_col in df.columns
            result['chaining'] = {'defined': True, 'valid': valid, 'missing': self.chain_col if not valid else None}
        else:
            result['chaining'] = {'defined': False, 'valid': False, 'missing': None}
        return result
