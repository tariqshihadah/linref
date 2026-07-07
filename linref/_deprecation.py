"""
Deprecation utilities for linref.

Provides a decorator for managing API deprecations, aligned with
conventions from pandas, scikit-learn, and geopandas.

Usage
-----
Deprecating a function or method::

    @deprecated(since="1.0.0", alternative="df.lr.relate()")
    def old_function():
        ...
"""

import warnings
import functools

from linref.errors import LinrefDeprecationWarning


def deprecated(since, message="", alternative=""):
    """
    Decorator to mark a function, method, or class as deprecated.

    Issues a ``FutureWarning`` on use and appends a deprecation note to the
    docstring.

    Parameters
    ----------
    since : str
        Version in which the deprecation was introduced.
    message : str, optional
        Additional context or migration instructions.
    alternative : str, optional
        Suggested replacement. Appended to warning if provided.
    """

    def decorator(obj):
        # Build warning message
        kind = "class" if isinstance(obj, type) else "function"
        name = obj.__qualname__
        warn_msg = f"{name} is deprecated since linref {since}."
        if alternative:
            warn_msg += f" Use {alternative} instead."
        if message:
            warn_msg += f" {message}"

        # Build docstring addendum
        doc_addendum = (
            f"\n\n.. deprecated:: {since}\n"
            f"   {warn_msg}\n"
        )

        if isinstance(obj, type):
            # Class deprecation: wrap __init__
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(warn_msg, LinrefDeprecationWarning, stacklevel=2)
                orig_init(self, *args, **kwargs)

            obj.__init__ = new_init
            obj.__doc__ = (obj.__doc__ or "") + doc_addendum
            return obj
        else:
            # Function/method deprecation
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                warnings.warn(warn_msg, LinrefDeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)

            wrapper.__doc__ = (obj.__doc__ or "") + doc_addendum
            return wrapper

    return decorator
