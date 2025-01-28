import numpy as np
from linref.events import common, base

def _method_require(**requirements):
    """
    Callable decorator to require that a class meets certain attribute or 
    property requirements.
    """
    def decorator(func):
        def wrapper(obj, *args, **kwargs):
            for key, value in requirements.items():
                if getattr(obj, key) != value:
                    raise ValueError(
                        f"The {func.__name__} method is only available "
                        f"for {obj.__class__.__name__} instances with {key}={value}."
                    )
            return func(obj, *args, **kwargs)
        return wrapper
    return decorator

def _prepare_data_array(data, name, ndim=1, dtype=None, copy=None):
    """
    Function for validating input data as a scalar np.array using the given 
    name, number of dimensions, and optional dtype and copy arguments.
    """
    # Initialize data as a numpy array
    try:
        try:
            data = np.asarray(data, dtype=dtype, copy=copy)
        except TypeError: # numpy 1.X compatibility
            data = np.array(data, dtype=dtype, copy=False)
        assert data.ndim == ndim
    except AssertionError:
        if dtype is None:
            raise ValueError(
                f"Invalid input data for `{name}`. Must be a {ndim}D array-"
                "like object."
            )
        else:
            raise ValueError(
                f"Invalid input data for `{name}`. Must be a {ndim}D array-"
                f"like object with dtype={dtype}. Provided array has shape="
                f"{data.shape} and dtype={data.dtype}."
            )
    return data

def _validate_scalar_or_array_input(events, value, name, dtype=None, fill=False, nonzero=False):
    """
    Function for validating input values as a scalar or array-like object with 
    the same length as the number of events in the input range.
    """
    # Validate input
    if np.isscalar(value):
        if fill:
            value = np.full(events.num_events, value)
    else:
        try:
            value = np.asarray(value, dtype=dtype)
            assert len(value) == events.num_events
        except:
            raise ValueError(
                f"Input '{name}' must be a scalar or an array-like with a "
                "length equal to the number of events in the input collection."
            )
    if nonzero:
        if np.any(value == 0):
            raise ValueError(
                f"Input '{name}' must be non-zero."
            )
    return value

def _stringify_instance(events):
    # Determine event type
    typologies = []
    typologies.append('grouped' if events.is_grouped else 'ungrouped')
    if events.is_point:
        typologies.append('point')
    else:
        if events.is_located:
            typologies.append('located')
        if events.is_monotonic:
            typologies.append('monotonic')
        else:
            typologies.append('non-monotonic')
        typologies.append('linear')
    event_type = ', '.join(typologies[:-1]) + ' ' + typologies[-1]

    # Set closed string
    closed = f", closed={events.closed}" if events.is_linear else ''

    # Create text string
    text = (
        f"{events.__class__.__name__}({events.num_events:,.0f} {event_type} events{closed})"
    )
    return text

def _represent_records(events):
    """
    Create a string representation of a EventsData instance, displaying only the 
    first and last few records.
    """
    # If no ranges present, return self as a string
    if events.num_events == 0:
        return str(events)
    # Determine number of records to show
    display_max = common.display_max
    if events.num_events > display_max:
        # Define head/skip/tail selections
        display_head = (display_max // 2) + (display_max % 2)
        display_tail = (display_max // 2)
        display_skip = events.num_events - display_max
        # Define bool mask
        display_select = np.array(
            [True]  * display_head + 
            [False] * display_skip + 
            [True]  * display_tail)
    else:
        # Default head/skip/tail selections
        display_head = events.num_events
        display_tail = display_skip = 0
        display_select = np.array([True] * events.num_events)
    # Determine numbers of left and right digits to display
    ld = len(str(int(events.arr[display_select].max())))
    rd = 3
    digits = ld + rd + 1
    # Create formatter
    records = []
    closed = events.closed
    if events.groups is not None:
        max_len = max([len(str(x)) for x in events.groups])
        group_strings = [str(x) if len(str(x)) <= 20 else str(x)[:17] + '...' for x in events.groups]
        groups = np.array(
            [f'group({x: <{min(max_len, 20)}}) ' for x in group_strings])
    else:
        groups = np.full(events.num_events, '')
    # Define record string template and select features to display
    if events.is_point:
        display_features = {
            'index': events.index[display_select],
            'groups': groups[display_select],
            'locs': events.locs[display_select],
            'modified_edges': events.modified_edges[display_select],
        }
        record_template = '{index}, {groups}@ {locs: >{digits}.{rd}f}'
    elif events.is_linear:
        display_features = {
            'index': events.index[display_select],
            'groups': groups[display_select],
            'begs': events.begs[display_select],
            'ends': events.ends[display_select],
            'modified_edges': events.modified_edges[display_select],
        }
        record_template = '{index}, {groups}{lb}{begs: >{digits}.{rd}f}, {ends: >{digits}.{rd}f}{rb}'
        if events.is_located:
            display_features['locs'] = events.locs[display_select]
            record_template += ' @ {locs: >{digits}.{rd}f}'

    # Iterate over selected features and create strings
    for i in range(sum(display_select)):
        params = {k: v[i] for k, v in display_features.items()}
        record = record_template.format(
            lb='[' if (closed in ['left','left_mod','both']) or params['modified_edges'] else '(',
            rb=']' if (closed in ['right','right_mod','both']) or params['modified_edges'] else ')',
            digits=digits, rd=rd, **params)
        records.append(record)

    # Create skipped record label if required
    if display_skip > 0:
        # Label skipped records
        spacer_label = '{:,.0f} records'.format(display_skip)
        # Format label
        spaces = max(ld*2 + rd*2 + 6 - len(spacer_label), 6)
        spacer = '.' * (spaces // 2) + spacer_label + \
            '.' * (spaces // 2 + spaces % 2)
        records = \
            records[:display_head] + [spacer] + records[-display_tail:]
    # Format full text string and return
    return '\n'.join(records) + '\n' + str(events)
