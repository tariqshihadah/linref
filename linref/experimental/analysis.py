import numpy as np
from rangel import RangeCollection

def rasterize_events(
    events,
    values=None,
    size=1,
    blur=0,
    blur_style='linear',
    normalize=True,
    bounds=None,
    fill='cut', 
    closed='left_mod',
    rc=None,
    **kwargs
):
    """
    Digitize and buffer events over a defined range, extending them across 
    uniform steps within the range and scaling their values relative to their 
    digital distance from their intersecting step location.
    
    Parameters
    ----------
    events : 1d or 2d array-like
        The locations of the target events being analyzed. If points, must be 
        flat array-like or of shape (x, 1). If ranges, must be array-like of 
        shape (x, 2). All event locations should fall on the range [beg, end] 
        (if provided) to be considered in the analysis.
    values : numeric or 1d array-like, optional
        The value(s) associated with each event being analyzed. If not 
        provided, all values will default to be 1.
    size : positive numeric, default 1
        The length of each pixel used to perform the analysis.
    blur : int, default 0
        The number of pixels to blur events across based on the blur style.
    blur_style : str or callable, default 'linear'
        The scaling function to be called at each blurring step to scale 
        original values. If a callable is provided, it must accept a single 
        integer input for the zero-indexed pixel number, returning a single 
        float scaling value. Predefined blurring functions can be called using 
        the following labels:
        
        Options
        -------
        linear : linearly scale down values from the original value to zero 
            at the first index outside the blurred pixel range
        none : do not scale down original values
        
    normalize : bool, default True
        Whether to normalize resulting buffered values so each event's total 
        buffers sum to 1.
    bounds : two-value numeric tuple, optional
        The values at which to begin and end the pixelation analysis. If not 
        provided, will default to the min and max location values in the events 
        data, respectively. If a predefined range collection is provided, that 
        will supersede these input parameters.
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
    
    closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
            'neither'}, default 'left'
        Whether window intervals are closed on the left-side, right-side, 
        both, or neither. Values of 'left' and 'right' will avoid double-
        counting and under-counting events which occur on the edges of 
        windows. If a predefined range collection is provided, that will 
        supersede these input parameters.

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
        
    rc : RangeCollection, optional
        A predefined valid consecutive range collection instance which defines 
        the analysis domain and which will be used to intersect with input 
        events.
            
    Created:  2022-01-06
    Modified: 2022-01-12
    """
    ##################
    # VALIDATE INPUT #
    ##################
    
    # Validate blur style
    if blur_style=='linear':
        blur_function = lambda n: (blur - n) / blur
    elif blur_style=='none':
        blur_function = lambda n: 1
    elif callable(blur_style):
        blur_function = blur_style
    else:
        raise ValueError("Input blur_style must be callable or label of "
                            "valid predefined scaling function.")            
    
    # Validate events
    try:
        events = np.asarray(events, dtype=float)
    except ValueError:
        raise TypeError("Could not convert input events to np.ndarray of "
                        "dtype float.")
    # - Determine shape and event type
    if events.ndim == 1:
        events = np.append(events, events).reshape(2,-1).T
    elif events.ndim == 2 and events.shape[1] == 1:
        events = np.append(events, events).reshape(2,-1).T
    elif events.ndim == 2 and events.shape[1] == 2:
        pass
    else:
        raise ValueError("Invalid events input. Must be array-like of "
                         "shape (x,), (x, 1), or (x, 2).")
    
    # Validate values data
    if not values is None:
        try:
            values = np.full(events.shape[0], float(values)) # extend scalar
        except:
            try:
                values = np.asarray(values, dtype=float).flatten() # coerce
            except:
                raise TypeError("Could not convert input values to np.ndarray "
                                "of dtype float.")
        if not values.shape[0] == events.shape[0]:
            raise ValueError("Number of event values must be equal to the "
                            "number of events provided. Provided: "
                            f"{events.shape[0]} events, {values.shape[0]} "
                            "values.")
    
    # Validate analysis range collection
    if rc is None:
        if bounds is None:
            beg = np.min(events) if bounds is None else bounds[0]
            end = np.max(events) if bounds is None else bounds
        else:
            try:
                beg, end = bounds
            except:
                raise ValueError("If used, bounds must be provided as "
                                 "two-value tuple of scalars defining the "
                                 "bounds of the analysis.")
        # Create a new analysis range collection
        rc = RangeCollection.from_steps(beg, end, length=size, steps=1, 
                                        fill=fill, closed=closed)
    elif not isinstance(rc, RangeCollection):
        raise ValueError("Input range collection is not valid.")
            
    ####################
    # PERFORM ANALYSIS #
    ####################
    
    # Intersect events with analysis range collection
    intx = rc.is_intersecting(beg=events[:,0],
                              end=events[:,1],
                              squeeze=False) * 1
    
    # Blur events
    scale = blur_function(0)
    data = intx * scale
    for step in range(1, blur):
        # Create and blur buffered data
        scale = blur_function(step)
        buff = np.pad(intx, ((step, step), (0, 0)), mode='constant') * scale
        # Apply buffered data
        forw = buff[:-step * 2, :]
        back = buff[step * 2:, :]
        data += forw + back
        
    # Normalize buffered data
    if normalize:
        denom = data.sum(axis=0)
        data = np.divide(data, denom, out=np.zeros_like(data), where=denom!=0)
    
    # Apply values if used
    if not values is None:
        return np.multiply(data, values)
    else:
        return data

def buffer_events(
    events, 
    size, 
    steps, 
    scaler='linear', 
    normalize=True, 
    beg=None, 
    end=None, 
    fill='cut', 
    closed='left_mod', 
    rc=None,
    values=None, # ADDED
):
    """
    Digitize and buffer events over a defined range, extending them across 
    uniform steps within the range and scaling their values relative to their 
    digital distance from their intersecting step location.
    
    Parameters
    ----------
    events : 1d or 2d array-like
        The locations of the target events being analyzed. If points, must be 
        flat array-like or of shape (x, 1). If ranges, must be array-like of 
        shape (x, 2). All event locations should fall on the range [beg, end] 
        (if provided) to be considered in the analysis.
    size : positive numeric
        The size of each buffering step used to perform the analysis.
    steps : int, default 10
        The number of steps of buffering to perform.
    scaler : str or callable, default 'linear'
        The scaling function to be called at each buffering step to scale 
        buffered values. If a callable is provided, it must accept a single 
        integer input for the zero-indexed buffering step, returning a single 
        float scaling value. Predefined scaling functions can be called using 
        the following labels:
        
        Options
        -------
        linear : linearly scale down values from the center value to the 
            edge value
        none : do not scale down buffered values
        
    normalize : bool, default True
        Whether to normalize resulting buffered values so each event's total 
        buffers sum to 1.
    beg, end : numeric, optional
        The locations at which to begin and end the buffering analysis. 
        If not provided, will default to the min and max values in the 
        events data, respectively. If a predefined range collection is 
        provided, that will supersede these input parameters.
    closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
            'neither'}, default 'left'
        Whether window intervals are closed on the left-side, right-side, 
        both, or neither. Values of 'left' and 'right' will avoid double-
        counting and under-counting events which occur on the edges of 
        windows.

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
        
    rc : RangeCollection, optional
        A predefined valid consecutive range collection instance which defines 
        the analysis domain and which will be used to intersect with input 
        events.
            
    Created:  2022-01-06
    Modified: 2022-01-06
    """
    ##################
    # VALIDATE INPUT #
    ##################
    
    # Validate scaler
    if scaler=='linear':
        scaler = lambda n: (steps - n) / steps
    elif scaler=='none':
        scaler = lambda n: 1
    else:
        if not callable(scaler):
            raise ValueError("Input scaler must be callable or label of valid "
                             "predefined scaler function.")
    
    # Validate events
    try:
        events = np.asarray(events, dtype=float)
    except ValueError:
        raise TypeError("Could not convert input events to np.ndarray of "
                        "dtype float.")
    # - Determine shape and event type
    if events.ndim == 1:
        events = np.append(events, events).reshape(2,-1).T
    elif events.ndim == 2 and events.shape[1] == 1:
        events = np.append(events, events).reshape(2,-1).T
    elif events.ndim == 2 and events.shape[1] == 2:
        pass
    else:
        raise ValueError("Invalid events input. Must be array-like of "
                         "shape (x,), (x, 1), or (x, 2).")
    
    # Validate values data
    if values is None:
        values = 1 # default
    try:
        values = np.full(events.shape[0], float(values)) # extend single value
    except:
        try:
            values = np.asarray(values, dtype=float).flatten() # coerce
        except:
            raise TypeError("Could not convert input values to np.ndarray "
                            "of dtype float.")
    if not values.shape[0] == events.shape[0]:
        raise ValueError("Number of event values must be equal to the number "
                         f"of events provided. Provided: {events.shape[0]} "
                         f"events, {values.shape[0]} values.")
            
    ####################
    # PERFORM ANALYSIS #
    ####################
    
    # Intersect events with range
    if rc is None:
        beg = np.min(events) if beg is None else beg
        end = np.max(events) if end is None else end
        rc = RangeCollection.from_steps(beg, end, length=size, steps=1, 
                                        fill=fill, closed=closed)
    elif not isinstance(rc, RangeCollection):
        raise ValueError("Input range collection is not valid.")
    intx = rc.is_intersecting(beg=events[:,0], end=events[:,1]) * 1
    
    # Buffer events
    scale = scaler(0)
    data = intx * scale
    for step in range(1, steps):
        # Create and scale buffered data
        scale = scaler(step)
        buff = np.pad(intx, ((step, step), (0, 0)), mode='constant') * scale
        # Apply buffered data
        forw = buff[:-step * 2, :]
        back = buff[step * 2:, :]
        data += forw + back
        
    # Normalize buffered data
    if normalize:
        denom = data.sum(axis=0)
        data = np.divide(data, denom, out=np.zeros_like(data), where=denom!=0)
    
    return data