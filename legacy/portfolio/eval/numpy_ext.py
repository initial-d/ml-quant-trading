"""
=========
numpy_ext
=========
An extension library for NumPy_ that implements common array operations not present in NumPy.
.. _numpy: https://numpy.org/
Installation
------------
**Regular installation**::
    pip install numpy_ext
**Installation for development**::
    git clone https://github.com/3jane/numpy_ext.git
    cd numpy_ext
    pip install -e .[dev]  # note: make sure you are using pip>=20
Window operations
-----------------
- :func:`numpy_ext.expanding`
- :func:`numpy_ext.expanding_apply`
- :func:`numpy_ext.rolling`
- :func:`numpy_ext.rolling_apply`
Operations with nans
--------------------
- :func:`numpy_ext.nans`
- :func:`numpy_ext.drop_na`
- :func:`numpy_ext.fill_na`
- :func:`numpy_ext.fill_not_finite`
- :func:`numpy_ext.prepend_na`
Others
------
- :func:`numpy_ext.apply_map`
- :func:`numpy_ext.expstep_range`
Functions
---------
"""
from functools import partial
from typing import Callable, Any, Union, Generator, Tuple, List

import numpy as np
from joblib import Parallel, delayed

Number = Union[int, float]


def expstep_range(
    start: Number,
    end: Number,
    min_step: Number = 1,
    step_mult: Number = 1,
    round_func: Callable = None
) -> np.ndarray:
    """
    Return spaced values within a given interval. Step is increased by a multiplier on each iteration.
    Parameters
    ----------
    start : int or float
        Start of interval, inclusive
    end : int or float
        End of interval, exclusive
    min_step : int or float, optional
        Minimal step between values. Must be bigger than 0. Default is 1.
    step_mult : int or float, optional
        Multiplier by which to increase the step on each iteration. Must be bigger than 0. Default is 1.
    round_func: Callable, optional
        Vectorized rounding function, e.g. np.ceil, np.floor, etc. Default is None.
    Returns
    -------
    np.ndarray
        Array of exponentially spaced values.
    Examples
    --------
    >>> expstep_range(1, 100, min_step=1, step_mult=1.5)
    array([ 1.        ,  2.        ,  3.5       ,  5.75      ,  9.125     ,
           14.1875    , 21.78125   , 33.171875  , 50.2578125 , 75.88671875])
    >>> expstep_range(1, 100, min_step=1, step_mult=1.5, round_func=np.ceil)
    array([ 1.,  2.,  4.,  6., 10., 15., 22., 34., 51., 76.])
    >>> expstep_range(start=-1, end=-100, min_step=1, step_mult=1.5)
    array([ -1.        ,  -2.        ,  -3.5       ,  -5.75      ,
            -9.125     , -14.1875    , -21.78125   , -33.171875  ,
           -50.2578125 , -75.88671875])
    Generate array of ints
    >>> expstep_range(start=100, end=1, min_step=1, step_mult=1.5).astype(int)
    array([100,  99,  97,  95,  91,  86,  79,  67,  50,  25])
    """
    if step_mult <= 0:
        raise ValueError('mult_step should be bigger than 0')

    if min_step <= 0:
        raise ValueError('min_step should be bigger than 0')

    last = start
    values = []
    step = min_step

    sign = 1 if start < end else -1

    while start < end and last < end or start > end and last > end:
        values.append(last)
        last += max(step, min_step) * sign
        step = abs(step * step_mult)

    values = np.array(values)
    if not round_func:
        return values

    values = np.array(round_func(values))
    _, idx = np.unique(values, return_index=True)
    return values[np.sort(idx)]


def apply_map(func: Callable[[Any], Any], array: Union[List, np.ndarray]) -> np.ndarray:
    """
    Apply a function element-wise to an array.
    Parameters
    ----------
    func : Callable[[Any], Any]
        Function that accepts one argument and returns a single value.
    array : Union[List, np.ndarray]
        Input array or a list. Any lists will be converted to np.ndarray first.
    Returns
    -------
    np.ndarray
        Resulting array.
    Examples
    --------
    >>> apply_map(lambda x: 0 if x < 3 else 1, [[2, 2], [3, 3]])
    array([[0, 0],
           [1, 1]])
    """
    array = np.array(array)
    array_view = array.flat
    array_view[:] = [func(x) for x in array_view]
    return array


#############################
# Operations with nans
#############################


def nans(shape: Union[int, Tuple[int]], dtype=np.float64) -> np.ndarray:
    """
    Return a new array of a given shape and type, filled with np.nan values.
    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., (2, 3) or 2.
    dtype: data-type, optional
    Returns
    -------
    np.ndarray
        Array of np.nans of the given shape.
    Examples
    --------
    >>> nans(3)
    array([nan, nan, nan])
    >>> nans((2, 2))
    array([[nan, nan],
           [nan, nan]])
    >>> nans(2, np.datetime64)
    array(['NaT', 'NaT'], dtype=datetime64)
    """
    if np.issubdtype(dtype, np.integer):
        dtype = np.float
    arr = np.empty(shape, dtype=dtype)
    arr.fill(np.nan)
    return arr


def drop_na(array: np.ndarray) -> np.ndarray:
    """
    Return a given array flattened and with nans dropped.
    Parameters
    ----------
    array : np.ndarray
        Input array.
    Returns
    -------
    np.ndarray
        New array without nans.
    Examples
    --------
    >>> drop_na(np.array([np.nan, 1, 2]))
    array([1., 2.])
    """
    return array[~np.isnan(array)]


def fill_na(array: np.ndarray, value: Any) -> np.ndarray:
    """
    Return a copy of array with nans replaced with a given value.
    Parameters
    ----------
    array : np.ndarray
        Input array.
    value : Any
        Value to replace nans with.
    Returns
    -------
    np.ndarray
        A copy of array with nans replaced with the given value.
    Examples
    --------
    >>> fill_na(np.array([np.nan, 1, 2]), -1)
    array([-1.,  1.,  2.])
    """
    ar = array.copy()
    ar[np.isnan(ar)] = value
    return ar


def fill_not_finite(array: np.ndarray, value: Any = 0) -> np.ndarray:
    """
    Return a copy of array with nans and infs replaced with a given value.
    Parameters
    ----------
    array : np.ndarray
        Input array.
    value : Any, optional
        Value to replace nans and infs with. Default is 0.
    Returns
    -------
    np.ndarray
        A copy of array with nans and infs replaced with the given value.
    Examples
    --------
    >>> fill_not_finite(np.array([np.nan, np.inf, 1, 2]), 99)
    array([99., 99.,  1.,  2.])
    """
    ar = array.copy()
    ar[~np.isfinite(array)] = value
    return ar


def prepend_na(array: np.ndarray, n: int) -> np.ndarray:
    """
    Return a copy of array with nans inserted at the beginning.
    Parameters
    ----------
    array : np.ndarray
        Input array.
    n : int
        Number of elements to insert.
    Returns
    -------
    np.ndarray
        New array with nans added at the beginning.
    Examples
    --------
    >>> prepend_na(np.array([1, 2]), 2)
    array([nan, nan,  1.,  2.])
    """
    return np.hstack(
        (
            nans(n, array[0].dtype) if len(array) and hasattr(array[0], 'dtype') else nans(n),
            array
        )
    )


#############################
# window operations
#############################


def rolling(
    array: np.ndarray,
    window: int,
    skip_na: bool = False,
    as_array: bool = False
) -> Union[Generator[np.ndarray, None, None], np.ndarray]:
    """
    Roll a fixed-width window over an array.
    The result is either a 2-D array or a generator of slices, controlled by `as_array` parameter.
    Parameters
    ----------
    array : np.ndarray
        Input array.
    window : int
        Size of the rolling window.
    skip_na : bool, optional
        If False, the sequence starts with (window-1) windows filled with nans. If True, those are omitted.
        Default is False.
    as_array : bool, optional
        If True, return a 2-D array. Otherwise, return a generator of slices. Default is False.
    Returns
    -------
    np.ndarray or Generator[np.ndarray, None, None]
        Rolling window matrix or generator
    Examples
    --------
    >>> rolling(np.array([1, 2, 3, 4, 5]), 2, as_array=True)
    array([[nan,  1.],
           [ 1.,  2.],
           [ 2.,  3.],
           [ 3.,  4.],
           [ 4.,  5.]])
    Usage with numpy functions
    >>> arr = rolling(np.array([1, 2, 3, 4, 5]), 2, as_array=True)
    >>> np.sum(arr, axis=1)
    array([nan,  3.,  5.,  7.,  9.])
    """
    if not any(isinstance(window, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong window type ({type(window)}) int expected')

    window = int(window)

    if array.size < window:
        raise ValueError('array.size should be bigger than window')

    def rows_gen():
        if not skip_na:
            yield from (prepend_na(array[:i + 1], (window - 1) - i) for i in np.arange(window - 1))

        starts = np.arange(array.size - (window - 1))
        yield from (array[start:end] for start, end in zip(starts, starts + window))

    return np.array([row for row in rows_gen()]) if as_array else rows_gen()


def rolling_apply(func: Callable, window: int, *arrays: np.ndarray, n_jobs: int = 1, **kwargs) -> np.ndarray:
    """
    Roll a fixed-width window over an array or a group of arrays, producing slices.
    Apply a function to each slice / group of slices, transforming them into a value.
    Perform computations in parallel, optionally.
    Return a new np.ndarray with the resulting values.
    Parameters
    ----------
    func : Callable
        The function to apply to each slice or a group of slices.
    window : int
        Window size.
    *arrays : list
        List of input arrays.
    n_jobs : int, optional
        Parallel tasks count for joblib. If 1, joblib won't be used. Default is 1.
    **kwargs : dict
        Input parameters (passed to func, must be named).
    Returns
    -------
    np.ndarray
    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> rolling_apply(sum, 2, arr)
    array([nan,  3.,  5.,  7.,  9.])
    >>> arr2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> func = lambda a1, a2, k: (sum(a1) + max(a2)) * k
    >>> rolling_apply(func, 2, arr, arr2, k=-1)
    array([  nan,  -5.5,  -8.5, -11.5, -14.5])
    """
    if not any(isinstance(window, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong window type ({type(window)}) int expected')

    window = int(window)

    if max(len(x.shape) for x in arrays) != 1:
        raise ValueError('Wrong array shape. Supported only 1D arrays')

    if len({array.size for array in arrays}) != 1:
        raise ValueError('Arrays must be the same length')

    def _apply_func_to_arrays(idxs):
        return func(*[array[idxs[0]:idxs[-1] + 1] for array in arrays], **kwargs)

    array = arrays[0]
    rolls = rolling(
        array if len(arrays) == n_jobs == 1 else np.arange(len(array)),
        window=window,
        skip_na=True
    )

    if n_jobs == 1:
        if len(arrays) == 1:
            arr = list(map(partial(func, **kwargs), rolls))
        else:
            arr = list(map(_apply_func_to_arrays, rolls))
    else:
        f = delayed(_apply_func_to_arrays)
        arr = Parallel(n_jobs=n_jobs)(f(idxs[[0, -1]]) for idxs in rolls)

    return prepend_na(arr, n=window - 1)


def expanding(
    array: np.ndarray,
    min_periods: int = 1,
    skip_na: bool = True,
    as_array: bool = False
) -> Union[Generator[np.ndarray, None, None], np.ndarray]:
    """
    Roll an expanding window over an array.
    The window size starts at min_periods and gets incremented by 1 on each iteration.
    The result is either a 2-D array or a generator of slices, controlled by `as_array` parameter.
    Parameters
    ----------
    array : np.ndarray
        Input array.
    min_periods : int, optional
        Minimum size of the window. Default is 1.
    skip_na : bool, optional
        If False, the windows of size less than min_periods are filled with nans. If True, they're dropped.
        Default is True.
    as_array : bool, optional
        If True, return a 2-D array. Otherwise, return a generator of slices. Default is False.
    Returns
    -------
    np.ndarray or Generator[np.ndarray, None, None]
    Examples
    --------
    >>> expanding(np.array([1, 2, 3, 4, 5]), 3, as_array=True)
    array([array([1, 2, 3]), array([1, 2, 3, 4]), array([1, 2, 3, 4, 5])],
          dtype=object)
    """
    if not any(isinstance(min_periods, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong min_periods type ({type(min_periods)}) int expected')

    min_periods = int(min_periods)

    if array.size < min_periods:
        raise ValueError('array.size should be bigger than min_periods')

    def rows_gen():
        if not skip_na:
            yield from (nans(i) for i in np.arange(1, min_periods))

        yield from (array[:i] for i in np.arange(min_periods, array.size + 1))

    return np.array([row for row in rows_gen()]) if as_array else rows_gen()


def expanding_apply(func: Callable, min_periods: int, *arrays: np.ndarray, n_jobs: int = 1, **kwargs) -> np.ndarray:
    """
    Roll an expanding window over an array or a group of arrays producing slices.
    The window size starts at min_periods and gets incremented by 1 on each iteration.
    Apply a function to each slice / group of slices, transforming them into a value.
    Perform computations in parallel, optionally.
    Return a new np.ndarray with the resulting values.
    Parameters
    ----------
    func : Callable
        The function to apply to each slice or a group of slices.
    min_periods : int
        Minimal size of expanding window.
    *arrays : list
        List of input arrays.
    n_jobs : int, optional
        Parallel tasks count for joblib. If 1, joblib won't be used. Default is 1.
    **kwargs : dict
        Input parameters (passed to func, must be named).
    Returns
    -------
    np.ndarray
    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> expanding_apply(sum, 2, arr)
    array([nan,  3.,  6., 10., 15.])
    >>> arr2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> func = lambda a1, a2, k: (sum(a1) + max(a2)) * k
    >>> expanding_apply(func, 2, arr, arr2, k=-1)
    array([  nan,  -5.5,  -9.5, -14.5, -20.5])
    """
    if not any(isinstance(min_periods, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong min_periods type ({type(min_periods)}) int expected')

    min_periods = int(min_periods)

    if max(len(x.shape) for x in arrays) != 1:
        raise ValueError('Supported only 1-D arrays')

    if len({array.size for array in arrays}) != 1:
        raise ValueError('Arrays must be the same length')

    def _apply_func_to_arrays(idxs):
        return func(*[array[idxs.astype(np.int)] for array in arrays], **kwargs)

    array = arrays[0]
    rolls = expanding(
        array if len(arrays) == n_jobs == 1 else np.arange(len(array)),
        min_periods=min_periods,
        skip_na=True
    )

    if n_jobs == 1:
        if len(arrays) == 1:
            arr = list(map(partial(func, **kwargs), rolls))
        else:
            arr = list(map(_apply_func_to_arrays, rolls))
    else:
        f = delayed(_apply_func_to_arrays)
        arr = Parallel(n_jobs=n_jobs)(map(f, rolls))

    return prepend_na(arr, n=min_periods - 1)
