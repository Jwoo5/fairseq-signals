from typing import Any, Hashable, List, Set, Sequence, Tuple, Union

import numpy as np
import pandas as pd

def to_list(obj: Any) -> List[Any]:
    """Convert some object to a list of object(s) unless already one.

    Parameters
    ----------
    obj : any
        The object to convert to a list.

    Returns
    -------
    list
        The processed object.

    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, (np.ndarray, set, dict)):
        return list(obj)

    return [obj]

def check_cols(
    data: pd.DataFrame,
    cols: Union[Hashable, Sequence[Hashable]],
    raise_err_on_unexpected: bool = False,
    raise_err_on_existing: bool = False,
    raise_err_on_missing: bool = False,
) -> Tuple[Set[Hashable], Set[Hashable], Set[Hashable]]:
    """
    Check DataFrame columns for expected columns and handle errors.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to check columns against.
    cols : hashable or list of Hashable
        The column(s) to check for in the DataFrame.
    raise_err_on_unexpected : bool, default False
        Raise an error if unexpected columns are found.
    raise_err_on_existing : bool, default False
        Raise an error if any of the specified columns already exist.
    raise_err_on_missing : bool, default False
        Raise an error if any of the specified columns are missing.

    Returns
    -------
    Tuple[Set[Hashable], Set[Hashable], Set[Hashable]]
        A tuple containing sets of unexpected, existing, and missing columns.
    """
    cols = set(to_list(cols))
    data_cols = set(data.columns)

    unexpected = data_cols - cols
    if raise_err_on_unexpected and len(unexpected) > 0:
        raise ValueError(f"Unexpected columns: {', '.join(unexpected)}")

    existing = data_cols.intersection(cols)
    if raise_err_on_existing and len(existing) > 0:
        raise ValueError(f"Existing columns: {', '.join(existing)}")

    missing = cols - data_cols
    if raise_err_on_missing and len(missing) > 0:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    return unexpected, existing, missing

def drop_na_cols(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns where all values are NaN.

    Parameters
    ----------
    data : pandas.DataFrame
        Data.

    Returns
    -------
    pandas.DataFrame
        Data without any all-NaN columns.
    """
    all_nan_cols = data.isna().all(axis=0)

    return data.drop(all_nan_cols.index[all_nan_cols], axis=1)

def explode_with_order(series: pd.Series, order_col: str = "order") -> pd.DataFrame:
    """
    Explode a series and add a column representing each element's indices in the original list.

    Requires a unique index.

    Useful for maintaining the order of exploded series.

    Parameters
    ----------
    series : pd.Series
        The input Pandas Series containing lists to be exploded.
    order_col : str, default "order"
        Name of the column to store the order of elements in the original list.

    Raises
    ------
    ValueError
        If the index of the input series is not unique.

    Returns
    -------
    pd.DataFrame
        Exploded series converted to a DataFrame to include the `order_col` column.
    """
    # Assert that the index is unique
    if not series.index.is_unique:
        raise ValueError("The input series must have a unique index.")

    # Explode the series and convert to DataFrame
    series_expl = series.explode().to_frame()

    # Add a column to represent the order of elements in the original list
    series_expl[order_col] = series_expl.groupby(series_expl.index).cumcount()

    return series_expl

def numpy_series_to_dataframe(series: pd.Series) -> pd.DataFrame:
    """
    Converts a pandas Series containing numpy arrays or NaN values into a DataFrame with separate columns for each array element.

    Parameters
    ----------
    series : pd.Series
        A pandas Series where each element is a numpy array of consistent size or a NaN.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each column corresponds to an element in the arrays of the input Series.

    Raises
    ------
    ValueError
        If the elements of the series are not all numpy arrays or NaN.
    """
    # Make a copy of the series to avoid modifying the original data
    series = series.copy()
    
    # Check for NaN values and replace them with numpy arrays full of NaNs of appropriate length
    isnull = series.isnull()
    if isnull.any():
        max_length = series.dropna().apply(len).max()  # Determine max length of arrays
        series.loc[isnull] = [np.full(max_length, np.nan)] * isnull.sum()
    
    # Convert the series of arrays to a DataFrame
    expanded = pd.DataFrame(series.tolist(), index=series.index)
    expanded.columns = series.name + '_' + expanded.columns.astype(str)

    return expanded

def and_conditions(conditions: List[pd.Series]) -> pd.Series:
    """
    Perform element-wise logical AND operation on a list of boolean Series.

    Parameters
    ----------
    conditions : list of pd.Series
        A list of boolean Pandas Series.

    Raises
    ------
    ValueError
        If the conditions are not Pandas boolean series.

    Returns
    -------
    pd.Series
        A new Pandas Series resulting from the element-wise logical AND operation.
    """
    for condition in conditions:
        is_bool_series(condition, raise_err=True)

    return reduce(lambda x, y: x & y, conditions)

def or_conditions(conditions: List[pd.Series]) -> pd.Series:
    """
    Perform element-wise logical OR operation on a list of boolean Series.

    Parameters
    ----------
    conditions : list of pd.Series
        A list of boolean Pandas Series.

    Raises
    ------
    ValueError
        If the conditions are not Pandas boolean series.

    Returns
    -------
    pd.Series
        A new Pandas Series resulting from the element-wise logical OR operation.
    """
    for condition in conditions:
        is_bool_series(condition, raise_err=True)

    return reduce(lambda x, y: x | y, conditions)
