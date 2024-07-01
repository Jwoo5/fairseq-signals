from typing import (
    Any,
    Generator,
    Hashable,
    Iterable,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import logging

from decimal import Decimal, getcontext
getcontext().prec = 10

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from sklearn.model_selection import GroupShuffleSplit

from fairseq_signals.utils.pandas import check_cols

def __normalize_fractions(
        fractions: Sequence[Union[float, int]],
    ) -> MutableSequence[Union[float, int]]:
    """
    Normalize the fractions to sum to 1.

    Parameters
    ----------
    fractions : Sequence[Union[float, int]]
        Sequence of values whose sum to normalize to 1.

    Raises
    ------
    ValueError
        If the sum is 0.

    Returns
    -------
    MutableSequence[Union[float, int]]
        Scaled sequence such that its sum is 1.
    """
    original_sum = sum(fractions)
    if original_sum == 0:
        raise ValueError("sum of sequence is zero.")

    return [value / original_sum for value in fractions]

def process_fractions(fractions: Union[str, int, float, Sequence[Union[float, int]]]):
    if isinstance(fractions, str):
        fraction_series = pd.Series(fractions).str.replace(
            '[^0-9.,]',
            '',
            regex=True,
        ).str.split(',').explode()
        fractions = list(fraction_series.astype(float))

        # If already adding up to 1, we don't need any further processing below
        # Decimal avoids issues of floating point imprecision creating another value
        if fraction_series.apply(Decimal).sum() == 1:
            return __normalize_fractions(fractions)

    if isinstance(fractions, Iterable):
        frac_list = list(fractions)
    else:
        frac_list = [fractions]

    if any(not isinstance(x, (float, int)) for x in frac_list):
        raise TypeError("fractions must be int or float.")

    if any(x < 0 for x in frac_list):
        raise ValueError("fractions must be non-negative.")

    if 0.0 <= sum(frac_list) < 1.0:
        # max() guarding against floating point imprecision
        frac_list.append(max(0.0, 1.0 - sum(frac_list)))

    return __normalize_fractions(frac_list)

def fractions_to_split(
        fractions: Union[int, float, Sequence[Union[float, int]]],
        n_samples: int,
    ) -> np.ndarray:
    """
    Create an array of index split points useful for dataset splitting.
    Created using the length of the data and the desired split fractions.

    Parameters
    ----------
    fractions : Union[int, float, Sequence[Union[float, int]]]
        Fraction(s) of samples to use for each split.
        If the sum of the sequence (or single value) is in the range
        (0, 1), an additional fraction will be assumed at the end of
        the sequence, such that the new sequence sums to 1.
        If a sequence (or single value) of 2 or more values sums to
        more than 1, it will first be normalized to sum to 1.
        Negative values are always an error.
    n_samples : int
        The total number of samples in the data being split.

    Returns
    -------
    np.ndarray
        Split indices to use in creating the desired split sizes.

    Raises
    ------
    ValueError
        If 'n_samples' is negative.
    TypeError
        If 'fractions' is not int or float.
    ValueError
        If 'fractions' contains negative values.

    Notes
    -----
    For sufficiently small values of a fraction, the corresponding split
    might be of zero size.
    """
    if n_samples < 0:
        raise ValueError("'n_samples' cannot be negative.")

    normalized_indexes = np.cumsum(process_fractions(fractions))

    return np.round(np.array(normalized_indexes[:-1]) * n_samples).astype(int)

def split_idx(
    fractions: Union[float, List[float]],
    n_samples: int,
    randomize: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Create disjoint subsets of indices.

    Parameters
    ----------
    fractions : Union[float, List[float]]
        Fraction(s) of samples between 0 and 1 to use for each split.
    n_samples : int
        The length of the data.
    randomize : bool, default=True
        Whether to randomize the data in the splits.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    Tuple[np.ndarray, ...]
        Disjoint subsets of indices.
    """
    split = fractions_to_split(fractions, n_samples)
    idx = np.arange(n_samples)

    # Optionally randomize
    if randomize:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    return tuple(np.split(idx, split))

def split_dataframe_col(
    data: pd.DataFrame,
    fractions: Union[float, List[float]],
    labels: Optional[List[str]] = None,
    split_col: Hashable = "split",
) -> pd.DataFrame:
    """
    Split a DataFrame into disjoint subsets based on fractions.
    
    Splits are defined by labels assigned in the split_col.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to split.
    fractions : Union[float, List[float]]
        Fraction(s) of samples between 0 and 1 to use for each split.
    labels : Optional[List[str]], default=None
        Labels for the splits. If not provided, a range of integer values will be used.
    split_col : hashable, default="split"
        Name of the column to store the split labels.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the split labels assigned.

    Raises
    ------
    AssertionError
        If the `split_col` already exists in the DataFrame.
    ValueError
        If the length of `labels` does not match the number of splits.
    """
    index = data.index
    data = data.reset_index(drop=True)

    assert split_col not in data.columns

    idxs = split_idx(fractions, len(data))

    if labels is None:
        labels = list(range(len(idxs)))
    elif len(labels) != len(idxs):
        raise ValueError("Length of `labels` must match the number of splits.")

    for i, idx in enumerate(idxs):
        data.loc[idx, split_col] = labels[i]

    return data.set_index(index)

def split_arrays_to_split_series(
    split_arrays: List[np.ndarray],
    labels: Optional[List[str]] = None,
) -> pd.Series:
    """
    Convert split arrays to a split series.

    Parameters
    ----------
    split_arrays : list of numpy.ndarray
        List of split arrays.
    labels : list of str, optional
        Split labels. If not provided, a range of integer values will be used.

    Returns
    -------
    pandas.Series
        A series containing the split indices.
    """
    length = int(sum([len(arr) for arr in split_arrays]))

    splits = pd.Series(np.nan, index=np.arange(length), dtype=np.float64)

    for ind, split_array in enumerate(split_arrays):
        splits.iloc[split_array] = ind

    splits = splits.astype(int)

    if labels is not None:
        if len(labels) != splits.max() + 1:
            raise ValueError("Incorrect number of labels.")

        return splits.map(dict(zip(range(len(labels)), labels)))

    return splits

def grouped_splits(
    groups: np.ndarray,
    fractions: Union[int, float, Sequence[Union[float, int]]],
    labels: Optional[List[str]] = None,
    **group_split_kwargs,
) -> pd.Series:
    """
    Perform grouped shuffle splits using sklearn.model_selection.GroupShuffleSplit.

    Parameters
    ----------
    groups : numpy.ndarray
        Group IDs for each sample.
    fractions : int, float, or sequence of int/float
        Fractions for splitting.
    labels : list of str, optional
        Split labels. If not provided, a range of integer values will be used.
    **group_split_kwargs
        Keyword arguments to pass to sklearn.model_selection.GroupShuffleSplit.

    Returns
    -------
    pandas.Series
        Series containing the split indices.
    """
    fractions = process_fractions(fractions)

    if groups.ndim != 1:
        raise ValueError("Group ID array must be 1-dimensional.")

    if labels is not None:
        if len(labels) != len(fractions):
            raise ValueError("Incorrect number of labels.")

    groups = pd.Series(groups)

    split_arrays = []
    for _ in range(len(fractions) - 1):
        splitter = GroupShuffleSplit(train_size=fractions[0], **group_split_kwargs)
        idx_split, idx = next(splitter.split(groups, groups=groups))
        split_arrays.append(groups.iloc[idx_split].index.values)
        groups = groups.iloc[idx]
        fractions = __normalize_fractions(fractions[1:])

    split_arrays.append(groups.index.values)

    return split_arrays_to_split_series(split_arrays, labels=labels).values

def temporal_splits(
    data: pd.DataFrame,
    fractions: Union[float, List[float]],
    date_col: Hashable,
    labels: Optional[List[str]] = None,
    split_col: Hashable = "split",
) -> pd.DataFrame:
    """
    Create temporal splits.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    fractions : float or list of float
        Fraction(s) of samples between 0 and 1 to use for an initial split. The ID
        assignments to avoid temporal overlap will change these fractions.
    overlapping: bool, default True
        remove_overlap
    date_col : str
        Date column in `data`.
    labels : list of str, optional
        Split labels. If not provided, a range of integer values will be used.
    split_col : str
        Resulting split column defined in `data`.

    Returns
    -------
    pandas.DataFrame
        Data with additional 'split' column indicating the split assignment for each row.
    """
    check_cols(data, date_col, raise_err_on_missing=True)
    check_cols(data, split_col, raise_err_on_existing=True)

    # Sort by date
    data = data.sort_values(date_col)

    # Split
    idxs = split_idx(
        fractions,
        len(data),
        randomize=False,
    )
    data[split_col] = -1
    for i, idx in enumerate(idxs):
        data.iloc[idx, data.columns.get_loc(split_col)] = i

    if labels is not None:
        labels_map = dict(zip(range(len(labels)), labels))
        data[split_col] = data[split_col].map(labels_map)

    # Get date transitions
    transitions = [data[date_col].iloc[idx[-1]] for idx in idxs[:-1]]

    return data, transitions

def grouped_temporal_splits(
    data: pd.DataFrame,
    fractions: Union[str, int, float, Sequence[Union[float, int]]],
    group_col: Hashable,
    date_col: Hashable,
    labels: Optional[List[str]] = None,
    split_col: Hashable = "split",
    filter_strategy: Union[bool, str] = False,
) -> pd.DataFrame:
    """
    Create temporal splits and optionally filter to avoid overlap of IDs.

    An initial split is performed based on the provided fractions to determine temporal
    boundaries. Grouping by 'id' and 'split', each ID is assigned a split based on
    maxizing the overall number of samples. The given fractions may not be represented
    in the splits due to the assignments/filtering needed to avoid temporal overlap.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    fractions : float or list of float
        Fraction(s) of samples between 0 and 1 to use for an initial split. The ID
        assignments to avoid temporal overlap will change these fractions.
    group_col : str
        Group column in `data`.
    date_col : str
        Date column in `data`.
    labels : list of str, optional
        Split labels. If not provided, a range of integer values will be used.
    split_col : str
        Resulting split column defined in `data`.
    filter_strategy : bool or str
        If False, there is no filtering based on temporal versus group assignments.
        If True, filters overlap across all splits. Otherwise, expects a split label
        with which to filter overlap between it and all other splits. E.g., using
        filter_strategy='train' could be used to filter overlap with evaluative splits.

    Returns
    -------
    pandas.DataFrame
        Data with additional split information.
    """
    check_cols(data, [group_col, date_col], raise_err_on_missing=True)
    check_cols(data, [split_col, split_col], raise_err_on_existing=True)

    if not isinstance(filter_strategy, bool):
        if labels is None:
            raise ValueError(
                "Must specify `labels` when using a split label filter strategy."
            )
        elif filter_strategy not in labels:
            raise ValueError(
                "`filter_strategy` must be an item in `labels` when using a "
                "split label filter strategy."
            )

    data, transitions = temporal_splits(
        data,
        fractions,
        date_col,
        labels=labels,
        split_col=split_col,
    )

    grouped = data.groupby([group_col, split_col])

    # Get the number of IDs in each split
    size = grouped.size()
    size.rename("size", inplace=True)
    size = size.reset_index()

    group_split_col = f"{group_col}_{split_col}"

    # Find the split with the maximum sample count
    idxmax = size.groupby(group_col)['size'].idxmax()
    id_split_assign = size.loc[idxmax]
    id_split_assign.drop("size", axis=1, inplace=True)
    id_split_assign.set_index(group_col, inplace=True)
    id_split_assign = id_split_assign[split_col]
    id_split_assign.rename(group_split_col, inplace=True)

    # Add in the ID assignments
    data = data.join(id_split_assign, on=group_col, how="left")

    if filter_strategy == False:
        pass
    elif filter_strategy == True:
        data = data[data[split_col] == data[group_split_col]].copy()

        # Can now drop id_split since it's now always equal to split
        data.drop(group_split_col, axis=1, inplace=True)
    else:
        is_split = data[[split_col, group_split_col]] == filter_strategy
        remove_cond = is_split.any(axis=1) & ~is_split.all(axis=1)
        data = data[~remove_cond].copy()

    return data, transitions

STRATEGY_TO_METHOD = {
    'random': '_random_split',
    'grouped': '_grouped_split',
    'temporal': '_temporal_split',
    'grouped_temporal': '_grouped_temporal_split'
}

STRATEGY_IS_GROUPED = {
    'random': False,
    'grouped': True,
    'temporal': False,
    'grouped_temporal': True
}

STRATEGY_IS_TEMPORAL = {
    'random': False,
    'grouped': False,
    'temporal': True,
    'grouped_temporal': True
}

class DatasetSplitter:
    def __init__(
        self,
        strategy: str,
        fractions: Union[float, List[float]], 
        split_labels: List[str],
        group_col: Optional[str] = None, 
        date_col: Optional[str] = None,
        split_col: str = 'split',
        **split_fn_kwargs,
    ):
        """
        Initialize the DatasetSplitter with a specific strategy, fractions for splits, 
        optional split names, and optional grouping/date columns.

        Parameters
        ----------
        strategy : str
            The strategy to use for splitting the dataset.
        fractions : float or List[float]
            Fractions of the dataset to allocate to each split.
        split_labels : List[str], optional
            Labels for each of the splits, used for labeling the split in the DataFrame.
        group_col : str, optional
            The column name used for grouping in grouped strategies.
        date_col : str, optional
            The column name representing dates in temporal strategies.
        """
        self.strategy = strategy
        self.fractions = process_fractions(fractions)
        self.split_labels = split_labels
        self.group_col = group_col
        self.date_col = date_col
        self.split_col = split_col
        self.split_fn_kwargs = split_fn_kwargs

        # Verify requirements for the selected strategy
        self._verify_init()

    @property
    def is_grouped(self):
        """Check if the strategy requires a grouping column."""
        return STRATEGY_IS_GROUPED[self.strategy]

    @property
    def is_temporal(self):
        """Check if the strategy requires a date column."""
        return STRATEGY_IS_TEMPORAL[self.strategy]

    def _verify_init(self):
        if self.strategy not in STRATEGY_IS_GROUPED:
            raise ValueError(f'Unrecognized strategy {self.strategy}.')

        if self.is_grouped:
            if self.group_col is None:
                raise ValueError(f"`group_col` is required for strategy '{self.strategy}'.")
        else:
            if self.group_col is not None:
                raise ValueError(f"Cannot specify `group_col` for strategy '{self.strategy}'.")

        if self.is_temporal:
            if self.date_col is None:
                raise ValueError(f"`date_col` is required for strategy '{self.strategy}'.")
        else:
            if self.date_col is not None:
                raise ValueError(f"Cannot specify `date_col` for strategy '{self.strategy}'.")

        if len(self.fractions) != len(self.split_labels):
            raise ValueError(
                f"Length of fractions ({self.fractions}) does not match that of split "
                f"names ({self.split_labels})"
            )

    def _verify_data(self, data):
        check_cols(data, self.split_col, raise_err_on_existing=True)

        if self.is_grouped:
            check_cols(data, self.group_col, raise_err_on_missing=True)
            
            group_na = data[self.group_col].isna()
            if group_na.any():
                data = data[~group_na].copy()
                logging.warning(f'Dropped {group_na.sum()} rows having NaN groups.')

        if self.is_temporal:
            check_cols(data, self.date_col, raise_err_on_missing=True)

        if self.strategy == 'grouped_temporal':
            check_cols(data, f'{self.group_col}_{self.split_col}', raise_err_on_existing=True)

        return data

    def __call__(self, data: pd.DataFrame):
        """
        Split the dataset according to the specified strategy.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to split.

        Returns
        -------
        pd.DataFrame
            DataFrame with an additional column(s) indicating split assignments.
        """
        data = self._verify_data(data)

        method_name = STRATEGY_TO_METHOD[self.strategy]
        method = getattr(self, method_name)
        return method(data, **self.split_fn_kwargs)

    def _random_split(self, data: pd.DataFrame, **kwargs):
        """Perform a random split."""
        return split_dataframe_col(
            data,
            self.fractions,
            labels=self.split_labels,
            split_col=self.split_col,
            **kwargs,
        )

    def _grouped_split(self, data: pd.DataFrame, **kwargs):
        """Perform a grouped split based on the group column."""
        data[self.split_col] = grouped_splits(
            data[self.group_col].values,
            self.fractions,
            labels=self.split_labels,
            **kwargs,
        )

        return data

    def _temporal_split(self, data: pd.DataFrame, **kwargs):
        """Perform a temporal split based on the date column."""
        data, transitions = temporal_splits(
            data,
            self.fractions,
            self.date_col,
            labels=self.split_labels,
            split_col = self.split_col,
            **kwargs,
        )
        logging.info(f'Date transitions used: {transitions}.')

        return data

    def _grouped_temporal_split(self, data: pd.DataFrame, **kwargs):
        """Perform a temporal split that also considers groups."""
        data, transitions = grouped_temporal_splits(
            data,
            self.fractions,
            self.group_col,
            self.date_col,
            labels=self.split_labels,
            split_col = self.split_col,
            **kwargs,
        )
        logging.info(f'Date transitions used: {transitions}.')

        return data
