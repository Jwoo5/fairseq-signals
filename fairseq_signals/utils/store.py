from typing import Any, Callable, Dict, Optional, Tuple
import os

import pickle

import numpy as np

import fairseq_signals.distributed.utils as dist_utils

def normalize_ext(ext: Optional[str], include_period: bool = True) -> Optional[str]:
    """Normalize the format of a file extension.

    Args:
        ext (str): The file extension.
        include_period (bool) : Whether to include a period in the returned extension.

    Returns:
        str or None: Processed file extension with or without a period, or None if ext is None.
    """
    if ext is None:
        return None

    if ext.startswith("."):
        if include_period:
            # Has period and we want the period
            return ext
        # Has period and we don't want the period
        return ext[1:]

    if include_period:
        # Doesn't have period and we want the period
        return "." + ext

    # Doesn't have period and we don't want the period
    return ext


def has_ext(
    file: Optional[str],
    ext: Optional[str],
    normalize: bool = True,
    case: bool = False,
    raise_err: bool = True,
) -> bool:
    """Check if a given file path has the given extension.

    Note that it will match on substrings, e.g., on ".gz" in ".nii.gz".

    Args:
        file (str): The file path to check. If None, True is returned.
        ext (str): The file extension to match. If None, True is returned.
        normalize (bool): Whether to normalize the extension to include a period.
        case (bool): Whether extension matching is case-sensitive.

    Returns:
        bool: Whether the file has the given extension if ext is provided, otherwise True.
    """
    if file is None:
        return True

    if ext is None:
        return True

    if normalize:
        ext = normalize_ext(ext, include_period=True)

    if case:
        return file.endswith(ext)

    has = file.lower().endswith(ext.lower())

    if raise_err and not has:
        raise ValueError(f"File {file} must have extension '{ext}'.")

    return has


class MemmapReader:
    """Wrapper for reading memory-mapped arrays.

    Enables reading a memmap keyword arguments by optionally

    Args:
        file (str): The file path for the memory-mapped store, should end with ".npy".
        shape (tuple): The shape of the memory-mapped array.
        dtype (str): The data type of the memory-mapped array.
        **memmap_kwargs (Any): Additional keyword arguments for np.memmap.

    Attributes:
        file (str): The file path for the memory-mapped store.
        shape (tuple): The shape of the memory-mapped array.
        dtype (str): The data type of the memory-mapped array.
        _read_array (np.ndarray): The memory-mapped array used for reading data.

    Raises:
        ValueError: If the file doesn't have a ".npy" extension.

    Notes:
        If data doesn't appear to be reading correctly, ensure that the
        data type being read matches that of `dtype`.
    """
    def __init__(
        self,
        file: str,
        shape: Tuple,
        dtype: str = "float32",
        **memmap_kwargs,
    ):
        has_ext(file, ".npy", raise_err=True)

        self.file = file
        self.shape = shape
        self.dtype = dtype

        if "mode" in memmap_kwargs:
            raise ValueError(f"Cannot pass mode when initializing {self.__class__.__name__}.")

        # Create memory-mapped array for reading
        self._read_array = np.memmap(
            file,
            mode='r',
            shape=shape,
            dtype=dtype,
            **memmap_kwargs,
        )


    @staticmethod
    def file_to_default_header(file: str) -> str:
        """Generate the default header file name.

        Args:
            file (str): The file path.

        Returns:
            str: The default header file name.
        """
        return file[:-4] + "_header.pkl"

    @staticmethod
    def _from_header(
        file: str,
        header_file: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Load metadata from a header file.

        Args:
            file (str): The file path.
            header_file (str, optional): The header file path.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing the file path and metadata dictionary.

        Raises:
            ValueError: If the header file is not found.
        """
        has_ext(file, ".npy", raise_err=True)
        has_ext(header_file, ".pkl", raise_err=True)

        header_file = header_file or MemmapReader.file_to_default_header(file)

        if not os.path.isfile(header_file):
            raise ValueError(f"Could not find header file {header_file}.")

        with open(header_file, 'rb') as hfile:
            return file, pickle.load(hfile)


    @classmethod
    def from_header(
        cls,
        file: str,
        header_file: Optional[str] = None,
    ):
        """Create the class instance from a header file.

        Args:
            file (str): The file path.
            header_file (str, optional): The header file path.
        """
        file, memmap_kwargs = cls._from_header(file, header_file=header_file)
        return cls(file, **memmap_kwargs)


    @property
    def array(self) -> np.ndarray:
        """Get the memory-mapped array for reading.

        Returns:
            np.ndarray: The memory-mapped array for reading.
        """
        return self._read_array


    def __getitem__(self, key: Any) -> Any:
        """Get items from the memory-mapped array.

        Args:
            key (Any): The key to access the array.

        Returns:
            Any: The item(s) from the array.
        """
        return self.array[key]


    def __repr__(self) -> str:
        return self.array.__repr__()


    def __str__(self) -> str:
        return self.array.__str__()


    def __len__(self) -> str:
        return self.shape[0]


class MemmapBatchWriter(MemmapReader):
    """Wrapper for iteratively writing to a memory-mapped array.

    Provides functionality to iteratively write over the first dimension in batches.

    Args:
        file (str): The file path for the memory-mapped store, should end with ".npy".
        shape (tuple): The shape of the memory-mapped array.
        transform (Callable, optional): A function to transform data before writing to the store.
        header_file (str, optional): The header file path.
        dtype (str, default "float32"): The data type of the memory-mapped array.
        **memmap_kwargs (Any): Additional keyword arguments for np.memmap.

    Attributes:
        transform (Callable, optional): A function to transform data before writing to the store.
            It should return a NumPy array with data type `dtype`.
        n (int): Number of samples written to the array.
        total_n (int): Number of samples which can be written to the array (the first dimension).
        _write_array (np.ndarray): The memory-mapped array used for writing data.
        _is_closed (bool): A flag indicating whether the memory-mapped write array is closed.

    Raises:
        ValueError: If the file doesn't have a ".npy" extension.


    Notes:
        If data doesn't appear to be writing or writing incorrectly, ensure that the
        data type being written is consistent and matches that of `dtype`.
    """
    def __init__(
        self,
        file: str,
        shape: Tuple,
        transform: Optional[Callable[[Any], np.ndarray]] = None,
        header_file: Optional[str] = None,
        dtype: str = "float32",
        **memmap_kwargs,
    ):
        has_ext(header_file, ".pkl", raise_err=True)

        self.total_n = shape[0]

        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

        # Create memory-mapped array for writing
        self._write_array = np.memmap(
            file,
            mode='w+',
            shape=shape,
            dtype=dtype,
            **memmap_kwargs,
        )

        # Call super init to set attributes and create _read_array
        super().__init__(file, shape, dtype, **memmap_kwargs)

        self.n = 0
        self._is_closed = False

        # Save header file
        header = {
            "shape": shape,
            "dtype": dtype,
            **memmap_kwargs,
        }
        self.save_header(header_file, **header)

    @classmethod
    def from_header(
        cls,
        file: str,
        header_file: Optional[str] = None,
        transform: Optional[Callable[[Any], np.ndarray]] = None,
    ):
        """ Create the class instance from a header file.

        Args:
            file (str): The file path.
        header_file (str, optional): The header file path.
        transform : (Callable, optional): A function to transform data before writing to the store.
        """
        file, memmap_kwargs = cls._from_header(file, header_file=header_file)
        return cls(file, transform=transform, **memmap_kwargs)


    def save_header(
        self,
        header_file: Optional[str] = None,
        **memmap_kwargs,
    ) -> None:
        """ Save header information to a header file.

        Args:
            header_file (str, optional): The header file path.
        **memmap_kwargs (Any): Additional keyword arguments for np.memmap.
        """
        header_file = header_file or self.file_to_default_header(self.file)
        with open(header_file, 'wb') as hfile:
            pickle.dump(memmap_kwargs, hfile)


    def __call__(self, batch: Any) -> None:
        """ Write a batch of data to the memory-mapped store.

        Args:
            batch (Any): The data batch to write.

        Raises:
            ValueError: If attempting to write more data than the store's capacity.
        """
        self.is_closed(raise_err=True)

        new_n = self.n + len(batch)
        if new_n > self.total_n:
            raise ValueError(
                f"Can only write {self.total_n} samples. Already wrote {self.n} "
                f"samples, where the batch size being written is {len(batch)}."
            )

        # Apply the transform function and write the batch to the store
        self._write_array[self.n: new_n] = self.transform(batch)
        self.n = new_n


    def close(self) -> None:
        """Close the memory-mapped store and release resources.
        """
        del self._write_array
        self._is_closed = True


    def is_closed(self, raise_err: bool = False) -> bool:
        """Check if the memory-mapped write array has been closed.

        Args:
            raise_err (bool, optional): Whether to raise an error if closed.

        Returns:
            bool: True if the store is closed, False otherwise.

        Raises:
            ValueError: If the write array has been closed and `raise_err` is True.
        """
        if self._is_closed and raise_err:
            raise ValueError(f"{self.__class__.__name__} is closed.")

        return self._is_closed

def initialize_store(
    dtype,
    save_file,
    shape,
    save_directory=None,
):
    # Handle stores
    if (
        dist_utils.get_data_parallel_world_size() == 1
        or dist_utils.get_data_parallel_rank() == 0
    ):
        if save_directory is not None:
            save_path = os.path.join(save_directory, save_file)

        store = MemmapBatchWriter(
            save_path,
            shape,
            dtype=dtype,
            transform=lambda batch: batch.detach().cpu().numpy(),
        )

        return store

    return None

def initialize_stores(
    dtype,
    store_id,
    outputs_shape,
    targets_shape,
    save_directory=None,
):
    # Handle stores
    if (
        dist_utils.get_data_parallel_world_size() == 1
        or dist_utils.get_data_parallel_rank() == 0
    ):
        outputs_path = f'outputs_{store_id}.npy'
        output_store = initialize_store(
            dtype,
            outputs_path,
            outputs_shape,
            save_directory=save_directory,
        )
        targets_path = f'targets_{store_id}.npy'
        target_store = initialize_store(
            dtype,
            targets_path,
            targets_shape,
            save_directory=save_directory,
        )

        return output_store, target_store

    return None

def initialize_stores_to_criterion(
    dtype,
    criterion,
    store_id,
    outputs_shape,
    targets_shape,
    save_directory=None,
):
    stores = initialize_stores(
        dtype,
        store_id,
        outputs_shape,
        targets_shape,
        save_directory=save_directory,
    )

    if stores is not None:
        output_store, target_store = stores
        criterion.set_output_store(output_store)
        criterion.set_target_store(target_store)

def store(
    store: MemmapBatchWriter,
    values: Any,
):
    if dist_utils.get_data_parallel_world_size() > 1:
        group = dist_utils.get_data_parallel_group()
        values = torch.cat(dist_utils.batch_all_gather(values, group=group))

    store(values)
