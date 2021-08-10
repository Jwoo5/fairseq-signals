from pathlib import Path
from typing import BinaryIO, Tuple, Union, List

import numpy as np

def get_physionet_weights(path_or_fp: Union[str, BinaryIO]) -> Tuple[List[set], np.ndarray]:
    def load_table():
        if isinstance(path_or_fp, str):
            ext = Path(path_or_fp).suffix
            if ext != '.csv':
                raise ValueError(f"Unsupported weights table format: {ext}")

        table = list()
        with open(path_or_fp, 'r') as f:
            for line in f:
                arrs = [arr.strip() for arr in line.split(',')]
                table.append(arrs)

        rows = table[0][1:]
        cols = [table[i+1][0] for i in range(len(rows))]

        assert (rows == cols)

        values = np.stack(
            np.array(
                [float(v) for v in row[1:]]
            ) for row in table[1:]
        )
        
        return rows, cols, values
    
    rows, cols, values = load_table()

    rows = [set(row.split('|')) for row in rows]
    cols = [set(col.split('|')) for col in cols]
    assert (rows == cols)

    return rows, values