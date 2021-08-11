from pathlib import Path
from typing import BinaryIO, Tuple, Union, List

import numpy as np
import torch

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
            [np.array(
                [float(v) for v in row[1:]]
            ) for row in table[1:]]
        )
        
        return rows, cols, values
    
    rows, cols, values = load_table()

    rows = [set(row.split('|')) for row in rows]
    cols = [set(col.split('|')) for col in cols]
    assert (rows == cols)

    return rows, values

def compute_scored_confusion_matrix(
    weights: np.ndarray,
    labels: np.ndarray,
    outputs: np.ndarray,
):
    norms = np.sum(
        np.any((labels, outputs), axis=0),
        axis=1
    )
    norms = [float(max(x, 1)) for x in norms]

    scores = np.zeros((outputs.shape[-1], outputs.shape[-1]))
    for i, norm in enumerate(norms):
        trg_indices = np.where(labels[i] == 1)[0]
        out_indices = np.where(outputs[i] == 1)[0]
        for trg_idx in trg_indices:
            scores[trg_idx, out_indices] += 1.0 / norm
    scores *= weights
    score = np.sum(scores)

    return score

def get_sinus_rhythm():
    return set(['426783006'])

def get_sinus_rhythm_index(classes = None):
    if classes:
        if get_sinus_rhythm() in classes:
            return classes.index(get_sinus_rhythm())
    else:
        return 14