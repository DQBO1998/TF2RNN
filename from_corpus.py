from itertools import product
from typing import Callable, Sequence
import pandas as pd
import numpy as np

def _prev_next_pair(text: str):
    return [[text[:i], text[:i + 1]] for i in range(len(text))]

def _make_trajs(corpus: Sequence[str]):
    nested = list(map(_prev_next_pair, corpus))
    flat = [[uid, *pair] for uid, nest in enumerate(nested) for pair in nest]
    return flat

def _encode_trajs(trajs: Sequence[list], fn: Callable):
    txt = list({t1 for _, t1, _ in trajs} | {t2 for _, _, t2 in trajs})
    Z = fn(txt)
    dims = Z.shape[1]
    txt_to_Z = {t: Z[i] for i, t in enumerate(txt)}
    X = np.zeros((len(trajs), dims))
    Y = np.zeros((len(trajs), dims))
    UID = np.zeros((len(trajs), 1), dtype=int)
    for num, (uid, t1, t2) in enumerate(trajs):
        X[num] = txt_to_Z[t1]
        Y[num] = txt_to_Z[t2]
        UID[num] = uid
    return UID, X, Y

def make_dataset(corpus: Sequence[str], fn: Callable, to_numpy: bool = None, to_pandas: bool = None):
    assert not (to_numpy and to_pandas)

    if to_numpy is None and to_pandas is None:
        to_numpy = True

    trajs = _make_trajs(corpus)
    UID, X, Y = _encode_trajs(trajs=trajs, fn=fn)
    if to_numpy:
        return UID.reshape(-1), X, Y
    heads = ['UID', *[f'X{d}' for d in range(X.shape[1])], *[f'Y{d}' for d in range(Y.shape[1])]]
    D = np.concatenate([UID, X, Y], axis=1)
    return pd.DataFrame(data=D, columns=heads)