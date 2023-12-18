from typing import Callable
from datasets import load_dataset
from from_corpus import make_dataset as trajectories
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from time import time
from datetime import timedelta
import argparse
import numpy as np
import os

def preprocess_batch(batch, fn: Callable):
    corpus = []

    for sentences in batch['set']:
        corpus += sentences

    uid, X, Y = trajectories(corpus=corpus, fn=fn, to_numpy=True)
    return tuple((x, y) for x, y in zip(X, Y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run this script to produce embeddings from a sentence transformer.')
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2', help='Full name of a sentence transformer.')
    parser.add_argument('--batch', type=int, default=1, help='Batch-size to split the dataset into.')
    parser.add_argument('--outdir', type=str, default=os.path.dirname(os.path.abspath(__file__)) + '/dataset/', help='Output directory.')
    parser.add_argument('--cuda', type=int, default=1, help='Use cuda? 1=yes 0=no')
    args = parser.parse_args()
    print(args)
    model = SentenceTransformer(args.model, device = 'cuda' if args.cuda == 1 else None)
    loader = load_dataset('embedding-data/sentence-compression', 
                        split='train')
    diffs = []
    uid = 0
    for i in (bar := tqdm(range(0, len(loader), args.batch))):
        T1 = time()
        batch = preprocess_batch(batch=loader[i:i + args.batch], fn=lambda txt: model.encode(txt))
        for x, y in batch:
            uid += 1
            np.savez(args.outdir + f'{uid}.npz', x=x, y=y)
        T2 = time()
        diffs.append(T2 - T1)
        if len(diffs) > 100:
            diffs.pop(0)
        remaining = len(loader) - (i + args.batch)
        expremtime = timedelta(seconds=remaining * sum(diffs) / len(diffs))
        bar.set_description(f'ERT: {expremtime}')

        if i > 1000:
            break
    