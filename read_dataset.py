import pandas as pd

chunks = pd.read_json('D:/Github/TF2RNN/dataset.json', orient='records', lines=True, precise_float=True, chunksize=32)

for batch in chunks:
    