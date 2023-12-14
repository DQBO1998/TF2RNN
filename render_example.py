from sentence_transformers import SentenceTransformer
from from_corpus import make_dataset
from matplotlib import pyplot as plt
import numpy as np
import hypertools as hyp

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

uid, X, Y = make_dataset(sentences, lambda txt: model.encode(txt))
print(uid.shape, X.shape, Y.shape)
hyp.plot(x=Y, hue=uid, legend=sentences)
plt.show()