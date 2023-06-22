import os
import pickle
import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from tqdm.notebook import tqdm

def make_bm25_embedding(DATA_DIR, tokenizer, full_ds, context, test=False):
	bm25 = BM25Okapi(context, tokenizer=tokenizer.tokenize)

	train_iterator = tqdm(full_ds['question'], desc="BM25 Embedding")
	bm25_embedding = []
	for query in train_iterator:
		scores = bm25.get_scores(tokenizer.tokenize(query))
		bm25_embedding.append(scores)

	bm25_embedding = np.array(bm25_embedding)

	if not test:
		with open(os.path.join(DATA_DIR, 'wiki_bm25_embedding.bin'), 'wb') as f:
			pickle.dump(bm25_embedding, f)
	else:
		with open(os.path.join(DATA_DIR, 'test_wiki_bm25_embedding.bin'), 'wb') as f:
			pickle.dump(bm25_embedding, f)

	return pd.DataFrame(bm25_embedding, index=full_ds['id'])
