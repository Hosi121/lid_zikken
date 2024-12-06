import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec
from nltk.corpus import reuters
import nltk

nltk.download('reuters')
nltk.download('punkt')

# ===== コーパス準備 =====
sentences = []
for fileid in reuters.fileids():
    text = reuters.raw(fileid)
    tokens = nltk.word_tokenize(text)
    sentences.append(tokens)

# ===== Word2Vec学習 =====
w2v_model = Word2Vec(sentences, vector_size=300, window=5, min_count=5, workers=4, sg=0)
words = list(w2v_model.wv.index_to_key)
embedding_matrix = np.array([w2v_model.wv[word] for word in words])
num_vectors = embedding_matrix.shape[0]
dim = embedding_matrix.shape[1]

print(f"Vocabulary Size: {num_vectors}, Embedding Dim: {dim}")

# LID計算用関数
def compute_lid(distances, k=5, eps=1e-12):
    distances = np.maximum(distances, eps)
    d_k = distances[-1]
    ratio = d_k / distances[:-1]
    lid_value = 1.0 / ((1.0/(k-1)) * np.sum(np.log(ratio)))
    return lid_value

k = 5
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embedding_matrix)
distances, indices = nbrs.kneighbors(embedding_matrix)

lid_values = []
for i in range(num_vectors):
    d = distances[i,1:]
    lid_val = compute_lid(d, k=k)
    if np.isinf(lid_val) or np.isnan(lid_val):
        lid_val = 0.0
    lid_values.append(lid_val)

lid_values = np.array(lid_values)

# CSV保存
df = pd.DataFrame({"word": words, "LID": lid_values})
df.to_csv("reuters_w2v_lid_results.csv", index=False)
print("LID結果を 'reuters_w2v_lid_results.csv' に保存しました。")

# 結果表示
plt.figure(figsize=(10,6))
plt.hist(lid_values, bins=50, edgecolor='black')
plt.title("LID Distribution for Reuters W2V Embeddings")
plt.xlabel("LID value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

harmonic_mean = len(lid_values) / np.sum(1.0/lid_values[np.nonzero(lid_values)])
print("Harmonic Mean of LID:", harmonic_mean)
