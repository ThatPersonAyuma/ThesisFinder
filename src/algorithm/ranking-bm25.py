import math
import re
from collections import Counter, defaultdict


class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)

        self.tokenized_docs = [self._tokenize(d) for d in documents]
        self.doc_length = [len(d) for d in self.tokenized_docs]
        self.avgdl = sum(self.doc_length) / self.N

        self.index = defaultdict(list)
        self.doc_freq = Counter()
        for doc_id, tokens in enumerate(self.tokenized_docs):
            tf = Counter(tokens)
            for term, freq in tf.items():
                self.index[term].append((doc_id, freq))
                self.doc_freq[term] += 1
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.doc_freq.items()
        }

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def score(self, query, doc_id):
        query_tokens = re.findall(r"\w+", query.lower())
        total_score = 0

        for term in query_tokens:
            if term not in self.idf:
                continue

            term_idf = self.idf[term]
            tf = 0

            for d_id, freq in self.index.get(term, []):
                if d_id == doc_id:
                    tf = freq
                    break

            norm = 1 - self.b + self.b * (self.doc_length[doc_id] / self.avgdl)
            tf_component = (tf * (self.k1 + 1)) / (tf * self.k1 * norm)
            total_score += term_idf * tf_component

        return total_score

    def search(self, query, top_k=None):
        lis = []
        for doc_id in range(self.N):
            s = self.score(query, doc_id)
            lis.append((doc_id, s))
        lis.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            return list[:top_k]
        return lis
