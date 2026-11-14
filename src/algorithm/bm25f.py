"""Okapi BM25F: BM25 dengan bobot dan normalisasi panjang per field.

Dokumen dapat berupa string (satu field ``body``) atau dict ``{field: teks}``.
"""

import math
import re
from collections import Counter, defaultdict

# Bobot default: judul lebih informatif daripada abstrak.
DEFAULT_FIELD_WEIGHTS = {"title": 3.0, "abstract": 1.0}
SINGLE_FIELD = "body"


class BM25F:
    """Peringkat dokumen multi-field dengan Okapi BM25F.

    ``documents`` berupa string (field ``body``) atau dict ``{field: teks}``.
    ``k1`` mengatur saturasi tf, ``b`` normalisasi panjang (boleh per field),
    dan ``field_weights`` bobot field (default: title 3.0, abstract 1.0).
    """

    def __init__(self, documents, k1=1.2, b=0.75, field_weights=None):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.fields = self._detect_fields(documents)

        # b dan bobot dapat diatur per field.
        if isinstance(b, dict):
            self.field_b = {f: float(b.get(f, 0.75)) for f in self.fields}
        else:
            self.field_b = {f: float(b) for f in self.fields}

        weights = DEFAULT_FIELD_WEIGHTS if field_weights is None else field_weights
        self.field_weights = {f: float(weights.get(f, 1.0)) for f in self.fields}

        # Tokenisasi dan panjang dokumen dihitung terpisah untuk setiap field.
        self.tokenized_docs = {
            f: [
                self._tokenize(doc.get(f) or "") if isinstance(doc, dict) else self._tokenize(doc)
                for doc in documents
            ]
            for f in self.fields
        }
        self.field_doc_length = {
            f: [len(tokens) for tokens in self.tokenized_docs[f]] for f in self.fields
        }
        self.field_avgdl = {
            f: (sum(lengths) / self.N) if self.N > 0 else 0.0
            for f, lengths in self.field_doc_length.items()
        }

        # Normalisasi panjang BM25 per field.
        self.doc_norm = {
            f: [
                (1 - self.field_b[f] + self.field_b[f] * (dl / self.field_avgdl[f]))
                if self.field_avgdl[f] > 0
                else 1.0
                for dl in self.field_doc_length[f]
            ]
            for f in self.fields
        }

        # Gabungkan bobot field dengan normalisasi panjang agar tf~ cukup
        # dihitung sebagai sum(field_boost * tf).
        self.field_boost = {
            f: [
                self.field_weights[f] / norm if norm > 0 else 0.0
                for norm in self.doc_norm[f]
            ]
            for f in self.fields
        }

        # Panjang total semua field untuk inspeksi/kompatibilitas.
        self.doc_length = [
            sum(self.field_doc_length[f][i] for f in self.fields)
            for i in range(self.N)
        ]
        self.avgdl = (sum(self.doc_length) / self.N) if self.N > 0 else 0.0

        # Inverted index: term -> {doc_id: {field: tf}}.
        self.index = defaultdict(dict)
        for f in self.fields:
            for doc_id, tokens in enumerate(self.tokenized_docs[f]):
                for term, freq in Counter(tokens).items():
                    self.index[term].setdefault(doc_id, {})[f] = freq

        # df dihitung per dokumen, bukan per field.
        self.doc_freq = Counter({term: len(postings) for term, postings in self.index.items()})
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.doc_freq.items()
        }

    @staticmethod
    def _detect_fields(documents):
        """String -> field ``body``; dict -> union semua key."""
        if not documents:
            return []
        if isinstance(documents[0], dict):
            fields = []
            for doc in documents:
                if not isinstance(doc, dict):
                    raise TypeError("Semua dokumen harus bertipe sama: dict field atau string.")
                for field in doc:
                    if field not in fields:
                        fields.append(field)
            return fields
        if any(not isinstance(doc, str) for doc in documents):
            raise TypeError("Semua dokumen harus bertipe sama: dict field atau string.")
        return [SINGLE_FIELD]

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def _weighted_tf(self, doc_id, field_tfs):
        """tf~ = sum_f w_f * tf_f / (1 - b_f + b_f * |d_f| / avgdl_f)."""
        return sum(
            self.field_boost[field][doc_id] * freq
            for field, freq in field_tfs.items()
        )

    def _term_score(self, idf, doc_id, field_tfs):
        """Kontribusi satu term: idf * tf~ * (k1 + 1) / (k1 + tf~)."""
        tf = self._weighted_tf(doc_id, field_tfs)
        if tf <= 0:
            return 0.0
        return idf * (tf * (self.k1 + 1)) / (tf + self.k1)

    def score(self, query, doc_id):
        """Skor BM25F dokumen ``doc_id`` terhadap ``query``."""
        query_tokens = self._tokenize(query)
        total_score = 0.0

        for term in query_tokens:
            postings = self.index.get(term)
            if not postings:
                continue
            field_tfs = postings.get(doc_id)
            if field_tfs:
                total_score += self._term_score(self.idf[term], doc_id, field_tfs)

        return total_score

    def search(self, query, top_k=None):
        """Kembalikan [(doc_id, skor)] terurut menurun; skor 0 tidak disertakan."""
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)

        # Traversal inverted index: hanya hitung dokumen yang memuat term query.
        for term in query_tokens:
            postings = self.index.get(term)
            if not postings:
                continue

            term_idf = self.idf[term]
            for doc_id, field_tfs in postings.items():
                scores[doc_id] += self._term_score(term_idf, doc_id, field_tfs)

        results = sorted(
            ((doc_id, score) for doc_id, score in scores.items() if score > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        if top_k is not None:
            return results[:top_k]
        return results


# Alias lama agar pemakaian BM25 satu field tetap berjalan.
BM25 = BM25F
