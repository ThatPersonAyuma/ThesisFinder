import math
import re
from collections import Counter, defaultdict

# Bobot default per field: judul lebih informatif daripada abstrak.
DEFAULT_FIELD_WEIGHTS = {"title": 3.0, "abstract": 1.0}
SINGLE_FIELD = "body"


class BM25F:
    """Okapi BM25F: BM25 dengan pembobotan dan normalisasi panjang per field.

    Dokumen dapat berupa string (diperlakukan sebagai satu field ``body``) atau
    dict ``{field: teks}``, misalnya ``{"title": "...", "abstract": "..."}``.

    Rumus yang dipakai::

        tf~(t, d) = sum_f w_f * tf(t, d, f) / (1 - b_f + b_f * |d_f| / avgdl_f)
        skor(t, d) = idf(t) * tf~(t, d) * (k1 + 1) / (k1 + tf~(t, d))

    dengan ``idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)`` dan df dihitung
    dari jumlah dokumen yang memuat term di field mana pun.
    """

    def __init__(self, documents, k1=1.2, b=0.75, field_weights=None):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.fields = self._detect_fields(documents)

        # b boleh berupa satu nilai untuk semua field atau dict per field.
        if isinstance(b, dict):
            self.field_b = {f: float(b.get(f, 0.75)) for f in self.fields}
        else:
            self.field_b = {f: float(b) for f in self.fields}

        weights = DEFAULT_FIELD_WEIGHTS if field_weights is None else field_weights
        self.field_weights = {f: float(weights.get(f, 1.0)) for f in self.fields}

        # Tokenisasi per field. Field yang tidak ada pada sebuah dokumen dianggap kosong.
        self.tokenized_docs = {
            f: [
                self._tokenize(doc.get(f) or "") if isinstance(doc, dict) else self._tokenize(doc)
                for doc in documents
            ]
            for f in self.fields
        }

        # Panjang dan rata-rata panjang dokumen dihitung terpisah untuk tiap field.
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

        # Gabungan bobot field dan normalisasi panjang: tf~ = sum(boost * tf).
        self.field_boost = {
            f: [
                self.field_weights[f] / norm if norm > 0 else 0.0
                for norm in self.doc_norm[f]
            ]
            for f in self.fields
        }

        # Panjang total (semua field) tetap tersedia untuk inspeksi/kompatibilitas.
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

        # df menghitung dokumen sekali, meskipun term muncul di beberapa field.
        self.doc_freq = Counter({term: len(postings) for term, postings in self.index.items()})
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.doc_freq.items()
        }

    @staticmethod
    def _detect_fields(documents):
        """Tentukan daftar field dari dokumen (union key, urutan kemunculan)."""
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
        """tf~: jumlah tf tiap field yang sudah dibobot dan dinormalisasi panjangnya."""
        return sum(
            self.field_boost[field][doc_id] * freq
            for field, freq in field_tfs.items()
        )

    def _term_score(self, idf, doc_id, field_tfs):
        tf = self._weighted_tf(doc_id, field_tfs)
        if tf <= 0:
            return 0.0
        return idf * (tf * (self.k1 + 1)) / (tf + self.k1)

    def score(self, query, doc_id):
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
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)

        # Inverted index traversal: hanya hitung dokumen yang mengandung kata kunci.
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


# Nama lama dipertahankan agar kode dan pengujian yang sudah ada tetap berjalan.
BM25 = BM25F
