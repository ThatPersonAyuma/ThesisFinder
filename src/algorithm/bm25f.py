"""Implementasi algoritma peringkat Okapi BM25F.

Modul ini menyediakan kelas :class:`BM25F`, yaitu perluasan BM25 untuk dokumen
yang memiliki beberapa field (misalnya ``title`` dan ``abstract``). Setiap field
dapat diberi bobot berbeda dan dinormalisasi panjangnya secara terpisah sebelum
digabungkan menjadi satu skor relevansi.

Rumus
-----
Untuk term ``t`` pada dokumen ``d``::

    tf~(t, d) = sum_f w_f * tf(t, d, f) / (1 - b_f + b_f * |d_f| / avgdl_f)
    skor(t, d) = idf(t) * tf~(t, d) * (k1 + 1) / (k1 + tf~(t, d))
    idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

Keterangan:
    - ``N``: jumlah dokumen dalam korpus.
    - ``df(t)``: jumlah dokumen yang memuat ``t`` di field mana pun.
    - ``tf(t, d, f)``: frekuensi ``t`` pada field ``f`` dokumen ``d``.
    - ``|d_f|``: panjang field ``f`` dokumen ``d`` (jumlah token).
    - ``avgdl_f``: rata-rata panjang field ``f``.
    - ``w_f``: bobot field ``f``.
    - ``b_f``: parameter normalisasi panjang field ``f``.
    - ``k1``: parameter saturasi frekuensi term.

BM25 vs BM25F
-------------
BM25 biasa menggabungkan seluruh isi dokumen menjadi satu field, sehingga
kecocokan pada judul dan abstrak mendapat bobot yang sama. BM25F menghitung
``tf`` per field, menormalkan panjang tiap field, lalu menjumlahkannya dengan
bobot field. Dengan begitu, kecocokan pada judul dapat diberi prioritas.

Contoh pemakaian
----------------
::

    from src.algorithm.bm25f import BM25F

    dokumen = [
        {"title": "Graph Neural Networks", "abstract": "Molecular property prediction."},
        {"title": "Image Classification", "abstract": "Deep learning for computer vision."},
    ]
    bm25f = BM25F(dokumen)
    hasil = bm25f.search("graph neural network", top_k=5)
    # hasil: [(doc_id, skor), ...] terurut dari skor tertinggi

Dokumen juga boleh berupa string biasa. Dalam kasus itu seluruh teks
diperlakukan sebagai satu field bernama ``body`` sehingga perilakunya setara
dengan BM25 satu field.
"""

import math
import re
from collections import Counter, defaultdict

# Bobot default per field. Judul diberi bobot lebih besar karena biasanya lebih
# padat informasi dan lebih menggambarkan topik dokumen daripada abstrak.
DEFAULT_FIELD_WEIGHTS = {"title": 3.0, "abstract": 1.0}

# Nama field yang dipakai saat dokumen diberikan sebagai string biasa.
SINGLE_FIELD = "body"


class BM25F:
    """Model peringkat Okapi BM25F untuk dokumen multi-field.

    Parameters
    ----------
    documents : list[str] | list[dict[str, str]]
        Korpus. Setiap dokumen berupa string (satu field ``body``) atau dict
        ``{nama_field: teks}``. Semua dokumen harus bertipe sama. Field yang
        tidak ada pada suatu dokumen dianggap kosong.
    k1 : float, default 1.2
        Parameter saturasi frekuensi term. Nilai lebih besar membuat kenaikan
        ``tf`` semakin cepat mendatar.
    b : float | dict[str, float], default 0.75
        Parameter normalisasi panjang. Satu nilai berlaku untuk semua field,
        atau dict per field, misalnya ``{"title": 0.5, "abstract": 0.75}``.
        ``b=0`` mematikan normalisasi panjang.
    field_weights : dict[str, float] | None, default None
        Bobot tiap field. Jika ``None``, field ``title`` berbobot 3.0 dan
        ``abstract`` 1.0, sedangkan field lain berbobot 1.0. Bobot menentukan
        seberapa besar kontribusi kemunculan term pada field tersebut.

    Attributes
    ----------
    fields : list[str]
        Nama field yang ditemukan pada korpus, sesuai urutan kemunculan.
    index : dict[str, dict[int, dict[str, int]]]
        Inverted index: term -> doc_id -> ``{field: frekuensi}``.
    idf : dict[str, float]
        Nilai IDF setiap term.
    doc_freq : collections.Counter
        Document frequency setiap term (dihitung per dokumen, bukan per field).
    field_weights : dict[str, float]
        Bobot efektif yang dipakai untuk setiap field.
    field_b : dict[str, float]
        Nilai ``b`` efektif untuk setiap field.
    field_avgdl : dict[str, float]
        Rata-rata panjang setiap field.
    field_boost : dict[str, list[float]]
        Faktor ``w_f / (1 - b_f + b_f * |d_f| / avgdl_f)`` per dokumen per
        field, dipra-hitung agar pencarian lebih cepat.
    N : int
        Jumlah dokumen.
    avgdl : float
        Rata-rata panjang total dokumen (semua field), untuk inspeksi.

    Notes
    -----
    - IDF memakai rumus smoothing yang sama dengan BM25.
    - Semua token diubah ke huruf kecil dan diambil dengan regex ``\\w+``.
    - Untuk korpus kosong, ``search`` dan ``score`` mengembalikan hasil kosong
      tanpa error.
    """

    def __init__(self, documents, k1=1.2, b=0.75, field_weights=None):
        # Simpan parameter mentah agar mudah diperiksa saat debugging.
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)

        # 1. Tentukan field yang dipakai korpus.
        self.fields = self._detect_fields(documents)

        # 2. Siapkan parameter b per field. Satu nilai float berlaku untuk semua
        #    field; dict memungkinkan normalisasi panjang berbeda per field.
        if isinstance(b, dict):
            self.field_b = {f: float(b.get(f, 0.75)) for f in self.fields}
        else:
            self.field_b = {f: float(b) for f in self.fields}

        # 3. Bobot per field. Field tanpa bobot eksplisit memakai 1.0.
        weights = DEFAULT_FIELD_WEIGHTS if field_weights is None else field_weights
        self.field_weights = {f: float(weights.get(f, 1.0)) for f in self.fields}

        # 4. Tokenisasi setiap field secara terpisah. Field yang tidak ada pada
        #    sebuah dokumen dianggap kosong.
        self.tokenized_docs = {
            f: [
                self._tokenize(doc.get(f) or "") if isinstance(doc, dict) else self._tokenize(doc)
                for doc in documents
            ]
            for f in self.fields
        }

        # 5. Hitung panjang dan rata-rata panjang dokumen per field. Normalisasi
        #    panjang harus memakai statistik field itu sendiri, bukan panjang
        #    total dokumen.
        self.field_doc_length = {
            f: [len(tokens) for tokens in self.tokenized_docs[f]] for f in self.fields
        }
        self.field_avgdl = {
            f: (sum(lengths) / self.N) if self.N > 0 else 0.0
            for f, lengths in self.field_doc_length.items()
        }

        # 6. Pra-hitung normalisasi panjang BM25 per field:
        #    1 - b_f + b_f * (|d_f| / avgdl_f).
        self.doc_norm = {
            f: [
                (1 - self.field_b[f] + self.field_b[f] * (dl / self.field_avgdl[f]))
                if self.field_avgdl[f] > 0
                else 1.0
                for dl in self.field_doc_length[f]
            ]
            for f in self.fields
        }

        # 7. Gabungkan bobot field dengan normalisasi panjang. Faktor ini
        #    memungkinkan tf~ dihitung sebagai sum(field_boost[f][doc_id] * tf).
        self.field_boost = {
            f: [
                self.field_weights[f] / norm if norm > 0 else 0.0
                for norm in self.doc_norm[f]
            ]
            for f in self.fields
        }

        # 8. Panjang total (semua field) tetap tersedia untuk inspeksi dan
        #    kompatibilitas dengan pemakaian BM25 satu field.
        self.doc_length = [
            sum(self.field_doc_length[f][i] for f in self.fields)
            for i in range(self.N)
        ]
        self.avgdl = (sum(self.doc_length) / self.N) if self.N > 0 else 0.0

        # 9. Bangun inverted index: term -> {doc_id: {field: tf}}. Struktur ini
        #    membuat pencarian hanya menyentuh dokumen yang memuat term query.
        self.index = defaultdict(dict)
        for f in self.fields:
            for doc_id, tokens in enumerate(self.tokenized_docs[f]):
                for term, freq in Counter(tokens).items():
                    self.index[term].setdefault(doc_id, {})[f] = freq

        # 10. Hitung document frequency. Satu dokumen dihitung sekali meskipun
        #     term muncul di beberapa field, lalu hitung IDF dengan smoothing.
        self.doc_freq = Counter({term: len(postings) for term, postings in self.index.items()})
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.doc_freq.items()
        }

    @staticmethod
    def _detect_fields(documents):
        """Tentukan nama field yang dipakai korpus.

        - Jika dokumen berupa string, seluruh teks dianggap satu field ``body``.
        - Jika dokumen berupa dict, field adalah gabungan semua key dengan urutan
          kemunculan pertama. Semua dokumen wajib bertipe sama.

        Raises
        ------
        TypeError
            Jika tipe dokumen tidak konsisten atau bukan ``str``/``dict``.
        """
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
        """Ubah teks menjadi daftar token huruf kecil.

        Token diambil dengan pola ``\\w+`` sehingga spasi dan tanda baca menjadi
        pemisah. Contoh: ``"Graph-based Model"`` -> ``["graph", "based", "model"]``.
        """
        return re.findall(r"\w+", text.lower())

    def _weighted_tf(self, doc_id, field_tfs):
        """Hitung ``tf~`` (tf terbobot) untuk satu term pada satu dokumen.

        ``field_tfs`` berisi ``{field: frekuensi}``. Setiap frekuensi dikalikan
        faktor bobot dan normalisasi panjang yang sudah dipra-hitung di
        ``field_boost``, lalu dijumlahkan.
        """
        return sum(
            self.field_boost[field][doc_id] * freq
            for field, freq in field_tfs.items()
        )

    def _term_score(self, idf, doc_id, field_tfs):
        """Hitung kontribusi satu term terhadap skor dokumen.

        Rumus: ``idf * tf~ * (k1 + 1) / (k1 + tf~)``. Mengembalikan 0.0 bila
        ``tf~`` nol, misalnya karena field tersebut berbobot 0.
        """
        tf = self._weighted_tf(doc_id, field_tfs)
        if tf <= 0:
            return 0.0
        return idf * (tf * (self.k1 + 1)) / (tf + self.k1)

    def score(self, query, doc_id):
        """Hitung skor BM25F satu dokumen terhadap query.

        Parameters
        ----------
        query : str
            Teks query; tokenisasinya sama dengan tokenisasi dokumen.
        doc_id : int
            Indeks dokumen pada ``documents``.

        Returns
        -------
        float
            Skor relevansi. Bernilai 0.0 jika tidak ada term query yang cocok.
        """
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
        """Cari dokumen relevan dan urutkan berdasarkan skor BM25F.

        Hanya dokumen yang mengandung minimal satu term query yang dihitung
        (traversal inverted index). Hasil diurutkan menurun berdasarkan skor;
        dokumen berskor 0 tidak disertakan.

        Parameters
        ----------
        query : str
            Teks query.
        top_k : int | None, default None
            Batasi jumlah hasil. ``None`` mengembalikan semua dokumen relevan.

        Returns
        -------
        list[tuple[int, float]]
            Pasangan ``(doc_id, skor)`` terurut dari skor tertinggi.
        """
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


# Alias nama lama dipertahankan agar kode dan pengujian yang sudah ada tetap
# berjalan tanpa perubahan. Implementasinya tetap BM25F, bukan BM25 satu field.
BM25 = BM25F
