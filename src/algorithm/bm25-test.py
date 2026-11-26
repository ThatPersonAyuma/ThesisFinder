import csv
import math
import sys
import unittest
from pathlib import Path

# Memastikan root project ada di sys.path agar modul src dapat diimpor.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithm.bm25f import BM25, BM25F

# Dataset dokumen acuan untuk pengujian Information Retrieval (IR)
BENCHMARK_DOCS = [
    # Doc 0: Graph Neural Networks & Molecule
    "Graph Neural Networks for Molecular Property Prediction: We investigate graph neural networks "
    "and deep learning architectures for predicting molecular properties in drug discovery.",
    # Doc 1: Computer Vision & Image Classification
    "Deep Learning for Real-Time Image Classification: Convolutional neural networks for fast image "
    "classification and object detection in computer vision applications.",
    # Doc 2: NLP & Transformers
    "Attention Mechanisms and Transformers in Natural Language Processing: Transformers improve "
    "natural language understanding and language modeling using self attention mechanisms.",
    # Doc 3: Graph Algorithms & Routing
    "Classical Graph Algorithms for Shortest Path and Network Routing: A comprehensive study of "
    "graph algorithms including Dijkstra and Bellman-Ford for network routing.",
    # Doc 4: Robotics & Reinforcement Learning
    "Reinforcement Learning for Autonomous Drone Navigation: Deep reinforcement learning algorithms "
    "for continuous control and policy optimization in robotics.",
    # Doc 5: Quantum & Cryptography
    "Quantum Cryptography and Post-Quantum Security Protocols: A survey of lattice-based cryptography "
    "and post-quantum encryption protocols for secure communications.",
]


class TestBM25Formulas(unittest.TestCase):
    """Pengujian kebenaran matematis dari komponen rumus Okapi BM25."""

    def setUp(self):
        self.bm = BM25(BENCHMARK_DOCS, k1=1.2, b=0.75)

    def test_idf_values_and_smoothing(self):
        """Memverifikasi bahwa nilai IDF non-negatif dan sesuai rumus perataan (smoothing)."""
        for term, idf_val in self.bm.idf.items():
            df = self.bm.doc_freq[term]
            expected_idf = math.log((self.bm.N - df + 0.5) / (df + 0.5) + 1)
            self.assertAlmostEqual(idf_val, expected_idf, places=6)
            self.assertGreater(idf_val, 0.0)

    def test_term_frequency_saturation(self):
        """Memverifikasi bahwa dokumen dengan frekuensi kata lebih tinggi mendapat skor lebih tinggi."""
        # Doc A: 1 kali muncul kata 'network', Doc B: 2 kali muncul kata 'network'
        corpus = [
            "neural network model",
            "neural network network model",
        ]
        bm = BM25(corpus, k1=1.2, b=0.0)  # b=0 untuk menonaktifkan penalti panjang dokumen
        score_single = bm.score("network", 0)
        score_double = bm.score("network", 1)

        self.assertGreater(score_double, score_single)

    def test_document_length_normalization(self):
        """Memverifikasi bahwa parameter 'b' memberikan penalti pada dokumen yang lebih panjang (TF sama)."""
        # Kedua dokumen memiliki 1 kemunculan kata 'graph', namun doc 1 jauh lebih panjang
        corpus = [
            "graph algorithm",
            "graph algorithm with deep neural network transformer attention machine learning system",
        ]
        bm = BM25(corpus, k1=1.2, b=0.75)
        score_short = bm.score("graph", 0)
        score_long = bm.score("graph", 1)

        # Dokumen pendek harus memiliki skor BM25 lebih tinggi karena konsentrasi informasinya lebih padat
        self.assertGreater(
            score_short,
            score_long,
            "Dokumen yang lebih pendek harus mendapat skor lebih tinggi saat b > 0.",
        )

    def test_score_matches_search_result(self):
        """Memverifikasi bahwa skor dari search() identik dengan score() langsung untuk tiap doc_id."""
        query = "graph neural network"
        search_results = self.bm.search(query)

        self.assertGreater(len(search_results), 0)
        for doc_id, s in search_results:
            direct_score = self.bm.score(query, doc_id)
            self.assertAlmostEqual(s, direct_score, places=6)


class TestBM25SearchProperties(unittest.TestCase):
    """Pengujian fungsionalitas dan sifat-sifat dasar pencarian BM25."""

    def setUp(self):
        self.bm = BM25(BENCHMARK_DOCS)

    def test_results_sorted_descending(self):
        """Memverifikasi bahwa hasil pencarian selalu terurut menurun berdasarkan skor relevansi."""
        results = self.bm.search("deep learning neural")
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_limiting(self):
        """Memverifikasi bahwa parameter top_k membatasi jumlah hasil yang dikembalikan."""
        query = "learning neural network"
        for k in (1, 2, 3):
            results = self.bm.search(query, top_k=k)
            self.assertLessEqual(len(results), k)

    def test_non_positive_top_k_returns_empty(self):
        """top_k <= 0 tidak boleh mengembalikan sebagian hasil."""
        for k in (0, -1):
            self.assertEqual(self.bm.search("learning", top_k=k), [])

    def test_ties_break_by_doc_id(self):
        """Dokumen dengan skor sama diurutkan menurut doc_id naik."""
        bm = BM25(["gamma", "gamma", "beta", "gamma", "gamma", "alpha"])
        self.assertEqual([doc_id for doc_id, _ in bm.search("alpha beta")], [2, 5])

    def test_unmatched_document_returns_zero(self):
        """Memverifikasi bahwa dokumen yang tidak mengandung kata kunci query menghasilkan skor 0.0."""
        # Doc 5 adalah tentang 'Quantum Cryptography', tidak ada kata 'image' atau 'vision'
        score = self.bm.score("image vision classification", 5)
        self.assertEqual(score, 0.0)

    def test_out_of_vocabulary_query(self):
        """Memverifikasi bahwa kata yang sama sekali tidak ada di korpus mengembalikan daftar kosong."""
        results = self.bm.search("unknown extraterrestrial signal")
        self.assertEqual(results, [])

    def test_empty_or_punctuation_query(self):
        """Memverifikasi bahwa query kosong atau hanya tanda baca tidak menyebabkan crash."""
        self.assertEqual(self.bm.search(""), [])
        self.assertEqual(self.bm.search("   "), [])
        self.assertEqual(self.bm.search("!@#$%^&*()_+"), [])

    def test_empty_corpus_handling(self):
        """Memverifikasi bahwa inisialisasi korpus kosong ditangani tanpa ZeroDivisionError."""
        bm_empty = BM25([])
        self.assertEqual(bm_empty.N, 0)
        self.assertEqual(bm_empty.avgdl, 0.0)
        self.assertEqual(bm_empty.search("test"), [])


class TestBM25FFields(unittest.TestCase):
    """Pengujian BM25F: bobot field, normalisasi per field, dan statistik korpus."""

    FIELD_DOCS = [
        {"title": "graph neural", "abstract": "image classification"},
        {"title": "image classification", "abstract": "graph neural"},
    ]

    def test_default_title_weight_outranks_abstract(self):
        """Kecocokan di title harus lebih tinggi daripada di abstract dengan tf sama."""
        bm = BM25F(self.FIELD_DOCS)
        results = bm.search("graph")

        self.assertEqual(results[0][0], 0)
        self.assertGreater(results[0][1], results[1][1])

        # Verifikasi rumus BM25F: tf~ title = 3 (bobot default), tf~ abstract = 1.
        idf = bm.idf["graph"]
        expected = idf * (3 * (bm.k1 + 1)) / (3 + bm.k1)
        self.assertAlmostEqual(bm.score("graph", 0), expected, places=6)

    def test_custom_field_weights(self):
        """Bobot field yang sama harus menghasilkan skor yang sama untuk dokumen simetris."""
        bm = BM25F(self.FIELD_DOCS, field_weights={"title": 1.0, "abstract": 1.0})
        self.assertAlmostEqual(bm.score("graph", 0), bm.score("graph", 1), places=6)

    def test_document_frequency_counts_document_once(self):
        """df dihitung per dokumen, bukan per kemunculan term di beberapa field."""
        docs = [
            {"title": "graph network", "abstract": "graph learning"},
            {"title": "network graph", "abstract": "learning"},
            {"title": "learning", "abstract": "network"},
        ]
        bm = BM25F(docs)

        self.assertEqual(bm.doc_freq["graph"], 2)
        self.assertEqual(bm.doc_freq["network"], 3)
        self.assertEqual(bm.doc_freq["learning"], 3)

    def test_missing_field_is_treated_as_empty(self):
        """Dokumen yang tidak memiliki salah satu field tetap dapat dicari."""
        bm = BM25F([{"title": "graph"}, {"abstract": "graph"}])
        results = bm.search("graph")

        self.assertEqual(len(results), 2)
        self.assertTrue(all(score > 0 for _, score in results))

    def test_score_matches_search_result(self):
        """Skor search() pada dokumen multi-field identik dengan score() langsung."""
        bm = BM25F(self.FIELD_DOCS)
        for doc_id, score in bm.search("graph neural classification"):
            self.assertAlmostEqual(score, bm.score("graph neural classification", doc_id), places=6)

    def test_bm25_alias_is_bm25f(self):
        """Nama lama BM25 tetap tersedia dan menunjuk ke implementasi BM25F."""
        self.assertIs(BM25, BM25F)

    def test_corpus_stats_override_local_statistics(self):
        """IDF dan avgdl korpus menggantikan statistik lokal kandidat."""
        docs = [
            {"title": "graph", "abstract": "graph"},
            {"title": "other", "abstract": "text"},
        ]
        bm = BM25F(docs, corpus_idf={"graph": 2.0},
                   field_avgdl={"title": 10.0, "abstract": 20.0})

        self.assertEqual(bm.field_avgdl["title"], 10.0)
        self.assertEqual(bm.field_avgdl["abstract"], 20.0)
        self.assertEqual(bm.idf["graph"], 2.0)
        self.assertEqual(bm.score("other", 1), 0.0)
        self.assertEqual([doc_id for doc_id, _ in bm.search("graph")], [0])

        tf_title = 3.0 / (1 - 0.75 + 0.75 * (1 / 10.0))
        tf_abstract = 1.0 / (1 - 0.75 + 0.75 * (1 / 20.0))
        tf = tf_title + tf_abstract
        expected = 2.0 * (tf * (bm.k1 + 1)) / (tf + bm.k1)
        self.assertAlmostEqual(bm.score("graph", 0), expected, places=6)


class TestBM25QueryDataset(unittest.TestCase):
    """Pengujian berbasis data query dari file query-bm25-test.csv."""

    def setUp(self):
        self.bm = BM25(BENCHMARK_DOCS)
        self.csv_path = Path(__file__).resolve().parent / "query-bm25-test.csv"
        self.assertTrue(self.csv_path.exists(), f"File {self.csv_path} tidak ditemukan.")

    def test_queries_from_csv(self):
        """Mengeksekusi setiap baris query di CSV dan memvalidasi ranking dokumen yang diharapkan."""
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertGreater(len(rows), 0, "File CSV query tidak boleh kosong.")

        for row in rows:
            query_id = row["query_id"]
            query = row["query"]
            expected_top = int(row["expected_top_doc_id"])
            relevant_docs = (
                [int(x) for x in row["relevant_doc_ids"].split(";") if x.strip()]
                if row["relevant_doc_ids"]
                else []
            )
            scenario = row.get("scenario", "")

            with self.subTest(query_id=query_id, query=query, scenario=scenario):
                results = self.bm.search(query, top_k=5)

                if expected_top == -1:
                    # Skenario Out-of-Vocabulary: harus mengembalikan []
                    self.assertEqual(
                        results,
                        [],
                        f"[Query #{query_id}] '{query}' seharusnya menghasilkan daftar kosong.",
                    )
                else:
                    self.assertGreater(
                        len(results),
                        0,
                        f"[Query #{query_id}] '{query}' seharusnya menghasilkan dokumen.",
                    )
                    top_doc_id = results[0][0]
                    top_score = results[0][1]

                    # Skor dokumen teratas harus bernilai positif
                    self.assertGreater(
                        top_score,
                        0.0,
                        f"[Query #{query_id}] Skor dokumen teratas harus > 0.",
                    )

                    # Dokumen teratas harus sesuai dengan expected_top_doc_id
                    self.assertEqual(
                        top_doc_id,
                        expected_top,
                        f"[Query #{query_id}] Query: '{query}'. Diharapkan top doc {expected_top}, didapat {top_doc_id}.",
                    )

                    # Dokumen teratas juga harus termasuk dalam daftar dokumen yang relevan
                    if relevant_docs:
                        self.assertIn(
                            top_doc_id,
                            relevant_docs,
                            f"[Query #{query_id}] Dokumen {top_doc_id} harus ada di {relevant_docs}.",
                        )


if __name__ == "__main__":
    unittest.main()
