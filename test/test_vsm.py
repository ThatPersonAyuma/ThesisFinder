import math
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.vsm.indexer import build_index, clean_keywords, normalize_batch
from src.vsm.searcher import VSMSearcher
from src.vsm.tfidf import analyze_text, inverse_document_frequency, light_stem, term_frequency

DOCS = [
    ("http://arxiv.org/abs/2101.00001v1", "Graph neural networks for molecules",
     "We study graph neural networks applied to molecular property prediction.", ["Alice Kipf"], ["cs.LG"], "Index Terms—graph, molecules"),
    ("http://arxiv.org/abs/2102.00002v1", "Deep learning for image classification",
     "Convolutional networks classify images with high accuracy.", ["Bob Smith"], ["cs.CV"], None),
    ("http://arxiv.org/abs/2203.00003v1", "Transformers in language modeling",
     "Attention based transformers improve language modeling and translation.", ["Alice Kipf", "Carol Lee"], ["cs.CL"], "Keywords: transformer; language model"),
    ("http://arxiv.org/abs/2204.00004v1", "A survey of graph algorithms",
     "We survey classical graph algorithms such as shortest path.", ["Dan Brown"], ["cs.DS"], None),
]


def make_df():
    raw = pd.DataFrame({
        "id": [d[0] for d in DOCS], "title": [d[1] for d in DOCS], "abstract": [d[2] for d in DOCS],
        "authors": [d[3] for d in DOCS], "categories": [d[4] for d in DOCS], "keywords": [d[5] for d in DOCS],
    })
    return normalize_batch(raw)


class TestFormulas(unittest.TestCase):
    def test_tf(self):
        np.testing.assert_allclose(term_frequency(np.array([1, 2, 10])), [1.0, 1 + math.log(2), 1 + math.log(10)])

    def test_idf_smooth(self):
        idf = inverse_document_frequency(np.array([1, 4, 0]), n_docs=4)
        self.assertAlmostEqual(idf[0], math.log(5 / 2) + 1)
        self.assertAlmostEqual(idf[1], math.log(5 / 5) + 1)  # muncul di semua dokumen -> 1.0
        self.assertEqual(idf[2], 0.0)

    def test_stem_and_stopword(self):
        self.assertEqual(light_stem("models"), "model")
        self.assertEqual(light_stem("learning"), "learn")
        self.assertEqual(light_stem("studies"), "study")
        self.assertEqual(analyze_text("The Models of Learning"), ["model", "learn"])


class TestDataset(unittest.TestCase):
    def test_clean_keywords(self):
        self.assertEqual(clean_keywords("Index Terms—meta-learning, evolution strategies, tpu"),
                         ["meta-learning", "evolution strategies", "tpu"])
        self.assertEqual(clean_keywords("KEYWORDS"), [])
        self.assertEqual(clean_keywords(None), [])
        self.assertEqual(clean_keywords("Keywords-MLOps; AutoML; machine learning, re-training;"),
                         ["MLOps", "AutoML", "machine learning", "re-training"])

    def test_normalize_derives_columns(self):
        df = make_df()
        self.assertEqual(list(df.year), [2021, 2021, 2022, 2022])
        self.assertEqual(list(df.month), [1, 2, 3, 4])
        self.assertEqual(df.pdf[0], "http://arxiv.org/pdf/2101.00001v1")
        self.assertEqual(df.primary_category[2], "cs.CL")


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        build_index([make_df()], cls.tmp.name, verbose=False)
        cls.s = VSMSearcher(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_relevant_first(self):
        r = self.s.search("graph neural network", top_k=3)
        self.assertEqual(r[0].doc_idx, 0)
        self.assertTrue(all(0 < x.score <= 1.0001 for x in r))

    def test_field_selection(self):
        r = self.s.search("kipf", fields=["authors"])
        self.assertEqual({x.doc_idx for x in r}, {0, 2})
        self.assertEqual(self.s.search("kipf", fields=["title"]), [])

    def test_multi_field_dict(self):
        r = self.s.search({"title": "graph", "authors": "kipf"})
        self.assertEqual(r[0].doc_idx, 0)  # cocok di title DAN authors

    def test_keywords_field(self):
        r = self.s.search("transformer", fields=["keywords"])
        self.assertEqual([x.doc_idx for x in r], [2])

    def test_candidate_filter(self):
        mask = np.array([False, False, False, True])
        r = self.s.search("graph", candidates=mask)
        self.assertEqual([x.doc_idx for x in r], [3])
        r = self.s.search("graph", candidates=[0, 1])
        self.assertEqual([x.doc_idx for x in r], [0])

    def test_unknown_field(self):
        with self.assertRaises(ValueError):
            self.s.search("graph", fields=["foo"])

    def test_matches_sklearn(self):
        """Skor per-field harus identik dengan TfidfVectorizer(sublinear_tf, smooth_idf, l2)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            self.skipTest("scikit-learn tidak terpasang")
        df = make_df()
        corpus = [" ".join(analyze_text(t)) for t in df.title]
        vec = TfidfVectorizer(sublinear_tf=True, smooth_idf=True, norm="l2", token_pattern=r"\S+")
        X = vec.fit_transform(corpus)
        q = vec.transform([" ".join(analyze_text("graph neural"))])
        expected = (X @ q.T).toarray().ravel()
        _, per_field = self.s.score_all("graph neural", fields=["title"])
        np.testing.assert_allclose(per_field["title"], expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()