import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.vsm.indexer import build_index, iter_papers
from src.vsm.searcher import VSMSearcher
from src.vsm.tfidf import (analyze_text, inverse_document_frequency, light_stem,
                           term_frequency)

DOCS = [
    ("http://arxiv.org/abs/2101.00001v1", "Graph neural networks for molecules",
     "We study graph neural networks applied to molecular property prediction."),
    ("http://arxiv.org/abs/2102.00002v1", "Deep learning for image classification",
     "Convolutional networks classify images with high accuracy."),
    ("http://arxiv.org/abs/2203.00003v1", "Transformers in language modeling",
     "Attention based transformers improve language modeling and translation."),
    ("http://arxiv.org/abs/2204.00004v1", "A survey of graph algorithms",
     "We survey classical graph algorithms such as shortest path."),
]


def make_df():
    return pd.DataFrame({"id": [d[0] for d in DOCS], "title": [d[1] for d in DOCS],
                         "abstract": [d[2] for d in DOCS]})


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

    def test_stopword_from_frek_but_topic_words_kept(self):
        self.assertEqual(analyze_text("often many graph neural network learning"),
                         ["graph", "neural", "network", "learn"])


class TestIndexer(unittest.TestCase):
    def test_iter_papers_reads_only_needed_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.parquet")
            df = make_df()
            df["authors"] = [["A"]] * 4
            df["keywords"] = ["x"] * 4
            df.to_parquet(path)
            parts = list(iter_papers(path, batch_size=3, limit=4))
            got = pd.concat(parts, ignore_index=True)
            self.assertEqual(list(got.columns), ["id", "title", "abstract"])
            self.assertEqual(len(got), 4)
            self.assertEqual(len(pd.concat(list(iter_papers(path, limit=2)))), 2)

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.parquet")
            make_df().drop(columns=["abstract"]).to_parquet(path)
            with self.assertRaises(ValueError):
                list(iter_papers(path))


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        build_index([make_df()], cls.tmp.name, verbose=False)
        cls.s = VSMSearcher(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_only_title_and_abstract_indexed(self):
        self.assertEqual(self.s.fields, ("title", "abstract"))
        names = set(os.listdir(self.tmp.name))
        self.assertTrue({"docvec_title.parquet", "docvec_abstract.parquet", "idf_title.npy", "df_abstract.npy",
                         "vocab.json", "meta.parquet", "config.json"} <= names)
        self.assertFalse(any("authors" in n or "keywords" in n for n in names))
        self.assertEqual(list(self.s.meta.columns), ["id", "title"])

    def test_relevant_first(self):
        r = self.s.search("graph neural network", top_k=3)
        self.assertEqual(r[0].doc_idx, 0)
        self.assertEqual(r[0].id, DOCS[0][0])
        self.assertTrue(all(0 < x.score <= 1.0001 for x in r))

    def test_field_selection(self):
        r = self.s.search("molecular", fields=["abstract"])
        self.assertEqual([x.doc_idx for x in r], [0])
        self.assertEqual(self.s.search("molecular", fields=["title"]), [])

    def test_multi_field_dict(self):
        r = self.s.search({"title": "graph", "abstract": "molecular"})
        self.assertEqual(r[0].doc_idx, 0)  # cocok di title DAN abstract

    def test_candidate_filter(self):
        mask = np.array([False, False, False, True])
        self.assertEqual([x.doc_idx for x in self.s.search("graph", candidates=mask)], [3])
        self.assertEqual([x.doc_idx for x in self.s.search("graph", candidates=[0, 1])], [0])

    def test_docvec_parquet_format(self):
        d = pd.read_parquet(os.path.join(self.tmp.name, "docvec_title.parquet"))
        self.assertEqual(list(d.columns), ["doc", "term", "w"])
        self.assertTrue((d.w > 0).all())  # nilai nol tidak disimpan
        self.assertTrue((np.diff(d.doc.to_numpy()) >= 0).all())  # terurut menurut dokumen
        norms = (d.assign(sq=d.w.astype("float64") ** 2).groupby("doc").sq.sum() ** 0.5)
        np.testing.assert_allclose(norms.to_numpy(), 1.0, atol=1e-5)  # tiap vektor unit-length

    def test_doc_vector_saved(self):
        v = self.s.doc_vector(0, "title")
        self.assertAlmostEqual(sum(w * w for w in v.values()), 1.0, places=5)  # vektor unit-length
        self.assertEqual(set(v), {"graph", "neural", "network", "molecule"})

    def test_unknown_field(self):
        for f in ("foo", "authors", "keywords"):
            with self.assertRaises(ValueError):
                self.s.search("graph", fields=[f])

    def test_matches_sklearn(self):
        """Skor per-field harus identik dengan TfidfVectorizer(sublinear_tf, smooth_idf, l2)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            self.skipTest("scikit-learn tidak terpasang")
        df = make_df()
        for field, query in (("title", "graph neural"), ("abstract", "graph networks")):
            corpus = [" ".join(analyze_text(t)) for t in df[field]]
            vec = TfidfVectorizer(sublinear_tf=True, smooth_idf=True, norm="l2", token_pattern=r"\S+")
            X = vec.fit_transform(corpus)
            q = vec.transform([" ".join(analyze_text(query))])
            expected = (X @ q.T).toarray().ravel()
            _, per_field = self.s.score_all(query, fields=[field])
            np.testing.assert_allclose(per_field[field], expected, atol=1e-5, err_msg=field)


if __name__ == "__main__":
    unittest.main()