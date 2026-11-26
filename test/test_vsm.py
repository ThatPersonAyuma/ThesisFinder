import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.vsm.cleaning.cleaning import clean_frequencies
from src.vsm.cleaning.stemming import stemming
from src.vsm.cleaning.stopword import hapus_stopword
from src.vsm.cleaning.tokenizer import Tokenizer, build_vocab_csv
from src.vsm.indexer import build_index, iter_papers
from src.vsm.searcher import VSMSearcher
from src.vsm.tfidf import (
    analyze_text,
    inverse_document_frequency,
    light_stem,
    term_frequency,
    tokenize,
)

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
    return pd.DataFrame(
        {"id": [d[0] for d in DOCS], "title": [d[1] for d in DOCS], "abstract": [d[2] for d in DOCS]}
    )


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
        self.assertEqual(light_stem("studied"), "study")
        self.assertEqual(light_stem("approaches"), "approach")
        self.assertEqual(analyze_text("The Models of Learning"), ["model", "learn"])

    def test_tokenize_strips_urls_unicode_and_pure_digits(self):
        tokens = tokenize("Naïve approaches in 2025 see https://github.com/user/repo!")
        self.assertEqual(tokens, ["naive", "approaches", "in", "see"])

    def test_stopword_from_frek_but_topic_words_kept(self):
        self.assertEqual(
            analyze_text("often many graph neural network learning"),
            ["graph", "neural", "network", "learn"],
        )


class TestCleaningAndTokenizer(unittest.TestCase):
    def test_cleaning_merges_punctuation_before_min_freq_and_builds_tokenizer(self):
        raw = pd.DataFrame({
            "word": ["Neural,", "neural.", "https://example.com", "2025", "rareword", "approaches"],
            "frequence": [3, 3, 20, 50, 2, 6],
        })
        cleaned = clean_frequencies(raw, min_freq=5, verbose=False)
        freq_map = dict(zip(cleaned["word"], cleaned["frequence"]))
        self.assertEqual(freq_map["neural"], 6)
        self.assertEqual(freq_map["approaches"], 6)
        self.assertNotIn("rareword", freq_map)
        self.assertNotIn("2025", freq_map)

        stopped = hapus_stopword(cleaned)
        stemmed = stemming(stopped)
        vocab_df = build_vocab_csv(stemmed)

        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = os.path.join(tmp, "vocab_clean.csv")
            vocab_df.to_csv(vocab_path, index=False)
            tok = Tokenizer(vocab_path)
            self.assertIn("neural", tok.vocab)
            self.assertIn("approach", tok.vocab)
            vec = tok.vectorize("Neural approaches and neural!")
            self.assertEqual(vec.shape, (len(tok),))
            self.assertEqual(vec[tok.vocab["neural"]], 2)
            self.assertEqual(vec[tok.vocab["approach"]], 1)

            idf = np.full(len(tok), 2.0, dtype=np.float32)
            tvec = tok.vectorize_tfidf("Neural approaches", idf)
            self.assertAlmostEqual(float(np.linalg.norm(tvec)), 1.0, places=5)


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

    def test_build_index_with_fixed_vocab_and_min_df(self):
        with tempfile.TemporaryDirectory() as tmp:
            d_vocab = os.path.join(tmp, "idx_vocab")
            build_index([make_df()], d_vocab, vocab=["graph", "neural"], verbose=False)
            s = VSMSearcher(d_vocab)
            self.assertEqual(s.terms, ["graph", "neural"])
            self.assertEqual(len(s.search("molecules")), 0)
            self.assertGreater(len(s.search("graph")), 0)

            d_mindf = os.path.join(tmp, "idx_mindf")
            build_index([make_df()], d_mindf, min_df=2, verbose=False)
            s2 = VSMSearcher(d_mindf)
            self.assertIn("graph", s2.vocab)
            self.assertNotIn("shortest", s2.vocab)

    def test_empty_corpus_builds_valid_empty_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_index([], tmp, verbose=False)
            self.assertEqual(cfg["n_docs"], 0)
            s = VSMSearcher(tmp)
            self.assertEqual(s.search("graph"), [])


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
        self.assertTrue(
            {"docvec_title.parquet", "docvec_abstract.parquet", "idf_title.npy", "df_abstract.npy",
             "vocab.json", "meta.parquet", "config.json"} <= names
        )
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
        self.assertEqual(r[0].doc_idx, 0)

    def test_candidate_filter_and_edge_cases(self):
        mask = np.array([False, False, False, True])
        self.assertEqual([x.doc_idx for x in self.s.search("graph", candidates=mask)], [3])
        self.assertEqual([x.doc_idx for x in self.s.search("graph", candidates=[0, 1])], [0])
        self.assertEqual(self.s.search("graph", candidates=[]), [])
        self.assertEqual(self.s.search("graph", candidates=[999]), [])
        self.assertEqual(self.s.search({"title": "   "}), [])
        self.assertEqual(self.s.search("graph", top_k=0), [])

        _, per_field = self.s.score_all("graph", candidates=[3])
        self.assertEqual(per_field["title"][0], 0.0)
        self.assertEqual(per_field["abstract"][0], 0.0)

    def test_docvec_parquet_format(self):
        d = pd.read_parquet(os.path.join(self.tmp.name, "docvec_title.parquet"))
        self.assertEqual(list(d.columns), ["doc", "term", "w"])
        self.assertTrue((d.w > 0).all())
        self.assertTrue((np.diff(d.doc.to_numpy()) >= 0).all())
        norms = d.assign(sq=d.w.astype("float64") ** 2).groupby("doc").sq.sum() ** 0.5
        np.testing.assert_allclose(norms.to_numpy(), 1.0, atol=1e-5)

    def test_doc_vector_saved_and_query_vector(self):
        v = self.s.doc_vector(0, "title")
        self.assertAlmostEqual(sum(w * w for w in v.values()), 1.0, places=5)
        self.assertEqual(set(v), {"graph", "neural", "network", "molecule"})
        qv = self.s.vectorize_query("graph neural", field="title")
        self.assertAlmostEqual(float(np.linalg.norm(qv)), 1.0, places=5)
        with self.assertRaises(IndexError):
            self.s.doc_vector(99, "title")

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
