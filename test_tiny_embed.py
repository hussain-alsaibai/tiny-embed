#!/usr/bin/env python3
"""
Comprehensive test suite for Tiny Embed.
Run with: python -m unittest test_tiny_embed
"""

import math
import os
import tempfile
import unittest

from tiny_embed import (
    TinyEmbed,
    VectorStore,
    GloVeEmbed,
    SparseVector,
    cosine_similarity,
    euclidean_distance,
    semantic_similarity,
    preprocess,
    tokenize,
    bag_of_words,
    tf_idf_vectors,
    ngrams,
    svd,
    top_k_nearest,
    project_svd,
    matmul,
    transpose,
)


class TestPreprocessing(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Hello, world!"), ["hello", "world"])

    def test_preprocess(self):
        tokens = preprocess("The quick brown fox.", stopwords={"the"})
        self.assertNotIn("the", tokens)
        self.assertIn("quick", tokens)

    def test_ngrams(self):
        tokens = ["the", "quick", "brown"]
        self.assertEqual(ngrams(tokens, 2), [("the", "quick"), ("quick", "brown")])
        self.assertEqual(ngrams(tokens, 3), [("the", "quick", "brown")])


class TestVectorMath(unittest.TestCase):
    def test_cosine_similarity(self):
        a = [1, 0, 0]
        b = [1, 0, 0]
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0)
        c = [0, 1, 0]
        self.assertAlmostEqual(cosine_similarity(a, c), 0.0)

    def test_euclidean_distance(self):
        a = [1, 2, 3]
        b = [1, 2, 3]
        self.assertAlmostEqual(euclidean_distance(a, b), 0.0)
        c = [4, 0, 0]
        self.assertAlmostEqual(euclidean_distance(a, c), math.sqrt(22))

    def test_semantic_similarity(self):
        a = [1, 0]
        b = [1, 0]
        self.assertAlmostEqual(semantic_similarity(a, b), 1.0)
        c = [0, 1]
        self.assertAlmostEqual(semantic_similarity(a, c), 0.5)


class TestFeatureExtraction(unittest.TestCase):
    def test_bag_of_words(self):
        texts = ["hello world", "hello hello"]
        vecs, vocab = bag_of_words(texts)
        self.assertEqual(len(vocab), 2)
        self.assertEqual(vecs[0][vocab["hello"]], 1.0)
        self.assertEqual(vecs[1][vocab["hello"]], 2.0)

    def test_tfidf(self):
        texts = ["hello world", "hello hello"]
        vecs, vocab, idf, dfs = tf_idf_vectors(texts)
        self.assertEqual(len(vocab), 2)
        # hello appears in both docs -> lower idf than world
        self.assertLess(idf[vocab["hello"]], idf[vocab["world"]])


class TestSparseVector(unittest.TestCase):
    def test_from_dense(self):
        sv = SparseVector.from_dense([0, 1, 0, 2])
        self.assertEqual(sv.data, {1: 1.0, 3: 2.0})

    def test_dot(self):
        a = SparseVector.from_dense([1, 0, 2])
        b = SparseVector.from_dense([0, 3, 4])
        self.assertEqual(a.dot(b), 1 * 0 + 0 * 3 + 2 * 4)


class TestSVD(unittest.TestCase):
    def test_svd_shape(self):
        A = [[1, 2], [3, 4], [5, 6]]
        U, S, Vt = svd(A, k=2)
        self.assertEqual(len(U), 3)
        self.assertEqual(len(U[0]), 2)
        self.assertEqual(len(S), 2)
        self.assertEqual(len(Vt), 2)
        self.assertEqual(len(Vt[0]), 2)

    def test_project_svd(self):
        vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        projected = project_svd(vectors, k=2)
        self.assertEqual(len(projected), 3)
        self.assertEqual(len(projected[0]), 2)


class TestMatrixOps(unittest.TestCase):
    def test_matmul(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = matmul(A, B)
        self.assertEqual(C, [[19, 22], [43, 50]])

    def test_transpose(self):
        A = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(transpose(A), [[1, 4], [2, 5], [3, 6]])


class TestTinyEmbed(unittest.TestCase):
    def test_fit_transform_bow(self):
        model = TinyEmbed(method="bow")
        texts = ["hello world", "hello there"]
        vecs = model.fit_transform(texts)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), len(model.vocab))

    def test_fit_transform_tfidf(self):
        model = TinyEmbed(method="tfidf")
        texts = ["the quick brown fox", "the lazy dog"]
        vecs = model.fit_transform(texts)
        self.assertEqual(len(vecs), 2)
        self.assertGreater(len(model.vocab), 0)

    def test_encode_single(self):
        model = TinyEmbed(method="bow")
        model.fit(["hello world", "foo bar"])
        vec = model.encode("hello world")
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), len(model.vocab))

    def test_encode_batch(self):
        model = TinyEmbed(method="bow")
        model.fit(["hello world", "foo bar"])
        vecs = model.encode(["hello world", "foo bar"])
        self.assertEqual(len(vecs), 2)

    def test_dimensionality_reduction(self):
        texts = ["hello world", "foo bar", "baz qux", "hello foo"]
        model = TinyEmbed(method="tfidf", dim=2)
        vecs = model.fit_transform(texts)
        self.assertEqual(len(vecs[0]), 2)

    def test_save_load(self):
        model = TinyEmbed(method="tfidf")
        model.fit(["hello world", "foo bar"])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            path = f.name
        try:
            model.save(path)
            loaded = TinyEmbed.load(path)
            self.assertEqual(loaded.method, "tfidf")
            self.assertEqual(loaded.vocab, model.vocab)
        finally:
            os.unlink(path)


class TestGloVeEmbed(unittest.TestCase):
    def test_encode(self):
        vectors = {
            "hello": [1.0, 2.0, 3.0],
            "world": [4.0, 5.0, 6.0],
        }
        model = GloVeEmbed(vectors, dim=3)
        vec = model.encode("hello world")
        expected = [(1 + 4) / 2, (2 + 5) / 2, (3 + 6) / 2]
        self.assertEqual(vec, expected)

    def test_encode_unknown(self):
        vectors = {"hello": [1.0, 2.0, 3.0]}
        model = GloVeEmbed(vectors, dim=3)
        vec = model.encode("xyz")
        self.assertEqual(vec, [0.0, 0.0, 0.0])


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore(dim=3)

    def test_add_get(self):
        self.store.add("a", [1, 2, 3], {"text": "hello"})
        entry = self.store.get("a")
        self.assertEqual(entry["vector"], [1, 2, 3])
        self.assertEqual(entry["metadata"]["text"], "hello")

    def test_delete(self):
        self.store.add("a", [1, 2, 3])
        self.assertTrue(self.store.delete("a"))
        self.assertIsNone(self.store.get("a"))

    def test_update(self):
        self.store.add("a", [1, 2, 3])
        self.store.update("a", vector=[4, 5, 6])
        self.assertEqual(self.store.get("a")["vector"], [4, 5, 6])

    def test_search_cosine(self):
        self.store.add("a", [1, 0, 0])
        self.store.add("b", [0, 1, 0])
        self.store.add("c", [1, 1, 0])
        results = self.store.search([1, 0, 0], top_k=2)
        self.assertEqual(len(results), 2)
        ids = [r[0] for r in results]
        self.assertIn("a", ids)

    def test_search_euclidean(self):
        store = VectorStore(dim=3, metric="euclidean")
        store.add("a", [0, 0, 0])
        store.add("b", [10, 10, 10])
        results = store.search([0, 0, 0], top_k=1)
        self.assertEqual(results[0][0], "a")

    def test_batch_add(self):
        items = [
            ("a", [1, 0, 0]),
            ("b", [0, 1, 0]),
            ("c", [0, 0, 1]),
        ]
        self.store.add_batch(items)
        self.assertEqual(self.store.count(), 3)

    def test_save_load(self):
        self.store.add("a", [1, 2, 3], {"x": 1})
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            path = f.name
        try:
            self.store.save(path)
            loaded = VectorStore.load(path)
            self.assertEqual(loaded.count(), 1)
            self.assertEqual(loaded.get("a")["vector"], [1, 2, 3])
        finally:
            os.unlink(path)


class TestTopKNearest(unittest.TestCase):
    def test_top_k(self):
        vectors = {
            "a": [1, 0, 0],
            "b": [0, 1, 0],
            "c": [0.9, 0.1, 0],
        }
        results = top_k_nearest([1, 0, 0], vectors, top_k=2)
        self.assertEqual(len(results), 2)
        ids = [r[0] for r in results]
        self.assertIn("a", ids)
        self.assertIn("c", ids)


class TestEdgeCases(unittest.TestCase):
    def test_empty_string(self):
        model = TinyEmbed(method="bow")
        model.fit(["hello world"])
        vec = model.encode("")
        self.assertEqual(len(vec), len(model.vocab))
        self.assertEqual(sum(vec), 0.0)

    def test_unicode(self):
        model = TinyEmbed(method="bow")
        model.fit(["こんにちは世界", "hello world"])
        vec = model.encode("こんにちは")
        self.assertEqual(len(vec), len(model.vocab))

    def test_long_text(self):
        model = TinyEmbed(method="tfidf")
        long_text = "word " * 1000
        model.fit([long_text, "short text"])
        vec = model.encode(long_text)
        self.assertEqual(len(vec), len(model.vocab))
        self.assertGreater(sum(vec), 0)


if __name__ == "__main__":
    unittest.main()
