#!/usr/bin/env python3
"""
Tiny Embed — Zero-Dependency Text Embeddings

A single-file, zero-dependency text embeddings library in pure Python.
No PyTorch. No NumPy. No transformers.

Like sentence-transformers, but tiny.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union


__version__ = "0.1.0"
__all__ = [
    "TinyEmbed",
    "VectorStore",
    "GloVeEmbed",
    "cosine_similarity",
    "euclidean_distance",
    "semantic_similarity",
    "preprocess",
    "tokenize",
    "bag_of_words",
    "tf_idf_vectors",
    "ngrams",
    "svd",
    "project_svd",
    "load_glove",
    "SparseVector",
    "top_k_nearest",
    "matmul",
    "transpose",
]


# -----------------------------------------------------------------------------
# Default stopwords (English)
# -----------------------------------------------------------------------------

DEFAULT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "will", "with", "i", "you", "your", "we", "our", "this", "these", "those",
    "they", "them", "their", "have", "had", "been", "being", "have", "do",
    "does", "did", "can", "could", "would", "should", "may", "might", "must",
    "shall", "am", "if", "or", "so", "but", "not", "no", "yes", "than",
    "then", "when", "where", "who", "what", "how", "why", "which", "while",
    "because", "about", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "all", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "than",
    "too", "very", "just", "now",
})


# -----------------------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase and strip whitespace."""
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()


_WORD_RE = re.compile(r"\b\w+\b")


def tokenize(text: str, pattern: re.Pattern = _WORD_RE) -> List[str]:
    """Extract word tokens from text using regex."""
    return pattern.findall(normalize(text))


def remove_stopwords(
    tokens: List[str],
    stopwords: Optional[Union[set, frozenset]] = None,
) -> List[str]:
    """Filter out stopwords from a token list."""
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    return [t for t in tokens if t not in stopwords]


def preprocess(
    text: str,
    stopwords: Optional[Union[set, frozenset]] = None,
    min_len: int = 1,
) -> List[str]:
    """Full preprocessing pipeline: normalize, tokenize, filter."""
    tokens = tokenize(text)
    if stopwords is not None:
        tokens = remove_stopwords(tokens, stopwords)
    if min_len > 1:
        tokens = [t for t in tokens if len(t) >= min_len]
    return tokens


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """Generate word n-grams from a list of tokens."""
    if n <= 1:
        return [(t,) for t in tokens]
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _build_vocabulary(
    texts: List[str],
    ngram_range: Tuple[int, int] = (1, 1),
    stopwords: Optional[Union[set, frozenset]] = None,
    min_df: int = 1,
    max_df_ratio: float = 1.0,
) -> Tuple[Dict[str, int], List[int]]:
    """Build vocabulary from corpus with n-gram support."""
    vocab: Dict[str, int] = {}
    doc_freq: Counter = Counter()
    n_docs = len(texts)

    for text in texts:
        tokens = preprocess(text, stopwords=stopwords)
        features: set = set()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for gram in ngrams(tokens, n):
                if n == 1:
                    features.add(gram[0])
                else:
                    features.add(" ".join(gram))
        for feat in features:
            doc_freq[feat] += 1

    max_df = int(max_df_ratio * n_docs)
    for feat, df in doc_freq.items():
        if min_df <= df <= max_df:
            vocab[feat] = len(vocab)

    # Store document frequencies aligned with vocab indices
    dfs = [0] * len(vocab)
    for feat, idx in vocab.items():
        dfs[idx] = doc_freq[feat]

    return vocab, dfs


def bag_of_words(
    texts: List[str],
    vocab: Optional[Dict[str, int]] = None,
    ngram_range: Tuple[int, int] = (1, 1),
    stopwords: Optional[Union[set, frozenset]] = None,
) -> Tuple[List[List[float]], Dict[str, int]]:
    """Compute raw bag-of-words (count) vectors."""
    if vocab is None:
        vocab, _ = _build_vocabulary(texts, ngram_range, stopwords)

    dim = len(vocab)
    vectors: List[List[float]] = []

    for text in texts:
        tokens = preprocess(text, stopwords=stopwords)
        counts = Counter()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for gram in ngrams(tokens, n):
                feat = gram[0] if len(gram) == 1 else " ".join(gram)
                if feat in vocab:
                    counts[feat] += 1
        vec = [0.0] * dim
        for feat, count in counts.items():
            vec[vocab[feat]] = float(count)
        vectors.append(vec)

    return vectors, vocab


def tf_idf_vectors(
    texts: List[str],
    vocab: Optional[Dict[str, int]] = None,
    ngram_range: Tuple[int, int] = (1, 1),
    stopwords: Optional[Union[set, frozenset]] = None,
    sublinear_tf: bool = False,
) -> Tuple[List[List[float]], Dict[str, int], List[float], List[float]]:
    """Compute TF-IDF weighted vectors."""
    if vocab is None:
        vocab, dfs = _build_vocabulary(texts, ngram_range, stopwords)
    else:
        # Recompute dfs if vocab provided
        dfs = [0] * len(vocab)
        doc_freq = Counter()
        for text in texts:
            tokens = preprocess(text, stopwords=stopwords)
            seen = set()
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for gram in ngrams(tokens, n):
                    feat = gram[0] if len(gram) == 1 else " ".join(gram)
                    if feat in vocab:
                        seen.add(feat)
            for feat in seen:
                doc_freq[feat] += 1
        for feat, idx in vocab.items():
            dfs[idx] = doc_freq.get(feat, 0)

    n_docs = len(texts)
    dim = len(vocab)

    # Precompute IDF
    idf = [0.0] * dim
    for i, df in enumerate(dfs):
        if df > 0:
            idf[i] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    vectors: List[List[float]] = []

    for text in texts:
        tokens = preprocess(text, stopwords=stopwords)
        counts = Counter()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for gram in ngrams(tokens, n):
                feat = gram[0] if len(gram) == 1 else " ".join(gram)
                if feat in vocab:
                    counts[feat] += 1

        vec = [0.0] * dim
        total_tokens = sum(counts.values())
        for feat, count in counts.items():
            idx = vocab[feat]
            tf = count
            if sublinear_tf:
                tf = 1.0 + math.log(tf) if tf > 0 else 0.0
            else:
                # Normalize by document length
                tf = count / max(total_tokens, 1)
            vec[idx] = tf * idf[idx]

        vectors.append(vec)

    return vectors, vocab, idf, dfs


# -----------------------------------------------------------------------------
# Vector math (dense)
# -----------------------------------------------------------------------------

def dot(a: List[float], b: List[float]) -> float:
    """Dot product of two dense vectors."""
    return sum(x * y for x, y in zip(a, b))


def norm(v: List[float]) -> float:
    """L2 norm of a dense vector."""
    return math.sqrt(sum(x * x for x in v))


def normalize_vector(v: List[float]) -> List[float]:
    """L2-normalize a dense vector."""
    n = norm(v)
    if n == 0:
        return [0.0] * len(v)
    return [x / n for x in v]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two dense vectors.
    Returns a value in [-1, 1]. For non-negative vectors, [0, 1].
    """
    n_a = norm(a)
    n_b = norm(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    return dot(a, b) / (n_a * n_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two dense vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def semantic_similarity(a: List[float], b: List[float]) -> float:
    """
    Semantic similarity score mapped to [0, 1].
    Uses cosine similarity with a sigmoid-like scaling.
    """
    cos = cosine_similarity(a, b)
    # Map [-1, 1] to [0, 1] smoothly
    return (cos + 1.0) / 2.0


# -----------------------------------------------------------------------------
# Sparse vector utilities
# -----------------------------------------------------------------------------

class SparseVector:
    """A simple sparse vector representation using a dictionary."""

    __slots__ = ("dim", "data")

    def __init__(self, data: Dict[int, float], dim: int):
        self.dim = dim
        self.data = {k: float(v) for k, v in data.items() if v != 0.0}

    def to_dense(self) -> List[float]:
        vec = [0.0] * self.dim
        for i, v in self.data.items():
            vec[i] = v
        return vec

    @classmethod
    def from_dense(cls, vec: List[float]) -> "SparseVector":
        data = {i: v for i, v in enumerate(vec) if v != 0.0}
        return cls(data, len(vec))

    def dot(self, other: "SparseVector") -> float:
        a, b = self.data, other.data
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.data.values()))

    def cosine(self, other: "SparseVector") -> float:
        n_self = self.norm()
        n_other = other.norm()
        if n_self == 0.0 or n_other == 0.0:
            return 0.0
        return self.dot(other) / (n_self * n_other)

    def __repr__(self) -> str:
        return f"SparseVector(dim={self.dim}, nnz={len(self.data)})"


# -----------------------------------------------------------------------------
# Dense matrix operations
# -----------------------------------------------------------------------------

def matmul(
    A: List[List[float]], B: List[List[float]]
) -> List[List[float]]:
    """Multiply two dense matrices (A @ B)."""
    m = len(A)
    if m == 0:
        return []
    n = len(B[0])
    p = len(B)
    # Initialize result
    C = [[0.0] * n for _ in range(m)]
    # Transpose B for better cache locality
    Bt = list(zip(*B))
    for i in range(m):
        Ai = A[i]
        for j in range(n):
            C[i][j] = sum(Ai[k] * Bt[j][k] for k in range(p))
    return C


def transpose(mat: List[List[float]]) -> List[List[float]]:
    """Transpose a dense matrix."""
    if not mat:
        return []
    return [list(row) for row in zip(*mat)]


def vector_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def vector_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def vector_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def matrix_sub(
    A: List[List[float]], B: List[List[float]]
) -> List[List[float]]:
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def outer(u: List[float], v: List[float]) -> List[List[float]]:
    """Outer product of two vectors."""
    return [[ui * vj for vj in v] for ui in u]


def matrix_vector_mul(A: List[List[float]], v: List[float]) -> List[float]:
    """Multiply a matrix by a vector."""
    return [sum(row[j] * v[j] for j in range(len(v))) for row in A]


# -----------------------------------------------------------------------------
# Truncated SVD (pure Python, power iteration)
# -----------------------------------------------------------------------------

def svd(
    A: List[List[float]],
    k: int,
    max_iter: int = 100,
    tol: float = 1e-10,
    seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """
    Truncated SVD via power iteration.

    Returns (U, S, Vt) where:
      - U is m x k (left singular vectors)
      - S is length k (singular values)
      - Vt is k x n (right singular vectors transposed)
    """
    if seed is not None:
        random.seed(seed)

    m = len(A)
    if m == 0:
        return [], [], []
    n = len(A[0])

    # Copy A so we can deflate it
    M = [row[:] for row in A]

    At = transpose(M)

    U_rows: List[List[float]] = []
    S: List[float] = []
    V_rows: List[List[float]] = []

    actual_k = min(k, min(m, n))

    for _ in range(actual_k):
        # Random init vector
        v = [random.random() for _ in range(n)]
        # Normalize
        v_norm = norm(v)
        v = [x / v_norm for x in v]

        for _ in range(max_iter):
            # M^T M v
            Mv = matrix_vector_mul(M, v)
            v_new = matrix_vector_mul(At, Mv)

            v_new_norm = norm(v_new)
            if v_new_norm < 1e-15:
                break
            v_new = [x / v_new_norm for x in v_new]

            # Check convergence
            diff = sum((a - b) ** 2 for a, b in zip(v, v_new))
            v = v_new
            if diff < tol:
                break

        sigma = norm(matrix_vector_mul(M, v))
        if sigma < 1e-15:
            u = [0.0] * m
        else:
            u = [x / sigma for x in matrix_vector_mul(M, v)]

        U_rows.append(u)
        S.append(sigma)
        V_rows.append(v)

        # Deflate
        if sigma > 1e-15:
            for i in range(m):
                for j in range(n):
                    M[i][j] -= sigma * u[i] * v[j]

    # Return in standard shapes: U is m x k, Vt is k x n
    U_final = transpose(U_rows)  # m x k
    Vt = [v[:] for v in V_rows]  # k x n
    return U_final, S, Vt


def project_svd(
    vectors: List[List[float]], k: int, seed: Optional[int] = None
) -> List[List[float]]:
    """Project dense vectors to k dimensions via truncated SVD."""
    if not vectors or k <= 0:
        return vectors
    U, S, Vt = svd(vectors, k, seed=seed)
    actual_k = len(Vt)
    if actual_k == 0:
        return vectors
    V = transpose(Vt)  # n x actual_k
    projected = []
    for row in vectors:
        proj = [sum(row[j] * V[j][i] for j in range(len(row))) for i in range(actual_k)]
        projected.append(proj)
    return projected


# -----------------------------------------------------------------------------
# GloVe-style word vectors
# -----------------------------------------------------------------------------

def load_glove(
    filepath: str,
    vocab_limit: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Load pre-trained word vectors from a GloVe-style text file.
    Each line: word val1 val2 val3 ...
    """
    vectors: Dict[str, List[float]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            vals = [float(x) for x in parts[1:]]
            vectors[word] = vals
            if vocab_limit and len(vectors) >= vocab_limit:
                break
    return vectors


class GloVeEmbed:
    """Simple wrapper for GloVe-style static word vectors."""

    def __init__(
        self,
        vectors: Dict[str, List[float]],
        dim: int,
        stopwords: Optional[Union[set, frozenset]] = None,
    ):
        self.vectors = vectors
        self.dim = dim
        self.stopwords = stopwords or DEFAULT_STOPWORDS
        self._zero = [0.0] * dim

    def encode(
        self,
        text: Union[str, List[str]],
        aggregation: str = "mean",
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode text(s) into dense vectors by averaging word embeddings.
        aggregation: 'mean' | 'sum'
        """
        if isinstance(text, str):
            return self._encode_one(text, aggregation)
        return [self._encode_one(t, aggregation) for t in text]

    def _encode_one(self, text: str, aggregation: str) -> List[float]:
        tokens = preprocess(text, stopwords=self.stopwords)
        vecs = [self.vectors.get(t, self._zero) for t in tokens]
        if not vecs:
            return self._zero[:]
        n = len(vecs)
        if aggregation == "mean":
            return [sum(v[i] for v in vecs) / n for i in range(self.dim)]
        elif aggregation == "sum":
            return [sum(v[i] for v in vecs) for i in range(self.dim)]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def __repr__(self) -> str:
        return f"GloVeEmbed(dim={self.dim}, vocab={len(self.vectors)})"


# -----------------------------------------------------------------------------
# Main embedding model
# -----------------------------------------------------------------------------

class TinyEmbed:
    """
    A tiny, zero-dependency text embedding model.

    Methods:
        - 'bow': raw bag-of-words counts
        - 'tfidf': TF-IDF weighted bag-of-words
        - 'glove': pre-trained word vectors (requires loading vectors first)
    """

    def __init__(
        self,
        method: str = "tfidf",
        dim: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        stopwords: Optional[Union[set, frozenset]] = None,
        sublinear_tf: bool = False,
        glove_vectors: Optional[Dict[str, List[float]]] = None,
        glove_dim: Optional[int] = None,
    ):
        self.method = method
        self.dim = dim
        self.ngram_range = ngram_range
        self.stopwords = stopwords or DEFAULT_STOPWORDS
        self.sublinear_tf = sublinear_tf
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self._glove = None
        if method == "glove":
            if not glove_vectors or not glove_dim:
                raise ValueError("glove method requires glove_vectors and glove_dim")
            self._glove = GloVeEmbed(glove_vectors, glove_dim, stopwords=self.stopwords)

    def fit(self, texts: List[str]) -> "TinyEmbed":
        """Fit vocabulary and IDF on a corpus."""
        if self.method in ("bow", "tfidf"):
            if self.method == "tfidf":
                _, self.vocab, self.idf, _ = tf_idf_vectors(
                    texts,
                    ngram_range=self.ngram_range,
                    stopwords=self.stopwords,
                    sublinear_tf=self.sublinear_tf,
                )
            else:
                _, self.vocab = bag_of_words(
                    texts,
                    ngram_range=self.ngram_range,
                    stopwords=self.stopwords,
                )
        return self

    def transform(self, texts: List[str]) -> List[List[float]]:
        """Transform texts into embedding vectors."""
        if self.method == "glove":
            if self._glove is None:
                raise RuntimeError("GloVe model not initialized")
            vecs = self._glove.encode(texts)
            if self.dim and self.dim < self._glove.dim:
                # Project via SVD if dim is smaller
                return project_svd(vecs, self.dim)
            return vecs

        if self.method == "tfidf":
            vecs, _, _, _ = tf_idf_vectors(
                texts,
                vocab=self.vocab,
                ngram_range=self.ngram_range,
                stopwords=self.stopwords,
                sublinear_tf=self.sublinear_tf,
            )
        elif self.method == "bow":
            vecs, _ = bag_of_words(
                texts,
                vocab=self.vocab,
                ngram_range=self.ngram_range,
                stopwords=self.stopwords,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.dim and self.dim < len(self.vocab):
            return project_svd(vecs, self.dim)

        return vecs

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def encode(
        self,
        texts: Union[str, List[str]],
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode text(s). Alias for transform with string handling.
        Matches sentence-transformers API style.
        """
        if isinstance(texts, str):
            return self.transform([texts])[0]
        return self.transform(texts)

    def save(self, path: str) -> None:
        """Serialize model to JSON."""
        state = {
            "method": self.method,
            "dim": self.dim,
            "ngram_range": self.ngram_range,
            "vocab": self.vocab,
            "idf": self.idf,
            "sublinear_tf": self.sublinear_tf,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str, glove_vectors: Optional[Dict[str, List[float]]] = None) -> "TinyEmbed":
        """Deserialize model from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        inst = cls(
            method=state["method"],
            dim=state.get("dim"),
            ngram_range=tuple(state["ngram_range"]),
            stopwords=None,  # use defaults
            sublinear_tf=state.get("sublinear_tf", False),
            glove_vectors=glove_vectors,
            glove_dim=len(next(iter(glove_vectors.values()))) if glove_vectors else None,
        )
        inst.vocab = state.get("vocab", {})
        inst.idf = state.get("idf", [])
        return inst

    def __repr__(self) -> str:
        return (
            f"TinyEmbed(method={self.method!r}, dim={self.dim}, "
            f"vocab={len(self.vocab)})"
        )


# -----------------------------------------------------------------------------
# Vector store
# -----------------------------------------------------------------------------

class VectorStore:
    """
    In-memory vector store with CRUD, batch ops, and nearest-neighbor search.
    Supports both dense and sparse vectors.
    """

    def __init__(self, dim: Optional[int] = None, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self._storage: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        id: str,
        vector: Union[List[float], SparseVector],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single vector."""
        if isinstance(vector, SparseVector):
            dense = vector.to_dense()
        else:
            dense = list(vector)

        if self.dim is None:
            self.dim = len(dense)
        elif len(dense) != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {len(dense)}")

        self._storage[id] = {
            "vector": dense,
            "metadata": metadata or {},
        }

    def add_batch(
        self,
        items: List[Tuple[str, Union[List[float], SparseVector], Optional[Dict[str, Any]]]],
    ) -> None:
        """Add multiple vectors in a batch."""
        for item in items:
            if len(item) == 2:
                id, vector = item
                meta = None
            else:
                id, vector, meta = item
            self.add(id, vector, meta)

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a vector and metadata by ID."""
        return self._storage.get(id)

    def delete(self, id: str) -> bool:
        """Delete a vector by ID. Returns True if deleted."""
        if id in self._storage:
            del self._storage[id]
            return True
        return False

    def update(
        self,
        id: str,
        vector: Optional[Union[List[float], SparseVector]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector and/or metadata."""
        if id not in self._storage:
            return False
        if vector is not None:
            if isinstance(vector, SparseVector):
                dense = vector.to_dense()
            else:
                dense = list(vector)
            if self.dim is not None and len(dense) != self.dim:
                raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {len(dense)}")
            self._storage[id]["vector"] = dense
        if metadata is not None:
            self._storage[id]["metadata"].update(metadata)
        return True

    def search(
        self,
        query: Union[List[float], SparseVector],
        top_k: int = 5,
        metric: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Brute-force nearest neighbor search.
        Returns list of (id, score, metadata) sorted by score descending.
        For cosine, higher is better. For euclidean, lower is better (returned as negative).
        """
        if not self._storage:
            return []

        if isinstance(query, SparseVector):
            q = query.to_dense()
        else:
            q = list(query)

        if self.dim is not None and len(q) != self.dim:
            raise ValueError(f"Query dimension mismatch: expected {self.dim}, got {len(q)}")

        use_metric = metric or self.metric
        results = []

        for id, entry in self._storage.items():
            vec = entry["vector"]
            if use_metric == "cosine":
                score = cosine_similarity(q, vec)
            elif use_metric == "euclidean":
                score = -euclidean_distance(q, vec)  # negate so higher is better
            elif use_metric == "dot":
                score = dot(q, vec)
            else:
                raise ValueError(f"Unknown metric: {use_metric}")

            results.append((id, score, entry["metadata"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def ids(self) -> List[str]:
        return list(self._storage.keys())

    def count(self) -> int:
        return len(self._storage)

    def save(self, path: str) -> None:
        """Serialize store to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "metric": self.metric,
                    "storage": self._storage,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Deserialize store from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        store = cls(dim=state.get("dim"), metric=state.get("metric", "cosine"))
        store._storage = state["storage"]
        return store

    def __repr__(self) -> str:
        return f"VectorStore(dim={self.dim}, metric={self.metric!r}, count={self.count()})"


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------

def top_k_nearest(
    query: List[float],
    vectors: Dict[str, List[float]],
    top_k: int = 5,
    metric: str = "cosine",
) -> List[Tuple[str, float]]:
    """Standalone top-k nearest neighbor search over a dict of vectors."""
    store = VectorStore(dim=len(query), metric=metric)
    for id, vec in vectors.items():
        store.add(id, vec)
    return [(r[0], r[1]) for r in store.search(query, top_k=top_k)]
