#!/usr/bin/env python3
"""
Tiny Embed — Performance Benchmarks

Measures:
  - Embedding throughput (docs/sec)
  - Similarity query throughput (queries/sec)
  - Vector store search latency
  - SVD projection time
"""

import random
import string
import time

from tiny_embed import (
    TinyEmbed,
    VectorStore,
    cosine_similarity,
    euclidean_distance,
    top_k_nearest,
    svd,
    project_svd,
)


def generate_documents(n: int, words_per_doc: int = 50) -> list:
    """Generate random synthetic documents."""
    word_pool = [
        "machine", "learning", "deep", "neural", "network", "data", "model",
        "training", "algorithm", "feature", "vector", "embedding", "similarity",
        "clustering", "classification", "regression", "python", "code", "library",
        "benchmark", "performance", "speed", "accuracy", "precision", "recall",
        "text", "document", "corpus", "vocabulary", "token", "word", "sentence",
        "paragraph", "article", "paper", "research", "science", "technology",
        "computer", "software", "hardware", "system", "application", "service",
    ]
    docs = []
    for _ in range(n):
        words = random.choices(word_pool, k=words_per_doc)
        docs.append(" ".join(words))
    return docs


def benchmark_embedding(model: TinyEmbed, texts: list, label: str = ""):
    """Benchmark embedding generation speed."""
    n = len(texts)
    start = time.perf_counter()
    vectors = model.fit_transform(texts)
    elapsed = time.perf_counter() - start
    docs_per_sec = n / elapsed
    print(f"  [{label}] Embeddings: {n:>6} docs in {elapsed:.3f}s → {docs_per_sec:,.0f} docs/sec")
    return vectors


def benchmark_similarity(vectors: list, label: str = ""):
    """Benchmark pairwise cosine similarity computation."""
    n = len(vectors)
    # Sample pairs
    n_pairs = min(n * (n - 1) // 2, 10000)
    pairs = []
    for _ in range(n_pairs):
        i, j = random.sample(range(n), 2)
        pairs.append((i, j))

    start = time.perf_counter()
    for i, j in pairs:
        cosine_similarity(vectors[i], vectors[j])
    elapsed = time.perf_counter() - start
    queries_per_sec = n_pairs / elapsed
    print(f"  [{label}] Cosine sim: {n_pairs:>6} pairs in {elapsed:.3f}s → {queries_per_sec:,.0f} pairs/sec")


def benchmark_vector_store_search(store: VectorStore, query_vectors: list, top_k: int = 5):
    """Benchmark vector store search speed."""
    n_queries = len(query_vectors)
    start = time.perf_counter()
    for qv in query_vectors:
        store.search(qv, top_k=top_k)
    elapsed = time.perf_counter() - start
    queries_per_sec = n_queries / elapsed
    print(f"  VectorStore search: {n_queries:>6} queries in {elapsed:.3f}s → {queries_per_sec:,.0f} queries/sec")


def benchmark_svd(vectors: list, k: int):
    """Benchmark SVD projection."""
    n = len(vectors)
    dim = len(vectors[0])
    start = time.perf_counter()
    projected = project_svd(vectors, k=k)
    elapsed = time.perf_counter() - start
    print(f"  SVD projection: {n}×{dim} → {n}×{k} in {elapsed:.3f}s")
    return projected


def main():
    print("=" * 60)
    print("Tiny Embed — Performance Benchmarks")
    print("=" * 60)
    print()

    # --- Benchmark 1: Small corpus ---
    print("Benchmark 1: Small Corpus (100 docs, ~50 words each)")
    print("-" * 60)
    docs_100 = generate_documents(100)

    model_bow = TinyEmbed(method="bow")
    vecs_bow = benchmark_embedding(model_bow, docs_100, "BOW")
    benchmark_similarity(vecs_bow, "BOW")

    model_tfidf = TinyEmbed(method="tfidf")
    vecs_tfidf = benchmark_embedding(model_tfidf, docs_100, "TF-IDF")
    benchmark_similarity(vecs_tfidf, "TF-IDF")
    print()

    # --- Benchmark 2: Medium corpus ---
    print("Benchmark 2: Medium Corpus (1,000 docs, ~50 words each)")
    print("-" * 60)
    docs_1k = generate_documents(1000)

    model_tfidf_1k = TinyEmbed(method="tfidf")
    vecs_1k = benchmark_embedding(model_tfidf_1k, docs_1k, "TF-IDF")
    benchmark_similarity(vecs_1k, "TF-IDF")
    print()

    # --- Benchmark 3: Vector Store ---
    print("Benchmark 3: Vector Store Search")
    print("-" * 60)
    store = VectorStore(dim=len(vecs_1k[0]))
    for i, vec in enumerate(vecs_1k):
        store.add(f"doc_{i}", vec)

    query_docs = generate_documents(100, words_per_doc=10)
    query_vecs = model_tfidf_1k.encode(query_docs)
    benchmark_vector_store_search(store, query_vecs, top_k=5)
    print()

    # --- Benchmark 4: SVD Projection ---
    print("Benchmark 4: Dimensionality Reduction (SVD)")
    print("-" * 60)
    benchmark_svd(vecs_1k[:200], k=50)  # subset for speed
    print()

    # --- Benchmark 5: Large corpus (encode only) ---
    print("Benchmark 5: Large Corpus (5,000 docs, encode only)")
    print("-" * 60)
    docs_5k = generate_documents(5000)
    start = time.perf_counter()
    _ = model_tfidf_1k.encode(docs_5k)  # reuse fitted vocab
    elapsed = time.perf_counter() - start
    docs_per_sec = 5000 / elapsed
    print(f"  TF-IDF encode: 5,000 docs in {elapsed:.3f}s → {docs_per_sec:,.0f} docs/sec")
    print()

    print("=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
