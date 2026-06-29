#!/usr/bin/env python3
"""
Tiny Embed — Working Examples

Demonstrates:
  1. Document similarity search
  2. Duplicate detection
  3. Simple RAG pipeline with tiny-embed + tiny-agent integration concept
"""

from tiny_embed import (
    TinyEmbed,
    VectorStore,
    cosine_similarity,
    euclidean_distance,
    top_k_nearest,
    preprocess,
)


# -----------------------------------------------------------------------------
# Example 1: Document Similarity Search
# -----------------------------------------------------------------------------

def document_similarity_search():
    print("=" * 60)
    print("Example 1: Document Similarity Search")
    print("=" * 60)

    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Python is a popular programming language for data science",
        "Data scientists often use Python and R for analysis",
        "The cat sat on the mat and looked at the mouse",
        "A feline rested on the rug watching a rodent",
    ]

    model = TinyEmbed(method="tfidf", dim=64)
    vectors = model.fit_transform(documents)

    store = VectorStore(dim=len(vectors[0]))
    for i, vec in enumerate(vectors):
        store.add(f"doc_{i}", vec, {"text": documents[i]})

    queries = [
        "A quick fox jumps over a dog",
        "Neural networks in artificial intelligence",
        "Programming languages for data analysis",
    ]

    for query_text in queries:
        query_vec = model.encode(query_text)
        results = store.search(query_vec, top_k=3)
        print(f"\nQuery: '{query_text}'")
        for rank, (doc_id, score, meta) in enumerate(results, 1):
            print(f"  {rank}. {doc_id} (score={score:.4f}): {meta['text']}")


# -----------------------------------------------------------------------------
# Example 2: Duplicate Detection
# -----------------------------------------------------------------------------

def duplicate_detection():
    print("\n" + "=" * 60)
    print("Example 2: Duplicate Detection")
    print("=" * 60)

    documents = [
        "How to make sourdough bread at home",
        "How to bake sourdough bread at home",   # near-duplicate
        "Best chocolate cake recipe for beginners",
        "Beginner chocolate cake recipe",          # near-duplicate
        "Introduction to machine learning",
        "Machine learning basics for beginners",
        "Deep learning with PyTorch tutorial",
        "PyTorch deep learning tutorial",        # near-duplicate
        "Unrelated document about space exploration",
    ]

    model = TinyEmbed(method="tfidf")
    vectors = model.fit_transform(documents)

    threshold = 0.85
    duplicates = []

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim > threshold:
                duplicates.append((i, j, sim))

    if duplicates:
        print(f"\nFound {len(duplicates)} potential duplicate pairs (threshold={threshold}):")
        for i, j, sim in duplicates:
            print(f"\n  Pair (similarity={sim:.4f}):")
            print(f"    [{i}] {documents[i]}")
            print(f"    [{j}] {documents[j]}")
    else:
        print("No duplicates found.")


# -----------------------------------------------------------------------------
# Example 3: Simple RAG Pipeline
# -----------------------------------------------------------------------------

def simple_rag_pipeline():
    print("\n" + "=" * 60)
    print("Example 3: Simple RAG Pipeline")
    print("=" * 60)

    knowledge_base = [
        {
            "id": "kb_001",
            "text": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
            "source": "Biology Textbook",
        },
        {
            "id": "kb_002",
            "text": "The mitochondria is the powerhouse of the cell, generating ATP through cellular respiration.",
            "source": "Cell Biology Guide",
        },
        {
            "id": "kb_003",
            "text": "Machine learning is a method of data analysis that automates analytical model building, using algorithms that iteratively learn from data.",
            "source": "AI Handbook",
        },
        {
            "id": "kb_004",
            "text": "Neural networks are computing systems inspired by biological neural networks, used in deep learning for pattern recognition.",
            "source": "Deep Learning Book",
        },
        {
            "id": "kb_005",
            "text": "Python is an interpreted, high-level programming language known for its readability and extensive standard library.",
            "source": "Python Docs",
        },
    ]

    # Step 1: Index knowledge base
    texts = [doc["text"] for doc in knowledge_base]
    model = TinyEmbed(method="tfidf", dim=128)
    vectors = model.fit_transform(texts)

    store = VectorStore(dim=len(vectors[0]))
    for doc, vec in zip(knowledge_base, vectors):
        store.add(doc["id"], vec, {
            "text": doc["text"],
            "source": doc["source"],
        })

    print("\nKnowledge base indexed:")
    for doc in knowledge_base:
        print(f"  [{doc['id']}] {doc['text'][:50]}...")

    # Step 2: Answer user queries with retrieval-augmented responses
    queries = [
        "How do plants make energy?",
        "What is a neural network used for?",
        "Tell me about Python programming",
        "How does cellular respiration work?",
    ]

    def mock_llm_generate(context: str, query: str) -> str:
        """
        Mock LLM generation — in practice, replace with real LLM API call.
        This demonstrates the RAG pattern: retrieve context, then generate.
        """
        return f"[Based on retrieved context] Answer to '{query}' using: {context[:120]}..."

    for query in queries:
        print(f"\n{'─' * 50}")
        print(f"User Query: '{query}'")
        print("─" * 50)

        # Retrieve relevant documents
        query_vec = model.encode(query)
        results = store.search(query_vec, top_k=2)

        # Build context from top results
        context_parts = []
        for doc_id, score, meta in results:
            context_parts.append(f"[{doc_id}] {meta['text']}")
        context = "\n".join(context_parts)

        # Generate response (mock)
        response = mock_llm_generate(context, query)

        print("Retrieved context:")
        for doc_id, score, meta in results:
            print(f"  • [{doc_id}] (score={score:.4f}) {meta['text'][:80]}...")
        print(f"\nGenerated response:\n  {response}")

    # Integration concept with tiny-agent
    print("\n" + "=" * 60)
    print("Integration Concept: tiny-embed + tiny-agent")
    print("=" * 60)
    print("""
# Conceptual integration with a tiny-agent system:

from tiny_embed import TinyEmbed, VectorStore
from tiny_agent import Agent, Tool  # hypothetical tiny-agent

# Create embedding tool for the agent
embed_tool = Tool(
    name="embed",
    description="Embed text into vectors",
    func=lambda text: model.encode(text),
)

# Create retrieval tool
retrieval_tool = Tool(
    name="retrieve",
    description="Retrieve relevant documents from knowledge base",
    func=lambda query: store.search(model.encode(query), top_k=3),
)

# Agent with RAG capabilities
agent = Agent(tools=[embed_tool, retrieval_tool])

# The agent can now:
# 1. Embed user queries
# 2. Retrieve relevant context
# 3. Use retrieved context to answer questions
# 4. All without heavy ML frameworks

response = agent.run("How do plants make energy?")
""")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    document_similarity_search()
    duplicate_detection()
    simple_rag_pipeline()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
