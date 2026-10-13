"""
part3_mini_rag.py
A beginner-friendly demonstration of a mini Retrieval-Augmented Generation (RAG) system.
Goal: Retrieve relevant source chunks from a knowledge base to ground answers,
contrasting grounded responses against risky ungrounded hallucinations.

Note: This is a tiny visible version of the RAG idea using basic Bag-of-Words and Cosine Similarity,
not a production enterprise RAG system with neural dense embeddings or vector databases.
"""

from collections import Counter
import math
import re


# Tiny in-memory knowledge base containing five lecture topics
KNOWLEDGE_BASE = [
    {
        "id": 1,
        "title": "Generative AI",
        "text": "Generative AI refers to artificial intelligence models capable of generating novel content, including text, images, code, and audio, based on patterns learned from training data."
    },
    {
        "id": 2,
        "title": "Attention",
        "text": "Attention in a transformer allows the model to dynamically focus on different parts of an input sequence, calculating weights that determine how much each word should influence other words."
    },
    {
        "id": 3,
        "title": "Cosine Similarity",
        "text": "Cosine similarity is a metric used to measure the angular orientation and directional similarity between two vectors in high-dimensional embedding space, independent of their magnitude."
    },
    {
        "id": 4,
        "title": "RAG",
        "text": "Retrieval-Augmented Generation (RAG) augments large language models by retrieving relevant external knowledge chunks from a database to ground the generation in verified factual context and reduce hallucinations."
    },
    {
        "id": 5,
        "title": "LoRA",
        "text": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes pretrained model weights and injects trainable rank-decomposition matrices into transformer layers to reduce compute and memory costs."
    }
]

# Simple stopword list to filter out non-informative words
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "with", "by", "and", "or", "it", "this", "that", "what",
    "do", "does", "how", "can"
}


def tokenize(text):
    """Lowercase text, remove punctuation, and remove simple stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def cosine_similarity(counter1, counter2):
    """Compute cosine similarity between two bag-of-words frequency Counters."""
    all_terms = set(counter1.keys()) | set(counter2.keys())
    dot_product = sum(counter1.get(t, 0) * counter2.get(t, 0) for t in all_terms)
    mag1 = math.sqrt(sum(v**2 for v in counter1.values()))
    mag2 = math.sqrt(sum(v**2 for v in counter2.values()))
    return dot_product / (mag1 * mag2) if mag1 and mag2 else 0.0


def search_knowledge_base(query):
    """Tokenize query and rank knowledge base chunks by cosine similarity."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_bow = Counter(query_tokens)
    scored_chunks = []

    for chunk in KNOWLEDGE_BASE:
        chunk_text = f"{chunk['title']} {chunk['text']}"
        chunk_tokens = tokenize(chunk_text)
        chunk_bow = Counter(chunk_tokens)
        score = cosine_similarity(query_bow, chunk_bow)
        scored_chunks.append((score, chunk))

    # Sort descending by similarity score
    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return scored_chunks


def answer_query(query):
    """Process question, retrieve top 2 chunks, and present ungrounded vs RAG outputs."""
    scored_chunks = search_knowledge_base(query)

    # Check if we have any valid matching chunk
    if not scored_chunks or scored_chunks[0][0] <= 0.0:
        print("\nResult: Not found in the provided knowledge base.\n")
        return

    top2 = scored_chunks[:2]
    best_score, best_chunk = top2[0]

    print("\n" + "=" * 60)
    print(f"Query: \"{query}\"")
    print("-" * 60)
    print("Retrieved Top Chunks:")
    for rank, (score, chunk) in enumerate(top2, start=1):
        print(f"[{rank}] Chunk ID: {chunk['id']} | Title: '{chunk['title']}' | Score: {score:.4f}")
        print(f"    Text: {chunk['text']}")

    print("\n--- Output Comparison ---")
    print("1. [Ungrounded Answer Style Warning]")
    print("   WARNING: Generating an answer purely from model parameters without retrieval risks")
    print("   hallucinations, obsolete data, or incorrect assumptions because it lacks verified citation.")

    print("\n2. [RAG Grounded Answer]")
    print(f"   According to Chunk {best_chunk['id']} ('{best_chunk['title']}'):")
    print(f"   \"{best_chunk['text']}\"")
    print("=" * 60 + "\n")


def main():
    print("=== Part 3: Mini-RAG (Retrieval-Augmented Generation) ===")
    print("Demonstrating how grounding answers in retrieved source chunks prevents hallucinations.")
    print("Note: This is a tiny visible teaching version of RAG, not an enterprise production system.\n")

    while True:
        try:
            query = input("Ask a question (or press Enter to quit): ").strip()
            if not query:
                print("Goodbye!")
                break
            answer_query(query)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
