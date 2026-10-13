"""
part2_attention_mechanism.py
A beginner-friendly demonstration of a tiny attention mechanism using the ambiguous word "bat".
Teaching analogy: Attention acts like a spotlight deciding which nearby words deserve more focus
and pull the meaning of an ambiguous word toward a specific context.

Note: This is a toy attention demo demonstrating cosine similarity + softmax weighting,
not a full multi-head transformer with learned Q/K/V projection matrices.
"""

import math
import re


# Dimensions: [animal, sport, motion, place, object, grammar]
WORD_VECTORS = {
    # Ambiguous target words
    "bat": [3.0, 3.0, 1.0, 0.0, 2.0, 0.0],
    "bats": [3.0, 3.0, 1.0, 0.0, 2.0, 0.0],
    # Sports-related words
    "ball": [0.0, 4.0, 1.0, 0.0, 4.0, 0.0],
    "baseball": [0.0, 5.0, 1.0, 1.0, 3.0, 0.0],
    "player": [1.0, 4.0, 2.0, 0.0, 0.0, 0.0],
    "swung": [0.0, 3.0, 4.0, 0.0, 0.0, 0.0],
    "hit": [0.0, 3.0, 4.0, 0.0, 0.0, 0.0],
    # Animal / Nature / Motion words
    "flew": [2.0, 0.0, 4.0, 0.0, 0.0, 0.0],
    "flying": [2.0, 0.0, 4.0, 0.0, 0.0, 0.0],
    "cave": [1.0, 0.0, 0.0, 5.0, 0.0, 0.0],
    "pet": [4.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animal": [5.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "john": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # Grammar / Function words
    "look": [0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
    "all": [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    "those": [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    "the": [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    "at": [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    "with": [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    "a": [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    "is": [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    "out": [0.0, 0.0, 1.0, 1.0, 0.0, 3.0],
    "of": [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
}


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x**2 for x in v1))
    mag2 = math.sqrt(sum(x**2 for x in v2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0


def softmax(scores):
    """Convert raw similarity scores into normalized attention weights summing to 1."""
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores] if total else [0.0] * len(scores)


def tokenize(sentence):
    """Lowercase sentence, remove punctuation, and extract tokens."""
    cleaned = re.sub(r"['’]s\b", "", sentence.lower())
    return re.findall(r"[a-z]+", cleaned)


def analyze_attention(sentence):
    """Compute and display attention weights of context words relative to target 'bat'/'bats'."""
    tokens = tokenize(sentence)
    target = None
    for t in tokens:
        if t in ("bat", "bats"):
            target = t
            break

    if not target:
        print(f"Error: Sentence '{sentence}' does not contain 'bat' or 'bats'.")
        return

    # Extract all other words in the sentence
    target_idx = tokens.index(target)
    context_words = [w for i, w in enumerate(tokens) if i != target_idx]

    if not context_words:
        print("Sentence must contain words other than the target word.")
        return

    target_vec = WORD_VECTORS[target]
    default_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    similarities = [
        cosine_similarity(target_vec, WORD_VECTORS.get(w, default_vec))
        for w in context_words
    ]
    weights = softmax(similarities)

    print(f"\nSentence: \"{sentence}\"")
    print(f"Target word: '{target}' (Spotlight analyzing surrounding context)")
    print("-" * 50)
    print(f"{'Word':<15} | {'Cosine Similarity':<18} | {'Attention Weight':<16}")
    print("-" * 50)
    for word, sim, wt in zip(context_words, similarities, weights):
        print(f"{word:<15} | {sim:<18.4f} | {wt:<16.4f}")
    print("-" * 50)

    # Plain-English interpretation of top pulling words
    ranked = sorted(zip(context_words, similarities, weights), key=lambda x: x[2], reverse=True)
    top_meaningful = [item for item in ranked if item[1] > 0.1][:2]

    if top_meaningful:
        pulled_by = ", ".join(f"'{w}' ({wt:.1%})" for w, _, wt in top_meaningful)
        print(f"Plain-English Interpretation: The meaning of '{target}' is pulled most strongly by {pulled_by}.")
    else:
        print(f"Plain-English Interpretation: No strong contextual clues found in surrounding words.")


def main():
    print("=== Part 2: Tiny Attention Mechanism Demo ===")
    print("Analogy: Attention acts like a spotlight illuminating context words that pull meaning.")
    print("Note: This is a toy demo, not a full multi-head transformer.\n")

    demo_sentences = [
        "The player swung the bat at the ball",
        "The bat flew out of the cave",
        "John's pet is a bat",
    ]

    print("--- Built-in Demo Sentences ---")
    for demo in demo_sentences:
        analyze_attention(demo)

    print("\n--- Try Your Own Sentence ---")
    print("Enter a sentence containing 'bat' or 'bats' (or press Enter to finish):")
    try:
        user_sent = input("Your sentence: ").strip()
        if user_sent:
            analyze_attention(user_sent)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
