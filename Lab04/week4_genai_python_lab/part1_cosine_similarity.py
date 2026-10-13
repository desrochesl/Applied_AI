"""
part1_cosine_similarity.py
A beginner-friendly demonstration of cosine similarity in a 2D meaning-space.
Note: This is a tiny toy teaching model, not a real production embedding model.
"""

import math


def dot_product(v1, v2):
    """Calculate the dot product of two vectors: sum of element-wise products."""
    return sum(x * y for x, y in zip(v1, v2))


def magnitude(v):
    """Calculate the Euclidean length (magnitude) of a vector."""
    return math.sqrt(sum(x**2 for x in v))


def cosine_similarity(v1, v2):
    """Compute cosine similarity: dot(v1, v2) / (|v1| * |v2|)."""
    mag1, mag2 = magnitude(v1), magnitude(v2)
    return dot_product(v1, v2) / (mag1 * mag2) if mag1 and mag2 else 0.0


def explain_similarity(score):
    """Provide a plain-English explanation for a cosine similarity score."""
    if score >= 0.8:
        return "High similarity: Vectors point in almost the same direction."
    elif score >= 0.3:
        return "Moderate similarity: Vectors share some directional alignment."
    elif score > 0.0:
        return "Low similarity: Weak relationship."
    else:
        return "Zero/Negative similarity: Vectors point in orthogonal or opposite directions."


def parse_vector(text):
    """Parse a comma-separated string into a list of floats."""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    print("=== Part 1: Cosine Similarity in Meaning-Space [mammal, sport] ===")
    print("Note: This is a tiny teaching model, not a real production embedding model.\n")

    # Built-in toy vectors
    vectors = {
        "bat_animal": [4, 1],
        "bat_sport": [1, 4],
        "cave": [5, 0],
        "baseball": [0, 5],
        "dog": [5, 0],
        "ball": [0, 5],
    }

    pairs = [
        ("bat_animal (4,1)", "cave (5,0)", vectors["bat_animal"], vectors["cave"]),
        ("bat_sport (1,4)", "baseball (0,5)", vectors["bat_sport"], vectors["baseball"]),
        ("dog (5,0)", "ball (0,5)", vectors["dog"], vectors["ball"]),
    ]

    print("--- Built-in Examples ---")
    for name1, name2, v1, v2 in pairs:
        sim = cosine_similarity(v1, v2)
        print(f"{name1} vs {name2} -> Cosine Similarity: {sim:.4f}")
        print(f"  Explanation: {explain_similarity(sim)}\n")

    print("--- Try Your Own Vectors ---")
    print("Enter two comma-separated vectors (e.g., '1,4' and '4,1') or press Enter to skip.")
    try:
        raw1 = input("Enter Vector A: ").strip()
        if raw1:
            raw2 = input("Enter Vector B: ").strip()
            v1, v2 = parse_vector(raw1), parse_vector(raw2)
            if len(v1) != len(v2):
                print(f"Error: Vector lengths do not match ({len(v1)} vs {len(v2)}).")
            elif not v1:
                print("Error: Vectors cannot be empty.")
            else:
                sim = cosine_similarity(v1, v2)
                print(f"Cosine Similarity between {v1} and {v2}: {sim:.4f}")
                print(f"Explanation: {explain_similarity(sim)}")
    except ValueError:
        print("Error: All vector values must be numeric numbers separated by commas.")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
