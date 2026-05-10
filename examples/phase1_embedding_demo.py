from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


def print_similarity_table(sentences: list[str], similarity_matrix: np.ndarray) -> None:
    header = "      |   s1 |   s2 |   s3"
    print(header)
    print("-" * len(header))
    for index, row in enumerate(similarity_matrix, start=1):
        formatted_row = " | ".join(f"{value:0.3f}" for value in row)
        print(f"s{index:<5} | {formatted_row}")

    print("\nLegend:")
    for index, sentence in enumerate(sentences, start=1):
        print(f"  s{index}: {sentence}")


def main() -> None:
    sentences = [
        "Alice leads the climate research project.",
        "Alice is in charge of the climate study.",
        "Bananas are rich in potassium.",
    ]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, normalize_embeddings=True)
    similarity_matrix = embeddings @ embeddings.T

    print_similarity_table(sentences, similarity_matrix)


if __name__ == "__main__":
    main()