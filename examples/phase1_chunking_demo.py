from __future__ import annotations


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    words = text.split()
    step = size - overlap
    return [
        " ".join(words[start : start + size])
        for start in range(0, len(words), step)
        if words[start : start + size]
    ]


def main() -> None:
    sample = (
        "Alice joined the city climate lab in 2021 and later led the flood-risk "
        "project with Bob. Together they published a report on river safety."
    )

    print("No overlap:")
    for chunk in chunk_words(sample, size=8, overlap=0):
        print(f"  - {chunk}")

    print("\nWith overlap:")
    for chunk in chunk_words(sample, size=8, overlap=2):
        print(f"  - {chunk}")


if __name__ == "__main__":
    main()