import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_FOLDER / "output" / "chunks.json"
OUTPUT_FILE = PROJECT_FOLDER / "output" / "embedded_chunks.json"


def main():
    # Step 1: Check that the extracted chunks exist.
    if not INPUT_FILE.exists():
        print("Could not find output/chunks.json.")
        print("Run ingest.py successfully first.")
        return

    # Step 2: Read the chunks.
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        print("chunks.json is empty. Check your PDF extraction.")
        return

    if any(
        not isinstance(chunk.get("text"), str)
        or not chunk["text"].strip()
        for chunk in chunks
    ):
        print("One or more chunks contain missing or empty text.")
        print("Check chunks.json before continuing.")
        return

    texts = [chunk["text"] for chunk in chunks]
    print(f"Loaded {len(texts)} chunks.")

    # Step 3: Load the embedding model.
    print("Loading the model...")
    print("The first run needs internet to download model files.")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    # Step 4: Check that no chunk exceeds the model's input limit.
    tokenized = model.tokenizer(
        texts,
        truncation=False,
        add_special_tokens=True
    )

    too_long = [
        index + 1
        for index, tokens in enumerate(tokenized["input_ids"])
        if len(tokens) > model.max_seq_length
    ]

    if too_long:
        print(
            f"{len(too_long)} chunks exceed the model's "
            f"{model.max_seq_length}-token limit."
        )
        print("First affected record numbers:", too_long[:10])
        print("Reduce the chunk size and run ingest.py again.")
        return

    # Step 5: Convert text into embeddings.
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Step 6: Keep each embedding with its source information.
    records = []

    for chunk, embedding in zip(chunks, embeddings):
        record = dict(chunk)
        record["embedding"] = embedding.tolist()
        records.append(record)

    output_data = {
        "model": MODEL_NAME,
        "normalized": True,
        "embedding_dimension": int(embeddings.shape[1]),
        "chunks": records
    }

    # Step 7: Save the result.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    print("\nEmbeddings created successfully!")
    print(f"Chunks processed: {len(records)}")
    print(f"Numbers per embedding: {embeddings.shape[1]}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()