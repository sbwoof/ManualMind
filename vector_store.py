import hashlib
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_FOLDER / "output" / "embedded_chunks.json"
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"


def main():
    # 1. Read the embeddings created in the previous step.
    if not INPUT_FILE.exists():
        print("Could not find output/embedded_chunks.json.")
        print("Run embed.py successfully first.")
        return

    file_bytes = INPUT_FILE.read_bytes()
    data = json.loads(file_bytes)
    chunks = data["chunks"]

    if not chunks:
        print("The file contains no chunks.")
        return

    expected_dimension = data["embedding_dimension"]

    if any(
        len(chunk["embedding"]) != expected_dimension
        for chunk in chunks
    ):
        print("Some embeddings have an incorrect length.")
        print("Run embed.py again.")
        return

    # 2. Open a database stored on your computer.
    client = chromadb.PersistentClient(
        path=str(DATABASE_FOLDER)
    )

    # Different input files get separate collections.
    # Rerunning the same file reuses the same collection.
    file_fingerprint = hashlib.sha256(file_bytes).hexdigest()[:16]
    collection_name = f"manualmind_{file_fingerprint}"

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}}
    )

    # 3. Save chunks in small batches.
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        collection.upsert(
            ids=[
                f"record_{start + index}"
                for index in range(len(batch))
            ],
            documents=[chunk["text"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[
                {
                    "document": chunk["document"],
                    "page": int(chunk["page"]),
                    "chunk_number": int(chunk["chunk_number"])
                }
                for chunk in batch
            ]
        )

    print(f"\nStored {collection.count()} chunks in ChromaDB.")
    print(f"Database location: {DATABASE_FOLDER}")

    # 4. Load the SAME model used to embed the manual.
    print("\nLoading the model for questions...")

    model = SentenceTransformer(data["model"], device="cpu")

    print("\nReady! Ask a question about your manual.")
    print("Type exit to stop.")

    # 5. Keep accepting questions until you type exit.
    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        question_tokens = model.tokenizer(
            question,
            truncation=False
        )["input_ids"]

        if len(question_tokens) > model.max_seq_length:
            print("Please shorten the question and try again.")
            continue

        question_embedding = model.encode(
            question,
            normalize_embeddings=data["normalized"],
            convert_to_numpy=True
        )

        # 6. Retrieve up to four matching passages.
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(4, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        # 7. Display the matching passages and source pages.
        print("\nMatching manual passages:")

        for index, text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][index]
            distance = results["distances"][0][index]

            print(f"\n--- Result {index + 1} ---")
            print(f"Manual: {metadata['document']}")
            print(f"PDF page: {metadata['page']}")
            print(f"Cosine distance: {distance:.4f}")
            print(text)


if __name__ == "__main__":
    main()