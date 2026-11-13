import json
import re
from pathlib import Path

import pymupdf


CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def clean_text(text):
    """Remove unnecessary line breaks and spaces."""

    # Join words separated by a hyphen and line break
    text = re.sub(r"-\n(?=\w)", "", text)

    # Replace repeated spaces and line breaks with one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_chunks(text):
    """Split page text into smaller overlapping sections."""

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Try not to split a word in the middle
        if end < len(text):
            last_space = text.rfind(" ", start, end)

            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def process_manual(pdf_path):
    """Extract text from every page of the manual."""

    document = pymupdf.open(pdf_path)
    manual_name = Path(pdf_path).name
    all_chunks = []

    for page_index, page in enumerate(document):
        page_number = page_index + 1

        raw_text = page.get_text("text", sort=True)
        cleaned_text = clean_text(raw_text)
        page_chunks = split_into_chunks(cleaned_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_information = {
                "id": f"page_{page_number}_chunk_{chunk_index + 1}",
                "document": manual_name,
                "page": page_number,
                "chunk_number": chunk_index + 1,
                "text": chunk_text
            }

            all_chunks.append(chunk_information)

        print(
            f"Page {page_number}: "
            f"{len(cleaned_text)} characters, "
            f"{len(page_chunks)} chunks"
        )

    document.close()
    return all_chunks


def save_chunks(chunks):
    """Create the output folder and save the extracted chunks."""

    project_folder = Path(__file__).resolve().parent
    output_folder = project_folder / "output"

    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "chunks.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"\nThe extracted text was saved to {output_path}")


def main():
    folder = Path(__file__).resolve().parent
    default = folder / "Samsung_user_manual.pdf"

    raw = input(f"PDF path (press Enter for {default.name}): ")
    cleaned = raw.strip().strip('"').strip("'")
    pdf_path = Path(cleaned) if cleaned else default

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        print("Files in this folder:")
        for f in folder.iterdir():
            print("  ", f.name)
        return

    chunks = process_manual(pdf_path)
    save_chunks(chunks)

    print(f"Total number of chunks created: {len(chunks)}")
    print("Step 1 completed successfully!")


if __name__ == "__main__":
    main()