from pathlib import Path
import fitz

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Documents" / "My_notes.pdf"

def load_pdf():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)
    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()

        pages.append({
            "text": text,
            "page": page_number
        })

    doc.close()
    return pages

if __name__ == "__main__":
    pages = load_pdf()

    print("PDF loaded successfully!")
    print(f"Total pages: {len(pages)}")

    if pages:
        print("\nFirst page text:")
        print(pages[0]["text"][:1000])
