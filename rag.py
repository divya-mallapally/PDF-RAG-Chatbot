from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.pkl"

print("Loading vector database...")

index = faiss.read_index(str(INDEX_PATH))

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def search_documents(question, k=3):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype("float32")

    distances, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx < len(chunks):
            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "distance": float(distance)
            })

    return results


def ask_question(question):

    results = search_documents(question)

    if not results:
        return "I could not find the answer in the uploaded document."

    context = "\n\n".join(
        f"Page {result['page']}:\n{result['text']}"
        for result in results
    )

    return context


if __name__ == "__main__":
    question = input("Ask a question about your PDF: ")

    results = search_documents(question)

    for result in results:
        print("\n--- Result ---")
        print("Page:", result["page"])
        print(result["text"])
