from pathlib import Path
import time   
import random


def load_documents():
    base_path = Path(__file__).resolve().parent.parent
    file_path = base_path / "data" / "docs.txt"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = content.split("\n\n")
    return chunks


def retrieve_relevant_chunks(question: str, chunks: list[str], k: int = 3):
    question_words = question.lower().split()

    scored_chunks = []

    for chunk in chunks:
        score = sum(word in chunk.lower() for word in question_words)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    return [chunk for score, chunk in scored_chunks[:k] if score > 0]


def get_rag_chain():
    chunks = load_documents()

    def fake_chain(question: str):

        time.sleep(random.uniform(0.8, 1.8)) 

        relevant_chunks = retrieve_relevant_chunks(question, chunks)

        if not relevant_chunks:
            return "No lo sé según los documentos."

        context = "\n\n".join(relevant_chunks)

        return f"""
Basado en los documentos encontrados:

{context}

Respuesta:
Según la información disponible, esta es la respuesta a tu pregunta.
""".strip()

    return fake_chain