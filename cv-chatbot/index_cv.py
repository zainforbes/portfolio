from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss, pickle, re

def load_pdf(path):
    reader = PdfReader(path)
    return "\n\n".join(p.extract_text() for p in reader.pages)

def chunk_text(text, max_tokens=300):
    sentences = re.split(r'(?<=[\.!?])\s+', text)
    chunks, curr, count = [], [], 0
    for s in sentences:
        t = len(s.split())
        if count + t > max_tokens:
            chunks.append(" ".join(curr)); curr, count = [], 0
        curr.append(s); count += t
    if curr: chunks.append(" ".join(curr))
    return chunks

if __name__ == "__main__":
    raw = load_pdf("data/test_cv.pdf")
    chunks = chunk_text(raw)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    emb = embedder.encode(chunks, show_progress_bar=True).astype("float32")

    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)
    faiss.write_index(idx, "cv.index")

    with open("cv_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Indexed {len(chunks)} chunks.")