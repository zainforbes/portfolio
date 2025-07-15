# tools.py
from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# —– Load your FAISS index + chunks
INDEX_PATH = Path(__file__).parent / "cv.index"
CHUNKS_PATH = Path(__file__).parent / "cv_chunks.pkl"

index = faiss.read_index(str(INDEX_PATH))
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(query: str, k: int = 5):
    q_emb = embedder.encode([query]).astype("float32")
    _, ids = index.search(q_emb, k)
    return [chunks[i] for i in ids[0]]

# —– LLM: drop-in transformers pipeline
#    (this will pull the model from HF the first time you run it)
hf_pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    trust_remote_code=True,
    max_length=512,
    do_sample=True,
    temperature=0.2,
)

llm = HuggingFacePipeline(pipeline=hf_pipe)

def qa_with_context(query: str) -> str:
    snippets = retrieve_chunks(query)
    context = "\n\n".join(snippets)
    prompt = (
        "You are interviewing a candidate. Use ONLY the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return llm(prompt)
