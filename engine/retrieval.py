"""A tiny vector database, built from scratch: chunk text, turn each chunk
into a TF-IDF vector, and rank chunks against a query by cosine similarity.

No FAISS, no Chroma, no external vector store, just numpy, so every number in
the ranking is something you can trace by hand. This is the same idea a
production vector database runs (embed text, store as vectors, rank by
similarity), just with a from-scratch, hand-computable embedding (word counts
weighted by how rare a word is) instead of a neural one. A neural embedding
model would place semantically similar chunks near each other even when they
share no words; TF-IDF only catches literal word overlap, that's the
trade-off for staying readable end to end. See ROADMAP.md for swapping in a
real embedding model once this project points at a real base model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Splits text into overlapping, word-boundary-aligned character windows.
    The overlap keeps a sentence that straddles a chunk boundary from being
    unfindable from either side. Snapping each edge to the nearest word
    boundary keeps a chunk from starting or ending mid-word — a fragment like
    "...loved the lighthouse. T" doesn't just read as broken, it glues onto
    whatever gets appended after it (a query, a prompt) just as unnaturally,
    which measurably hurt the retrieval-augmented generation demo before this
    fix: the model was reading a spliced-together word fragment it had never
    seen, not a natural continuation.
    """
    text = text.strip()
    if not text:
        return []

    step = max(1, chunk_size - overlap)
    n = len(text)
    chunks = []
    for start in range(0, n, step):
        end = min(start + chunk_size, n)
        words = text[start:end].split()
        if len(words) > 1:
            mid_word_start = start > 0 and not text[start - 1].isspace() and not text[start].isspace()
            mid_word_end = end < n and not text[end - 1].isspace() and not text[end].isspace()
            if mid_word_start:
                words = words[1:]
            if mid_word_end and len(words) > 1:
                words = words[:-1]
        chunk = " ".join(words).strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
    return chunks


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class TfidfIndex:
    """One TF-IDF vector per chunk, ranked against a query vector by cosine
    similarity. `build()` does the indexing (like inserting documents into a
    vector database); `search()` does the query (like a similarity search)."""

    chunks: list[str] = field(default_factory=list)
    vocab: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray | None = None
    vectors: np.ndarray | None = None  # (n_chunks, vocab_size), L2-normalized

    @classmethod
    def build(cls, chunks: list[str]) -> "TfidfIndex":
        docs_words = [_words(c) for c in chunks]
        vocab: dict[str, int] = {}
        for words in docs_words:
            for w in set(words):
                vocab.setdefault(w, len(vocab))

        n_docs = len(chunks)
        doc_freq = np.zeros(len(vocab))
        tf = np.zeros((n_docs, len(vocab)))
        for i, words in enumerate(docs_words):
            for w in words:
                tf[i, vocab[w]] += 1
            for w in set(words):
                doc_freq[vocab[w]] += 1
            if tf[i].sum() > 0:
                tf[i] /= tf[i].sum()

        # smoothed idf (+1 everywhere) so a word in every chunk still gets a
        # small positive weight instead of collapsing to zero
        idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1
        vectors = tf * idf
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        return cls(chunks=chunks, vocab=vocab, idf=idf, vectors=vectors)

    def _embed_query(self, query: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        words = [w for w in _words(query) if w in self.vocab]
        for w in words:
            vec[self.vocab[w]] += 1
        if vec.sum() > 0:
            vec /= vec.sum()
        vec = vec * self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Returns up to top_k (chunk, cosine_similarity) pairs, best match
        first. Chunks with zero similarity (no shared vocabulary at all) are
        left out rather than padding the results with unrelated text."""
        if not self.chunks:
            return []
        query_vec = self._embed_query(query)
        scores = self.vectors @ query_vec  # both sides L2-normalized, so dot product is cosine similarity
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def _cli() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Search a text file with a from-scratch TF-IDF vector index.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    text = Path(args.input).read_text()
    index = TfidfIndex.build(chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap))
    results = index.search(args.query, top_k=args.top_k)

    if not results:
        print("no matching chunks")
        return
    for chunk, score in results:
        print(f"[{score:.3f}] {chunk[:200]}")


if __name__ == "__main__":
    _cli()
