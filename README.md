# llamaindex_korbm25

Korean BM25 retriever for LlamaIndex, originally built in 2024 while preparing for an AI competition.

## Background

At the time, LlamaIndex's BM25 retrieval path was not convenient for Korean morphological tokenization. For Korean text, whitespace tokenization alone often fragments retrieval quality because particles and inflections remain attached to words.

This project implemented a small LlamaIndex-compatible `BaseRetriever` that applies Korean morphological analyzers before BM25 scoring.

Supported tokenizers:

- Okt
- Kkma
- Kiwi

## What this project solved

The goal was not to replace LlamaIndex, but to fix a concrete retrieval problem encountered in a Korean RAG pipeline:

1. Receive LlamaIndex nodes.
2. Tokenize Korean text with a morphological analyzer.
3. Build a BM25 index with `rank_bm25`.
4. Return `NodeWithScore` results through the LlamaIndex retriever interface.

## Example

```python
retriever = KorBM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5,
    mode="kiwi",
)

results = retriever.retrieve("한국어 검색 예시")
```

`mode` can be `"okt"`, `"kkma"`, or `"kiwi"`.

## Historical / legacy status

This repository is kept primarily as a record of the retrieval problem and the solution implemented in 2024.

Modern LlamaIndex and BM25 tooling have evolved significantly since then, with more flexible retrieval pipelines, persistence, filtering, hybrid retrieval, reranking, and tokenizer customization. For a new production system, I would evaluate the current LlamaIndex/BM25 ecosystem first instead of using this implementation as-is.

The useful part of this repository today is the engineering story: identifying a Korean retrieval limitation in an existing framework, tracing it to tokenization, and implementing a compatible retriever to solve it.

## Possible modern follow-up

A current extension of this work would be to benchmark Korean retrieval approaches such as:

- vanilla BM25
- Kiwi-based BM25
- dense retrieval
- hybrid BM25 + dense retrieval
- hybrid retrieval + reranking

That would make it possible to measure where Korean morphological tokenization still helps in a modern RAG stack.
