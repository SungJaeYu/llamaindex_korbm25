# llamaindex_korbm25

LlamaIndex에서 한국어 BM25 검색을 사용하기 위해 만든 커스텀 Retriever입니다.

이 Retriever는 **공군 2024년 해커톤(AI 프로그램)** 참가 프로젝트의 RAG 검색 성능을 개선하는 과정에서 개발했으며, 해당 프로젝트는 **장려상**을 수상했습니다.

## 한국어

### 배경

공군 2024년 해커톤(AI 프로그램)을 준비하면서 LlamaIndex 기반의 한국어 RAG 파이프라인을 구성했습니다. 당시 LlamaIndex의 BM25 Retriever는 한국어 형태소 분석기를 자연스럽게 적용하기 어려웠고, 한국어는 조사와 어미가 단어에 결합되는 특성이 있어 단순한 토큰화만으로는 lexical retrieval 품질이 떨어질 수 있었습니다.

이 문제를 해결하기 위해 LlamaIndex의 `BaseRetriever` 인터페이스를 유지하면서 BM25 점수 계산 전에 한국어 형태소 분석을 적용하는 커스텀 Retriever를 구현했습니다.

지원 형태소 분석기:

- Okt
- Kkma
- Kiwi

### 해결한 문제

목표는 LlamaIndex 자체를 대체하는 것이 아니라, 실제 한국어 RAG 파이프라인에서 발생한 검색 문제를 해결하는 것이었습니다.

1. LlamaIndex node를 입력받습니다.
2. 한국어 형태소 분석기로 문서를 토큰화합니다.
3. `rank_bm25`를 사용해 BM25 index를 구성합니다.
4. 검색 결과를 LlamaIndex의 `NodeWithScore` 형식으로 반환합니다.

### 사용 예시

```python
retriever = KorBM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5,
    mode="kiwi",
)

results = retriever.retrieve("한국어 검색 예시")
```

`mode`는 `"okt"`, `"kkma"`, `"kiwi"` 중 하나를 사용할 수 있습니다.

### 프로젝트 결과

이 Retriever는 **공군 2024년 해커톤(AI 프로그램)** 참가 프로젝트의 한국어 RAG 검색 파이프라인을 개선하는 과정에서 개발했습니다.

해당 프로젝트는 대회에서 **장려상**을 수상했습니다.

이 저장소는 대회 전체 프로젝트가 아니라, 당시 검색 성능 문제를 해결하기 위해 별도로 구현했던 BM25 Retriever 코드를 정리한 것입니다.

### 현재 상태

이 저장소는 현재 운영 환경에서 그대로 사용하기 위한 최신 BM25 패키지라기보다, **2024년 당시 한국어 retrieval 문제를 발견하고 해결했던 과정을 기록하기 위한 프로젝트**로 유지하고 있습니다.

그 이후 LlamaIndex와 BM25 생태계는 크게 발전했습니다. 현재는 persistence, metadata filtering, hybrid retrieval, reranking, tokenizer customization 등 더 유연한 기능을 제공하는 도구들이 있으므로, 새로운 시스템을 구축한다면 최신 LlamaIndex/BM25 생태계를 먼저 검토하는 것이 적절합니다.

이 프로젝트의 현재 가치는 특정 프레임워크를 단순 사용한 것이 아니라,

> 한국어 검색 성능 문제를 발견하고 → 원인을 tokenization으로 좁히고 → 기존 LlamaIndex 인터페이스와 호환되는 Retriever를 직접 구현했다는 점

에 있습니다.

### 현대적인 확장 방향

이 프로젝트를 현재 기준으로 확장한다면 다음 검색 방식들을 동일한 한국어 데이터셋에서 비교하는 형태가 적절합니다.

- 기본 BM25
- Kiwi 기반 BM25
- Dense Retrieval
- BM25 + Dense Hybrid Retrieval
- Hybrid Retrieval + Reranker

이를 통해 현대적인 RAG 환경에서도 한국어 형태소 분석이 lexical retrieval 품질에 어느 정도 영향을 주는지 정량적으로 비교할 수 있습니다.

---

## English

A custom Korean BM25 retriever for LlamaIndex.

This retriever was developed while improving the RAG retrieval pipeline for a project submitted to the **2024 Republic of Korea Air Force Hackathon (AI Program)**. The project received an **Honorable Mention**.

### Background

While preparing for the 2024 Republic of Korea Air Force Hackathon (AI Program), I built a Korean RAG pipeline based on LlamaIndex. At the time, LlamaIndex's BM25 retrieval path was not convenient for Korean morphological tokenization. Because Korean particles and inflections are attached to words, simple tokenization can reduce lexical retrieval quality.

To address this, I implemented a small LlamaIndex-compatible `BaseRetriever` that applies Korean morphological analysis before BM25 scoring.

Supported tokenizers:

- Okt
- Kkma
- Kiwi

### What this project solved

The goal was not to replace LlamaIndex, but to solve a concrete retrieval problem encountered in a Korean RAG pipeline:

1. Receive LlamaIndex nodes.
2. Tokenize Korean text with a morphological analyzer.
3. Build a BM25 index with `rank_bm25`.
4. Return `NodeWithScore` results through the LlamaIndex retriever interface.

### Example

```python
retriever = KorBM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5,
    mode="kiwi",
)

results = retriever.retrieve("한국어 검색 예시")
```

`mode` can be `"okt"`, `"kkma"`, or `"kiwi"`.

### Competition result

This retriever was developed as part of the effort to improve Korean RAG retrieval for a project submitted to the **2024 Republic of Korea Air Force Hackathon (AI Program)**.

The project received an **Honorable Mention**.

This repository contains the BM25 retriever implementation created to address the retrieval issue; it is not the full hackathon project.

### Historical / legacy status

This repository is kept primarily as a record of the retrieval problem and the solution implemented in 2024.

Modern LlamaIndex and BM25 tooling have evolved significantly since then, with more flexible retrieval pipelines, persistence, filtering, hybrid retrieval, reranking, and tokenizer customization. For a new production system, I would evaluate the current LlamaIndex/BM25 ecosystem first instead of using this implementation as-is.

The useful part of this repository today is the engineering story: identifying a Korean retrieval limitation in an existing framework, tracing it to tokenization, and implementing a compatible retriever to solve it.

### Possible modern follow-up

A current extension of this work would be to benchmark Korean retrieval approaches such as:

- vanilla BM25
- Kiwi-based BM25
- dense retrieval
- hybrid BM25 + dense retrieval
- hybrid retrieval + reranking

That would make it possible to measure where Korean morphological tokenization still helps in a modern RAG stack.
