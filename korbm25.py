import logging
from typing import List, Optional, cast

import numpy as np
from konlpy.tag import Kkma, Okt
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:
    raise ImportError("Could not import rank_bm25. Install it with `pip install rank-bm25`.") from exc

try:
    from kiwipiepy import Kiwi
except ImportError as exc:
    raise ImportError(
        "Could not import kiwipiepy. Install it with `pip install kiwipiepy`."
    ) from exc


logger = logging.getLogger(__name__)

okt = Okt()
kkma = Kkma()
kiwi_tokenizer = Kiwi()


def tokenize_kiwi(text: str) -> List[str]:
    return [token.form for token in kiwi_tokenizer.tokenize(text)]


def tokenize_okt(text: str) -> List[str]:
    return okt.morphs(text)


def tokenize_kkma(text: str) -> List[str]:
    return kkma.morphs(text)


TOKENIZERS = {
    "okt": tokenize_okt,
    "kkma": tokenize_kkma,
    "kiwi": tokenize_kiwi,
}


class KorBM25Retriever(BaseRetriever):
    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        mode: str = "okt",
    ) -> None:
        self.similarity_top_k = similarity_top_k

        if nodes is None:
            raise ValueError("Please pass nodes.")

        if mode not in TOKENIZERS:
            supported = ", ".join(TOKENIZERS)
            raise ValueError(f"Unsupported mode: {mode}. Choose one of: {supported}.")

        self.mode = mode
        self.tokenize = TOKENIZERS[mode]
        self.corpus = [node_to_metadata_dict(node) for node in nodes]
        self.corpus_tokens = [self.tokenize(node.get_content()) for node in nodes]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        mode: str = "okt",
    ) -> "KorBM25Retriever":
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert nodes is not None

        return cls(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            mode=mode,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][: self.similarity_top_k]

        nodes: List[NodeWithScore] = []
        for idx in top_n:
            node_dict = self.corpus[int(idx)]
            node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=float(scores[idx])))

        return nodes
