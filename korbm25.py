import logging
from typing import Any, Callable, List, Optional, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("could not import")

from konlpy.tag import Kkma, Okt

import numpy as np


okt = Okt()
kkma = Kkma()

try:
    from kiwipiepy import Kiwi
except ImportError:
    raise ImportError(
        "Could not import kiwipiepy, please install with `pip install " "kiwipiepy`."
    )

kiwi_tokenizer = Kiwi()


def tokenize_kiwi(text: str) -> List[str]:
    return [token.form for token in kiwi_tokenizer.tokenize(text)]

def tokenize_okt(text: str) -> List[str]:
    return [token for token in okt.morphs(text)]

def tokenize_kkma(text: str) -> List[str]:
    return [token for token in kkma.morphs(text)]


logger = logging.getLogger(__name__)


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
            raise ValueError("Please pass nodes or an existing BM25 object.")

        self.corpus = [node_to_metadata_dict(node) for node in nodes]
        if mode == 'okt':
            self.corpus_tokens = [tokenize_okt(node.get_content()) for node in nodes]
        elif mode == 'kkma':
            self.corpus_tokens = [tokenize_kkma(node.get_content()) for node in nodes]
        elif mode == 'kiwi':
            self.corpus_tokens = [tokenize_kiwi(node.get_content()) for node in nodes]
        else:
            pass
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
        mode: str = 'okt',
    ) -> "KorBM25Retriever":

        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        return cls(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            mode=mode,
        )


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        tokenized_query = tokenize_kkma(query)

        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:self.similarity_top_k]
        nodes: List[NodeWithScore] = []
        for idx in top_n:
            # idx can be an int or a dict of the node
            if isinstance(idx, dict):
                node = metadata_dict_to_node(idx)
            else:
                node_dict = self.corpus[int(idx)]
                node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=scores[idx]))

        return nodes