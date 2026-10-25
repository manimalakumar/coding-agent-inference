"""Independent offline acceptance checks; never provided to the coding agent."""

from copy import deepcopy
from unittest.mock import MagicMock
import numpy as np
import pytest
from langchain_chroma import Chroma
from langchain_chroma.vectorstores import (
    _results_to_docs_and_scores,
    _results_to_docs_and_vectors,
)
from langchain_core.embeddings import FakeEmbeddings


@pytest.mark.parametrize(
    "documents", [[None, "kept", ""], [None, None, None], ["first", "second", "third"]]
)
@pytest.mark.parametrize("kind", ["scores", "vectors", "ids", "retriever"])
def test_retrieval(documents, kind):
    metadata = [{} if kind == "ids" else None, {"n": 2}, {"n": 3}]
    ids = ["a", "b", "c"]
    vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    scores = [0.1, 0.2, 0.3]
    data = {
        "documents": [documents],
        "metadatas": [metadata],
        "ids": [ids],
        "distances": [scores],
        "embeddings": [vectors],
    }
    original = deepcopy(data)
    store = Chroma(client=MagicMock(), embedding_function=FakeEmbeddings(size=2))
    if kind == "scores":
        output = _results_to_docs_and_scores(data)
    elif kind == "vectors":
        output = _results_to_docs_and_vectors(data)
    elif kind == "ids":
        store._chroma_collection.get.return_value = {
            "documents": documents,
            "metadatas": metadata,
            "ids": ids,
        }
        output = [(d, None) for d in store.get_by_ids(ids)]
    else:
        store._chroma_collection.query.return_value = data
        output = [(d, None) for d in store.as_retriever().invoke("offline")]
    expected = [i for i, d in enumerate(documents) if d is not None]
    assert len(output) == len(expected)
    for (document, associated), i in zip(output, expected):
        assert document.page_content == documents[i]
        assert document.metadata == (metadata[i] or {})
        assert document.id == ids[i]
        if kind == "scores":
            assert associated == scores[i]
        if kind == "vectors":
            np.testing.assert_array_equal(associated, vectors[i])
    assert data == original
