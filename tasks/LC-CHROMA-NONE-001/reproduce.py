"""Offline reproduction using the public Chroma retriever."""
from unittest.mock import MagicMock
from langchain_chroma import Chroma
from langchain_core.embeddings import FakeEmbeddings

store = Chroma(embedding_function=FakeEmbeddings(size=3), client=MagicMock())
store._chroma_collection.query.return_value = {
    "documents": [["retained content", None]],
    "metadatas": [[{"kind": "valid"}, None]],
    "ids": [["valid-id", "missing-id"]],
    "distances": [[0.1, 0.2]],
}
result = store.as_retriever().invoke("offline query")
assert len(result) == 1 and result[0].page_content == "retained content"
assert result[0].id == "valid-id" and result[0].metadata == {"kind": "valid"}
print("Missing-content retrieval succeeded")
