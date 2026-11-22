# was getting hf token warning without this
import os

from memvault.embeddings.base import BaseEmbedder

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


# this prefix yiels better results with the bge model
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder(BaseEmbedder):
    """
    Local embedding provider using BGE-small via sentence transformers

    Downloads the model on first use, cached locally
    RUNS ON CPU by default

    Install the optional dependency
    pip install eviloomemvault[local]

    """
    
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    def __init__(self, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Local embeddings need sentence-transformers. "
                "Install it with: pip install 'eviloomemvault[local]'"
           ) from e  
        self._model = SentenceTransformer(self.MODEL_NAME, device=device)


    def embed(self, text: str) -> list[float]:
        """Embed content text for storage"""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed the query with BGE prefix for better retrieval"""
        vector = self._model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
        )

        return vector.tolist()

    @property
    def dimensions(self) -> int:
        return 384
