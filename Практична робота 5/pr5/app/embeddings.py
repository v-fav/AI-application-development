import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)

_MODEL = None


def get_model() -> SentenceTransformer:
    global _MODEL

    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)

    return _MODEL


def embed_passages(texts: list[str]) -> np.ndarray:
    model = get_model()

    prepared = [
        f"passage: {text}"
        for text in texts
    ]

    vectors = model.encode(
        prepared,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    model = get_model()

    vector = model.encode(
        [f"query: {text}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(
        vector[0],
        dtype=np.float32,
    )