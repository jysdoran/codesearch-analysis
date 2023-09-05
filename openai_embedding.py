from dataclasses import dataclass
import itertools
from pathlib import Path
import re
from typing import Iterable, List

import numpy as np
import openai
from openai.embeddings_utils import get_embeddings
import tiktoken


from data import (
    batched,
    JSONLDataset,
    DatasetConfig,
    DATASET_TEST,
    SYNTHETIC_DATASETS,
    DATASET_TRAIN,
)

openai.api_key_path = "/your/key/here"


EMBEDDING_DIR = Path("./embeddings/")
CODE_FILENAME = "code.npy"
DOCSTRING_FILENAME = "docstring.npy"


@dataclass
class EmbeddingModel:
    name: str
    dims: int
    max_tokens: int = 8191


# Recommended and cheaper
text_embedding_ada_002 = EmbeddingModel(
    "text-embedding-ada-002", 1536
)  # $0.0001 / 1K tokens

"""With the -001 text embeddings (not -002, and not code embeddings), we suggest replacing newlines (\n)
in your input with a single space, as we have seen worse results when newlines are present."""
# code_search_ada_code_001 = EmbeddingModel(
#     "code-search-ada-code-001", 1536
# )  # $0.004 / 1K tokens
# code_search_ada_text_001 = EmbeddingModel("code-search-ada-text-001", 1536)

# code_search_babbage_code_001 = EmbeddingModel(
#     "code-search-babbage-code-001", 1536
# )  # $0.005 / 1K tokens
# code_search_babbage_text_001 = EmbeddingModel("code-search-babbage-text-001", 1536)


def remove_docstring(code: str):
    return re.sub(r'"""[\s\S]*?"""', "", code, count=1)


def get_embedding_dir(
    dataset_config: DatasetConfig, model: EmbeddingModel = text_embedding_ada_002
):
    return EMBEDDING_DIR / model.name / dataset_config.name


def load_embeddings(
    dataset_config: DatasetConfig, model: EmbeddingModel = text_embedding_ada_002
):
    embedding_dir = get_embedding_dir(dataset_config, model)
    try:
        code_embeddings = np.load(embedding_dir / CODE_FILENAME)
    except FileNotFoundError:
        code_embeddings = None
    try:
        docstring_embeddings = np.load(embedding_dir / DOCSTRING_FILENAME)
    except FileNotFoundError:
        docstring_embeddings = None

    return code_embeddings, docstring_embeddings


def embed_list_long(
    dataset: Iterable[str],
    model: EmbeddingModel = text_embedding_ada_002,
    batch_size: int = 2048,
):
    assert batch_size <= 2048, "The batch size should not be larger than 2048."
    embeddings = []
    for batch in batched(dataset, n=batch_size):
        try:
            embeddings.append(np.array(get_embeddings(batch, engine=model.name)))

        except Exception as e:
            print("Failed to embed batch", e)
            embeddings.append(np.full((len(batch), model.dims), np.nan))

    return np.concatenate(embeddings)


def get_code(example):
    code = example.get("function")
    if code is None:
        code = example.get("code")

    if not code:
        raise ValueError("No code found in example", example)

    return code


def get_docstring_summary(example):
    docstring = example.get("docstring_summary")
    if docstring is None:
        tokens = example.get("docstring_tokens")
        if tokens is not None:
            docstring = " ".join(tokens)
    
    if not docstring:
        raise ValueError("No docstring found in example", example)
                         
    return docstring


def clip_token_lengths(s, encoding: tiktoken.Encoding, max_tokens):
    tokens = encoding.encode(s)

    n_tokens = len(tokens)
    if n_tokens > max_tokens:
        raise ValueError("Input is too long", s)
        s = s[:max_tokens]
        tokens = tokens[:max_tokens]
        s = encoding.decode(tokens)
    return s


def embed_dataset(
    dataset_config,
    limit=12800,
    embed_code: bool = False,
    embed_docstring: bool = False,
    model=text_embedding_ada_002,
):
    dataset = JSONLDataset(dataset_config.path)

    embedding_dir = get_embedding_dir(dataset_config, model)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    encoding = tiktoken.encoding_for_model(model.name)

    def clip_with_encoding(s):
        return clip_token_lengths(s, encoding, model.max_tokens)

    if embed_code:
        codes = list(
            map(
                clip_with_encoding,
                map(remove_docstring, map(get_code, itertools.islice(dataset, limit))),
            )
        )

        result = embed_list_long(codes, model)
        np.save(embedding_dir / CODE_FILENAME, result)

    if embed_docstring:
        docstrings = list(
            map(
                clip_with_encoding,
                map(get_docstring_summary, itertools.islice(dataset, limit)),
            )
        )

        result = embed_list_long(docstrings, model)
        np.save(embedding_dir / DOCSTRING_FILENAME, result)


def embed_test_dataset():
    embed_dataset(DATASET_TEST, limit=None, embed_code=True, embed_docstring=True)


def embed_all_datasets():
    datasets = (DATASET_TRAIN,) + SYNTHETIC_DATASETS

    for dataset_config in datasets[-1:]:
        print("Embedding dataset:", dataset_config)
        embed_dataset(
            dataset_config, limit=12800, embed_code=True, embed_docstring=True
        )


if __name__ == "__main__":
    # embed_test_dataset()
    embed_all_datasets()
