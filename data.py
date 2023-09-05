import itertools
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

DATASET_PATH = Path(
    "/home/james/Thesis/jysdoran-CodeXGLUE/Text-Code/NL-code-search-Adv/dataset"
)
SYNTH_DATASET_PATH = Path("/home/james/Thesis/LLM-data-synthesis/datasets")


@dataclass
class DatasetConfig:
    name: str
    path: Path


DATASET_TEST = DatasetConfig("test", DATASET_PATH / "test.jsonl")
DATASET_TRAIN = DatasetConfig("train", DATASET_PATH / "train.jsonl")

SYNTH_DATASET_C2D = DatasetConfig(
    "c2d_semisynthetic", SYNTH_DATASET_PATH / "c2d_semisynthetic.jsonl"
)
SYNTH_DATASET_D2C = DatasetConfig(
    "d2c_semisynthetic", SYNTH_DATASET_PATH / "d2c_semisynthetic.jsonl"
)
SYNTH_DATASET_HARDNEG = DatasetConfig(
    "hardnegative_semisynthetic",
    SYNTH_DATASET_PATH / "hardnegative_semisynthetic.jsonl",
)
SYNTH_DATASET_HARDPOS = DatasetConfig(
    "hardpositive_semisynthetic",
    SYNTH_DATASET_PATH / "hardpositive_semisynthetic.jsonl",
)
SYNTH_DATASET_SUBCONCEPTS = DatasetConfig(
    "subconcepts_8_synthetic", SYNTH_DATASET_PATH / "subconcepts_8_synthetic.jsonl"
)

SYNTHETIC_DATASETS = (
    SYNTH_DATASET_C2D,
    SYNTH_DATASET_D2C,
    SYNTH_DATASET_HARDNEG,
    SYNTH_DATASET_HARDPOS,
    SYNTH_DATASET_SUBCONCEPTS,
)


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch


class JSONLDataset(list):
    """Class for managing a dataset stored in a JSONL file."""

    def __init__(self, path: Optional[Path] = None):
        self.path = None
        if path is not None:
            self.load_jsonl(path)

    def load_jsonl(self, path):
        self.clear()
        self.path = path
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                self.append(js)

    def save_jsonl(self, path=None):
        path = path if path is not None else self.path
        self.path = path
        with open(self.path, "w") as f:
            for js in self:
                f.write(json.dumps(js) + "\n")

    def update_range(self, new_dicts, start=0, end=None):
        if end is None:
            end = start + len(new_dicts)
        assert len(new_dicts) == end - start
        self[start:end] = new_dicts


@dataclass
class CodeSearchAdvExample:
    """A single training example for the CodeSearchAdv dataset."""

    idx: int
    func_name: str
    original_string: str
    code: str
    docstring: str
    code_tokens: List[str]
    docstring_tokens: List[str]

    def __init__(self, js):
        self.idx = js["idx"]
        self.func_name = js["func_name"]
        self.original_string = js["original_string"]
        self.code = js["code"]
        self.docstring = js["docstring"]
        self.code_tokens = js["code_tokens"]
        self.docstring_tokens = js["docstring_tokens"]


class CodeSearchAdvDataset(list):
    def __init__(self, path: Optional[Path] = None):
        super().__init__()
        if path is not None:
            self.load_jsonl(path)
        self.offset = 0

    def load_jsonl(
        self,
        path=DATASET_PATH / "train.jsonl",
        start=0,
        end=None,
    ):
        self.clear()
        with open(path) as f:
            for line in itertools.islice(f, start, end):
                line = line.strip()
                js = json.loads(line)
                # assert js["language"] == "python"
                self.append(CodeSearchAdvExample(js))
            self.offset = start

    def __getitem__(self, item):
        if isinstance(item, slice):
            raise NotImplementedError
        return super().__getitem__(item - self.offset)

    def save_jsonl(self, path):
        if self.offset > 0:
            raise NotImplementedError
        with open(path, "w") as f:
            for example in self:
                f.write(json.dumps(example.__dict__) + "\n")


def save_dicts_as_jsonl(dicts: List[Dict], path):
    with open(path, "w") as f:
        for data_dict in dicts:
            f.write(json.dumps(data_dict) + "\n")


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, url, idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.idx = idx


def convert_examples_to_features(js, tokenizer, args):
    # code
    if "code_tokens" in js:
        code = " ".join(js["code_tokens"])
    else:
        code = " ".join(js["function_tokens"])
    code_tokens = tokenizer.tokenize(code)[: args.block_size - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.block_size - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = " ".join(js["docstring_tokens"])
    nl_tokens = tokenizer.tokenize(nl)[: args.block_size - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.block_size - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js["url"], js["idx"])


def load_data_codesearch_adv(
    file_path=DATASET_PATH / "train.jsonl",
    start=0,
    end=None,
) -> List[CodeSearchAdvExample]:
    examples = []
    with open(file_path) as f:
        for line in itertools.islice(f, start, end):
            line = line.strip()
            js = json.loads(line)
            assert js["language"] == "python"
            examples.append(CodeSearchAdvExample(js))

    return examples


def main():
    dataset = CodeSearchAdvDataset()
    dataset.load_jsonl(DATASET_PATH / "train.jsonl", start=9, end=10)

    for example in dataset:
        print(example)
    # dataset[:10].save_jsonl("slice.jsonl")


if __name__ == "__main__":
    main()
