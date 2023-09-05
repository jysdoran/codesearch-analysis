from functools import lru_cache, cached_property
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import wandb

from vector_rank import *

PROJECT_ID = "jysdoran/codebert-search-Adv-grid"


def get_runs():
    api = wandb.Api()

    return list(api.runs(PROJECT_ID, filters=None, per_page=200))


def make_wandb_df():
    runs = get_runs()
    # Extract rundata to dataframe
    config_keys = set()
    summary_keys = set()
    for run in runs:
        config_keys.update(run.config.keys())
        summary_keys.update(run.summary.keys())

    data = {k: [] for k in ("run_name", "run_id", "created_at", *config_keys, *summary_keys)}
    for run in runs:
        data["run_name"].append(run.name)
        data["run_id"].append(run.id)
        data["created_at"].append(run.created_at)
        for k in config_keys:
            data[k].append(run.config.get(k))
        for k in summary_keys:
            data[k].append(run.summary.get(k))

    # create a pandas dataframe
    wandb_df = pd.DataFrame(data)

    return wandb_df


class RunData:
    RANK_TRUNC = 100
    CACHED_PROPERTIES = (
        "predictions",
        "code_vecs",
        "nl_vecs",
        "scores",
        "ranks",
        "correct_rank",
    )
    TEST_OFFSET = 261424

    def __init__(self, run_id):
        self.run_id = run_id
        self.api = None
        self.run = None
        self.recalculate_predictions = False
        self.refresh_data()

    def refresh_data(self):
        self.api = wandb.Api()
        self.run = self.api.run(f"{PROJECT_ID}/{str(self.run_id)}")

        for prop in self.CACHED_PROPERTIES:
            self.__dict__.pop(prop, None)

        self.get_run_history.cache_clear()

    @lru_cache(maxsize=None)
    def get_run_history(self, samples=10000):
        # Use api to retrieve and plot metric history over training steps
        return self.run.history(samples=samples)

    @cached_property
    def predictions(self) -> pd.DataFrame:
        if self.recalculate_predictions:
            raise wandb.errors.CommError("predictions are disabled")
        artifact = self.api.artifact(
            f"{PROJECT_ID}/run-{self.run_id}-predictions:latest"
        )
        table = artifact.get("predictions")

        return pd.DataFrame(data=table.data, columns=table.columns)

    @cached_property
    def code_vecs(self) -> np.array:
        artifact = self.api.artifact(f"{PROJECT_ID}/run-{self.run_id}-code_vecs:latest")
        table = artifact.get("code_vecs")

        return np.concatenate(table.data)

    @cached_property
    def nl_vecs(self) -> np.array:
        artifact = self.api.artifact(f"{PROJECT_ID}/run-{self.run_id}-nl_vecs:latest")
        table = artifact.get("nl_vecs")

        return np.concatenate(table.data)

    @cached_property
    def scores(self) -> np.array:
        return np.matmul(self.nl_vecs, self.code_vecs.T)

    @cached_property
    def ranks(self) -> np.array:
        try:
            return np.array(self.predictions["pred_idxs"].to_list()) - self.TEST_OFFSET
        except wandb.errors.CommError:
            return np.argsort(self.scores, axis=-1, kind="quicksort", order=None)[
                :, : -self.RANK_TRUNC - 1 : -1
            ]

    @cached_property
    def correct_rank(self) -> np.array:
        return correct_rank(self.ranks)


def get_all_histories(runs_df: pd.DataFrame):
    all_dfs = []
    for index, row in runs_df.iterrows():
        history_df = RunData.run_history(row["run_id"])
        missing_cols = list(set(row.index).difference(history_df.columns))
        # print(row[missing_cols])
        history_df[missing_cols] = row[missing_cols].values
        all_dfs.append(history_df)

    return pd.concat(all_dfs, axis=0, ignore_index=True)


class WandBData:
    SMALL_THRESHOLD = 12800

    def __init__(self):
        self.raw_df = None
        self.df = None

        self.refresh_data()

    def update_raw_df(self):
        self.raw_df = make_wandb_df()

    def preprocess(self):
        df = self.raw_df.dropna(
            axis="rows", subset=["num_synthetic_examples", "test_mrr"]
        ).copy()
        df.replace({None: np.nan, "": np.nan}, inplace=True)

        df["eval_mrr_max"] = df["eval_mrr"].apply(
            lambda x: x.get("max", np.nan) if isinstance(x, dict) else np.nan
        )

        df["created_at"] = pd.to_datetime(df["created_at"])

        # Info columns
        df["has_synthetic_examples"] = df["num_synthetic_examples"] > 0
        df["num_total_examples"] = (
            df["num_synthetic_examples"] + df["num_train_examples"]
        )

        # Homogenise configuration labels
        df["synthetic_dataset"] = df["synthetic_data_file"].str.split("/").str[-1]
        df["synthetic_dataset"] = df["synthetic_dataset"].str.split(".").str[0]
        df.loc[
            df["synthetic_dataset"] == "hardnegative", "synthetic_dataset"
        ] = "hardnegative_synthetic"

        mask_hard_semisynthetic = df["synthetic_dataset"] == "hard_semisynthetic"
        df.loc[
            mask_hard_semisynthetic, "synthetic_dataset"
        ] = "hardnegative_semisynthetic"

        mask_sample_synthetic_subset = df["sample_synthetic_subset"] == True
        df.loc[mask_sample_synthetic_subset, "synthetic_dataset_strategy"] = "sample"
        df.drop(columns=["sample_synthetic_subset"], inplace=True)

        df["overlap"] = (df["train_example_offset"] == df["synthetic_example_offset"]) \
            & ~df["synthetic_dataset_strategy"].isin(("sample", "paired", "paired-negative")) \
            & df["synthetic_dataset"].str.contains("semisynthetic") \
            & (df["num_synthetic_examples"] == df["num_train_examples"]) \
            & (df["synthetic_dataset"] != "hardnegative_semisynthetic")

        # Mask out synthetic dataset related columns if no synthetic examples
        df.loc[
            df["num_synthetic_examples"] == 0,
            ["synthetic_dataset", "synthetic_dataset_strategy"],
        ] = [np.nan, np.nan]

        self.df = df

    def refresh_data(self):
        self.update_raw_df()
        self.preprocess()
        self.filtered_df.cache_clear()

    def filter(self, **kwargs):
        for k, v in kwargs.items():
            self.df = self.df[self.df[k] == v]
        return self

    @lru_cache(maxsize=None)
    def filtered_df(
        self,
        synthetic_datasets=tuple(),
        seeds=(0,),
        small=True,
        strategies=(np.nan,),
        overlap=False,
        new=True,
    ):
        df = self.df
        if small:
            df = df[df["num_train_examples"] <= self.SMALL_THRESHOLD]
        if seeds:
            df = df[df["seed"].isin(seeds)]
        if synthetic_datasets:
            df = df[df["synthetic_dataset"].isin(synthetic_datasets)]
        if strategies:
            df = df[df["synthetic_dataset_strategy"].isin(strategies)]
        if not overlap:
            df = df[~df["overlap"]]
        if new:
            df = df[(df["created_at"] > pd.to_datetime("2023-08-12")) | ~df["has_synthetic_examples"]]

        return df
