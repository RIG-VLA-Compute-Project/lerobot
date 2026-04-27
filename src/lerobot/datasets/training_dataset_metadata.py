#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

import packaging.version
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
from lerobot.datasets.utils import (
    DEFAULT_SUBTASKS_PATH,
    DEFAULT_TASKS_PATH,
    EPISODES_DIR,
    INFO_PATH,
    check_version_compatibility,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


class _EpisodeRows:
    """
    Lightweight row-access wrapper over a pyarrow Table.

    Supports:
      - len(meta.episodes)
      - meta.episodes[ep_idx] -> dict[str, python_scalar_or_list]
      - meta.episodes[column_name] -> list[python_scalar_or_list]
      - meta.episodes.to_pandas()
    """

    def __init__(self, table: pa.Table):
        self._table = table
        self.column_names = tuple(table.column_names)
        self._column_cache: dict[str, list] = {}
        self._row_cache: dict[int, dict] = {}
        self._pandas_cache: pd.DataFrame | None = None

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, idx: int | str | slice) -> dict | list:
        if isinstance(idx, str):
            if idx not in self.column_names:
                raise KeyError(idx)
            if idx not in self._column_cache:
                self._column_cache[idx] = self._table[idx].to_pylist()
            return self._column_cache[idx]

        if isinstance(idx, slice):
            return [self[row_idx] for row_idx in range(*idx.indices(self._table.num_rows))]

        if not isinstance(idx, int):
            raise TypeError(f"Episode index must be int, str, or slice, got {type(idx)}")

        if idx < 0:
            idx += self._table.num_rows

        if idx < 0 or idx >= self._table.num_rows:
            raise IndexError(f"Episode index {idx} out of range")

        if idx in self._row_cache:
            return self._row_cache[idx]

        row = {}
        for key in self.column_names:
            value = self._table[key][idx]
            row[key] = value.as_py() if hasattr(value, "as_py") else value
        self._row_cache[idx] = row
        return row

    def __iter__(self):
        for idx in range(self._table.num_rows):
            yield self[idx]

    def to_pandas(self) -> pd.DataFrame:
        if self._pandas_cache is None:
            self._pandas_cache = self._table.to_pandas()
        return self._pandas_cache


class LeRobotTrainingDatasetMetadata:
    """
    Cacheless metadata loader for training.
    """

    EXCLUDED_EPISODE_COLUMN_PREFIXES = ("stats/",)

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.info = self._load_info()
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)

        self.tasks = self._load_tasks()
        self.subtasks = self._load_subtasks()
        self._episodes_table = self._load_episodes_table()
        self.episodes = _EpisodeRows(self._episodes_table)

    def _load_info(self) -> dict:
        fpath = self.root / INFO_PATH
        with open(fpath) as f:
            info = json.load(f)

        for ft in info["features"].values():
            if "shape" in ft:
                ft["shape"] = tuple(ft["shape"])

        return info

    def _load_tasks(self) -> pd.DataFrame:
        path = self.root / DEFAULT_TASKS_PATH
        tasks = pd.read_parquet(path)
        tasks.index.name = "task"
        return tasks

    def _load_subtasks(self) -> pd.DataFrame | None:
        path = self.root / DEFAULT_SUBTASKS_PATH
        if not path.exists():
            return None
        subtasks = pd.read_parquet(path)
        if subtasks.index.name is None and "subtask" in subtasks.columns:
            subtasks = subtasks.set_index("subtask")
        return subtasks

    def _episode_paths(self) -> list[Path]:
        paths = sorted((self.root / EPISODES_DIR).glob("*/*.parquet"))
        if len(paths) == 0:
            raise FileNotFoundError(
                f"No episode metadata parquet files found under {self.root / EPISODES_DIR}"
            )
        return paths

    def _load_episodes_table(self) -> pa.Table:
        paths = self._episode_paths()
        columns = [
            name
            for name in pq.read_schema(paths[0]).names
            if not name.startswith(self.EXCLUDED_EPISODE_COLUMN_PREFIXES)
        ]
        return pq.read_table([str(path) for path in paths], columns=columns)

    @property
    def _version(self) -> packaging.version.Version:
        return packaging.version.parse(self.info["codebase_version"])

    @property
    def data_path(self) -> str:
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        return self.info["video_path"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    def get_data_file_path(self, ep_index: int) -> Path:
        ep = self.episodes[ep_index]
        return Path(
            self.data_path.format(
                chunk_index=ep["data/chunk_index"],
                file_index=ep["data/file_index"],
            )
        )

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep = self.episodes[ep_index]
        return Path(
            self.video_path.format(
                video_key=vid_key,
                chunk_index=ep[f"videos/{vid_key}/chunk_index"],
                file_index=ep[f"videos/{vid_key}/file_index"],
            )
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"repo_id={self.repo_id!r}, "
            f"root={str(self.root)!r}, "
            f"total_episodes={self.total_episodes}, "
            f"total_frames={self.total_frames})"
        )
