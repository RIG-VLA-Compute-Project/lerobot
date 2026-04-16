#!/usr/bin/env python

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.utils

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
)
from lerobot.datasets.video_training_utils import decode_video_frames, get_safe_default_codec
from lerobot.utils.constants import HF_LEROBOT_HOME


class LeRobotTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        decode_camera_streams: list[str] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 3e-4,
        revision: str | None = None,
        video_backend: str | None = None,
        required_keys: set[str] | None = None,
        videos_hw: dict[str, tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.decode_camera_streams = set(decode_camera_streams) if decode_camera_streams else None
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.required_keys = set(required_keys or [])
        self.delta_indices = None
        self.videos_hw = videos_hw

        self._current_episode_table: pa.Table | None = None
        self._keep_columns: list[str] | None = None

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        # Open metadata once, extract only lightweight state, then forget it.
        meta = self._open_meta()

        self._fps = meta.fps
        self._features = meta.features
        self._video_keys = tuple(meta.video_keys)
        self._camera_keys = tuple(meta.camera_keys)
        self._total_frames = meta.total_frames
        self._total_episodes = meta.total_episodes
        self._task_names = tuple(meta.tasks.index.tolist())
        self._subtask_names = None if meta.subtasks is None else tuple(meta.subtasks.index.tolist())
        self._current_episode_idx: int | None = None
        self._current_episode_cache: dict | None = None

        self._validate_decode_camera_streams()
        self._meta_video_feature_keys = self._get_video_feature_keys_from_meta()

        try:
            if not self._check_local_episodes_sufficient(meta):
                raise FileNotFoundError(
                    f"Local dataset at {self.root} does not contain all required files for episodes."
                )

            self._num_frames = self._total_frames

            keep = set(self.required_keys) - self._meta_video_feature_keys
            keep |= {"episode_index", "index", "timestamp", "task_index"}
            if self._subtask_names is not None:
                keep.add("subtask_index")

            self._keep_columns = sorted(keep)
            if len(self._keep_columns) == 0:
                raise ValueError("No parquet columns requested")
        finally:
            del meta

        self._abs_idx_to_episode_idx = {}

        episode_iter = range(self._total_episodes)
        meta = self._open_meta()
        try:
            for ep_idx in episode_iter:
                ep = meta.episodes[ep_idx]
                ep_start = ep["dataset_from_index"].item() if hasattr(ep["dataset_from_index"], "item") else ep["dataset_from_index"]
                ep_end = ep["dataset_to_index"].item() if hasattr(ep["dataset_to_index"], "item") else ep["dataset_to_index"]
                for abs_idx in range(ep_start, ep_end):
                    self._abs_idx_to_episode_idx[abs_idx] = ep_idx
        finally:
            del meta

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _open_meta(self) -> LeRobotDatasetMetadata:
        return LeRobotDatasetMetadata(
            self.repo_id,
            self.root,
            self.revision,
            force_cache_sync=False,
        )

    def _get_current_episode_cache(self, episode_idx: int) -> dict:
        if self._current_episode_idx == episode_idx and self._current_episode_cache is not None:
            return self._current_episode_cache

        meta = self._open_meta()
        try:
            ep = meta.episodes[episode_idx]
            decode_video_keys = self._get_decode_video_keys()

            def scalar(x):
                return x.item() if isinstance(x, torch.Tensor) else x

            chunk_index = scalar(ep["data/chunk_index"])
            file_index = scalar(ep["data/file_index"])
            dataset_from_index = scalar(ep["dataset_from_index"])
            dataset_to_index = scalar(ep["dataset_to_index"])

            parquet_path = self.root / meta.data_path.format(
                chunk_index=chunk_index,
                file_index=file_index,
            )

            cache = {
                "dataset_from_index": dataset_from_index,
                "dataset_to_index": dataset_to_index,
                "episode_length": dataset_to_index - dataset_from_index,
                "parquet_path": parquet_path,
                "episode_index": episode_idx,
                "video_from_timestamps": {
                    vid_key: scalar(ep[f"videos/{vid_key}/from_timestamp"])
                    for vid_key in decode_video_keys
                },
                "video_paths": {
                    vid_key: self.root / meta.get_video_file_path(episode_idx, vid_key)
                    for vid_key in decode_video_keys
                },
            }

            self._current_episode_idx = episode_idx
            self._current_episode_cache = cache
            return cache
        finally:
            del meta

    def _get_validated_parquet_columns(self, parquet_path: Path) -> list[str]:
        parquet_file = pq.ParquetFile(parquet_path)
        available_columns = set(parquet_file.schema_arrow.names)

        missing_columns = [c for c in self._keep_columns if c not in available_columns]
        if missing_columns:
            available_sorted = sorted(available_columns)
            missing_sorted = sorted(missing_columns)
            raise KeyError(
                "Requested parquet columns are missing.\n"
                f"File: {parquet_path}\n"
                f"Missing: {missing_sorted}\n"
                f"Available: {available_sorted}"
            )

        return list(self._keep_columns)
    
    def _read_episode_table(self, episode_cache: dict, columns: list[str]) -> pa.Table:
        table = pq.read_table(
            episode_cache["parquet_path"],
            filters=[
                ("index", ">=", episode_cache["dataset_from_index"]),
                ("index", "<", episode_cache["dataset_to_index"]),
            ],
        )

        available_columns = set(table.column_names)
        missing_columns = [c for c in columns if c not in available_columns]
        if missing_columns:
            raise KeyError(
                "Requested parquet columns are missing after reading filtered episode table.\n"
                f"File: {episode_cache['parquet_path']}\n"
                f"Episode: {episode_cache['episode_index']}\n"
                f"Missing: {sorted(missing_columns)}\n"
                f"Available: {sorted(available_columns)}"
            )

        return table.select(columns)

    def _ensure_current_episode_table(self, episode_cache: dict) -> pa.Table:
        if (
            self._current_episode_table is not None
            and self._current_episode_idx == episode_cache["episode_index"]
        ):
            return self._current_episode_table

        columns = self._get_validated_parquet_columns(episode_cache["parquet_path"])
        table = self._read_episode_table(episode_cache, columns)

        index_values = table["index"].to_pylist()
        expected = set(range(
            episode_cache["dataset_from_index"],
            episode_cache["dataset_to_index"],
        ))
        actual = set(index_values)

        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                f"Unexpected index contents for episode {episode_cache['episode_index']} "
                f"in {episode_cache['parquet_path']}. "
                f"Missing indices: {missing[:10]} "
                f"Extra indices: {extra[:10]}"
            )

        episode_cache["index_to_row"] = {
            abs_idx: row_idx for row_idx, abs_idx in enumerate(index_values)
        }
        
        if table.num_rows != episode_cache["episode_length"]:
            raise RuntimeError(
                f"Unexpected number of rows for episode {episode_cache['episode_index']} "
                f"in {episode_cache['parquet_path']}: "
                f"{table.num_rows} != {episode_cache['episode_length']}"
            )

        self._current_episode_table = table
        return table

    def _check_local_episodes_sufficient(
        self,
        meta: LeRobotDatasetMetadata,
    ) -> bool:
        requested_episodes = (
            set(range(meta.total_episodes))
        )

        required_video_keys = self._get_decode_video_keys()

        for ep_idx in requested_episodes:
            ep = meta.episodes[ep_idx]

            def scalar(x):
                return x.item() if isinstance(x, torch.Tensor) else x

            parquet_path = self.root / meta.data_path.format(
                chunk_index=scalar(ep["data/chunk_index"]),
                file_index=scalar(ep["data/file_index"]),
            )
            if not parquet_path.exists():
                return False

            for vid_key in required_video_keys:
                video_path = self.root / meta.get_video_file_path(ep_idx, vid_key)
                if not video_path.exists():
                    return False

        return True

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_episodes(self) -> int:
        return self._total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self._features

    @property
    def video_keys(self) -> tuple[str, ...]:
        return self._video_keys

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return self._camera_keys

    def get_episode_info(self, episode_idx: int):
        meta = self._open_meta()
        ep = meta.episodes[episode_idx]
        return ep

    def get_episode_len(self, episode_idx: int) -> int:
        ep = self.get_episode_info(episode_idx)
        return ep["length"]

    def _validate_decode_camera_streams(self) -> None:
        if self.decode_camera_streams is None:
            return

        unknown_streams = sorted(self.decode_camera_streams - set(self.video_keys))
        if unknown_streams:
            raise ValueError(
                f"Unknown decode_camera_streams: {unknown_streams}. "
                f"Available video streams: {self.video_keys}."
            )
        
    def _get_video_feature_keys_from_meta(self) -> set[str]:
        video_keys = set()

        for feature_name, feature_def in self.features.items():
            dtype = feature_def.get("dtype") if isinstance(feature_def, dict) else getattr(feature_def, "dtype", None)
            if dtype == "video":
                video_keys.add(feature_name)

        return video_keys

    def _get_decode_video_keys(self) -> list[str]:
        if self.decode_camera_streams is None:
            return list(self.video_keys)
        return [key for key in self.video_keys if key in self.decode_camera_streams]
    
    def _episode_local_index(self, abs_idx: int, episode_cache: dict) -> int:
        try:
            return episode_cache["index_to_row"][abs_idx]
        except KeyError as e:
            raise KeyError(
                f"Absolute index {abs_idx} not found in episode {episode_cache['episode_index']} "
                f"for parquet file {episode_cache['parquet_path']}"
            ) from e

    def _arrow_scalar_to_python(self, value):
        if hasattr(value, "as_py"):
            return value.as_py()
        return value

    def _table_row_to_item(self, table: pa.Table, row_idx: int) -> dict:
        item = {}
        for key in table.column_names:
            value = table[key][row_idx]
            value = self._arrow_scalar_to_python(value)

            if isinstance(value, list):
                item[key] = torch.tensor(value)
            elif isinstance(value, bool):
                item[key] = torch.tensor(value)
            elif isinstance(value, int):
                item[key] = torch.tensor(value)
            elif isinstance(value, float):
                item[key] = torch.tensor(value)
            else:
                item[key] = value

        return item

    def _get_query_indices(
        self,
        abs_idx: int,
        episode_cache: dict,
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep_start = episode_cache["dataset_from_index"]
        ep_end = episode_cache["dataset_to_index"]

        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        episode_table: pa.Table,
        episode_cache: dict,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        ts_col = episode_table["timestamp"]

        for key in self.video_keys:
            if query_indices is not None and key in query_indices:
                local_indices = [
                    self._episode_local_index(idx, episode_cache)
                    for idx in query_indices[key]
                ]
                timestamps = pc.take(ts_col, pa.array(local_indices))
                query_timestamps[key] = [x.as_py() for x in timestamps]
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_episode_table(
        self,
        episode_table: pa.Table,
        episode_cache: dict,
        query_indices: dict[str, list[int]],
    ) -> dict:
        result = {}

        for key, q_idx in query_indices.items():
            if key in self.video_keys:
                continue

            local_indices = [
                self._episode_local_index(idx, episode_cache)
                for idx in q_idx
            ]
            taken = pc.take(episode_table[key], pa.array(local_indices))
            values = [self._arrow_scalar_to_python(x) for x in taken]

            if len(values) > 0 and isinstance(values[0], list):
                result[key] = torch.tensor(values)
            else:
                result[key] = torch.tensor(values)

        return result

    def _query_videos(
        self,
        query_timestamps: dict[str, list[float]],
        episode_cache: dict,
    ) -> dict[str, torch.Tensor]:
        item = {}

        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = episode_cache["video_from_timestamps"][vid_key]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = episode_cache["video_paths"][vid_key]
            video_hw = self.videos_hw.get(vid_key, None) if self.videos_hw is not None else None
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self.tolerance_s,
                self.video_backend,
                shape=video_hw,
            )
            item[vid_key] = frames.squeeze(0)

        return item

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        abs_idx = idx

        ep_idx = self._abs_idx_to_episode_idx.get(abs_idx)
        if ep_idx is None:
            raise IndexError(f"Index out of bounds: {idx}")

        episode_cache = self._get_current_episode_cache(ep_idx)
        episode_table = self._ensure_current_episode_table(episode_cache)

        local_idx = self._episode_local_index(abs_idx, episode_cache)
        item = self._table_row_to_item(episode_table, local_idx)

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, episode_cache)
            query_result = self._query_episode_table(episode_table, episode_cache, query_indices)
            item = {**item, **padding, **query_result}

        decode_video_keys = self._get_decode_video_keys()
        if len(decode_video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(
                episode_table, episode_cache, current_ts, query_indices
            )
            query_timestamps = {
                key: value for key, value in query_timestamps.items()
                if key in decode_video_keys
            }

            item = {
                key: value
                for key, value in item.items()
                if not key.endswith("_is_pad")
                or key[:-len("_is_pad")] not in self.video_keys
                or key[:-len("_is_pad")] in decode_video_keys
            }

            video_frames = self._query_videos(query_timestamps, episode_cache)
            item = {**video_frames, **item}

        task_idx = item["task_index"].item()
        item["task"] = self._task_names[task_idx]

        if "subtask_index" in item and self._subtask_names is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._subtask_names[subtask_idx]

        return item