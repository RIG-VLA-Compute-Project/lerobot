#!/usr/bin/env python

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from bisect import bisect_right
import torch
import torch.utils

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
from lerobot.datasets.training_dataset_metadata import LeRobotTrainingDatasetMetadata
from lerobot.datasets.feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
)
from lerobot.datasets.video_training_utils import (
    VideoDecoderCache,
    decode_video_frames,
    get_safe_default_codec,
)
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

        self._current_parquet_path: Path | None = None
        self._current_parquet_file: pq.ParquetFile | None = None
        self._current_row_group_locator: list[tuple[int, int, int]] | None = None
        self._parquet_index_bounds_cache: dict[Path, tuple[int, int]] = {}
        self._current_rows_cache_key: tuple[Path, tuple[int, ...]] | None = None
        self._current_rows_table: pa.Table | None = None
        self._current_rows_index_to_row: dict[int, int] | None = None
        self._keep_columns: list[str] | None = None
        self._video_decoder_cache = VideoDecoderCache()

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self._current_episode_idx: int | None = None
        self._current_episode_cache: dict | None = None
        self._episode_starts: list[int] = []
        self._episode_ends: list[int] = []

        self.meta = self._open_meta()
        meta = self.meta
        try:
            self._fps = meta.fps
            self._features = meta.features
            self._video_keys = tuple(meta.video_keys)
            self._camera_keys = tuple(meta.camera_keys)
            self._total_frames = meta.total_frames
            self._total_episodes = meta.total_episodes
            self._task_names = tuple(meta.tasks.index.tolist())
            self._subtask_names = None if meta.subtasks is None else tuple(meta.subtasks.index.tolist())

            self._validate_decode_camera_streams()
            self._meta_video_feature_keys = self._get_video_feature_keys_from_meta()

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

            def scalar(x):
                return x.item() if hasattr(x, "item") else x

            for ep_idx in range(self._total_episodes):
                ep = meta.episodes[ep_idx]
                self._episode_starts.append(scalar(ep["dataset_from_index"]))
                self._episode_ends.append(scalar(ep["dataset_to_index"]))
        finally:
            del meta

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _open_meta(self) -> LeRobotTrainingDatasetMetadata:
        return LeRobotTrainingDatasetMetadata(
            self.repo_id,
            self.root,
            self.revision,
        )

    def _get_current_episode_cache(self, episode_idx: int) -> dict:
        if self._current_episode_idx == episode_idx and self._current_episode_cache is not None:
            return self._current_episode_cache

        meta = self.meta
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

        # TODO - this seems to be necessary for AgiBotWorldBeta specifically.
        # Maybe this is an issue for any dataset which at some point used the official lerobot conversion code?
        # In any case we should harden this to wrap around the chunk index too
        if not self._parquet_contains_index(parquet_path, dataset_from_index):
            next_parquet_path = self.root / meta.data_path.format(
                chunk_index=chunk_index,
                file_index=file_index + 1,
            )

            if self._parquet_contains_index(next_parquet_path, dataset_from_index):
                parquet_path = next_parquet_path
            else:
                raise RuntimeError(
                    f"Could not locate dataset_from_index={dataset_from_index} "
                    f"for episode {episode_idx} in metadata parquet file {parquet_path} "
                    f"or fallback parquet file {next_parquet_path}"
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
                vid_key: self.root / meta.video_path.format(
                    video_key=vid_key,
                    chunk_index=scalar(ep[f"videos/{vid_key}/chunk_index"]),
                    file_index=scalar(ep[f"videos/{vid_key}/file_index"]),
                )
                for vid_key in decode_video_keys
            },
        }

        if self._current_episode_cache is not None:
            self._video_decoder_cache.clear_except_paths(cache["video_paths"].values())

        if self._current_parquet_path is not None and self._current_parquet_path != parquet_path:
            self._clear_current_parquet_state()

        self._current_episode_idx = episode_idx
        self._current_episode_cache = cache
        return cache

    def _clear_current_rows_cache(self) -> None:
        self._current_rows_cache_key = None
        self._current_rows_table = None
        self._current_rows_index_to_row = None

    def _clear_current_parquet_state(self) -> None:
        parquet_file = self._current_parquet_file
        if parquet_file is not None:
            close = getattr(parquet_file, "close", None)
            if close is not None:
                close()

        self._current_parquet_path = None
        self._current_parquet_file = None
        self._current_row_group_locator = None
        self._clear_current_rows_cache()

    def _clear_video_decoder_cache(self) -> None:
        decoder_cache = getattr(self, "_video_decoder_cache", None)
        if decoder_cache is not None:
            decoder_cache.clear()

    def __del__(self):
        self._clear_video_decoder_cache()
        self._clear_current_parquet_state()

    def _ensure_current_parquet_locator(
        self,
        episode_cache: dict,
    ) -> tuple[pq.ParquetFile, list[tuple[int, int, int]]]:
        parquet_path = episode_cache["parquet_path"]

        if (
            self._current_parquet_path == parquet_path
            and self._current_parquet_file is not None
            and self._current_row_group_locator is not None
        ):
            return self._current_parquet_file, self._current_row_group_locator

        self._clear_current_parquet_state()
        parquet_file = pq.ParquetFile(parquet_path)

        available_columns = set(parquet_file.schema_arrow.names)
        missing_columns = [c for c in self._keep_columns if c not in available_columns]
        if missing_columns:
            raise KeyError(
                "Requested parquet columns are missing.\n"
                f"File: {parquet_path}\n"
                f"Missing: {sorted(missing_columns)}\n"
                f"Available: {sorted(available_columns)}"
            )

        try:
            index_col_idx = parquet_file.schema_arrow.names.index("index")
        except ValueError as e:
            raise KeyError(f"'index' column not found in parquet file: {parquet_path}") from e

        locator = []
        for rg_idx in range(parquet_file.metadata.num_row_groups):
            col_meta = parquet_file.metadata.row_group(rg_idx).column(index_col_idx)
            stats = col_meta.statistics
            if stats is None or stats.min is None or stats.max is None:
                raise RuntimeError(
                    f"Parquet row-group statistics for 'index' are missing in {parquet_path}. "
                    "Path B relies on min/max row-group stats for fast sparse reads."
                )
            locator.append((int(stats.min), int(stats.max), rg_idx))

        self._current_parquet_path = parquet_path
        self._current_parquet_file = parquet_file
        self._current_row_group_locator = locator
        return parquet_file, locator

    def _read_rows_for_abs_indices(
        self,
        episode_cache: dict,
        abs_indices: list[int],
    ) -> tuple[pa.Table, dict[int, int]]:
        parquet_file, locator = self._ensure_current_parquet_locator(episode_cache)

        target_indices = sorted(set(int(i) for i in abs_indices))
        if len(target_indices) == 0:
            raise ValueError("No indices requested")

        needed_row_groups = [
            rg_idx
            for rg_min, rg_max, rg_idx in locator
            if any(rg_min <= target_idx <= rg_max for target_idx in target_indices)
        ]
        if len(needed_row_groups) == 0:
            raise RuntimeError(
                f"No row groups found for requested indices {target_indices[:10]} "
                f"in {episode_cache['parquet_path']}"
            )

        rows_cache_key = (episode_cache["parquet_path"], tuple(needed_row_groups))
        if (
            self._current_rows_cache_key == rows_cache_key
            and self._current_rows_table is not None
            and self._current_rows_index_to_row is not None
        ):
            table = self._current_rows_table
            row_by_index = self._current_rows_index_to_row
        else:
            table = parquet_file.read_row_groups(
                needed_row_groups,
                columns=self._keep_columns,
            )

            index_values = table["index"].to_pylist()
            row_by_index = {abs_idx: row_idx for row_idx, abs_idx in enumerate(index_values)}

            self._current_rows_cache_key = rows_cache_key
            self._current_rows_table = table
            self._current_rows_index_to_row = row_by_index

        missing = [abs_idx for abs_idx in target_indices if abs_idx not in row_by_index]
        if missing:
            raise RuntimeError(
                f"Requested indices not found after sparse parquet read for "
                f"{episode_cache['parquet_path']}: {missing[:10]}"
            )

        return table, row_by_index
    
    def _parquet_contains_index(self, parquet_path: Path, target_index: int) -> bool:
        if not parquet_path.exists():
            return False

        bounds = self._parquet_index_bounds(parquet_path)
        if bounds is None:
            return False

        min_index, max_index = bounds
        return min_index <= target_index <= max_index

    def _parquet_index_bounds(self, parquet_path: Path) -> tuple[int, int] | None:
        cached = self._parquet_index_bounds_cache.get(parquet_path)
        if cached is not None:
            return cached

        parquet_file = pq.ParquetFile(parquet_path)
        try:
            try:
                index_col_idx = parquet_file.schema_arrow.names.index("index")
            except ValueError as e:
                raise KeyError(f"'index' column not found in parquet file: {parquet_path}") from e

            mins = []
            maxes = []
            for rg_idx in range(parquet_file.metadata.num_row_groups):
                col_meta = parquet_file.metadata.row_group(rg_idx).column(index_col_idx)
                stats = col_meta.statistics
                if stats is None or stats.min is None or stats.max is None:
                    raise RuntimeError(
                        f"Parquet row-group statistics for 'index' are missing in {parquet_path}. "
                        "Training dataset uses min/max row-group stats for sparse reads."
                    )
                mins.append(int(stats.min))
                maxes.append(int(stats.max))
        finally:
            close = getattr(parquet_file, "close", None)
            if close is not None:
                close()

        if len(mins) == 0:
            return None

        bounds = (min(mins), max(maxes))
        self._parquet_index_bounds_cache[parquet_path] = bounds
        return bounds

    def _check_local_episodes_sufficient(
        self,
        meta: LeRobotTrainingDatasetMetadata,
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
                video_path = self.root / meta.video_path.format(
                    video_key=vid_key,
                    chunk_index=scalar(ep[f"videos/{vid_key}/chunk_index"]),
                    file_index=scalar(ep[f"videos/{vid_key}/file_index"]),
                )
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
        return self.meta.episodes[episode_idx]

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
        rows_table: pa.Table,
        row_by_index: dict[int, int],
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        ts_col = rows_table["timestamp"]

        for key in self.video_keys:
            if query_indices is not None and key in query_indices:
                row_ids = [row_by_index[idx] for idx in query_indices[key]]
                timestamps = pc.take(ts_col, pa.array(row_ids))
                query_timestamps[key] = [x.as_py() for x in timestamps]
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_rows_table(
        self,
        rows_table: pa.Table,
        row_by_index: dict[int, int],
        query_indices: dict[str, list[int]],
    ) -> dict:
        result = {}

        for key, q_idx in query_indices.items():
            if key in self.video_keys:
                continue

            row_ids = [row_by_index[idx] for idx in q_idx]
            taken = pc.take(rows_table[key], pa.array(row_ids))
            values = [self._arrow_scalar_to_python(x) for x in taken]
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
                decoder_cache=self._video_decoder_cache,
            )
            item[vid_key] = frames.squeeze(0)

        return item

    def _episode_idx_from_abs_idx(self, abs_idx: int) -> int:
        ep_idx = bisect_right(self._episode_starts, abs_idx) - 1
        if ep_idx < 0 or abs_idx >= self._episode_ends[ep_idx]:
            raise IndexError(f"Index out of bounds: {abs_idx}")
        return ep_idx

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        abs_idx = idx

        ep_idx = self._episode_idx_from_abs_idx(abs_idx)

        episode_cache = self._get_current_episode_cache(ep_idx)

        query_indices = None
        padding = {}
        needed_indices = [abs_idx]

        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, episode_cache)
            for q_idx in query_indices.values():
                needed_indices.extend(q_idx)

        rows_table, row_by_index = self._read_rows_for_abs_indices(
            episode_cache,
            needed_indices,
        )

        item = self._table_row_to_item(rows_table, row_by_index[abs_idx])

        if query_indices is not None:
            query_result = self._query_rows_table(rows_table, row_by_index, query_indices)
            item = {**item, **padding, **query_result}

        decode_video_keys = self._get_decode_video_keys()
        if len(decode_video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(
                rows_table,
                row_by_index,
                current_ts,
                query_indices,
            )
            query_timestamps = {
                key: value
                for key, value in query_timestamps.items()
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
