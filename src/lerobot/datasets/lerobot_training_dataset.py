#!/usr/bin/env python

from collections.abc import Callable
from pathlib import Path

import datasets
import torch
import torch.utils

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from lerobot.datasets.io_utils import hf_transform_to_torch, load_nested_dataset
from lerobot.datasets.video_training_utils import decode_video_frames, get_safe_default_codec
from lerobot.utils.constants import HF_LEROBOT_HOME


class LeRobotTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
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
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.required_keys = set(required_keys or [])
        self.delta_indices = None
        self._absolute_to_relative_idx = None
        self.videos_hw = videos_hw

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

        self._validate_decode_camera_streams()

        temp_hf_dataset = self._open_hf_dataset()
        try:
            if not self._check_local_episodes_sufficient(temp_hf_dataset, meta):
                raise FileNotFoundError(
                    f"Local dataset at {self.root} does not contain all required files for episodes={self.episodes}"
                )

            self._num_frames = len(temp_hf_dataset) if self.episodes is not None else self._total_frames

            if self.episodes is not None:
                self._absolute_to_relative_idx = {
                    abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
                    for rel_idx, abs_idx in enumerate(temp_hf_dataset["index"])
                }
        finally:
            del temp_hf_dataset
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

    def _open_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)

        keep = set(self.required_keys)
        keep |= {"episode_index", "index", "timestamp", "task_index"}
        if "subtask_index" in hf_dataset.column_names:
            keep.add("subtask_index")

        keep = [c for c in hf_dataset.column_names if c in keep]
        hf_dataset = hf_dataset.select_columns(keep)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_local_episodes_sufficient(
        self,
        hf_dataset: datasets.Dataset,
        meta: LeRobotDatasetMetadata,
    ) -> bool:
        if hf_dataset is None or len(hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in hf_dataset.unique("episode_index")
        }

        requested_episodes = (
            set(range(meta.total_episodes))
            if self.episodes is None
            else set(self.episodes)
        )

        if not requested_episodes.issubset(available_episodes):
            return False

        required_video_keys = self._get_decode_video_keys()
        if len(required_video_keys) > 0:
            for ep_idx in requested_episodes:
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

    def _get_decode_video_keys(self) -> list[str]:
        if self.decode_camera_streams is None:
            return list(self.video_keys)
        return [key for key in self.video_keys if key in self.decode_camera_streams]

    def _get_query_indices(
        self,
        abs_idx: int,
        episode_info,
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep_start = episode_info["dataset_from_index"]
        ep_end = episode_info["dataset_to_index"]

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
        hf_dataset: datasets.Dataset,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(
        self,
        hf_dataset: datasets.Dataset,
        query_indices: dict[str, list[int]],
    ) -> dict:
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.video_keys:
                continue

            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )

            try:
                result[key] = torch.stack(hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(hf_dataset[relative_indices][key])

        return result

    def _query_videos(
        self,
        meta: LeRobotDatasetMetadata,
        query_timestamps: dict[str, list[float]],
        ep_idx: int,
        episode_info,
    ) -> dict[str, torch.Tensor]:
        item = {}

        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = episode_info[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / meta.get_video_file_path(ep_idx, vid_key)
            video_hw = self.videos_hw.get(vid_key, None) if self.videos_hw is not None else None
            frames = decode_video_frames(video_path, shifted_query_ts, self.tolerance_s, self.video_backend, shape=video_hw)
            item[vid_key] = frames.squeeze(0)

        return item

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        meta = self._open_meta()
        hf_dataset = self._open_hf_dataset()

        try:
            item = hf_dataset[idx]
            ep_idx = item["episode_index"].item()
            abs_idx = item["index"].item()
            episode_info = meta.episodes[ep_idx]

            query_indices = None
            if self.delta_indices is not None:
                query_indices, padding = self._get_query_indices(abs_idx, episode_info)
                query_result = self._query_hf_dataset(hf_dataset, query_indices)
                item = {**item, **padding}
                for key, val in query_result.items():
                    item[key] = val

            decode_video_keys = self._get_decode_video_keys()
            if len(decode_video_keys) > 0:
                current_ts = item["timestamp"].item()
                query_timestamps = self._get_query_timestamps(hf_dataset, current_ts, query_indices)
                query_timestamps = {key: value for key, value in query_timestamps.items() if key in decode_video_keys}

                item = {
                    key: value
                    for key, value in item.items()
                    if not key.endswith("_is_pad")
                    or key[:-len("_is_pad")] not in self.video_keys
                    or key[:-len("_is_pad")] in decode_video_keys
                }

                video_frames = self._query_videos(meta, query_timestamps, ep_idx, episode_info)
                item = {**video_frames, **item}

            task_idx = item["task_index"].item()
            item["task"] = self._task_names[task_idx]

            if "subtask_index" in item and self._subtask_names is not None:
                subtask_idx = item["subtask_index"].item()
                item["subtask"] = self._subtask_names[subtask_idx]

            return item
        finally:
            del hf_dataset
            del meta