#!/usr/bin/env python

"""Cluster-optimized LeRobotTrainingDataset.

Drop-in replacement for the cacheless ``LeRobotTrainingDataset`` for the
marigold_data workload on a shared filesystem.

What marigold's ``InfiniteDataReader`` actually does (per worker, with
``multiprocessing_context="forkserver"``):

  * Each worker is assigned a handful of subdatasets via the sharding plan.
  * Each subdataset gets a ``SubdatasetState`` → ``Sampler`` →
    ``LeRobotTrainingDataset``, all eagerly constructed once per worker.
  * After that, the worker runs forever:
      pick one of its datasets (weighted random) →
      pull the next sample from that dataset →
      occasionally cross an episode boundary in that dataset.
  * ``max_active_subdatasets`` defaults to ``None`` and is typically left off,
    so no unload/reload of samplers happens.

What that means for the NFS picture:

  * Each ``LeRobotTrainingDataset`` lives for the worker's lifetime.
  * The per-instance state we set up at init survives all the way through.
  * The only NFS-touching work we control during training is what
    ``__getitem__`` does.

Three optimisations targeted at exactly this pattern:

  1. **Bounded per-instance parquet LRU.** First access to a given data
     parquet file reads the whole file into RAM (with column projection) and
     stores it. Subsequent accesses to any row of that parquet are pure RAM
     lookups. Cache size is bounded — RAM per worker scales with
     ``parquet_cache_size`` × largest-parquet, not with total dataset size.
     Inside one episode the hit rate is 100%; at episode boundaries it is
     also ~100% in LeRobot 3.0 because consecutive episodes share parquets
     within a chunk.

  2. **Persistent video decoders.** Decoders are NOT evicted on episode
     transitions. With LeRobot 3.0's shared mp4 layout (multiple episodes
     per mp4 per camera), closing the decoder when an episode ends would
     reopen the same NFS handle one batch later. An optional LRU cap on
     total open decoders is supported for FD-budget-constrained setups.

  3. **Vectorised delta-timestamp reads.** Each ``__getitem__`` typically
     pulls ~20 rows for delta-timestamp expansion (proprio + action windows).
     These now go through one ``pc.take`` per column instead of a Python
     row-by-row loop.

Public interface matches the cacheless ``LeRobotTrainingDataset``: same
constructor, same properties, same ``__getitem__`` output. Two new
optional kwargs (``parquet_cache_size`` and ``max_open_video_decoders``)
have defaults that preserve existing call sites.

LeRobot 3.0 layout assumptions baked into the design:
  * Multiple episodes can share a parquet file. The parquet LRU is keyed
    by *path*, not by episode, so all episodes in the same parquet share
    one cache entry. ``_CachedTable.abs_to_row`` maps the global ``index``
    column for the entire parquet, not just one episode.
  * Multiple episodes can share an mp4 file per camera. Each episode stores
    ``videos/<key>/from_timestamp`` indicating where it begins in the
    shared mp4. The decoder cache is keyed by mp4 path, so shared mp4s
    share one decoder; query timestamps are shifted by ``from_timestamp``.
"""

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

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


# ---------------------------------------------------------------------------
# Per-instance parquet LRU cache
# ---------------------------------------------------------------------------


@dataclass
class _CachedTable:
    """A parquet table loaded into RAM plus an ``abs_index → row_idx`` map.

    The map is built once when the table is loaded by scanning the 'index'
    column. It handles multi-episode parquets (LeRobot 3.0): all rows from
    all episodes in the parquet are indexed in a single dict. The 'index'
    column is globally unique in a LeRobot dataset, so the map is
    unambiguous even when the parquet contains many episodes.
    """

    table: pa.Table
    abs_to_row: dict[int, int]


def _read_and_index(path: Path, columns: list[str]) -> _CachedTable:
    """Read a parquet file fully into RAM and build the abs_idx → row map.

    On a pyarrow read failure, attempts a schema read to produce a clearer
    error message about missing columns (one extra NFS read on failure
    only).
    """
    try:
        table = pq.read_table(path, columns=columns)
    except (KeyError, pa.ArrowInvalid):
        schema = pq.read_schema(path)
        available = set(schema.names)
        missing = [c for c in columns if c not in available]
        if missing:
            raise KeyError(
                "Requested parquet columns are missing.\n"
                f"File: {path}\n"
                f"Missing: {sorted(missing)}\n"
                f"Available: {sorted(available)}"
            ) from None
        raise
    abs_to_row = {abs_idx: row_idx for row_idx, abs_idx in enumerate(table["index"].to_pylist())}
    return _CachedTable(table=table, abs_to_row=abs_to_row)


class _ParquetTableLRU:
    """Bounded LRU of ``_CachedTable`` objects keyed by path.

    One per ``LeRobotTrainingDataset`` instance. Cache misses read outside
    the lock so concurrent misses on different paths don't serialise.
    """

    def __init__(self, max_tables: int) -> None:
        if max_tables < 1:
            raise ValueError("max_tables must be >= 1")
        self._max_tables = max_tables
        self._entries: OrderedDict[Path, _CachedTable] = OrderedDict()
        self._lock = Lock()

    def get(self, path: Path, columns: list[str]) -> _CachedTable:
        with self._lock:
            cached = self._entries.get(path)
            if cached is not None:
                self._entries.move_to_end(path)
                return cached

        # Cache miss: read outside the lock so other paths can be served in
        # parallel.
        cached = _read_and_index(path, columns)

        with self._lock:
            existing = self._entries.get(path)
            if existing is not None:
                self._entries.move_to_end(path)
                return existing
            self._entries[path] = cached
            self._entries.move_to_end(path)
            while len(self._entries) > self._max_tables:
                self._entries.popitem(last=False)
        return cached

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Video decoder cache: persistent across episode transitions
# ---------------------------------------------------------------------------


class _PersistentVideoDecoderCache(VideoDecoderCache):
    """VideoDecoderCache that does NOT evict decoders on episode transitions.

    LeRobot 3.0 shares mp4 files across multiple episodes per camera. Closing
    the decoder on every episode hop would re-open the exact same NFS handle
    a few samples later. Instead we keep handles open and optionally cap the
    total number via LRU.
    """

    def __init__(self, max_open_decoders: Optional[int] = None) -> None:
        super().__init__()
        self._max_open_decoders = max_open_decoders
        self._lru: OrderedDict = OrderedDict()
        self._lru_lock = Lock()

    def get_decoder(self, video_path, shape=None):
        decoder = super().get_decoder(video_path, shape=shape)
        if self._max_open_decoders is not None:
            key = (str(video_path), tuple(shape) if shape is not None else None)
            with self._lru_lock:
                if key in self._lru:
                    self._lru.move_to_end(key)
                else:
                    self._lru[key] = None
                while len(self._lru) > self._max_open_decoders:
                    stale_key, _ = self._lru.popitem(last=False)
                    self._evict_one(stale_key)
        return decoder

    def _evict_one(self, key: tuple[str, Optional[tuple[int, int]]]) -> None:
        with self._lock:
            decoder = self._decoders.pop(key, None)
        if decoder is not None:
            self._close_decoders([decoder])

    def clear_except_paths(self, video_paths) -> None:
        # Intentional no-op: keep handles open across episode transitions so
        # shared mp4s in the LeRobot 3.0 layout are not repeatedly re-opened.
        return None


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_DEFAULT_PARQUET_CACHE_SIZE = _env_int("LEROBOT_PARQUET_CACHE_SIZE", 4)


class LeRobotTrainingDataset(torch.utils.data.Dataset):
    """Bounded-RAM, NFS-aware LeRobotTrainingDataset for marigold_data.

    Same constructor, same public attributes/properties, same ``__getitem__``
    output shape as the cacheless ``LeRobotTrainingDataset``.

    Optional extra kwargs (backward-compatible defaults):
      parquet_cache_size: max number of fully-loaded parquet tables held in
        RAM per dataset instance. Defaults to the
        ``LEROBOT_PARQUET_CACHE_SIZE`` env var, falling back to 4. For
        marigold's pattern (one active episode at a time, episodes share
        parquets via LeRobot 3.0 chunking), values of 2–4 hit ~100% within
        an episode and across most episode transitions.
      max_open_video_decoders: optional LRU cap on open decoders. Defaults
        to None (no cap — keep decoders open for instance lifetime).
    """

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
        parquet_cache_size: int = _DEFAULT_PARQUET_CACHE_SIZE,
        max_open_video_decoders: int | None = None,
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

        # Per-access scratch state.
        self._current_episode_idx: Optional[int] = None
        self._current_episode_cache: Optional[dict] = None

        # Episode bookkeeping (derived from metadata at init).
        self._episode_starts: list[int] = []
        self._episode_ends: list[int] = []
        self._episode_naive_paths: list[Path] = []
        self._episode_dataset_from_index: list[int] = []

        # Lazy path resolution state (per-episode, populated on first access).
        self._episode_resolved_paths: dict[int, Path] = {}
        self._path_resolution_lock = Lock()

        # Per-instance lazy caches.
        self._parquet_cache = _ParquetTableLRU(max_tables=parquet_cache_size)
        self._video_decoder_cache = _PersistentVideoDecoderCache(
            max_open_decoders=max_open_video_decoders,
        )

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

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
            self._subtask_names = (
                None if meta.subtasks is None else tuple(meta.subtasks.index.tolist())
            )

            self._validate_decode_camera_streams()
            self._meta_video_feature_keys = self._get_video_feature_keys_from_meta()

            if not self._check_local_episodes_sufficient(meta):
                raise FileNotFoundError(
                    f"Local dataset at {self.root} does not contain all required files for episodes."
                )

            self._num_frames = self._total_frames

            # Column projection: only load columns we actually need from data
            # parquet files. Excludes video features (those come from mp4s).
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
                ep_start = scalar(ep["dataset_from_index"])
                ep_end = scalar(ep["dataset_to_index"])
                self._episode_starts.append(ep_start)
                self._episode_ends.append(ep_end)
                self._episode_dataset_from_index.append(ep_start)

                chunk_index = scalar(ep["data/chunk_index"])
                file_index = scalar(ep["data/file_index"])
                self._episode_naive_paths.append(
                    self.root / meta.data_path.format(
                        chunk_index=chunk_index,
                        file_index=file_index,
                    )
                )
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

    # ---- Public properties (interface parity with cacheless) ----

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

    def __len__(self) -> int:
        return self.num_frames

    def get_episode_info(self, episode_idx: int):
        return self.meta.episodes[episode_idx]

    def get_episode_len(self, episode_idx: int) -> int:
        ep = self.get_episode_info(episode_idx)
        return ep["length"]

    # ---- Validation ----

    def _validate_decode_camera_streams(self) -> None:
        if self.decode_camera_streams is None:
            return
        unknown_streams = sorted(self.decode_camera_streams - set(self._video_keys))
        if unknown_streams:
            raise ValueError(
                f"Unknown decode_camera_streams: {unknown_streams}. "
                f"Available video streams: {self._video_keys}."
            )

    def _get_video_feature_keys_from_meta(self) -> set[str]:
        video_keys = set()
        for feature_name, feature_def in self._features.items():
            dtype = (
                feature_def.get("dtype")
                if isinstance(feature_def, dict)
                else getattr(feature_def, "dtype", None)
            )
            if dtype == "video":
                video_keys.add(feature_name)
        return video_keys

    def _check_local_episodes_sufficient(
        self,
        meta: LeRobotTrainingDatasetMetadata,
    ) -> bool:
        requested_episodes = set(range(meta.total_episodes))
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

    def _get_decode_video_keys(self) -> list[str]:
        if self.decode_camera_streams is None:
            return list(self._video_keys)
        return [key for key in self._video_keys if key in self.decode_camera_streams]

    # ---- Path resolution (lazy, cached per episode) ----

    def _resolve_episode_path(self, episode_idx: int) -> Path:
        """Return the parquet path for ``episode_idx``, applying the
        ``file_index + 1`` fallback if needed.

        Cached after first resolution. The fallback check uses parquet
        row-group statistics (metadata-only NFS read).
        """
        resolved = self._episode_resolved_paths.get(episode_idx)
        if resolved is not None:
            return resolved

        with self._path_resolution_lock:
            resolved = self._episode_resolved_paths.get(episode_idx)
            if resolved is not None:
                return resolved

            naive = self._episode_naive_paths[episode_idx]
            target_index = self._episode_dataset_from_index[episode_idx]

            if self._candidate_table_contains_index(naive, target_index):
                self._episode_resolved_paths[episode_idx] = naive
                return naive

            meta = self.meta
            ep = meta.episodes[episode_idx]

            def scalar(x):
                return x.item() if isinstance(x, torch.Tensor) else x

            fallback = self.root / meta.data_path.format(
                chunk_index=scalar(ep["data/chunk_index"]),
                file_index=scalar(ep["data/file_index"]) + 1,
            )
            if self._candidate_table_contains_index(fallback, target_index):
                self._episode_resolved_paths[episode_idx] = fallback
                return fallback

            raise RuntimeError(
                f"Could not locate dataset_from_index={target_index} "
                f"for episode {episode_idx} in {naive} or fallback {fallback}"
            )

    @staticmethod
    def _candidate_table_contains_index(path: Path, target_index: int) -> bool:
        """Metadata-only check whether ``path`` is a parquet whose 'index'
        column covers ``target_index``. Opens a ``pq.ParquetFile`` briefly
        to read row-group statistics and closes it immediately."""
        if not path.exists():
            return False

        parquet_file = pq.ParquetFile(path)
        try:
            try:
                index_col_idx = parquet_file.schema_arrow.names.index("index")
            except ValueError:
                return False
            for rg_idx in range(parquet_file.metadata.num_row_groups):
                stats = parquet_file.metadata.row_group(rg_idx).column(index_col_idx).statistics
                if stats is None or stats.min is None or stats.max is None:
                    return False
                if int(stats.min) <= target_index <= int(stats.max):
                    return True
            return False
        finally:
            close = getattr(parquet_file, "close", None)
            if close is not None:
                close()

    # ---- Per-episode cache (video paths + from_timestamps) ----

    def _get_current_episode_cache(self, episode_idx: int) -> dict:
        if self._current_episode_idx == episode_idx and self._current_episode_cache is not None:
            return self._current_episode_cache

        meta = self.meta
        ep = meta.episodes[episode_idx]
        decode_video_keys = self._get_decode_video_keys()

        def scalar(x):
            return x.item() if isinstance(x, torch.Tensor) else x

        dataset_from_index = scalar(ep["dataset_from_index"])
        dataset_to_index = scalar(ep["dataset_to_index"])

        cache = {
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_to_index,
            "episode_length": dataset_to_index - dataset_from_index,
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

        # NOTE: deliberately no decoder cache eviction here. With shared mp4s
        # in LeRobot 3.0, consecutive episodes often map to the same mp4 path
        # — evicting on episode change would just re-open the same NFS file.

        self._current_episode_idx = episode_idx
        self._current_episode_cache = cache
        return cache

    def _clear_video_decoder_cache(self) -> None:
        decoder_cache = getattr(self, "_video_decoder_cache", None)
        if decoder_cache is not None:
            decoder_cache.clear()

    def __del__(self):
        # Release video decoder file handles. Parquet cache is pure RAM,
        # nothing to close.
        self._clear_video_decoder_cache()

    # ---- Row materialisation ----

    @staticmethod
    def _arrow_scalar_to_python(value):
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

    # ---- Delta-timestamp expansion (batched via pc.take) ----

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
        cached: _CachedTable,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        """Read 'timestamp' values for video query indices from the cached
        table in one batched ``pc.take`` per video key.

        All query indices for a __getitem__ are within the current episode,
        which is contained in a single parquet file (the ``cached`` table).
        """
        query_timestamps: dict[str, list[float]] = {}
        ts_col = cached.table["timestamp"]

        for key in self._video_keys:
            if query_indices is not None and key in query_indices:
                rows = pa.array([cached.abs_to_row[q] for q in query_indices[key]])
                query_timestamps[key] = pc.take(ts_col, rows).to_pylist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_rows(
        self,
        cached: _CachedTable,
        query_indices: dict[str, list[int]],
    ) -> dict:
        """Batched delta-timestamp value reads for non-video keys.

        All queries hit the single cached parquet table for the current
        episode. Each column is fetched in one ``pc.take`` call.
        """
        result: dict = {}

        for key, q_idx_list in query_indices.items():
            if key in self._video_keys:
                continue

            rows = pa.array([cached.abs_to_row[q] for q in q_idx_list])
            taken = pc.take(cached.table[key], rows).to_pylist()
            result[key] = torch.tensor(taken)

        return result

    def _query_videos(
        self,
        query_timestamps: dict[str, list[float]],
        episode_cache: dict,
    ) -> dict[str, torch.Tensor]:
        item: dict = {}
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

    # ---- Episode → abs_idx lookup ----

    def _episode_idx_from_abs_idx(self, abs_idx: int) -> int:
        ep_idx = bisect_right(self._episode_starts, abs_idx) - 1
        if ep_idx < 0 or abs_idx >= self._episode_ends[ep_idx]:
            raise IndexError(f"Index out of bounds: {abs_idx}")
        return ep_idx

    # ---- Main __getitem__ ----

    def __getitem__(self, idx) -> dict:
        abs_idx = idx

        ep_idx = self._episode_idx_from_abs_idx(abs_idx)
        episode_cache = self._get_current_episode_cache(ep_idx)

        path = self._resolve_episode_path(ep_idx)
        cached = self._parquet_cache.get(path, self._keep_columns)

        row_idx = cached.abs_to_row[abs_idx]
        item = self._table_row_to_item(cached.table, row_idx)

        query_indices = None
        padding: dict = {}
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, episode_cache)
            query_result = self._query_rows(cached, query_indices)
            item = {**item, **padding, **query_result}

        decode_video_keys = self._get_decode_video_keys()
        if len(decode_video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(cached, current_ts, query_indices)
            query_timestamps = {
                key: value
                for key, value in query_timestamps.items()
                if key in decode_video_keys
            }

            item = {
                key: value
                for key, value in item.items()
                if not key.endswith("_is_pad")
                or key[:-len("_is_pad")] not in self._video_keys
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
