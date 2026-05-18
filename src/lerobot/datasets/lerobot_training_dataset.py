#!/usr/bin/env python

"""Cluster-optimized LeRobotTrainingDataset for the marigold_data workload.

Designed for ``multiprocessing_context="forkserver"`` data loaders running on
a shared filesystem with hundreds of workers per node and thousands across the
cluster. Each worker gets a handful of subdatasets via a sharding plan;
``max_active_subdatasets`` is left at the default (``None``), so every dataset
the worker is assigned to lives for the whole worker lifetime.

Five optimisations layered on top of the cacheless ``LeRobotTrainingDataset``:

  1. **Worker-wide shared parquet LRU.** A single module-level
     ``_ParquetTableLRU`` is shared by all ``LeRobotTrainingDataset`` instances
     in a worker process. With forkserver each worker is its own process, so
     "module-level" == "per-worker", and the bound on total parquet RAM is one
     number per worker rather than ``per_instance_size × num_instances``.

  2. **Worker-wide shared video decoder cache** with the same scoping. Bounded
     by default — every distinct mp4 ever touched does NOT stay open forever,
     which was the source of the long-step memory growth in the 8-node job.

  3. **Decoders never evicted on episode transition.** LeRobot 3.0 reuses mp4
     files across multiple episodes per camera; evicting on the per-episode
     boundary just reopens the same NFS handle moments later. Eviction is
     governed by the LRU cap only, not episode hops.

  4. **Fast startup.** ``_check_local_episodes_sufficient`` previously stat'd
     every (episode × camera + episode) path — millions of NFS metadata
     ops cluster-wide. We now (a) deduplicate paths before stat'ing, which
     for a multi-episode-chunk layout drops the work by 2–3 orders of
     magnitude, and (b) honor the ``LEROBOT_SKIP_FILE_CHECK`` env var to skip
     the check entirely. Per-episode bookkeeping at init is built from bulk
     pyarrow column reads instead of one-dict-per-episode iteration.

  5. **Vectorised delta-timestamp reads via ``pc.take``.** Per-``__getitem__``
     CPU win when the policy uses long proprio/action windows.

Public interface matches the cacheless ``LeRobotTrainingDataset`` exactly.

Tunable env vars (all per worker process):
  LEROBOT_PARQUET_CACHE_SIZE          (default 4)    parquet tables held in RAM
  LEROBOT_VIDEO_DECODER_CACHE_SIZE    (default 64)   open video decoders
  LEROBOT_SKIP_FILE_CHECK             (default 0)    skip per-file stat check

LeRobot 3.0 layout invariants used by this design:
  * Multiple episodes can share a parquet file. The parquet LRU is keyed by
    *path*, not episode, so all episodes within a parquet share one entry.
  * Multiple episodes can share an mp4 per camera. Each episode stores
    ``videos/<key>/from_timestamp``; query timestamps are shifted accordingly.
    The decoder cache is keyed by mp4 path, so shared mp4s share a decoder.
"""

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
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
# Cache types
# ---------------------------------------------------------------------------


@dataclass
class _CachedTable:
    """A parquet table loaded into RAM plus an ``abs_index → row_idx`` map.

    The map handles multi-episode parquets (LeRobot 3.0): all rows from all
    episodes in the parquet are indexed in one dict. The 'index' column is
    globally unique across a LeRobot dataset, so the map is unambiguous.
    """

    table: pa.Table
    abs_to_row: dict[int, int]


def _read_and_index(path: Path, columns: list[str]) -> _CachedTable:
    """Read a parquet file fully into RAM and build the abs_idx → row map."""
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
    """Bounded LRU of ``_CachedTable`` objects keyed by parquet path.

    Cache misses read outside the lock so concurrent misses on different
    paths don't serialise.
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


class _PersistentVideoDecoderCache(VideoDecoderCache):
    """VideoDecoderCache that does NOT evict on episode transitions and is
    bounded by an explicit total-decoder LRU cap.

    LeRobot 3.0 shares mp4 files across multiple episodes per camera; closing
    a decoder on every episode hop would re-open the exact same NFS handle a
    few samples later. Eviction is instead driven by the configured cap.
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
        # Intentional no-op: keep handles open across episode transitions.
        return None


# ---------------------------------------------------------------------------
# Module-level (per-worker-process) shared caches
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_PARQUET_CACHE_SIZE = _env_int("LEROBOT_PARQUET_CACHE_SIZE", 4)
_VIDEO_DECODER_CACHE_SIZE = _env_int("LEROBOT_VIDEO_DECODER_CACHE_SIZE", 64)
_SKIP_FILE_CHECK = _env_flag("LEROBOT_SKIP_FILE_CHECK", default=False)


_SHARED_PARQUET_CACHE: Optional[_ParquetTableLRU] = None
_SHARED_VIDEO_DECODER_CACHE: Optional[_PersistentVideoDecoderCache] = None
_SHARED_CACHE_INIT_LOCK = Lock()


def _get_shared_caches() -> tuple[_ParquetTableLRU, _PersistentVideoDecoderCache]:
    """Lazily construct the per-worker singletons used by every dataset in
    this process. With forkserver, every worker is a fresh process, so each
    one gets its own pair sized by the env vars above."""
    global _SHARED_PARQUET_CACHE, _SHARED_VIDEO_DECODER_CACHE
    if _SHARED_PARQUET_CACHE is not None and _SHARED_VIDEO_DECODER_CACHE is not None:
        return _SHARED_PARQUET_CACHE, _SHARED_VIDEO_DECODER_CACHE
    with _SHARED_CACHE_INIT_LOCK:
        if _SHARED_PARQUET_CACHE is None:
            _SHARED_PARQUET_CACHE = _ParquetTableLRU(max_tables=_PARQUET_CACHE_SIZE)
        if _SHARED_VIDEO_DECODER_CACHE is None:
            _SHARED_VIDEO_DECODER_CACHE = _PersistentVideoDecoderCache(
                max_open_decoders=_VIDEO_DECODER_CACHE_SIZE,
            )
    return _SHARED_PARQUET_CACHE, _SHARED_VIDEO_DECODER_CACHE


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


class LeRobotTrainingDataset(torch.utils.data.Dataset):
    """Bounded-RAM, NFS-aware LeRobotTrainingDataset for marigold_data.

    Same constructor, same public attributes/properties, same ``__getitem__``
    output shape as the cacheless ``LeRobotTrainingDataset``.

    The ``parquet_cache_size`` and ``max_open_video_decoders`` kwargs are
    retained for source compatibility but are *not* per-instance anymore;
    the operative bounds are the env vars listed at the top of this module
    (one shared cache per worker process).
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
        parquet_cache_size: int | None = None,
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

        # Per-access scratch.
        self._current_episode_idx: Optional[int] = None
        self._current_episode_cache: Optional[dict] = None

        # Lazy per-episode path resolution state.
        self._episode_resolved_paths: dict[int, Path] = {}
        self._path_resolution_lock = Lock()

        # Per-worker shared caches.
        self._parquet_cache, self._video_decoder_cache = _get_shared_caches()

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

            # Column projection.
            keep = set(self.required_keys) - self._meta_video_feature_keys
            keep |= {"episode_index", "index", "timestamp", "task_index"}
            if self._subtask_names is not None:
                keep.add("subtask_index")
            self._keep_columns = sorted(keep)
            if len(self._keep_columns) == 0:
                raise ValueError("No parquet columns requested")

            # Bulk-load per-episode bookkeeping from the metadata table.
            # Avoids the previous per-episode dict materialisation, which
            # iterated 20+ column lookups per episode (now 1 column lookup
            # converted to numpy in one shot).
            self._load_episode_arrays(meta)

            self._num_frames = self._total_frames

            if not _SKIP_FILE_CHECK and not self._check_local_episodes_sufficient(meta):
                raise FileNotFoundError(
                    f"Local dataset at {self.root} does not contain all required files for episodes."
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

    def _load_episode_arrays(self, meta: LeRobotTrainingDatasetMetadata) -> None:
        """Build the per-episode arrays we need at access time, in one bulk
        pyarrow column read each. Replaces the per-episode ``meta.episodes[ep]``
        dict materialisation that the cacheless variant used in its init loop.
        """
        table = meta._episodes_table  # the in-RAM episode metadata table

        starts = table["dataset_from_index"].to_numpy(zero_copy_only=False)
        ends = table["dataset_to_index"].to_numpy(zero_copy_only=False)
        chunk_indices = table["data/chunk_index"].to_numpy(zero_copy_only=False)
        file_indices = table["data/file_index"].to_numpy(zero_copy_only=False)

        self._episode_starts: list[int] = starts.astype(np.int64).tolist()
        self._episode_ends: list[int] = ends.astype(np.int64).tolist()
        self._episode_dataset_from_index: list[int] = self._episode_starts

        data_path_fmt = meta.data_path
        root = self.root
        # ``format`` is fast enough at this scale; vectorising it is not worth
        # the complexity.
        self._episode_naive_paths: list[Path] = [
            root / data_path_fmt.format(
                chunk_index=int(c),
                file_index=int(f),
            )
            for c, f in zip(chunk_indices, file_indices)
        ]

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
        """Stat-check that every required parquet and video file exists.

        With LeRobot 3.0 multi-episode-per-chunk, most paths are shared across
        episodes. We deduplicate paths from the bulk pyarrow columns and only
        stat each unique path once. With ~10k episodes, ~50 chunks, and 3
        cameras this drops the work from O(40k) to O(200) stat calls per
        dataset.

        Can be disabled entirely via the ``LEROBOT_SKIP_FILE_CHECK`` env var.
        """
        required_video_keys = self._get_decode_video_keys()
        table = meta._episodes_table

        # Unique parquet paths from the bulk-read chunk/file columns.
        data_chunks = table["data/chunk_index"].to_numpy(zero_copy_only=False)
        data_files = table["data/file_index"].to_numpy(zero_copy_only=False)
        unique_parquet_pairs = {(int(c), int(f)) for c, f in zip(data_chunks, data_files)}

        for chunk_idx, file_idx in unique_parquet_pairs:
            parquet_path = self.root / meta.data_path.format(
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            if not parquet_path.exists():
                return False

        # Unique video paths per camera key.
        for vid_key in required_video_keys:
            v_chunks = table[f"videos/{vid_key}/chunk_index"].to_numpy(zero_copy_only=False)
            v_files = table[f"videos/{vid_key}/file_index"].to_numpy(zero_copy_only=False)
            unique_video_pairs = {(int(c), int(f)) for c, f in zip(v_chunks, v_files)}
            for chunk_idx, file_idx in unique_video_pairs:
                video_path = self.root / meta.video_path.format(
                    video_key=vid_key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
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
        column covers ``target_index``. Opens a ``pq.ParquetFile`` briefly to
        read row-group statistics and closes it immediately."""
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

        self._current_episode_idx = episode_idx
        self._current_episode_cache = cache
        return cache

    def __del__(self):
        # Do NOT clear the shared decoder cache here: other instances in this
        # worker rely on it.
        pass

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
