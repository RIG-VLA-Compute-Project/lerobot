#!/usr/bin/env python

"""Cluster-optimized LeRobotTrainingDataset for the marigold_data workload.

Designed for shared-filesystem deployments where many workers across many
nodes read from one NFS server. Optimisations are layered onto the cacheless
``LeRobotTrainingDataset`` and target the actual steady-state and startup
bottlenecks observed in 8-node jobs.

Steady-state (per ``__getitem__``):

  1. **mp4 prefetch into the OS page cache on first decoder open** (opt-in via
     ``LEROBOT_PREFETCH_MP4``). LeRobot 3.0 puts many episodes into one mp4,
     so one big sequential NFS read converts the subsequent many-small-seeks-
     from-ffmpeg access pattern into page-cache hits. Disproportionately
     helps multi-node jobs because NFS RPC overhead is what gets multiplied
     by node count.

  2. **Parallel multi-camera video decode** via a small thread pool. Each
     camera's decoder is a separate object on a separate mp4, so concurrent
     decodes are safe; this issues N concurrent NFS reads instead of N
     serial ones. The HF variant has done this for a while; we now do too.

  3. **Bounded video decoder LRU** keyed by mp4 path. Decoders are *not*
     evicted on episode transitions because consecutive episodes typically
     share the same mp4 in LeRobot 3.0. Eviction is governed only by the
     LRU cap.

  4. **Bounded parquet table LRU**. Whole parquet read into RAM on first
     access, all subsequent accesses to any row of that parquet are pure
     RAM lookups. Bound stops memory growth.

  5. **Vectorised delta-timestamp reads via ``pc.take``** so a query of 20
     anchor offsets becomes one columnar arrow op per column, not a Python
     row-by-row loop.

Startup (per ``__init__``):

  6. **Dedup the per-episode file-existence check** before stat'ing. With
     multi-episode-per-chunk layout, ``data/<chunk>/<file>.parquet`` and
     ``videos/<key>/<chunk>/<file>.mp4`` paths are shared across many
     episodes; the previous implementation stat'd them once per episode.
     We now stat each unique path once. The check can also be skipped
     entirely with ``LEROBOT_SKIP_FILE_CHECK``.

  7. **Bulk pyarrow column loads** to build the per-episode arrays in one
     shot instead of materialising one dict per episode.

Caches are module-level so a process has one parquet LRU and one decoder LRU
regardless of how many ``LeRobotTrainingDataset`` instances it holds. Module
globals are scoped to the process; this works correctly under any
multiprocessing context the data loader chooses.

Tunable env vars (all per process):
  LEROBOT_PARQUET_CACHE_SIZE          (default 4)    parquet tables held in RAM
  LEROBOT_VIDEO_DECODER_CACHE_SIZE    (default 64)   open video decoders
  LEROBOT_SKIP_FILE_CHECK             (default 0)    skip per-file stat check
  LEROBOT_PREFETCH_MP4                (default 0)    page-cache prefetch on
                                                     first decoder open
  LEROBOT_VIDEO_DECODE_THREADS        (default = num cameras, cap 8)
                                                     thread pool size for
                                                     parallel multi-camera
                                                     decode; set to 0/1 to
                                                     disable threading.

LeRobot 3.0 layout invariants used by this design:
  * Multiple episodes can share a parquet file. The parquet LRU is keyed by
    *path*, not episode, so all episodes within a parquet share one entry.
  * Multiple episodes can share an mp4 per camera. Each episode stores
    ``videos/<key>/from_timestamp``; query timestamps are shifted
    accordingly. The decoder cache is keyed by mp4 path.
"""

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
# Env-var helpers
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
_PREFETCH_MP4 = _env_flag("LEROBOT_PREFETCH_MP4", default=False)
_VIDEO_DECODE_THREADS = _env_int("LEROBOT_VIDEO_DECODE_THREADS", 0)


# ---------------------------------------------------------------------------
# mp4 page-cache prefetch
# ---------------------------------------------------------------------------


def _prefetch_into_page_cache(path: Path) -> None:
    """Read the file once with sequential-read advice to warm the OS page
    cache. Bytes are not retained in the Python heap; they live only in the
    OS page cache, where they'll be served on subsequent decoder reads
    without an NFS round trip.

    Best-effort: any IO error is swallowed (the worst case is that the
    decoder open later in the call path fails with a clearer error).
    """
    BUF = 4 << 20  # 4 MiB
    try:
        with open(path, "rb") as f:
            fd = f.fileno()
            if hasattr(os, "posix_fadvise"):
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                except OSError:
                    pass
            while f.read(BUF):
                pass
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Cache types
# ---------------------------------------------------------------------------


@dataclass
class _CachedTable:
    """A parquet table loaded into RAM plus an ``abs_index → row_idx`` map.

    Handles multi-episode parquets (LeRobot 3.0): the 'index' column is
    globally unique in a LeRobot dataset, so the map covers every row from
    every episode in this parquet without ambiguity.
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
    """Bounded LRU of ``_CachedTable`` objects keyed by (parquet path,
    projected column set).

    The cache is shared process-wide while reads are projected to each
    dataset's ``_keep_columns``, so the path alone is NOT a sufficient key:
    two datasets over the same file with different column sets would poison
    each other's reads (KeyError on the missing columns). Same-column-set
    instances still share entries.

    Cache misses read outside the lock so concurrent misses on different
    paths don't serialise.
    """

    def __init__(self, max_tables: int) -> None:
        if max_tables < 1:
            raise ValueError("max_tables must be >= 1")
        self._max_tables = max_tables
        self._entries: OrderedDict[tuple, _CachedTable] = OrderedDict()
        self._lock = Lock()

    def get(self, path: Path, columns: list[str]) -> _CachedTable:
        key = (path, tuple(sorted(columns)))
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached

        cached = _read_and_index(path, columns)

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
            self._entries[key] = cached
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_tables:
                self._entries.popitem(last=False)
        return cached

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class _PersistentVideoDecoderCache(VideoDecoderCache):
    """VideoDecoderCache that does not evict on episode transitions and is
    bounded by an explicit total-decoder LRU cap.

    On a miss, optionally prefetches the mp4 into the OS page cache before
    creating the decoder, so that ffmpeg's subsequent seek/read pattern is
    served from RAM rather than NFS.
    """

    def __init__(
        self,
        max_open_decoders: Optional[int] = None,
        prefetch: bool = False,
    ) -> None:
        super().__init__()
        self._max_open_decoders = max_open_decoders
        self._prefetch = prefetch
        self._lru: OrderedDict = OrderedDict()
        self._lru_lock = Lock()

    def get_decoder(self, video_path, shape=None):
        path_str = str(video_path)
        shape_key = tuple(shape) if shape is not None else None
        key = (path_str, shape_key)

        # Fast path: read the decoder reference under self._lock, then
        # RELEASE before touching the LRU. _maybe_evict acquires the two
        # locks in the opposite order (self._lru_lock then self._lock), so
        # nesting them here would deadlock the moment two threads hit
        # get_decoder concurrently — one cache-hit and one cache-miss
        # whose eviction sweep needs self._lock back.
        cached_decoder = None
        with self._lock:
            cached_decoder = self._decoders.get(key)
        if cached_decoder is not None:
            self._touch_lru(key)
            return cached_decoder

        # Cache miss. Prefetch the mp4 into page cache before any NFS read
        # from ffmpeg.
        if self._prefetch:
            _prefetch_into_page_cache(Path(video_path))

        # Delegate decoder creation to the parent class (still locks
        # internally and handles the double-check). The parent will read
        # the (now-cached) mp4 to populate its own state.
        decoder = super().get_decoder(video_path, shape=shape)

        self._touch_lru(key)
        self._maybe_evict()
        return decoder

    def _touch_lru(self, key: tuple) -> None:
        if self._max_open_decoders is None:
            return
        with self._lru_lock:
            if key in self._lru:
                self._lru.move_to_end(key)
            else:
                self._lru[key] = None

    def _maybe_evict(self) -> None:
        if self._max_open_decoders is None:
            return
        with self._lru_lock:
            while len(self._lru) > self._max_open_decoders:
                stale_key, _ = self._lru.popitem(last=False)
                self._evict_one_locked_outside(stale_key)

    def _evict_one_locked_outside(self, key: tuple) -> None:
        # Called while holding self._lru_lock. Decoder dict lock is acquired
        # separately by parent.
        with self._lock:
            decoder = self._decoders.pop(key, None)
        if decoder is not None:
            self._close_decoders([decoder])

    def clear_except_paths(self, video_paths) -> None:
        # Intentional no-op: keep handles open across episode transitions.
        return None


# ---------------------------------------------------------------------------
# Module-level (per-process) caches
# ---------------------------------------------------------------------------


_SHARED_PARQUET_CACHE: Optional[_ParquetTableLRU] = None
_SHARED_VIDEO_DECODER_CACHE: Optional[_PersistentVideoDecoderCache] = None
_SHARED_CACHE_INIT_LOCK = Lock()


def _get_shared_caches() -> tuple[_ParquetTableLRU, _PersistentVideoDecoderCache]:
    """Lazily construct per-process singletons used by every dataset
    instance in this process. Module globals are per-process; this is true
    under any multiprocessing context the data loader chooses."""
    global _SHARED_PARQUET_CACHE, _SHARED_VIDEO_DECODER_CACHE
    if _SHARED_PARQUET_CACHE is not None and _SHARED_VIDEO_DECODER_CACHE is not None:
        return _SHARED_PARQUET_CACHE, _SHARED_VIDEO_DECODER_CACHE
    with _SHARED_CACHE_INIT_LOCK:
        if _SHARED_PARQUET_CACHE is None:
            _SHARED_PARQUET_CACHE = _ParquetTableLRU(max_tables=_PARQUET_CACHE_SIZE)
        if _SHARED_VIDEO_DECODER_CACHE is None:
            _SHARED_VIDEO_DECODER_CACHE = _PersistentVideoDecoderCache(
                max_open_decoders=_VIDEO_DECODER_CACHE_SIZE,
                prefetch=_PREFETCH_MP4,
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
    retained for source compatibility but are not per-instance; the
    operative bounds are the env vars at the top of this module (one shared
    cache per process).
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
        # NOTE: distinguish an explicit empty list (decode no cameras) from
        # None (default: decode all). Collapsing `[]` to None would silently
        # decode every on-disk stream — see `_get_decode_video_keys`.
        self.decode_camera_streams = (
            set(decode_camera_streams) if decode_camera_streams is not None else None
        )
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

        # Lazy per-episode path resolution.
        self._episode_resolved_paths: dict[int, Path] = {}
        self._path_resolution_lock = Lock()

        # Per-process shared caches.
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

            keep = set(self.required_keys) - self._meta_video_feature_keys
            keep |= {"episode_index", "index", "timestamp", "task_index"}
            if self._subtask_names is not None:
                keep.add("subtask_index")
            self._keep_columns = sorted(keep)
            if len(self._keep_columns) == 0:
                raise ValueError("No parquet columns requested")

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

        # Decode thread pool: lazy-init in _query_videos when we know how
        # many cameras to decode and whether threading is enabled.
        self._decode_pool: Optional[ThreadPoolExecutor] = None

    def _open_meta(self) -> LeRobotTrainingDatasetMetadata:
        return LeRobotTrainingDatasetMetadata(
            self.repo_id,
            self.root,
            self.revision,
        )

    def _load_episode_arrays(self, meta: LeRobotTrainingDatasetMetadata) -> None:
        """Build per-episode arrays via bulk pyarrow column reads."""
        table = meta._episodes_table

        starts = table["dataset_from_index"].to_numpy(zero_copy_only=False)
        ends = table["dataset_to_index"].to_numpy(zero_copy_only=False)
        chunk_indices = table["data/chunk_index"].to_numpy(zero_copy_only=False)
        file_indices = table["data/file_index"].to_numpy(zero_copy_only=False)

        self._episode_starts: list[int] = starts.astype(np.int64).tolist()
        self._episode_ends: list[int] = ends.astype(np.int64).tolist()
        self._episode_dataset_from_index: list[int] = self._episode_starts

        data_path_fmt = meta.data_path
        root = self.root
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
        """Stat-check every required parquet and video file exists.

        LeRobot 3.0 multi-episode-per-chunk means most paths repeat across
        episodes. We deduplicate from bulk pyarrow column reads and stat
        each unique path once. Can be disabled with
        ``LEROBOT_SKIP_FILE_CHECK``.
        """
        required_video_keys = self._get_decode_video_keys()
        table = meta._episodes_table

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

    def _get_decode_video_keys(self, camera_streams: "set[str] | list[str] | None" = None) -> list[str]:
        allowed = (
            set(self._video_keys)
            if self.decode_camera_streams is None
            else self.decode_camera_streams
        )
        if camera_streams is not None:
            requested = set(camera_streams)
            unknown = sorted(requested - allowed)
            if unknown:
                raise ValueError(
                    f"Per-call camera_streams not in the constructed decode set: {unknown}. "
                    f"Constructed streams: {sorted(allowed)}."
                )
            allowed = requested
        return [key for key in self._video_keys if key in allowed]

    # ---- Path resolution ----

    def _resolve_episode_path(self, episode_idx: int) -> Path:
        """Return the parquet path for ``episode_idx``, applying the
        ``file_index + 1`` fallback if needed. Cached after first
        resolution. The fallback uses parquet row-group statistics
        (metadata-only NFS read)."""
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
        # Shared caches outlive the instance — do not clear them here.
        pool = getattr(self, "_decode_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False)
            except Exception:
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

    # ---- Delta-timestamp expansion ----

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

    # ---- Video query (parallel multi-camera) ----

    def _decode_single_camera(
        self,
        vid_key: str,
        query_ts: list[float],
        episode_cache: dict,
    ) -> tuple[str, torch.Tensor]:
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
        return vid_key, frames.squeeze(0)

    def _query_videos(
        self,
        query_timestamps: dict[str, list[float]],
        episode_cache: dict,
    ) -> dict[str, torch.Tensor]:
        items = list(query_timestamps.items())

        # Single camera: no thread-pool overhead.
        if len(items) <= 1:
            return {k: v for k, v in (self._decode_single_camera(k, ts, episode_cache) for k, ts in items)}

        # Decide pool size: env var override, else one thread per camera
        # (capped at 8 to bound thread fan-out per worker).
        if _VIDEO_DECODE_THREADS > 0:
            n_threads = min(_VIDEO_DECODE_THREADS, len(items))
        elif _VIDEO_DECODE_THREADS == 0:
            n_threads = min(len(items), 8)
        else:
            n_threads = 1  # negative => disable threading

        if n_threads <= 1:
            return {k: v for k, v in (self._decode_single_camera(k, ts, episode_cache) for k, ts in items)}

        if self._decode_pool is None or getattr(self._decode_pool, "_max_workers", 0) < n_threads:
            if self._decode_pool is not None:
                try:
                    self._decode_pool.shutdown(wait=False)
                except Exception:
                    pass
            self._decode_pool = ThreadPoolExecutor(
                max_workers=n_threads,
                thread_name_prefix="lerobot-decode",
            )

        futures = [
            self._decode_pool.submit(self._decode_single_camera, k, ts, episode_cache)
            for k, ts in items
        ]
        return dict(f.result() for f in futures)

    # ---- Episode → abs_idx lookup ----

    def _episode_idx_from_abs_idx(self, abs_idx: int) -> int:
        ep_idx = bisect_right(self._episode_starts, abs_idx) - 1
        if ep_idx < 0 or abs_idx >= self._episode_ends[ep_idx]:
            raise IndexError(f"Index out of bounds: {abs_idx}")
        return ep_idx

    # ---- Main __getitem__ ----

    def __getitem__(self, idx, camera_streams: "set[str] | list[str] | None" = None) -> dict:
        """``camera_streams``: optional per-call subset of the constructed
        decode streams — only these videos are decoded for this query
        (m3_data per-sample camera selection). Must be a subset of the
        construction-time ``decode_camera_streams``; the plain ``ds[idx]``
        DataLoader path is unaffected."""
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

        decode_video_keys = self._get_decode_video_keys(camera_streams)
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
