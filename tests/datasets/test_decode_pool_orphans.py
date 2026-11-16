#!/usr/bin/env python

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A decode future never outlives its call, and one cached torchcodec decoder is never used by two
threads at once (concurrent get_frames_at on one VideoDecoder segfaults a DataLoader worker).

CPU only. Run (container, no pytest there): python -m unittest tests.datasets.test_decode_pool_orphans
"""

import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch

import lerobot.datasets.lerobot_training_dataset as ltd
from lerobot.datasets.video_training_utils import (
    VideoDecoderCache,
    decode_video_frame_groups,
    decode_video_frames,
)

VIDEO_ROOT = Path(
    os.environ.get(
        "LEROBOT_TEST_VIDEO_ROOT", "/datasets/MolmoAct2BimanualYam/allenai__18122025-foldclo-11/videos/"
    )
)


class _Tracker:
    """Records the peak number of simultaneous decodes per path and the completion of each."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active: dict[str, int] = {}
        self.peak: dict[str, int] = {}
        self.finished: set[str] = set()

    def enter(self, path):
        with self.lock:
            self.active[path] = self.active.get(path, 0) + 1
            self.peak[path] = max(self.peak.get(path, 0), self.active[path])

    def leave(self, path, done=True):
        with self.lock:
            self.active[path] -= 1
            if done:
                self.finished.add(path)


def _bare_dataset(pool: ThreadPoolExecutor) -> ltd.LeRobotTrainingDataset:
    ds = object.__new__(ltd.LeRobotTrainingDataset)
    ds._get_decode_pool = lambda n: pool
    ds._video_decoder_cache = None
    ds.videos_hw = None
    ds.tolerance_s = 1e-3
    ds.video_backend = "torchcodec"
    return ds


class OrphanDecodeTest(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.addCleanup(self.pool.shutdown, wait=True)
        self.tr = _Tracker()
        self.saved_threads = ltd._VIDEO_DECODE_THREADS
        ltd._VIDEO_DECODE_THREADS = 0
        self.addCleanup(setattr, ltd, "_VIDEO_DECODE_THREADS", self.saved_threads)

    def _fake_decode(self, key):
        """cam_a fails at once, cam_b is slow, cam_c is slow and fails too (later in camera order)."""
        self.tr.enter(key)
        try:
            if key == "cam_a":
                raise IndexError("Invalid frame index a")
            time.sleep(0.4)
            if key == "cam_c":
                raise IndexError("Invalid frame index c")
        finally:
            self.tr.leave(key)
        return key, torch.zeros(1)

    def test_query_videos_waits_for_siblings_before_raising(self):
        ds = _bare_dataset(self.pool)
        ds._decode_single_camera = lambda k, ts, ep: self._fake_decode(k)
        with self.assertRaisesRegex(IndexError, "index a"):  # first error in camera order
            ds._query_videos({"cam_a": [0.0], "cam_b": [0.0], "cam_c": [0.0]}, {})
        # nothing may still be decoding when the error reaches the caller
        self.assertEqual(sum(self.tr.active.values()), 0)
        self.assertEqual(self.tr.finished, {"cam_a", "cam_b", "cam_c"})
        # the next sample's decode of the same path starts only after, so never overlaps
        self._fake_decode("cam_b")
        self.assertEqual(self.tr.peak["cam_b"], 1)

    def test_query_videos_success_unchanged(self):
        ds = _bare_dataset(self.pool)
        ds._decode_single_camera = lambda k, ts, ep: (k, torch.full((1,), float(ord(k[-1]))))
        out = ds._query_videos({"cam_a": [0.0], "cam_b": [0.0]}, {})
        self.assertEqual(list(out), ["cam_a", "cam_b"])
        self.assertEqual(out["cam_b"].item(), float(ord("b")))

    def test_decode_video_groups_waits_for_siblings_before_raising(self):
        ds = _bare_dataset(self.pool)
        ds._get_current_episode_cache = lambda ep: {
            "video_from_timestamps": {"cam_a": 0.0, "cam_b": 0.0},
            "video_paths": {"cam_a": "cam_a", "cam_b": "cam_b"},
        }
        saved = ltd.decode_video_frame_groups
        ltd.decode_video_frame_groups = lambda path, groups, *a, **k: self._fake_decode(path)[1]
        self.addCleanup(setattr, ltd, "decode_video_frame_groups", saved)
        with self.assertRaisesRegex(IndexError, "index a"):
            ds.decode_video_groups(0, {"cam_a": [[0.0]], "cam_b": [[0.0]]})
        self.assertEqual(sum(self.tr.active.values()), 0)
        self.assertEqual(self.tr.finished, {"cam_a", "cam_b"})


class _FakeDecoder:
    """Stands in for a torchcodec VideoDecoder; records overlapping get_frames_at calls."""

    def __init__(self):
        self.metadata = SimpleNamespace(average_fps=10.0)
        self.active = 0
        self.peak = 0
        self.guard = threading.Lock()

    def get_frames_at(self, indices):
        with self.guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self.guard:
            self.active -= 1
        n = len(indices)
        return SimpleNamespace(
            data=torch.zeros(n, 3, 2, 2, dtype=torch.uint8),
            pts_seconds=torch.tensor([i / 10.0 for i in indices]),
        )


class DecoderLockTest(unittest.TestCase):
    def _hammer(self, cache, fn):
        errs = []

        def run():
            try:
                fn()
            except Exception as e:  # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=run) for _ in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errs, [])

    def test_lock_serializes_one_cached_decoder(self):
        for cache in (VideoDecoderCache(), ltd._PersistentVideoDecoderCache(max_open_decoders=4)):
            fake = _FakeDecoder()
            cache._decoders[("v.mp4", None)] = fake
            self._hammer(cache, lambda: decode_video_frames("v.mp4", [0.1, 0.2], 1e-3, "torchcodec", decoder_cache=cache))
            self.assertEqual(fake.peak, 1)
            fake.peak = 0
            self._hammer(
                cache,
                lambda: decode_video_frame_groups("v.mp4", [[0.1], [0.2, 0.3]], 1e-3, "torchcodec", decoder_cache=cache),
            )
            self.assertEqual(fake.peak, 1)

    def test_distinct_decoders_do_not_block_each_other(self):
        cache = VideoDecoderCache()
        a, b = _FakeDecoder(), _FakeDecoder()
        cache._decoders[("a.mp4", None)] = a
        cache._decoders[("b.mp4", None)] = b
        self.assertIsNot(cache.decoder_lock("a.mp4"), cache.decoder_lock("b.mp4"))
        self.assertIs(cache.decoder_lock("a.mp4"), cache.decoder_lock("a.mp4", None))
        self.assertIsNot(cache.decoder_lock("a.mp4"), cache.decoder_lock("a.mp4", (4, 4)))


@unittest.skipUnless((VIDEO_ROOT / "observation.images.left/chunk-000/file-001.mp4").exists(), "no MolmoAct video")
class RealDecodeIdentityTest(unittest.TestCase):
    """Cached (locked), uncached and pooled decodes return the same frames bit for bit, and equal a
    plain torchcodec read of those frames."""

    TS = [90.8 + 0.0333 * i for i in range(6)]

    def _path(self, cam):
        return str(VIDEO_ROOT / f"observation.images.{cam}/chunk-000/file-001.mp4")

    def test_identical_with_and_without_cache_and_pool(self):
        from torchcodec.decoders import VideoDecoder

        for shape in (None, (96, 128)):
            for cam in ("left", "top"):
                p = self._path(cam)
                plain = decode_video_frames(p, self.TS, 3e-2, "torchcodec", shape=shape)
                cache = VideoDecoderCache()
                cached = decode_video_frames(p, self.TS, 3e-2, "torchcodec", shape=shape, decoder_cache=cache)
                with ThreadPoolExecutor(2) as pool:
                    pooled = [
                        pool.submit(decode_video_frames, self._path(c), self.TS, 3e-2, "torchcodec", shape, cache)
                        for c in ("left", "top")
                    ]
                    pooled = [f.result() for f in pooled][("left", "top").index(cam)]
                self.assertEqual(plain.dtype, torch.uint8)
                self.assertTrue(torch.equal(plain, cached))
                self.assertTrue(torch.equal(plain, pooled))
                g, sel, ids = decode_video_frame_groups(
                    p, [self.TS[:3], self.TS[2:]], 3e-2, "torchcodec", shape=shape, decoder_cache=cache
                )
                g0, sel0, ids0 = decode_video_frame_groups(
                    p, [self.TS[:3], self.TS[2:]], 3e-2, "torchcodec", shape=shape
                )
                self.assertTrue(torch.equal(g, g0))
                self.assertEqual((sel, ids), (sel0, ids0))
                self.assertTrue(torch.equal(g[sel[0]], plain[:3]))
                self.assertTrue(torch.equal(g[sel[1]], plain[2:]))
        # plain torchcodec reference (no lerobot code), full resolution
        d = VideoDecoder(self._path("left"), seek_mode="approximate", num_ffmpeg_threads=1)
        ref = d.get_frames_at([round(t * d.metadata.average_fps) for t in self.TS]).data
        self.assertTrue(torch.equal(ref, decode_video_frames(self._path("left"), self.TS, 3e-2, "torchcodec")))

    def test_two_threads_same_cached_decoder_stress(self):
        """race.py 'same' mode through the real entry point: crashed (segfault) without the lock."""
        cache = VideoDecoderCache()
        p = self._path("left")
        fps = 30.0
        base = round(90.8 * 30) + 1045

        def work(off):
            ts = [(base + off + 16 * k) / fps for k in range(128)]
            return decode_video_frames(p, ts, 3e-2, "torchcodec", decoder_cache=cache)

        for _ in range(int(os.environ.get("LEROBOT_TEST_STRESS_ITERS", "20"))):
            with ThreadPoolExecutor(2) as pool:
                futs = [pool.submit(work, 0), pool.submit(work, 4)]
                [f.result() for f in futs]


if __name__ == "__main__":
    unittest.main()
