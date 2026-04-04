#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
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
import importlib
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import av
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature, _FEATURE_TYPES
from torchcodec.decoders import VideoDecoder
from torchcodec.transforms import Resize

logger = logging.getLogger(__name__)

# List of hardware encoders to probe for auto-selection. Availability depends on the platform and FFmpeg build.
# Determines the order of preference for auto-selection when vcodec="auto" is used.
HW_ENCODERS = [
    "h264_videotoolbox",  # macOS
    "hevc_videotoolbox",  # macOS
    "h264_nvenc",  # NVIDIA GPU
    "hevc_nvenc",  # NVIDIA GPU
    "h264_vaapi",  # Linux Intel/AMD
    "h264_qsv",  # Intel Quick Sync
]

VALID_VIDEO_CODECS = {"h264", "hevc", "libsvtav1", "auto"} | set(HW_ENCODERS)


def _get_codec_options(
    vcodec: str,
    g: int | None = 2,
    crf: int | None = 30,
    preset: int | None = None,
) -> dict:
    """Build codec-specific options dict for video encoding."""
    options = {}

    # GOP size (keyframe interval) - supported by VideoToolbox and software encoders
    if g is not None and (vcodec in ("h264_videotoolbox", "hevc_videotoolbox") or vcodec not in HW_ENCODERS):
        options["g"] = str(g)

    # Quality control (codec-specific parameter names)
    if crf is not None:
        if vcodec in ("h264", "hevc", "libsvtav1"):
            options["crf"] = str(crf)
        elif vcodec in ("h264_videotoolbox", "hevc_videotoolbox"):
            quality = max(1, min(100, int(100 - crf * 2)))
            options["q:v"] = str(quality)
        elif vcodec in ("h264_nvenc", "hevc_nvenc"):
            options["rc"] = "constqp"
            options["qp"] = str(crf)
        elif vcodec in ("h264_vaapi",):
            options["qp"] = str(crf)
        elif vcodec in ("h264_qsv",):
            options["global_quality"] = str(crf)

    # Preset (only for libsvtav1)
    if vcodec == "libsvtav1":
        options["preset"] = str(preset) if preset is not None else "12"

    return options


def detect_available_hw_encoders() -> list[str]:
    """Probe PyAV/FFmpeg for available hardware video encoders."""
    available = []
    for codec_name in HW_ENCODERS:
        try:
            av.codec.Codec(codec_name, "w")
            available.append(codec_name)
        except Exception:  # nosec B110
            logger.debug("HW encoder '%s' not available", codec_name)  # nosec B110
    return available


def resolve_vcodec(vcodec: str) -> str:
    """Validate vcodec and resolve 'auto' to best available HW encoder, fallback to libsvtav1."""
    if vcodec not in VALID_VIDEO_CODECS:
        raise ValueError(f"Invalid vcodec '{vcodec}'. Must be one of: {sorted(VALID_VIDEO_CODECS)}")
    if vcodec != "auto":
        logger.info(f"Using video codec: {vcodec}")
        return vcodec
    available = detect_available_hw_encoders()
    for encoder in HW_ENCODERS:
        if encoder in available:
            logger.info(f"Auto-selected video codec: {encoder}")
            return encoder
    logger.info("No hardware encoder available, falling back to software encoder 'libsvtav1'")
    return "libsvtav1"


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logger.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..
        shape (tuple[int, int], optional): Target HxW shape for the decoded frames. Only supported with torchcodec backend.

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s, shape=shape)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logger.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
            f"\nbackend: {backend}"
        )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f"Number of retrieved frames ({len(closest_frames)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )
    return closest_frames


class FrameTimestampError(ValueError):
    """Helper error to indicate the retrieved timestamps exceed the queried ones"""

    pass


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    decoder_cache = None,
    shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Args:
        video_path: Path to the video file.
        timestamps: List of timestamps to extract frames.
        tolerance_s: Allowed deviation in seconds for frame retrieval.
        log_loaded_timestamps: Whether to log loaded timestamps.
        decoder_cache: Optional decoder cache instance. Uses default if None.
        shape: Optional target HxW shape for the decoded frames.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """

    transforms = [Resize(shape)] if shape is not None else None
    decoder = VideoDecoder(str(video_path), seek_mode="approximate", num_ffmpeg_threads=1, transforms=transforms)

    loaded_ts = []
    loaded_frames = []

    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    # convert timestamps to frame indices
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logger.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
        )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    # convert to float32 in [0,1] range
    #closest_frames = (closest_frames / 255.0).type(torch.float32)

    if not len(timestamps) == len(closest_frames):
        raise FrameTimestampError(
            f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
        )
    
    # clostest frames is uin8 in 0-255 range
    return closest_frames


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    if "VideoFrame" not in _FEATURE_TYPES:
        register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.WARNING)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.WARNING)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_video_duration_in_s(video_path: Path | str) -> float:
    """
    Get the duration of a video file in seconds using PyAV.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration of the video in seconds.
    """
    with av.open(str(video_path)) as container:
        # Get the first video stream
        video_stream = container.streams.video[0]
        # Calculate duration: stream.duration * stream.time_base gives duration in seconds
        if video_stream.duration is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            # Fallback to container duration if stream duration is not available
            duration = float(container.duration / av.time_base)
    return duration