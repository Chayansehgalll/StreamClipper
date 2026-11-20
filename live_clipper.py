"""Authorized live-stream clipper: chat spike -> rolling buffer -> vertical clip.

Quick start:
    1. Install FFmpeg and Streamlink, and ensure both are on PATH.
    2. Install Python dependencies: uv sync
    3. Set GROQ_API_KEY in the project .env file.
    4. Run: uv run --env-file .env python live_clipper.py
    5. Open the dashboard and enter the stream URL to start clipping.

Only use this with streams you own or are authorized to capture and republish.
This tool does not bypass DRM, access controls, or platform restrictions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from review_server import ClipReviewServer


# Configuration
STREAM_URL = ""  # Optional URL prefilled in the dashboard.
CHAT_WINDOW_SECONDS = 5
CHAT_BASELINE_SECONDS = 300
CHAT_SPIKE_MULTIPLIER = 1.5
CHAT_SPIKE_MIN_MESSAGES = 2
CLIP_DURATION = 46
CLIP_PRE_ROLL_SECONDS = 10
COOLDOWN_PERIOD = 10
WHISPER_MODEL_SIZE = "small"  # "small" or "base"
TRANSCRIPTION_TIMEOUT_SECONDS = 15
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SEGMENT_SECONDS = 1
MAX_CONCURRENT_CLIPS = 2
MAX_CLIPS_PER_RUN = 5
VOICE_ONLY_AUDIO = False
VOICE_STEM_MODEL = "UVR_MDXNET_KARA_2.onnx"
GAMEPLAY_CROP_FOCUS_X = 0.5
GAMEPLAY_CROP_RATIO = 0.75
STREAMER_PANEL_HEIGHT = 540
FACECAM_AUTO_DETECT = True
FACECAM_CROP_X = 0.70
FACECAM_CROP_Y = 0.68
FACECAM_CROP_WIDTH = 0.30
FACECAM_CROP_HEIGHT = 0.32
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("STREAMCLIPPER_DATA_DIR", str(ROOT_DIR))).resolve()
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = DATA_DIR / "output"
YOUTUBE_CLIENT_SECRETS_FILE = ROOT_DIR / "youtube_client_secret.json"
YOUTUBE_TOKEN_FILE = (
    Path(os.getenv("LOCALAPPDATA", str(Path.home())))
    / "live-stream-clipper"
    / "youtube_token.json"
)
SEGMENT_DIR = TEMP_DIR / "segments"
SEGMENT_PATTERN = "segment_*.ts"
_DEFERRED_CLEANUP_PATHS: set[Path] = set()
FACE_MODEL_PATH = (
    Path(os.getenv("LOCALAPPDATA", str(ROOT_DIR)))
    / "live-stream-clipper"
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)
FACE_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "f12e12798e8314f7c074a6656816c048dcc95b7a/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
FACE_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
_FACE_MODEL_LOCK = threading.Lock()
VOICE_MODEL_DIR = (
    Path(os.getenv("LOCALAPPDATA", str(ROOT_DIR)))
    / "live-stream-clipper"
    / "audio-separator-models"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
LOGGER = logging.getLogger("live-clipper")


class ConfigurationError(RuntimeError):
    """Raised when an input or required local dependency is unavailable."""


@dataclass(frozen=True)
class ClipSettings:
    stream_url: str = STREAM_URL
    chat_spike_multiplier: float = CHAT_SPIKE_MULTIPLIER
    clip_duration: int = CLIP_DURATION
    clip_pre_roll_seconds: int = CLIP_PRE_ROLL_SECONDS
    max_clips: int = MAX_CLIPS_PER_RUN
    min_chat_messages: int = CHAT_SPIKE_MIN_MESSAGES
    cooldown_seconds: int = COOLDOWN_PERIOD

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ClipSettings:
        stream_url = payload.get("stream_url")
        if not isinstance(stream_url, str) or not stream_url.strip():
            raise ConfigurationError("Enter a Twitch, YouTube, or Kick stream URL.")
        stream_url = stream_url.strip()
        if len(stream_url) > 2048:
            raise ConfigurationError("Stream URL must be at most 2048 characters.")
        detect_platform(stream_url)

        multiplier_value = payload.get("chat_spike_multiplier")
        if isinstance(multiplier_value, bool):
            raise ConfigurationError("Chat spike multiplier must be a number.")
        try:
            multiplier = float(multiplier_value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("Chat spike multiplier must be a number.") from exc
        if not math.isfinite(multiplier) or not 0.1 <= multiplier <= 20:
            raise ConfigurationError("Chat spike multiplier must be between 0.1 and 20.")

        settings = cls(
            stream_url=stream_url,
            chat_spike_multiplier=multiplier,
            clip_duration=_int_setting(payload, "clip_duration", 5, 180),
            clip_pre_roll_seconds=_int_setting(payload, "clip_pre_roll_seconds", 0, 179),
            max_clips=_int_setting(payload, "max_clips", 1, 100),
            min_chat_messages=_int_setting(payload, "min_chat_messages", 1, 100),
            cooldown_seconds=_int_setting(payload, "cooldown_seconds", 0, 3600),
        )
        if settings.clip_pre_roll_seconds >= settings.clip_duration:
            raise ConfigurationError("Pre-roll must be shorter than the total clip duration.")
        return settings


def _int_setting(payload: dict[str, Any], key: str, minimum: int, maximum: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{key.replace('_', ' ').capitalize()} must be a whole number.")
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{key.replace('_', ' ').capitalize()} must be between {minimum} and {maximum}."
        )
    return value


def detect_platform(url: str) -> str:
    """Return the platform name for supported public channel/live URLs."""
    url = url.strip()
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigurationError("Enter a valid http(s) live-stream URL.")

    host = parsed.hostname.lower().removeprefix("www.")
    if host == "twitch.tv" or host.endswith(".twitch.tv"):
        return "twitch"
    if host in {"youtube.com", "m.youtube.com", "youtu.be"} or host.endswith(".youtube.com"):
        return "youtube"
    if host == "kick.com" or host.endswith(".kick.com"):
        return "kick"
    raise ConfigurationError("Supported URLs are Twitch, YouTube, and Kick.")


def youtube_video_id(url: str) -> str | None:
    """Extract a video ID from common YouTube watch and live-video URLs."""
    if "://" not in url:
        url = f"https://{url.strip()}"
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0] or None

    video_id = parse_qs(parsed.query).get("v")
    if video_id and video_id[0]:
        return video_id[0]

    parts = [part for part in parsed.path.split("/") if part]
    for route in ("live", "embed", "shorts"):
        if route in parts:
            index = parts.index(route)
            return parts[index + 1] if index + 1 < len(parts) else None
    return None


def streamer_name(url: str) -> str:
    """Get a readable channel identifier for prompts and output metadata."""
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname.lower().endswith("youtu.be"):
        return parsed.path.strip("/").split("/")[0] or "streamer"
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.hostname and "youtube.com" in parsed.hostname.lower():
        if "@" in (parts[0] if parts else ""):
            return parts[0].lstrip("@")
        if "channel" in parts:
            index = parts.index("channel")
            return parts[index + 1] if index + 1 < len(parts) else "streamer"
        video_id = youtube_video_id(url)
        if video_id:
            return video_id
    return parts[0].lstrip("@") if parts else "streamer"


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "clip"


def ensure_ffmpeg_on_path() -> bool:
    """Find WinGet's standard FFmpeg install location when it is not on PATH."""
    if shutil.which("ffmpeg"):
        return True

    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        return False

    packages_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    if not packages_dir.is_dir():
        return False

    for package_dir in packages_dir.glob("Gyan.FFmpeg*"):
        for executable in package_dir.glob("**/bin/ffmpeg.exe"):
            if executable.is_file():
                os.environ["PATH"] = (
                    str(executable.parent) + os.pathsep + os.environ.get("PATH", "")
                )
                LOGGER.info("Found FFmpeg in the WinGet package directory.")
                return True
    return False


def message_timestamp(message: dict[str, Any]) -> float:
    """Normalize ChatDownloader timestamps to Unix seconds."""
    value = message.get("timestamp")
    if value is None:
        value = message.get("timestamp_in_seconds")
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        return time.time()
    # Chat sources may report seconds, milliseconds, or microseconds.
    if stamp > 1e14:
        stamp /= 1_000_000
    elif stamp > 1e11:
        stamp /= 1_000
    return stamp


def is_chat_spike(
    message_times: deque[float],
    now: float,
    monitoring_started_at: float,
    spike_multiplier: float | None = None,
    minimum_messages: int | None = None,
) -> tuple[bool, int, int, float]:
    if spike_multiplier is None:
        spike_multiplier = CHAT_SPIKE_MULTIPLIER
    if minimum_messages is None:
        minimum_messages = CHAT_SPIKE_MIN_MESSAGES
    recent_cutoff = now - CHAT_WINDOW_SECONDS
    baseline_start = max(monitoring_started_at, now - CHAT_BASELINE_SECONDS)
    baseline_duration = max(0.0, recent_cutoff - baseline_start)
    recent_count = sum(timestamp >= recent_cutoff for timestamp in message_times)
    baseline_count = sum(
        baseline_start <= timestamp < recent_cutoff
        for timestamp in message_times
    )
    baseline_expected = (
        baseline_count * CHAT_WINDOW_SECONDS / baseline_duration
        if baseline_duration > 0
        else 0
    )
    threshold = max(
        minimum_messages,
        math.ceil(spike_multiplier * baseline_expected),
    )
    return recent_count >= threshold, recent_count, threshold, baseline_expected


class ChatReader:
    """Adapt ChatDownloader's synchronous iterator to an asyncio consumer."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=500)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._chat: Any = None
        self._chat_lock = threading.Lock()

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._read_chat,
            name="chat-downloader",
            daemon=True,
        )
        self.thread.start()

    def _read_chat(self) -> None:
        try:
            if detect_platform(self.url) == "youtube":
                video_id = youtube_video_id(self.url)
                if video_id:
                    self._read_youtube_chat(video_id)
                else:
                    self._read_chat_downloader()
            else:
                self._read_chat_downloader()
        except Exception as exc:
            self.events.put(("error", exc))
        finally:
            self.events.put(("end", None))

    def _read_chat_downloader(self) -> None:
        from chat_downloader import ChatDownloader

        chat = ChatDownloader().get_chat(self.url)
        with self._chat_lock:
            self._chat = chat
        for message in chat:
            if self.stop_event.is_set():
                break
            self.events.put(("message", message))

    def _read_youtube_chat(self, video_id: str) -> None:
        import pytchat

        LOGGER.info("Using pytchat for YouTube live chat.")
        # This reader runs in a worker thread; pytchat's default installs a
        # SIGINT handler, which is only allowed from Python's main thread.
        try:
            chat = pytchat.create(video_id=video_id, interruptable=False)
        except pytchat.InvalidVideoIdException as exc:
            if self.stop_event.is_set():
                return
            detail = getattr(exc, "doc", str(exc))
            LOGGER.warning(
                "Pytchat could not resolve YouTube video %s (%s); trying the "
                "ChatDownloader fallback.",
                video_id,
                detail,
            )
            self._read_chat_downloader()
            return
        with self._chat_lock:
            self._chat = chat
        while chat.is_alive() and not self.stop_event.is_set():
            received_message = False
            for item in chat.get().sync_items():
                if self.stop_event.is_set():
                    break
                self.events.put(
                    (
                        "message",
                        {
                            "timestamp": item.timestamp,
                            "message": item.message,
                        },
                    )
                )
                received_message = True
            if not received_message:
                self.stop_event.wait(0.2)

    def stop(self) -> None:
        self.stop_event.set()
        with self._chat_lock:
            chat = self._chat
        close = getattr(chat, "close", None)
        if not callable(close):
            close = getattr(chat, "terminate", None)
        if callable(close):
            close()

    async def next_event(self) -> tuple[str, Any]:
        return await asyncio.to_thread(self.events.get)


def _cleanup_temporary_files(paths: tuple[Path, ...], *, deferred: bool = False) -> None:
    for path in paths:
        if deferred:
            _DEFERRED_CLEANUP_PATHS.discard(path)
        elif path in _DEFERRED_CLEANUP_PATHS:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("Could not remove temporary file %s.", path)


def _finish_cancelled_worker(
    task: asyncio.Task[Any],
    cleanup_paths: tuple[Path, ...],
) -> None:
    try:
        error = task.exception()
    except asyncio.CancelledError:
        error = None
    if error is not None:
        LOGGER.error("Clip worker failed after stop was requested: %s", error)
    _cleanup_temporary_files(cleanup_paths, deferred=True)


async def _run_in_thread_cancellable(
    function: Callable[..., Any],
    *args: Any,
    cleanup_after_cancel: tuple[Path, ...],
) -> Any:
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        _DEFERRED_CLEANUP_PATHS.update(cleanup_after_cancel)
        worker.add_done_callback(
            lambda completed: _finish_cancelled_worker(completed, cleanup_after_cancel)
        )
        raise


async def _terminate_process(
    process: asyncio.subprocess.Process,
    timeout: float = 1.0,
) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


async def _communicate_process(
    process: asyncio.subprocess.Process,
) -> tuple[bytes, bytes]:
    try:
        return await process.communicate()
    except asyncio.CancelledError:
        await _terminate_process(process)
        raise


class RollingStreamCapture:
    """Capture Streamlink output into a bounded rolling set of MPEG-TS segments."""

    def __init__(self, url: str, clip_duration: int = CLIP_DURATION) -> None:
        self.url = url
        self.clip_duration = clip_duration
        self.stop_event = asyncio.Event()
        self._segment_lock = asyncio.Lock()
        self._stderr_tasks: list[asyncio.Task[None]] = []
        self._buffer_started_at: float | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def _next_segment_number(self) -> int:
        numbers = []
        for path in SEGMENT_DIR.glob(SEGMENT_PATTERN):
            match = re.search(r"segment_(\d+)\.ts$", path.name)
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers, default=-1) + 1

    async def _drain_stderr(
        self,
        stream: asyncio.StreamReader | None,
        label: str,
    ) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode(errors="replace").strip()
            if text:
                LOGGER.debug("%s: %s", label, text)

    async def _pump(
        self,
        source: asyncio.StreamReader,
        target: asyncio.StreamWriter,
        source_process: asyncio.subprocess.Process,
    ) -> None:
        try:
            while True:
                data = await source.read(64 * 1024)
                if not data:
                    break
                target.write(data)
                await target.drain()
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.warning("FFmpeg input closed while Streamlink was still writing.")
            await self._terminate(source_process)
        finally:
            target.close()
            try:
                await target.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    async def _terminate(self, process: asyncio.subprocess.Process | None) -> None:
        if process is not None:
            await _terminate_process(process)

    async def run(self) -> None:
        SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
        backoff = 2
        while not self.stop_event.is_set():
            streamlink: asyncio.subprocess.Process | None = None
            ffmpeg: asyncio.subprocess.Process | None = None
            prune_task: asyncio.Task[None] | None = None
            self._stderr_tasks = []
            try:
                self._buffer_started_at = time.time()
                start_number = self._next_segment_number()
                streamlink = await asyncio.create_subprocess_exec(
                    "streamlink",
                    "--stdout",
                    self.url,
                    "best",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                ffmpeg = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-i",
                    "pipe:0",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-c",
                    "copy",
                    "-f",
                    "segment",
                    "-segment_format",
                    "mpegts",
                    "-segment_time",
                    str(SEGMENT_SECONDS),
                    "-reset_timestamps",
                    "1",
                    "-segment_start_number",
                    str(start_number),
                    str(SEGMENT_DIR / "segment_%09d.ts"),
                    stdin=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                if streamlink.stdout is None or ffmpeg.stdin is None:
                    raise RuntimeError("Failed to create the capture process pipes.")

                self._stderr_tasks = [
                    asyncio.create_task(self._drain_stderr(streamlink.stderr, "streamlink")),
                    asyncio.create_task(self._drain_stderr(ffmpeg.stderr, "ffmpeg")),
                ]
                prune_task = asyncio.create_task(self._prune_loop())
                pump_task = asyncio.create_task(
                    self._pump(streamlink.stdout, ffmpeg.stdin, streamlink)
                )
                await pump_task
                await streamlink.wait()
                await ffmpeg.wait()

                if not self.stop_event.is_set():
                    LOGGER.warning(
                        "Capture ended (Streamlink exit=%s, FFmpeg exit=%s); reconnecting.",
                        streamlink.returncode,
                        ffmpeg.returncode,
                    )
            except FileNotFoundError as exc:
                raise ConfigurationError(
                    "Streamlink and FFmpeg must both be installed and available on PATH."
                ) from exc
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Capture failed; reconnecting.")
            finally:
                await self._terminate(ffmpeg)
                await self._terminate(streamlink)
                for task in self._stderr_tasks:
                    if not task.done():
                        task.cancel()
                if self._stderr_tasks:
                    await asyncio.gather(*self._stderr_tasks, return_exceptions=True)
                if prune_task is not None:
                    prune_task.cancel()
                    await asyncio.gather(prune_task, return_exceptions=True)

            if not self.stop_event.is_set():
                await self._prune_segments()
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 30)

    async def _prune_segments(self) -> None:
        cutoff = time.time() - self.clip_duration - 20
        async with self._segment_lock:
            for path in SEGMENT_DIR.glob(SEGMENT_PATTERN):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except FileNotFoundError:
                    continue

    async def has_full_buffer(self) -> bool:
        if self._buffer_started_at is None:
            return False

        cutoff = max(
            self._buffer_started_at,
            time.time() - self.clip_duration - SEGMENT_SECONDS,
        )
        async with self._segment_lock:
            timestamps = []
            for path in SEGMENT_DIR.glob(SEGMENT_PATTERN):
                try:
                    timestamp = path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if timestamp >= cutoff:
                    timestamps.append(timestamp)

        if len(timestamps) < 2:
            return False
        buffered_seconds = max(timestamps) - min(timestamps) + SEGMENT_SECONDS
        return buffered_seconds >= self.clip_duration

    async def _prune_loop(self) -> None:
        while not self.stop_event.is_set():
            await self._prune_segments()
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass

    async def create_raw_clip(self, destination: Path) -> None:
        """Snapshot enough recent segments and remux them to an MP4."""
        async with self._segment_lock:
            paths = sorted(
                SEGMENT_DIR.glob(SEGMENT_PATTERN),
                key=lambda path: path.stat().st_mtime,
            )
            cutoff = time.time() - self.clip_duration - SEGMENT_SECONDS
            paths = [path for path in paths if path.stat().st_mtime >= cutoff]
            if not paths:
                raise RuntimeError("No buffered stream segments are available yet.")

            snapshot_dir = Path(tempfile.mkdtemp(prefix="clip_segments_", dir=TEMP_DIR))
            copied: list[Path] = []
            for path in paths:
                target = snapshot_dir / path.name
                shutil.copy2(path, target)
                copied.append(target)

        concat_file = snapshot_dir / "segments.txt"
        with concat_file.open("w", encoding="utf-8", newline="\n") as handle:
            for path in copied:
                normalized = path.resolve().as_posix().replace("'", r"'\''")
                handle.write(f"file '{normalized}'\n")

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-t",
            str(self.clip_duration),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        try:
            await run_command(command)
        finally:
            shutil.rmtree(snapshot_dir)


async def run_command(command: list[str]) -> None:
    """Run a local media/metadata command and surface its complete error output."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Required executable not found: {command[0]}") from exc
    stdout, stderr = await _communicate_process(process)
    if process.returncode != 0:
        details = stderr.decode(errors="replace").strip()
        if stdout:
            details = f"{details}\n{stdout.decode(errors='replace').strip()}".strip()
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {command[0]}\n{details}"
        )


@dataclass(frozen=True)
class CaptionWord:
    text: str
    start: float
    end: float


def _srt_time(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def format_subtitles(words: list[CaptionWord], words_per_line: int = 4) -> str:
    """Group timed words into readable, centered 3-5 word SRT captions."""
    if not words:
        return ""
    chunks = [words[index:index + words_per_line] for index in range(0, len(words), words_per_line)]
    if len(chunks) > 1 and len(chunks[-1]) < 3:
        remainder = chunks.pop()
        previous = chunks.pop()
        combined = previous + remainder
        if len(remainder) == 1:
            chunks.append(combined)
        else:
            chunks.extend((combined[:3], combined[3:]))
    entries = []
    for index, chunk in enumerate(chunks, 1):
        text = " ".join(word.text.strip() for word in chunk if word.text.strip()).strip()
        if not text:
            continue
        start = min(word.start for word in chunk)
        end = max(word.end for word in chunk)
        entries.append(f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    return "\n".join(entries)


class Transcriber:
    def __init__(self, model_size: str) -> None:
        self.model_size = model_size
        self._model: Any = None
        self._device: str | None = None

    def _load_cpu_model(self) -> Any:
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="float16",
            )
            self._device = "cuda"
            LOGGER.info("Loaded Whisper model on CUDA.")
        except Exception as exc:
            LOGGER.warning("CUDA Whisper initialization failed; falling back to CPU: %s", exc)
            self._model = self._load_cpu_model()
            self._device = "cpu"
        return self._model

    def _transcribe_with_model(
        self,
        model: Any,
        audio_path: Path,
    ) -> tuple[list[CaptionWord], str]:
        segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
        )
        words: list[CaptionWord] = []
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())
            if segment.words:
                for word in segment.words:
                    words.append(CaptionWord(word.word.strip(), float(word.start), float(word.end)))
            else:
                tokens = segment.text.split()
                if tokens:
                    duration = max(float(segment.end) - float(segment.start), 0.01)
                    step = duration / len(tokens)
                    words.extend(
                        CaptionWord(token, float(segment.start) + i * step, float(segment.start) + (i + 1) * step)
                        for i, token in enumerate(tokens)
                    )
        transcript = " ".join(part for part in transcript_parts if part)
        detected_language = getattr(info, "language", None)
        contains_devanagari = any("\u0900" <= character <= "\u097f" for character in transcript)
        if detected_language != "en" or contains_devanagari:
            LOGGER.info(
                "Captions skipped: speech is not English; only English captions are enabled."
            )
            words = []
        return words, transcript

    def _transcribe_sync(self, audio_path: Path) -> tuple[list[CaptionWord], str]:
        model = self._load_model()
        try:
            return self._transcribe_with_model(model, audio_path)
        except RuntimeError as exc:
            if self._device != "cuda":
                raise
            LOGGER.warning("CUDA Whisper inference failed; retrying on CPU: %s", exc)
            self._model = None
            del model
            self._model = self._load_cpu_model()
            self._device = "cpu"
            return self._transcribe_with_model(self._model, audio_path)

    async def transcribe(self, audio_path: Path) -> tuple[list[CaptionWord], str]:
        return await _run_in_thread_cancellable(
            self._transcribe_sync,
            audio_path,
            cleanup_after_cancel=(audio_path,),
        )


def _subtitle_filter_path(path: Path) -> str:
    # FFmpeg filter syntax needs the drive colon and apostrophes escaped on Windows.
    value = path.resolve().as_posix().replace("\\", "\\\\")
    value = value.replace(":", r"\:").replace("'", r"\'")
    return value


def _ensure_face_detector_model() -> Path | None:
    with _FACE_MODEL_LOCK:
        if FACE_MODEL_PATH.is_file():
            digest = hashlib.sha256(FACE_MODEL_PATH.read_bytes()).hexdigest()
            if digest != FACE_MODEL_SHA256:
                raise RuntimeError(
                    f"Cached face-detection model failed integrity check: {FACE_MODEL_PATH}"
                )
            return FACE_MODEL_PATH

        try:
            with urlopen(FACE_MODEL_URL, timeout=20) as response:
                model_data = response.read()
        except (OSError, URLError, TimeoutError) as exc:
            LOGGER.warning(
                "Could not download the local face detector; using the configured facecam crop: %s",
                exc,
            )
            return None

        if hashlib.sha256(model_data).hexdigest() != FACE_MODEL_SHA256:
            raise RuntimeError("Downloaded face-detection model failed integrity check.")

        FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=FACE_MODEL_PATH.parent,
            prefix="yunet_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(model_data)
        try:
            temporary_path.replace(FACE_MODEL_PATH)
        finally:
            temporary_path.unlink(missing_ok=True)
        LOGGER.info("Cached the local face-detection model at %s.", FACE_MODEL_PATH)
        return FACE_MODEL_PATH


def detect_facecam_crop(video_path: Path) -> tuple[float, float, float, float] | None:
    """Find a stable face location and center the configured crop around it."""
    model_path = _ensure_face_detector_model()
    if model_path is None:
        return None

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        LOGGER.warning("Could not read the clip for facecam detection; using the configured crop.")
        return None

    observations: list[tuple[int, float, float, float, float]] = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            LOGGER.warning("Clip duration is unavailable for facecam detection; using configured crop.")
            return None

        duration = frame_count / fps
        sample_count = min(15, max(3, round(duration / 3)))
        detector = None
        for sample_index in range(sample_count):
            position_seconds = duration * (sample_index + 0.5) / sample_count
            capture.set(cv2.CAP_PROP_POS_MSEC, position_seconds * 1000)
            has_frame, frame = capture.read()
            if not has_frame:
                continue

            height, width = frame.shape[:2]
            if detector is None:
                detector = cv2.FaceDetectorYN.create(
                    str(model_path),
                    "",
                    (width, height),
                    0.65,
                    0.3,
                    5000,
                )
            else:
                detector.setInputSize((width, height))
            _status, faces = detector.detect(frame)
            if faces is None:
                continue
            for face in faces:
                face_x, face_y, face_width, face_height = face[:4]
                observations.append(
                    (
                        sample_index,
                        float((face_x + face_width / 2) / width),
                        float((face_y + face_height / 2) / height),
                        float((face_width * face_height) / (width * height)),
                        float(face[14]),
                    )
                )
    except cv2.error as exc:
        LOGGER.warning(
            "Face detection failed; using the configured facecam crop: %s",
            exc,
        )
        return None
    finally:
        capture.release()

    clusters: list[list[tuple[int, float, float, float, float]]] = []
    for observation in observations:
        _frame_index, center_x, center_y, _area, _confidence = observation
        nearest_cluster = None
        nearest_distance = 0.10
        for cluster in clusters:
            cluster_x = median(item[1] for item in cluster)
            cluster_y = median(item[2] for item in cluster)
            distance = math.hypot(center_x - cluster_x, center_y - cluster_y)
            if distance < nearest_distance:
                nearest_cluster = cluster
                nearest_distance = distance
        if nearest_cluster is None:
            clusters.append([observation])
        else:
            nearest_cluster.append(observation)

    if not clusters:
        LOGGER.warning("No face was consistently detected; using the configured facecam crop.")
        return None

    best_cluster = max(
        clusters,
        key=lambda cluster: (
            len({item[0] for item in cluster}),
            median(item[3] for item in cluster),
            median(item[4] for item in cluster),
        ),
    )
    supporting_frames = len({item[0] for item in best_cluster})
    minimum_support = max(2, (sample_count + 3) // 4)
    if supporting_frames < minimum_support:
        LOGGER.warning(
            "Face detections were not stable across the clip (%d/%d samples); using the configured crop.",
            supporting_frames,
            sample_count,
        )
        return None

    center_x = median(item[1] for item in best_cluster)
    center_y = median(item[2] for item in best_cluster)
    crop_x = max(0.0, min(1.0 - FACECAM_CROP_WIDTH, center_x - FACECAM_CROP_WIDTH / 2))
    crop_y = max(0.0, min(1.0 - FACECAM_CROP_HEIGHT, center_y - FACECAM_CROP_HEIGHT / 2))
    LOGGER.info(
        "Detected a stable facecam location in %d/%d sampled frames.",
        supporting_frames,
        sample_count,
    )
    return crop_x, crop_y, FACECAM_CROP_WIDTH, FACECAM_CROP_HEIGHT


async def render_vertical(
    raw_clip: Path,
    subtitle_file: Path | None,
    output_file: Path,
    voice_audio_file: Path | None = None,
) -> None:
    if not 0 <= GAMEPLAY_CROP_FOCUS_X <= 1:
        raise ConfigurationError("GAMEPLAY_CROP_FOCUS_X must be between 0 and 1.")
    if not 9 / 16 <= GAMEPLAY_CROP_RATIO <= 1:
        raise ConfigurationError("GAMEPLAY_CROP_RATIO must be between 9/16 and 1.")
    if not 0 < STREAMER_PANEL_HEIGHT < 1920 or STREAMER_PANEL_HEIGHT % 2:
        raise ConfigurationError("STREAMER_PANEL_HEIGHT must be an even number between 1 and 1919.")
    if (
        FACECAM_CROP_X < 0
        or FACECAM_CROP_Y < 0
        or FACECAM_CROP_WIDTH <= 0
        or FACECAM_CROP_HEIGHT <= 0
        or FACECAM_CROP_X + FACECAM_CROP_WIDTH > 1
        or FACECAM_CROP_Y + FACECAM_CROP_HEIGHT > 1
    ):
        raise ConfigurationError("The normalized facecam crop must fit within the source frame.")

    facecam_crop = None
    if FACECAM_AUTO_DETECT:
        facecam_crop = await _run_in_thread_cancellable(
            detect_facecam_crop,
            raw_clip,
            cleanup_after_cancel=(raw_clip,),
        )
    if facecam_crop is None:
        facecam_crop = (
            FACECAM_CROP_X,
            FACECAM_CROP_Y,
            FACECAM_CROP_WIDTH,
            FACECAM_CROP_HEIGHT,
        )
    facecam_x, facecam_y, facecam_width, facecam_height = facecam_crop
    gameplay_height = round(1080 / GAMEPLAY_CROP_RATIO)
    gameplay_panel_height = 1920 - STREAMER_PANEL_HEIGHT

    filters = [
        "[0:v]split=2[gamesrc][facesrc]",
        (
            "[gamesrc]crop=ih*"
            f"{GAMEPLAY_CROP_RATIO}:ih:(iw-ow)*{GAMEPLAY_CROP_FOCUS_X}:0,"
            f"scale=1080:{gameplay_height}:flags=lanczos,"
            f"crop=1080:{gameplay_panel_height}[gamefit]"
        ),
        (
            "[facesrc]crop="
            f"iw*{facecam_width}:ih*{facecam_height}:"
            f"iw*{facecam_x}:ih*{facecam_y},"
            f"scale=1080:{STREAMER_PANEL_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop=1080:{STREAMER_PANEL_HEIGHT}[face]"
        ),
    ]
    gameplay = "[gamefit]"
    if subtitle_file is not None:
        subtitle_path = _subtitle_filter_path(subtitle_file)
        force_style = (
            "FontName=Arial,FontSize=12,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,Bold=1,Outline=3,Shadow=1,"
            "Alignment=2,MarginV=36,WrapStyle=2"
        )
        filters.append(
            f"[gamefit]subtitles=filename='{subtitle_path}':"
            f"force_style='{force_style}'[gamecaptioned]"
        )
        gameplay = "[gamecaptioned]"
    filters.append(f"[face]{gameplay}vstack=inputs=2,format=yuv420p[vout]")
    filter_graph = ";".join(filters)
    input_args = ["-i", str(raw_clip)]
    audio_map = "0:a?"
    if voice_audio_file is not None:
        input_args.extend(["-i", str(voice_audio_file)])
        audio_map = "1:a:0"
    await run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *input_args,
            "-map",
            audio_map,
            "-filter_complex",
            filter_graph,
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output_file),
        ]
    )


async def has_audio_stream(path: Path) -> bool:
    """Return whether ffprobe finds an audio stream in the media file."""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ConfigurationError("ffprobe is required for voice-only audio processing.") from exc

    stdout, stderr = await _communicate_process(process)
    if process.returncode != 0:
        details = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffprobe failed for {path}:\n{details}")
    return bool(stdout.strip())


class VoiceStemSeparator:
    """Create a best-effort vocal stem from a clip's mixed audio."""

    def __init__(self) -> None:
        self.output_dir = TEMP_DIR / "voice_stems"
        self._separator: Any = None
        self._lock = threading.Lock()

    async def isolate(self, raw_clip: Path, stem: str) -> Path | None:
        if not await has_audio_stream(raw_clip):
            LOGGER.warning("No audio stream was found; rendering this clip without audio.")
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        input_audio = self.output_dir / f"{stem}_input.wav"
        try:
            await run_command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(raw_clip),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    "44100",
                    "-c:a",
                    "pcm_s16le",
                    str(input_audio),
                ]
            )
            voice_audio = self.output_dir / f"{input_audio.stem}_voice.wav"
            return await _run_in_thread_cancellable(
                self._separate_sync,
                input_audio,
                cleanup_after_cancel=(input_audio, voice_audio),
            )
        finally:
            _cleanup_temporary_files((input_audio,))

    def _separate_sync(self, input_audio: Path) -> Path:
        with self._lock:
            if self._separator is None:
                try:
                    from audio_separator.separator import Separator
                except ModuleNotFoundError as exc:
                    if exc.name not in {"audio_separator", "audio_separator.separator"}:
                        raise
                    raise ConfigurationError(
                        "Voice-only audio processing requires the optional dependency. "
                        "Install it with 'uv sync --extra music-separation'."
                    ) from exc

                self.output_dir.mkdir(parents=True, exist_ok=True)
                VOICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                separator = Separator(
                    output_dir=str(self.output_dir),
                    model_file_dir=str(VOICE_MODEL_DIR),
                    output_format="WAV",
                    output_single_stem="Vocals",
                    log_level=logging.WARNING,
                )
                separator.load_model(model_filename=VOICE_STEM_MODEL)
                self._separator = separator
                LOGGER.info(
                    "Loaded local vocal-separation model %s.",
                    VOICE_STEM_MODEL,
                )

            output_names = {"Vocals": f"{input_audio.stem}_voice"}
            output_files = self._separator.separate(str(input_audio), output_names)
            if len(output_files) != 1:
                raise RuntimeError(
                    f"Expected one vocal stem, but the separator returned {len(output_files)} files."
                )
            voice_audio = Path(output_files[0])
            if not voice_audio.is_absolute():
                voice_audio = self.output_dir / voice_audio
            if not voice_audio.is_file():
                raise RuntimeError(f"Audio separator did not create its vocal stem: {voice_audio}")
            return voice_audio


class MetadataGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is missing from this process environment. "
                "Add it to the project .env file and launch with "
                "'uv run --env-file .env'."
            )
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model

    async def generate(self, transcript: str, channel: str) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "short_form_metadata",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "maxLength": 49},
                            "description": {"type": "string"},
                            "hashtags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 5,
                                "maxItems": 5,
                            },
                        },
                        "required": ["title", "description", "hashtags"],
                        "additionalProperties": False,
                    },
                },
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create accurate short-form metadata for GTA 6 gameplay from the supplied "
                        "transcript. The transcript may be Hindi, Hinglish, or English: understand "
                        "its meaning, but write the title, description, and all hashtags in natural "
                        "English. Make the title and description clearly about GTA 6, and keep all "
                        "five hashtags relevant to GTA 6 gaming. Do not invent events, missions, "
                        "characters, or claims not supported by the transcript. Return JSON with "
                        "title, description, and hashtags (an array of exactly five strings, each "
                        "beginning with #). The title must be under 50 characters and may use emojis."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Game: Grand Theft Auto VI (GTA 6)\n"
                        f"Streamer: {channel}\nTranscript:\n{transcript[:12000]}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty metadata response.")
        data = json.loads(content)
        title = data.get("title")
        description = data.get("description")
        hashtags = data.get("hashtags")
        if (
            not isinstance(title, str)
            or not title.strip()
            or len(title) >= 50
            or not isinstance(description, str)
            or not isinstance(hashtags, list)
            or len(hashtags) != 5
            or any(not isinstance(tag, str) or not tag.startswith("#") for tag in hashtags)
        ):
            raise RuntimeError("Groq metadata did not match the required title/description/hashtag format.")
        return {
            "title": title.strip(),
            "description": description.strip(),
            "hashtags": hashtags,
            "streamer": channel,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "transcript": transcript,
        }


class ClipPipeline:
    def __init__(
        self,
        capture: RollingStreamCapture,
        transcriber: Transcriber,
        metadata: MetadataGenerator,
        channel: str,
        voice_separator: VoiceStemSeparator | None = None,
    ) -> None:
        self.capture = capture
        self.transcriber = transcriber
        self.metadata = metadata
        self.channel = channel
        self.voice_separator = voice_separator or VoiceStemSeparator()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLIPS)

    async def create_clip(
        self,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        async with self.semaphore:
            def report_progress(message: str) -> None:
                if progress_callback is not None:
                    progress_callback(message)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = f"clip_{safe_stem(self.channel)}_{timestamp}"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            raw_clip = TEMP_DIR / f"raw_{stem}.mp4"
            subtitle_file = OUTPUT_DIR / f"{stem}.srt"
            final_file = OUTPUT_DIR / f"{stem}.mp4"
            metadata_file = OUTPUT_DIR / f"{stem}.json"
            voice_audio: Path | None = None
            transcription_task: asyncio.Task[tuple[list[CaptionWord], str]] | None = None
            clip_completed = False

            try:
                report_progress("Cutting a clip from the stream buffer…")
                await self.capture.create_raw_clip(raw_clip)
                if VOICE_ONLY_AUDIO:
                    report_progress("Separating audio…")
                    LOGGER.info("Separating a local vocal stem; this may take several minutes.")
                    voice_audio = await self.voice_separator.isolate(raw_clip, stem)
                if VOICE_ONLY_AUDIO and voice_audio is None:
                    words, transcript = [], ""
                else:
                    transcription_audio = voice_audio or raw_clip
                    report_progress("Transcribing speech…")
                    transcription_task = asyncio.create_task(
                        self.transcriber.transcribe(transcription_audio)
                    )
                    try:
                        words, transcript = await asyncio.wait_for(
                            asyncio.shield(transcription_task),
                            timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        LOGGER.warning(
                            "Speech transcription exceeded %g seconds; continuing without "
                            "captions or transcript metadata.",
                            TRANSCRIPTION_TIMEOUT_SECONDS,
                        )
                        report_progress("Transcription timed out; continuing without captions…")
                        words, transcript = [], ""
                subtitle_text = format_subtitles(words)
                if not subtitle_text:
                    if not transcript.strip():
                        LOGGER.warning("No speech was detected; exporting the clip without burned subtitles.")
                    subtitle_file.write_text("", encoding="utf-8")
                    subtitle_argument = None
                else:
                    subtitle_file.write_text(subtitle_text, encoding="utf-8")
                    subtitle_argument = subtitle_file

                report_progress("Rendering the vertical clip…")
                await render_vertical(
                    raw_clip,
                    subtitle_argument,
                    final_file,
                    voice_audio,
                )
                LOGGER.info("Rendered clip video: %s", final_file)
                metadata = {
                    "title": "",
                    "description": "",
                    "hashtags": [],
                    "streamer": self.channel,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "transcript": transcript,
                }
                if transcript.strip():
                    report_progress("Generating title and description…")
                    metadata.update(await self.metadata.generate(transcript, self.channel))
                else:
                    LOGGER.warning("Skipping Groq metadata because no speech transcript was detected.")
                report_progress("Saving the completed clip…")
                temporary_metadata = metadata_file.with_suffix(".json.tmp")
                temporary_metadata.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary_metadata.replace(metadata_file)
                LOGGER.info("Created clip: %s", final_file)
                clip_completed = True
                report_progress("Clip is ready for review.")
            except Exception:
                LOGGER.exception("Clip processing failed for %s.", stem)
                raise
            finally:
                if transcription_task is not None and not transcription_task.done():
                    _DEFERRED_CLEANUP_PATHS.add(raw_clip)

                    def transcription_finished(
                        task: asyncio.Task[tuple[list[CaptionWord], str]],
                    ) -> None:
                        if not task.cancelled():
                            error = task.exception()
                            if error is not None:
                                LOGGER.error(
                                    "Transcription failed after the clip stopped waiting: %s",
                                    error,
                                )
                        _cleanup_temporary_files((raw_clip,), deferred=True)

                    transcription_task.add_done_callback(transcription_finished)
                if not clip_completed:
                    _cleanup_temporary_files(
                        (
                            subtitle_file,
                            final_file,
                            metadata_file,
                            metadata_file.with_suffix(".json.tmp"),
                        )
                    )
                temporary_files = [raw_clip]
                if voice_audio is not None:
                    temporary_files.append(voice_audio)
                _cleanup_temporary_files(tuple(temporary_files))


async def monitor_chat(
    reader: ChatReader,
    pipeline: ClipPipeline,
    capture: RollingStreamCapture,
    settings: ClipSettings | None = None,
    stop_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    settings = settings or ClipSettings(
        stream_url=getattr(reader, "url", ""),
        chat_spike_multiplier=CHAT_SPIKE_MULTIPLIER,
        clip_duration=CLIP_DURATION,
        clip_pre_roll_seconds=CLIP_PRE_ROLL_SECONDS,
        max_clips=MAX_CLIPS_PER_RUN,
        min_chat_messages=CHAT_SPIKE_MIN_MESSAGES,
        cooldown_seconds=COOLDOWN_PERIOD,
    )
    if settings.max_clips < 1:
        raise ConfigurationError("The clip limit must be at least 1.")
    if not 0 <= settings.clip_pre_roll_seconds < settings.clip_duration:
        raise ConfigurationError(
            "Pre-roll must be non-negative and shorter than the total clip duration."
        )

    message_times: deque[float] = deque()
    monitoring_started_at = time.monotonic()
    last_trigger = float("-inf")
    last_activity_log = float("-inf")
    spike_active = False
    clips_started = 0
    buffer_wait_logged = False
    tasks: set[asyncio.Task[None]] = set()
    active_clip_progress: dict[int, str] = {}

    def publish_progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    def publish_clip_progress() -> None:
        if stop_event is not None and stop_event.is_set():
            return
        if active_clip_progress:
            publish_progress(
                " | ".join(
                    f"Clip {number}: {message}"
                    for number, message in sorted(active_clip_progress.items())
                )
            )
        elif clips_started >= settings.max_clips:
            publish_progress("Finishing the configured clip run…")
        else:
            publish_progress("Watching chat for a fast-activity spike…")

    def completed(task: asyncio.Task[None]) -> None:
        tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            LOGGER.error("Clip task ended with an error: %s", task.exception())

    async def next_event_or_stop() -> tuple[str, Any] | None:
        if stop_event is None:
            return await reader.next_event()
        event_task = asyncio.create_task(reader.next_event())
        stop_task = asyncio.create_task(stop_event.wait())
        done, _ = await asyncio.wait(
            (event_task, stop_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task in done and stop_event.is_set():
            if not event_task.done():
                event_task.cancel()
            await asyncio.gather(event_task, return_exceptions=True)
            return None
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
        return event_task.result()

    reader.start()
    publish_progress("Watching chat for a fast-activity spike…")
    LOGGER.info(
        "Watching chat for fast activity: trigger at %.1fx usual rate or at least "
        "%d messages in %ds; clips are %ds total with %ds before the spike; "
        "maximum %d clips.",
        settings.chat_spike_multiplier,
        settings.min_chat_messages,
        CHAT_WINDOW_SECONDS,
        settings.clip_duration,
        settings.clip_pre_roll_seconds,
        settings.max_clips,
    )
    try:
        while clips_started < settings.max_clips:
            event = await next_event_or_stop()
            if event is None:
                break
            kind, value = event
            if stop_event is not None and stop_event.is_set():
                break
            if kind == "end":
                raise RuntimeError("Chat connection ended.")
            if kind == "error":
                raise RuntimeError(f"Chat reader failed: {value}") from value
            now = time.monotonic()
            message_times.append(now)
            baseline_cutoff = now - CHAT_BASELINE_SECONDS
            while message_times and message_times[0] < baseline_cutoff:
                message_times.popleft()
            spike, recent_count, threshold, baseline_expected = is_chat_spike(
                message_times,
                now,
                monitoring_started_at,
                settings.chat_spike_multiplier,
                settings.min_chat_messages,
            )
            if not spike:
                spike_active = False
            if now - last_activity_log >= 10:
                if baseline_expected > 0:
                    pace_ratio = recent_count / baseline_expected
                    if pace_ratio < 1:
                        comparison = (
                            f"{pace_ratio:.0%} of usual "
                            f"(about {baseline_expected:.1f} messages per "
                            f"{CHAT_WINDOW_SECONDS}s)"
                        )
                    else:
                        comparison = (
                            f"{pace_ratio:.1f}x usual "
                            f"(about {baseline_expected:.1f} messages per "
                            f"{CHAT_WINDOW_SECONDS}s)"
                        )
                else:
                    comparison = "usual rate not established yet"
                LOGGER.info(
                    "Chat pace: %d messages in the last %ds; %s; "
                    "%d needed to trigger a clip.",
                    recent_count,
                    CHAT_WINDOW_SECONDS,
                    comparison,
                    threshold,
                )
                last_activity_log = now
            if (
                spike
                and not spike_active
                and now - last_trigger >= settings.cooldown_seconds
            ):
                if not await capture.has_full_buffer():
                    if not buffer_wait_logged:
                        LOGGER.info(
                            "Building the %d-second stream buffer; clips can start once it is ready.",
                            settings.clip_duration,
                        )
                        publish_progress("Building the initial stream buffer…")
                        buffer_wait_logged = True
                    continue
                if buffer_wait_logged:
                    LOGGER.info("Stream buffer is ready; automatic clipping is active.")
                    buffer_wait_logged = False

                clip_number = clips_started + 1
                wait_seconds = settings.clip_duration - settings.clip_pre_roll_seconds
                active_clip_progress[clip_number] = (
                    f"Chat spike found; capturing {wait_seconds}s more…"
                )
                publish_clip_progress()

                async def create_spike_clip(
                    clip_number: int,
                    wait_seconds: int,
                ) -> None:
                    try:
                        if progress_callback is None:
                            if stop_event is None:
                                await asyncio.sleep(wait_seconds)
                            else:
                                try:
                                    await asyncio.wait_for(
                                        stop_event.wait(),
                                        timeout=wait_seconds,
                                    )
                                except asyncio.TimeoutError:
                                    pass
                        else:
                            deadline = asyncio.get_running_loop().time() + wait_seconds
                            while (remaining := deadline - asyncio.get_running_loop().time()) > 0:
                                active_clip_progress[clip_number] = (
                                    f"Chat spike found; capturing "
                                    f"{math.ceil(remaining)}s more…"
                                )
                                publish_clip_progress()
                                timeout = min(remaining, 1.0)
                                if stop_event is None:
                                    await asyncio.sleep(timeout)
                                else:
                                    try:
                                        await asyncio.wait_for(
                                            stop_event.wait(),
                                            timeout=timeout,
                                        )
                                    except asyncio.TimeoutError:
                                        pass
                                    if stop_event.is_set():
                                        return
                        if stop_event is not None and stop_event.is_set():
                            return
                        if progress_callback is None:
                            await pipeline.create_clip()
                        else:
                            def update_clip_progress(message: str) -> None:
                                active_clip_progress[clip_number] = message
                                publish_clip_progress()

                            await pipeline.create_clip(update_clip_progress)
                            publish_clip_progress()
                    finally:
                        active_clip_progress.pop(clip_number, None)
                        publish_clip_progress()

                last_trigger = now
                spike_active = True
                LOGGER.info(
                    "Fast chat detected: %d messages in %ds (trigger threshold: %d).",
                    recent_count,
                    CHAT_WINDOW_SECONDS,
                    threshold,
                )
                task = asyncio.create_task(create_spike_clip(clip_number, wait_seconds))
                tasks.add(task)
                clips_started += 1
                task.add_done_callback(completed)
                if clips_started == settings.max_clips:
                    LOGGER.info(
                        "Per-run clip limit reached (%d); stopping chat monitoring.",
                        settings.max_clips,
                    )
    finally:
        reader.stop()
        capture.stop()
        pending_tasks = tuple(tasks)
        if pending_tasks:
            gather_task = asyncio.gather(*pending_tasks, return_exceptions=True)
            if stop_event is None:
                await gather_task
            else:
                stop_task = asyncio.create_task(stop_event.wait())
                done, _ = await asyncio.wait(
                    (gather_task, stop_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done and stop_event.is_set() and not gather_task.done():
                    active_tasks = tuple(task for task in pending_tasks if not task.done())
                    if active_tasks:
                        LOGGER.info(
                            "Stop requested; canceling %d in-progress clip task(s).",
                            len(active_tasks),
                        )
                        for task in active_tasks:
                            task.cancel()
                    await gather_task
                else:
                    stop_task.cancel()
                    await asyncio.gather(stop_task, return_exceptions=True)
                    await gather_task


def validate_runtime_dependencies() -> None:
    ensure_ffmpeg_on_path()
    missing_programs = [
        name for name in ("streamlink", "ffmpeg") if shutil.which(name) is None
    ]
    if VOICE_ONLY_AUDIO and shutil.which("ffprobe") is None:
        missing_programs.append("ffprobe")
    if missing_programs:
        missing_list = ", ".join(missing_programs)
        if missing_programs == ["ffmpeg"]:
            raise ConfigurationError(
                "FFmpeg was not found on PATH or in WinGet's standard install "
                "directory. Install it with 'winget install --id Gyan.FFmpeg -e' "
                "and ensure its bin folder is on PATH."
            )
        raise ConfigurationError(
            f"Required executable(s) not found on PATH: {missing_list}. "
            "Run 'uv sync' for Streamlink and install FFmpeg separately."
        )


async def run_clip_session(
    settings: ClipSettings,
    stop_event: asyncio.Event,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    url = settings.stream_url
    if "://" not in url:
        url = f"https://{url}"
    platform = detect_platform(url)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Detected %s URL; channel=%s.", platform, streamer_name(url))
    capture = RollingStreamCapture(url, settings.clip_duration)
    reader = ChatReader(url)
    pipeline = ClipPipeline(
        capture=capture,
        transcriber=Transcriber(WHISPER_MODEL_SIZE),
        metadata=MetadataGenerator(GROQ_API_KEY, GROQ_MODEL),
        channel=streamer_name(url),
    )
    capture_task = asyncio.create_task(capture.run(), name="stream-capture")
    try:
        await monitor_chat(
            reader,
            pipeline,
            capture,
            settings,
            stop_event,
            progress_callback,
        )
    finally:
        reader.stop()
        capture.stop()
        if not capture_task.done():
            capture_task.cancel()
        await asyncio.gather(capture_task, return_exceptions=True)


class ClipperController:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._lock = threading.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._state: dict[str, Any] = {
            "status": "idle",
            "stream_url": "",
            "error": "",
            "progress": "",
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = ClipSettings.from_payload(payload)
        future = asyncio.run_coroutine_threadsafe(self._start(settings), self.loop)
        try:
            return future.result(timeout=10)
        except TimeoutError as exc:
            future.cancel()
            raise RuntimeError("Could not start the clipper; check the server logs.") from exc

    async def _start(self, settings: ClipSettings) -> dict[str, Any]:
        if self._task is not None and not self._task.done():
            raise RuntimeError("A stream is already being monitored.")
        self._stop_event = asyncio.Event()
        with self._lock:
            self._state = {
                "status": "starting",
                "stream_url": settings.stream_url,
                "settings": asdict(settings),
                "error": "",
                "progress": "Starting the clipper…",
            }
        self._task = asyncio.create_task(
            self._run(settings, self._stop_event),
            name="clipper-session",
        )
        return {"ok": True, "status": "starting"}

    async def _run(self, settings: ClipSettings, stop_event: asyncio.Event) -> None:
        try:
            self._set_progress("Checking required tools…")
            validate_runtime_dependencies()
            if stop_event.is_set():
                with self._lock:
                    self._state["status"] = "stopped"
                    self._state["progress"] = "Stopped."
                return
            with self._lock:
                self._state["status"] = "observing"
            self._set_progress("Connecting to the stream…")
            await run_clip_session(settings, stop_event, self._set_progress)
            final_status = "stopped" if stop_event.is_set() else "completed"
            with self._lock:
                self._state["status"] = final_status
                self._state["progress"] = (
                    "Stopped." if final_status == "stopped" else "Run complete."
                )
        except Exception as exc:
            LOGGER.exception("Clipper session failed.")
            with self._lock:
                self._state["status"] = "error"
                self._state["error"] = str(exc)

    def _set_progress(self, message: str) -> None:
        with self._lock:
            if self._state.get("status") in {"starting", "observing"}:
                self._state["progress"] = message

    def stop(self) -> dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(self._request_stop(), self.loop)
        try:
            return future.result(timeout=10)
        except TimeoutError as exc:
            future.cancel()
            raise RuntimeError("Could not stop the clipper; check the server logs.") from exc

    async def _request_stop(self) -> dict[str, Any]:
        if self._task is None or self._task.done():
            return {"ok": True, "status": self.status()["status"]}
        with self._lock:
            self._state["status"] = "stopping"
            self._state["progress"] = "Stopping and discarding the current clip…"
        if self._stop_event is not None:
            self._stop_event.set()
        return {"ok": True, "status": "stopping"}

    async def shutdown(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            await self._task


async def run_dashboard() -> None:
    host = os.getenv("WEB_HOST", "127.0.0.1")
    username = os.getenv("STREAMCLIPPER_USERNAME", "")
    password = os.getenv("STREAMCLIPPER_PASSWORD", "")
    if host not in {"127.0.0.1", "localhost", "::1"} and not (username and password):
        raise ConfigurationError(
            "Set STREAMCLIPPER_USERNAME and STREAMCLIPPER_PASSWORD before binding "
            "the dashboard to a public interface."
        )
    try:
        port = int(os.getenv("PORT", os.getenv("WEB_PORT", "55842")))
    except ValueError as exc:
        raise ConfigurationError("PORT or WEB_PORT must be a valid port number.") from exc
    if not 0 <= port <= 65535:
        raise ConfigurationError("PORT or WEB_PORT must be between 0 and 65535.")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    controller = ClipperController(asyncio.get_running_loop())
    review_server = ClipReviewServer(
        OUTPUT_DIR,
        YOUTUBE_CLIENT_SECRETS_FILE,
        YOUTUBE_TOKEN_FILE,
        controller=controller,
        default_settings=asdict(ClipSettings()),
        username=username,
        password=password,
        host=host,
        port=port,
    )
    review_server.start()
    LOGGER.info("StreamClipper dashboard: %s", review_server.url)
    if host in {"127.0.0.1", "localhost", "::1"}:
        try:
            webbrowser.open(review_server.url, new=2)
        except (webbrowser.Error, OSError):
            LOGGER.exception("Could not open the dashboard automatically.")
    try:
        await asyncio.to_thread(review_server.wait_until_closed)
    finally:
        await controller.shutdown()
        review_server.stop()


def main() -> None:
    try:
        asyncio.run(run_dashboard())
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user.")
    except ConfigurationError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    except Exception as exc:
        LOGGER.exception("Application stopped: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
