import json
import unittest
import os
import threading
import tempfile
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from live_clipper import (
    CaptionWord,
    ChatReader,
    ClipPipeline,
    ConfigurationError,
    detect_platform,
    ensure_ffmpeg_on_path,
    format_subtitles,
    is_chat_spike,
    message_timestamp,
    MetadataGenerator,
    monitor_chat,
    render_vertical,
    streamer_name,
    Transcriber,
    youtube_video_id,
)


class PlatformTests(unittest.TestCase):
    def test_detects_supported_platform_urls(self):
        self.assertEqual(detect_platform("https://twitch.tv/example"), "twitch")
        self.assertEqual(detect_platform("https://youtube.com/watch?v=abc"), "youtube")
        self.assertEqual(detect_platform("https://kick.com/example"), "kick")

    def test_detects_platform_url_without_scheme(self):
        self.assertEqual(detect_platform("twitch.tv/Lysium"), "twitch")

    def test_rejects_non_live_platform_url(self):
        with self.assertRaises(ConfigurationError):
            detect_platform("https://example.com/watch")

    def test_extracts_youtube_channel_handle(self):
        self.assertEqual(streamer_name("https://youtube.com/@creator/live"), "creator")

    def test_extracts_youtube_video_id_from_supported_urls(self):
        self.assertEqual(
            youtube_video_id("https://www.youtube.com/watch?v=video123"),
            "video123",
        )
        self.assertEqual(youtube_video_id("https://youtu.be/video123"), "video123")
        self.assertEqual(
            youtube_video_id("https://youtube.com/live/video123"),
            "video123",
        )
        self.assertEqual(streamer_name("https://youtube.com/watch?v=video123"), "video123")


class YouTubeChatReaderTests(unittest.TestCase):
    def test_uses_pytchat_and_normalizes_messages(self):
        timestamp = 1_794_992_978_642

        class FakeChat:
            checks = 0
            terminated = False

            def is_alive(self):
                self.checks += 1
                return self.checks == 1

            def get(self):
                return SimpleNamespace(
                    sync_items=lambda: iter(
                        [SimpleNamespace(timestamp=timestamp, message="hello")]
                    )
                )

            def terminate(self):
                self.terminated = True

        fake_chat = FakeChat()
        create_kwargs = {}

        def create_chat(**kwargs):
            create_kwargs.update(kwargs)
            if (
                threading.current_thread() is not threading.main_thread()
                and kwargs.get("interruptable", True)
            ):
                raise ValueError("pytchat cannot install signal handlers in a worker thread")
            return fake_chat

        reader = ChatReader("https://www.youtube.com/watch?v=video123")
        with patch.dict(
            "sys.modules",
            {"pytchat": SimpleNamespace(create=create_chat)},
        ):
            reader.start()
            self.assertIsNotNone(reader.thread)
            reader.thread.join(timeout=2)
            self.assertFalse(reader.thread.is_alive())

        self.assertEqual(
            create_kwargs,
            {"video_id": "video123", "interruptable": False},
        )
        kind, message = reader.events.get_nowait()
        self.assertEqual(kind, "message")
        self.assertEqual(message["message"], "hello")
        self.assertAlmostEqual(
            message_timestamp(message),
            timestamp / 1000,
        )
        self.assertEqual(reader.events.get_nowait(), ("end", None))
        reader.stop()
        self.assertTrue(fake_chat.terminated)


class SubtitleTests(unittest.TestCase):
    def test_rebalances_short_final_caption(self):
        words = [
            CaptionWord(f"w{index}", float(index), float(index) + 0.5)
            for index in range(5)
        ]
        result = format_subtitles(words)
        captions = [part for part in result.strip().split("\n\n") if part]
        self.assertEqual(len(captions), 1)
        self.assertEqual(len(captions[0].splitlines()[2].split()), 5)

    def test_rebalances_two_word_remainder(self):
        words = [
            CaptionWord(f"w{index}", float(index), float(index) + 0.5)
            for index in range(6)
        ]
        result = format_subtitles(words)
        captions = [part for part in result.strip().split("\n\n") if part]
        self.assertEqual([len(caption.splitlines()[2].split()) for caption in captions], [3, 3])


class FfmpegPathTests(unittest.TestCase):
    def test_discovers_winget_ffmpeg_and_adds_its_bin_directory_to_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = (
                Path(temp_dir)
                / "Microsoft"
                / "WinGet"
                / "Packages"
                / "Gyan.FFmpeg_test"
                / "ffmpeg-build"
                / "bin"
            )
            bin_dir.mkdir(parents=True)
            (bin_dir / "ffmpeg.exe").touch()
            with patch.dict(os.environ, {"LOCALAPPDATA": temp_dir, "PATH": ""}):
                with patch("live_clipper.shutil.which", return_value=None):
                    self.assertTrue(ensure_ffmpeg_on_path())
                self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], str(bin_dir))


class TranscriberTests(unittest.TestCase):
    def test_only_english_transcripts_keep_caption_words(self):
        transcriber = Transcriber("base")

        for language, text, expect_captions in (
            ("en", "Hello there", True),
            ("hi", "नमस्ते", False),
            ("en", "Hello नमस्ते", False),
        ):
            with self.subTest(language=language, text=text):
                class FakeModel:
                    def transcribe(self, *_args, **_kwargs):
                        segment = SimpleNamespace(
                            text=text,
                            start=0.0,
                            end=1.0,
                            words=[SimpleNamespace(word="Hello", start=0.0, end=0.5)],
                        )
                        return [segment], SimpleNamespace(language=language)

                words, transcript = transcriber._transcribe_with_model(
                    FakeModel(),
                    Path("clip.mp4"),
                )

                self.assertEqual(bool(words), expect_captions)
                self.assertEqual(transcript, text)

    def test_cuda_runtime_failure_retries_transcription_on_cpu(self):
        transcriber = Transcriber("base")
        cuda_model = object()
        cpu_model = object()
        transcriber._device = "cuda"

        with (
            patch.object(transcriber, "_load_model", return_value=cuda_model),
            patch.object(
                transcriber,
                "_transcribe_with_model",
                side_effect=[RuntimeError("Library cublas64_12.dll is not found"), ([], "ok")],
            ) as transcribe,
            patch.object(transcriber, "_load_cpu_model", return_value=cpu_model),
        ):
            result = transcriber._transcribe_sync(Path("clip.mp4"))

        self.assertEqual(result, ([], "ok"))
        self.assertEqual(transcriber._device, "cpu")
        self.assertIs(transcriber._model, cpu_model)
        self.assertEqual(transcribe.call_args_list[0].args[0], cuda_model)
        self.assertEqual(transcribe.call_args_list[1].args[0], cpu_model)


class RenderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        face_detection = patch("live_clipper.detect_facecam_crop", return_value=None)
        face_detection.start()
        self.addCleanup(face_detection.stop)

    async def test_portrait_render_fills_frame_with_gameplay_and_facecam_overlay(self):
        with patch("live_clipper.run_command", new_callable=AsyncMock) as run_command:
            await render_vertical(Path("raw.mp4"), None, Path("final.mp4"))

        command = run_command.await_args.args[0]
        filter_graph = command[command.index("-filter_complex") + 1]
        self.assertIn("split=2[gamesrc][facesrc]", filter_graph)
        self.assertIn("crop=ih*0.75:ih:(iw-ow)*0.5:0", filter_graph)
        self.assertIn("scale=1080:1440:flags=lanczos", filter_graph)
        self.assertIn("pad=1080:1920:0:240:color=black[game]", filter_graph)
        self.assertIn("crop=iw*0.3:ih*0.32:iw*0.7:ih*0.68", filter_graph)
        self.assertIn("overlay=W-w-40:40", filter_graph)
        self.assertNotIn("subtitles=", filter_graph)
        self.assertIn("[vout]", command)
        self.assertNotIn("0:v:0", command)

    async def test_portrait_render_uses_detected_facecam_position(self):
        with (
            patch("live_clipper.detect_facecam_crop", return_value=(0.25, 0.4, 0.3, 0.32)),
            patch("live_clipper.run_command", new_callable=AsyncMock) as run_command,
        ):
            await render_vertical(Path("raw.mp4"), None, Path("final.mp4"))

        command = run_command.await_args.args[0]
        filter_graph = command[command.index("-filter_complex") + 1]
        self.assertIn("crop=iw*0.3:ih*0.32:iw*0.25:ih*0.4", filter_graph)

    async def test_portrait_render_uses_separated_audio_track(self):
        voice_audio = Path("voice.wav")
        with patch("live_clipper.run_command", new_callable=AsyncMock) as run_command:
            await render_vertical(
                Path("raw.mp4"),
                None,
                Path("final.mp4"),
                voice_audio,
            )

        command = run_command.await_args.args[0]
        self.assertIn(str(voice_audio), command)
        map_index = command.index("-map")
        self.assertEqual(command[map_index + 1], "1:a:0")
        self.assertNotIn("0:a?", command)

    async def test_subtitles_are_applied_to_the_fitted_source_frame(self):
        with patch("live_clipper.run_command", new_callable=AsyncMock) as run_command:
            await render_vertical(
                Path("raw.mp4"),
                Path("captions.srt"),
                Path("final.mp4"),
            )

        command = run_command.await_args.args[0]
        filter_graph = command[command.index("-filter_complex") + 1]
        self.assertIn("[gamefit]subtitles=filename=", filter_graph)
        self.assertIn("[gamecaptioned]", filter_graph)
        self.assertIn("[gamecaptioned]pad=1080:1920:0:240:color=black[game]", filter_graph)
        self.assertIn("[game][face]overlay=", filter_graph)


class MetadataTests(unittest.TestCase):
    def test_metadata_client_uses_groq_compatible_endpoint(self):
        with patch("openai.AsyncOpenAI") as client:
            generator = MetadataGenerator("test-key", "openai/gpt-oss-20b")

        client.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        self.assertEqual(generator.model, "openai/gpt-oss-20b")


class MetadataResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_requests_strict_json_and_validates_the_result(self):
        response_data = {
            "title": "Wild stream moment!",
            "description": "A surprising moment from the live stream.",
            "hashtags": ["#Shorts", "#Live", "#Gaming", "#Streamer", "#Clip"],
        }
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(response_data))
                    )
                ]
            )
        )
        generator = MetadataGenerator("test-key", "openai/gpt-oss-20b")
        generator.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        result = await generator.generate("A transcript.", "creator")

        self.assertEqual(result["title"], response_data["title"])
        self.assertEqual(result["hashtags"], response_data["hashtags"])
        response_format = create.await_args.kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])


class ClipLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_stops_after_configured_number_of_clip_triggers(self):
        class FakeReader:
            stopped = False

            def start(self):
                pass

            async def next_event(self):
                return "message", {"timestamp": time.time()}

            def stop(self):
                self.stopped = True

        class FakePipeline:
            clips_started = 0

            async def create_clip(self):
                self.clips_started += 1

        class FakeCapture:
            stopped = False

            async def has_full_buffer(self):
                return True

            def stop(self):
                self.stopped = True

        reader = FakeReader()
        pipeline = FakePipeline()
        capture = FakeCapture()
        with (
            patch("live_clipper.MAX_CLIPS_PER_RUN", 1),
            patch("live_clipper.CHAT_SPIKE_MIN_MESSAGES", 1),
            patch("live_clipper.CHAT_WINDOW_SECONDS", 5),
            patch("live_clipper.COOLDOWN_PERIOD", 0),
            patch("live_clipper.CLIP_DURATION", 1),
            patch("live_clipper.CLIP_PRE_ROLL_SECONDS", 0),
            patch("live_clipper.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            await monitor_chat(reader, pipeline, capture)

        sleep.assert_awaited_once_with(1)
        self.assertEqual(pipeline.clips_started, 1)
        self.assertTrue(reader.stopped)
        self.assertTrue(capture.stopped)

    async def test_waits_until_stream_buffer_is_full_before_spending_limit(self):
        class FakeReader:
            stopped = False
            messages = iter(
                [
                    ("message", {"timestamp": time.time()}),
                    ("message", {"timestamp": time.time()}),
                ]
            )

            def start(self):
                pass

            async def next_event(self):
                return next(self.messages)

            def stop(self):
                self.stopped = True

        class FakePipeline:
            clips_started = 0

            async def create_clip(self):
                self.clips_started += 1

        class FakeCapture:
            stopped = False
            readiness = iter([False, True])

            async def has_full_buffer(self):
                return next(self.readiness)

            def stop(self):
                self.stopped = True

        reader = FakeReader()
        pipeline = FakePipeline()
        capture = FakeCapture()
        with (
            patch("live_clipper.MAX_CLIPS_PER_RUN", 1),
            patch("live_clipper.CHAT_SPIKE_MIN_MESSAGES", 1),
            patch("live_clipper.CHAT_WINDOW_SECONDS", 5),
            patch("live_clipper.COOLDOWN_PERIOD", 0),
            patch("live_clipper.CLIP_DURATION", 1),
            patch("live_clipper.CLIP_PRE_ROLL_SECONDS", 0),
            patch("live_clipper.asyncio.sleep", new_callable=AsyncMock),
        ):
            await monitor_chat(reader, pipeline, capture)

        self.assertEqual(pipeline.clips_started, 1)
        self.assertTrue(reader.stopped)
        self.assertTrue(capture.stopped)

    async def test_schedules_only_one_clip_for_a_sustained_spike(self):
        class FakeReader:
            stopped = False
            events = iter(
                [
                    ("message", {"timestamp": time.time()}),
                    ("message", {"timestamp": time.time()}),
                    ("end", None),
                ]
            )

            def start(self):
                pass

            async def next_event(self):
                return next(self.events)

            def stop(self):
                self.stopped = True

        class FakePipeline:
            clips_started = 0

            async def create_clip(self):
                self.clips_started += 1

        class FakeCapture:
            stopped = False

            async def has_full_buffer(self):
                return True

            def stop(self):
                self.stopped = True

        reader = FakeReader()
        pipeline = FakePipeline()
        capture = FakeCapture()
        with (
            patch("live_clipper.MAX_CLIPS_PER_RUN", 5),
            patch("live_clipper.CHAT_SPIKE_MIN_MESSAGES", 1),
            patch("live_clipper.CHAT_WINDOW_SECONDS", 5),
            patch("live_clipper.COOLDOWN_PERIOD", 0),
            patch("live_clipper.CLIP_DURATION", 1),
            patch("live_clipper.CLIP_PRE_ROLL_SECONDS", 0),
            patch("live_clipper.asyncio.sleep", new_callable=AsyncMock),
        ):
            with self.assertRaises(RuntimeError):
                await monitor_chat(reader, pipeline, capture)

        self.assertEqual(pipeline.clips_started, 1)
        self.assertTrue(reader.stopped)
        self.assertTrue(capture.stopped)


class ChatSpikeTests(unittest.TestCase):
    def test_requires_minimum_burst_when_no_baseline_messages_exist(self):
        spike, count, threshold, baseline_expected = is_chat_spike(
            deque([4.5, 4.9]),
            now=5.0,
            monitoring_started_at=0.0,
        )
        self.assertTrue(spike)
        self.assertEqual((count, threshold, baseline_expected), (2, 2, 0))

        spike, count, threshold, baseline_expected = is_chat_spike(
            deque([4.9]),
            now=5.0,
            monitoring_started_at=0.0,
        )
        self.assertFalse(spike)
        self.assertEqual((count, threshold, baseline_expected), (1, 2, 0))

    def test_requires_three_times_the_recent_baseline_rate(self):
        with (
            patch("live_clipper.CHAT_SPIKE_MIN_MESSAGES", 5),
            patch("live_clipper.CHAT_SPIKE_MULTIPLIER", 3),
        ):
            baseline = deque(float(index) for index in range(60))
            below_threshold = baseline + deque(
                60.1 + index * 0.3 for index in range(10)
            )
            spike, count, threshold, baseline_expected = is_chat_spike(
                below_threshold,
                now=65.0,
                monitoring_started_at=0.0,
            )
            self.assertFalse(spike)
            self.assertEqual((count, threshold, baseline_expected), (10, 15, 5))

            burst = baseline + deque(
                60.1 + index * 0.3 for index in range(16)
            )
            spike, count, threshold, baseline_expected = is_chat_spike(
                burst,
                now=65.0,
                monitoring_started_at=0.0,
            )
            self.assertTrue(spike)
            self.assertEqual((count, threshold, baseline_expected), (16, 15, 5))


class SilentClipTests(unittest.IsolatedAsyncioTestCase):
    async def test_silent_clip_skips_metadata_generation(self):
        class FakeCapture:
            async def create_raw_clip(self, destination):
                destination.write_bytes(b"raw")

        class FakeTranscriber:
            async def transcribe(self, _audio_path):
                return [], ""

        metadata = SimpleNamespace(generate=AsyncMock())
        pipeline = ClipPipeline(
            FakeCapture(),
            FakeTranscriber(),
            metadata,
            "creator",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("live_clipper.TEMP_DIR", root / "temp"),
                patch("live_clipper.OUTPUT_DIR", root / "output"),
                patch("live_clipper.VOICE_ONLY_AUDIO", False),
                patch("live_clipper.render_vertical", new_callable=AsyncMock) as render,
            ):
                await pipeline.create_clip()

            metadata.generate.assert_not_awaited()
            self.assertIsNone(render.await_args.args[1])
            self.assertEqual(len(list((root / "output").glob("clip_*.srt"))), 1)
            self.assertEqual(list((root / "output").glob("*.json")), [])


class VoiceOnlyClipTests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_stem_is_used_for_transcription_and_output(self):
        class FakeCapture:
            async def create_raw_clip(self, destination):
                destination.write_bytes(b"raw")

        class FakeVoiceSeparator:
            def __init__(self, voice_audio):
                self.voice_audio = voice_audio

            async def isolate(self, _raw_clip, _stem):
                return self.voice_audio

        class FakeTranscriber:
            audio_path = None

            async def transcribe(self, audio_path):
                self.audio_path = audio_path
                return [], ""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voice_audio = root / "voice.wav"
            voice_audio.write_bytes(b"separated audio")
            transcriber = FakeTranscriber()
            pipeline = ClipPipeline(
                FakeCapture(),
                transcriber,
                SimpleNamespace(generate=AsyncMock()),
                "creator",
                FakeVoiceSeparator(voice_audio),
            )
            with (
                patch("live_clipper.TEMP_DIR", root / "temp"),
                patch("live_clipper.OUTPUT_DIR", root / "output"),
                patch("live_clipper.VOICE_ONLY_AUDIO", True),
                patch("live_clipper.render_vertical", new_callable=AsyncMock) as render,
            ):
                await pipeline.create_clip()

            self.assertEqual(transcriber.audio_path, voice_audio)
            self.assertEqual(render.await_args.args[3], voice_audio)
            self.assertFalse(voice_audio.exists())


if __name__ == "__main__":
    unittest.main()
