# Live Stream Clipper

This application watches chat for spikes, snapshots recent live-stream segments,
transcribes speech, preserves the full source frame in a 9:16 composition, and
writes JSON metadata. Its optional voice-only audio processing is experimental.
Use it only with streams you own or have explicit permission to capture and
republish. It does not bypass DRM or platform access controls.

## Requirements

- Python 3.10+
- FFmpeg with `ffmpeg` on `PATH` (the `subtitles` filter requires an FFmpeg build
  with libass; H.264 output requires libx264)
- Streamlink on `PATH`
- A Groq API key (metadata generation)
- CUDA is optional; Whisper falls back to CPU if CUDA initialization or inference fails

Streamlink and chat-provider support depends on the stream's availability.
YouTube watch/live-video URLs use `pytchat` for live chat; Twitch and Kick use
ChatDownloader. A URL can be recognized by this application even when the
provider cannot access its live video or chat; such errors are reported rather
than silently treated as success.

## Install and run

Install FFmpeg separately on Windows; it is a system executable, not a Python
package managed by `uv`. Activating `.venv` does not install or add FFmpeg:

```powershell
winget install --id Gyan.FFmpeg -e
```

After the installation completes, close the current PowerShell window and open a
new one so it picks up the updated system `PATH`. Verify that
`Get-Command ffmpeg` finds FFmpeg and `ffmpeg -version` runs. If `winget` reports
that FFmpeg is already installed but PowerShell still cannot find it, add the
installed executable's folder to the current PowerShell session. The app also
searches WinGet's standard `Gyan.FFmpeg` package directory automatically:

```powershell
$ffmpeg = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*" `
  -Filter ffmpeg.exe -File -Recurse |
  Select-Object -First 1 -ExpandProperty FullName
if (-not $ffmpeg) { throw "Could not locate winget's ffmpeg.exe" }
$env:Path = "$(Split-Path $ffmpeg);$env:Path"
ffmpeg -version
```

This PATH change lasts for the current PowerShell window. Install the Python
dependencies, including the optional CPU audio-separation package used by the
default voice-only mode:

```powershell
uv sync --extra music-separation
```

The audio-separation extra includes PyTorch and ONNX Runtime. The selected
vocal-separation model (about 53 MB) downloads on first use and is cached under
`%LOCALAPPDATA%`. Audio processing runs locally; stream audio is not uploaded.

Create a Groq API key in GroqCloud, then store it in the project-root `.env`
file (which is ignored by Git):

```dotenv
GROQ_API_KEY=your-groq-key-goes-here
```

Load `.env` when launching with `uv`:

```powershell
uv run --extra music-separation --env-file .env python .\live_clipper.py
```

The variable name must be exactly `GROQ_API_KEY`. Do not share or commit your
key. `uv run` does not load `.env` automatically unless you pass `--env-file`.
Metadata uses Groq's OpenAI-compatible API with the `openai/gpt-oss-20b` model
by default. Set `GROQ_MODEL` in `.env` to select another Groq-supported model.

At the prompt, enter only the stream URL, for example
`https://www.twitch.tv/Lysium` or `twitch.tv/Lysium` (the scheme is optional).
Run shell commands such as
`uv run --extra music-separation --env-file .env python .\live_clipper.py`
at the PowerShell prompt, not inside the app's URL prompt. You can instead set
`STREAM_URL` near the top of `live_clipper.py`.

The app writes final MP4s and SRT files to `output\`, and writes metadata JSON
when speech is transcribed and Groq returns valid metadata. Silent clips are
rendered without subtitles and skip metadata generation.
Temporary clips and the rolling stream-segment buffer are written under `temp\`.
The segment buffer is pruned as it ages. Initial buffering takes approximately
the configured clip duration, and clips can only be generated after enough live
segments have arrived.

Voice-only processing is best-effort music-vocal separation, not a reliable
music detector or speech-versus-game-audio separator. It is applied to every
clip when `VOICE_ONLY_AUDIO` is enabled. It may suppress game sound effects or
leave sung vocals in the output, and it cannot guarantee a clip will avoid a
copyright claim. For reliable results, capture a separate streamer microphone
track at the source and use only audio you have permission to publish.

Vertical exports crop gameplay to a wider 3:4 view and place it in a 1080x1920
portrait frame with 240-pixel black margins above and below. The facecam stays
at the same top-right position. Adjust `GAMEPLAY_CROP_RATIO` in `live_clipper.py`
to tune the zoom; the default is 0.75. Captions are burned only when Whisper
detects English and the transcript contains no Devanagari text. Other-language
speech remains available in the transcript and metadata but does not get burned
captions.
Automatic facecam detection samples
the clip and looks for a stable face position; its small YuNet model downloads
from OpenCV Zoo on first use and is cached under `%LOCALAPPDATA%`. If detection
is unavailable or uncertain, the app uses the configured crop (default:
x=0.70, y=0.68, width=0.30, height=0.32). Detection is best-effort and can
mistake a stable in-game face for the streamer, so review new layouts and adjust
the crop constants in `live_clipper.py` when needed. Set
`FACECAM_AUTO_DETECT = False` to disable detection. Exports use high-quality
Lanczos scaling and H.264 CRF 18. Captions are white, bold, outlined, and
bottom-centered with a margin intended to keep them clear of typical player
controls.

## Configuration

Edit the constants near the top of `live_clipper.py`:

- `STREAM_URL`: set a URL, or leave empty for runtime input
- `CHAT_WINDOW_SECONDS`: burst window used to measure recent chat speed (5 seconds by default)
- `CHAT_BASELINE_SECONDS`: recent history used as the normal chat-rate baseline (300 seconds)
- `CHAT_SPIKE_MULTIPLIER`: how many times faster chat must be than its baseline (1.3 by default)
- `CHAT_SPIKE_MIN_MESSAGES`: minimum messages in the burst window, even when baseline chat is quiet (2 by default)
- `CLIP_DURATION`: total clip length (45 seconds by default)
- `CLIP_PRE_ROLL_SECONDS`: how many seconds before the detected spike to include (12 by default)
- `COOLDOWN_PERIOD`: minimum seconds between spike triggers
- `MAX_CLIPS_PER_RUN`: maximum clip attempts before the app stops (currently `5`)
- `WHISPER_MODEL_SIZE`: `small` or `base`
- `VOICE_ONLY_AUDIO`: enable best-effort local vocal-stem processing
- `VOICE_STEM_MODEL`: audio-separation model used for the vocal stem
- `GAMEPLAY_CROP_FOCUS_X`: horizontal gameplay focus, from `0` (left) to `1` (right)
- `GAMEPLAY_CROP_RATIO`: gameplay crop width divided by its height; larger values zoom out slightly
- `FACECAM_AUTO_DETECT`: locate a stable face position before cropping
- `FACECAM_CROP_X`, `FACECAM_CROP_Y`, `FACECAM_CROP_WIDTH`, `FACECAM_CROP_HEIGHT`: normalized facecam rectangle in the landscape source
- `FACECAM_OUTPUT_WIDTH`: facecam inset width in output pixels
- `FACECAM_MARGIN`: inset distance from the top and right edges in pixels
- `GROQ_API_KEY`: preferably stored in the ignored `.env` file
- `GROQ_MODEL`: optional Groq model ID (defaults to `openai/gpt-oss-20b`)

The rolling buffer is assembled from short MPEG-TS segments; segment boundaries
are constrained by the live stream's keyframes, so clip edges may vary slightly.
When chat activity reaches the adaptive spike threshold, the app waits long
enough to capture the configured total clip length with its pre-roll included.
Periodic chat activity logs report the current rate as a multiple of the recent
baseline; the ratio is unavailable until prior chat messages establish a baseline.

Run the focused tests with:

```powershell
uv run --extra music-separation python -m unittest -v
```
