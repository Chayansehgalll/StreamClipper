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
YouTube watch/live-video URLs use `pytchat` first, with ChatDownloader as a
fallback if Pytchat cannot resolve the video's channel ID; Twitch and Kick use
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
uv sync --extra music-separation --extra youtube-publishing
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
uv run --extra music-separation --extra youtube-publishing --env-file .env python .\live_clipper.py
```

The variable name must be exactly `GROQ_API_KEY`. Do not share or commit your
key. `uv run` does not load `.env` automatically unless you pass `--env-file`.
Metadata uses Groq's OpenAI-compatible API with the `openai/gpt-oss-20b` model
by default. Set `GROQ_MODEL` in `.env` to select another Groq-supported model.

The app opens the StreamClipper dashboard at `http://127.0.0.1:55842/`.
Enter a Twitch, YouTube, or Kick URL in the form and choose the chat threshold,
clip length, pre-roll, and clip limit before starting. You can set `STREAM_URL`
near the top of `live_clipper.py` to prefill the URL field.

The app writes final MP4s, SRT files, and editable review metadata JSON files to
`output\`. Generated title, description, and hashtags are added when speech is
transcribed. Metadata is written in English and focused on GTA 6 gameplay, even
when the transcript is Hindi or Hinglish. Silent clips have blank metadata
fields for you to fill in during review and are rendered without subtitles.
Temporary clips and the rolling stream-segment buffer are written under `temp\`.
The segment buffer is pruned as it ages. Initial buffering takes approximately
the configured clip duration, and clips can only be generated after enough live
segments have arrived.

## Hosted dashboard

The dashboard and clipper run together on one always-on server. Netlify and
Vercel are not suitable for this worker: it needs to keep monitoring a stream
and running FFmpeg for longer than a serverless request. **Render is one
click-through option** that deploys the included Dockerfile:

1. Push this repository to GitHub.
2. In Render, choose **New → Web Service**, connect GitHub, and select this
   repository. Render detects the root `Dockerfile`.
3. Choose a **paid** service plan. Render's free service can sleep and has no
   persistent disk, so monitoring would stop and files would be lost.
4. Add a persistent disk mounted at `/app/data`.
5. Add environment variables `STREAMCLIPPER_USERNAME`,
   `STREAMCLIPPER_PASSWORD` (at least 16 characters), and optionally
   `GROQ_API_KEY` for automatic metadata. The Dockerfile already sets the
   required host and persistent-data paths. Set Render's health check path to
   `/healthz` if configuring a health check.
6. Deploy. Open the `https://…onrender.com` address Render gives you and share
   that URL plus the dashboard username/password with your friend.

The persistent disk stores clips, temporary stream segments, and model files.
Keep the login credentials private; anyone with them can start/stop clipping
and manage the clips.

For a VPS instead, copy `.env.example` to `.env`, set a unique password, and run
`docker compose up --build -d`. The Compose file binds to `127.0.0.1:8000`;
place it behind an HTTPS-enabled reverse proxy before sharing it outside the
server. On another container host, use the included `Dockerfile` and attach a
persistent disk at `/app/data`.

Clips, temporary stream segments, and model/token files need persistent disk
storage. The first run may download Whisper and audio-separation models. The
current YouTube OAuth flow is designed for a desktop browser; hosted YouTube
publishing requires a web-based OAuth callback setup in addition to the
dashboard.

## Review and publishing

Use **Start observing and clipping** to begin and **Stop clipping** to end
monitoring. A stop request lets an already-processing clip finish; the dashboard
then returns to its start form. Monitoring also stops after the configured clip
limit. Completed clips appear below the controls with editable title,
description, and hashtags. Nothing is uploaded until you click an upload
action. YouTube uploads default to **private**; choose unlisted or public
explicitly in the review page if desired.

To enable YouTube uploads, create a Google Cloud project, enable the YouTube
Data API v3, and create an OAuth client ID for a Desktop app. Download its JSON
file as `youtube_client_secret.json` in the project root. That file is ignored
by Git. The first desktop YouTube upload opens Google's consent flow; the
resulting token is stored under `%LOCALAPPDATA%\live-stream-clipper\`.

Instagram direct publishing is not enabled. The review page's Instagram action
saves the details, copies the caption, and opens Instagram for manual upload.
Direct Instagram publishing requires a Professional account linked to a
Facebook Page and publicly reachable video hosting.

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

- `STREAM_URL`: prefill the dashboard's stream URL field
- `CHAT_WINDOW_SECONDS`: burst window used to measure recent chat speed (5 seconds by default)
- `CHAT_BASELINE_SECONDS`: recent history used as the normal chat-rate baseline (300 seconds)
- `CHAT_SPIKE_MULTIPLIER`: how many times faster chat must be than its baseline (1.5 by default)
- `CHAT_SPIKE_MIN_MESSAGES`: minimum messages in the burst window, even when baseline chat is quiet (2 by default)
- `CLIP_DURATION`: total clip length (46 seconds by default)
- `CLIP_PRE_ROLL_SECONDS`: how many seconds before the detected spike to include (10 by default)
- `COOLDOWN_PERIOD`: minimum seconds between spike triggers (10 seconds by default)
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
