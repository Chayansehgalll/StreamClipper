"""Local review and approval UI for generated stream clips."""

from __future__ import annotations

import base64
import hmac
import ipaddress
import json
import logging
import mimetypes
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


LOGGER = logging.getLogger("live-clipper.review")
_CLIP_FILE_PATTERN = re.compile(r"clip_[A-Za-z0-9_-]+_\d{8}_\d{6}\.mp4")
_CLIP_ID_PATTERN = re.compile(r"clip_[A-Za-z0-9_-]+_\d{8}_\d{6}")
_YOUTUBE_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]


class ClipReviewServer:
    """Serve the clipper dashboard and review actions."""

    def __init__(
        self,
        output_dir: Path,
        client_secrets_file: Path,
        token_file: Path,
        controller: Any | None = None,
        default_settings: dict[str, Any] | None = None,
        username: str = "",
        password: str = "",
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.client_secrets_file = client_secrets_file.resolve()
        self.token_file = token_file.resolve()
        self.controller = controller
        self.default_settings = default_settings or {}
        self.username = username
        self.password = password
        self.host = host
        if bool(username) != bool(password):
            raise ValueError("Both dashboard username and password must be configured.")
        if password and len(password) < 16:
            raise ValueError("Dashboard password must be at least 16 characters.")
        if not self._is_loopback_host(host) and not (username and password):
            raise ValueError("Dashboard authentication is required on a public interface.")
        self._token = secrets.token_urlsafe(32)
        self._closed = threading.Event()
        self._upload_lock = threading.Lock()
        owner = self

        class RequestHandler(BaseHTTPRequestHandler):
            server_version = "StreamClipperReview/1.0"

            def log_message(self, format_string: str, *args: Any) -> None:
                LOGGER.debug(format_string, *args)

            def _authorize(self) -> bool:
                if not owner.username:
                    return True
                expected = "Basic " + base64.b64encode(
                    f"{owner.username}:{owner.password}".encode("utf-8")
                ).decode("ascii")
                if hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                    return True
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="StreamClipper"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False

            def do_GET(self) -> None:
                if urlparse(self.path).path == "/healthz":
                    self._send_bytes(200, b"ok", "text/plain; charset=utf-8")
                    return
                if not self._authorize():
                    return
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_bytes(
                        200,
                        owner._review_page().encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                elif parsed.path == "/api/status":
                    self._send_json(200, owner._control_status())
                elif parsed.path == "/api/clips":
                    self._send_json(200, owner._list_clips())
                elif parsed.path.startswith("/media/"):
                    self._send_media(unquote(parsed.path.removeprefix("/media/")))
                else:
                    self._send_json(404, {"error": "Not found."})

            def do_POST(self) -> None:
                if not self._authorize():
                    return
                if not hmac.compare_digest(
                    self.headers.get("X-Review-Token", ""),
                    owner._token,
                ):
                    self._send_json(403, {"error": "Review session token is invalid."})
                    return

                try:
                    payload = self._read_json()
                    route = urlparse(self.path).path
                    if route == "/api/update":
                        result = owner._update_metadata(payload)
                        self._send_json(200, result)
                    elif route == "/api/start":
                        result = owner._start_monitoring(payload)
                        self._send_json(200, result)
                    elif route == "/api/stop":
                        result = owner._stop_monitoring()
                        self._send_json(200, result)
                    elif route == "/api/youtube-upload":
                        result = owner._upload_to_youtube(payload)
                        self._send_json(200, result)
                    elif route == "/api/delete":
                        result = owner._delete_clip(payload)
                        self._send_json(200, result)
                    elif route == "/api/instagram-prep":
                        result = owner._update_metadata(payload)
                        self._send_json(
                            200,
                            {
                                **result,
                                "message": (
                                    "Details saved. Upload the video manually on Instagram; "
                                    "the review page can copy the caption for you."
                                ),
                            },
                        )
                    else:
                        self._send_json(404, {"error": "Not found."})
                except (ValueError, RuntimeError) as exc:
                    self._send_json(400, {"error": str(exc)})
                except Exception:
                    LOGGER.exception("Review request failed.")
                    self._send_json(500, {"error": "The review action failed; see app logs."})

            def _read_json(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ValueError("Invalid request length.") from exc
                if length < 1 or length > 1_000_000:
                    raise ValueError("Request body must be between 1 byte and 1 MB.")
                try:
                    payload = json.loads(self.rfile.read(length))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError("Request body must contain valid JSON.") from exc
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                return payload

            def _send_json(self, status: int, payload: Any) -> None:
                self._send_bytes(
                    status,
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )

            def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_media(self, filename: str) -> None:
                if not _CLIP_FILE_PATTERN.fullmatch(filename):
                    self._send_json(404, {"error": "Clip not found."})
                    return
                path = owner.output_dir / filename
                if not path.is_file():
                    self._send_json(404, {"error": "Clip not found."})
                    return
                file_size = path.stat().st_size
                start = 0
                end = file_size - 1
                status = 200
                byte_range = self.headers.get("Range")
                if byte_range:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", byte_range)
                    if not match or (not match.group(1) and not match.group(2)):
                        self._send_bytes(416, b"", "text/plain")
                        return
                    if match.group(1):
                        start = int(match.group(1))
                        end = int(match.group(2)) if match.group(2) else end
                    else:
                        suffix_length = int(match.group(2))
                        start = max(0, file_size - suffix_length)
                    if start >= file_size or end < start:
                        self._send_bytes(416, b"", "text/plain")
                        return
                    end = min(end, file_size - 1)
                    status = 206
                content_length = end - start + 1
                content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(content_length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("X-Content-Type-Options", "nosniff")
                if status == 206:
                    self.send_header(
                        "Content-Range",
                        f"bytes {start}-{end}/{file_size}",
                    )
                self.end_headers()
                with path.open("rb") as media_file:
                    media_file.seek(start)
                    remaining = content_length
                    while remaining:
                        chunk = media_file.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

        self.server = ThreadingHTTPServer((host, port), RequestHandler)
        self.server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        if host.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("The clip review server has already been started.")
        self._thread = threading.Thread(
            target=self._serve,
            name="clip-review-server",
            daemon=True,
        )
        self._thread.start()

    def _serve(self) -> None:
        try:
            self.server.serve_forever()
        finally:
            self.server.server_close()
            self._closed.set()

    def wait_until_closed(self) -> None:
        self._closed.wait()

    def stop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.server.shutdown()
            self._thread.join(timeout=5)
        elif self._thread is None:
            self.server.server_close()

    def _review_page(self) -> str:
        return (
            _REVIEW_PAGE.replace("__REVIEW_TOKEN__", json.dumps(self._token))
            .replace("__DEFAULT_SETTINGS__", json.dumps(self.default_settings))
        )

    def _control_status(self) -> dict[str, Any]:
        if self.controller is None:
            return {"status": "idle", "stream_url": "", "error": ""}
        return self.controller.status()

    def _start_monitoring(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.controller is None:
            raise RuntimeError("The clipper controls are not configured.")
        return self.controller.start(payload)

    def _stop_monitoring(self) -> dict[str, Any]:
        if self.controller is None:
            raise RuntimeError("The clipper controls are not configured.")
        return self.controller.stop()

    def _clip_paths(self, clip_id: Any) -> tuple[Path, Path]:
        if not isinstance(clip_id, str) or not _CLIP_ID_PATTERN.fullmatch(clip_id):
            raise ValueError("Invalid clip ID.")
        video_path = self.output_dir / f"{clip_id}.mp4"
        metadata_path = self.output_dir / f"{clip_id}.json"
        if not video_path.is_file() or not metadata_path.is_file():
            raise ValueError("The clip is not ready for review.")
        return video_path, metadata_path

    def _list_clips(self) -> list[dict[str, Any]]:
        clips = []
        for video_path in sorted(self.output_dir.glob("clip_*.mp4")):
            if not _CLIP_FILE_PATTERN.fullmatch(video_path.name):
                continue
            metadata_path = video_path.with_suffix(".json")
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except (OSError, json.JSONDecodeError):
                LOGGER.exception("Could not read review metadata %s.", metadata_path)
                continue
            if not isinstance(metadata, dict):
                LOGGER.error("Review metadata is not an object: %s", metadata_path)
                continue
            upload_info = metadata.get("youtube_upload")
            clips.append(
                {
                    "id": video_path.stem,
                    "video_url": f"/media/{video_path.name}",
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "hashtags": metadata.get("hashtags", []),
                    "youtube_upload": upload_info if isinstance(upload_info, dict) else None,
                }
            )
        return clips

    @staticmethod
    def _read_metadata(metadata_path: Path) -> dict[str, Any]:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read review metadata: {metadata_path}") from exc
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Review metadata must be a JSON object: {metadata_path}")
        return metadata

    @staticmethod
    def _write_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)

    def _update_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        _, metadata_path = self._clip_paths(payload.get("clip_id"))
        title = payload.get("title")
        description = payload.get("description")
        hashtags = payload.get("hashtags")
        if not isinstance(title, str) or len(title.strip()) > 100:
            raise ValueError("Title must be a string of at most 100 characters.")
        if not isinstance(description, str) or len(description) > 5000:
            raise ValueError("Description must be a string of at most 5000 characters.")
        if isinstance(hashtags, str):
            hashtags = hashtags.split()
        if (
            not isinstance(hashtags, list)
            or len(hashtags) > 30
            or any(
                not isinstance(tag, str)
                or not tag.startswith("#")
                or len(tag) > 100
                for tag in hashtags
            )
        ):
            raise ValueError("Enter up to 30 hashtags, each beginning with #.")
        metadata = self._read_metadata(metadata_path)
        metadata.update(
            {
                "title": title.strip(),
                "description": description.strip(),
                "hashtags": hashtags,
            }
        )
        self._write_metadata(metadata_path, metadata)
        return {"ok": True, "clip_id": payload["clip_id"]}

    def _delete_clip(self, payload: dict[str, Any]) -> dict[str, Any]:
        clip_id = payload.get("clip_id")
        with self._upload_lock:
            video_path, metadata_path = self._clip_paths(clip_id)
            metadata = self._read_metadata(metadata_path)
            upload_info = metadata.get("youtube_upload")
            youtube_upload_retained = (
                isinstance(upload_info, dict) and bool(upload_info.get("video_id"))
            )
            for path in (
                video_path,
                video_path.with_suffix(".srt"),
                metadata_path,
            ):
                if path.is_file():
                    path.unlink()
        LOGGER.info(
            "Deleted local clip %s; YouTube upload retained=%s.",
            clip_id,
            youtube_upload_retained,
        )
        return {
            "ok": True,
            "clip_id": clip_id,
            "youtube_upload_retained": youtube_upload_retained,
        }

    def _upload_to_youtube(self, payload: dict[str, Any]) -> dict[str, Any]:
        clip_id = payload.get("clip_id")
        video_path, metadata_path = self._clip_paths(clip_id)
        visibility = payload.get("visibility", "private")
        if visibility not in {"private", "unlisted", "public"}:
            raise ValueError("Visibility must be private, unlisted, or public.")
        self._update_metadata(payload)
        with self._upload_lock:
            metadata = self._read_metadata(metadata_path)
            upload_info = metadata.get("youtube_upload")
            if isinstance(upload_info, dict) and upload_info.get("video_id"):
                raise ValueError("This clip has already been uploaded to YouTube.")
            if not metadata.get("title"):
                raise ValueError("Add a title before uploading.")
            video_id = self._upload_video(video_path, metadata, visibility)
            metadata["youtube_upload"] = {
                "video_id": video_id,
                "privacy_status": visibility,
            }
            self._write_metadata(metadata_path, metadata)
        return {
            "ok": True,
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

    def _upload_video(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        visibility: str,
    ) -> str:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:
            raise RuntimeError(
                "YouTube upload packages are missing. Run "
                "'uv sync --extra youtube-publishing'."
            ) from exc

        if not self.client_secrets_file.is_file():
            raise RuntimeError(
                "YouTube OAuth setup is missing. Add the downloaded desktop OAuth "
                f"client file at {self.client_secrets_file}."
            )

        credentials = None
        if self.token_file.is_file():
            credentials = Credentials.from_authorized_user_file(
                str(self.token_file),
                _YOUTUBE_SCOPE,
            )
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        if not credentials or not credentials.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.client_secrets_file),
                _YOUTUBE_SCOPE,
            )
            credentials = flow.run_local_server(port=0)

        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(credentials.to_json(), encoding="utf-8")
        youtube = build("youtube", "v3", credentials=credentials)
        description = metadata.get("description", "").strip()
        hashtags = metadata.get("hashtags", [])
        if hashtags:
            description = f"{description}\n\n{' '.join(hashtags)}".strip()
        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": metadata["title"],
                    "description": description,
                    "tags": [tag.lstrip("#") for tag in hashtags],
                },
                "status": {
                    "privacyStatus": visibility,
                    "selfDeclaredMadeForKids": False,
                },
            },
            media_body=MediaFileUpload(
                str(video_path),
                mimetype="video/mp4",
                resumable=True,
            ),
        )
        response = None
        while response is None:
            _, response = request.next_chunk()
        video_id = response.get("id")
        if not isinstance(video_id, str) or not video_id:
            raise RuntimeError("YouTube did not return an uploaded video ID.")
        return video_id


_REVIEW_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreamClipper review</title>
<style>
body{font:16px system-ui,sans-serif;background:#111827;color:#f3f4f6;margin:0}
header{padding:20px 5%;background:#1f2937;position:sticky;top:0;z-index:1}
h1{font-size:1.35rem;margin:0 0 6px}p{color:#cbd5e1}.toolbar{display:flex;gap:10px}
main{max-width:1100px;margin:24px auto;padding:0 16px}
.card{background:#1f2937;border:1px solid #374151;border-radius:12px;padding:18px;margin:18px 0}
.settings-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:4px 16px}
.hidden,[hidden]{display:none!important}.error{color:#fca5a5}.run-status{font-weight:700}
h2{display:flex;align-items:center;gap:10px}
.badge{display:inline-block;padding:3px 8px;border-radius:999px;font-size:.72rem;font-weight:700}
.uploaded-badge{background:#166534;color:#dcfce7}
.layout{display:grid;grid-template-columns:minmax(260px,1fr) minmax(280px,1.3fr);gap:20px}
video{width:100%;max-height:70vh;background:#000;border-radius:8px}
label{display:block;font-weight:600;margin:10px 0 4px}
input,textarea,select{box-sizing:border-box;width:100%;padding:10px;border-radius:6px;border:1px solid #4b5563;background:#111827;color:#f9fafb}
textarea{min-height:90px;resize:vertical}.buttons{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
button{border:0;border-radius:6px;padding:10px 14px;cursor:pointer;background:#374151;color:white;font-weight:600}
button.primary{background:#2563eb}button.instagram{background:#be185d}button.danger{background:#b91c1c}button:disabled{opacity:.55;cursor:wait}
.status{min-height:24px;color:#93c5fd}.uploaded{color:#86efac}.muted{color:#9ca3af}
@media(max-width:720px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>StreamClipper</h1>
  <p>Start monitoring a live stream, then review completed clips below. Uploads still require your approval.</p>
  <div class="toolbar"><span id="run-status" class="run-status">Idle</span><button id="refresh">Refresh clips</button></div>
</header>
<main>
  <section id="setup" class="card">
    <h2>Start observing and clipping</h2>
    <form id="control-form">
      <div class="settings-grid">
        <label>Stream URL<input id="stream_url" name="stream_url" type="text" required placeholder="https://www.youtube.com/watch?v=…"></label>
        <label>Chat spike multiplier<input id="chat_spike_multiplier" name="chat_spike_multiplier" type="number" min="0.1" max="20" step="0.1" required></label>
        <label>Clip duration (seconds)<input id="clip_duration" name="clip_duration" type="number" min="5" max="180" step="1" required></label>
        <label>Pre-roll (seconds)<input id="clip_pre_roll_seconds" name="clip_pre_roll_seconds" type="number" min="0" max="179" step="1" required></label>
        <label>Maximum clips<input id="max_clips" name="max_clips" type="number" min="1" max="100" step="1" required></label>
        <label>Minimum chat messages per 5s<input id="min_chat_messages" name="min_chat_messages" type="number" min="1" max="100" step="1" required></label>
        <label>Cooldown between clips (seconds)<input id="cooldown_seconds" name="cooldown_seconds" type="number" min="0" max="3600" step="1" required></label>
      </div>
      <div class="buttons"><button class="primary" type="submit">Start observing and clipping</button></div>
      <p id="control-message" class="status" role="status"></p>
    </form>
  </section>
  <section id="active" class="card" hidden>
    <h2>Monitoring stream</h2>
    <p id="active-stream"></p>
    <p id="active-message" class="status">Starting…</p>
    <button id="stop" class="danger" type="button">Stop clipping</button>
  </section>
  <h2>Review completed clips</h2>
  <div id="clips"><p>Looking for completed clips…</p></div>
</main>
<script>
const reviewToken=__REVIEW_TOKEN__;
const defaults=__DEFAULT_SETTINGS__;
const clipsRoot=document.querySelector("#clips");
const setup=document.querySelector("#setup");
const active=document.querySelector("#active");
const controlMessage=document.querySelector("#control-message");
const statusNames={idle:"Idle",starting:"Starting…",observing:"Observing",stopping:"Stopping…",stopped:"Stopped",completed:"Finished",error:"Error"};
for(const key of ["stream_url","chat_spike_multiplier","clip_duration","clip_pre_roll_seconds","max_clips","min_chat_messages","cooldown_seconds"]){
  const input=document.querySelector("#"+key);input.value=defaults[key]??"";
}
async function post(path,payload){
  const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":reviewToken},body:JSON.stringify(payload)});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||"Request failed");
  return data;
}
function makeLabel(text,control){
  const label=document.createElement("label");label.textContent=text;label.append(control);return label;
}
function field(tag,value,maxLength){
  const element=document.createElement(tag);element.value=value||"";
  if(maxLength)element.maxLength=maxLength;
  return element;
}
function button(text,className,action){
  const item=document.createElement("button");item.textContent=text;
  if(className)item.className=className;
  item.addEventListener("click",action);return item;
}
function formPayload(){
  return{
    stream_url:document.querySelector("#stream_url").value,
    chat_spike_multiplier:Number(document.querySelector("#chat_spike_multiplier").value),
    clip_duration:Number(document.querySelector("#clip_duration").value),
    clip_pre_roll_seconds:Number(document.querySelector("#clip_pre_roll_seconds").value),
    max_clips:Number(document.querySelector("#max_clips").value),
    min_chat_messages:Number(document.querySelector("#min_chat_messages").value),
    cooldown_seconds:Number(document.querySelector("#cooldown_seconds").value)
  };
}
async function updateStatus(){
  try{
    const response=await fetch("/api/status",{cache:"no-store"});
    if(!response.ok)throw new Error("Could not read clipper status.");
    const state=await response.json();
    const name=statusNames[state.status]||state.status;
    document.querySelector("#run-status").textContent=name;
    if(["starting","observing","stopping"].includes(state.status)){
      setup.hidden=true;active.hidden=false;
      document.querySelector("#active-stream").textContent=state.stream_url||"";
      document.querySelector("#active-message").textContent=state.status==="error"?state.error:name;
      document.querySelector("#stop").disabled=state.status==="stopping";
    }else{
      active.hidden=true;setup.hidden=false;
      if(state.status==="stopped")controlMessage.textContent="Monitoring stopped. Ready to start another stream.";
      else if(state.status==="completed")controlMessage.textContent="Run finished. Ready to start another stream.";
      else if(state.status==="error"){controlMessage.textContent=state.error||"The clipper stopped with an error.";controlMessage.classList.add("error");}
      else if(state.status==="idle")controlMessage.textContent="";
    }
  }catch(error){document.querySelector("#run-status").textContent=error.message;}
}
document.querySelector("#control-form").addEventListener("submit",async event=>{
  event.preventDefault();controlMessage.textContent="Starting…";controlMessage.classList.remove("error");
  try{await post("/api/start",formPayload());await updateStatus();}
  catch(error){controlMessage.textContent=error.message;controlMessage.classList.add("error");}
});
document.querySelector("#stop").addEventListener("click",async()=>{
  if(!confirm("Stop observing this stream? Any clip already being processed will finish."))return;
  document.querySelector("#stop").disabled=true;
  document.querySelector("#active-message").textContent="Stopping…";
  try{await post("/api/stop",{});await updateStatus();}
  catch(error){document.querySelector("#active-message").textContent=error.message;document.querySelector("#stop").disabled=false;}
});
async function loadClips(){
  try{
    const response=await fetch("/api/clips",{cache:"no-store"});
    if(!response.ok)throw new Error("Could not load clips");
    const clips=await response.json();clipsRoot.replaceChildren();
    if(!clips.length){const empty=document.createElement("p");empty.textContent="No completed clips yet. New clips will appear here.";clipsRoot.append(empty);return;}
    for(const clip of clips){
      const card=document.createElement("section");card.className="card";
      const heading=document.createElement("h2");heading.textContent=clip.id;
      if(clip.youtube_upload?.video_id){
        const badge=document.createElement("span");badge.className="badge uploaded-badge";badge.textContent="Uploaded";heading.append(badge);
      }
      card.append(heading);
      const layout=document.createElement("div");layout.className="layout";
      const video=document.createElement("video");video.controls=true;video.preload="metadata";video.src=clip.video_url;layout.append(video);
      const form=document.createElement("div");
      const title=field("input",clip.title,100);
      const description=field("textarea",clip.description,5000);
      const hashtags=field("input",(clip.hashtags||[]).join(" "),1000);hashtags.placeholder="#tag1 #tag2";
      const visibility=document.createElement("select");
      for(const value of ["private","unlisted","public"]){const option=document.createElement("option");option.value=value;option.textContent=value;visibility.append(option);}
      visibility.value=clip.youtube_upload?.privacy_status||"private";
      form.append(makeLabel("Title",title),makeLabel("Description",description),makeLabel("Hashtags",hashtags),makeLabel("YouTube visibility",visibility));
      const status=document.createElement("p");status.className="status";
      if(clip.youtube_upload?.video_id){status.classList.add("uploaded");status.textContent="Uploaded: https://www.youtube.com/watch?v="+clip.youtube_upload.video_id;}
      const actions=document.createElement("div");actions.className="buttons";
      const payload=()=>({clip_id:clip.id,title:title.value,description:description.value,hashtags:hashtags.value.split(/\s+/).filter(Boolean)});
      const saveButton=button("Save details","",async()=>{try{await post("/api/update",payload());status.textContent="Details saved."; }catch(error){status.textContent=error.message;}});
      actions.append(saveButton);
      const uploadButton=button(clip.youtube_upload?.video_id?"Already uploaded":"Upload to YouTube","primary",async()=>{
        if(clip.youtube_upload?.video_id)return;
        if(!confirm("Upload this clip to YouTube as "+visibility.value+"?"))return;
        uploadButton.disabled=true;status.textContent="Uploading…";
        try{
          const result=await post("/api/youtube-upload",{...payload(),visibility:visibility.value});
          status.classList.add("uploaded");status.textContent="Uploaded: "+result.url;uploadButton.textContent="Already uploaded";
          if(!heading.querySelector(".uploaded-badge")){
            const badge=document.createElement("span");badge.className="badge uploaded-badge";badge.textContent="Uploaded";heading.append(badge);
          }
        }
        catch(error){status.textContent=error.message;uploadButton.disabled=false;}
      });
      uploadButton.disabled=Boolean(clip.youtube_upload?.video_id);
      const deleteButton=button("Delete clip","danger",async()=>{
        if(!confirm("Delete this clip's local video, subtitle, and metadata files? Any YouTube upload will remain online."))return;
        deleteButton.disabled=true;status.textContent="Deleting local files…";
        try{
          const result=await post("/api/delete",{clip_id:clip.id});
          const notice=document.createElement("p");notice.className="status";
          notice.textContent=result.youtube_upload_retained
            ?"Local clip deleted. Its YouTube upload remains online."
            :"Local clip deleted.";
          card.replaceWith(notice);
        }catch(error){status.textContent=error.message;deleteButton.disabled=false;}
      });
      actions.append(uploadButton,button("Prepare Instagram manual upload","instagram",async()=>{
        const caption=[description.value,hashtags.value].filter(Boolean).join("\n\n");
        window.open("https://www.instagram.com/","_blank","noopener");
        try{await navigator.clipboard.writeText(caption);status.textContent="Caption copied. Choose this video in Instagram and paste the caption.";}
        catch(error){status.textContent="Copy the caption above, then upload the video manually on Instagram.";}
        try{await post("/api/instagram-prep",payload());}
        catch(error){status.textContent=error.message;}
      }),deleteButton);
      form.append(actions,status);layout.append(form);card.append(layout);clipsRoot.append(card);
    }
  }catch(error){clipsRoot.textContent=error.message;}
}
document.querySelector("#refresh").addEventListener("click",loadClips);
loadClips();setInterval(()=>{if(!clipsRoot.contains(document.activeElement))loadClips();},8000);
updateStatus();setInterval(updateStatus,1500);
</script>
</body>
</html>
"""
