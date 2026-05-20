"""Minimal VR input WebSocket server.

Serves the WebXR HTML page on `GET /` and accepts pose JSON messages on
`/ws`. Stores the latest wrist + hand keypoint poses in thread-safe state
that the main teleop loop can poll at any rate.

Message format (sent by `web/index.html`):

    {
      "type": "hands",
      "left":  {"wrist": [x,y,z, qw,qx,qy,qz],
                "joints": {"thumb-tip": [x,y,z, qw,qx,qy,qz], ...}}
                | null,
      "right": {...} | null
    }

Pose vectors are 7-element lists: position + quaternion (wxyz, normalized).
The WebXR session uses the standard Y-up frame; this server does *not* apply
any reframe — coordinate convention conversion is the IK/retargeting layer's
responsibility.

We avoid any heavy framework: just `aiohttp` (already needed for the
shadow-hand frontend / WebXR streaming) and `asyncio`. The server runs in a
background thread so the main loop stays synchronous.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

from aiohttp import web, WSMsgType


class VRInputServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8012,
        web_dir: Path | str | None = None,
        cert_file: Path | str | None = None,
        key_file: Path | str | None = None,
    ):
        self.host = host
        self.port = int(port)
        self.web_dir = Path(web_dir) if web_dir is not None else (
            Path(__file__).resolve().parent / "web"
        )
        # Optional TLS — Meta Browser requires HTTPS for WebXR sessions on
        # non-localhost origins. If both cert + key paths exist, the server
        # binds with TLS and the client URL becomes `https://...`.
        self.cert_file = Path(cert_file) if cert_file else None
        self.key_file = Path(key_file) if key_file else None

        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {
            "timestamp": 0.0,
            "left": None,
            "right": None,
        }

        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ---------------- public API ----------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._serve_thread, name="vr-input-server", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._loop is None or self._runner is None:
            return
        self._stop_event.set()
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            future.result(timeout=3.0)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self) -> dict[str, Any]:
        """Snapshot of the most recent VR pose payload."""
        with self._lock:
            return dict(self._latest)

    # ---------------- aiohttp internals ----------------

    async def _shutdown(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    def _serve_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    async def _serve(self) -> None:
        import ssl
        app = web.Application(client_max_size=1 << 20)
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)
        if self.web_dir.exists():
            app.router.add_static("/static/", path=str(self.web_dir), show_index=False)
        runner = web.AppRunner(app)
        await runner.setup()
        ssl_ctx = None
        if self.cert_file and self.key_file and self.cert_file.exists() and self.key_file.exists():
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(certfile=str(self.cert_file), keyfile=str(self.key_file))
        site = web.TCPSite(runner, self.host, self.port, ssl_context=ssl_ctx)
        await site.start()
        self._runner = runner

        # Sleep loop — the server is driven by aiohttp; we only need to keep the
        # event loop alive until stop() is called.
        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)

    async def _handle_index(self, request: web.Request) -> web.Response:
        index_path = self.web_dir / "index.html"
        if not index_path.exists():
            return web.Response(
                text="VR input server is up. Place a WebXR client at "
                f"{self.web_dir}/index.html or POST poses to /ws.",
                content_type="text/plain",
            )
        return web.FileResponse(index_path)

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=1 << 20)
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") != "hands":
                    continue
                self._store(payload)
            elif msg.type == WSMsgType.ERROR:
                break
        return ws

    def _store(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._latest = {
                "timestamp": time.time(),
                "left": payload.get("left"),
                "right": payload.get("right"),
            }


def hand_payload_to_keypoints(
    side_payload: dict | None,
) -> tuple[dict | None, dict | None]:
    """Convert one side of the WebXR payload to the format `HandSolver.step` expects.

    Returns (wrist_pose, keypoints_dict). `wrist_pose` is
    `{"position": [x,y,z], "quaternion_wxyz": [w,x,y,z]}` or None.
    `keypoints_dict` includes the wrist plus all joints.
    """
    if not side_payload:
        return None, None

    wrist_arr = side_payload.get("wrist")
    joints = side_payload.get("joints") or {}

    keypoints: dict[str, dict] = {}
    if wrist_arr is not None and len(wrist_arr) == 7:
        keypoints["wrist"] = {
            "position": list(wrist_arr[:3]),
            "quaternion_wxyz": list(wrist_arr[3:7]),
        }
    for jname, vec in joints.items():
        if vec is None or len(vec) != 7:
            continue
        keypoints[jname] = {
            "position": list(vec[:3]),
            "quaternion_wxyz": list(vec[3:7]),
        }

    return keypoints.get("wrist"), keypoints
