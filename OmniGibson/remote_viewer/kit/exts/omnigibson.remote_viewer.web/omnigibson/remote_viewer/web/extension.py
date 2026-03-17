from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import carb
import omni.ext
from omni.services.core import main, routers
from fastapi import Request, Response, APIRouter
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles


# Session endpoint setup
_SESSION_ENDPOINT = "/v1/streaming/session"
_session_router = routers.ServiceAPIRouter()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _split_host(value: str) -> str:
    if ":" in value and not value.startswith("["):
        return value.split(":", 1)[0]
    return value


@_session_router.post(_SESSION_ENDPOINT, summary="Create remote viewer session")
async def create_session_endpoint(request: Request) -> Dict[str, Any]:
    stream_host = os.getenv("PUBLIC_IP", "").strip()

    xf_host = request.headers.get("x-forwarded-host")
    if not stream_host and xf_host:
        stream_host = _split_host(xf_host)
    if not stream_host and request.client:
        stream_host = request.client.host or "127.0.0.1"
    stream_host = stream_host.strip()

    # Determine streaming mode (webrtc or websocket)
    streaming_mode = os.getenv("OMNIGIBSON_REMOTE_STREAMING", "webrtc").lower()
    if streaming_mode not in ["webrtc", "websocket"]:
        streaming_mode = "webrtc"

    # Determine signaling and media ports based on streaming mode
    if streaming_mode == "websocket":
        # WebSocket mode: use HTTP server port (same port for everything)
        # Try env var first, then fall back to request host header
        signaling_port = _env_int("EXTERNAL_HTTP_PORT", 0)
        if signaling_port == 0:
            # Extract port from request's host header
            host_header = request.headers.get("host", "")
            if ":" in host_header:
                signaling_port = int(host_header.split(":")[-1])
            else:
                # No port in host header, use standard HTTP port
                signaling_port = 80 if request.url.scheme == "http" else 443
        media_port = signaling_port  # Same port for websocket mode
    else:
        # WebRTC uses separate signaling and media ports
        signaling_port = _env_int("EXTERNAL_STREAM_SIGNALING_PORT", 49100)
        media_port = _env_int("EXTERNAL_STREAM_MEDIA_PORT", 47998)

    # Determine backend URL for client reference
    xf_proto = request.headers.get("x-forwarded-proto", request.url.scheme or "http")
    xf_host_full = request.headers.get("x-forwarded-host")
    if xf_proto and xf_host_full:
        backend_url = f"{xf_proto}://{xf_host_full}".rstrip("/")
    else:
        host = request.headers.get("host", "")
        backend_url = f"{request.url.scheme or 'http'}://{host}".rstrip("/")

    response: Dict[str, Any] = {
        "streamSignalingHost": stream_host,
        "signalingPort": int(signaling_port),
        "mediaPort": int(media_port),
        "backendUrl": backend_url,
        "streamingMode": streaming_mode,
    }

    return response


class WebUIExtension(omni.ext.IExt):
    """Serves the remote viewer web UI static files and session API via Kit's HTTP server."""

    def on_startup(self, _ext_id: str) -> None:
        # Register session API endpoint
        main.register_router(_session_router)

        try:
            self._startup_impl(_ext_id)
        except Exception as e:
            carb.log_error(f"Web extension startup failed: {e}")
            import traceback

            carb.log_error(traceback.format_exc())

    def _startup_impl(self, _ext_id: str) -> None:
        # Get the extension path from Kit's extension manager
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        ext_path = manager.get_extension_path(_ext_id)

        if not ext_path:
            carb.log_error("Failed to get extension path from Kit")
            return

        # Extension is at: .../remote_viewer/kit/exts/omnigibson.remote_viewer.web
        # Bundled web UI is at: .../omnigibson.remote_viewer.web/web-ui
        web_dir = str(Path(ext_path) / "web-ui")

        if not Path(web_dir).is_dir():
            carb.log_error(
                f"Web UI directory not found: {web_dir}\n"
                f"Make sure web-ui/ exists in the extension directory.\n"
                f"Run setup: cd {web_dir} && ./setup.sh"
            )
            return

        index_file = Path(web_dir) / "index.html"
        if not index_file.exists():
            carb.log_error(f"index.html not found in {web_dir}")
            carb.log_error(f"Directory contents: {list(Path(web_dir).iterdir())}")
            return

        # Get the FastAPI app
        app = main.get_app()
        if not app:
            carb.log_error("Failed to get FastAPI app")
            return

        # Mount assets directory if it exists
        assets_path = Path(web_dir) / "assets"
        if assets_path.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_path)), name="web-assets")

        # Helper to serve index.html
        def read_index():
            with open(index_file, "r") as f:
                return Response(content=f.read(), media_type="text/html", headers={"Cache-Control": "no-store"})

        # Register web UI routes
        web_router = APIRouter(include_in_schema=False)

        @web_router.get("/")
        async def serve_root():
            return read_index()

        @web_router.get("/index.html")
        async def serve_index():
            return read_index()

        app.include_router(web_router)

        # Catch-all for static files with SPA fallback
        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa_catchall(full_path: str):
            # Preserve API routes
            if full_path.startswith("v1/") or full_path.startswith("api/"):
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="Not found")

            # Try to serve static file
            file_path = Path(web_dir) / full_path
            if file_path.is_file():
                return FileResponse(file_path)

            # SPA fallback
            return read_index()

        # Store references
        self._app = app
        self._web_dir = web_dir
        self._index_file = index_file

    def on_shutdown(self) -> None:
        # Deregister session API endpoint
        main.deregister_router(_session_router)
