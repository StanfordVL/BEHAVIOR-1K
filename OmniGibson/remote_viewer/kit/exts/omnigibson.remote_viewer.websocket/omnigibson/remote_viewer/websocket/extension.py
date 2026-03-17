"""
WebSocket-based TCP-only streaming for OmniGibson.
Captures viewport frames and streams them as MJPEG over WebSocket.
"""

from __future__ import annotations

import asyncio
import io
from typing import Set

import carb
import omni.ext
import omni.kit.app
import omni.ui as ui
from fastapi import WebSocket, WebSocketDisconnect
from omni.kit.widget.viewport.capture import ByteCapture
from omni.services.core import main
from PIL import Image


class WebSocketStreamingExtension(omni.ext.IExt):
    """Streams viewport frames over WebSocket (TCP-only)."""

    def on_startup(self, _ext_id: str) -> None:
        self._clients: Set[WebSocket] = set()
        self._streaming = False
        self._stream_task = None

        # Get FastAPI app
        app = main.get_app()
        if not app:
            carb.log_error("Failed to get FastAPI app for WebSocket streaming")
            return

        # Register WebSocket endpoint
        @app.websocket("/streaming/client/")
        async def websocket_stream(websocket: WebSocket):
            await websocket.accept()
            self._clients.add(websocket)
            carb.log_info(f"WebSocket client connected. Total clients: {len(self._clients)}")

            # Start streaming if not already running
            if not self._streaming:
                self._start_streaming()

            try:
                # Keep connection alive (view-only, no interaction)
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(websocket)
                carb.log_info(f"WebSocket client disconnected. Remaining: {len(self._clients)}")

                # Stop streaming if no clients
                if not self._clients:
                    self._stop_streaming()

        self._app = app
        carb.log_info("WebSocket streaming endpoint registered at /streaming/client/")

    def _start_streaming(self) -> None:
        """Start the frame capture and streaming loop."""
        if self._streaming:
            return

        self._streaming = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        carb.log_info("Started WebSocket streaming")

    def _stop_streaming(self) -> None:
        """Stop the streaming loop."""
        self._streaming = False
        if self._stream_task:
            self._stream_task.cancel()
            self._stream_task = None
        carb.log_info("Stopped WebSocket streaming")

    async def _stream_loop(self) -> None:
        """Capture viewport frames and send to all connected clients."""
        try:
            while self._streaming and self._clients:
                # Capture frame from viewport
                frame_data = await self._capture_frame()

                if frame_data and self._clients:
                    # Send to all clients
                    disconnected = set()
                    for client in self._clients:
                        try:
                            await client.send_bytes(frame_data)
                        except Exception as e:
                            carb.log_warn(f"Failed to send frame to client: {e}")
                            disconnected.add(client)

                    # Remove disconnected clients
                    self._clients -= disconnected

                # Target 30 FPS
                await asyncio.sleep(1 / 30)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            carb.log_error(f"Streaming loop error: {e}")
            import traceback

            carb.log_error(traceback.format_exc())

    async def _capture_frame(self) -> bytes | None:
        """Capture current viewport frame as JPEG."""
        try:
            # Get viewport API
            import omni.kit.viewport.utility as vp_util

            viewport_api = vp_util.get_viewport_from_window_name("Viewport")

            if not viewport_api:
                return None

            # Create a future to wait for capture completion
            future = asyncio.Future()
            jpeg_data = None

            class JPEGCapture(ByteCapture):
                def on_capture_completed(self, buffer, buffer_size, width, height, format):
                    nonlocal jpeg_data
                    try:
                        import ctypes
                        import numpy as np

                        # Extract pointer from PyCapsule
                        # The buffer is a PyCapsule wrapping a void* pointer
                        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

                        # Get the raw pointer address (pass None for name)
                        ptr_address = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)

                        # Cast to unsigned byte pointer
                        buffer_ptr = ctypes.cast(ptr_address, ctypes.POINTER(ctypes.c_ubyte))

                        # Create numpy array from buffer pointer
                        arr = np.ctypeslib.as_array(buffer_ptr, shape=(buffer_size,))

                        # Determine number of channels from format
                        if format == ui.TextureFormat.RGBA8_UNORM:
                            channels = 4
                        elif format == ui.TextureFormat.RGB8_UNORM:
                            channels = 3
                        else:
                            channels = 4  # Default to RGBA

                        # Reshape to image dimensions
                        arr = arr.reshape((height, width, channels))

                        # Convert RGBA to RGB if needed
                        if channels == 4:
                            arr = arr[:, :, :3]

                        # Encode as JPEG
                        img = Image.fromarray(arr, mode="RGB")
                        buffer_io = io.BytesIO()
                        img.save(buffer_io, format="JPEG", quality=85, optimize=True)
                        jpeg_data = buffer_io.getvalue()

                        future.set_result(True)
                    except Exception as e:
                        carb.log_error(f"Failed to encode JPEG: {e}")
                        import traceback

                        carb.log_error(traceback.format_exc())
                        future.set_exception(e)

            # Schedule capture
            viewport_api.schedule_capture(JPEGCapture())

            # Wait for completion with timeout
            try:
                await asyncio.wait_for(future, timeout=1.0)
                return jpeg_data
            except asyncio.TimeoutError:
                carb.log_warn("Frame capture timed out")
                return None

        except Exception as e:
            carb.log_warn(f"Frame capture failed: {e}")
            return None

    def on_shutdown(self) -> None:
        """Clean up WebSocket connections."""
        self._stop_streaming()

        # Close all client connections
        for client in list(self._clients):
            try:
                asyncio.create_task(client.close())
            except:
                pass

        self._clients.clear()
