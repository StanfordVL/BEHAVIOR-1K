"""
Adapted from https://github.com/Physical-Intelligence/openpi
"""

import asyncio
import functools
import http
import logging
import msgpack
import numpy as np
import requests
import time
import torch as th
import traceback
import websockets
import websockets.asyncio.server as _server
import websockets.sync.client
from copy import deepcopy
from omnigibson.macros import gm
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["WebsocketClientPolicy", "WebsocketPolicyServer"]


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        scheme: str = "ws",
        api_key: Optional[str] = None,
        allow_reconnect: bool = False,
    ) -> None:
        """
        Initializes the websocket client policy.

        Args:
            host (str): Hostname of the websocket server to connect to.
            port (int): Port of the websocket server to connect to. Defaults to 8000.
            scheme (str): WebSocket scheme to use. Either "wss" (secure) or "ws" (insecure). Defaults to "ws".
            api_key (str, optional): API key to include in the Authorization header when connecting to the websocket server, if required.
            allow_reconnect (bool): Whether to allow automatic reconnection if the websocket connection is lost.
                If False, the client will raise an error if the connection is lost.
                If True, the client will attempt to reconnect indefinitely with a delay between attempts.
        """
        self._uri = f"{scheme}://{host}:{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = None, None
        self._allow_reconnect = allow_reconnect

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        parsed = urlparse(self._uri)
        host = parsed.hostname
        port = parsed.port
        http_scheme = "https" if parsed.scheme == "wss" else "http"
        health_url = f"{http_scheme}://{host}:{port}/healthz"

        # First, wait for the health check to pass
        while True:
            try:
                response = requests.get(health_url, timeout=2)
                if response.ok:
                    logger.info("Health check passed, attempting websocket connection...")
                    break
            except Exception:
                pass
            logger.info(f"Health check failed, waiting for server at {http_scheme}://{host}:{port}...")
            time.sleep(5)

        # Now attempt websocket connection (rest of the code remains the same)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=60,
                    ping_timeout=300,
                )
                metadata = unpackb(conn.recv())
                logger.info("Connected to server!")
                return conn, metadata
            except (ConnectionRefusedError, websockets.exceptions.InvalidMessage, EOFError) as e:
                logger.info(f"Websocket connection failed ({e}), retrying...")
                time.sleep(5)

    def act(self, obs: Dict) -> th.Tensor:
        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()

        data = self._packer.pack(obs)
        max_retries = 2
        response = None

        for attempt in range(max_retries + 1):
            try:
                self._ws.send(data)
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")

                action_dict = unpackb(response)
                if "action" not in action_dict:
                    if attempt < max_retries:
                        logger.warning(
                            f"Server response missing 'action' key, retrying ({attempt + 1}/{max_retries})..."
                        )
                        continue
                    raise RuntimeError(f"Server response missing 'action' key: {action_dict}")
                action = th.from_numpy(deepcopy(action_dict["action"])).to(th.float32)
                return action

            except websockets.exceptions.ConnectionClosedError as e:
                if self._allow_reconnect and attempt < max_retries:
                    logger.warning(f"Connection lost, reconnecting (attempt {attempt + 1}/{max_retries})...")
                    self._ws, self._server_metadata = self._wait_for_server()
                    continue
                raise RuntimeError(f"Websocket connection error: {e}")

    def reset(self) -> None:
        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()

        data = self._packer.pack({"reset": True})
        self._ws.send(data)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        logger.info(f"Starting websocket server on {self._host}:{self._port}...")
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    self._policy.reset()
                    continue

                obs = deepcopy(result)

                infer_time = time.monotonic()
                action = self._policy.act(obs)
                infer_time = time.monotonic() - infer_time

                action = {
                    "action": action.cpu().numpy(),
                }
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.error(f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}")
                if gm.DEBUG:
                    await websocket.send(traceback.format_exc())
                try:
                    # Try new websockets API first
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except AttributeError:
                    # Fallback for older websockets versions
                    await websocket.close(code=1011, reason="Internal server error")
                raise


def _health_check(connection, request) -> Optional[Any]:
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        else:
            # For older websockets versions, return a simple response
            return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    # Continue with the normal request handling.
    return None


"""
Adds NumPy array and PyTorch tensor support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

This module supports serializing both NumPy arrays and PyTorch tensors. PyTorch tensors are converted to
NumPy arrays (zero-copy when possible) before serialization. On deserialization, arrays are returned as NumPy arrays.

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""


def pack_data(obj):
    if isinstance(obj, th.Tensor):
        data = obj.detach().cpu().numpy()
        return {
            b"__ndarray__": True,
            b"data": data.tobytes(),
            b"dtype": data.dtype.str,
            b"shape": data.shape,
        }

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
packb = functools.partial(msgpack.packb, default=pack_data)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)
