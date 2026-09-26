"""Blocking MCP HTTP server with a persistent Python/pymxs namespace.

Execute this script from MAXScript to listen at http://127.0.0.1:8765/mcp.
HTTP requests and tool calls run sequentially on the script's main thread.
Keep this script and submitted code Python 3.7-compatible.
"""

import ast
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import tokenize
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 - used in Python 3.7 type comments

import pymxs

rt = pymxs.runtime
_GLOBALS = {"__name__": "__3dsmax_mcp_repl__", "pymxs": pymxs, "rt": rt}


RUN_PYTHON_TOOL = "run_python"
EXECUTE_SCRIPT_TOOL = "execute_script"


def _safe_repr(value):
    try:
        return repr(value)
    except BaseException:
        return "<unrepresentable {}>".format(type(value).__name__)


class _OutputBuffer(io.StringIO):
    """Bound captured output so accidental print loops do not exhaust Max's RAM."""

    def __init__(self):
        super().__init__()
        self.remaining = 1024 * 1024
        self.truncated = False

    def write(self, value):
        length = len(value)
        overflow = length > self.remaining
        super().write(value[:self.remaining])
        self.remaining = max(0, self.remaining - length)
        if overflow and not self.truncated:
            super().write("\n<output truncated at 1 MiB of characters>\n")
            self.truncated = True
        return length


def _jsonable(value, depth=0, seen=None):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= 10:
        return "<maximum nesting depth>"
    if isinstance(value, (list, tuple, dict)):
        if seen is None:
            seen = set()
        if id(value) in seen:
            return "<circular reference>"
        seen.add(id(value))
        try:
            if isinstance(value, dict):
                return {str(k): _jsonable(v, depth + 1, seen) for k, v in value.items()}
            return [_jsonable(x, depth + 1, seen) for x in value]
        finally:
            seen.remove(id(value))
    return _safe_repr(value)


def _point3_tuple(value):  # type: (Any) -> Optional[Tuple[float, float, float]]
    try:
        return (float(value.x), float(value.y), float(value.z))
    except Exception:
        return None


def _safe_class_name(value):  # type: (Any) -> Optional[str]
    if value is None:
        return None
    try:
        return str(rt.classOf(value))
    except Exception:
        return type(value).__name__


def _material_summary(mat, depth=0):  # type: (Any, int) -> Optional[Dict[str, Any]]
    if mat is None:
        return None
    summary = {
        "name": str(getattr(mat, "name", "")),
        "class": _safe_class_name(mat),
    }
    if depth >= 2:
        return summary

    for attr in ("diffuse", "diffuseColor", "reflection", "refraction", "texmap_diffuse", "fileName"):
        if rt.isProperty(mat, attr):
            try:
                summary[attr] = _jsonable(getattr(mat, attr))
            except Exception as exc:
                summary[attr] = "<error: {}>".format(exc)

    if rt.isProperty(mat, "materialList"):
        try:
            summary["submaterials"] = [
                _material_summary(sub_mat, depth + 1) for sub_mat in list(mat.materialList) if sub_mat is not None
            ]
        except Exception as exc:
            summary["submaterials_error"] = str(exc)
    return summary


def _node_summary(obj, include_material=True):  # type: (Any, bool) -> Dict[str, Any]
    summary = {
        "name": str(obj.name),
        "class": _safe_class_name(obj),
        "isHidden": bool(getattr(obj, "isHidden", False)),
        "isSelected": bool(getattr(obj, "isSelected", False)),
        "position": _point3_tuple(getattr(obj, "pos", None)),
        "parent": str(obj.parent.name) if getattr(obj, "parent", None) else None,
        "material": None,
    }
    try:
        summary["wireColor"] = _jsonable(getattr(obj, "wireColor", None))
    except Exception:
        pass
    if include_material:
        summary["material"] = _material_summary(getattr(obj, "material", None))
    try:
        if rt.classOf(obj) == rt.Editable_Poly:
            summary["poly"] = {
                "verts": int(rt.polyop.GetNumVerts(obj)),
                "faces": int(rt.polyop.GetNumFaces(obj)),
                "dead_structs": int(rt.polyop.GetHasDeadStructs(obj)),
            }
    except Exception as exc:
        summary["poly_error"] = str(exc)
    return summary


def get_scene_state(max_nodes=100):  # type: (int) -> Dict[str, Any]
    """Return compact read-only state for the currently open 3ds Max scene."""
    objects = list(rt.objects)
    selected = list(rt.selection)
    cameras = list(rt.cameras)
    lights = list(rt.lights)
    materials = list(rt.scenematerials)
    state = {
        "file": {"path": str(rt.maxFilePath), "name": str(rt.maxFileName)},
        "counts": {
            "objects": len(objects),
            "selected": len(selected),
            "cameras": len(cameras),
            "lights": len(lights),
            "scene_materials": len(materials),
        },
        "selection": [str(obj.name) for obj in selected[:max_nodes]],
        "objects": [_node_summary(obj, include_material=False) for obj in objects[:max_nodes]],
        "cameras": [str(camera.name) for camera in cameras[:max_nodes]],
        "lights": [str(light.name) for light in lights[:max_nodes]],
        "materials": [_material_summary(mat) for mat in materials[:max_nodes]],
        "render": {},
    }
    for attr in ("rendWidth", "rendHeight", "rendTimeType", "rendOutputFilename"):
        try:
            state["render"][attr] = _jsonable(getattr(rt, attr))
        except Exception:
            pass
    try:
        state["render"]["renderer"] = _safe_class_name(rt.renderers.current)
    except Exception:
        pass
    return state


def capture_viewport(path=None):  # type: (Optional[str]) -> str
    """Save the active viewport image to a file and return the path.

    This uses the MAXScript graphics-window API (`gw.getViewportDib`) through pymxs,
    so it captures whatever viewport/layout is currently active.
    """
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "3dsmax_mcp_{}.png".format(uuid.uuid4().hex))
    path = os.path.abspath(path)
    bitmap = rt.gw.getViewportDib()
    try:
        bitmap.filename = path
        if not rt.save(bitmap, quiet=True):
            raise IOError("Could not save viewport to {}".format(path))
    finally:
        rt.close(bitmap)
    return path


def _execute_code(code, filename="<3dsmax-mcp>"):  # type: (str, str) -> Tuple[Any, str, str, Optional[str]]
    stdout = _OutputBuffer()
    stderr = _OutputBuffer()
    result = None
    error = None
    result_key = "_mcp_result_" + uuid.uuid4().hex
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            tree = ast.parse(code, filename=filename, mode="exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                expression = tree.body[-1]
                tree.body[-1] = ast.copy_location(ast.Assign(
                    targets=[ast.Name(id=result_key, ctx=ast.Store())], value=expression.value
                ), expression)
                ast.fix_missing_locations(tree)
            exec(compile(tree, filename, "exec", dont_inherit=True), _GLOBALS, _GLOBALS)
            result = _GLOBALS.get(result_key)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                error = traceback.format_exc()
        except BaseException:
            error = traceback.format_exc()
        finally:
            _GLOBALS.pop(result_key, None)
    return result, stdout.getvalue(), stderr.getvalue(), error


def _base_response(ok, stdout, stderr, error):  # type: (bool, str, str, Optional[str]) -> Dict[str, Any]
    response = {
        "ok": ok,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
    }
    try:
        response["scene_state"] = get_scene_state()
    except Exception:
        response["scene_state"] = None
        response["scene_state_error"] = traceback.format_exc()
    return response


def _maybe_capture_view(response, capture_view=False, viewport_path=None):  # type: (Dict[str, Any], bool, Optional[str]) -> Dict[str, Any]
    if capture_view:
        try:
            response["viewport_image"] = capture_viewport(viewport_path)
        except Exception:
            response["viewport_image_error"] = traceback.format_exc()
    return response


def run_python(code, capture_view=False, viewport_path=None):  # type: (str, bool, Optional[str]) -> Dict[str, Any]
    result, stdout, stderr, error = _execute_code(code)
    response = _base_response(error is None, stdout, stderr, error)
    response["result_repr"] = _safe_repr(result)
    try:
        response["result_json"] = _jsonable(result)
    except Exception:
        response["result_json"] = None
        response["result_serialization_error"] = traceback.format_exc()
    return _maybe_capture_view(response, capture_view, viewport_path)


def execute_script(script_path, script_args=None, capture_view=False, viewport_path=None):
    # type: (str, Optional[List[str]], bool, Optional[str]) -> Dict[str, Any]
    """Execute a Python script from disk in the stateful 3ds Max namespace."""
    if not isinstance(script_path, str) or not os.path.isabs(script_path):
        raise ValueError("script_path must be an absolute path")
    script_path = os.path.abspath(script_path)
    stdout = _OutputBuffer()
    stderr = _OutputBuffer()
    error = None
    old_argv = sys.argv
    old_path = sys.path
    old_specials = {key: _GLOBALS[key] for key in ("__file__", "__name__", "__package__", "__spec__") if key in _GLOBALS}
    try:
        with tokenize.open(script_path) as script_file:
            code = script_file.read()
        sys.argv = [script_path] + list(script_args or [])
        sys.path = [os.path.dirname(script_path)] + old_path
        _GLOBALS.update(__file__=script_path, __name__="__main__", __package__=None, __spec__=None)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(code, script_path, "exec", dont_inherit=True), _GLOBALS, _GLOBALS)
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    error = traceback.format_exc()
            except BaseException:
                error = traceback.format_exc()
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.argv = old_argv
        sys.path = old_path
        for key in ("__file__", "__name__", "__package__", "__spec__"):
            _GLOBALS.pop(key, None)
        _GLOBALS.update(old_specials)

    response = _base_response(error is None, stdout.getvalue(), stderr.getvalue(), error)
    response["script_path"] = script_path
    return _maybe_capture_view(response, capture_view, viewport_path)


def _run_python_schema():  # type: () -> Dict[str, Any]
    return {
        "name": RUN_PYTHON_TOOL,
        "description": "Execute Python 3.7-compatible code in the live 3ds Max pymxs session and return stdio, repr(result), scene state, and optionally a viewport screenshot path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python 3.7-compatible code to execute. The final expression is returned."},
                "capture_view": {"type": "boolean", "default": False},
                "viewport_path": {"type": "string", "description": "Optional PNG path for the active viewport capture."},
            },
            "required": ["code"],
        },
    }


def _execute_script_schema():  # type: () -> Dict[str, Any]
    return {
        "name": EXECUTE_SCRIPT_TOOL,
        "description": "Execute a Python 3.7-compatible script from its full path inside the live 3ds Max pymxs session and return stdout/stderr plus scene state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "script_path": {"type": "string", "description": "Full path to the Python script on disk."},
                "script_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Optional sys.argv[1:] values for the script.",
                },
                "capture_view": {"type": "boolean", "default": False},
                "viewport_path": {"type": "string", "description": "Optional PNG path for the active viewport capture."},
            },
            "required": ["script_path"],
        },
    }


def _tool_schemas():  # type: () -> List[Dict[str, Any]]
    return [_run_python_schema(), _execute_script_schema()]


def _handle_tool_call(params):  # type: (Dict[str, Any]) -> Dict[str, Any]
    tool_name = params.get("name")
    args = params.get("arguments", {})
    schemas = {schema["name"]: schema["inputSchema"] for schema in _tool_schemas()}
    if not isinstance(tool_name, str) or tool_name not in schemas:
        raise ValueError("Unknown tool: {}".format(tool_name))
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    schema = schemas[tool_name]
    for key in schema["required"]:
        if key not in args:
            raise ValueError("Missing required argument: {}".format(key))
    for key, value in args.items():
        if key not in schema["properties"]:
            raise ValueError("Unknown argument: {}".format(key))
        expected = schema["properties"][key]["type"]
        valid = ((expected == "string" and isinstance(value, str)) or
                 (expected == "boolean" and isinstance(value, bool)) or
                 (expected == "array" and isinstance(value, list) and all(isinstance(x, str) for x in value)))
        if not valid:
            raise ValueError("Invalid type for {}: expected {}".format(key, expected))
    if tool_name == RUN_PYTHON_TOOL:
        return run_python(args.get("code", ""), bool(args.get("capture_view", False)), args.get("viewport_path"))
    if tool_name == EXECUTE_SCRIPT_TOOL:
        return execute_script(
            args.get("script_path", ""),
            args.get("script_args", []),
            bool(args.get("capture_view", False)),
            args.get("viewport_path"),
        )
    raise ValueError("Unknown tool: {}".format(tool_name))


PROTOCOL_VERSIONS = ("2025-03-26", "2025-06-18", "2025-11-25")
_server = None


def _rpc_error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _handle(request):  # type: (Dict[str, Any]) -> Optional[Dict[str, Any]]
    if (not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or
            not isinstance(request.get("method"), str)):
        return _rpc_error(None, -32600, "Expected one JSON-RPC 2.0 request or notification")
    method = request.get("method")
    request_id = request.get("id")
    if "id" in request and (isinstance(request_id, bool) or not isinstance(request_id, (str, int))):
        return _rpc_error(None, -32600, "Request id must be a string or integer")
    # Never execute a tools/call notification: it has no response channel.
    if "id" not in request:
        return None
    params = request.get("params", {})
    if not isinstance(params, dict):
        return _rpc_error(request_id, -32602, "params must be an object")
    if method == "initialize":
        version = params.get("protocolVersion")
        if not isinstance(version, str):
            return _rpc_error(request_id, -32602, "protocolVersion must be a string")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": version if version in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "behavior-1k-3dsmax-repl", "version": "0.3.0"},
                "instructions": "Code runs in 3ds Max Python {}. Keep snippets Python 3.7-compatible.".format(
                    sys.version.split()[0]),
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tool_schemas()}}
    if method == "tools/call":
        try:
            payload = _handle_tool_call(params)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2, allow_nan=False)}],
                    "isError": not payload["ok"],
                },
            }
        except ValueError as exc:
            return _rpc_error(request_id, -32602, str(exc))
        except Exception as exc:
            return _rpc_error(request_id, -32603, str(exc))
    if request_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "Unknown method: {}".format(method)},
        }
    return None


class _HTTPServer(HTTPServer):
    # Avoid Windows allowing two listeners on the same port.
    allow_reuse_address = False


class _MCPHandler(BaseHTTPRequestHandler):
    """Stateless Streamable HTTP with JSON responses (no optional SSE stream)."""

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        # Max's streams are for captured user code, never HTTP logging.
        pass

    def _reply(self, status, payload=None, allow=None):
        body = b"" if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if payload is not None:
            self.send_header("Content-Type", "application/json")
        if allow:
            self.send_header("Allow", allow)
        self.end_headers()
        self.close_connection = True
        if body:
            self.wfile.write(body)

    def _check_request(self):
        port = self.server.server_port
        allowed_hosts = {"127.0.0.1:{}".format(port), "localhost:{}".format(port)}
        if port == 80:
            allowed_hosts.update(("127.0.0.1", "localhost"))
        if self.headers.get("Host", "").lower() not in allowed_hosts:
            self._reply(403, {"error": "Invalid Host"})
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() not in {"http://" + host for host in allowed_hosts}:
            self._reply(403, {"error": "Invalid Origin"})
            return False
        if urlsplit(self.path).path != "/mcp":
            self._reply(404, {"error": "Use /mcp"})
            return False
        version = self.headers.get("MCP-Protocol-Version")
        if version is not None and version not in PROTOCOL_VERSIONS:
            self._reply(400, {"error": "Unsupported MCP-Protocol-Version"})
            return False
        return True

    def do_GET(self):
        if self._check_request():
            self._reply(405, allow="POST")

    do_DELETE = do_GET
    do_OPTIONS = do_GET

    def do_POST(self):
        if not self._check_request():
            return
        if self.headers.get_content_type() != "application/json":
            self._reply(415, {"error": "Content-Type must be application/json"})
            return
        accepted = {part.split(";", 1)[0].strip() for part in self.headers.get("Accept", "").split(",")}
        if not {"application/json", "text/event-stream"}.issubset(accepted):
            self._reply(406, {"error": "Accept must include application/json and text/event-stream"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
            self._reply(400, {"error": "One Content-Length required; transfer encoding is unsupported"})
            return
        try:
            length = int(lengths[0])
        except ValueError:
            length = -1
        if not 0 < length <= 4 * 1024 * 1024:
            self._reply(413 if length > 0 else 400, {"error": "Request body must be 1 byte to 4 MiB"})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete request body")
            request = json.loads(body.decode("utf-8"), parse_constant=self._invalid_constant)
        except (ValueError, UnicodeError, RecursionError):
            self._reply(400, _rpc_error(None, -32700, "Invalid JSON"))
            return
        except OSError:
            self._reply(408, {"error": "Timed out reading request body"})
            return
        version = self.headers.get("MCP-Protocol-Version", "2025-03-26")
        if isinstance(request, list) and version == "2025-03-26" and request:
            # Batching was removed in 2025-06-18; retain it only for the older version.
            response = [_handle(item) for item in request]
            response = [item for item in response if item is not None] or None
        else:
            response = _handle(request)
        try:
            self._reply(202 if response is None else 200, response)
        except OSError:
            # Disconnection does not undo an executed request.
            pass

    @staticmethod
    def _invalid_constant(value):
        raise ValueError("Invalid JSON constant: " + value)


def stop_server():
    """Stop after sending the current response; callable through run_python."""
    if _server is not None:
        _server.stopped = True


def main(port=8765):
    """Serve requests synchronously until stop_server() is called or interrupted."""
    global _server
    if _server is not None:
        raise RuntimeError("MCP server is already running")
    with _HTTPServer(("127.0.0.1", port), _MCPHandler) as server:
        _server = server
        server.stopped = False
        server.timeout = 0.25
        _GLOBALS["stop_server"] = stop_server
        try:
            while not server.stopped:
                server.handle_request()
        finally:
            _server = None
            _GLOBALS.pop("stop_server", None)


if __name__ == "__main__":
    main()
