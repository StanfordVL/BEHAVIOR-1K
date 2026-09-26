"""Standalone regression tests: python -m unittest discover -s <this directory>.

No running 3ds Max or third-party packages required. The pymxs runtime is mocked;
HTTP tests use actual loopback sockets and execute tools on the test main thread.
"""
import ast
import http.client
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock


class FakeRuntime:
    maxFilePath = "/tmp/"
    maxFileName = "fake.max"
    objects = []
    selection = []
    cameras = []
    lights = []
    scenematerials = []
    renderers = types.SimpleNamespace(current="renderer")

    def classOf(self, value):
        return type(value).__name__

    def isProperty(self, value, name):
        return hasattr(value, name)


SERVER_PATH = Path(__file__).with_name("mcp_repl_server.py")
spec = importlib.util.spec_from_file_location("max_mcp_tested", str(SERVER_PATH))
server = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"pymxs": types.SimpleNamespace(runtime=FakeRuntime())}):
    spec.loader.exec_module(server)


def request(method, params=None, request_id=1):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        server._GLOBALS.clear()
        server._GLOBALS.update(__name__="__3dsmax_mcp_repl__", rt=server.rt)

    def test_python37_syntax(self):
        ast.parse(SERVER_PATH.read_text(encoding="utf-8"), feature_version=(3, 7))

    def test_state_and_stream_capture(self):
        result = server.run_python('import sys\nx = 40\nprint("out")\nprint("err", file=sys.stderr)\nx + 2')
        self.assertEqual(result["result_json"], 42)
        self.assertEqual(result["stdout"], "out\n")
        self.assertEqual(result["stderr"], "err\n")
        self.assertEqual(server.run_python("x")["result_json"], 40)

    def test_compile_entire_snippet_before_side_effects(self):
        result = server.run_python("side_effect = True\n(await missing())")
        self.assertFalse(result["ok"])
        self.assertNotIn("side_effect", server._GLOBALS)

    def test_exits_and_interrupts_do_not_escape(self):
        for code, ok in [("raise SystemExit(0)", True), ("raise SystemExit(2)", False),
                         ("raise KeyboardInterrupt()", False)]:
            self.assertEqual(server.run_python(code)["ok"], ok)

    def test_script_context_encoding_imports_and_restoration(self):
        old_argv, old_path = sys.argv, sys.path
        server._GLOBALS["__file__"] = "original.py"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "script.py"
            (Path(directory) / "mcp_test_sibling.py").write_text("value = 123\n")
            path.write_bytes(('# coding: latin-1\nimport sys\nfrom mcp_test_sibling import value\n'
                              'if __name__ == "__main__":\n    print("caf\xe9", sys.argv[1], value)\n'
                              'script_value = 55\nraise SystemExit(0)\n').encode("latin-1"))
            result = server.execute_script(str(path), ["arg"])
        sys.modules.pop("mcp_test_sibling", None)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["stdout"], "caf\xe9 arg 123\n")
        self.assertEqual(server._GLOBALS["script_value"], 55)
        self.assertEqual(server._GLOBALS["__name__"], "__3dsmax_mcp_repl__")
        self.assertEqual(server._GLOBALS["__file__"], "original.py")
        self.assertIs(sys.argv, old_argv)
        self.assertIs(sys.path, old_path)
        with self.assertRaises(ValueError):
            server.execute_script("relative.py")

    def test_missing_script_and_scene_failure_preserve_output(self):
        with tempfile.TemporaryDirectory() as directory:
            result = server.execute_script(os.path.join(directory, "absent.py"))
        self.assertFalse(result["ok"])
        with mock.patch.object(server, "get_scene_state", side_effect=RuntimeError("bad plugin")):
            result = server.run_python('print("kept"); 7')
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "kept\n")
        self.assertIn("bad plugin", result["scene_state_error"])

    def test_cycles_nonfinite_and_bad_repr(self):
        result = server.run_python('a = []; a.append(a); {"cycle": a, "inf": float("inf")}')
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["result_json"]["inf"], "inf")
        result = server.run_python('class Bad:\n    def __repr__(self):\n        raise ValueError()\nBad()')
        self.assertIn("unrepresentable", result["result_repr"])

    def test_output_is_bounded(self):
        result = server.run_python('print("x" * (2 * 1024 * 1024))')
        self.assertLess(len(result["stdout"]), 1024 * 1024 + 100)
        self.assertIn("truncated", result["stdout"])

    def test_bad_result_key_does_not_lose_stdout(self):
        result = server.run_python('class Key:\n    def __str__(self):\n        raise ValueError("bad key")\n'
                                   'print("kept"); {Key(): 1}')
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "kept\n")
        self.assertIn("bad key", result["result_serialization_error"])

    def test_viewport_filename_and_cleanup_on_failure(self):
        bitmap = types.SimpleNamespace()
        runtime = mock.Mock()
        runtime.gw.getViewportDib.return_value = bitmap
        runtime.save.return_value = True
        with mock.patch.object(server, "rt", runtime):
            path = server.capture_viewport()
            self.assertEqual(bitmap.filename, path)
            runtime.save.assert_called_once_with(bitmap, quiet=True)
            runtime.close.assert_called_once_with(bitmap)
            runtime.close.reset_mock()
            runtime.save.return_value = False
            with self.assertRaises(IOError):
                server.capture_viewport()
            runtime.close.assert_called_once_with(bitmap)

    def test_mcp_validation_and_error_flag(self):
        for params in ({"name": "run_python"}, {"name": "run_python", "arguments": []},
                       {"name": []}, {"name": "run_python", "arguments": {"code": "1", "capture_view": "false"}}):
            self.assertEqual(server._handle(request("tools/call", params))["error"]["code"], -32602)
        response = server._handle(request("tools/call", {"name": "run_python", "arguments": {"code": "1/0"}}))
        self.assertTrue(response["result"]["isError"])
        self.assertIn("ZeroDivisionError", response["result"]["content"][0]["text"])
        notification = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "run_python"}}
        with mock.patch.object(server, "run_python") as execute:
            self.assertIsNone(server._handle(notification))
            execute.assert_not_called()


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.http = server._HTTPServer(("127.0.0.1", 0), server._MCPHandler)
        self.http.timeout = 0.01

    def tearDown(self):
        self.http.server_close()

    def exchange(self, payload=None, method="POST", headers=None, path="/mcp", raw=None):
        output, errors = [], []
        def client():
            connection = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=3)
            try:
                actual_headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                                  "MCP-Protocol-Version": "2025-11-25"}
                actual_headers.update(headers or {})
                body = raw if raw is not None else json.dumps(payload)
                connection.request(method, path, body.encode("utf-8") if method == "POST" else None, actual_headers)
                response = connection.getresponse()
                data = response.read()
                output.append((response.status, json.loads(data) if data else None))
            except BaseException as exc:
                errors.append(exc)
            finally:
                connection.close()
        worker = threading.Thread(target=client)
        worker.start()
        deadline = time.monotonic() + 5
        while worker.is_alive() and time.monotonic() < deadline:
            self.http.handle_request()
            worker.join(0.001)
        self.assertFalse(worker.is_alive())
        if errors:
            raise errors[0]
        return output[0]

    def test_initialize_negotiate_discover_ping(self):
        for version in server.PROTOCOL_VERSIONS + ("unknown",):
            status, body = self.exchange(request("initialize", {"protocolVersion": version}))
            self.assertEqual(status, 200)
            self.assertEqual(body["result"]["protocolVersion"], version if version in server.PROTOCOL_VERSIONS else "2025-11-25")
        self.assertEqual(self.exchange({"jsonrpc": "2.0", "method": "notifications/initialized"}), (202, None))
        self.assertEqual(len(self.exchange(request("tools/list"))[1]["result"]["tools"]), 2)
        self.assertEqual(self.exchange(request("ping"))[1]["result"], {})

    def test_tool_and_scene_inspection_execute_on_main_thread(self):
        main_thread = threading.get_ident()
        def scene():
            self.assertEqual(threading.get_ident(), main_thread)
            return {}
        with mock.patch.object(server, "get_scene_state", side_effect=scene):
            status, body = self.exchange(request("tools/call", {"name": "run_python", "arguments": {
                "code": "import threading; threading.get_ident()"}}))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body["result"]["content"][0]["text"])["result_json"], main_thread)

    def test_http_validation_and_server_survives(self):
        for headers, expected in [({"Origin": "https://evil.example"}, 403),
                                  ({"Host": "evil.example"}, 403),
                                  ({"Content-Type": "text/plain"}, 415),
                                  ({"Accept": "application/json"}, 406),
                                  ({"MCP-Protocol-Version": "unsupported"}, 400),
                                  ({"Content-Length": "5000000"}, 413),
                                  ({"Transfer-Encoding": "chunked"}, 400)]:
            self.assertEqual(self.exchange(request("ping"), headers=headers)[0], expected)
        self.assertEqual(self.exchange(raw="{")[0], 400)
        self.assertEqual(self.exchange(raw='{"x":NaN}')[0], 400)
        self.assertEqual(self.exchange(request("ping"), path="/missing")[0], 404)
        self.assertEqual(self.exchange(method="GET"), (405, None))
        self.assertEqual(self.exchange(method="DELETE"), (405, None))
        self.assertEqual(self.exchange(request("ping"))[0], 200)

    def test_old_protocol_batch_and_invalid_envelopes(self):
        status, body = self.exchange([request("ping", request_id=0), {"jsonrpc": "2.0", "method": "notifications/initialized"}],
                                     headers={"MCP-Protocol-Version": "2025-03-26"})
        self.assertEqual(status, 200)
        self.assertEqual(body, [{"jsonrpc": "2.0", "id": 0, "result": {}}])
        for value in ([], [request("ping")], 42, None, {}, request("ping", request_id=True)):
            self.assertEqual(self.exchange(value)[1]["error"]["code"], -32600)


class LifecycleTests(unittest.TestCase):
    def test_main_serves_on_calling_thread_and_stops_after_reply(self):
        results, errors = [], []
        main_ident = threading.get_ident()

        def client():
            connection = None
            try:
                deadline = time.monotonic() + 3
                while server._server is None and time.monotonic() < deadline:
                    time.sleep(0.001)
                if server._server is None:
                    raise AssertionError("Server did not start")
                connection = http.client.HTTPConnection("127.0.0.1", server._server.server_port, timeout=3)
                code = "import threading; stop_server(); threading.get_ident()"
                payload = request("tools/call", {"name": "run_python", "arguments": {"code": code}})
                connection.request("POST", "/mcp", json.dumps(payload), {
                    "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-11-25"})
                response = connection.getresponse()
                results.append((response.status, json.loads(response.read())))
            except BaseException as exc:
                errors.append(exc)
            finally:
                if connection is not None:
                    connection.close()
                server.stop_server()

        worker = threading.Thread(target=client)
        worker.start()
        try:
            server.main(port=0)
        finally:
            worker.join(5)
        self.assertFalse(worker.is_alive())
        if errors:
            raise errors[0]
        self.assertIsNone(server._server)
        self.assertEqual(results[0][0], 200)
        payload = json.loads(results[0][1]["result"]["content"][0]["text"])
        self.assertEqual(payload["result_json"], main_ident)

    def test_script_entry_point_starts_server_and_closes_socket(self):
        namespace = {"__name__": "__main__"}
        listener = mock.Mock()
        listener.__enter__ = mock.Mock(return_value=listener)
        listener.__exit__ = mock.Mock(return_value=False)
        def handle_request():
            namespace["stop_server"]()
        listener.handle_request.side_effect = handle_request
        with mock.patch("http.server.HTTPServer.__new__", return_value=listener), mock.patch.dict(
                sys.modules, {"pymxs": types.SimpleNamespace(runtime=FakeRuntime())}):
            exec(compile(SERVER_PATH.read_text(encoding="utf-8"), str(SERVER_PATH), "exec"), namespace)
        listener.handle_request.assert_called_once()
        listener.__exit__.assert_called_once()
        self.assertIsNone(namespace["_server"])


if __name__ == "__main__":
    unittest.main()
