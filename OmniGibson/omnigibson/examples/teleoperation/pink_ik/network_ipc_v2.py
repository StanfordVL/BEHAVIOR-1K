#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:    Ji Yingwei
Created:   2025-11-20
Description:
    This module uses a lightweight network-based IPC layer to exchange structured messages between processes.
"""
import multiprocessing
import time
import numpy as np
import zmq


# -------- minimal ZMQ-based IPC  --------
def _to_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj
    return np.array(obj)


class NetworkIPC:
    """
    A tiny network IPC to mirror your SharedMemoryIPC API using ZeroMQ REQ/REP.

    Pattern:
      - Client.wait()  -> asks server for current snapshot (SNAPSHOT) and loads it.
      - Client.commit()-> sends queued updates (UPDATE) to server (ACK on success).
      - Server.wait()  -> blocks until it receives an UPDATE, applies it, replies ACK.
                          If it gets a WAIT first, it returns a SNAPSHOT and keeps waiting.
      - Server.commit()-> no-op (kept for symmetry).

    Security: uses pickle via send/recv_pyobj; keep to trusted networks or replace with safe serialization.
    """
    def __init__(self, name, config, is_server=False, host="127.0.0.1", port=5555, linger_ms=0):
        self.name = name
        self.is_server = is_server
        self.config = {k: (tuple(shape), np.dtype(dt)) for k, (shape, dt) in config.items()}
        self.data = {k: np.zeros(shape, dtype=dt) for k, (shape, dt) in self.config.items()}
        self.pending = {}

        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.REP if is_server else zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, linger_ms)

        addr = f"tcp://{host}:{port}"
        if is_server:
            self.socket.bind(addr)
            print(f"[Server] bound at {addr}")
        else:
            # retry connect a bit to allow server to start
            for attempt in range(60):
                try:
                    self.socket.connect(addr)
                    print(f"[Client] connected to {addr}")
                    break
                except zmq.ZMQError:
                    time.sleep(0.1)
            else:
                raise RuntimeError(f"Failed to connect to {addr}")

    # ---- helpers ----
    def _validate_and_assign(self, updates: dict):
        for k, v in updates.items():
            if k not in self.config:
                raise KeyError(f"Unknown key '{k}'")
            v = _to_numpy(v)
            shape, dt = self.config[k]
            if v.shape != shape:
                raise ValueError(f"Shape mismatch for '{k}': got {v.shape}, expected {shape}")
            if v.dtype != dt:
                v = v.astype(dt, copy=False)
            self.data[k][...] = v

    def get(self, key):
        return self.data[key].copy()

    def set(self, key, value):
        if self.is_server:
            # immediately publish to snapshot so client sees it on next WAIT
            self._validate_and_assign({key: value})
        else:
            self.pending[key] = _to_numpy(value)

    # ---- client side ----
    def _client_wait(self):
        self.socket.send_pyobj({"type": "WAIT"})
        msg = self.socket.recv_pyobj()
        if not (isinstance(msg, dict) and msg.get("type") == "SNAPSHOT"):
            raise RuntimeError(f"Unexpected server reply to WAIT: {msg}")
        self._validate_and_assign(msg["data"])

    def _client_commit(self):
        self.socket.send_pyobj({"type": "UPDATE", "data": self.pending})
        self.pending.clear()
        msg = self.socket.recv_pyobj()
        if not (isinstance(msg, dict) and msg.get("type") == "ACK"):
            raise RuntimeError(f"Unexpected server reply to UPDATE: {msg}")

    # ---- server side ----
    def _server_wait(self):
        while True:
            msg = self.socket.recv_pyobj()
            if not isinstance(msg, dict):
                self.socket.send_pyobj({"type": "ERR", "reason": "bad message"})
                continue
            typ = msg.get("type")
            if typ == "WAIT":
                self.socket.send_pyobj({"type": "SNAPSHOT", "data": self.data})
            elif typ == "UPDATE":
                updates = msg.get("data", {})
                self._validate_and_assign(updates)
                self.socket.send_pyobj({"type": "ACK"})
                break
            else:
                self.socket.send_pyobj({"type": "ERR", "reason": f"unknown type {typ}"})

    def _server_commit(self):
        # no-op
        pass

    # ---- public API (mirrors original) ----
    def wait(self):
        if self.is_server:
            self._server_wait()
        else:
            self._client_wait()

    def commit(self):
        if self.is_server:
            self._server_commit()
        else:
            self._client_commit()
    
    def talk(self):
        self.commit()
        self.wait()

    def close(self):
        if self.is_server:
            self._server_wait()
        else:
            self._client_commit()
        try:
            self.socket.close()
        except Exception:
            pass


# -------- your original demo flow, unchanged in spirit --------
def writer_process(config, host, port):
    # acts like your original "client" process
    time.sleep(1.0)  # small delay so server binds first
    ipc = NetworkIPC("test", config, is_server=False, host=host, port=port)
    for i in range(5):
        # send cam_high to server for next round
        ipc.set("cam_high", np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8))
        print(f"[Client] Sent cam_high {i}")

        ipc.talk()  # fetch server snapshot (contains 'state')

        state = ipc.get("state")
        print(f"[Client] Received state mean: {state.mean():.3f}")
    
    # ipc.commit()
    ipc.close()
    print("client close")


def reader_process(config, host, port):
    # acts like your original "server" process
    ipc = NetworkIPC("test", config, is_server=True, host=host, port=port)
    for i in range(5):
        ipc.talk()  # wait until client sent UPDATE (cam_high)
        cam = ipc.get("cam_high")
        print(f"[Server] Received cam_high sum: {int(cam.sum())}")

        # dummy inference
        time.sleep(0.2)

        # publish next state for client to read on next WAIT
        ipc.set("state", np.ones((1, 14), dtype=np.float32) * i)

        print(f"[Server] set state sum: {i}")
    # time.sleep(1)
    # ipc.wait()
    ipc.close()

    print("server close")


if __name__ == "__main__":
    # same config as your reference 2
    config = {
        "state":    ((1, 14), np.float32),
        "cam_high": ((1, 224, 224, 3), np.uint8),
    }

    HOST = "127.0.0.1"  # set to server machine IP for cross-PC
    PORT = 5555

    writer = multiprocessing.Process(target=writer_process, args=(config, HOST, PORT))
    reader = multiprocessing.Process(target=reader_process, args=(config, HOST, PORT))

    # start server then client (order matters for bind/connect)
    reader.start()
    writer.start()

    writer.join()
    reader.join()
