"""Authenticated local Web Host transport for a UNI-owned episode executor."""
from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit
from http.client import HTTPConnection, HTTPException


class HostUnavailable(RuntimeError):
    """The DSH Web Host stopped answering; planner state and robot stop are unknown."""


class WebHostHarness:
    """Run once on the existing DSH Host; never reconnect or replay admission."""

    def __init__(self, url: str, config_path: str, run_id: str, uni_token: str):
        target = urlsplit(url)
        if target.scheme != "http" or target.hostname != "127.0.0.1" or not target.port:
            raise ValueError("DSH Web Host must use an explicit loopback HTTP port")
        self.host, self.port = target.hostname, target.port
        self.config_path, self.run_id, self.uni_token = config_path, run_id, uni_token
        self.token = os.environ["RSS_DSH_HOST_TOKEN"]
        self.stopped = threading.Event()
        self.settled = threading.Event()
        self.lost = False

    def __enter__(self):
        return self

    def _request(self, operation: str, payload: dict, timeout=None):
        connection = HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            connection.request("POST", f"/api/rss-agent/{operation}", json.dumps(payload),
                               {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
            response = connection.getresponse()
            result = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError(result.get("error", f"DSH Host HTTP {response.status}"))
            return result
        finally:
            connection.close()

    def run(self, prompt: str):
        if self.stopped.is_set():
            raise RuntimeError("DSH run cancelled before admission")
        try:
            self._request("run", {"configPath": self.config_path, "uniToken": self.uni_token})
        except (OSError, HTTPException) as error:
            self.lost = True
            raise HostUnavailable(f"DSH Web Host at {self.host}:{self.port} stopped answering during the episode: "
                                  f"{type(error).__name__}: {error}") from error
        self.settled.set()
        return SimpleNamespace(finish_reason="completed", final_response="")

    def close(self):
        self.stopped.set()
        if self.settled.is_set() or self.lost:
            return
        try:
            # The acknowledgement covers planner disposal, never physical robot stop.
            self._request("cancel", {"runId": self.run_id}, timeout=120)
        except (OSError, HTTPException) as error:
            self.lost = True
            raise HostUnavailable(f"DSH Web Host did not acknowledge cancellation: {type(error).__name__}: {error}") from error

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception:
            # Keep the failure that ended the run; a cleanup error must not replace it.
            if exc_type is None:
                raise
