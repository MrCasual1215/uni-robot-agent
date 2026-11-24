"""DeepSeek Harness planner backend.

An episode uses an existing local Web Host or one DeepSeek Harness SDK runtime. The runtime profile mounts a
plugin that registers the ``/rss-episode`` command; the command receives this
episode's prompt, tool declarations and a private MCP endpoint over the UNI
executor, and runs one or more Harness agents against it. Model routes and
credentials belong to the Harness profile, so no UNI provider gateway is used.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from uni_robot_agent.agents.contracts import AgentCapabilities, AgentInterrupted, AgentResult
from uni_robot_agent.events import AgentRunStarted
from uni_robot_agent.services.mcp.http import HttpMcpServer

TOKEN_ENV = "RSS_UNI_MCP_TOKEN"
SPEC_KEYS = {"agents", "protocol", "maxRounds", "dshBin", "profile", "patches", "env", "proposalTimeoutS",
             "peerGraceS", "maxNudges", "nativeTools", "permissionPreset", "jointCompaction", "judge", "webHost"}


def load_spec(path: str) -> dict[str, Any]:
    """Read and validate the run specification once, before services start."""
    spec = json.loads(Path(path).read_text())
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise ValueError(f"Unknown DeepSeek Harness spec keys: {sorted(unknown)}")
    if "webHost" in spec:
        from urllib.parse import urlsplit
        if not isinstance(spec["webHost"], str):
            raise ValueError("webHost must be a loopback HTTP URL")
        target = urlsplit(spec["webHost"])
        if (target.scheme != "http" or target.hostname != "127.0.0.1" or not target.port
                or target.username or target.password or target.path not in ("", "/") or target.query or target.fragment):
            raise ValueError("webHost must be http://127.0.0.1:<port>")
    agents = spec.get("agents")
    if not isinstance(agents, list) or not 1 <= len(agents) <= 2:
        raise ValueError("A DeepSeek Harness spec declares one or two agents")
    for agent in agents:
        if not all(isinstance(agent.get(key), str) and agent[key] for key in ("id", "provider", "model")):
            raise ValueError("Each agent requires non-empty id, provider and model")
    if not isinstance(spec.get("protocol"), str):
        raise ValueError("A DeepSeek Harness spec requires a protocol")
    if not isinstance(spec.get("patches", []), list):
        raise ValueError("patches must be a list of patch files")
    return spec


class DshAgent:
    name = "dsh"
    # No UNI gateway is created: SessionConfig leaves provider_url unset for this backend.
    model_protocol = "responses"
    capabilities = AgentCapabilities(tool_event_source="execution")

    @classmethod
    def add_cli_arguments(cls, parser) -> None:
        group = parser.add_argument_group("DeepSeek Harness")
        group.add_argument("--dsh-spec", help="JSON run specification: agents, protocol, profile patches and runtime")

    @classmethod
    def resolve_defaults(cls, args) -> None:
        pass

    @classmethod
    def validate_config(cls, args, explicit: set[str]) -> None:
        if not getattr(args, "dsh_spec", None):
            raise ValueError("The dsh backend requires --dsh-spec")
        if args.provider_url:
            raise ValueError("Configure model routes in the DeepSeek Harness profile, not --provider-url")
        if getattr(args, "interaction_mode", "tool_calling") != "tool_calling":
            raise ValueError("The dsh backend supports tool_calling interaction only")
        load_spec(args.dsh_spec)

    @classmethod
    def config_options(cls, args) -> dict[str, Any]:
        return {"spec_path": args.dsh_spec}

    def __init__(self, *, model: str, spec_path: str, timeout_s: float | None = None):
        self.model = model
        self.spec = load_spec(spec_path)
        self.timeout_s = timeout_s
        self.harness = None
        self.stop_reason: str | None = None
        self.cancel_error: str | None = None
        self._lock = threading.Lock()

    def model_runtime(self, profile, event_bus):
        raise RuntimeError("The dsh backend uses DeepSeek Harness model routes")

    def cancel(self, reason: str) -> None:
        with self._lock:
            self.stop_reason = self.stop_reason or reason
            harness = self.harness
        if harness is not None:
            # Closing the runtime stops planning; it does not stop a submitted robot action.
            try:
                harness.close()
            except Exception as error:  # noqa: BLE001 - run() reports the episode outcome.
                self.cancel_error = f"{type(error).__name__}: {error}"

    def close(self) -> None:
        self.cancel(self.stop_reason or "closed")

    @staticmethod
    def finalize_without_provider(result: AgentResult, session_dir: Path | None) -> None:
        if session_dir is not None:
            result.provider_log_path = str(Path(session_dir) / "dsh")

    def _tools(self, executor) -> list[dict[str, Any]]:
        tools = []
        for schema in executor.get_tool_schemas():
            policy = executor.tool_policy(schema["name"])
            tools.append({"name": schema["name"], "description": schema.get("description", ""),
                          "input_schema": schema.get("input_schema", {"type": "object"}),
                          "gated": bool(policy and policy.mutates_world)})
        if not any(tool["gated"] for tool in tools):
            raise ValueError("The executor declares no world-mutating tools to coordinate")
        return tools

    def _write_run_config(self, *, prompt, executor, session_dir: Path, url: str) -> Path:
        root = session_dir / "dsh"
        images = []
        for index, image in enumerate(prompt.images):
            path = root / "prompt-images" / f"{index:02d}.{image.mime_type.split('/')[-1]}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image.data)
            images.append({"mime": image.mime_type, "path": str(path)})
        keys = ("agents", "protocol", "maxRounds", "proposalTimeoutS", "peerGraceS", "maxNudges", "nativeTools",
                "permissionPreset", "jointCompaction", "judge")
        config = {key: self.spec[key] for key in keys if key in self.spec}
        config.update(runId=session_dir.name, outputDir=str(root / "coordination"), workdir=str(root / "work"),
                      scratchDir=str(session_dir / "scratch"),
                      system=prompt.system or "", user=prompt.user, images=images,
                      uni={"url": url, "tokenEnv": TOKEN_ENV}, tools=self._tools(executor))
        path = root / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))
        return path

    def run(self, *, prompt, task, toolkit, executor, max_turns=0, event_bus=None, session_dir=None,
            model_endpoint=None, code_policy=None) -> AgentResult:
        if model_endpoint is not None:
            raise ValueError("The dsh backend does not use a UNI model endpoint")
        if code_policy is not None:
            raise ValueError("The dsh backend supports tool_calling interaction only")
        if session_dir is None:
            raise ValueError("The dsh backend requires a session directory")
        session_dir = Path(session_dir).absolute()
        started = time.monotonic()
        token = secrets.token_urlsafe(32)
        server = HttpMcpServer(executor, event_bus=event_bus, auth_token=token)
        url = server.start()
        result = AgentResult(trace_format="dsh")
        timer = None
        try:
            run_config = self._write_run_config(prompt=prompt, executor=executor, session_dir=session_dir, url=url)
            if event_bus is not None:
                event_bus.emit(AgentRunStarted(instruction=task, model=self.model, agent_backend=self.name,
                                               tool_event_source=self.capabilities.tool_event_source))
            env = {TOKEN_ENV: token, **{name: os.environ[name] for name in self.spec.get("env", []) if name in os.environ}}
            first = self.spec["agents"][0]
            if self.spec.get("webHost"):
                from .web_host import WebHostHarness
                harness = WebHostHarness(self.spec["webHost"], str(run_config), session_dir.name, token)
            else:
                from deepseek_harness import DeepSeekHarness
                harness = DeepSeekHarness(
                    dsh_home=str(session_dir / "dsh" / "home"), cwd=str(session_dir / "dsh" / "work"),
                    profile=self.spec.get("profile", "sdk"), patches=tuple(self.spec.get("patches", [])),
                    dsh_bin=self.spec.get("dshBin"), provider=first["provider"], model=first["model"],
                    env=env, initialize_timeout_seconds=120,
                )
            (session_dir / "dsh" / "work").mkdir(parents=True, exist_ok=True)
            with self._lock:
                if self.stop_reason:
                    raise RuntimeError(f"Agent stopped: {self.stop_reason}")
                self.harness = harness
            if self.timeout_s:
                timer = threading.Timer(self.timeout_s, self.cancel, args=("wall_time_limit",))
                timer.daemon = True
                timer.start()
            with harness:
                response = harness.run(f"/rss-episode {run_config}")
            summary_path = Path(json.loads(run_config.read_text())["outputDir"]) / "summary.json"
            if not summary_path.is_file():
                raise RuntimeError(f"No episode summary; the runtime ended with {response.finish_reason}: "
                                   f"{response.final_response[-4000:] or 'no result text'}")
            summary = json.loads(summary_path.read_text())
            if summary.get("error"):
                raise RuntimeError(f"Episode failed in DeepSeek Harness: {summary['error']}")
            result.success = bool(summary.get("finished"))
            result.message = summary.get("finishSummary", "")
            result.steps_used = result.turns_used = int(summary.get("decisionSteps", 0))
            result.main_thread_id = ",".join(summary.get("sessions", {}).values())
        except KeyboardInterrupt:
            result.stop_reason = "interrupted"
            result.error = "Interrupted by user"
            raise AgentInterrupted(result)
        except Exception as error:  # noqa: BLE001 - recorded as the agent outcome.
            result.error = f"{type(error).__name__}: {error}"
            if type(error).__name__ == "HostUnavailable":
                # Infrastructure loss, distinct from a planner or model failure.
                result.stop_reason = "host_unavailable"
        finally:
            if timer is not None:
                timer.cancel()
            with self._lock:
                self.harness = None
            server.stop()
            result.stop_reason = result.stop_reason or self.stop_reason
            result.cleanup_error = result.cleanup_error or self.cancel_error
            if result.stop_reason:
                result.success = False
            result.tool_calls = executor.tool_execution_seq
            result.elapsed_s = time.monotonic() - started
        return result
