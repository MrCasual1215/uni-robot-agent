"""Prepare the pinned LIBERO-PRO runtime and launch one owned episode."""
from pathlib import Path

from uni_robot_agent.services.deployment.config import LaunchPlan, Recipe, Runtime
from uni_robot_agent.services.deployment.runtime import check_python
from uni_robot_agent.services.deployment.setup import Preparation

from .server import DATA_SUITES, PORT, REVISION, SUITES


def recipe(_args) -> Recipe:
    return Recipe.load(__package__, "deployment.toml")


def validate(args) -> None:
    if args.runtime_profile or args.worker_python:
        raise ValueError("LIBERO-PRO uses one native runtime without --runtime-profile or --worker-python")
    if args.command == "setup":
        return
    if args.suite not in SUITES or not args.task:
        raise ValueError("LIBERO-PRO deployment requires --task and --suite (" + ", ".join(SUITES) + ")")
    if any(getattr(args, field) is not None for field in ("layout", "layout_sha256", "task_dir", "cameras")):
        raise ValueError("LIBERO-PRO uses suite initial states and its two fixed cameras")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.init_state_index is not None and args.init_state_index < 0:
        raise ValueError("--init-state-index must be nonnegative")
    if any(value is not None and value <= 0 for value in (args.settle_steps, args.max_steps)):
        raise ValueError("--settle-steps and --max-steps must be positive")
    if getattr(args, "tool_profile", None) == "clean" and args.depth:
        raise ValueError("The clean profile requires RGB observations; use --no-depth")


def external_data(args) -> dict[str, str]:
    if args.native_root is None:
        raise ValueError("An external LIBERO-PRO runtime requires --native-root")
    return {"native_root": str(args.native_root.expanduser().resolve())}


def prepare(context: Preparation) -> None:
    settings = context.recipe.settings
    context.data["native_root"] = context.step("source", lambda: context.git(
        settings["repository"], REVISION, context.root / "source"))
    patterns = [f"{directory}/{suite}/*" for directory in ("bddl_files", "init_files") for suite in DATA_SUITES]
    context.data["task_data_root"] = context.step("task-data", lambda: context.snapshot(
        settings["task_data_repository"], settings["task_data_revision"], context.root / "task-data",
        patterns=patterns))
    context.data["task_data_revision"] = settings["task_data_revision"]
    context.step("torch", lambda: context.pip(settings["torch"], options=["--index-url", settings["torch_index"]]))
    context.step("dependencies", lambda: context.pip(*settings["packages"]))
    context.environment.setdefault("MUJOCO_GL", "egl")
    # Official initial-state archives contain NumPy arrays in Torch's pickle format.
    context.environment.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def verify(runtime: Runtime) -> dict:
    native = Path(runtime.data["native_root"]) / "libero/libero"
    for name in ("assets", "benchmark", "bddl_files", "init_files"):
        if not (native / name).is_dir():
            raise ValueError(f"Missing LIBERO-PRO directory {native / name}")
    if task_data_root := runtime.data.get("task_data_root"):
        for suite in DATA_SUITES:
            for name in ("bddl_files", "init_files"):
                if not (Path(task_data_root) / name / suite).is_dir():
                    raise ValueError(f"Missing LIBERO-PRO task data for {suite} under {name}")
    return check_python(runtime.python, ["bddl.parsing", "mujoco", "robosuite.utils.camera_utils", "torch", "yaml"],
                        environment=runtime.environment)


def plan(args, runtime: Runtime, output: Path, run_id: str) -> LaunchPlan:
    port = args.port or PORT
    seed = args.seed if args.seed is not None else 0
    state_index = args.init_state_index if args.init_state_index is not None else 0
    command = [
        runtime.python, "-m", "uni_robot_agent.integrations.libero_pro.server",
        "--native-root", runtime.data["native_root"], "--suite", args.suite, "--task", args.task,
        "--run-id", run_id, "--output-dir", str(output / "environment"), "--seed", str(seed),
        "--init-state-index", str(state_index), "--host", args.host, "--port", str(port),
        "--depth" if args.depth else "--no-depth",
    ]
    for option in ("settle_steps", "max_steps"):
        if (value := getattr(args, option)) is not None:
            command.extend(("--" + option.replace("_", "-"), str(value)))
    if task_data_root := runtime.data.get("task_data_root"):
        if args.suite.endswith("_env"):
            raise ValueError("The managed LIBERO-PRO data does not contain _env suites; use a prepared --env-url")
        if args.suite in DATA_SUITES:
            command.extend(("--task-data-root", task_data_root))
    expected = {"integration": "libero_pro", "episode_id": run_id, "metadata.native_revision": REVISION,
                "metadata.suite": args.suite, "metadata.seed": seed, "metadata.init_state_index": state_index}
    if args.task.isdecimal():
        expected["metadata.task_index"] = int(args.task)
    else:
        expected["metadata.task_id"] = args.task
    connect_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    return LaunchPlan(tuple(command), f"http://{connect_host}:{port}", expected, runtime.environment, output,
                      listen_fd_option="--listen-fd")
