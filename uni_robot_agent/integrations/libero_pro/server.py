"""Serve a pinned LIBERO-PRO task without generating or rewriting its assets."""
import os
from pathlib import Path

from uni_robot_agent.integrations.libero_plus.server import BUDGETS, parser as family_parser, prepare
from uni_robot_agent.integrations.simulation import serve_episode

REVISION = "eafdb809426b13153aa1e4c42d6601844217dfec"
PORT = 18782
DATA_SUITES = tuple(base + suffix for base in BUDGETS for suffix in ("_object", "_swap", "_task", "_lan"))
SUITES = (*BUDGETS, *DATA_SUITES, *(base + "_env" for base in BUDGETS))


def parser():
    result = family_parser(suites=SUITES, description=__doc__, port=PORT)
    result.add_argument("--task-data-root", type=Path, help="Official bddl_files/ and init_files/ tree")
    result.add_argument("--listen-fd", type=int, help="Inherit a bound TCP socket from the deployment owner")
    return result


def main():
    args = parser().parse_args()
    if args.listen_fd is not None:
        os.set_inheritable(args.listen_fd, False)
    serve_episode(lambda: prepare(args, benchmark_name="libero_pro", revision=REVISION,
                                  task_data_root=args.task_data_root),
                  args, "libero_pro", listen_fd=args.listen_fd)


if __name__ == "__main__":
    main()
