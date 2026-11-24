"""Kinematic dry run of a move through the executor's own motion program.

``MotionExecutor.move`` runs unchanged against ``KinematicPreviewBackend``:
the same waypoints, stopping rules and goal-region recovery as a real move.
Each pose step is planned by the environment (``plan_pose``: native IK and
its joint-delta guard from the predicted joints) and the arm is assumed to
track the planned joints exactly. No physics is stepped, so the preview says
whether the executor's planner accepts the path and what it costs; it does
not model collisions, contact or tracking lag.
"""
from __future__ import annotations

import numpy as np

from uni_robot_agent.services.environment.contracts import (
    EnvironmentState,
    GoalRegionRequest,
    Pose,
    PoseCommand,
    PosePlanRequest,
    RequestRejected,
    StepReceipt,
)
from uni_robot_agent.tools.gripper.contracts import MoveRequest, motion_profiles
from uni_robot_agent.tools.gripper.motion import MotionExecutor


class KinematicPreviewBackend:
    """A primitive-step backend whose steps are environment plans applied to a predicted state."""

    def __init__(self, client, state: EnvironmentState) -> None:
        self.client = client
        self.description = client.info.environment
        self._state = state.model_copy(deep=True)
        # The environment checks plans against the real, unchanged world.
        self._version = state.state_version
        self.joint_path_rad = 0.0
        self.max_step_joint_delta_rad = 0.0
        self.planned_steps = 0

    def read_state(self) -> EnvironmentState:
        return self._state.model_copy(deep=True)

    def step_pose(self, command: PoseCommand) -> StepReceipt:
        joints = {name: list(arm.joints) for name, arm in self._state.arms.items()}
        reply = self.client.plan_pose(PosePlanRequest(command=command, joints=joints,
                                                      expected_state_version=self._version))
        if not reply.get("accepted"):
            return StepReceipt(accepted=False, state=self.read_state(), diagnostics=reply.get("diagnostics", {}))
        for name, planned in reply["arms"].items():
            arm = self._state.arms[name]
            delta = np.asarray(planned["joints"], float) - np.asarray(arm.joints, float)
            self.joint_path_rad += float(np.linalg.norm(delta))
            self.max_step_joint_delta_rad = max(self.max_step_joint_delta_rad, float(np.max(np.abs(delta))))
            arm.joints = list(planned["joints"])
            arm.control_pose = Pose(**planned["control_pose"])
        self._state.native_steps += 1
        self.planned_steps += 1
        return StepReceipt(accepted=True, state=self.read_state(), diagnostics=reply.get("diagnostics", {}))

    def solve_goal_region(self, request: GoalRegionRequest):
        # The search starts from the measured joints, not the predicted ones.
        return self.client.solve_goal_region(request)

    def _unsupported(self, *_args, **_kwargs):
        raise RequestRejected("Move previews support stepwise pose motion only")

    step_joints = step_native = step_path = _unsupported


def preview_move(client, state: EnvironmentState, request: MoveRequest) -> dict:
    """Dry-run the pose part of a move; gripper motion does not change the kinematic path."""
    if not client.info.environment.metadata.get("pose_plan"):
        raise RequestRejected("This environment does not support pose plans")
    pose_only = request.model_copy(update={"targets": [target.model_copy(update={"gripper": None})
                                                       for target in request.targets]})
    backend = KinematicPreviewBackend(client, state)
    executor = MotionExecutor(backend)
    if executor.wholepath:
        raise RequestRejected("Move previews support stepwise motion execution only")
    feedback = executor.move(pose_only)
    limits = request.parameters(motion_profiles(client.info.environment.metadata.get("motion_profiles")))
    final = backend.read_state()
    return {
        "accepted": bool(feedback["pose_reached"]),
        "stop_reason": feedback["stop_reason"],
        "stop_evidence": feedback["stop_evidence"],
        "goal_recovery": feedback["goal_recovery"] is not None,
        "predicted_steps": backend.planned_steps,
        "joint_path_rad": backend.joint_path_rad,
        "max_step_joint_delta_rad": backend.max_step_joint_delta_rad,
        # The executor's own arrival tolerance for this move.
        "tolerance": {"position_m": limits.position_tolerance_m, "orientation_rad": limits.orientation_tolerance_rad},
        "final": {name: {"control_pose": final.arms[name].control_pose.model_dump(mode="json"),
                         "joints": final.arms[name].joints}
                  for name in (target.arm for target in request.targets)},
    }
