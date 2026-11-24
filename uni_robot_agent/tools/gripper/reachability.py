"""Read-only reachability queries for proposed grasp-site goals.

``preview_move`` dry-runs a whole move through the executor's motion program
(see ``preview.py``); ``check_reachability`` only asks for a final-goal IK solution.

Uses the environment's goal-region IK search, which solves without stepping
physics, and reports how far each goal is from the measured pose. It does not
predict collisions or the tracking behaviour of an interpolated move.
"""
from __future__ import annotations

import numpy as np

from uni_robot_agent.execution.scheduler import ToolExecutionPolicy
from uni_robot_agent.services.environment.contracts import GoalRegionRequest, Pose, RequestRejected
from uni_robot_agent.tools.gripper.contracts import MOTION_PROFILES, MoveRequest, motion_profiles
from uni_robot_agent.tools.gripper.pose import quaternion_distance, rpy_to_quaternion
from uni_robot_agent.tools.gripper.preview import preview_move
from uni_robot_agent.tools.gripper.provider import POSE6
from uni_robot_agent.tools.toolkit import ToolOutput, ToolProvider, ToolSpec, bind_tools, schema

NAME = "check_reachability"
PREVIEW = "preview_move"


def tool_policies():
    # The service serializes the solve with motion; it never advances the world.
    return {name: ToolExecutionPolicy(resource_key="episode", concurrency="exclusive") for name in (NAME, PREVIEW)}


class ReachabilityTools(ToolProvider):
    provider_id = "reachability"

    def __init__(self, episode):
        self.episode = episode

    def get_tools(self):
        arms = [arm.name for arm in self.episode.description.environment.arms]
        target = schema({"arm": {"type": "string", "enum": arms}, "pose": POSE6}, ("arm", "pose"))
        spec = ToolSpec(NAME,
            "Check whether absolute grasp-site goals have an IK solution from the measured joints, without moving. "
            "Reports found, the solver's residual errors, and the translation and rotation change from the measured "
            "pose, plus the retained gripper opening. A found goal is not a collision check and does not guarantee that an interpolated move tracks it.",
            schema({"targets": {"type": "array", "items": target, "minItems": 1, "maxItems": len(arms)},
                    "profile": {"type": "string", "enum": list(MOTION_PROFILES), "default": "standard"}},
                   ("targets",)), readonly=True)
        move_target = schema({"arm": {"type": "string", "enum": arms}, "pose": POSE6,
                              "gripper_opening_m": {"type": "number"}}, ("arm", "pose"))
        preview = ToolSpec(PREVIEW,
            "Dry-run a move with the executor's own motion program and IK planner, without stepping physics. "
            "Takes the move arguments and reports whether the planner accepts the whole path (accepted, stop_reason), "
            "its cost (predicted_steps, joint_path_rad) and the predicted final grasp-site poses and gripper openings. "
            "Assumes exact tracking; it is not a collision or contact check.",
            schema({"targets": {"type": "array", "items": move_target, "minItems": 1, "maxItems": len(arms)},
                    "profile": {"type": "string", "enum": list(MOTION_PROFILES), "default": "standard"},
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 24, "default": 24},
                    "position_tolerance_m": {"type": "number"},
                    "orientation_tolerance_rad": {"type": "number"}},
                   ("targets",)), readonly=True)
        return bind_tools([spec, preview], provider_id=self.provider_id,
                          handlers={NAME: self.check_reachability, PREVIEW: self.preview_move})

    def preview_move(self, targets, profile="standard", max_steps=24,
                     position_tolerance_m=None, orientation_tolerance_rad=None):
        state = self.episode.state
        request = MoveRequest(profile=profile, max_steps=max_steps, position_tolerance_m=position_tolerance_m,
                              orientation_tolerance_rad=orientation_tolerance_rad,
                              targets=[{"arm": target["arm"], "pose": target["pose"]} for target in targets])
        result = preview_move(self.episode.env, state, request)
        for target in targets:
            metadata = state.arms[target["arm"]].metadata
            # An omitted opening keeps the retained drive target.
            final = result["final"][target["arm"]]
            final["opening_m"] = target.get(
                "gripper_opening_m", metadata.get("gripper_target_opening_m", metadata.get("gripper_opening_m")))
            # How far the commanded opening moves from the measured one: the gripper's share of the state change.
            measured = metadata.get("gripper_opening_m")
            if isinstance(final["opening_m"], (int, float)) and isinstance(measured, (int, float)):
                final["opening_change_m"] = abs(float(final["opening_m"]) - float(measured))
        return ToolOutput(extra={**result, "state_version": state.state_version, "profile": profile})

    def check_reachability(self, targets, profile="standard"):
        description = self.episode.description.environment
        if not description.metadata.get("goal_region_search"):
            raise RequestRejected("This environment does not support goal-region search")
        limits = motion_profiles(description.metadata.get("motion_profiles"))[profile]
        state = self.episode.state
        results = {}
        for target in targets:
            arm, pose = target["arm"], target["pose"]
            measured = state.arms[arm].control_pose
            metadata = state.arms[arm].metadata
            goal_quaternion = rpy_to_quaternion(pose[3:])
            reply = self.episode.env.solve_goal_region(GoalRegionRequest(
                arm=arm, goal=Pose(position=pose[:3], quaternion=goal_quaternion.tolist()),
                position_tolerance_m=limits.position_tolerance_m,
                orientation_tolerance_rad=limits.orientation_tolerance_rad,
                expected_state_version=state.state_version))
            results[arm] = {
                "found": reply.found,
                "translation_m": float(np.linalg.norm(np.asarray(pose[:3]) - np.asarray(measured.position))),
                "rotation_change_rad": quaternion_distance(np.asarray(measured.quaternion), goal_quaternion),
                "diagnostics": reply.diagnostics,
                # What a move that omits gripper_opening_m keeps: the retained drive target.
                "retained_opening_m": metadata.get("gripper_target_opening_m", metadata.get("gripper_opening_m")),
            }
        return ToolOutput(extra={"targets": results, "state_version": state.state_version, "profile": profile})
