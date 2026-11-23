"""Read-only reachability query for proposed grasp-site goals.

Uses the environment's goal-region IK search, which solves without stepping
physics, and reports how far each goal is from the measured pose. It does not
predict collisions or the tracking behaviour of an interpolated move.
"""
from __future__ import annotations

import numpy as np

from uni_robot_agent.execution.scheduler import ToolExecutionPolicy
from uni_robot_agent.services.environment.contracts import GoalRegionRequest, Pose, RequestRejected
from uni_robot_agent.tools.gripper.contracts import MOTION_PROFILES, motion_profiles
from uni_robot_agent.tools.gripper.pose import quaternion_distance, rpy_to_quaternion
from uni_robot_agent.tools.gripper.provider import POSE6
from uni_robot_agent.tools.toolkit import ToolOutput, ToolProvider, ToolSpec, bind_tools, schema

NAME = "check_reachability"


def tool_policies():
    # The service serializes the solve with motion; it never advances the world.
    return {NAME: ToolExecutionPolicy(resource_key="episode", concurrency="exclusive")}


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
            "pose. A found goal is not a collision check and does not guarantee that an interpolated move tracks it.",
            schema({"targets": {"type": "array", "items": target, "minItems": 1, "maxItems": len(arms)},
                    "profile": {"type": "string", "enum": list(MOTION_PROFILES), "default": "standard"}},
                   ("targets",)), readonly=True)
        return bind_tools([spec], provider_id=self.provider_id, handlers={NAME: self.check_reachability})

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
            }
        return ToolOutput(extra={"targets": results, "state_version": state.state_version, "profile": profile})
