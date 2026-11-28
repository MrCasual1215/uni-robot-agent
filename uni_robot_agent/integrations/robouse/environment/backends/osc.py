"""Panda OSC backends with measured fingertip-centre reference points."""
import numpy as np
from ..control import state, quaternion, rotation
from scipy.spatial.transform import Rotation
from uni_robot_agent.services.environment.contracts import ArmControlCapabilities, RequestRejected


# Damped least squares for kinematic pose plans: damping, iterations, per-iteration joint step and convergence.
IK_DAMPING = 0.05
IK_ITERATIONS = 60
IK_MAX_STEP_RAD = 0.2
IK_POSITION_TOLERANCE_M = 1e-4
IK_ROTATION_TOLERANCE_RAD = 1e-3


class OSC:
    hz = 20.
    constraint = "Full world-frame pose control at the midpoint of the finger pads."

    def __init__(self, backend, kind):
        self.b, self.kind = backend, kind
        self.model, self.data = backend.mj_model_data()
        self.robots = backend.env.robots
        self.names = tuple("arm" if len(self.robots) == 1 else f"arm{i}" for i in range(len(self.robots)))
        # Only the reviewed LIBERO controller is admitted for actor control.
        self.actor_control = ArmControlCapabilities(commands=["pose"]) if kind == "libero" else None
        self.control_point = "panda_fingerpad_midpoint"
        self.entries = []
        offset = 0
        for robot in self.robots:
            if kind == "libero":
                gripper, controller, site = robot.gripper, robot.controller, robot.eef_site_id
                arm_slice, grip_index = slice(0, 6), 6
            else:
                if len(robot.arms) != 1:
                    raise ValueError("Only one Panda arm per robosuite robot is admitted")
                arm = robot.arms[0]
                gripper, controller, site = robot.gripper[arm], robot.part_controllers[arm], robot.eef_site_id[arm]
                start, end = robot._action_split_indexes[arm]
                gs, ge = robot._action_split_indexes[f"{arm}_gripper"]
                if end - start != 6 or ge - gs != 1:
                    raise ValueError("Expected a six-axis OSC arm and scalar gripper")
                arm_slice, grip_index = slice(offset + start, offset + end), offset + gs
            if type(gripper).__name__ != "PandaGripper" or controller.name != "OSC_POSE":
                raise ValueError("Only PandaGripper with OSC_POSE is admitted in this release")
            pads = [gripper.important_geoms[k] for k in ("left_fingerpad", "right_fingerpad")]
            ids = [[self.model.geom(n).id for n in group] for group in pads]
            if getattr(controller, "input_ref_frame", "world") != "world":
                raise ValueError("OSC controller must accept world-frame deltas")
            self.entries.append((controller, int(site), ids, arm_slice, grip_index, robot, gripper))
            offset += robot.action_dim

    def _pose(self, entry, data=None):
        data = self.data if data is None else data
        _, site, pads, _, _, robot, _ = entry
        points = [data.geom_xpos[ids].mean(axis=0) for ids in pads]
        centre = np.mean(points, axis=0)
        quat = quaternion(data.site_xmat[site])
        indexes = robot._ref_gripper_joint_pos_indexes
        if isinstance(indexes, dict):
            indexes = indexes[robot.arms[0]]
        width = float(np.abs(data.qpos[indexes]).sum())
        return centre, quat, width

    def read(self, commands):
        result = {}
        for name, entry in zip(self.names, self.entries):
            centre, quat, width = self._pose(entry)
            result[name] = state(centre, quat, width / .08, commands[name])
        return result

    def encode(self, targets):
        action = np.zeros(self.b.action_spec.dim)
        for name, entry in zip(self.names, self.entries):
            controller, site, _, arm_slice, grip_index, _, gripper = entry
            target = targets[name]
            centre, quat, width = self._pose(entry)
            current_rotation, target_rotation = rotation(quat), rotation(target.pose.quaternion)
            local_offset = current_rotation.inv().apply(centre - self.data.site_xpos[site])
            native_target = np.asarray(target.pose.position) - target_rotation.apply(local_offset)
            delta = np.r_[native_target - self.data.site_xpos[site], (target_rotation * current_rotation.inv()).as_rotvec()]
            low, high = np.asarray(controller.output_min), np.asarray(controller.output_max)
            ilow, ihigh = np.asarray(controller.input_min), np.asarray(controller.input_max)
            if np.any(high <= low):
                raise RequestRejected("OSC controller has invalid scaling")
            action[arm_slice] = np.clip(ilow + (delta-low)*(ihigh-ilow)/(high-low), ilow, ihigh)
            opening_command = (1. + float(np.asarray(gripper.current_action).ravel()[0])) / 2
            error = opening_command - target.gripper
            action[grip_index] = 0. if abs(error) < float(gripper.speed) / 4 + 1e-8 else np.sign(error)
        return action.tolist()

    def plan(self, targets, joints):
        """One pose step from supplied arm joints, on a kinematic copy of the model; the live simulation is untouched.

        The OSC goal is formed as ``encode`` forms it (fingerpad offset, then the controller's per-step output
        clipping) and solved by damped least squares within the joint limits. The arm is assumed to reach the
        solved joints in one step: no physics, contact, torque limits or OSC tracking lag. An unreachable goal
        returns the closest joints found, so the motion program's progress checks decide when a move stalls.
        ``joints`` maps an arm to its seven joint angles, or to an empty list for the measured joints.
        """
        import mujoco

        data = mujoco.MjData(self.model)
        data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = self.data.qpos, self.data.mocap_pos, self.data.mocap_quat
        starts = {}
        for name, entry in zip(self.names, self.entries):
            indexes = entry[5]._ref_joint_pos_indexes
            values = joints.get(name) or [float(self.data.qpos[index]) for index in indexes]
            if len(values) != len(indexes):
                raise RequestRejected(f"Pose plans need {len(indexes)} joints for {name}")
            data.qpos[indexes] = starts[name] = np.asarray(values, float)
        mujoco.mj_kinematics(self.model, data)
        result = {}
        for name, entry in zip(self.names, self.entries):
            controller, site, _, _, _, robot, _ = entry
            target = targets[name]
            centre, quat, _ = self._pose(entry, data)
            current_rotation, target_rotation = rotation(quat), rotation(target.pose.quaternion)
            local_offset = current_rotation.inv().apply(centre - data.site_xpos[site])
            native_target = np.asarray(target.pose.position) - target_rotation.apply(local_offset)
            delta = np.r_[native_target - data.site_xpos[site], (target_rotation * current_rotation.inv()).as_rotvec()]
            delta = np.clip(delta, np.asarray(controller.output_min), np.asarray(controller.output_max))
            goal_position = data.site_xpos[site] + delta[:3]
            goal_rotation = Rotation.from_rotvec(delta[3:]) * current_rotation
            joint_ids, dofs = robot._ref_joint_indexes, robot._ref_joint_vel_indexes
            low, high = self.model.jnt_range[joint_ids, 0], self.model.jnt_range[joint_ids, 1]
            q = starts[name].copy()
            position_error = rotation_error = np.zeros(3)
            for _ in range(IK_ITERATIONS):
                mujoco.mj_kinematics(self.model, data)
                position_error = goal_position - data.site_xpos[site]
                rotation_error = (goal_rotation * Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).inv()).as_rotvec()
                if (np.linalg.norm(position_error) < IK_POSITION_TOLERANCE_M
                        and np.linalg.norm(rotation_error) < IK_ROTATION_TOLERANCE_RAD):
                    break
                mujoco.mj_comPos(self.model, data)
                jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, data, jacp, jacr, site)
                jacobian = np.vstack([jacp[:, dofs], jacr[:, dofs]])
                error = np.r_[position_error, rotation_error]
                step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + IK_DAMPING ** 2 * np.eye(6), error)
                step *= min(1., IK_MAX_STEP_RAD / max(float(np.max(np.abs(step))), 1e-12))
                q = np.clip(q + step, low, high)
                data.qpos[robot._ref_joint_pos_indexes] = q
            mujoco.mj_kinematics(self.model, data)
            centre, quat, _ = self._pose(entry, data)
            result[name] = {
                "joints": q.tolist(), "joint_delta": (q - starts[name]).tolist(),
                "control_pose": {"position": centre.tolist(), "quaternion": quat},
                "residual_m": float(np.linalg.norm(position_error)), "residual_rad": float(np.linalg.norm(rotation_error)),
                "at_joint_limit": bool(np.any(np.isclose(q, low, atol=1e-4) | np.isclose(q, high, atol=1e-4))),
            }
        return result
