from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import numpy as np
import math

if TYPE_CHECKING:
    from .task_space_impedance_cfg import TaskSpaceImpedanceControllerCfg


class TaskSpaceImpedanceController:
    def __init__(self, cfg: TaskSpaceImpedanceControllerCfg, num_envs: int, num_dof: int, device: str):
        
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_dof = num_dof
        self._device = device

        self._task_space_target = torch.zeros((self.num_envs, 7), device=self._device)  # 3 for position, 4 for quaternion

        self._p_gains = torch.zeros((self.num_envs, 6), device=self._device)
        self._p_gains[:] = torch.tensor(self.cfg.stiffness, device=self._device)
        self._d_gains = torch.zeros((self.num_envs, 6), device=self._device)
        self._d_gains[:] = 2 * torch.sqrt(self._p_gains) * torch.tensor(self.cfg.damping_ratio, device=self._device)

        self._task_wrench = torch.zeros(self.num_envs, 6, device=self._device)
        self._desired_torques = torch.zeros(self.num_envs, self.num_dof, device=self._device)


    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.cfg.is_restricted:
            return 4 # (dx, dy, dz, dyaw)
        else:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)


    def initialize(self):
        pass


    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        pass


    def set_command(
        self, command: torch.Tensor
    ) -> None:
        # store command
        self._task_space_target[:] = command  # position(x, y, z), quaternion(w, x, y, z)

    
    def compute(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        current_ee_pose_b: torch.Tensor,
        current_ee_vel_b: torch.Tensor,
        jacobian_b: torch.Tensor,
        arm_mass_matrix: torch.Tensor,
        gravity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self._task_wrench = torch.zeros((self.num_envs, 6), device=self._device)
        pos_error, axis_angle_error = math_utils.compute_pose_error(
            current_ee_pose_b[:, :3],
            current_ee_pose_b[:, 3:7],
            self._task_space_target[:, :3],
            self._task_space_target[:, 3:7],
            rot_error_type="axis_angle"
        )
        delta_ee_pose = torch.cat((pos_error, axis_angle_error), dim=1)

        # Set tau = k_p * task_pos_error - k_d * task_vel_error
        task_wrench_motion = self._apply_task_space_gains(
            delta_ee_pose=delta_ee_pose,
            ee_linvel=current_ee_vel_b[:, :3],
            ee_angvel=current_ee_vel_b[:, 3:6],
            p_gains=self._p_gains,
            d_gains=self._d_gains
        )
        self._task_wrench = self._task_wrench + task_wrench_motion

        jacobian_T = torch.transpose(jacobian_b, dim0=1, dim1=2)
        self._desired_torques[:, 0:7] = (jacobian_T @ self._task_wrench.unsqueeze(-1)).squeeze(-1)

        # # adapted from https://gitlab-master.nvidia.com/carbon-gym/carbgym/-/blob/b4bbc66f4e31b1a1bee61dbaafc0766bbfbf0f58/python/examples/franka_cube_ik_osc.py#L70-78
        # # roboticsproceedings.org/rss07/p31.pdf
        arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
        arm_mass_matrix_task = torch.inverse(
            jacobian_b @ arm_mass_matrix_inv @ jacobian_T
        )
        j_eef_inv = arm_mass_matrix_task @ jacobian_b @ arm_mass_matrix_inv
        default_dof_pos_tensor = torch.tensor(self.cfg.default_dof_pos_tensor, device=self._device).repeat((self.num_envs, 1))

        # nullspace computation
        distance_to_default_dof_pos = default_dof_pos_tensor - joint_pos[:, 0:7]
        distance_to_default_dof_pos = (distance_to_default_dof_pos + np.pi) % (
            2 * np.pi
        ) - np.pi # normalize to [-pi, pi]
        u_null = self.cfg.kp_null * distance_to_default_dof_pos - self.cfg.kd_null * joint_vel[:, :7]
        u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
        torque_null = (torch.eye(7, device=self._device).unsqueeze(0) - torch.transpose(jacobian_b, 1, 2) @ j_eef_inv) @ u_null
        self._desired_torques[:, 0:7] += torque_null.squeeze(-1)

        # Gravity compensation
        if self.cfg.gravity_compensation:
            self._desired_torques += gravity

        self._desired_torques = torch.clamp(self._desired_torques, min=-100.0, max=100.0)

        return self._desired_torques, self._task_wrench

        
    @staticmethod
    def _apply_task_space_gains(
        delta_ee_pose: torch.Tensor,
        ee_linvel: torch.Tensor,
        ee_angvel: torch.Tensor,
        p_gains: torch.Tensor,
        d_gains: torch.Tensor
    ) -> torch.Tensor:
        
        task_wrench = torch.zeros_like(delta_ee_pose)

        # Apply gains to linear error components
        lin_error = delta_ee_pose[:, :3]
        ang_error = delta_ee_pose[:, 3:6]

        # Apply gains to rotational error components
        task_wrench[:, :3] = p_gains[:, :3] * lin_error + d_gains[:, :3] * (0.0 - ee_linvel)
        task_wrench[:, 3:] = p_gains[:, 3:6] * ang_error + d_gains[:, 3:6] * (0.0 - ee_angvel)

        return task_wrench
    
    
    @staticmethod
    def get_pose_error(
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        target_ee_pos: torch.Tensor,
        target_ee_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        pos_error = target_ee_pos - ee_pos

        quat_dot = (target_ee_quat * ee_quat).sum(dim=1, keepdim=True)
        target_ee_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, target_ee_quat, -target_ee_quat
        )

        ee_quat_norm = math_utils.quat_mul(
            ee_quat, math_utils.quat_conjugate(ee_quat)
        )[
            :, 0
        ] # scalar component
        ee_quat_inv = math_utils.quat_conjugate(
            ee_quat
        ) / ee_quat_norm.unsqueeze(1)
        quat_error = math_utils.quat_mul(target_ee_quat, ee_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = math_utils.axis_angle_from_quat(quat_error)

        return pos_error, axis_angle_error

    @staticmethod
    def get_delta_dof_pos(
        delta_pose: torch.Tensor,
        ik_method: str,
        jacobian: torch.Tensor,
        device: str
    ) -> torch.Tensor:
        """Get delta Franka DOF position from delta pose using specified IK method."""
        # References:
        # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
        # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

        if ik_method == "pinv":  # Jacobian pseudoinverse
            k_val = 1.0
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == "trans":  # Jacobian transpose
            k_val = 1.0
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == "dls":  # damped least squares (Levenberg-Marquardt)
            lambda_val = 0.1  # 0.1
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
            delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == "svd":  # adaptive SVD
            k_val = 1.0
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            min_singular_value = 1.0e-5
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        return delta_dof_pos