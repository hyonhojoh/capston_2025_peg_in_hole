# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat
from isaacsim.core.utils.prims import set_prim_property
from pxr import Gf
import wandb
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_rew(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a tensor of zeros for the reward function."""
    return torch.zeros(env.num_envs, device=env.device)

def compute_intermediate_values(env: ManagerBasedRLEnv,
                                hole_height: float = float(0.05),
                                robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
                                peg_cfg: SceneEntityCfg = SceneEntityCfg("peg")) -> torch.Tensor:
    """Compute intermediate values for the reward function.

    This function computes the intermediate values for the reward function. It is useful for debugging and visualization.
    """
    '''Scene Asset'''
    robot_asset: Articulation = env.scene[robot_cfg.name]
    hole_asset: Articulation | RigidObject = env.scene[hole_cfg.name]
    peg_asset: Articulation | RigidObject = env.scene[peg_cfg.name]

    '''World Frame Data'''
    base_pose_w = robot_asset.data.root_state_w[:, :7]
    peg_pose_w = peg_asset.data.root_state_w[:, :7]
    hole_pose_w = hole_asset.data.root_state_w[:, :7]

    '''Base Frame Data'''
    peg_pos_b, peg_quat_b = math_utils.subtract_frame_transforms(
        base_pose_w[:, :3], base_pose_w[:, 3:7], peg_pose_w[:, :3], peg_pose_w[:, 3:7]
    )
    hole_pos_b, hole_quat_b = math_utils.subtract_frame_transforms(
        base_pose_w[:, :3], base_pose_w[:, 3:7], hole_pose_w[:, :3], hole_pose_w[:, 3:7]
    )
    hole_top_pos_b = hole_pos_b.clone() + torch.tensor([0.0, 0.0, hole_height], device=env.device).expand(env.num_envs, -1)

    ''' Compute Error "from End-Effector to Hole" '''
    pos_error, quat_error = math_utils.compute_pose_error(peg_pos_b, peg_quat_b,
                                                          hole_top_pos_b, hole_quat_b,
                                                          rot_error_type="quat")
    # quat_error = math_utils.normalize_and_unique_quat(quat_error)

    z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=peg_pos_b.device).repeat(peg_pos_b.shape[0], 1)
    peg_z_axis_world = math_utils.quat_apply(peg_quat_b, z_axis_local)
    hole_z_axis_world = math_utils.quat_apply(hole_quat_b, z_axis_local)
    dot_product = torch.sum(peg_z_axis_world * hole_z_axis_world, dim=-1)
    d_ori_tilt = torch.arccos(torch.clamp(dot_product, -1.0, 1.0))

    # abs value
    dist_tip_to_opening_xy = torch.norm(pos_error[:, :2], dim=1)
    dist_tip_to_opening = torch.norm(pos_error, dim=1)
    d_ori = 2 * torch.arccos(quat_error[:, 0])
    z_depth = hole_top_pos_b[:, 2] - peg_pos_b[:, 2]
    z = peg_pos_b[:, 2]
    
    gripper_joint_pos = robot_asset.data.joint_pos[:, -1] # average: 0.0038
    
    '''Condition'''
    success_condition = (dist_tip_to_opening_xy < 0.003) & (z_depth > 0.001)
    final_success_condition = (dist_tip_to_opening_xy < 0.003) & (z_depth > (hole_height / 2)) # 절반은 들어가야 최종 성공
    # is_aligned = (dist_tip_to_opening_xy < 0.01) & (d_ori < 0.1)
    is_missing = gripper_joint_pos < 0.002

    success_env_ids = torch.nonzero(success_condition).squeeze(-1)
    nonsuccess_env_ids = torch.nonzero(~success_condition).squeeze(-1)

    # Store each variable value in a dictionary
    env.shared_dict["rewards"] = {
        "hole_pos_b": hole_pos_b,  # Position of the hole
        "hole_top_pos_b": hole_top_pos_b,  # Top position of the hole
        "peg_pos_b": peg_pos_b,  # Position of the peg
        "dist_tip_to_opening": dist_tip_to_opening,  # Distance from the tip to the opening
        "dist_tip_to_opening_xy": dist_tip_to_opening_xy,  # Distance in the XY plane from the tip to the opening
        "d_ori": d_ori,  # Rotation error angle
        "d_ori_tilt": d_ori_tilt,
        "z_depth": z_depth,  # Difference in top Z position
        "z": z,  # Z position of the peg
        "hole_height": hole_height,  # Height of the object
        "gripper_joint_pos": gripper_joint_pos,  # Position of the gripper joint
        "is_missing": is_missing,  # Condition for missing peg
        
        "success_condition": success_condition,  # Condition for success
        "final_success_condition": final_success_condition,  # Final condition for success
        # "is_aligned": is_aligned,  # Condition for alignment
        "success_env_ids": success_env_ids,  # IDs of successful environments
        "nonsuccess_env_ids": nonsuccess_env_ids,  # IDs of non-successful environments
    }

    for idx in success_env_ids:
        env_prim_path = f"/World/envs/env_{int(idx)}"
        table_prim_path = env_prim_path + "/Table"
        shader_prim_path = table_prim_path + "/geometry/material/Shader"

        # Shader의 diffuseColor 속성 변경
        set_prim_property(
            prim_path=shader_prim_path,  # Shader의 경로
            property_name="inputs:diffuseColor",  # Shader의 diffuseColor 속성
            property_value=Gf.Vec3f(0.0, 0.0, 1.0)  # 파란색
        )
    
    for idx in nonsuccess_env_ids:
        env_prim_path = f"/World/envs/env_{int(idx)}"
        table_prim_path = env_prim_path + "/Table"
        shader_prim_path = table_prim_path + "/geometry/material/Shader"

        # Shader의 diffuseColor 속성 변경
        set_prim_property(
            prim_path=shader_prim_path,  # Shader의 경로
            property_name="inputs:diffuseColor",  # Shader의 diffuseColor 속성
            property_value=Gf.Vec3f(0.5, 0.5, 0.5)  # 파란색
        )

    # print(f"success_rate: {success_condition.float().mean().item() * 100}%")

    return torch.zeros(env.num_envs, device=env.device) # Just For calculating intermediate values



def comprehensive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    # 중간 값 가져오기
    dist_xy = env.shared_dict["rewards"]["dist_tip_to_opening_xy"]
    z_depth = env.shared_dict["rewards"]["z_depth"]
    # z = env.shared_dict["rewards"]["z"]
    # d_ori = env.shared_dict["rewards"]["d_ori"]
    d_ori_tilt = env.shared_dict["rewards"]["d_ori_tilt"]
    hole_height = env.shared_dict["rewards"]["hole_height"]
    success_condition = env.shared_dict["rewards"]["success_condition"]

    # --- 정렬 보상 (항상 계산) ---
    xy_rew = torch.exp(-15.0 * dist_xy)  # XY 평면에서의 거리 보상 # 1.0
    z_rew = torch.exp(-10.0 * torch.abs(z_depth)) # Z 깊이 보상
    z_rew = torch.where((dist_xy < 0.01) * (z_depth > -0.003), 1.0 + torch.exp(-20.0 * torch.abs(z_depth + 0.001)), z_rew) # Z 깊이가 음수일 때 보상 감소
    ori_rew = torch.exp(-5.0 * d_ori_tilt)
    align_reward = xy_rew + ori_rew
    
    pre_insertion_reward = align_reward + z_rew # 정렬 전 보상
    # --- 삽입 보상 (정렬 후에만 활성화될 보상) ---
    insert_progress = torch.where(success_condition, z_depth / hole_height, 0.0)
    post_insertion_reward = 5.0 + 3.0 * insert_progress # 정렬 후 보상. 기본 보너스가 2.0 더 높음.
    
    reward = torch.where(success_condition, post_insertion_reward, pre_insertion_reward)
    
    env.shared_dict["rewards"]["xy_rew"] = xy_rew
    env.shared_dict["rewards"]["z_rew"] = z_rew
    env.shared_dict["rewards"]["ori_rew"] = ori_rew
    env.shared_dict["rewards"]["insert_progress"] = insert_progress
    env.shared_dict["rewards"]["comprehensive_reward"] = reward
    
    return reward


def action_smoothness_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = str("arm_action")
) -> torch.Tensor:
    last_actions = env.action_manager.get_term(action_name).raw_actions
    actions = env.action_manager.action

    pen = - torch.norm(actions - last_actions, dim=1)
    env.shared_dict["rewards"]["action_smoothness_penalty"] = pen

    return pen



def missing_peg_penalty(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    gripper_joint_pos = env.shared_dict["rewards"]["gripper_joint_pos"]

    # gripper_joint_pos < threshold => Gripper is close -> missing peg -< penalty
    missing_peg_penalty = torch.where(gripper_joint_pos < threshold, -2.0, 0.0)
    
    env.shared_dict["rewards"]["missing_peg_penalty"] = missing_peg_penalty

    return missing_peg_penalty


def fast_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    success_condition = env.shared_dict["rewards"]["success_condition"]
    fast_success_reward = torch.clamp(torch.where(success_condition,
                                                  1.0 - (env.episode_length_buf / env.max_episode_length), 0.0),
                                      0.0, 1.0)
    env.shared_dict["rewards"]["fast_success_reward"] = fast_success_reward

    return fast_success_reward


def final_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    final_success_condition = env.shared_dict["rewards"]["final_success_condition"]
    final_success_reward = torch.where(final_success_condition, 5.0, 0.0)
    env.shared_dict["rewards"]["final_success_reward"] = final_success_reward

    return final_success_reward


def wandb_log(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Extract required variables from the shared_dict
    dist_tip_to_opening_xy = env.shared_dict["rewards"]["dist_tip_to_opening_xy"]
    z = env.shared_dict["rewards"]["z"]
    d_ori = env.shared_dict["rewards"]["d_ori_tilt"]
    success_condition = env.shared_dict["rewards"]["success_condition"]
    final_success_condition = env.shared_dict["rewards"]["final_success_condition"]

    comprehensive_reward = env.shared_dict["rewards"]["comprehensive_reward"]
    action_smoothness_penalty = 0.01 * env.shared_dict["rewards"]["action_smoothness_penalty"]
    
    fast_success_reward = env.shared_dict["rewards"]["fast_success_reward"]
    final_success_reward = env.shared_dict["rewards"]["final_success_reward"]

    # missing_peg_penalty = env.shared_dict["rewards"]["missing_peg_penalty"]
    xy_rew = env.shared_dict["rewards"]["xy_rew"]
    z_rew = env.shared_dict["rewards"]["z_rew"]
    ori_rew = env.shared_dict["rewards"]["ori_rew"]
    insert_progress = env.shared_dict["rewards"]["insert_progress"]

    total_reward = (comprehensive_reward + \
                    action_smoothness_penalty + \
                    fast_success_reward + \
                    final_success_reward)
                    # (missing_peg_penalty)
                    

    # Compute metrics
    metrics = {
        "error/xy (mm)": dist_tip_to_opening_xy.mean().item() * 1e3,
        "error/z (mm)": z.mean().item() * 1e3,
        "error/ori (deg)": d_ori.mean().item() * (180 / torch.pi),

        "condition/success (%)": success_condition.float().mean().item() * 100,
        "condition/final_success (%)": final_success_condition.float().mean().item() * 100,
        "condition/missing": env.shared_dict["rewards"]["is_missing"].float().mean().item() * 100,

        "reward/xy_rew": xy_rew.mean().item(),
        "reward/z_rew": z_rew.mean().item(),
        "reward/ori_rew": ori_rew.mean().item(),
        "reward/insert_progress (%)": insert_progress.mean().item() * 100,
        "reward/fast_success": fast_success_reward.mean().item(),
        "reward/final_success": final_success_reward.mean().item(),

        # "penalty/missing_peg": missing_peg_penalty.mean().item(),
        "penalty/action_smoothness": action_smoothness_penalty.mean().item(),

        "reward/total": total_reward.mean().item(),
    }

    # print(f"success_rate : {final_success_condition.float().mean().item() * 100}")

    # Log metrics using wandb
    wandb.log(metrics)

    return torch.zeros(env.num_envs, device=env.device) # Just For calculating intermediate values