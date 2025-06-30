# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import os
from PIL import Image
import datetime
import numpy as np

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationData, RigidObjectData
from isaaclab.sensors import FrameTransformerData, ContactSensor, Camera, RayCasterCamera, TiledCamera
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.low_pass_filter import LowPassFilter
from collections import defaultdict
from .pretrained_model.vision_auto_encoder import Autoencoder

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ee_pose_w(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
)-> torch.Tensor:
    
    ee_data: FrameTransformerData = env.scene[ee_cfg.name].data

    ee_pos_w = ee_data.target_pos_w[:, 0, :]
    ee_quat_w = ee_data.target_quat_w[:, 0, :]

    return torch.cat((ee_pos_w, math_utils.normalize_and_unique_quat(ee_quat_w)), dim=1)


def ee_pose_b(env: ManagerBasedRLEnv,
              robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
              ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """The pose of the end-effector relative to the environment origins.
    ee_frame(sensor data, not accurate)"""

    robot_data: ArticulationData = env.scene[robot_cfg.name].data    

    base_pose_w = robot_data.root_state_w[:, :7]
    ee_pose_w_ = ee_pose_w(env, ee_cfg)

    # 3. Base 좌표계를 기준으로 Target의 상대 포즈 계산
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        base_pose_w[:, 0:3], base_pose_w[:, 3:7], ee_pose_w_[:, 0:3], ee_pose_w_[:, 3:7]
    )
    ee_pose_b = torch.cat((ee_pos_b, math_utils.normalize_and_unique_quat(ee_quat_b)), dim=1)

    return ee_pose_b


def ee_pos_b(env: ManagerBasedRLEnv,
             robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
             ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """The pose of the end-effector relative to the environment origins.
    ee_frame(sensor data, not accurate)"""

    return ee_pose_b(env, robot_cfg, ee_cfg)[:, 0:3]


def ee_quat_b(env: ManagerBasedRLEnv,
              robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
              ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """The pose of the end-effector relative to the environment origins.
    ee_frame(sensor data, not accurate)"""
    
    return ee_pose_b(env, robot_cfg, ee_cfg)[:, 3:7]


def ee_vel_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_name: str = str("panda_hand")
) -> torch.Tensor:
    """The linear velocity of the end-effector in the world frame."""
    robot_asset: Articulation = env.scene[robot_cfg.name]
    body_ids, body_names = robot_asset.find_bodies(ee_name)
    if len(body_ids) != 1:
        raise ValueError(
            f"Expected one match for the body name: {ee_name}. Found {len(body_ids)}: {body_names}."
        )
    ee_idx = body_ids[0]

    ee_linvel_w = robot_asset.data.body_state_w[:, ee_idx, 7:10]
    ee_angvel_w = robot_asset.data.body_state_w[:, ee_idx, 10:13]

    return torch.cat((ee_linvel_w, ee_angvel_w), dim=1)  # (B,6)


def ee_vel_b(env: ManagerBasedRLEnv,
             robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
             ee_name: str = str("panda_hand")) -> torch.Tensor:
    """The linear velocity of the end-effector in the base frame."""
    robot_asset: Articulation = env.scene[robot_cfg.name]

    base_quat_w = robot_asset.data.root_state_w[:, 3:7]
    base_quat_w_conj = math_utils.quat_conjugate(base_quat_w)

    ee_vel_w_ = ee_vel_w(env, robot_cfg, ee_name)
    ee_linvel_b = math_utils.quat_rotate(base_quat_w_conj, ee_vel_w_[:, 0:3])
    ee_angvel_b = math_utils.quat_rotate(base_quat_w_conj, ee_vel_w_[:, 3:6])

    ee_vel_b = torch.cat((ee_linvel_b, ee_angvel_b), dim=1)
    
    return ee_vel_b


def ee_linvel_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_name: str = str("panda_hand"),
) -> torch.Tensor:
    """The linear velocity of the end-effector in the base frame."""
    return ee_vel_b(env, robot_cfg, ee_name)[:, 0:3]


def ee_angvel_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_name: str = str("panda_hand"),
) -> torch.Tensor:
    """The linear velocity of the end-effector in the base frame."""
    return ee_vel_b(env, robot_cfg, ee_name)[:, 3:6]


def noisy_object_pose_b(env: ManagerBasedRLEnv,
                        noise_name: str = str("noisy_hole_pose_b")) -> torch.Tensor:

    if noise_name in env.shared_dict["events"]:
        noisy_object_pose_b = env.shared_dict["events"][noise_name]
    else:
        noisy_object_pose_b = torch.zeros((env.num_envs, 7), device=env.device)

    return noisy_object_pose_b

def noisy_object_pos_b(
    env: ManagerBasedRLEnv,
    noise_name: str = str("noisy_hole_pose_b"),
) -> torch.Tensor:

    return noisy_object_pose_b(env, noise_name)[:, 0:3]

def noisy_object_quat_b(
    env: ManagerBasedRLEnv,
    noise_name: str = str("noisy_hole_pose_b"),
) -> torch.Tensor:

    return noisy_object_pose_b(env, noise_name)[:, 3:7]


def object_pose_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("peg")
) -> torch.Tensor:
    """The pose of the object relative to the environment origins."""

    object_data: RigidObjectData = env.scene[object_cfg.name].data

    object_pos_w = object_data.root_state_w[:, 0:3]
    object_quat_w = object_data.root_state_w[:, 3:7]

    return torch.cat((object_pos_w, math_utils.normalize_and_unique_quat(object_quat_w)), dim=1)


def object_pose_b(
    env: ManagerBasedRLEnv,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """The pose of the object relative to the base frame (robot)."""

    base_data: ArticulationData = env.scene[base_cfg.name].data
    object_pose_w_ = object_pose_w(env, object_cfg)
    
    base_pose_w = base_data.root_state_w[:, :7]

    # 3. Base 좌표계를 기준으로 Target의 상대 포즈 계산
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        base_pose_w[:, 0:3], base_pose_w[:, 3:7], object_pose_w_[:, 0:3], object_pose_w_[:, 3:7]
    )
    return torch.cat((pos_b, math_utils.normalize_and_unique_quat(quat_b)), dim=1)


def object_pos_b(
    env: ManagerBasedRLEnv,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """The position of the object relative to the base frame (robot)."""
    return object_pose_b(env, base_cfg, object_cfg)[:, 0:3]

def object_quat_b(
    env: ManagerBasedRLEnv,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """The quaternion of the object relative to the base frame (robot)."""
    return object_pose_b(env, base_cfg, object_cfg)[:, 3:7]

def peg_from_ee(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position error between the end-effector and the peg."""
    peg_pose_w_ = object_pose_w(env, peg_cfg)
    ee_pose_w_ = ee_pose_w(env, ee_cfg)

    pos_error, quat_error = math_utils.subtract_frame_transforms(
        ee_pose_w_[:, :3], ee_pose_w_[:, 3:7], peg_pose_w_[:, :3], peg_pose_w_[:, 3:7]
    )
    peg_from_ee = torch.cat((pos_error, math_utils.normalize_and_unique_quat(quat_error)), dim=1)

    return peg_from_ee


def peg_pos_from_ee(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    
    return peg_from_ee(env, peg_cfg, ee_cfg)[:, 0:3]

def peg_quat_from_ee(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    
    return peg_from_ee(env, peg_cfg, ee_cfg)[:, 3:7]


def hole_from_peg(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """The position error between two objects."""
    peg_pose_w_ = object_pose_w(env, peg_cfg)
    hole_pose_w_ = object_pose_w(env, hole_cfg)

    pos_error, quat_error = math_utils.subtract_frame_transforms(
        peg_pose_w_[:, :3], peg_pose_w_[:, 3:7], hole_pose_w_[:, :3], hole_pose_w_[:, 3:7]
    )
    hole_from_peg = torch.cat((pos_error, math_utils.normalize_and_unique_quat(quat_error)), dim=1)
    return hole_from_peg

def hole_pos_from_peg(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """The position error between two objects."""
    return hole_from_peg(env, hole_cfg, peg_cfg)[:, 0:3]

def hole_quat_from_peg(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """The quaternion error between two objects."""
    return hole_from_peg(env, hole_cfg, peg_cfg)[:, 3:7]


def noisy_hole_from_peg(env: ManagerBasedRLEnv,
                        base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                        peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
                        noise_name: str = str("noisy_hole_pose_b"),) -> torch.Tensor:
    """The position error between two objects with Noise."""
    peg_pose_b_ = object_pose_b(env, base_cfg, peg_cfg)
    noisy_hole_pose_b = noisy_object_pose_b(env, noise_name)

    # 3. Base 좌표계를 기준으로 Target의 상대 포즈 계산
    noisy_pos_error, noisy_quat_error = math_utils.subtract_frame_transforms(
        peg_pose_b_[:, :3], peg_pose_b_[:, 3:7], noisy_hole_pose_b[:, :3], noisy_hole_pose_b[:, 3:7]
    )
    noisy_hole_from_peg = torch.cat((noisy_pos_error,
                                  math_utils.normalize_and_unique_quat(noisy_quat_error)), dim=1)

    return noisy_hole_from_peg


def noisy_hole_pos_from_peg(
    env: ManagerBasedRLEnv,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    noise_name: str = str("noisy_hole_pose_b"),
) -> torch.Tensor:
    """The position error between two objects with Noise."""
    return noisy_hole_from_peg(env, base_cfg, peg_cfg, noise_name)[:, 0:3]

def noisy_hole_quat_from_peg(
    env: ManagerBasedRLEnv,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    noise_name: str = str("noisy_hole_pose_b"),
) -> torch.Tensor:
    """The position error between two objects with Noise."""
    return noisy_hole_from_peg(env, base_cfg, peg_cfg, noise_name)[:, 3:7]

def noisy_hole_from_ee(
    env: ManagerBasedRLEnv,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    noise_name: str = str("noisy_hole_pose_b")
) -> torch.Tensor:
    """The position error between two objects with Noise."""
    ee_pose_b_ = ee_pose_b(env, base_cfg, ee_cfg)
    noisy_hole_pose_b = noisy_object_pose_b(env, noise_name)

    # 3. Base 좌표계를 기준으로 Target의 상대 포즈 계산
    noisy_pos_error, noisy_quat_error = math_utils.subtract_frame_transforms(
        ee_pose_b_[:, :3], ee_pose_b_[:, 3:7], noisy_hole_pose_b[:, :3], noisy_hole_pose_b[:, 3:7]
    )
    noisy_hole_from_ee = torch.cat((noisy_pos_error,
                                  math_utils.normalize_and_unique_quat(noisy_quat_error)), dim=1)

    return noisy_hole_from_ee

def noisy_hole_pos_from_ee(
    env: ManagerBasedRLEnv,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    noise_name: str = str("noisy_hole_pose_b")
) -> torch.Tensor:
    """The position error between two objects with Noise."""
    return noisy_hole_from_ee(env, base_cfg, ee_cfg, noise_name)[:, 0:3]

def noisy_hole_quat_from_ee(
    env: ManagerBasedRLEnv,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    noise_name: str = str("noisy_hole_pose_b")
) -> torch.Tensor:
    """The position error between two objects with Noise."""
    return noisy_hole_from_ee(env, base_cfg, ee_cfg, noise_name)[:, 3:7]


def net_contact_force_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_L"),
) -> torch.Tensor:
    """The net contact force on the object in the base frame, with EMA filtering."""
    # 1) raw force 계산
    robot_asset: Articulation = env.scene[robot_cfg.name]
    sensor_asset: ContactSensor = env.scene[contact_sensor_cfg.name]
    base_quat_w      = robot_asset.data.root_quat_w           # (B,4)
    net_force_w      = sensor_asset.data.net_forces_w[:, 0, :]# (B,3)
    base_quat_w_conj = math_utils.quat_conjugate(base_quat_w)
    raw_force_b      = math_utils.quat_rotate(base_quat_w_conj, net_force_w)  # (B,3)

    return raw_force_b


def net_contact_force_ee(env: ManagerBasedRLEnv,
                         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                         ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                         contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_L")
) -> torch.Tensor:
    """The net contact force on the end-effector."""
    net_contact_force_b_ = net_contact_force_b(env, robot_cfg, contact_sensor_cfg)
    ee_pose_b_ = ee_pose_b(env, robot_cfg, ee_cfg)
 
    net_contact_force_ee = math_utils.quat_rotate(
        math_utils.quat_conjugate(ee_pose_b_[:, 3:7]), net_contact_force_b_)

    return net_contact_force_ee


def net_contact_force_x_ee(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_L"),
) -> torch.Tensor:

    return net_contact_force_ee(env, robot_cfg, ee_cfg, contact_sensor_cfg)[:, 0].unsqueeze(-1)  # (B,1)


def FT_sensor_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body-frame wrench from FT sensor on the panda_hand joint, with EMA filtering."""

    robot_asset: Articulation = env.scene[robot_cfg.name]
    base_quat_w      = robot_asset.data.root_quat_w
    base_quat_w_conj = math_utils.quat_conjugate(base_quat_w)

    # get_measured_joint_forces() 의 shape: (B, num_joints, 6)
    # 마지막에서 두 번째 joint 가 panda_hand_joint 이므로 index -> 8
    panda_hand_joint_idx = 8
    wrench_w = robot_asset.get_measured_joint_forces()[0, :, panda_hand_joint_idx, :]  # (B,6)
    force_w  = wrench_w[:, 0:3]  # (B,3)
    torque_w = wrench_w[:, 3:6]  # (B,3)

    # world → body frame
    force_b  = math_utils.quat_rotate(base_quat_w_conj, force_w)
    torque_b = math_utils.quat_rotate(base_quat_w_conj, torque_w)
    raw_wrench_b = torch.cat((force_b, torque_b), dim=1)  # (B,6)

    return raw_wrench_b


def FT_sensor_ee(env: ManagerBasedRLEnv,
                 robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    # base frame 기준의 F/T 데이터
    force_torque_data_b = FT_sensor_b(env, robot_cfg)  # shape: (num_envs, 6)
    f_b = force_torque_data_b[:, :3]
    t_b = force_torque_data_b[:, 3:]

    # ee의 pose (base frame 기준)
    ee_pose_b_ = ee_pose_b(env, robot_cfg, ee_cfg)  # shape: (num_envs, 7)

    # 회전 행렬
    rot_b_to_ee = math_utils.quat_conjugate(ee_pose_b_[:, 3:7])  # world to ee frame

    # 각각 회전
    f_ee = math_utils.quat_rotate(rot_b_to_ee, f_b)
    t_ee = math_utils.quat_rotate(rot_b_to_ee, t_b)

    # 합치기
    force_torque_data_ee = torch.cat((f_ee, t_ee), dim=-1)

    return force_torque_data_ee


def F_sensor_ee(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return FT_sensor_ee(robot_cfg, ee_cfg)[:, 0:3]  # (B,3)

def T_sensor_ee(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return FT_sensor_ee(env, robot_cfg, ee_cfg)[:, 3:6]  # (B,3)


def F_filtered_sensor_ee(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Body-frame wrench from FT sensor on the panda_hand joint, with EMA filtering."""
    F_ee_raw = T_sensor_ee(env, robot_cfg, ee_cfg)  # (B,3)

    dt_sim = env.physics_dt   
    obs_store = env.shared_dict.setdefault("observations", defaultdict(dict))
    if "lp_force_filter" not in obs_store:
        obs_store["lp_force_filter"] = LowPassFilter(
            cutoff_hz=25.0,
            dt=dt_sim,
            init_mode="first",
        )

    lp_force_filter: LowPassFilter = obs_store["lp_force_filter"]

    filtered_force_ee = lp_force_filter(F_ee_raw)

    return filtered_force_ee


def T_filtered_sensor_ee(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Body-frame wrench from FT sensor on the panda_hand joint, with EMA filtering."""
    T_ee_raw = T_sensor_ee(env, robot_cfg, ee_cfg)  # (B,3)

    dt_sim = env.physics_dt   
    obs_store = env.shared_dict.setdefault("observations", defaultdict(dict))
    if "lp_torque_filter" not in obs_store:
        obs_store["lp_torque_filter"] = LowPassFilter(
            cutoff_hz=25.0,
            dt=dt_sim,
            init_mode="first",
        )

    lp_torque_filter: LowPassFilter = obs_store["lp_torque_filter"]

    filtered_torque_ee = lp_torque_filter(T_ee_raw)

    return filtered_torque_ee


def F_xy_filtered_sensor_ee(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    return F_filtered_sensor_ee(env, robot_cfg, ee_cfg)[:, 0:2]

def F_z_filtered_sensor_ee(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    return F_filtered_sensor_ee(env, robot_cfg, ee_cfg)[:, 2].unsqueeze(-1)  # (B,1)


# --- 수정된 save_image 함수 ---
def save_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    save_dir: str = "/home/home/ai_ws/src/image_feature_compress/data", # 기본 저장 경로 설정
    folder_name: str = "all_collected_images"
) -> torch.Tensor:
    """
    강화학습 환경의 센서에서 이미지를 받아 단일 폴더에 저장합니다.
    파일명에 스텝 번호와 환경 인덱스를 활용하여 고유성을 보장합니다.

    Args:
        env: 센서가 위치한 환경.
        sensor_cfg: 읽어올 센서 설정. 기본값은 SceneEntityCfg("tiled_camera").
        data_type: 센서에서 가져올 데이터 타입 ("rgb", "depth" 등). 기본값은 "rgb".
        save_dir: 이미지를 저장할 상위 디렉토리. 기본값은 "/home/home/ai_ws/src/image_feature_compress/data".

    Returns:
        강화학습 파이프라인을 위해 더미 텐서 (shape: (num_envs, 1))를 반환합니다.
        실제 이미지 데이터는 파일로 저장됩니다.
    """
    current_save_path = os.path.join(save_dir, data_type, folder_name)
    os.makedirs(current_save_path, exist_ok=True) # 폴더 없으면 생성

    # 강화학습 스텝 번호 가져오기 (안전하게)
    try:
        # env.episode_length_buf는 (num_envs,) 텐서이므로, 각 환경별 스텝을 가져옵니다.
        # 이 시점에는 env.episode_length_buf가 존재해야 합니다.
        episode_lengths_tensor = env.episode_length_buf
    except AttributeError:
        # 만약 RL 환경 초기화 시점에 save_image가 호출되고
        # episode_length_buf가 아직 존재하지 않는다면 (AttributeError 발생 시)
        # 모든 환경에 대해 스텝을 0으로 간주합니다.
        print("Warning: env.episode_length_buf not found. Using step 0 for all environments.")
        episode_lengths_tensor = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


    # 사용된 센서 객체 추출
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # 입력 이미지 획득
    images = sensor.data.output[data_type] # (num_envs, H, W, C)

    # 이미지 저장 루프
    for i, img_data in enumerate(images):
        # 각 환경의 현재 스텝 번호
        # episode_lengths_tensor는 (num_envs,) 텐서이므로 [i].item()으로 각 환경의 스텝을 가져옵니다.
        step = episode_lengths_tensor[i].item() 

        # 파일명 (예: rgb_env0_step00001.png)
        filename = f"{data_type}_env{i}_step{step:05d}.png" # 5자리로 패딩
        filepath = os.path.join(current_save_path, filename)
        
        if data_type == "rgb":
            img_np = img_data.detach().cpu().numpy().astype(np.uint8)
            try:
                img_pil = Image.fromarray(img_np)
                img_pil.save(filepath)
                # print(f"  Saved RGB image: {filepath}") # 로깅은 필요에 따라 활성화
            except Exception as e:
                print(f"Error saving RGB image {filepath}: {e}")

        elif "distance_to" in data_type or "depth" in data_type:
            # Depth 이미지 처리 (float 값 -> 시각화용 PNG & 원본 .npy)
            img_np = img_data.detach().cpu().numpy().squeeze() # (H, W, 1) -> (H, W)
            img_np[img_np == float("inf")] = 0 # 무한대 값 처리

            # 시각화용 그레이스케일 PNG 저장 (0-255로 스케일링)
            max_val = np.max(img_np)
            min_val = np.min(img_np)
            if max_val != min_val:
                img_display = ((img_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                img_display = np.zeros_like(img_np, dtype=np.uint8)
            
            try:
                img_pil = Image.fromarray(img_display, mode='L') # L for grayscale
                img_pil.save(filepath)
                # print(f"  Saved Depth image (normalized for display): {filepath}")
            except Exception as e:
                print(f"Error saving Depth image {filepath}: {e}")
            
            # 원본 float depth 데이터 저장 (.npy 파일)
            npy_filename = f"{data_type}_env{i}_step{step:05d}.npy"
            npy_filepath = os.path.join(current_save_path, npy_filename)
            np.save(npy_filepath, img_np)
            # print(f"  Saved raw Depth data: {npy_filepath}")

        else:
            print(f"Warning: Data type '{data_type}' not supported for direct image saving.")

    # 강화학습 파이프라인의 일관성을 위해 더미 텐서 반환
    return torch.zeros(env.num_envs, 1, device=env.device)



def image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone()


class compressed_image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.
    
    This term first extracts high-dimensional features from images using a pre-trained
    Theia model, and then compresses them into a low-dimensional latent vector
    using a trained Autoencoder. The final output is the compressed latent vector.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        # autoencoder_checkpoint_path는 필수 인자이므로 get() 대신 직접 접근하거나 에러를 발생시킵니다.
        if "autoencoder_checkpoint_path" not in cfg.params:
            raise ValueError("'autoencoder_checkpoint_path' must be provided in the observation term config.")
        self.autoencoder_checkpoint_path: str = cfg.params["autoencoder_checkpoint_path"]
        
        self.model_name: str = cfg.params.get("model_name", "theia-tiny-patch16-224-cddsv")
        self.model_device: str = cfg.params.get("model_device", env.device)
        self.sensor_cfg: SceneEntityCfg = cfg.params.get("sensor_cfg", SceneEntityCfg("tiled_camera"))
        
        # --- Theia 모델 준비 ---
        theia_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
        self.theia_model = theia_config["model"]()
        self.theia_inference_fn = theia_config["inference"]

        # --- Autoencoder 모델 준비 ---
        autoencoder_config = self._prepare_autoencoder_model(self.autoencoder_checkpoint_path, self.model_device)
        self.autoencoder_model = autoencoder_config["model"]()
        self.autoencoder_inference_fn = autoencoder_config["inference"]
        
        # Observation 공간의 차원은 autoencoder의 latent_dim과 같습니다.
        # latent_dim은 체크포인트에서 읽어옵니다.
        self._shape = (self.autoencoder_model.decoder[0].in_features,) # decoder의 첫 레이어 입력 차원 = latent_dim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        data_type: str,
        convert_perspective_to_orthogonal: bool,
        model_name: str,
        autoencoder_checkpoint_path: str,
        model_device: str,
    ) -> torch.Tensor:
        # Hydra가 넘겨준 sensor_cfg, data_type 등은 무시하고
        # 생성자에서 저장해둔 self.sensor_cfg, "rgb", self.autoencoder_* 속성을 사용합니다.

        # 1) 센서로부터 이미지 가져오기
        image_data = image(
            env,
            sensor_cfg=self.sensor_cfg,
            data_type="rgb",
            normalize=False,
        )

        # 2) Theia 모델로 고차원 특징 추출
        features = self.theia_inference_fn(self.theia_model, image_data)

        # 3) Autoencoder로 압축
        compressed_features = self.autoencoder_inference_fn(self.autoencoder_model, features)

        # 4) 환경 디바이스로 이동 후 반환
        return compressed_features.to(env.device)


    """
    Helper functions.
    """

    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Theia 모델과 추론 함수를 준비합니다."""
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Theia 모델을 로드, 평가 모드로 설정하고 지정된 디바이스로 보냅니다."""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Theia 모델을 사용하여 이미지에서 [CLS] 토큰 특징을 추출합니다."""
            # 이미지 텐서를 모델 디바이스로 이동하고 차원 순서를 (B, C, H, W)로 변경
            # image() 함수가 (B, H, W, C)를 반환한다고 가정
            image_proc = images.to(model_device).permute(0, 3, 1, 2).float() / 255.0
            
            # ImageNet 표준 정규화
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            with torch.no_grad():
                features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
                # [CLS] 토큰 (인덱스 0) 특징 벡터만 반환
                cls_token_features = features.last_hidden_state[:, 0]
            
            return cls_token_features

        return {"model": _load_model, "inference": _inference}


    def _prepare_autoencoder_model(self, checkpoint_path: str, model_device: str) -> dict:
        """학습된 Autoencoder 모델과 추론 함수를 준비합니다."""

        def _load_model() -> torch.nn.Module:
            """체크포인트에서 Autoencoder 모델을 로드합니다."""
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(model_device))

            # 체크포인트에서 차원 정보 로드
            input_dim = checkpoint.get('feature_dim')
            latent_dim = checkpoint.get('latent_dim')

            if input_dim is None or latent_dim is None:
                raise ValueError("Checkpoint must contain 'feature_dim' and 'latent_dim' information.")

            # Autoencoder 모델 초기화 및 학습된 가중치 로드
            autoencoder = Autoencoder(input_dim, latent_dim)
            autoencoder.load_state_dict(checkpoint['model_state_dict'])
            autoencoder.eval() # 평가 모드로 설정

            return autoencoder.to(model_device)

        def _inference(model, features: torch.Tensor) -> torch.Tensor:
            """Autoencoder의 encoder를 사용하여 특징을 압축합니다."""
            # 입력 특징을 모델 디바이스로 이동
            features = features.to(model_device)

            # print(f"features : {features.shape}")
             
            with torch.no_grad():
                compressed_features = model.encode(features)

            # print(f"compressed_features : {compressed_features.shape}")
            
            return compressed_features

        # reset 함수는 Autoencoder에 필요 없으므로 반환하지 않음
        return {"model": _load_model, "inference": _inference}