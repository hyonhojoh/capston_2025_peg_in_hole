from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.managers import SceneEntityCfg
from pxr import Gf
from isaacsim.core.utils.prims import set_prim_property
import matplotlib.pyplot as plt
import numpy as np
from isaaclab.utils.low_pass_filter import LowPassFilter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



import math
import numpy as np
import matplotlib.pyplot as plt

def set_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    friction: float = 0.75,
):
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = friction  # Static friction.
    materials[..., 1] = friction  # Dynamic friction.
    env_ids_ = torch.arange(env.num_envs, device="cpu")
    asset.root_physx_view.set_material_properties(materials, env_ids_)

def set_body_inertia(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    inertias = asset.root_physx_view.get_inertias()
    offset = torch.zeros_like(inertias)
    offset[:, :, [0, 4, 8]] += 0.01
    new_inertias = inertias + offset
    asset.root_physx_view.set_inertias(new_inertias, torch.arange(env.num_envs))
    
def move_tcp_using_controller(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pos_offset: tuple[float, float, float] | None,
    quat_offset: tuple[float, float, float, float] | None,
    noise: dict[str, tuple[float, float]] | None,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_name: str = "peg",
    target_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    sim_step: int = 500,
    traj_steps: int = 500,
):
    # 1. Target 및 Base Assets 가져오기
    target_asset: RigidObject = env.scene[target_cfg.name]
    base_asset: Articulation = env.scene[base_cfg.name]

    body_ids, body_names = base_asset.find_bodies(ee_name)
    if len(body_ids) != 1:
        raise ValueError(
            f"Expected one match for the body name: {ee_name}. Found {len(body_ids)}: {body_names}."
        )
    ee_idx = body_ids[0]

    # 2. Base와 Target의 현재 포즈 계산 (월드 좌표계)
    base_pose_w = base_asset.data.root_state_w[:, :7].clone()
    target_pose_w = target_asset.data.root_state_w[:, :7].clone()
    
    # 3. Base 좌표계를 기준으로 Target의 상대 포즈 계산
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        base_pose_w[:, 0:3], base_pose_w[:, 3:7],
        target_pose_w[:, 0:3], target_pose_w[:, 3:7]
    )

    # 4. Offset 적용
    # 4.1 위치 offset
    if pos_offset is not None:
        pos_offset_tensor = torch.tensor(pos_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    else:
        pos_offset_tensor = torch.zeros_like(target_pos_b)

    # 4.2 쿼터니언 offset
    if quat_offset is not None:
        quat_offset_tensor = torch.tensor(quat_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    else:
        quat_offset_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

    # 4.3 Offset 적용: base 및 offset을 결합
    target_pos_b, target_quat_b = math_utils.combine_frame_transforms(
        target_pos_b, target_quat_b, pos_offset_tensor, quat_offset_tensor
    )

    # 5. Noise 적용
    range_list = [noise.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=target_asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (env.num_envs, 6), device=target_asset.device)
    noisy_target_pos_b = target_pos_b + rand_samples[:, 0:3]
    
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    noisy_target_quat_b = math_utils.quat_mul(target_quat_b, orientations_delta)

    # 최종 목표 pose (position + axis-angle)
    noisy_target_axis_angle_b = math_utils.axis_angle_from_quat(noisy_target_quat_b)
    target_pose_cmd = torch.cat((noisy_target_pos_b, noisy_target_axis_angle_b), dim=1)

    # 6. Controller 관련 설정
    arm_action = env.action_manager.get_term("arm_action")
    arm_action.cfg.controller.use_relative_mode = False
    arm_action.cfg.scale = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    arm_action.reset()

    # 현재 TCP pose 계산: base_asset에서 지정된 ee body의 정보를 이용
    try:
        start_pose_w = base_asset.data.body_state_w[:, ee_idx, :7].clone()
        start_pos_b, start_quat_b = math_utils.subtract_frame_transforms(
            base_pose_w[:, 0:3], base_pose_w[:, 3:7],
            start_pose_w[:, 0:3], start_pose_w[:, 3:7]
        )
        start_pose_b = torch.cat((start_pos_b, start_quat_b), dim=1)
    except AttributeError:
        start_pose_b = target_pose_cmd.clone()

    # 초기 (start) 위치 및 회전
    start_pos = start_pose_b[:, :3]      # shape: [num_envs, 3]
    start_quat = start_pose_b[:, 3:7]    # shape: [num_envs, 4]

    # 목표 (end) 위치 및 회전
    end_pos = noisy_target_pos_b        # shape: [num_envs, 3]
    end_quat = noisy_target_quat_b      # shape: [num_envs, 4]

    # Trajectory 구간 설정
    sim_steps_per_waypoint = sim_step // traj_steps
    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()

    # 에러 기록용 리스트 (환경 0만 기록 예시)
    step_indices = []
    desired_pos_record = []  
    actual_pos_record = []   
    orientation_error_record = []
    global_step = 0

    # 7. Trajectory 생성 및 Controller에 명령 전달
    #    - 위치: 3차 보간 (start, end에서 속도=0)
    #      pos(t) = start + (end - start)*(3*t^2 - 2*t^3), t in [0..1]
    #    - 회전: SLERP를 그대로 사용 (3차 보간과 다른 개념)
    for i in range(traj_steps + 1):
        # 보간 계수 t in [0..1]
        tau = i / float(traj_steps)

        # === Cubic interpolation for position ===
        #   0 <= tau <= 1
        #   pos(tau) = start + (end - start)*(3*tau^2 - 2*tau^3)
        cubic_val = 3.0 * tau**2 - 2.0 * tau**3
        interp_pos = start_pos + (end_pos - start_pos) * cubic_val  # [num_envs, 3]

        # === SLERP for orientation ===
        #   각 env별로 start_quat[j] -> end_quat[j]를 tau로 보간
        interp_quat_list = []
        for j in range(env.num_envs):
            q_interp = math_utils.quat_slerp(start_quat[j], end_quat[j], tau)
            interp_quat_list.append(q_interp.unsqueeze(0))
        interp_quat = torch.cat(interp_quat_list, dim=0)  # [num_envs, 4]
        interp_axis_angle = math_utils.axis_angle_from_quat(interp_quat)

        # 보간된 pose 형성 (position + axis-angle)
        # interp_pose_b = torch.cat((interp_pos, interp_axis_angle), dim=1)
        interp_pose_b = torch.cat((interp_pos, interp_quat), dim=1)

        # 8. 최종 목표 pose 이벤트 기록
        env.shared_dict["events"]["noisy_hole_pose_b"] = torch.cat(
            (noisy_target_pos_b, math_utils.normalize_and_unique_quat(noisy_target_quat_b)), dim=1
        )

        # 해당 waypoint 이후, sim_steps_per_waypoint 만큼 시뮬레이션 step 실행
        for _ in range(sim_steps_per_waypoint):
            arm_action.apply_actions()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
                env.sim.render()
            env.scene.update(dt=env.physics_dt)

            # # 현재 end-effector pose (world 좌표계) 추출
            # current_pose_w = base_asset.data.body_state_w[:, ee_idx, :7].clone()
            # current_pos_b, current_quat_b = math_utils.subtract_frame_transforms(
            #     base_pose_w[:, 0:3], base_pose_w[:, 3:7],
            #     current_pose_w[:, 0:3], current_pose_w[:, 3:7]
            # )

            # # 에러 기록 (환경 0 기준)
            # desired_p = interp_pos[0].detach().cpu().numpy()  
            # actual_p  = current_pos_b[0].detach().cpu().numpy()
            # desired_pos_record.append(desired_p)
            # actual_pos_record.append(actual_p)

            # # 회전 에러: desired vs. actual (환경 0)
            # desired_q = interp_quat[0]
            # actual_q = current_quat_b[0]
            # dot_val = torch.abs(torch.dot(desired_q, actual_q))
            # dot_val = torch.clamp(dot_val, 0.0, 1.0)
            # error_angle_rad = torch.acos(dot_val)
            # error_angle_deg = error_angle_rad.item() * 180.0 / np.pi
            # orientation_error_record.append(error_angle_deg)

            # step_indices.append(global_step)
            # global_step += 1

    arm_action.cfg.controller.use_relative_mode = True
    arm_action.cfg.scale = (0.005, 0.005, 0.005, 0.01, 0.01, 0.01)
    arm_action.reset() # refresh command type: 7 -> 6

    # # 9. Plot (환경 0만 예시)
    # desired_pos_record = np.array(desired_pos_record)
    # actual_pos_record = np.array(actual_pos_record)
    # step_indices = np.array(step_indices)

    # plt.figure(figsize=(10, 5))
    # plt.plot(step_indices, desired_pos_record[:, 0], "r--", label="Desired X")
    # plt.plot(step_indices, actual_pos_record[:, 0], "r-", label="Actual X")
    # plt.plot(step_indices, desired_pos_record[:, 1], "g--", label="Desired Y")
    # plt.plot(step_indices, actual_pos_record[:, 1], "g-", label="Actual Y")
    # plt.plot(step_indices, desired_pos_record[:, 2], "b--", label="Desired Z")
    # plt.plot(step_indices, actual_pos_record[:, 2], "b-", label="Actual Z")
    # plt.xlabel("Simulation Steps")
    # plt.ylabel("Position (m)")
    # plt.title("Desired vs Actual End-Effector Position (Cubic Interpolation)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(10, 4))
    # plt.plot(step_indices, orientation_error_record, "m-", label="Orientation Error")
    # plt.xlabel("Simulation Steps")
    # plt.ylabel("Orientation Error (deg)")
    # plt.title("Orientation Tracking Error")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


def set_tcp_pose_to_noisy_hole(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pos_offset: tuple[float, float, float] | None,
    quat_offset: tuple[float, float, float, float] | None,
    noise: dict[str, tuple[float, float]] | None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    ee_name: str = "peg",
):
    robot_asset: Articulation = env.scene[robot_cfg.name]
    hole_asset: RigidObject = env.scene[hole_cfg.name]

    joint_ids, joint_names = robot_asset.find_joints(["panda_joint.*"])
    body_ids, body_names = robot_asset.find_bodies(ee_name)
    if len(body_ids) != 1:
        raise ValueError(
            f"Expected one match for the body name: {ee_name}. Found {len(body_ids)}: {body_names}."
        )
    ee_idx = body_ids[0]
    jacobi_body_idx = ee_idx - 1
    jacobi_joint_ids = joint_ids

    # 2. Base와 Target의 현재 포즈 계산 (월드 좌표계)
    base_pose_w = robot_asset.data.root_state_w[:, :7].clone()
    hole_pose_w = hole_asset.data.root_state_w[:, :7].clone()
    
    # 3. Base 좌표계를 기준으로 Target의 상대 포즈 계산
    hole_pos_b, hole_quat_b = math_utils.subtract_frame_transforms(
        base_pose_w[:, 0:3], base_pose_w[:, 3:7],
        hole_pose_w[:, 0:3], hole_pose_w[:, 3:7]
    )

    # 4. Offset 적용
    # 4.1 위치 offset
    if pos_offset is not None:
        pos_offset_tensor = torch.tensor(pos_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    else:
        pos_offset_tensor = torch.zeros_like(hole_pos_b)

    # 4.2 쿼터니언 offset
    if quat_offset is not None:
        quat_offset_tensor = torch.tensor(quat_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    else:
        quat_offset_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

    # 4.3 Offset 적용: base 및 offset을 결합
    target_pos_b, target_quat_b = math_utils.combine_frame_transforms(
        hole_pos_b, hole_quat_b, pos_offset_tensor, quat_offset_tensor
    )

    # 5. Noise 적용
    range_list = [noise.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=hole_asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (env.num_envs, 6), device=env.device)
    noisy_target_pos_b = target_pos_b + rand_samples[:, 0:3]
    
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    noisy_target_quat_b = math_utils.quat_mul(target_quat_b, orientations_delta)

    env.shared_dict["events"]["noisy_hole_pose_b"] = torch.cat(
        (noisy_target_pos_b, math_utils.normalize_and_unique_quat(noisy_target_quat_b)), dim=1
    )

    # Compute IK
    joint_pos = robot_asset.data.joint_pos.clone()
    joint_pos_limits = robot_asset.data.soft_joint_pos_limits[env_ids]
    arm_action = env.action_manager.get_term("arm_action")
    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    max_iters = 10
    
    for _ in range(max_iters):
        # 1. 현재 ee_pose
        ee_pose_w = robot_asset.data.body_state_w[:, ee_idx, :7].clone()
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            base_pose_w[:, :3], base_pose_w[:, 3:],
            ee_pose_w[:, :3], ee_pose_w[:, 3:]
        )

        # 2. error
        pos_err, ang_err = math_utils.compute_pose_error(
            ee_pos_b, ee_quat_b,
            noisy_target_pos_b, noisy_target_quat_b,
            rot_error_type="axis_angle"
        )
        pose_error = torch.cat((pos_err, ang_err), dim=1)

        # 3. 재계산된 Jacobian
        jacobian_w = robot_asset.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids]
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(robot_asset.data.root_quat_w))
        jacobian_b = torch.zeros_like(jacobian_w)
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        # 4. SVD 방식 IK
        U, S, Vh = torch.linalg.svd(jacobian_b, full_matrices=False)
        S_inv = 1.0 / S
        S_inv = torch.where(S > 1e-5, S_inv, torch.zeros_like(S_inv))
        J_pinv = (
            torch.transpose(Vh, 1, 2)[:, :, :6]
            @ torch.diag_embed(S_inv)
            @ torch.transpose(U, 1, 2)
        )
        delta_q = J_pinv @ pose_error.unsqueeze(-1)
        delta_q = delta_q.squeeze(-1)

        # 5. 업데이트 및 적용
        joint_pos = joint_pos + delta_q
        joint_pos = torch.clamp(joint_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
        
        robot_asset.write_joint_position_to_sim(
            position=joint_pos, joint_ids=joint_ids
        )
        
        env.scene.write_data_to_sim()
        env.sim.step()
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

        # 6. 종료 조건
        if (torch.norm(pose_error, dim=1) < 1e-3).any():
            # arm_action.process_actions(
            #     torch.concatenate(
            #         (noisy_target_pos_b, math_utils.axis_angle_from_quat(noisy_target_quat_b)), dim=1
            #     ), delta=False,
            # )
            arm_action.process_actions(
                torch.concatenate(
                    (noisy_target_pos_b, noisy_target_quat_b), dim=1
                ), delta=False,
            )
            for _ in range(150):
                arm_action.apply_actions()
                env.scene.write_data_to_sim()
                env.sim.step()
                if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
                    env.sim.render()
                env.scene.update(dt=env.physics_dt)
            break

    
def set_noisy_hole_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    noise: dict[str, tuple[float, float]] | None,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
):
    for _ in range(10):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    
    # 1. Target 및 Base Assets 가져오기
    target_asset: RigidObject | Articulation  = env.scene[target_cfg.name]
    base_asset: RigidObject | Articulation  = env.scene[base_cfg.name]

    # 2. Base와 Target의 현재 포즈 계산 (월드 좌표계)
    base_pose_w = base_asset.data.root_state_w[:, :7].clone()
    target_pose_w = target_asset.data.root_state_w[:, :7].clone()
    
    # 3. Base 좌표계를 기준으로 Target의 상대 포즈 계산
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        base_pose_w[:, 0:3], base_pose_w[:, 3:7], target_pose_w[:, 0:3], target_pose_w[:, 3:7]
    )

    # Apply Noise
    range_list = [noise.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=target_asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (env.num_envs, 6), device=target_asset.device)
    noisy_target_pos_b = target_pos_b + rand_samples[:, 0:3]
    
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    noisy_target_quat_b = math_utils.quat_mul(target_quat_b, orientations_delta)

    noisy_hole_pose_b = torch.cat((noisy_target_pos_b,
                                   math_utils.quat_unique(math_utils.normalize(noisy_target_quat_b))), dim=1)
    
    noisy_hole_pos_w = target_pose_w[:, 0:3] + rand_samples[:, 0:3]
    noisy_hole_quat_w = math_utils.quat_mul(target_pose_w[:, 3:7], orientations_delta)
    env.shared_dict["events"]["noisy_hole_pose_w"] = torch.cat((noisy_hole_pos_w, noisy_hole_quat_w), dim=1)
    env.shared_dict["events"]["noisy_hole_pose_b"] = noisy_hole_pose_b


def reset_table_color(env: ManagerBasedRLEnv, env_ids: torch.Tensor,):
    for idx in range(env.num_envs):
        env_prim_path = f"/World/envs/env_{int(idx)}"
        table_prim_path = env_prim_path + "/Table"
        shader_prim_path = table_prim_path + "/geometry/material/Shader"

        # Shader의 diffuseColor 속성 변경
        set_prim_property(
            prim_path=shader_prim_path,  # Shader의 경로
            property_name="inputs:diffuseColor",  # Shader의 diffuseColor 속성
            property_value=Gf.Vec3f(0.5, 0.5, 0.5)  # 회색
        )


def move_gripper_to_peg(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pos_offset: tuple[float, float, float] | None,
    quat_offset: tuple[float, float, float, float] | None,
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_name: str = "panda_hand",
    target_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    sim_step: int = 500,
    traj_steps: int = 500,
    pre_grasp_height_offset: float = 0.08,  # << NEW: 페그 위로 이동할 높이 (z-축 오프셋)
):
    """
    그리퍼를 2단계에 걸쳐 페그로 이동시킵니다:
    1. 페그 위 지정된 높이(pre_grasp_height_offset)까지 접근합니다.
    2. 수직으로 하강하여 페그를 잡을 위치로 이동합니다.
    """
    # 1. Target 및 Base Assets 가져오기
    target_asset: RigidObject | Articulation = env.scene[target_cfg.name]
    base_asset: RigidObject | Articulation = env.scene[base_cfg.name]

    body_ids, body_names = base_asset.find_bodies(ee_name)
    if len(body_ids) != 1:
        raise ValueError(
            f"Expected one match for the body name: {ee_name}. Found {len(body_ids)}: {body_names}."
        )
    ee_idx = body_ids[0]

    # 2. Base와 Target의 현재 포즈 계산 (월드 좌표계)
    base_pose_w = base_asset.data.root_state_w[:, :7].clone()
    target_pose_w = target_asset.data.root_state_w[:, :7].clone()

    # 3. Base 좌표계를 기준으로 Target의 상대 포즈 계산
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        base_pose_w[:, 0:3], base_pose_w[:, 3:7],
        target_pose_w[:, 0:3], target_pose_w[:, 3:7]
    )

    # 4. Offset 적용
    # 4.1 위치 offset
    if pos_offset is not None:
        pos_offset_tensor = torch.tensor(pos_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    else:
        pos_offset_tensor = torch.zeros_like(target_pos_b)

    # 4.2 쿼터니언 offset
    if quat_offset is not None:
        quat_offset_tensor = torch.tensor(quat_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    else:
        quat_offset_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

    # 4.3 Offset 적용: base 및 offset을 결합
    target_pos_b, target_quat_b = math_utils.combine_frame_transforms(
        target_pos_b, target_quat_b, pos_offset_tensor, quat_offset_tensor
    )
    
    # 6. Controller 관련 설정
    arm_action = env.action_manager.get_term("arm_action")
    gripper_action = env.action_manager.get_term("gripper_action")

    # 그리퍼 명령 사전 정의
    close_gripper_command = -torch.ones(env.num_envs, 1, device=env.device)
    open_gripper_command = torch.zeros_like(close_gripper_command)

    # 현재 TCP pose 계산
    start_pose_w = base_asset.data.body_state_w[:, ee_idx, :7].clone()
    start_pos_b, start_quat_b = math_utils.subtract_frame_transforms(
        base_pose_w[:, 0:3], base_pose_w[:, 3:7],
        start_pose_w[:, 0:3], start_pose_w[:, 3:7]
    )

    # Gain For Reset Movement (이동 시 높은 게인 사용)
    arm_action.set_gain(
        stiffness=[800, 800, 800, 30, 30, 30],
        damping_ratio=[0.7, 0.7, 0.7, 0.1, 0.1, 0.1],
    )

    # 7. 2단계 궤적 생성 및 실행
    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()

    # 7.1. 목표 지점들 정의
    # 시작 포즈
    start_pos = start_pos_b
    start_quat = start_quat_b
    # 최종 파지 포즈
    final_grasp_pos = target_pos_b
    final_grasp_quat = target_quat_b
    # 중간 경유지 (Pre-grasp) 포즈: 최종 목표보다 z축 방향으로 위
    pre_grasp_pos = final_grasp_pos + torch.tensor([0.0, 0.0, pre_grasp_height_offset], device=env.device)
    pre_grasp_quat = final_grasp_quat # 회전은 최종 목표와 동일
    
    # 각 궤적에 할당할 스텝 수
    approach_traj_steps = traj_steps // 2
    descend_traj_steps = traj_steps - approach_traj_steps
    
    sim_steps_per_waypoint = sim_step // traj_steps if traj_steps > 0 else 1

    # === 궤적 1: Pre-grasp 위치로 이동 ===
    for i in range(approach_traj_steps + 1):
        tau = i / float(approach_traj_steps) if approach_traj_steps > 0 else 1.0
        cubic_val = 3.0 * tau**2 - 2.0 * tau**3
        
        interp_pos = start_pos + (pre_grasp_pos - start_pos) * cubic_val
        interp_quat_list = [math_utils.quat_slerp(start_quat[j], pre_grasp_quat[j], tau).unsqueeze(0) for j in range(len(env_ids))]
        interp_quat = torch.cat(interp_quat_list, dim=0)
        
        interp_pose_b = torch.cat((interp_pos, interp_quat), dim=1)
        arm_action.set_command(command=interp_pose_b)

        # 궤적 이동 중에는 그리퍼를 계속 열어둠
        gripper_action.process_actions(open_gripper_command)

        for _ in range(sim_steps_per_waypoint):
            arm_action.apply_actions()
            gripper_action.apply_actions()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
                env.sim.render()
            env.scene.update(dt=env.physics_dt)

    for i in range(50): # 안정화를 위한 추가 스텝
        arm_action.apply_actions()
        gripper_action.apply_actions()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    # === 궤적 2: Pre-grasp 위치에서 Final-grasp 위치로 하강 ===
    for i in range(descend_traj_steps + 1):
        tau = i / float(descend_traj_steps) if descend_traj_steps > 0 else 1.0
        cubic_val = 3.0 * tau**2 - 2.0 * tau**3
        
        interp_pos = pre_grasp_pos + (final_grasp_pos - pre_grasp_pos) * cubic_val
        # 회전은 pre_grasp와 final_grasp가 동일하므로 보간 결과도 동일
        interp_quat_list = [math_utils.quat_slerp(pre_grasp_quat[j], final_grasp_quat[j], tau).unsqueeze(0) for j in range(len(env_ids))]
        interp_quat = torch.cat(interp_quat_list, dim=0)
        
        interp_pose_b = torch.cat((interp_pos, interp_quat), dim=1)
        arm_action.set_command(command=interp_pose_b)

        # 궤적 이동 중에는 그리퍼를 계속 열어둠
        gripper_action.process_actions(open_gripper_command)
        
        for _ in range(sim_steps_per_waypoint):
            arm_action.apply_actions()
            gripper_action.apply_actions()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
                env.sim.render()
            env.scene.update(dt=env.physics_dt)

    # 8. 최종 위치에서 페그 파지 및 안정화
    # 그리퍼 닫기 명령 전달
    gripper_action.process_actions(close_gripper_command)

    for i in range(50): # 안정화를 위한 추가 스텝
        arm_action.set_command(command=torch.cat((target_pos_b, target_quat_b), dim=1))
        gripper_action.process_actions(close_gripper_command) # 닫기 명령 유지

        arm_action.apply_actions()
        gripper_action.apply_actions()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    # Gain For Action Movement (다음 동작을 위해 낮은 게인으로 설정)
    arm_action.set_gain(
        stiffness=[100, 100, 100, 30, 30, 30],
        damping_ratio=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )






def set_filter_cutoff_hz(env: ManagerBasedRLEnv,
                        env_ids: torch.Tensor,
                        force_cutoff_hz: float = 50.0,
                        torque_cutoff_hz: float = 50.0,
):
    dt_sim = env.physics_dt  
    lp_force_filter = LowPassFilter(force_cutoff_hz, dt_sim, init_mode="first")
    lp_torque_filter = LowPassFilter(torque_cutoff_hz, dt_sim, init_mode="first")

    env.shared_dict["observations"]["lp_force_filter"] = lp_force_filter
    env.shared_dict["observations"]["lp_torque_filter"] = lp_torque_filter


def reset_peg_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pos_offset: tuple[float, float, float] | None,
    quat_offset: tuple[float, float, float, float] | None,
    pose_range: dict[str, tuple[float, float]],
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
):
    peg_asset: RigidObject | Articulation  = env.scene[peg_cfg.name]

    hole_pos_w = env.shared_dict["events"]["noisy_hole_pose_w"][:, 0:3]
    hole_quat_w = env.shared_dict["events"]["noisy_hole_pose_w"][:, 3:7]

    # Offset
    if pos_offset is not None:
        pos_offset_tensor = torch.tensor(pos_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    else:
        pos_offset_tensor = torch.zeros_like(hole_pos_w)

    if quat_offset is not None:
        quat_offset_tensor = torch.tensor(quat_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    else:
        quat_offset_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

    # 4.3 Offset 적용: base 및 offset을 결합
    target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
        hole_pos_w, hole_quat_w, pos_offset_tensor, quat_offset_tensor
    )
    
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    positions = target_pos_w + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(target_quat_w, orientations_delta)

    velocities = torch.zeros((env.num_envs, 6), device=env.device)  # [num_envs, 6]

    # set into the physics simulation
    peg_asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    peg_asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)