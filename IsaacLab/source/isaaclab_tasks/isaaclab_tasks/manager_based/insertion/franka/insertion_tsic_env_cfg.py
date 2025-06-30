# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.05, 0.05, 0.05)

import numpy as np
from . import mdp

##
# Pre-defined configs
##


##
# Scene definition
##

@configclass
class InsertionSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 0.91, 0.64),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5,0.5,0.5), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.32),
        ),
    )

    hole: ArticulationCfg = MISSING

    peg: ArticulationCfg = MISSING
    
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Contact sensors
    contact_sensor_L = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.02,
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ContactSensor_L"),
        track_pose=True,
        track_air_time=True,
        force_threshold=0.5,
    )
    contact_sensor_R = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.02,
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ContactSensor_R"),
        track_pose=True,
        track_air_time=True,
        force_threshold=0.5,
    )


##
# mdp_franka settings
##

@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    set_intertia = EventTerm(
        func=mdp.set_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )

    set_hole_friction = EventTerm(
        func=mdp.set_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("hole"),
            "friction": 0.75,
        }
    )
    set_peg_friction = EventTerm(
        func=mdp.set_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("peg"),
            "friction": 0.75,
        }
    )
    set_robot_friction = EventTerm(
        func=mdp.set_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction": 0.75,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_table_color = EventTerm(func=mdp.reset_table_color, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.125, 0.125),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_hole = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("hole"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0),
                           "yaw": (-2*np.pi, 2*np.pi)},
            "velocity_range": {"x": (-0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "yaw": (-0.0, 0.0)}
        }
    )

    set_noisy_hole_pose = EventTerm(
        func=mdp.set_noisy_hole_pose,
        mode="reset",
        params={
            "base_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("hole"),
            "noise": {"x": (-0.001, 0.001), "y": (-0.001, 0.001), "z": (0.0, 0.0),
                      "yaw": (-0.01, 0.01)},
            # "noise": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0),
            #           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.0, 0.0)},
        },
    )

    reset_peg = EventTerm(
        func=mdp.reset_peg_pose,
        mode="reset",
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "pos_offset": (0.0, 0.0, 0.1),
            "quat_offset": (1.0, 0.0, 0.0, 0.0),
            "pose_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.03, 0.03),
                           "yaw": (-0.1, 0.1)},
            # "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0),
            #                "yaw": (-0.0, 0.0)},
        }
    )

    
    move_gripper_to_peg = EventTerm(
        func=mdp.move_gripper_to_peg,
        mode="reset",
        params={
            "base_cfg": SceneEntityCfg("robot"),
            "ee_name": "panda_hand",
            "target_cfg": SceneEntityCfg("peg"),
            "sim_step": 300,
            "traj_steps": 50,
            "pos_offset": (0.0, 0.0, 0.193),
            "quat_offset": (0.0, 1.0, 0.0, 0.0),
        }
    )

    set_filter_cutoff_hz = EventTerm(
        func=mdp.set_filter_cutoff_hz,
        mode="startup",
        params={
            "force_cutoff_hz": 5.0,
            "torque_cutoff_hz": 5.0,
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the mdp_franka."""

    arm_action: mdp.TaskSpaceImpedanceControllerActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the mdp_franka."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        last_action = ObsTerm(func=mdp.last_action, # 6
                              params={"action_name": "arm_action"})
        
        joint_pos = ObsTerm(func=mdp.joint_pos) # 7
        joint_vel = ObsTerm(func=mdp.joint_vel) # 7

        ee_pos_b = ObsTerm(func=mdp.ee_pos_b, # 3
                           params={"robot_cfg": SceneEntityCfg("robot"),
                                   "ee_cfg": SceneEntityCfg("ee_frame")})
        ee_quat_b = ObsTerm(func=mdp.ee_quat_b, # 4
                            params={"robot_cfg": SceneEntityCfg("robot"),
                                    "ee_cfg": SceneEntityCfg("ee_frame")})
        
        noisy_hole_pos_b = ObsTerm(func=mdp.noisy_object_pos_b,
                                   params={"noise_name": "noisy_hole_pose_b"})
        noisy_hole_quat_b = ObsTerm(func=mdp.noisy_object_quat_b,
                                    params={"noise_name": "noisy_hole_pose_b"})

        net_contact_force_x_ee_L = ObsTerm(func=mdp.net_contact_force_x_ee, # 1
                                         params={
                                            "robot_cfg": SceneEntityCfg("robot"),
                                            "ee_cfg": SceneEntityCfg("ee_frame"),
                                            "contact_sensor_cfg": SceneEntityCfg("contact_sensor_L"),
                                         })
        net_contact_force_x_ee_R = ObsTerm(func=mdp.net_contact_force_x_ee, # 1
                                         params={
                                            "robot_cfg": SceneEntityCfg("robot"),
                                            "ee_cfg": SceneEntityCfg("ee_frame"),
                                            "contact_sensor_cfg": SceneEntityCfg("contact_sensor_R"),
                                         })

        F_xy_filtered_sensor_ee = ObsTerm(func=mdp.F_xy_filtered_sensor_ee, # 2
                                          params={"robot_cfg": SceneEntityCfg("robot"),
                                                  "ee_cfg": SceneEntityCfg("ee_frame")})
        F_z_filtered_sensor_ee = ObsTerm(func=mdp.F_z_filtered_sensor_ee, # 1
                                         params={"robot_cfg": SceneEntityCfg("robot"),
                                                 "ee_cfg": SceneEntityCfg("ee_frame")})

        def __post_init__(self):
            self.enable_corruption = True # False: Do not add noise to observations
            self.concatenate_terms = True
 

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        last_action = ObsTerm(func=mdp.last_action,
                              params={"action_name": "arm_action"}) # 6
        
        joint_pos = ObsTerm(func=mdp.joint_pos) # 7
        joint_vel = ObsTerm(func=mdp.joint_vel) # 7

        ee_pos_b = ObsTerm(func=mdp.ee_pos_b, # 3
                           params={"robot_cfg": SceneEntityCfg("robot"),
                                   "ee_cfg": SceneEntityCfg("ee_frame")})
        ee_quat_b = ObsTerm(func=mdp.ee_quat_b, # 4
                           params={"robot_cfg": SceneEntityCfg("robot"),
                                   "ee_cfg": SceneEntityCfg("ee_frame")})

        ee_linvel_b = ObsTerm(func=mdp.ee_linvel_b, # 3
                              params={"robot_cfg": SceneEntityCfg("robot"),
                                      "ee_name": str("panda_hand")})
        ee_angvel_b = ObsTerm(func=mdp.ee_angvel_b, # 3
                              params={"robot_cfg": SceneEntityCfg("robot"),
                                      "ee_name": str("panda_hand")})

        peg_pos_b = ObsTerm(func=mdp.object_pos_b, # 3
                             params={"base_cfg": SceneEntityCfg("robot"),
                                     "object_cfg": SceneEntityCfg("peg")})
        peg_quat_b = ObsTerm(func=mdp.object_quat_b, # 4
                             params={"base_cfg": SceneEntityCfg("robot"),
                                     "object_cfg": SceneEntityCfg("peg")})

        hole_pos_b = ObsTerm(func=mdp.object_pos_b, # 3
                             params={"base_cfg": SceneEntityCfg("robot"),
                                     "object_cfg": SceneEntityCfg("hole")})
        hole_quat_b = ObsTerm(func=mdp.object_quat_b, # 4
                             params={"base_cfg": SceneEntityCfg("robot"),
                                     "object_cfg": SceneEntityCfg("hole")})
        
        peg_pos_from_ee = ObsTerm(func=mdp.peg_pos_from_ee, # 3
                                  params={"ee_cfg": SceneEntityCfg("ee_frame"),
                                          "peg_cfg": SceneEntityCfg("peg")})
        peg_quat_from_ee = ObsTerm(func=mdp.peg_quat_from_ee, # 4
                                   params={"ee_cfg": SceneEntityCfg("ee_frame"),
                                           "peg_cfg": SceneEntityCfg("peg")})
        
        net_contact_force_ee_L = ObsTerm(func=mdp.net_contact_force_ee, # 3
                                           params={
                                              "robot_cfg": SceneEntityCfg("robot"),
                                              "ee_cfg": SceneEntityCfg("ee_frame"),
                                              "contact_sensor_cfg": SceneEntityCfg("contact_sensor_L"),
                                           })
        net_contact_force_ee_R = ObsTerm(func=mdp.net_contact_force_ee, # 3
                                           params={
                                              "robot_cfg": SceneEntityCfg("robot"),
                                              "ee_cfg": SceneEntityCfg("ee_frame"),
                                              "contact_sensor_cfg": SceneEntityCfg("contact_sensor_R"),
                                           })

        F_filtered_sensor_ee = ObsTerm(func=mdp.F_filtered_sensor_ee, # 3
                                       params={"robot_cfg": SceneEntityCfg("robot"),
                                               "ee_cfg": SceneEntityCfg("ee_frame")})

        T_filtered_sensor_ee = ObsTerm(func=mdp.T_filtered_sensor_ee, # 3
                                       params={"robot_cfg": SceneEntityCfg("robot"),
                                               "ee_cfg": SceneEntityCfg("ee_frame")})
        
        
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the mdp_franka."""
    # zero_reward = RewTerm(func=mdp.zero_rew, weight=1.0)
    compute_intermediate_values = RewTerm(func=mdp.compute_intermediate_values,
                                          params={"hole_height": 0.05,
                                                  "robot_cfg": SceneEntityCfg("robot"),
                                                  "hole_cfg": SceneEntityCfg("hole"),
                                                  "peg_cfg": SceneEntityCfg("peg")},
                                          weight=1.0)
    
    comprehensive_reward = RewTerm(func=mdp.comprehensive_reward, weight=1.0)
    action_smoothness_penalty = RewTerm(func=mdp.action_smoothness_penalty,
                                        params={"action_name": "arm_action"}, weight=0.01)
    # missing_peg_penalty = RewTerm(func=mdp.missing_peg_penalty,
    #                               params={"threshold": 0.002}, weight=1.0)
    fast_success_reward = RewTerm(func=mdp.fast_success_reward, weight=1.0)
    final_success_reward = RewTerm(func=mdp.final_success_reward, weight=1.0)
    
    """Logging terms."""
    wandb_log = RewTerm(func=mdp.wandb_log, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the mdp_franka."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    

##
# Environment configuration
##


@configclass
class InsertionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the insertion environment."""

    # Scene settings
    scene: InsertionSceneCfg = InsertionSceneCfg(num_envs=1024, env_spacing=1.8)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # mdp settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 512 / 15
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 120Hz
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.render_interval = self.decimation
        self.disable_contact_processing = False # False: Enable contact processing
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_max_num_partitions = 1
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        )