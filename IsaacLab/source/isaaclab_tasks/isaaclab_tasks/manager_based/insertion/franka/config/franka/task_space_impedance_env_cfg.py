# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.task_space_impedance_cfg import TaskSpaceImpedanceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import TaskSpaceImpedanceControllerActionCfg, BinaryJointPositionActionCfg
from isaaclab.utils import configclass

from isaaclab.sensors import FrameTransformerCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import wandb
import os
import time

##
# Pre-defined configs
##

from ..assets.franka.franka import FRANKA_PANDA_CFG
from ...insertion_tsic_env_cfg import InsertionEnvCfg  # isort: skip
from ...insertion_tsic_camera_env_cfg import InsertionCameraEnvCfg  # isort: skip

from ..assets.hole import HOLE_FOUR_CFG
from ..assets.peg import PEG_FOUR_CFG


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.01, 0.01, 0.01)


@configclass
class HoleCfg:

    @configclass
    class UsdPathCfg:
        current_dir = os.path.dirname(__file__)
        hole_foler: str = os.path.join(current_dir, "..", "assets", "usd", "hole")

        three: str = os.path.join(hole_foler, "hole_three.usd")
        four: str = os.path.join(hole_foler, "hole_four.usd")
        five: str = os.path.join(hole_foler, "hole_five.usd")
        six: str = os.path.join(hole_foler, "hole_six.usd")
        seven: str = os.path.join(hole_foler, "hole_seven.usd")
        eight: str = os.path.join(hole_foler, "hole_eight.usd")
        trapezoid: str = os.path.join(hole_foler, "hole_trapezoid.usd")
        circle: str = os.path.join(hole_foler, "hole_circle.usd")
        lan: str = os.path.join(hole_foler, "hole_lan.usd")
        usb: str = os.path.join(hole_foler, "hole_usb.usd")

    @configclass
    class ScaleCfg:
        three: tuple = (1.016, 1.016, 1)
        four: tuple = (1.2, 1.2, 1) # size up
        five: tuple = (1.2, 1.2, 3) # 0205
        six: tuple = (1.018, 1.018, 3)
        seven: tuple = (1.016, 1.016, 1) #018
        eight: tuple = (1.0138, 1.0138, 3)
        trapezoid: tuple = (1.015, 1.015, 3)
        circle: tuple = (1.021, 1.021, 1)
        lan: tuple = (1.05, 1.05, 1) # size up
        usb: tuple = (1.05, 1.05, 1) # size up


@configclass
class PegCfg:

    @configclass
    class UsdPathCfg:
        current_dir = os.path.dirname(__file__)
        peg_foler: str = os.path.join(current_dir, "..", "assets", "usd", "peg")

        # three: str = os.path.join(peg_foler, "peg_three.usd")
        four: str = os.path.join(peg_foler, "peg_four.usd")
        five: str = os.path.join(peg_foler, "peg_five.usd")
        six: str = os.path.join(peg_foler, "peg_six.usd")
        # seven: str = os.path.join(peg_foler, "peg_seven.usd")
        eight: str = os.path.join(peg_foler, "peg_eight.usd")
        trapezoid: str = os.path.join(peg_foler, "peg_trapezoid.usd")
        # circle: str = os.path.join(peg_foler, "peg_circle.usd")
        lan: str = os.path.join(peg_foler, "peg_lan.usd")
        # usb: str = os.path.join(peg_foler, "peg_usb.usd")


hole_usd_path = HoleCfg.UsdPathCfg()
hole_scale = HoleCfg.ScaleCfg()

peg_usd_path = PegCfg.UsdPathCfg()


@configclass
class FrankaInsertionEnvCfg(InsertionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=FRANKA_PANDA_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.64),
                rot=(1.0, 0.0, 0.0, 0.0)
            )
        )
        self.scene.robot.spawn.activate_contact_sensors = True

        self.scene.hole = HOLE_FOUR_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Hole",
        )
        self.scene.peg = PEG_FOUR_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Peg",
        )

        self.actions.arm_action = TaskSpaceImpedanceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            clip_action=True,
            controller=TaskSpaceImpedanceControllerCfg(
                stiffness=(800, 800, 800, 30, 30, 30),
                damping_ratio=(0.7, 0.7, 0.7, 0.1, 0.1, 0.1),
                gravity_compensation=True,
                default_dof_pos_tensor=(-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754),
                kp_null=10.0,
                kd_null=6.3246,
                is_restricted=False,
            ),
            body_offset=TaskSpaceImpedanceControllerActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            scale=(0.02, 0.02, 0.02, 0.097, 0.097, 0.097)
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_link7",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

@configclass
class FrankaInsertionCameraEnvCfg(InsertionCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=FRANKA_PANDA_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.64),
                rot=(1.0, 0.0, 0.0, 0.0)
            )
        )
        self.scene.robot.spawn.activate_contact_sensors = True

        self.scene.hole = HOLE_FOUR_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Hole",
        )
        self.scene.peg = PEG_FOUR_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Peg",
        )

        self.actions.arm_action = TaskSpaceImpedanceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            clip_action=True,
            controller=TaskSpaceImpedanceControllerCfg(
                stiffness=(800, 800, 800, 30, 30, 30),
                damping_ratio=(0.7, 0.7, 0.7, 0.1, 0.1, 0.1),
                gravity_compensation=True,
                default_dof_pos_tensor=(-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754),
                kp_null=10.0,
                kd_null=6.3246,
                is_restricted=False,
            ),
            body_offset=TaskSpaceImpedanceControllerActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            scale=(0.02, 0.02, 0.02, 0.097, 0.097, 0.097)
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_link7",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )



########### Environment Configuration ###############
"""Environment Configuration"""
@configclass
class FrankaInsertionEnvFourCfg(FrankaInsertionEnvCfg):
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 호출
        super().__post_init__()

        # Initialize wandb
        if wandb.run is None:
            wandb.init(project="capston-four-v0", name=time.strftime('%m%d-%H:%M:%S'))


@configclass
class FrankaInsertionEnvFiveCfg(FrankaInsertionEnvCfg):
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 호출
        super().__post_init__()

        # Initialize wandb
        if wandb.run is None:
            wandb.init(project="capston-five-v0", name=time.strftime('%m%d-%H:%M:%S'))

        self.scene.hole.spawn.usd_path = hole_usd_path.five
        self.scene.hole.spawn.scale = hole_scale.five

        self.scene.peg.spawn.usd_path = peg_usd_path.five


@configclass
class FrankaInsertionEnvLanCfg(FrankaInsertionEnvCfg):
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 호출
        super().__post_init__()

        # Initialize wandb
        if wandb.run is None:
            wandb.init(project="capston-lan-v0", name=time.strftime('%m%d-%H:%M:%S'))

        self.scene.hole.spawn.usd_path = hole_usd_path.lan
        self.scene.hole.spawn.scale = hole_scale.lan

        self.scene.peg.spawn.usd_path = peg_usd_path.lan




############### Camera Environment Configuration ###############
@configclass
class FrankaInsertionCameraEnvFourCfg(FrankaInsertionCameraEnvCfg):
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 호출
        super().__post_init__()

        # Initialize wandb
        if wandb.run is None:
            wandb.init(project="capston-four-v1", name=time.strftime('%m%d-%H:%M:%S'))


@configclass
class FrankaInsertionCameraEnvFiveCfg(FrankaInsertionCameraEnvCfg):
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 호출
        super().__post_init__()

        # Initialize wandb``
        if wandb.run is None:
            wandb.init(project="capston-five-v1", name=time.strftime('%m%d-%H:%M:%S'))

        self.scene.hole.spawn.usd_path = hole_usd_path.five
        self.scene.hole.spawn.scale = hole_scale.five

        self.scene.peg.spawn.usd_path = peg_usd_path.five


@configclass
class FrankaInsertionCameraEnvLanCfg(FrankaInsertionCameraEnvCfg):
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 호출
        super().__post_init__()

        # Initialize wandb
        if wandb.run is None:
            wandb.init(project="capston-lan-v1", name=time.strftime('%m%d-%H:%M:%S'))

        self.scene.hole.spawn.usd_path = hole_usd_path.lan
        self.scene.hole.spawn.scale = hole_scale.lan

        self.scene.peg.spawn.usd_path = peg_usd_path.lan