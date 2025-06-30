import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
import os

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

@configclass
class HoleCfg:

    @configclass
    class UsdPathCfg:
        current_dir = os.path.dirname(__file__)
        hole_foler: str = os.path.join(current_dir, "usd", "hole")

        three: str = os.path.join(hole_foler, "hole_three.usd")
        # four: str = os.path.join(hole_foler, "factory_rectangular_hole_12mm_subdiv_3x.usd")
        four: str = f"{ASSET_DIR}/factory_hole_8mm.usd"
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
        four: tuple = (1.0, 1.0, 2) # size up
        five: tuple = (1.2, 1.2, 3) # 0205
        six: tuple = (1.018, 1.018, 3)
        seven: tuple = (1.016, 1.016, 1) #018
        eight: tuple = (1.0138, 1.0138, 3)
        trapezoid: tuple = (1.015, 1.015, 3)
        circle: tuple = (1.021, 1.021, 1)
        lan: tuple = (1.05, 1.05, 1) # size up
        usb: tuple = (1.05, 1.05, 1) # size up


hole_usd_path = HoleCfg.UsdPathCfg()
hole_scale = HoleCfg.ScaleCfg()

HOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=hole_usd_path.four,
        scale=hole_scale.four,
        activate_contact_sensors=False,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.0, 0.0), metallic=0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.45 + 0.15, 0.0, 0.64), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

HOLE_FOUR_CFG = HOLE_CFG.copy()
HOLE_FOUR_CFG.spawn.usd_path = hole_usd_path.four
HOLE_FOUR_CFG.spawn.scale = hole_scale.four
