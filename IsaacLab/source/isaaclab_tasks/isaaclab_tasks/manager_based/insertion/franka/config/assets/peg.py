import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
import os

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

@configclass
class PegCfg:

    @configclass
    class UsdPathCfg:
        current_dir = os.path.dirname(__file__)
        peg_foler: str = os.path.join(current_dir, "usd", "peg")

        # three: str = os.path.join(peg_foler, "peg_three.usd")
        four: str = f"{ASSET_DIR}/factory_peg_8mm.usd"
        # four: str = os.path.join(peg_foler, "factory_rectangular_peg_12mm_loose_subdiv_3x.usd")
        five: str = os.path.join(peg_foler, "peg_five.usd")
        six: str = os.path.join(peg_foler, "peg_six.usd")
        # seven: str = os.path.join(peg_foler, "peg_seven.usd")
        eight: str = os.path.join(peg_foler, "peg_eight.usd")
        trapezoid: str = os.path.join(peg_foler, "peg_trapezoid.usd")
        # circle: str = os.path.join(peg_foler, "peg_circle.usd")
        lan: str = os.path.join(peg_foler, "peg_lan.usd")
        # usb: str = os.path.join(peg_foler, "peg_usb.usd")
    
    @configclass
    class ScaleCfg:
        four: tuple = (1, 1, 2)  # size up

peg_usd_path = PegCfg.UsdPathCfg()
peg_scale = PegCfg.ScaleCfg()


PEG_CFG = ArticulationCfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Peg",
    spawn=sim_utils.UsdFileCfg(
        usd_path=peg_usd_path.four,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
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
        pos=(0.45 + 0.15, 0.0, 0.64 + 0.2), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

PEG_FOUR_CFG = PEG_CFG.copy()
PEG_FOUR_CFG.spawn.usd_path = peg_usd_path.four
PEG_FOUR_CFG.spawn.scale = peg_scale.four