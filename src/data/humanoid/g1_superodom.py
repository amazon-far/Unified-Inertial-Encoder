"""G1 29-DOF articulated model and the canonical IMU site to URDF link mapping."""

from ...interface.data  import ArticulatedModel
from ...interface.human import T_CANONICAL_NAMES
from ...utility.urdf    import get_urdf


G1_29DOF_Model = ArticulatedModel(
    names=(
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    urdf =get_urdf("unitree", "g1_description/g1_29dof_rev_1_0.urdf")
)

# Canonical human IMU site -> G1 URDF link carrying the virtual sensor. Each segment is the
# parent link of the next joint down the chain. Deliberately partial: torso_t8 and the
# shoulders have no good G1 analog, so they are dropped rather than mapped.
G1_WholeBody_Link_Mapping: dict[T_CANONICAL_NAMES, str] = {
    "pelvis"         : "imu_in_pelvis",         # dedicated IMU frame, fixed to pelvis
    "head"           : "head_link",             # fixed to torso (G1 has no neck DOF)

    "left_upper_arm" : "left_shoulder_yaw_link",
    "left_forearm"   : "left_elbow_link",
    "left_hand"      : "left_rubber_hand",
    "right_upper_arm": "right_shoulder_yaw_link",
    "right_forearm"  : "right_elbow_link",
    "right_hand"     : "right_rubber_hand",

    "left_upper_leg" : "left_hip_yaw_link",
    "left_lower_leg" : "left_knee_link",
    "left_foot"      : "left_ankle_roll_link",
    "right_upper_leg": "right_hip_yaw_link",
    "right_lower_leg": "right_knee_link",
    "right_foot"     : "right_ankle_roll_link",
}

# Device-name container for whole-body synthesized IMU data at the sites above.
G1_WholeBody_IMU_Model = ArticulatedModel(names=tuple(G1_WholeBody_Link_Mapping.keys()), urdf=None)
