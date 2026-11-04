"""Joint selection shared by actions, observations, rewards and symmetry.

Isaac Lab's default (preserve_order=False) resolves this selection in simulator
joint order. The list is a selection, NOT an export/deployment index contract.
"""
K1_LEG_JOINT_NAMES = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
K1_ARM_JOINT_NAMES = [
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll",
    "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll",
    "Right_Elbow_Pitch", "Right_Elbow_Yaw",
]
K1_WALK_JOINT_NAMES = K1_LEG_JOINT_NAMES + K1_ARM_JOINT_NAMES
# Mirrored nominal pose, radians. Arms lowered; elbows slightly bent.
K1_ARM_DEFAULT_POS = {
    "ALeft_Shoulder_Pitch": 0.0, "Left_Shoulder_Roll": -1.35,
    "Left_Elbow_Pitch": 0.0, "Left_Elbow_Yaw": -0.5,
    "ARight_Shoulder_Pitch": 0.0, "Right_Shoulder_Roll": 1.35,
    "Right_Elbow_Pitch": 0.0, "Right_Elbow_Yaw": 0.5,
}
