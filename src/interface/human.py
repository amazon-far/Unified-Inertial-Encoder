"""Canonical body-site names; the index of a name is its sensor id."""
from typing import Literal, get_args


T_CANONICAL_NAMES = Literal[
    "pelvis", "head", "torso_t8",

    "left_shoulder", "left_upper_arm", "left_forearm", "left_hand",
    "right_shoulder", "right_upper_arm", "right_forearm", "right_hand",

    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_upper_leg", "right_lower_leg", "right_foot",
]
CANONICAL_NAMES: tuple[T_CANONICAL_NAMES]       = get_args(T_CANONICAL_NAMES)
CANONICAL_IDX  : dict[T_CANONICAL_NAMES, int]   = {name: idx for idx, name in enumerate(CANONICAL_NAMES)}
