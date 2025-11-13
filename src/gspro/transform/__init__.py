"""
3D transform module.

Provides high-performance geometric transformations for 3D Gaussian Splatting.
"""

from gspro.transform.api import (
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotate,
    rotation_matrix_to_quaternion,
    scale,
    transform,
    translate,
)

__all__ = [
    "translate",
    "rotate",
    "scale",
    "transform",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "euler_to_quaternion",
    "quaternion_to_euler",
]
