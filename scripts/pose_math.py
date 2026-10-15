"""Quaternion and Euler conversions for pose editors (wxyz, XYZ degrees)."""
import math


def _normalize_quat(quat: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in quat))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(value) / norm for value in quat)  # type: ignore[return-value]


def _quat_from_euler_xyz_deg(euler_deg: list[float] | tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(float(value)) for value in euler_deg)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return _normalize_quat(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _euler_xyz_deg_from_quat(quat: list[float] | tuple[float, float, float, float]) -> list[float]:
    w, x, y, z = _normalize_quat(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


