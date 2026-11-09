"""Supplied NumPy forward and position-only inverse Panda kinematics.

The robot base is fixed at the world origin with no base rotation. The returned
hand frame is the MJCF `hand` body, not a fingertip or a carried-object centre.
This module does not import Genesis or integrate physics.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

MODEL = json.loads(Path(__file__).with_name('panda_kinematics.json').read_text(encoding='utf-8'))
JOINT_BOUNDS_RAD = np.array([r['limits_rad'] for r in MODEL['records'] if 'joint_name' in r], dtype=np.float64)
JOINT_BOUNDS_RAD.setflags(write=False)


def point(value):
    """Return an independent seven-angle float64 array, in native joint order."""
    q = np.array(value, dtype=np.float64, copy=True)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('a configuration must contain seven finite joint angles')
    return q


def quaternion_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)


@dataclass(frozen=True)
class HandPose:
    position_m: np.ndarray
    rotation_matrix: np.ndarray


def body_frames(q):
    """Map body names to their world homogeneous transforms at seven angles."""
    q = point(q)
    if np.any(q < JOINT_BOUNDS_RAD[:, 0]) or np.any(q > JOINT_BOUNDS_RAD[:, 1]):
        raise ValueError('joint angles exceed the supplied Panda source limits')
    T = np.eye(4)
    frames = {'link0': T.copy()}
    joint_index = 0
    for record in MODEL['records']:
        local = np.eye(4)
        local[:3, :3] = quaternion_matrix(record['quaternion_wxyz'])
        local[:3, 3] = record['position_m']
        T = T @ local
        if 'joint_name' in record:
            rotation = np.eye(4)
            rotation[:3, :3] = axis_rotation(record['axis'], q[joint_index])
            T = T @ rotation
            joint_index += 1
        frames[record['body']] = T.copy()
    return frames


def forward_kinematics(q):
    """Return the world hand position (m) and a 3×3 orientation matrix."""
    T = body_frames(q)['hand']
    return HandPose(T[:3, 3].copy(), T[:3, :3].copy())


def inverse_kinematics(position_m, seed_q):
    """Return seven bounded angles for a world hand position in metres.

    This supplied local solver starts at the seven-angle seed without random
    restarts. It returns only when FK position error is at most 1e-5 m. It uses
    at most 100 bounded updates and raises RuntimeError on nonconvergence.
    Invalid inputs raise ValueError. Orientation, collision freedom and the
    closest possible solution to the seed are not promised.
    """
    try:
        target = np.array(position_m, dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        raise ValueError('IK position must contain three finite coordinates in metres') from None
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('IK position must contain three finite coordinates in metres')
    try:
        q = point(seed_q)
    except (TypeError, ValueError):
        raise ValueError('IK seed must contain seven finite joint angles') from None
    frames = body_frames(q)  # Also validates the source joint limits.
    tolerance_m = 1e-5
    # A conservative chain-length bound also rejects huge finite inputs safely.
    reach_m = sum(np.linalg.norm(r['position_m']) for r in MODEL['records'])
    if np.max(np.abs(target)) > reach_m + tolerance_m:
        raise RuntimeError('position IK failed: target exceeds the source-chain reach bound')

    for _ in range(100):
        hand = frames['hand'][:3, 3]
        error = target - hand
        residual_m = float(np.linalg.norm(error))
        if residual_m <= tolerance_m:
            return q.copy()

        columns = []
        for record in MODEL['records']:
            if 'joint_name' in record:
                T = frames[record['body']]
                axis = np.asarray(record['axis'], dtype=float)
                world_axis = T[:3, :3] @ (axis / np.linalg.norm(axis))
                columns.append(np.cross(world_axis, hand - T[:3, 3]))
        jacobian = np.column_stack(columns)
        # Damped least squares gives a local position correction. Each update
        # has a seven-angle Euclidean length of at most 0.15 rad.
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.01**2 * np.eye(3), error)
        step *= min(1., 0.15 / max(float(np.linalg.norm(step)), 1e-15))

        for scale in (1., .5, .25, .125, .0625, .03125, .015625, .0078125):
            candidate = np.clip(q + scale * step,
                                JOINT_BOUNDS_RAD[:, 0], JOINT_BOUNDS_RAD[:, 1])
            candidate_frames = body_frames(candidate)
            candidate_error = target - candidate_frames['hand'][:3, 3]
            if np.linalg.norm(candidate_error) < residual_m:
                q, frames = candidate, candidate_frames
                break
        else:
            break

    residual_m = float(np.linalg.norm(target - frames['hand'][:3, 3]))
    if residual_m <= tolerance_m:
        return q.copy()
    raise RuntimeError(f'position IK did not converge: residual {residual_m:.6g} m exceeds 1e-5 m')


def capsule_segments(q):
    """Return source-geometry enclosing capsules at the current seven angles.

    Local fitted capsules enclose collision-mesh vertices and fingertip boxes.
    The hand-frame records include both fingers fixed at .04 m. This does not
    test self-collision, certify sampled connections or establish execution.
    """
    frames = body_frames(q)
    result = []
    for record in MODEL['source_collision_capsules']:
        if record['frame'] == 'link0':
            continue  # The fixed mounting base may touch its floor.
        T = frames[record['frame']]
        a, b = (T[:3,:3] @ np.asarray(record[key]) + T[:3,3] for key in ('a_m', 'b_m'))
        result.append((a, b, record['radius_m']))
    return tuple(result)
