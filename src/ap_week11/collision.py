"""Supplied conservative capsule/box teaching geometry checks.

Expanding every box axis by a capsule radius encloses its rounded expansion.
Rejecting an intersection is conservative for these enclosing capsules. The
fitted capsules enclose source collision vertices, including fixed open fingers.
Edge checks still sample joint interpolation. Self-collision, continuous-time
collision freedom and physical execution remain separate checks.
"""
import numpy as np
from .kinematics import point, capsule_segments, JOINT_BOUNDS_RAD


def segment_hits_box(a, b, lower, upper):
    """Closed-segment/AABB slab test. Boundary contact counts as intersection."""
    a, b = np.asarray(a), np.asarray(b)
    enter, leave = 0., 1.
    for start, delta, low, high in zip(a, b-a, lower, upper):
        if abs(delta) < 1e-15:
            if start < low or start > high:
                return False
        else:
            first, last = sorted(((low-start)/delta, (high-start)/delta))
            enter, leave = max(enter, first), min(leave, last)
            if enter > leave + 1e-12:
                return False
    return True


def state_is_valid(q, lower=None, upper=None):
    q = point(q)
    if np.any(q < JOINT_BOUNDS_RAD[:, 0]) or np.any(q > JOINT_BOUNDS_RAD[:, 1]):
        return False
    for a, b, radius in capsule_segments(q):
        if min(a[2], b[2]) - radius < -1e-12:
            return False
        if lower is not None and segment_hits_box(a, b, np.asarray(lower)-radius, np.asarray(upper)+radius):
            return False
    return True


def edge_is_valid(a, b, lower=None, upper=None, resolution=.04):
    """Sample every joint by at most resolution radians, including endpoints."""
    a, b = point(a), point(b)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution must be finite and positive')
    samples = max(1, int(np.ceil(np.max(np.abs(b-a))/resolution)))
    return all(state_is_valid(a+(b-a)*s/samples, lower, upper) for s in range(samples+1))
