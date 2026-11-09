"""Supplied seven-joint problem, budget and results, continuing Week 10's API."""
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from .kinematics import point, JOINT_BOUNDS_RAD, inverse_kinematics
from .collision import state_is_valid, edge_is_valid

START_POSITION_M = np.array([.35, -.10, .72])
GOAL_POSITION_M = np.array([.35, -.02, .72])
IK_SEED_Q = np.array([-.3, -.6, 0., -2., 0., 1.8, .8])
# Supplied local, position-only IK. It makes endpoints, not a planned route.
DEFAULT_START = inverse_kinematics(START_POSITION_M, IK_SEED_Q)
DEFAULT_GOAL = inverse_kinematics(GOAL_POSITION_M, DEFAULT_START)
DEFAULT_BOUNDS = np.column_stack((np.minimum(DEFAULT_START, DEFAULT_GOAL)-.10,
                                  np.maximum(DEFAULT_START, DEFAULT_GOAL)+.10))
for _constant in (START_POSITION_M, GOAL_POSITION_M, IK_SEED_Q,
                  DEFAULT_START, DEFAULT_GOAL, DEFAULT_BOUNDS):
    _constant.setflags(write=False)


def distance(left, right):
    return float(np.linalg.norm(np.asarray(right) - np.asarray(left)))


@dataclass(frozen=True)
class PlanningProblem:
    start: np.ndarray = field(default_factory=lambda: DEFAULT_START.copy())
    goal: np.ndarray = field(default_factory=lambda: DEFAULT_GOAL.copy())
    bounds: np.ndarray = field(default_factory=lambda: DEFAULT_BOUNDS.copy())
    obstacle_lower_m: np.ndarray | None = None
    obstacle_upper_m: np.ndarray | None = None
    edge_is_valid: object = None
    state_is_valid: object = None

    def __post_init__(self):
        start, goal = point(self.start), point(self.goal)
        bounds = np.array(self.bounds, dtype=float, copy=True)
        lower, upper = self.obstacle_lower_m, self.obstacle_upper_m
        if (lower is None) != (upper is None):
            raise ValueError('supply both obstacle bounds or neither')
        if lower is not None:
            lower, upper = (np.array(x, dtype=float, copy=True) for x in (lower, upper))
            if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or np.any(lower >= upper):
                raise ValueError('obstacle bounds must be finite ordered XYZ coordinates')
        if bounds.shape != (7, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError('bounds must be finite ordered shape (7,2)')
        if np.any(bounds[:, 0] < JOINT_BOUNDS_RAD[:, 0]) or np.any(bounds[:, 1] > JOINT_BOUNDS_RAD[:, 1]):
            raise ValueError('teaching bounds must stay inside Panda source joint limits')
        for q in (start, goal):
            if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
                raise ValueError('start and goal must be within planning bounds')
        state_callback = self.state_is_valid
        edge_callback = self.edge_is_valid
        if state_callback is None:
            state_callback = lambda q: state_is_valid(q, lower, upper)
        if edge_callback is None:
            edge_callback = lambda a, b: edge_is_valid(a, b, lower, upper)
        if not callable(state_callback) or not callable(edge_callback):
            raise TypeError('validity checks must be callable')
        if not all(state_callback(q) for q in (start, goal)):
            raise ValueError('start and goal must satisfy the supplied teaching state rule')
        for array in (start, goal, bounds, lower, upper):
            if array is not None:
                array.setflags(write=False)
        for name, value in [('start', start), ('goal', goal), ('bounds', bounds),
                            ('obstacle_lower_m', lower), ('obstacle_upper_m', upper),
                            ('state_is_valid', state_callback), ('edge_is_valid', edge_callback)]:
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PlanningResult:
    status: str
    path: np.ndarray | None
    cost: float
    checks: int
    nodes: int
    elapsed_s: float


class EdgeBudget:
    def __init__(self, callback, max_checks):
        if type(max_checks) is not int or max_checks < 0:
            raise ValueError('max_checks must be a nonnegative integer')
        self.callback, self.max_checks, self.used = callback, max_checks, 0

    @property
    def remaining(self):
        return self.max_checks-self.used

    def check(self, left, right):
        if not self.remaining:
            return False
        self.used += 1
        return bool(self.callback(left, right))


def steer(q_from, q_target, step_size):
    """Supplied Week 10 bounded-step formula, now supporting all seven entries."""
    origin, target = point(q_from), point(q_target)
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError('step_size must be finite and positive')
    delta = target-origin
    length = np.linalg.norm(delta)
    return target.copy() if length <= step_size else origin+step_size*delta/length


def prepare_tree(problem, seed, max_checks, step_size, goal_bias=.15):
    if not np.isfinite(step_size) or not .15 <= step_size <= .5:
        raise ValueError('this exercise requires 0.15 <= step_size <= 0.50 rad')
    if not np.isfinite(goal_bias) or not .05 <= goal_bias <= .35:
        raise ValueError('this exercise requires 0.05 <= goal_bias <= 0.35')
    return (np.random.default_rng(seed), EdgeBudget(problem.edge_is_valid, max_checks),
            [problem.start.copy()], [None], perf_counter())


def tree_path(nodes, parents, index):
    indices = []
    while index is not None:
        indices.append(index)
        index = parents[index]
    return np.array([nodes[i] for i in reversed(indices)], dtype=np.float64)


def finish(path, budget, node_count, started):
    if path is None:
        return PlanningResult('BUDGET_EXHAUSTED', None, float('inf'), budget.used,
                              node_count, perf_counter()-started)
    path = np.array(path, dtype=np.float64, copy=True)
    cost = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return PlanningResult('FOUND', path, cost, budget.used, node_count, perf_counter()-started)
