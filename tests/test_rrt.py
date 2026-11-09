"""Four supplied checks of the student's RRT expansion. No test writing is required."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from ap_week11.domain import PlanningProblem
from ap_week11.rrt import single_tree_rrt
from ap_week11.application import validate_path


def test_valid_short_route_in_open_space():
    """Nearby IK endpoints and the actual RRT route are accepted edge by edge."""
    problem = PlanningProblem()
    assert problem.obstacle_lower_m is None and problem.obstacle_upper_m is None
    assert problem.edge_is_valid(problem.start, problem.goal)
    result = single_tree_rrt(problem, seed=7)
    assert result.status == 'FOUND'
    path = validate_path(result.path, problem)
    assert path.shape[1] == 7 and len(path) >= 2
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert np.all(lengths <= .3+1e-12)
    np.testing.assert_allclose(result.cost, lengths.sum(), rtol=1e-12, atol=1e-12)



def test_rejection_consumes_budget_without_adding_nodes():
    """An all-rejecting supplied oracle must terminate at the announced budget."""
    calls = []
    def reject(a, b):
        calls.append((a.copy(), b.copy()))
        return False
    problem = PlanningProblem(edge_is_valid=reject)
    result = single_tree_rrt(problem, max_checks=5)
    assert result.status == 'BUDGET_EXHAUSTED' and result.path is None
    assert result.checks == len(calls) == 5
    assert result.nodes == 1



class _ScriptedTargets:
    """Supply a deterministic stream while preserving the original sampler calls."""
    def __init__(self, targets):
        self.targets = iter(targets)

    def random(self):
        return .9

    def uniform(self, lower, upper):
        q = np.array(next(self.targets), dtype=float, copy=True)
        assert np.all(q >= lower) and np.all(q <= upper)
        return q


def _tree_fixture(goal, targets, edge_check):
    original = PlanningProblem()
    points = np.array([original.start, goal, *targets])
    bounds = np.column_stack((points.min(axis=0)-.01, points.max(axis=0)+.01))
    return replace(original, goal=goal, bounds=bounds,
                   state_is_valid=lambda q: True, edge_is_valid=edge_check)


def test_nearest_node_uses_first_inserted_on_tie():
    """Observed edge origins distinguish a tie from a nearer non-root node."""
    start = PlanningProblem().start.copy()
    def at(x):
        q = start.copy()
        q[0] += x
        return q
    targets = [at(1/16), at(-1/16), at(1/32), at(3/32)]
    goal = start.copy()
    goal[2] += .5  # Keep supplied goal completion outside this small tree fixture.
    proposals = []
    def accept(a, b):
        proposals.append((a.copy(), b.copy()))
        return True
    problem = _tree_fixture(goal, targets, accept)
    with patch('ap_week11.domain.np.random.default_rng', return_value=_ScriptedTargets(targets)):
        result = single_tree_rrt(problem, max_checks=4, step_size=.15)
    assert result.status == 'BUDGET_EXHAUSTED' and result.checks == 4 and result.nodes == 5
    assert len(proposals) == 4
    # The third target is exactly halfway between the start and first new node.
    np.testing.assert_array_equal(proposals[2][0], start)
    # The fourth target is closer to the earlier positive branch than to the root.
    np.testing.assert_array_equal(proposals[3][0], targets[0])
    for (_, reached), target in zip(proposals, targets):
        np.testing.assert_array_equal(reached, target)


def test_multibranch_path_preserves_checked_parents_and_step_bound():
    """Recover only actual checked edges when growth returns to an earlier branch."""
    start = PlanningProblem().start.copy()
    def at(x, y):
        q = start.copy()
        q[:2] += [x, y]
        return q
    targets = [at(.25, 0), at(0, .25), at(.25, 0), at(.25, .125)]
    goal = at(.25, .20)
    checked = []
    def accept(a, b):
        checked.append((a.copy(), b.copy()))
        return True
    problem = _tree_fixture(goal, targets, accept)
    with patch('ap_week11.domain.np.random.default_rng', return_value=_ScriptedTargets(targets)):
        result = single_tree_rrt(problem, max_checks=5, step_size=.15)
    assert result.status == 'FOUND' and result.checks == 5 and result.nodes == 6
    assert len(checked) == 5
    np.testing.assert_allclose(checked[0][1], at(.15, 0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(checked[1][1], at(0, .15), rtol=0, atol=1e-12)
    # Growth returns to the first branch, not the most recently inserted branch.
    np.testing.assert_array_equal(checked[2][0], checked[0][1])
    assert result.path.shape == (5, 7)
    assert not any(np.array_equal(q, checked[1][1]) for q in result.path)
    np.testing.assert_array_equal(result.path[0], problem.start)
    np.testing.assert_array_equal(result.path[-1], problem.goal)
    for a, b in zip(result.path, result.path[1:]):
        assert any(np.array_equal(a, left) and np.array_equal(b, right) for left, right in checked)
    assert all(np.linalg.norm(b-a) <= .15+1e-12 for a, b in checked)
    lengths = np.linalg.norm(np.diff(result.path, axis=0), axis=1)
    assert np.all(lengths <= .15+1e-12)
    np.testing.assert_allclose(result.cost, lengths.sum(), rtol=1e-12, atol=1e-12)
