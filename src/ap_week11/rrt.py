"""Supplied single-tree scaffold with one student expansion region."""
import numpy as np
from .domain import distance, steer, prepare_tree, tree_path, finish


def single_tree_rrt(problem, seed=7, max_checks=1200, step_size=.3, goal_bias=.15):
    """Return a PlanningResult with a start-to-goal (N,7) path, or failure.

    Edit only TODO 1. Reuse the relevant Week 10 nearest-node, extension,
    accepted-edge and node/parent insertion lines if helpful. Sampling, goal
    completion, path recovery and result packaging are already supplied.

    problem.start and problem.goal are read-only (7,) angle arrays in radians.
    problem.bounds has shape (7,2). nodes stores independent seven-angle arrays.
    parents[i] is the index of the node from which nodes[i] was accepted.
    The start has parent None. Do not change these input arrays.

    Each loop supplies target, a (7,) angle array selected from the goal or
    the bounds. In TODO 1, assign nearest to the index of the closest stored
    node. distance(a,b) returns Euclidean joint distance in radians. Equal
    distances choose the first inserted node.

    Assign new_state using steer from that nearest node toward target with
    step_size. steer(a,b,step_size) returns an independent (7,) array at most
    step_size radians from a. It does not check the edge.

    Call budget.check(a,b) for the edge from the nearest node to new_state.
    It consumes one edge check even on rejection. On False, continue without
    changing nodes or parents. On True, append new_state to nodes and append
    its actual nearest-node index to parents. Leave goal tests and returns
    to the supplied code after TODO 1.
    """
    rng, budget, nodes, parents, started = prepare_tree(problem, seed, max_checks, step_size, goal_bias)
    if np.array_equal(problem.start, problem.goal):
        return finish(np.array(nodes), budget, len(nodes), started)
    while budget.remaining:
        target = problem.goal if rng.random() < goal_bias else rng.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
        # ----- BEGIN TODO 1 -----
        raise NotImplementedError('TODO 1: choose, check and append one nearest-node extension')
        # ----- END TODO 1 -----
        last = len(nodes)-1
        if np.array_equal(new_state, problem.goal):
            return finish(tree_path(nodes, parents, last), budget, len(nodes), started)
        if distance(new_state, problem.goal) <= step_size and budget.remaining:
            if budget.check(new_state, problem.goal):
                nodes.append(problem.goal.copy())
                parents.append(last)
                return finish(tree_path(nodes, parents, len(nodes)-1), budget, len(nodes), started)
    return finish(None, budget, len(nodes), started)
