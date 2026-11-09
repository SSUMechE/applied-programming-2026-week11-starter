"""Run the student's one-tree implementation and record its actual result."""
import argparse
from pathlib import Path
from ap_week11.domain import PlanningProblem, PlanningResult
from ap_week11.rrt import single_tree_rrt
from ap_week11.application import save_plan, validate_path

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=7)
p.add_argument('--step-size', type=float, default=.3, help='0.15..0.50 radians')
p.add_argument('--goal-bias', type=float, default=.15, help='0.05..0.35')
args = p.parse_args()
problem = PlanningProblem()
out = Path(__file__).resolve().parents[1] / 'artifacts/rrt'
save_plan(PlanningResult('RUNNING', None, float('inf'), 0, 0, 0.), problem, out)
try:
    result = single_tree_rrt(problem, seed=args.seed, max_checks=1200,
                             step_size=args.step_size, goal_bias=args.goal_bias)
    if result.path is not None:
        validate_path(result.path, problem)
except Exception:
    save_plan(PlanningResult('FAILED', None, float('inf'), 0, 0, 0.), problem, out)
    raise
print(f'RRT: {result.status} | checks={result.checks} | nodes={result.nodes}')
if result.path is not None:
    print(f'Joint-space length: {result.cost:.3f} rad | waypoints={len(result.path)}')
    print('Sampled teaching geometry: PASS')
else:
    print('No path found within the edge-check budget. No execution was attempted.')
save_plan(result, problem, out, seed=args.seed, step_size=args.step_size, goal_bias=args.goal_bias)
print('Saved: artifacts/rrt/path.json')
if result.path is not None:
    print('Open: artifacts/rrt/planned_configurations.html (kinematics only)')
