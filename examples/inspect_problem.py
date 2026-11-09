"""Complete supplied example. Runs before the RRT TODO is completed."""
import numpy as np
from ap_week11.domain import PlanningProblem, START_POSITION_M, GOAL_POSITION_M, IK_SEED_Q
from ap_week11.kinematics import forward_kinematics, inverse_kinematics

problem = PlanningProblem()
start_q = inverse_kinematics(START_POSITION_M, IK_SEED_Q)
goal_q = inverse_kinematics(GOAL_POSITION_M, start_q)
pose = forward_kinematics(goal_q)
print('Start target (m):', START_POSITION_M.tolist())
print('Goal target (m):', GOAL_POSITION_M.tolist())
print('Start q from IK (rad):', np.round(start_q, 6).tolist())
print('Goal q from IK (rad):', np.round(goal_q, 6).tolist())
print('Goal hand from FK (m):', np.round(pose.position_m, 6).tolist())
print('Goal position error (m):', f'{np.linalg.norm(pose.position_m-GOAL_POSITION_M):.3e}')
print('Direct edge allowed:', problem.edge_is_valid(problem.start, problem.goal))
print('Position-only IK: orientation is free. Both FK and IK are supplied.')
print('Open space. All seven angles may change. Base and open fingers are fixed.')
print('Sampled ground check. Self-collision and physics are separate checks.')
