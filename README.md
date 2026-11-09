# Week 11: Seven-Joint Path Design

This is the public Week 11 starter template. Read the Week 11 Reading and its integrated Assignment first. The starter includes supplied FK/IK, an unfinished RRT exercise and the checks used in the Reading.

## 1. Purpose and edit target

Move one fixed-base Panda hand between two nearby positions in open space. The targets are [0.35, -0.10, 0.72] m and [0.35, -0.02, 0.72] m in the world frame. Supplied position-only IK converts each target into seven joint angles. All seven arm angles may vary. There is no box obstacle, and the direct connection is allowed by the supplied teaching geometry.

Edit only TODO 1 in `src/ap_week11/rrt.py`, inside `single_tree_rrt`. Remove its `raise NotImplementedError(...)` and write between BEGIN/END. You may reuse the relevant Week 10 lines for nearest-node selection, extension, edge rejection and node/parent insertion. Keep the supplied sampling, goal-completion code, signatures and all other code unchanged.

The supplied loop selects `target`, a seven-angle array in radians. In TODO 1, assign `nearest` to the closest stored node's index, choosing the first inserted node on a tie. Use the supplied `steer` to assign the independent seven-angle array `new_state`. Check the edge from the nearest node with `budget.check`. On rejection, continue without changing the tree. On acceptance, append `new_state` to `nodes` and its actual parent index to `parents`. Each `nodes` entry has shape `(7,)`, and `parents[i]` names the stored node from which entry `i` was accepted. The function's docstring specifies these names and calls locally. Sampling, checked goal connection, path recovery and all returns are supplied.

Both forward kinematics (FK) and inverse kinematics (IK), source-model data, teaching checks, recording, native replay and four tests are supplied. Do not implement FK or IK, write tests, or change robot, targets, physics or checker settings. This is an empty-hand path exercise. Actual recorded bead behavior in the Reading explains the later transport difficulty.

## 2. Prepare the repository and existing environment

Open the [Week 11 template](https://github.com/SSUMechE/applied-programming-2026-week11-starter), choose **Use this template → Create a new repository**, and create your private `applied-programming-w11-YOUR-STUDENT-ID` repository. Invite the exact collaborator account `SSUMechE`.

Open **Anaconda Prompt**. Replace the course folder, GitHub ID and student ID with your values. Run each command on its own line.

```bat
cd /d "C:\your-course-folder"
git clone https://github.com/YOUR-GITHUB-ID/applied-programming-w11-YOUR-STUDENT-ID.git
cd applied-programming-w11-YOUR-STUDENT-ID
git remote -v
set PYTHONUTF8=1
conda activate applied-programming-cuda
python -m pip install "setuptools>=80.9.0" pytest==9.1.1
python -m pip install -e . --no-deps --no-build-isolation
python scripts/verify_environment.py
```

Both remote addresses must identify your own private Week 11 repository. Reuse the existing all-in-one course environment. Do not create a Week 11 environment or replace PyTorch, CUDA, Genesis or NumPy. Expected final status: `ENVIRONMENT PASS`. It confirms the checkout, imports and a small CUDA calculation without calling the unfinished RRT.

If the course environment is missing, follow the previously distributed [all-in-one setup command](https://github.com/SSUMechE/applied-programming-2026-week05-starter/releases/download/cuda-setup-v12-20261012/INSTALL_COMMAND.txt), then return here. Do not rerun that installer for an existing environment. Report an import/version failure instead of independently replacing shared packages.

## 3. Inspect before implementation

This complete example runs before TODO 1 is finished:

```bat
python examples/inspect_problem.py
```

It prints the two targets, their seven-angle IK results, the goal position checked by FK and `Direct edge allowed: True`. In `src/ap_week11/kinematics.py`, `inverse_kinematics(position_m, seed_q)` accepts a world position in metres and a nearby seven-angle seed in radians. It returns a new seven-angle array with position error at most 0.00001 m. It raises an explicit error if this local search fails. It does not constrain hand orientation or plan a route. The supplied case first uses a comfortable seed for the start target, then uses that returned start state as the seed for the nearby goal.

`forward_kinematics(goal_q)` returns a `HandPose`. Its `position_m` is the world hand position. Its `rotation_matrix` is the hand orientation. Neither is the seven-angle input. The example assigns that returned object to `pose`, then prints its position rounded to six decimal places. Both functions run before the RRT TODO is completed. Orientation is free in this unloaded exercise. This IK is not a container-upright controller.

The supplied checker uses capsules fitted around the original collision-mesh vertices and fingertip boxes. The fingers stay at their fixed open setting. It checks source joint limits and these enclosing shapes against the ground at sampled connecting joint states. The small sampling bounds cover the two endpoint states with 0.10 rad margin per joint. It does not check self-collision, continuous motion or carried contents. Passing this check is a planning result, separate from the native replay's observed contacts and tracking.

## 4. Implement, plan and check

After completing TODO 1, run:

```bat
python examples/run_rrt.py
python -m pytest -q
```

The four supplied checks cover a valid short route, rejected edges consuming the budget without insertion, nearest-node selection with the first inserted node winning a tie, and correct parent-chain recovery with bounded steps in a branching tree. The last two use deterministic seven-angle tree fixtures and supplied oracles. You do not write tests, sampling or final goal completion. A completed correct implementation ends with `4 passed`.

The example reports status, edge checks, tree nodes and joint-space length in radians. It saves the actual returned route as `artifacts/rrt/path.json`. Open `artifacts/rrt/planned_configurations.html` to inspect its declared geometry and saved configurations. This is a kinematic view, not a physical execution recording. A failed attempt records its failure rather than substituting an instructor route or reusing an earlier success.

Valid implementations may produce different paths and counts. Exact path length, tree size and execution time are not grading targets. A budget-exhausted result does not prove a route is impossible.

You may inspect a different design choice while keeping seed and problem fixed:

```bat
python examples/run_rrt.py --seed 7 --step-size 0.2
```

This optional command replaces the saved planning result. After it finishes, open `artifacts/rrt/planned_configurations.html` to inspect that new plan.

The permitted step size is 0.15–0.50 rad. The optional `--goal-bias` choice is 0.05–0.35. Defaults are 0.30 rad and 0.15. Smaller or larger is not automatically better. No optimization run, additional report or copied output is required.

Use the supplied native runner after the planning checks:

```bat
python examples/run_robot_path.py --capture --no-browser
```

This command replans with the default step size 0.30 rad and goal bias 0.15. It does not replay an earlier optional parameter comparison. Inspect this run's execution status and open `artifacts/robot_rrt/preview.html` to inspect its saved camera frames. The `--no-browser` flag leaves the browser launch to you. This native output is separate from the planning view under `artifacts/rrt`. Native tracking/contact checks, sampled planning geometry and real-robot safety are different claims. Passing the four CPU tests does not establish physical execution. Generated artifacts are checking support, not separate LMS uploads.

## 5. Submit the required revision

From your private repository root:

```bat
git diff
git status --short
git add src/ap_week11/rrt.py
git diff --staged
git commit -m "Complete Week 11 joint-space RRT"
git push origin main
git status --short
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Press `q` to leave a diff pager. Confirm an empty final status and matching full local/remote commit IDs. Check private visibility and `SSUMechE` access. A correctly addressed invitation sent on time and still awaiting instructor acceptance is not a student omission.

Submit these two items in LMS within one week of the laboratory session, by the exact LMS date and time:

- Your private repository URL.
- The matching full 40-character commit ID.

GitHub upload alone is not LMS submission. Follow the announced mandatory-format and submission policy. A failed algorithm check alone is not an automatic zero for the entire Assignment. No additional ZIP, report, student tests or screenshot submission is required.
