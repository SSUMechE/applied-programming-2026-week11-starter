"""Supplied Week 11 native Panda execution, adapted from released Week 10.

This runner accepts an already planned seven-angle path. It does not replan.
Geometry checks, actual dynamics, tracking and camera capture remain distinct.
The empty hand carries no vessel. A completed run is not a bead-retention test.
"""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path
import time

import numpy as np

DT_S = 0.01
INITIAL_SETTLE_STEPS = 60
RAMP_STEPS = 80
HOLD_STEPS = 80
ENDPOINT_TOLERANCE_RAD = 0.025
WALL_LIMIT_S = 600.0


def run_scene(path_q, obstacles, output_dir, capture=True):
    """Execute N-by-7 radians and supplied world boxes, recording observations.

    obstacles may be None or an empty sequence for the open-space exercise.
    Every supplied obstacle has center_m and size_m, each a three-vector in metres.
    output_dir must not exist. Input waypoints are unchanged throughout replay.
    The returned result and result.json describe only this native execution.
    """
    points = np.asarray(path_q, dtype=float)
    if points.ndim != 2 or points.shape[1] != 7 or not 1 <= len(points) <= 512:
        raise ValueError('path_q must contain 1 to 512 rows of seven angles')
    if not np.isfinite(points).all():
        raise ValueError('Every joint angle must be finite')
    if len(points) > 1 and np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0):
        raise ValueError('Adjacent waypoints must differ')
    boxes = []
    for index, item in enumerate(() if obstacles is None else obstacles):
        center, size = np.asarray(item['center_m'], float), np.asarray(item['size_m'], float)
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite(np.r_[center, size]).all() or np.any(size <= 0):
            raise ValueError('Obstacle centre and positive size must be three finite metres')
        boxes.append({'id': str(item.get('id', f'box_{index}')),
                      'center_m': center.tolist(), 'size_m': size.tolist()})
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'executed': False, 'path_q_rad': points.tolist(),
              'obstacles': boxes, 'scene_case': 'open_space' if not boxes else 'fixed_boxes',
              'dt_s': DT_S, 'frames': [],
              'capture_status': 'PENDING' if capture else 'NOT_REQUESTED',
              'capture_errors': [],
              'scope': 'One empty Panda arm. Native observed tracking and solver contacts. '
                       'No vessel, bead transport, continuous collision proof or real-robot safety claim.'}
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    import torch
    import genesis as gs
    if not torch.cuda.is_available():
        result.update(status='FAILED', error='The existing course CUDA environment is required')
        (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        raise RuntimeError(result['error'])
    torch.set_num_threads(2)
    start = time.monotonic()
    step_limit = INITIAL_SETTLE_STEPS + (RAMP_STEPS + HOLD_STEPS) * (len(points) - 1)
    steps, trace, waypoints = 0, [], []
    maximum = {'joint_violation_rad': 0.0, 'joint_speed_rad_s': 0.0,
               'ramp_error_rad': 0.0, 'self_contacts': 0,
               'nonbase_floor_contacts': 0, 'obstacle_contacts': 0}

    def emit(event, **fields):
        print(json.dumps({'event': event, 'step': steps, **fields}, allow_nan=False), flush=True)

    def array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if not np.isfinite(value).all():
            raise RuntimeError('A native simulator observation is nonfinite')
        return value

    emit('initializing', waypoints=len(points))
    initialized = False
    try:
        gs.init(backend=gs.cuda, precision='32', logging_level='warning', seed=0)
        initialized = True
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT_S, gravity=(0, 0, -9.81)),
            rigid_options=gs.options.RigidOptions(enable_collision=True,
                enable_joint_limit=True, box_box_detection=True),
            vis_options=gs.options.VisOptions(background_color=(1, 1, 1),
                ambient_light=(0.30, 0.30, 0.30), shadow=True, plane_reflection=False,
                lights=[gs.options.vis.DirectionalLight(dir=(-1, -0.6, -4),
                    color=(1, 1, 1), intensity=2.8)]),
            renderer=gs.renderers.Rasterizer(), show_viewer=False, show_FPS=False)
        floor = scene.add_entity(gs.morphs.Plane(),
            surface=gs.surfaces.Default(color=(0.82, 0.84, 0.87), roughness=0.85))
        barriers = [scene.add_entity(gs.morphs.Box(pos=box['center_m'], size=box['size_m'], fixed=True),
                    surface=gs.surfaces.Default(color=(0.85, 0.62, 0.22), roughness=0.8)) for box in boxes]
        arm = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
            surface=gs.surfaces.Default(roughness=0.4, metallic=0.05))
        camera = scene.add_camera(res=(1200, 900), pos=(1.30, -1.60, 1.00),
            lookat=(0.10, 0, 0.45), fov=40, GUI=False) if capture else None
        scene.build()
        motors, fingers = np.arange(7), np.arange(7, 9)
        arm.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 500, 500]))
        arm.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 50, 50]))
        force = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100], dtype=float)
        arm.set_dofs_force_range(-force, force)
        lower, upper = (array(item) for item in arm.get_dofs_limit(motors))
        if np.any(points < lower) or np.any(points > upper):
            raise ValueError('A planned target exceeds native Panda joint limits')
        base_index = arm.get_link('link0').idx
        hand = arm.get_link('hand')
        arm.set_qpos(np.r_[points[0], 0.04, 0.04])  # Initial state only.
        emit('scene_built')

        def step(target, segment, phase):
            nonlocal steps
            if steps >= step_limit or time.monotonic() - start > WALL_LIMIT_S:
                raise RuntimeError('Bounded native execution budget exceeded')
            arm.control_dofs_position(target, motors)
            arm.control_dofs_position(np.array([0.04, 0.04]), fingers)
            scene.step()
            steps += 1
            q = array(arm.get_dofs_position(motors))
            velocity = array(arm.get_dofs_velocity(motors))
            xyz = array(hand.get_pos()).reshape(3)
            error = float(np.max(np.abs(q - target)))
            violation = float(max(0.0, np.max(lower - q), np.max(q - upper)))
            self_count = len(arm.get_contacts(with_entity=arm)['geom_a'])
            floor_contact = arm.get_contacts(with_entity=floor)
            link_a = array(floor_contact['link_a']).astype(int)
            link_b = array(floor_contact['link_b']).astype(int)
            floor_count = int(np.count_nonzero((link_a != base_index) & (link_b != base_index)))
            obstacle_observations = [arm.get_contacts(with_entity=box) for box in barriers]
            obstacle_count = sum(len(contact['geom_a']) for contact in obstacle_observations)
            maximum['joint_violation_rad'] = max(maximum['joint_violation_rad'], violation)
            maximum['joint_speed_rad_s'] = max(maximum['joint_speed_rad_s'], float(np.max(np.abs(velocity))))
            maximum['self_contacts'] = max(maximum['self_contacts'], self_count)
            maximum['nonbase_floor_contacts'] = max(maximum['nonbase_floor_contacts'], floor_count)
            maximum['obstacle_contacts'] = max(maximum['obstacle_contacts'], obstacle_count)
            if phase == 'ramp':
                maximum['ramp_error_rad'] = max(maximum['ramp_error_rad'], error)
            trace.append({'step': steps, 'time_s': steps * DT_S, 'segment': segment, 'phase': phase,
                'hand_x_m': float(xyz[0]), 'hand_y_m': float(xyz[1]), 'hand_z_m': float(xyz[2]),
                'max_joint_error_rad': error, 'self_contacts': self_count,
                'nonbase_floor_contacts': floor_count, 'obstacle_contacts': obstacle_count,
                **{f'q{i + 1}_rad': float(value) for i, value in enumerate(q)}})
            if violation > 1e-4 or self_count or floor_count or obstacle_count:
                names = {link.idx: link.name for link in arm.links}
                rows = []
                for box_id, contact in enumerate(obstacle_observations):
                    for k in range(len(contact['geom_a'])):
                        la = int(array(contact['link_a'])[k])
                        lb = int(array(contact['link_b'])[k])
                        rows.append({'obstacle_id': boxes[box_id]['id'],
                            'link_a': names.get(la, str(la)), 'link_b': names.get(lb, str(lb)),
                            'geom_a': int(array(contact['geom_a'])[k]),
                            'geom_b': int(array(contact['geom_b'])[k])})
                result['failure_observation'] = {'step': steps, 'segment': segment, 'phase': phase,
                    'target_q_rad': np.asarray(target).tolist(), 'measured_q_rad': q.tolist(),
                    'obstacle_contact_pairs': rows}
                raise RuntimeError('Observed native joint-limit or solver-contact condition failed')

        def reached(index):
            q = array(arm.get_dofs_position(motors))
            xyz = array(hand.get_pos()).reshape(3)
            error = float(np.max(np.abs(q - points[index])))
            waypoints.append({'waypoint': index, 'step': steps, 'target_q_rad': points[index].tolist(),
                'measured_q_rad': q.tolist(), 'hand_position_m': xyz.tolist(), 'max_error_rad': error})
            if error > ENDPOINT_TOLERANCE_RAD:
                raise RuntimeError('Panda did not settle within the endpoint tolerance')
            if camera is not None:
                name = f'{index:02d}_waypoint.png'
                try:
                    rgb, _, _, _ = camera.render(rgb=True)
                    gs.tools.save_img_arr(rgb, filename=str(out / name))
                    result['frames'].append({'waypoint': index, 'step': steps, 'file': name})
                except Exception as error:
                    result['capture_errors'].append({'waypoint': index, 'type': type(error).__name__, 'message': str(error)})
                    emit('capture_failed', waypoint=index, message=str(error))
            emit('waypoint_reached', waypoint=index, max_error_rad=error)

        for _ in range(INITIAL_SETTLE_STEPS):
            step(points[0], 'initial', 'initial_settle')
        reached(0)
        for index in range(1, len(points)):
            left, right = points[index - 1], points[index]
            segment = f'{index - 1}->{index}'
            for frame in range(1, RAMP_STEPS + 1):
                u = frame / RAMP_STEPS
                step(left + (u * u * (3 - 2 * u)) * (right - left), segment, 'ramp')
            for _ in range(HOLD_STEPS):
                step(right, segment, 'hold')
            reached(index)
        result.update(status='EXECUTED', executed=True,
            checks={'joint_limits': maximum['joint_violation_rad'] <= 1e-4,
                    'waypoint_tracking': all(w['max_error_rad'] <= ENDPOINT_TOLERANCE_RAD for w in waypoints),
                    'observed_contacts': not any(maximum[k] for k in ['self_contacts','nonbase_floor_contacts','obstacle_contacts'])})
        if capture:
            result['capture_status'] = 'FAILED' if result['capture_errors'] else 'SAVED'
        emit('execution_complete', status=result['status'], steps=steps)
    except BaseException as error:
        result.update(status='FAILED', executed=False,
                      error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        result.update(steps=steps, simulation_time_s=steps * DT_S,
            wall_time_s=time.monotonic() - start, maximum=maximum,
            endpoint_tolerance_rad=ENDPOINT_TOLERANCE_RAD, waypoints=waypoints)
        (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        if trace:
            with (out / 'trace.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                writer.writeheader()
                writer.writerows(trace)
        if result['frames']:
            content = ''.join('<figure><img src="' + escape(f['file'], quote=True) + '"><figcaption>Waypoint '
                      + str(f['waypoint']) + ', simulation step ' + str(f['step']) + '</figcaption></figure>' for f in result['frames'])
            (out / 'preview.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<title>Week 11 native Panda frames</title><style>body{font:18px Georgia,serif;margin:2rem;color:black;background:white}'
                'figure{margin:1rem 0}img{max-width:100%;height:auto}</style><h1>Native Panda execution: '
                + escape(result['status']) + '</h1><p>Actual saved camera frames from this run. '
                'The empty hand carries no vessel. These images are not a bead-transport result.</p>' + content + '</html>', encoding='utf-8')
        if initialized:
            gs.destroy()
    return result
