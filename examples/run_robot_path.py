"""Supplied native replay of the student's current seven-angle RRT result."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
from uuid import uuid4
import webbrowser

from ap_week11.application import obstacle_records, validate_path
from ap_week11.domain import PlanningProblem
from ap_week11.rrt import single_tree_rrt


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--no-browser', action='store_true', help='save the preview without opening a browser')
    args = parser.parse_args()
    output = Path('artifacts/robot_rrt').resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_') + uuid4().hex[:12]
    current = output / 'runs' / run_id
    current.parent.mkdir(exist_ok=True)
    metadata = {'run_id': run_id, 'seed': args.seed,
                'artifact_directory': current.relative_to(output).as_posix()}
    report = {**metadata, 'status': 'RUNNING', 'executed': False, 'frames': []}
    write_json(output / 'result.json', report)
    write_json(output / 'path.json', {**metadata, 'status': 'RUNNING', 'units': 'rad', 'path_q_rad': None})
    (output / 'preview.html').write_text('<!doctype html><meta charset="utf-8"><h1>Week 11 execution: RUNNING</h1>'
        '<p>No completed frames for this run.</p>', encoding='utf-8')
    try:
        problem = PlanningProblem()
        planned = single_tree_rrt(problem, seed=args.seed, max_checks=1200, step_size=.3, goal_bias=.15)
        if planned.status != 'FOUND' or planned.path is None:
            raise RuntimeError('The planner did not find a path. No physics run was started.')
        path = validate_path(planned.path, problem)
        write_json(output / 'path.json', {**metadata, 'status': 'PLANNED', 'units': 'rad',
            'path_q_rad': path.tolist(), 'cost_rad': float(planned.cost),
            'checks': int(planned.checks), 'nodes': int(planned.nodes)})
        from ap_week11.robot_scene import run_scene
        report = run_scene(path, obstacle_records(problem), current, capture=args.capture)
        report.update(metadata)
        write_json(current / 'result.json', report)
        write_json(current / 'path.json', json.loads((output / 'path.json').read_text(encoding='utf-8')))
    except Exception as error:
        if (current / 'result.json').is_file():
            report = json.loads((current / 'result.json').read_text(encoding='utf-8'))
        report.update(metadata)
        report.update(status='FAILED', executed=False,
                      error={'type': type(error).__name__, 'message': str(error)})
        current.mkdir(parents=True, exist_ok=True)
        write_json(current / 'result.json', report)
        write_json(output / 'result.json', report)
        path_record = json.loads((output / 'path.json').read_text(encoding='utf-8'))
        path_record['status'] = 'FAILED'
        write_json(output / 'path.json', path_record)
        write_json(current / 'path.json', path_record)
        (output / 'preview.html').write_text('<!doctype html><meta charset="utf-8"><h1>Week 11 execution: FAILED</h1><p>'
            + escape(str(error)) + '</p><p>No earlier successful frames are shown as this result.</p>', encoding='utf-8')
        print('GPU execution FAILED:', error, file=sys.stderr)
        print('Run report:', output / 'result.json', file=sys.stderr)
        return 1
    write_json(output / 'result.json', report)
    path_record = json.loads((output / 'path.json').read_text(encoding='utf-8'))
    path_record['status'] = 'EXECUTED'
    write_json(output / 'path.json', path_record)
    write_json(current / 'path.json', path_record)
    preview = current / 'preview.html'
    relative = preview.relative_to(output).as_posix()
    message = '<p>Capture was not requested.</p>'
    if args.capture and preview.is_file():
        message = '<p><a href="' + escape(relative, quote=True) + '">Open this run\'s saved camera frames</a></p>'
    elif args.capture:
        message = '<p>Physics execution completed. Camera capture is unavailable. Inspect result.json.</p>'
    (output / 'preview.html').write_text('<!doctype html><meta charset="utf-8"><h1>Week 11 execution: EXECUTED</h1>'
        + message, encoding='utf-8')
    print('GPU execution complete')
    print('Execution status:', report['status'])
    print('Path file:', output / 'path.json')
    print('Run report:', output / 'result.json')
    if args.capture:
        print('Capture status:', report['capture_status'])
        print('Image preview:', output / 'preview.html')
        if preview.is_file() and not args.no_browser:
            try:
                webbrowser.open((output / 'preview.html').as_uri())
            except Exception as error:
                print('Open the saved preview manually:', error)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
