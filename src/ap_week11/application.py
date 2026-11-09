"""Supplied validation and recording. No simulation and no replacement planner."""
import json
from pathlib import Path
import numpy as np
from .kinematics import forward_kinematics, capsule_segments


def validate_path(path, problem):
    """Return a checked independent Nx7 path under the declared teaching rule."""
    points = np.array(path, dtype=np.float64, copy=True)
    if points.ndim != 2 or points.shape[1] != 7 or not 1 <= len(points) <= 512 or not np.isfinite(points).all():
        raise ValueError('the supplied viewer accepts 1..512 finite seven-angle waypoints')
    if not np.array_equal(points[0], problem.start) or not np.array_equal(points[-1], problem.goal):
        raise ValueError('path must preserve exact requested endpoints')
    if np.any(points < problem.bounds[:, 0]) or np.any(points > problem.bounds[:, 1]):
        raise ValueError('path leaves the supplied planning bounds')
    if any(np.array_equal(a, b) for a, b in zip(points, points[1:])):
        raise ValueError('consecutive repeated waypoints are not a motion edge')
    if not all(problem.state_is_valid(q) for q in points):
        raise ValueError('a path state violates the declared teaching geometry')
    if not all(problem.edge_is_valid(a, b) for a, b in zip(points, points[1:])):
        raise ValueError('a path edge violates the declared sampled teaching geometry')
    return points


def obstacle_records(problem):
    if problem.obstacle_lower_m is None:
        return []
    return [{'center_m': ((problem.obstacle_lower_m+problem.obstacle_upper_m)/2).tolist(),
             'size_m': (problem.obstacle_upper_m-problem.obstacle_lower_m).tolist()}]


def save_plan(result, problem, output_dir, *, seed=7, step_size=.3, goal_bias=.15):
    """Save the actual returned path. This cannot turn failure into a route."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if result.path is None:
        record = {'status': result.status, 'path_q_rad': None, 'checks': result.checks,
                  'nodes': result.nodes, 'executed': False}
    else:
        path = validate_path(result.path, problem)
        record = {'status': result.status, 'path_q_rad': path.tolist(), 'cost_rad': result.cost,
            'checks': result.checks, 'nodes': result.nodes, 'elapsed_s': result.elapsed_s,
            'seed': seed, 'step_size_rad': step_size, 'goal_bias': goal_bias,
            'hand_positions_m': [forward_kinematics(q).position_m.tolist() for q in path],
            'capsule_states_m': [[[a.tolist(), b.tolist(), r] for a, b, r in capsule_segments(q)] for q in path],
            'obstacles': obstacle_records(problem), 'executed': False,
            'scope': 'Source-fitted enclosing capsules/ground and any declared box, fixed-open fingers, sampled edges. No self-collision, continuous-motion certificate or physics acceptance.'}
    target = out / 'path.json'
    target.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if result.path is not None:
        (out / 'planned_configurations.html').write_text(viewer_html(record), encoding='utf-8')
    else:
        (out / 'planned_configurations.html').write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Current planning status</title>'
            '<h1>'+record['status']+'</h1><p>This attempt has no accepted planned path. No physics execution is recorded.</p></html>',
            encoding='utf-8')
    return target


def viewer_html(record):
    """A supplied labelled kinematic view of saved data, never a fake run screen."""
    data = json.dumps(record, allow_nan=False).replace('</', '<\\/')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Planned configurations</title>
<style>body{font:22px 'Times New Roman';margin:24px;color:black;background:white;max-width:1000px}svg{border:1px solid black;width:100%;height:auto}label{display:block;margin:12px 0}code{font-family:Consolas}input{width:60%}</style>
<h1>Planned configurations</h1><p>Declared capsule/ground geometry and any saved obstacles. Kinematics only. This is not simulator execution.</p>
<svg viewBox="0 0 900 600" id="view"></svg><label>Waypoint <input id="frame" type="range" min="0" value="0"><span id="index"></span></label>
<p id="values"></p><p>Black line: hand positions at saved waypoints. The hand trace is not a separate Cartesian plan.</p>
<script>const D='''+data+''';const V=document.getElementById('view'),S=document.getElementById('frame');S.max=D.path_q_rad.length-1;
function project(p){return [450+440*(p[0]-.55*p[1]),510-440*(p[2]+.32*p[1])];}
function segment(a,b,width=2,color='black'){const A=project(a),B=project(b);return `<line x1="${A[0]}" y1="${A[1]}" x2="${B[0]}" y2="${B[1]}" stroke="${color}" stroke-width="${width}" stroke-linecap="round"/>`;}
function draw(){let x='';for(const [axis,label]of[[[.2,0,0],'X'],[[0,.2,0],'Y'],[[0,0,.2],'Z']]){x+=segment([0,0,0],axis);let p=project(axis);x+=`<text x="${p[0]+8}" y="${p[1]}" font-size="22">${label} (m)</text>`;}
for(const o of D.obstacles){let C=o.center_m,T=o.size_m,P=[];for(let k=0;k<8;k++)P.push(C.map((v,j)=>v+((k>>j&1)?1:-1)*T[j]/2));for(let k=0;k<8;k++)for(let j=0;j<3;j++)if(!(k>>j&1))x+=segment(P[k],P[k|1<<j],1);}
for(let i=1;i<D.hand_positions_m.length;i++)x+=segment(D.hand_positions_m[i-1],D.hand_positions_m[i],1);
const n=+S.value;for(const[a,b,r]of D.capsule_states_m[n]){x+=segment(a,b,r*880+2);x+=segment(a,b,r*880,'white');x+=segment(a,b,1);}
V.innerHTML=x;document.getElementById('index').textContent=` ${n+1}/${D.path_q_rad.length}`;document.getElementById('values').textContent='q (rad): '+D.path_q_rad[n].map(v=>v.toFixed(3)).join(', ');}
S.oninput=draw;draw();</script></html>'''
