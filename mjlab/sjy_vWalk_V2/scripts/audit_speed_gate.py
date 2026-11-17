"""Offline shadow comparison of V2 gate candidates; never changes a policy or cap."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def mean(xs):
    return sum(xs)/len(xs)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def audit_window(rows, start, duration, cap=1.):
    end = start+duration
    # The logger measures position and heading relative to reset at t=0.
    if start == 0:
        x0 = y0 = heading0 = 0.
    else:
        anchors = [r for r in rows if abs(r['time_s']-start) < 1e-7]
        if len(anchors) != 1:
            raise ValueError('Window start needs an exact logged sample')
        x0, y0, heading0 = (anchors[0][k] for k in ('forward_m','lateral_m','heading_deg'))
    measured = [r for r in rows if start+1.-1e-8 <= r['time_s'] <= end+1e-8]
    if duration < 1.5 or len(measured) < 20 or rows[-1]['time_s'] < end-1e-8:
        raise ValueError('Incomplete diagnostic window')
    target = measured[0]['command_vx_mps']
    if any(abs(r['command_vx_mps']-target)>1e-8 for r in measured):
        raise ValueError('Command changed inside window')
    heading0 = math.radians(heading0)
    counts = {k:0 for k in ('vx','vy','instant_yaw','heading','lateral','non_yaw_joint','original_joint')}
    for r in measured:
        dx,dy = r['forward_m']-x0,r['lateral_m']-y0
        flags = dict(vx=abs(r['body_vx_mps']-target)<=.15+.15*abs(target),
            vy=abs(r['body_vy_mps'])<=.15,
            instant_yaw=abs(r['body_wz_radps'])<=.2,
            heading=abs(wrap(math.radians(r['heading_deg'])-heading0))<=math.radians(10),
            lateral=abs(-math.sin(heading0)*dx+math.cos(heading0)*dy)<=.15)
        for k,v in flags.items():
            counts[k]+=v
        counts['non_yaw_joint']+=all(flags[k] for k in ('vx','vy','heading','lateral'))
        counts['original_joint']+=all(flags.values())
    fractions={k:v/len(measured) for k,v in counts.items()}
    rates=[r['body_wz_radps'] for r in measured]
    signed=mean(rates)
    rms=math.sqrt(mean([x*x for x in rates]))
    base=fractions['non_yaw_joint'] >= .8
    eligible=target >= .8*cap  # Supplied diagnostics use vy=yaw command zero.
    return dict(start_s=start,end_s=end,command_vx_mps=target,eligible=eligible,
        samples=len(measured),pass_fractions=fractions,
        mean_body_wz_radps=signed,rms_body_wz_radps=rms,
        original_pass=eligible and fractions['original_joint']>=.8,
        candidate_mean_only_pass=eligible and base and abs(signed)<=.2,
        candidate_mean_rms035_pass=eligible and base and abs(signed)<=.2 and rms<=.35,
        candidate_mean_rms050_pass=eligible and base and abs(signed)<=.2 and rms<=.5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audits=[]
    for path in sorted(args.diagnostics_dir.rglob('state.csv')):
        summary=json.loads(path.with_name('summary.json').read_text(encoding='utf-8'))
        if summary['status']!='duration_completed':
            raise ValueError(f'Incomplete/error run requires separate handling: {path}')
        with path.open(encoding='utf-8',newline='') as f:
            rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        if not rows or not all(all(math.isfinite(v) for v in r.values()) for r in rows):
            raise ValueError(f'Invalid numeric rows: {path}')
        if any(abs(r['time_s']-(i+1)*.02)>1e-7 for i,r in enumerate(rows)):
            raise ValueError('Expected uninterrupted 50Hz samples')
        windows=[]
        for duration in (2.,3.,4.):
            # Disjoint within each duration; different duration sets overlap.
            for index in range(int((rows[-1]['time_s']+1e-7)//duration)):
                windows.append(audit_window(rows,index*duration,duration))
        eligible=[w for w in windows if w['eligible']]
        counts={key:sum(w[key] for w in eligible) for key in
                ('original_pass','candidate_mean_only_pass','candidate_mean_rms035_pass','candidate_mean_rms050_pass')}
        audits.append(dict(source_folder=path.parent.name,
            state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint=summary['checkpoint'],command_vx_mps=summary['command_vx_mps'],
            eligible_windows=len(eligible),passed_windows=counts,windows=windows))
    if not audits:
        raise ValueError('No state.csv found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(
        note='Hypothetical windows from fixed-command play; NOT training gate replay or independent trials. No promotion is performed.',
        threshold_note='RMS .35/.50 are sensitivity candidates, not validated safety or promotion thresholds.',
        audits=audits),indent=2),encoding='utf-8')
    for a in audits:
        print(a['command_vx_mps'],a['eligible_windows'],a['passed_windows'])


if __name__=='__main__':
    main()

