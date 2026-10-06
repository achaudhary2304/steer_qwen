"""Append steering training to the completed main LoRA checkpoint.

A waiting process polls stage statuses every five minutes. It never changes or
signals the parent trainer. The child initializes BOTH its bottleneck and LoRA
parameters from the main run; it does not combine independent pilot adapters.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concept_retrofit.io import save_json


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def parent_state(main, pid):
    main = Path(main)
    statuses = {}
    for stage in ('frozen', 'lora'):
        path = main/stage/'status.json'
        statuses[stage] = json.loads(path.read_text()) if path.exists() else {'state':'not_started'}
    complete = all(statuses[stage]['state'] == 'complete' for stage in statuses)
    checkpoint = main/'lora/best.pt'
    alive = process_alive(pid)
    if complete and checkpoint.exists() and not alive:
        return 'ready', statuses, checkpoint
    if not alive:
        return 'parent_stopped', statuses, checkpoint
    return 'waiting', statuses, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    parser.add_argument('--parent-pid', required=True, type=int)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default='configs/runnable/qwen35-08b-steering-scaled.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--merged', action='store_true', help='interleave four steering/capability phases on one checkpoint')
    parser.add_argument('--poll-seconds', type=int, default=300)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        raise ValueError('poll interval must be positive')
    status_file = Path(args.out+'-queue.json')
    while True:
        state, stages, checkpoint = parent_state(args.main, args.parent_pid)
        save_json(status_file, {'state':state,'main':args.main,'source_checkpoint':str(checkpoint),
                              'parent_pid':args.parent_pid,
                              'stages':{key:{k:v.get(k) for k in ('state','step','training_tokens')} for key,v in stages.items()}})
        print('STEERING_QUEUE '+state+' '+json.dumps({key:v['state'] for key,v in stages.items()}),flush=True)
        if state == 'parent_stopped':
            raise RuntimeError('Main trainer stopped before both stages completed; queue did not start training')
        if state == 'ready':
            break
        time.sleep(args.poll_seconds)
    command = [sys.executable, '-u', str(Path(__file__).with_name('merged_retrofit.py' if args.merged else 'steering_pilot.py')),
               '--data', args.data, '--checkpoint', str(checkpoint), '--out', args.out,
               '--config', args.config, '--lexicon-documents', '100000']
    if not args.merged:
        command += ['--expanded-prompts','--strengths','1','--max-new-tokens','64']
    if (Path(args.out)/'source.pt').exists():
        command += ['--resume']
    save_json(status_file, {'state':'running','main':args.main,'source_checkpoint':str(checkpoint),
                           'command':command})
    result = subprocess.run(command)
    completed = Path(args.out)/'status.json'
    success = result.returncode == 0 and completed.exists() and json.loads(completed.read_text())['state'] == 'complete'
    save_json(status_file, {'state':'complete' if success else 'failed', 'exit_code':result.returncode,
                           'source_checkpoint':str(checkpoint),'main':args.main})
    if not success:
        raise RuntimeError('Combined steering stage failed; check its log and saved checkpoint')
    print('COMBINED_STEERING_COMPLETE '+args.out,flush=True)


if __name__ == '__main__':
    main()
