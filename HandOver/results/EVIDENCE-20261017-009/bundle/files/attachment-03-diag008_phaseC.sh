#!/usr/bin/env bash
# PROFILE-DIAG-20261017-008 phase C (read-only acceptance), transcribed from
# HandOver/L2S-20261017-008.md lines 180-214, plus the closing read-only GPU query
# allowed by line 225 (new output file, no preemption).
set -euo pipefail
cd /home/wangguangjie/djs/association-code || exit 1
# shellcheck source=/home/wangguangjie/djs/diag008_wrapper.sh
source /home/wangguangjie/djs/diag008_wrapper.sh
diag008_init

task_attempt=${ATTEMPT:-1}
task_prep=runs/PREP-NATIVE-20261017-007-attempt-1
task_profile="runs/PROFILE-DIAG-20261017-008-attempt-$task_attempt"
task_gpu_uuid=$(cat "$task_root/allocation-$task_attempt/stdout.log")
echo "C_START=$(date +'%Y-%m-%dT%H:%M:%S%z') uuid=$task_gpu_uuid"

if diag008_passed "profile-$task_attempt"; then
  if diag008_start review && diag008_run review "accept-$task_attempt" env CUDA_VISIBLE_DEVICES= \
    "$task_python" - "$task_prep" "$task_profile" "$task_gpu_uuid" <<'PY'
import json, math, pathlib, sys
from association.config import load_config
from association.runtime import code_fingerprint, digest, object_hash
p,out=map(pathlib.Path,sys.argv[1:3]); c=load_config('configs/pilot.json')
m=json.loads((p/'manifest.json').read_text()); r=json.loads((out/'profile.json').read_text()); run=json.loads((out/'run.json').read_text())
assert r['status']==run['status']=='COMPLETED' and run['profile_sha256']==digest(out/'profile.json')
assert r['manifest_sha256']==run['manifest_sha256']==m['manifest_sha256']
assert r['config_hash']==run['config_hash']==object_hash(c) and run['config']==c
assert r['code_fingerprint']==run['code_fingerprint']==code_fingerprint()
assert run['provenance']['cuda_visible_devices']==sys.argv[3]
assert (r['role'],r['branch'],r['seed'])==('A0','ce',20)
assert r['training_scene_ids']==r['normalizer']['training_scene_ids']==m['splits']['T_core']
assert r['completed_steps']==r['requested_steps']==run['completed_steps']==6
assert r['warmup_steps']==5 and r['measured_steady_steps']==1 and r['parameters_changed']
assert r['gpu_measured'] and r['device']=='cuda:0' and r['peak_reserved_bytes']>=r['peak_allocated_bytes']>0
assert r['memory_snapshot_completed_steps']==6 and 'memory_measurement_error' not in r
assert r['joint_completion'] and r['completion_weight']==0.1 and r['pair_batch_size']==r['batch_tracks']==64
assert not r['eligible_reference'] and not run['eligible_reference'] and not r['checkpoint_written'] and not list(out.rglob('*.pt'))
assert not r['validation_measurements_read'] and not r['test_measurements_read']
assert len(r['step_phase_seconds'])==len(r['step_positive_pair_counts'])==6
for name in ('step_seconds','losses','gradient_norms'):
    assert len(r[name])==6 and all(math.isfinite(x) for x in r[name])
for phases in r['step_phase_seconds']:
    assert set(phases)=={'batch_preparation','encode_tracks','association_loss','completion_auxiliary','backward_optimizer'}
    assert all(math.isfinite(x) and x>=0 for x in phases.values())
assert all(0<=n<=total for n,total in zip(r['step_positive_pair_counts'],r['step_pair_counts']))
assert r['cost_projection']['full_reference_seconds'] is None
print(json.dumps({'status':'COMPLETED','diagnostic_only':True,'profile_sha256':digest(out/'profile.json'),
                  'run_sha256':digest(out/'run.json'),'peak_allocated_bytes':r['peak_allocated_bytes'],
                  'peak_reserved_bytes':r['peak_reserved_bytes'],'phase_seconds':r['phase_seconds'],
                  'step_phase_seconds':r['step_phase_seconds'],'step_positive_pair_counts':r['step_positive_pair_counts']},indent=2))
PY
  then
    printf '%s\n' '6步诊断验收通过；只回传，不启动完整训练。'
  else
    task_rc=$?; printf '验收失败rc=%s，保留原报告供本地核验。\n' "$task_rc"
  fi
else
  printf '%s\n' '本次未成功，不执行完成验收。决定收尾时启动review并整理失败证据。'
fi

# Closing read-only snapshot of the selected card (line 225); never preempts, never resets.
if diag008_run review "gpu-end-snapshot" "$task_python" tools/setup/gpu_allocation.py \
    --device "$task_gpu_uuid" --output "$task_root/allocation-end.json"; then
  printf '%s\n' '结束快照已记录。'
else
  task_rc=$?; printf '结束查询失败 rc=%s；记 unknown，不否定已完成的 profile。\n' "$task_rc"
fi

echo "C_END=$(date +'%Y-%m-%dT%H:%M:%S%z')"
for lbl in "accept-$task_attempt" gpu-end-snapshot; do
  if [ -f "$task_root/$lbl/exit_code.txt" ]; then
    printf '%s rc=%s\n' "$lbl" "$(cat "$task_root/$lbl/exit_code.txt")"
  else
    printf '%s missing\n' "$lbl"
  fi
done
