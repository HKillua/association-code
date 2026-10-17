#!/usr/bin/env bash
# PROFILE-DIAG-20261017-008 phases A+B, transcribed from HandOver/L2S-20261017-008.md
# lines 110-173. Attempt 2/3 are re-invoked by the operator, not looped here.
set -euo pipefail
cd /home/wangguangjie/djs/association-code || exit 1
# shellcheck source=/home/wangguangjie/djs/diag008_wrapper.sh
source /home/wangguangjie/djs/diag008_wrapper.sh
diag008_init
echo "AB_START=$(date +'%Y-%m-%dT%H:%M:%S%z')"

task_prep=runs/PREP-NATIVE-20261017-007-attempt-1
if diag008_start prep && diag008_run prep prepared-check env CUDA_VISIBLE_DEVICES= \
  "$task_python" - "$task_prep" "$task_archive" <<'PY'
import json, pathlib, platform, sys
import association, numpy, torch
from association.config import load_config, validate_manifest_config
from association.runtime import digest
p=pathlib.Path(sys.argv[1]); archive=pathlib.Path(sys.argv[2]); c=load_config('configs/pilot.json')
assert platform.python_version()=='3.11.7' and numpy.__version__=='1.26.4' and torch.__version__=='2.5.1+cu121'
assert torch.version.cuda=='12.1' and pathlib.Path(association.__file__).resolve().parent==pathlib.Path('association').resolve()
expected={'manifest.json':'fb83308d1747b21f55510eb2c5fc69730917fa8781f280a8c59d68f974aa7071',
          'audit.json':'032cd85c1016488dd2203a36569c7d47958d464b54bdc7a75998b5b534754060'}
assert all(digest(p/n)==h for n,h in expected.items())
assert digest('configs/pilot.json')=='d547497656b54ae0625725c26f610c9a3aaaa13e5cda6f6b038016e654a26cc6'
m=json.loads((p/'manifest.json').read_text()); a=json.loads((p/'audit.json').read_text())
validate_manifest_config(m,c)
assert a['status']=='COMPLETED' and not a['synthetic'] and a['length']==64
assert a['manifest_sha256']==m['manifest_sha256']=='0b4df901d075776973c209e835ac6d2a7bd95dc978c7f88a718b6ea274ee354c'
assert archive.stat().st_size==m['archive']['bytes']==924486019
assert m['archive']['sha256']=='513d96e9d763544575ddec99c7e91f834aec43fc8da5a39494d2760f920d7e78'
assert a['test_measurements_read']==a['reference_members_read']==0
print(json.dumps({'status':'COMPLETED','prepared':str(p),'files':expected,'archive':m['archive'],
                  'manifest_sha256':m['manifest_sha256'],'reused_without_preparing':True},indent=2))
PY
then
  printf '%s\n' 'prepared核验通过，直接复用。'
else
  task_rc=$?; printf '前置检查失败rc=%s，停止并保留证据，不重建prepared。\n' "$task_rc"
fi

task_attempt=${ATTEMPT:-1}
task_gpu_request=${GPU_REQUEST:-auto}
if diag008_passed prepared-check; then
  if [[ "$task_attempt" =~ ^[123]$ ]] && diag008_start gpu && \
    diag008_run gpu "allocation-$task_attempt" "$task_python" tools/setup/gpu_allocation.py \
      --device "$task_gpu_request" --output "$task_root/allocation-$task_attempt.json"; then
    if task_gpu_uuid=$(cat "$task_root/allocation-$task_attempt/stdout.log") && \
      [[ "$task_gpu_uuid" =~ ^GPU-[a-fA-F0-9-]+$ ]]; then
      task_profile="runs/PROFILE-DIAG-20261017-008-attempt-$task_attempt"
      echo "selected_uuid=$task_gpu_uuid profile_dir=$task_profile"
      if diag008_run gpu "profile-$task_attempt" "${task_timing[@]}" env CUDA_VISIBLE_DEVICES="$task_gpu_uuid" \
        "$task_python" -m association.cli profile --archive "$task_archive" --prepared "$task_prep" \
        --config configs/pilot.json --role A0 --branch ce --seed 20 --steps 6 --max-minutes 30 \
        --device cuda:0 --cpu-threads 4 --output "$task_profile"; then
        printf '%s\n' '6步命令完成，进入只读验收。'
      else
        task_rc=$?; printf 'profile rc=%s，保留最后进度；按失败类别决定是否同预算换卡。\n' "$task_rc"
      fi
    else
      printf '%s\n' 'UUID读取无效，停止本次启动。'
    fi
  else
    task_rc=$?; printf '选卡/前提失败rc=%s，不启动计算。\n' "$task_rc"
  fi
else
  printf '%s\n' 'prepared尚未核验，不启动GPU阶段。'
fi

echo "AB_END=$(date +'%Y-%m-%dT%H:%M:%S%z')"
for lbl in prepared-check "allocation-$task_attempt" "profile-$task_attempt"; do
  if [ -f "$task_root/$lbl/exit_code.txt" ]; then
    printf '%s rc=%s start=%s end=%s\n' "$lbl" "$(cat "$task_root/$lbl/exit_code.txt")" \
      "$(cat "$task_root/$lbl/started_at.txt")" "$(cat "$task_root/$lbl/ended_at.txt")"
  else
    printf '%s missing\n' "$lbl"
  fi
done
