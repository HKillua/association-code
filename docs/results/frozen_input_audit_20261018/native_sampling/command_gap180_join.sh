#!/bin/bash
set -u
task_audit_root=/Users/a1/Projects/biye-ronghe/docs/reviews/2026-10-18_experiment_reaudit/native_sampling_audit
env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /Users/a1/Projects/association-code/.venv/bin/python "$task_audit_root/join_gap180_sampling.py" \
  > "$task_audit_root/gap180_join_stdout.log" 2> "$task_audit_root/gap180_join_stderr.log"
task_audit_rc=$?
printf '%s\n' "$task_audit_rc" > "$task_audit_root/gap180_join_exit_code.txt"
exit "$task_audit_rc"
