# PROFILE-DIAG-20261017-008 launcher, transcribed from HandOver/L2S-20261017-008.md lines 48-106.
# Kept outside the repository (permitted by line 46): this harness does not keep Bash state
# between tool calls, so the one-time initialisation is separated into diag008_init and
# guarded by existence checks. The three normative functions are byte-for-byte unchanged.

set -euo pipefail
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_archive=/home/wangguangjie/djs/data/mtad/数据集level2.zip
task_root=HandOver/results/PROFILE-DIAG-20261017-008
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
task_timing=()
if /usr/bin/time --version >/dev/null 2>&1; then task_timing=(/usr/bin/time -v); fi

diag008_init() (
  set -euo pipefail
  cd /home/wangguangjie/djs/association-code || exit 1
  test -d "$task_root" || mkdir "$task_root" || exit 1
  if test ! -e "$task_root/actual_commit.txt"; then
    (set -C; git rev-parse HEAD > "$task_root/actual_commit.txt") || exit 1
  fi
  if test ! -e "$task_root/worktree_changes.txt"; then
    (set -C; git diff --name-only > "$task_root/worktree_changes.txt") || exit 1
  fi
)

diag008_start() (
  set -euo pipefail
  (( $# == 1 )) || exit 2
  case "$1" in gpu) task_budget=1800 ;; prep|review) task_budget=300 ;; *) exit 2 ;; esac
  if [[ "$1" != review && -e "$task_root/review_deadline_epoch.txt" ]]; then exit 2; fi
  task_deadline_file="$task_root/$1_deadline_epoch.txt"
  if test ! -e "$task_deadline_file"; then
    task_now=$(date +%s) || exit 1
    (set -C; printf '%s\n' "$(( task_now + task_budget ))" > "$task_deadline_file") || exit 1
  fi
)

diag008_run() (
  set -euo pipefail
  (( $# >= 3 )) || exit 2
  task_phase=$1 task_label=$2
  shift 2 || exit 2
  case "$task_phase" in prep|gpu|review) ;; *) exit 2 ;; esac
  if [[ "$task_phase" != review && -e "$task_root/review_deadline_epoch.txt" ]]; then exit 2; fi
  [[ "$task_label" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
  task_deadline=$(cat "$task_root/${task_phase}_deadline_epoch.txt") || exit 1
  [[ "$task_deadline" =~ ^[0-9]+$ ]] || exit 2
  task_now=$(date +%s) || exit 1
  task_remaining=$(( task_deadline - task_now ))
  (( task_remaining > 15 )) || exit 124
  task_evidence="$task_root/$task_label"
  mkdir "$task_evidence" || exit 1
  date -u +%FT%TZ > "$task_evidence/started_at.txt" || exit 1
  printf '%q ' "$@" > "$task_evidence/command.txt" || exit 1
  printf '\n' >> "$task_evidence/command.txt" || exit 1
  task_rc=0
  timeout --signal=TERM --kill-after=15s "$(( task_remaining - 15 ))s" \
    "$@" > "$task_evidence/stdout.log" 2> "$task_evidence/stderr.log" || task_rc=$?
  printf '%s\n' "$task_rc" > "$task_evidence/exit_code.txt" || exit 1
  date -u +%FT%TZ > "$task_evidence/ended_at.txt" || exit 1
  exit "$task_rc"
)

diag008_passed() (
  (( $# == 1 )) || exit 2
  [[ "$1" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
  task_saved_rc=$(cat "$task_root/$1/exit_code.txt") || exit 1
  [[ "$task_saved_rc" == 0 ]]
)
