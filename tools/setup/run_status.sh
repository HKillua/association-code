#!/usr/bin/env bash
# Source from a set -euo pipefail task. The outer timeout exit code is authoritative.

task_status_init() {
  task_status_dir=$1
  task_status_prefix=${2-}
  [[ -d "$task_status_dir" && "$task_status_prefix" =~ ^[a-zA-Z0-9_-]*$ ]] || return 2
  task_status_stem=${task_status_prefix:+${task_status_prefix}_}
  task_stage=not_started
  task_stage_complete=0
  task_signal=none
  task_completed=0
  trap 'task_signal=TERM; exit 143' TERM
  trap 'task_signal=INT; exit 130' INT
  trap 'task_signal=HUP; exit 129' HUP
  trap 'task_finish "$?"' EXIT
  date -u +%FT%TZ > "$task_status_dir/${task_status_stem}started_at.txt"
}

task_begin() {
  [[ "$1" =~ ^[a-zA-Z0-9_-]+$ && "$task_completed" = 0 ]] || return 2
  task_stage=$1
  task_stage_complete=0
  printf '%s\t%s\tstart\n' "$(date -u +%FT%TZ)" "$task_stage" >> "$task_status_dir/stages.tsv"
}

task_done() {
  [[ "$task_stage" != not_started && "$task_stage_complete" = 0 ]] || return 2
  printf '%s\t%s\tcomplete\n' "$(date -u +%FT%TZ)" "$task_stage" >> "$task_status_dir/stages.tsv"
  task_stage_complete=1
}

task_run_logged() {
  local task_name=$1 task_log=$2 task_command_rc=0
  shift 2
  task_begin "$task_name"
  "$@" > "$task_log" 2>&1 || task_command_rc=$?
  printf '%s\n' "$task_command_rc" > "$task_status_dir/${task_name}_exit_code.txt"
  if (( task_command_rc != 0 )); then
    return "$task_command_rc"
  fi
  task_done
}

task_complete() {
  [[ "$task_stage_complete" = 1 ]] || return 2
  printf 'completed\n' > "$task_status_dir/${task_status_stem}completed.txt"
  task_completed=1
}

task_finish() {
  local task_final_rc=$1
  trap - EXIT
  # A clean shell exit without explicit final acceptance is not a completed task.
  if (( task_final_rc == 0 )) && [[ "$task_completed" != 1 ]]; then
    task_final_rc=1
  fi
  set +e
  printf '%s\n' "$task_final_rc" > "$task_status_dir/${task_status_stem}exit_code.txt"
  printf '%s\n' "$task_signal" > "$task_status_dir/${task_status_stem}signal.txt"
  date -u +%FT%TZ > "$task_status_dir/${task_status_stem}ended_at.txt"
  printf '%s\t%s\texit=%s\tsignal=%s\n' "$(date -u +%FT%TZ)" "$task_stage" "$task_final_rc" "$task_signal" >> "$task_status_dir/stages.tsv"
  exit "$task_final_rc"
}
