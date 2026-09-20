#!/usr/bin/env bash
set -euo pipefail

stage= account= partition= qos= node= memory= time_limit= cpus= profile=
while (($#)); do
  case "$1" in
    --stage) stage=$2; shift 2 ;;
    --account) account=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;;
    --qos) qos=$2; shift 2 ;;
    --node) node=$2; shift 2 ;;
    --memory) memory=$2; shift 2 ;;
    --time) time_limit=$2; shift 2 ;;
    --cpus) cpus=$2; shift 2 ;;
    --resource-profile) profile=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$stage" =~ ^(A0|A1|A2|B|C|D-quality|D-match|D-qps|D-tune-quality|D-tune-select|D-tune-qps)$ ]] || { echo 'invalid --stage' >&2; exit 2; }
[[ "$profile" == exploratory || "$profile" == exclusive ]] || { echo 'invalid --resource-profile' >&2; exit 2; }
[[ -n "$partition" && -n "$qos" && -n "$memory" && -n "$time_limit" && -n "$cpus" ]] || {
  echo 'partition, qos, memory, time and cpus are required' >&2; exit 2;
}
[[ "$cpus" =~ ^[1-9][0-9]*$ ]] || { echo 'invalid --cpus' >&2; exit 2; }
[[ -n "${RESIDUAL_RUN_ROOT:-}" && -n "${RESIDUAL_PYTHON:-}" &&
   -n "${RESIDUAL_REFERENCE_BUILD:-}" && -n "${RESIDUAL_PERFORMANCE_BUILD:-}" ]] || {
  echo 'set RESIDUAL_RUN_ROOT, RESIDUAL_PYTHON and both build paths' >&2; exit 2;
}
mkdir -p "$RESIDUAL_RUN_ROOT/logs"
export RESIDUAL_STAGE="$stage" RESIDUAL_RESOURCE_PROFILE="$profile"
args=(--parsable --partition="$partition" --qos="$qos" --mem="$memory"
      --time="$time_limit" --cpus-per-task="$cpus" --export=ALL
      --job-name="$stage"
      --output="$RESIDUAL_RUN_ROOT/logs/$stage-%j.out"
      --error="$RESIDUAL_RUN_ROOT/logs/$stage-%j.err")
[[ -z "$account" ]] || args+=(--account="$account")
[[ -z "$node" ]] || args+=(--nodelist="$node")
[[ "$profile" != exclusive ]] || args+=(--exclusive)
sbatch "${args[@]}" scripts/v0/residual_estimator/run_prototype.slurm
