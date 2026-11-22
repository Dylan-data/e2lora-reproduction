#!/bin/bash
# Submit pending GLUE runs as TC2 slots open.
set -euo pipefail
cd "$(dirname "$0")/../.."
out="${GLUE_FULL_OUT:-$PWD/outputs/dylan/glue-full-seed42}"
mkdir -p "$out"
exec 9>"$out/follow.lock"
flock -n 9 || exit 0
queue="$out/follow-jobs.tsv"
touch "$queue"
# These jobs were submitted before the monitor started.
for id in 54129 54134; do
    if ! awk -v id="$id" '$1 == id {found=1} END {exit !found}' "$queue"; then
        printf '%s\texisting\n' "$id" >> "$queue"
    fi
done
while true; do
    remaining=0
    while IFS= read -r run; do
        task=${run%%:*}
        method=${run#*:}
        if ! compgen -G "$out/results/dylan/glue-$task-$method-original-*/result.json" >/dev/null &&
           ! compgen -G "$out/results/text-understanding/glue-$task-$method-original-*/result.json" >/dev/null; then
            remaining=$((remaining + 1))
            last=$(awk -v run="$run" '$2 == run {id=$1} END {print id}' "$queue")
            if [[ -n "$last" ]]; then
                state=$(sacct -X -j "$last" -n -P -o State | head -1)
                case "$state" in
                    TIMEOUT*) ;;
                    COMPLETED*|FAILED*|CANCELLED*|OUT_OF_MEMORY*|NODE_FAIL*)
                        printf '%s %s %s requires inspection\n' "$run" "$last" "$state" >&2
                        continue ;;
                    *) continue ;;
                esac
            fi
            slots=$(squeue -u "$USER" -h -o '%i' | wc -l)
            (( slots < 2 )) || continue
            if result=$(sbatch --parsable -p MGPU-TC2 --qos=normal --gres=gpu:1 -c 4 --mem=20G \
                 -t 06:00:00 -J "glue-$task-$method" --output="$out/slurm-%j.log" \
                 experiments/dylan/glue_full_tc2.sh "$run" 2>&1); then
                id=${result%%;*}
                printf '%s\t%s\n' "$id" "$run" >> "$queue"
                printf '%s %s submitted as %s\n' "$(date -Is)" "$run" "$id"
            elif [[ "$result" != *QOSMaxSubmitJobPerUserLimit* ]]; then
                printf '%s\n' "$result" >&2
                exit 1
            fi
        fi
    done < experiments/dylan/glue_follow_runs.txt
    (( remaining > 0 )) || exit 0
    sleep 60
done
