#!/usr/bin/env bash
# Is training making progress?
#
# Not "is the process alive" — a wedged decode or a hung CUDA call keeps the process
# alive indefinitely, and that is the failure a supervisor is meant to catch. The trainer
# refreshes `heartbeat` on every log line (mining, training and validation all log every
# couple of thousand items), so a stale file means stalled, not dead.
#
# Exit 0 healthy, 1 unhealthy — Docker's contract.
set -uo pipefail

BEAT="${CBIR_RUNS:-/app/data/runs}/heartbeat"
MAX_AGE="${CBIR_HEARTBEAT_MAX_AGE:-900}"   # 15 min: one epoch is ~15, one log line ~2

if [[ ! -f "${BEAT}" ]]; then
  # Absent is not unhealthy: the first heartbeat only lands once the run starts logging.
  # Compose's `start_period` covers that window; failing here would kill every cold start.
  echo "no heartbeat yet at ${BEAT}"
  exit 1
fi

age=$(( $(date +%s) - $(stat -c %Y "${BEAT}") ))
if (( age > MAX_AGE )); then
  echo "stalled: heartbeat ${age}s old (limit ${MAX_AGE}s) — $(cat "${BEAT}")"
  exit 1
fi

echo "ok: heartbeat ${age}s old — $(cat "${BEAT}")"
