#!/usr/bin/env bash

NUM_WORKER=${1}
NUM_PS=${2}
CODE=${3}

WORKER_GRPC_URLS="grpc://tf-worker0:2222"
WORKER_URLS="tf-worker0:2222"
IDX=1
while true; do
  if [[ "${IDX}" == "${NUM_WORKER}" ]]; then
    break
  fi

  WORKER_GRPC_URLS="${WORKER_GRPC_URLS} grpc://tf-worker${IDX}:2222"
  WORKER_URLS="${WORKER_URLS},tf-worker${IDX}:2222"
  ((IDX++))
done

PS_URLS="tf-ps0:2222"
IDX=1
while true; do
  if [[ "${IDX}" == "${NUM_PS}" ]]; then
    break
  fi

  PS_URLS="${PS_URLS},tf-ps${IDX}:2222"
  ((IDX++))
done

echo "WORKERS = ${WORKER_URLS}"
echo "PARAMETER_SERVERS = ${PS_URLS}"
echo "Running ${CODE}"
WKR_LOG_PREFIX="/tmp/worker"
URLS=($WORKER_GRPC_URLS)

IDX=0
((NUM_WORKER--))
while true; do
  if [[ "${IDX}" == "${NUM_WORKER}" ]]; then
    break
  fi

  WORKER_GRPC_URL="${URLS[IDX]}"
  python "${CODE}" \
      --worker_grpc_url="${WORKER_GRPC_URL}" \
      --worker_index=${IDX} \
      --workers=${WORKER_URLS} \
      --parameter_servers=${PS_URLS} > "${WKR_LOG_PREFIX}${IDX}.log" &
  echo "Worker ${IDX}: "
  echo "  GRPC URL: ${WORKER_GRPC_URL}"
  echo "  log file: ${WKR_LOG_PREFIX}${IDX}.log"

  ((IDX++))
done

WORKER_GRPC_URL="${URLS[IDX]}"
python "${CODE}" \
    --worker_grpc_url="${WORKER_GRPC_URL}" \
    --worker_index=${IDX} \
    --workers=${WORKER_URLS} \
    --parameter_servers=${PS_URLS}

echo "Done!"
