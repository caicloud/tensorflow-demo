#!/usr/bin/env bash

# Helper functions
die() {
  echo $@
  exit 1
}

WORKER_GRPC_URLS="grpc://180.101.191.78:30001 grpc://180.101.191.78:30002 grpc://180.101.191.78:30003"
WORKER_URLS="180.101.191.78:30001,180.101.191.78:30002,180.101.191.78:30003"

PS_URLS="tf-ps0:2222,tf-ps1:2222"

N_WORKERS=3

# Process additional input arguments
echo "WORKERS = ${WORKER_URLS}"
echo "PARAMETER_SERVERS = ${PS_URLS}"

# Current working directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNIST_REPLICA="${DIR}/word2vector.py"
echo "Running $MNIST_REPLICA"
WKR_LOG_PREFIX="/tmp/worker"


IDX=0
((N_WORKERS--))
while true; do
  WORKER_GRPC_URL="${URLS[IDX]}"
  python "${MNIST_REPLICA}" \
      --worker_grpc_url="${WORKER_GRPC_URL}" \
      --worker_index=${IDX} \
      --workers=${WORKER_URLS} \
      --parameter_servers=${PS_URLS} > "${WKR_LOG_PREFIX}${IDX}.log" &
  echo "Worker ${IDX}: "
  echo "  GRPC URL: ${WORKER_GRPC_URL}"
  echo "  log file: ${WKR_LOG_PREFIX}${IDX}.log"

  ((IDX++))
  if [[ "${IDX}" == "${N_WORKERS}" ]]; then
    break
  fi
done

WORKER_GRPC_URL="${URLS[IDX]}"
python "${MNIST_REPLICA}" \
    --worker_grpc_url="${WORKER_GRPC_URL}" \
    --worker_index=${IDX} \
    --workers=${WORKER_URLS} \
    --parameter_servers=${PS_URLS}

echo "Done!"
