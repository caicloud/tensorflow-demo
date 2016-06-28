#!/usr/bin/env bash

cd /
# run file server.
python /file_server.py &

cd notebooks
# run tensorboard
tensorboard --logdir=/log &

# run jupyter
bash /run_jupyter.sh
