#!/bin/bash

ANNOTATED_DIR=$1
for H in H1 H2 H3 H4 H5 H6; do
    Hk_DIR="${ANNOTATED_DIR}/${H}"
    echo "Checking for cached data for ${H} in ${Hk_DIR}..."
    if [ -d "${Hk_DIR}" ]; then
        echo "Preprocessing cached data for ${H}..."
        python tools/dashcam/train/preprocess_cache.py ${Hk_DIR} --gpu
    fi
done