#!/bin/bash
set -euo pipefail

# Start CARLA 0.9.15 in Docker with GPU and offscreen rendering

IMAGE_TAG=${IMAGE_TAG:-"mycarla:0.9.15_maps_fixtown06"}

EXTRA_ARGS="-it"
if [[ "${DETACH:-}" != "" ]]; then
  EXTRA_ARGS="-d"
fi

docker kill carla_sim || true
docker run \
  --name carla_sim \
  --rm \
  --gpus all \
  --net=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ${EXTRA_ARGS} \
  ${IMAGE_TAG} \
  /bin/bash ./CarlaUE4.sh -opengl -nosound -RenderOffScreen -benchmark -fps=20 -quality-level=Low

