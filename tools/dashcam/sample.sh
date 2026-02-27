DATASET_DIR="data/dual_camera_train/Town04_003"
rm -rf $DATASET_DIR
python tools/dashcam/run.py \
    --town Town04_Opt \
    --high-quality \
    --camera-height 1.6 \
    --perfect-cam \
    --record-modeld $DATASET_DIR \
    --max-frames 20000 \
    --fast