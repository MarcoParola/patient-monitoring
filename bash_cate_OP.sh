#!/bin/bash


# OpenPose configurations for hand and face
OPENPOSE_CONFIGS=(
  "True False"
  "False False"
)


FPS_VALUES=(3)
CAMERA_TYPES=(2 1 0)
# Iterate over OpenPose configurations
for CONFIG in "${OPENPOSE_CONFIGS[@]}"; do
  read DETECT_HANDS DETECT_FACE <<< "$CONFIG"
  for FPS in "${FPS_VALUES[@]}"; do
    for CAMERA_TYPE in "${CAMERA_TYPES[@]}"; do
      echo "Running OpenPose training with DETECT_HANDS=$DETECT_HANDS, DETECT_FACE=$DETECT_FACE, FPS=$FPS, CAMERA_TYPE=$CAMERA_TYPE"

      # Update OpenPose parameters in the config file
      sed -i "s|^\s*conv_backbone: .*|conv_backbone: \"OpenPose\"|" "$CONFIG_FILE"
      sed -i "s/^  detect_hands: .*/  detect_hands: $DETECT_HANDS/" "$CONFIG_FILE"
      sed -i "s/^  detect_face: .*/  detect_face: $DETECT_FACE/" "$CONFIG_FILE"
      sed -i "s|^  processed_csv: .*|  processed_csv: dataset/keypoints/keypoints_OpenPose_${DETECT_HANDS}_${DETECT_FACE}_${FPS}_${CAMERA_TYPE}.csv|" "$CONFIG_FILE"
      sed -i "s/^  fps: .*/  fps: $FPS/" "$CONFIG_FILE"
      sed -i "/^train:/,/^val:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"
      sed -i "/^val:/,/^test:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"
      sed -i "/^test:/,/^\s*$/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"
      sed -i "s/^  resize_w: .*/  resize_w: 640/" "$CONFIG_FILE"
      sed -i "s/^  resize_h: .*/  resize_h: 480/" "$CONFIG_FILE"

    
      # Execute the Python script
      python train_cate.py
    done
  done
done