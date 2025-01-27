

#!/bin/bash

# Configuration files
CONFIG_FILE="config/config_cate.yaml"



FPS_VALUES=(3)
CAMERA_TYPES=(2 1 0)

# CNN configuration
CNN_CONV_BACKBONE="CNN3D"
CNN_RESIZE_W=256
CNN_RESIZE_H=256

for FPS in "${FPS_VALUES[@]}"; do
  for CAMERA_TYPE in "${CAMERA_TYPES[@]}"; do
    echo "Running CNN training with FPS=$FPS, CAMERA_TYPE=$CAMERA_TYPE"

    # Update CNN parameters in the config file
    sed -i "s|^\s*conv_backbone: .*|conv_backbone: \"$CNN_CONV_BACKBONE\"|" "$CONFIG_FILE"
    sed -i "s/^  resize_w: .*/  resize_w: $CNN_RESIZE_W/" "$CONFIG_FILE"
    sed -i "s/^  resize_h: .*/  resize_h: $CNN_RESIZE_H/" "$CONFIG_FILE"
    sed -i "s/^  fps: .*/  fps: $FPS/" "$CONFIG_FILE"
    sed -i "s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"

        # Update patient_ids for train, val, and test in the config file
    sed -i "/^train:/,/^val:/s/^  patient_ids: .*/  patient_ids: [123,234]/" "$CONFIG_FILE"
    sed -i "/^val:/,/^test:/s/^  patient_ids: .*/  patient_ids: [345]/" "$CONFIG_FILE"
    sed -i "/^test:/,/^$/s/^  patient_ids: .*/  patient_ids: [345]/" "$CONFIG_FILE"

     # Add or update the 'end' and 'dataset' parameters
    if ! grep -q "^end:" "$CONFIG_FILE"; then
      echo "end: \"mlp\"" >> "$CONFIG_FILE"
    else
      sed -i "s/^end: .*/end: \"mlp\"/" "$CONFIG_FILE"
    fi

    if ! grep -q "^dataset:" "$CONFIG_FILE"; then
      echo "dataset: \"Custom\"" >> "$CONFIG_FILE"
    else
      sed -i "s/^dataset: .*/dataset: \"Custom\"/" "$CONFIG_FILE"
    fi

    # Execute the Python script for CNN trial
    python train_cate.py
  done
done
