# #!/bin/bash

# # Configuration files
# CONFIG_FILE="config/config_cate.yaml"

# # YOLO model sizes
# YOLO_MODEL_SIZES=("n" "s" "m" "l" "x")

# # OpenPose configurations
# OPENPOSE_CONFIGS=(
#   "True True"
#   "True False"
#   "False True"
#   "False False"
# )

# # Virtual environment directory
# VENV_DIR="C:\Users\cater\anaconda3\envs\venv"

# # OpenPose binaries setup
# OPENPOSE_BINARIES_URL="https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.6.0/openpose-1.6.0-binaries-win64-only_cpu-flir-3d.zip"
# OPENPOSE_BINARIES_ZIP="openpose-1.6.0-binaries-win64-only_cpu-flir-3d.zip"
# OPENPOSE_EXTRACTED_DIR="src/models/skeleton/openpose"

# # Create and activate a virtual environment
# if [ ! -d "$VENV_DIR" ]; then
#   echo "Creating virtual environment..."
#   python3 -m venv "$VENV_DIR"
# fi

# # Activate the virtual environment
# source "$VENV_DIR/Scripts/activate"

# # Install dependencies
# if [ -f "requirements.txt" ]; then
#   echo "Installing requirements..."
#   pip install --upgrade pip
#   python.exe -m pip install --upgrade pip
#   pip install -r requirements.txt
# else
#   echo "requirements.txt not found. Please provide one."
#   exit 1
# fi

# # Function to download OpenPose binaries
# download_openpose() {
#   echo "Checking if OpenPose binaries are already downloaded..."
#   if [ ! -f "$OPENPOSE_BINARIES_ZIP" ]; then
#     echo "Downloading OpenPose binaries..."
#     curl -L -o "$OPENPOSE_BINARIES_ZIP" "$OPENPOSE_BINARIES_URL"
#   else
#     echo "OpenPose binaries already downloaded."
#   fi
# }

# # Function to extract OpenPose binaries
# extract_openpose() {
#   echo "Checking if OpenPose binaries are already extracted..."
#   if [ ! -d "$OPENPOSE_EXTRACTED_DIR" ]; then
#     echo "Extracting OpenPose binaries..."
#     unzip "$OPENPOSE_BINARIES_ZIP" -d "src/models/skeleton/openposedir"
#     chmod +w src/models/skeleton/openposedir/*
#     chmod +w src/models/skeleton/openpose/*
#     echo "Organizing extracted files..."
#     mv src/models/skeleton/openposedir/openpose/* src/models/skeleton/openposedir/
#     rm -rf src/models/skeleton/openposedir/openpose
#     mv src/models/skeleton/openposedir src/models/skeleton/openpose
#     echo "Moving necessary files to OpenPose directory..."
#     cp src/models/skeleton/openpose_skeleton.py "$OPENPOSE_EXTRACTED_DIR/openpose_skeleton.py"
#     cp pose_iter_584000.caffemodel "$OPENPOSE_EXTRACTED_DIR/models/pose/body_25/pose_iter_584000.caffemodel"
#     cp pose_iter_440000.caffemodel "$OPENPOSE_EXTRACTED_DIR/models/pose/coco/pose_iter_440000.caffemodel"
#     cp pose_iter_102000.caffemodel "$OPENPOSE_EXTRACTED_DIR/models/hand/pose_iter_102000.caffemodel"
#     cp pose_iter_116000.caffemodel "$OPENPOSE_EXTRACTED_DIR/models/face/pose_iter_116000.caffemodel"
#   else
#     echo "OpenPose binaries already extracted."
#   fi
# }

# # # Download and extract OpenPose binaries
# # download_openpose
# # extract_openpose

# # Iterate over OpenPose configurations
# for CONFIG in "${OPENPOSE_CONFIGS[@]}"; do
#   read DETECT_HANDS DETECT_FACE <<< "$CONFIG"
#   echo "Running training with OpenPose DETECT_HANDS=$DETECT_HANDS, DETECT_FACE=$DETECT_FACE"
#   sed -i "s|^\s*conv_backbone: .*|conv_backbone: \"OpenPose\"|" "$CONFIG_FILE"
#   sed -i "s/^  detect_hands: .*/  detect_hands: $DETECT_HANDS/" "$CONFIG_FILE"
#   sed -i "s/^  detect_face: .*/  detect_face: $DETECT_FACE/" "$CONFIG_FILE"
#   sed -i "s|^  processed_csv: .*|  processed_csv: dataset/keypoints/keypoints_OpenPose_${DETECT_HANDS}_${DETECT_FACE}.csv|" "$CONFIG_FILE"
#   sed -i "s/^  resize_w: .*/  resize_w: 640/" "$CONFIG_FILE"
#   sed -i "s/^  resize_h: .*/  resize_h: 480/" "$CONFIG_FILE"


#   # Execute the Python script
#   #python train2_cate.py
# done

# # Iterate over YOLO model sizes
# for YOLO_MODEL_SIZE in "${YOLO_MODEL_SIZES[@]}"; do
#   echo "Running training with YOLO_MODEL_SIZE=$YOLO_MODEL_SIZE"

#   # Update YOLO parameters in the config file
#   sed -i "s|^\s*conv_backbone: .*|conv_backbone: \"YOLO\"|" "$CONFIG_FILE"
#   sed -i "s/^  model_size: .*/  model_size: \"$YOLO_MODEL_SIZE\"/" "$CONFIG_FILE"
#   sed -i "s|^  processed_csv: .*|  processed_csv: dataset/keypoints/keypoints_YOLO_${YOLO_MODEL_SIZE}.csv|" "$CONFIG_FILE"
#   sed -i "s/^  resize_w: .*/  resize_w: 640/" "$CONFIG_FILE"
#   sed -i "s/^  resize_h: .*/  resize_h: 480/" "$CONFIG_FILE"

#   # Execute the Python script
#   #python train2_cate.py
# done

# # CNN configuration
# echo "Running training with CNN configuration"
# CNN_CONV_BACKBONE="CNN3D"
# CNN_RESIZE_W=256
# CNN_RESIZE_H=256

# # Update CNN parameters in the config file
# sed -i "s|^\s*conv_backbone: .*|conv_backbone: \"$CNN_CONV_BACKBONE\"|" "$CONFIG_FILE"
# sed -i "s/^  resize_w: .*/  resize_w: $CNN_RESIZE_W/" "$CONFIG_FILE"
# sed -i "s/^  resize_h: .*/  resize_h: $CNN_RESIZE_H/" "$CONFIG_FILE"

# # Execute the Python script for CNN trial
# python train2_cate.py

# # Deactivate the virtual environment
# deactivate


#!/bin/bash

# Configuration files
CONFIG_FILE="config/config_cate.yaml"

# YOLO model sizes
YOLO_MODEL_SIZES=("n" "s" "m" "l" "x")

# OpenPose configurations
OPENPOSE_CONFIGS=(
  "True True"
  "True False"
  "False True"
  "False False"
)

# OpenPose binaries setup
OPENPOSE_BINARIES_URL="https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.6.0/openpose-1.6.0-binaries-win64-only_cpu-flir-3d.zip"
OPENPOSE_BINARIES_ZIP="openpose-1.6.0-binaries-win64-only_cpu-flir-3d.zip"
OPENPOSE_EXTRACTED_DIR="src/models/skeleton/openpose"

# Virtual environment directory
VENV_DIR="C:\Users\cater\anaconda3\envs\venv"

# FPS and camera type combinations
FPS_VALUES=(1 2 3)
CAMERA_TYPES=(0 1 "")

# Create and activate a virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/Scripts/activate"

# # Install dependencies
# if [ -f "requirements.txt" ]; then
#   echo "Installing requirements..."
#   pip install --upgrade pip
#   python.exe -m pip install --upgrade pip
#   pip install -r requirements.txt
# else
#   echo "requirements.txt not found. Please provide one."
#   exit 1
# fi



# # Iterate over OpenPose configurations
# for CONFIG in "${OPENPOSE_CONFIGS[@]}"; do
#   read DETECT_HANDS DETECT_FACE <<< "$CONFIG"
#   for FPS in "${FPS_VALUES[@]}"; do
#     for CAMERA_TYPE in "${CAMERA_TYPES[@]}"; do
#       echo "Running OpenPose training with DETECT_HANDS=$DETECT_HANDS, DETECT_FACE=$DETECT_FACE, FPS=$FPS, CAMERA_TYPE=$CAMERA_TYPE"

#       # Update OpenPose parameters in the config file
#       sed -i "s|^\s*conv_backbone: .*|conv_backbone: \"OpenPose\"|" "$CONFIG_FILE"
#       sed -i "s/^  detect_hands: .*/  detect_hands: $DETECT_HANDS/" "$CONFIG_FILE"
#       sed -i "s/^  detect_face: .*/  detect_face: $DETECT_FACE/" "$CONFIG_FILE"
#       sed -i "s|^  processed_csv: .*|  processed_csv: dataset/keypoints/keypoints_OpenPose_${DETECT_HANDS}_${DETECT_FACE}_${FPS}_${CAMERA_TYPE}.csv|" "$CONFIG_FILE"
#       sed -i "s/^  fps: .*/  fps: $FPS/" "$CONFIG_FILE"
#        # Update camera_type for train, val, and test sections
#       sed -i "/^train:/,/^val:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"
#       sed -i "/^val:/,/^test:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"
#       sed -i "/^test:/,/^\s*$/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"

#       sed -i "s/^  resize_w: .*/  resize_w: 640/" "$CONFIG_FILE"
#       sed -i "s/^  resize_h: .*/  resize_h: 480/" "$CONFIG_FILE"

#       # Execute the Python script
#       python train2_cate.py
#     done
#   done
# done

# Iterate over YOLO model sizes
for YOLO_MODEL_SIZE in "${YOLO_MODEL_SIZES[@]}"; do
  for FPS in "${FPS_VALUES[@]}"; do
    for CAMERA_TYPE in "${CAMERA_TYPES[@]}"; do
      echo "Running YOLO training with YOLO_MODEL_SIZE=$YOLO_MODEL_SIZE, FPS=$FPS, CAMERA_TYPE=$CAMERA_TYPE"

      # Update YOLO parameters in the config file
      sed -i "s|^\s*conv_backbone: .*|conv_backbone: \"YOLO\"|" "$CONFIG_FILE"
      sed -i "s/^  model_size: .*/  model_size: \"$YOLO_MODEL_SIZE\"/" "$CONFIG_FILE"
      sed -i "s|^  processed_csv: .*|  processed_csv: dataset/keypoints/keypoints_YOLO_${YOLO_MODEL_SIZE}_${FPS}_${CAMERA_TYPE}.csv|" "$CONFIG_FILE"
      sed -i "s/^  fps: .*/  fps: $FPS/" "$CONFIG_FILE"
       # Update camera_type for train, val, and test sections
      sed -i "/^train:/,/^val:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"
      sed -i "/^val:/,/^test:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"
      sed -i "/^test:/,/^\s*$/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"

      sed -i "s/^  resize_w: .*/  resize_w: 640/" "$CONFIG_FILE"
      sed -i "s/^  resize_h: .*/  resize_h: 480/" "$CONFIG_FILE"

      # Execute the Python script
      python train2_cate.py
    done
  done
done

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
    sed -i "s|^  processed_csv: .*|  processed_csv: dataset/keypoints/keypoints_CNN_${FPS}_${CAMERA_TYPE}.csv|" "$CONFIG_FILE"
    sed -i "s/^  fps: .*/  fps: $FPS/" "$CONFIG_FILE"
    sed -i "s/^  camera_type: .*/  camera_type: \"$CAMERA_TYPE\"/" "$CONFIG_FILE"

    # Execute the Python script for CNN trial
    python train2_cate.py
  done
done

# Deactivate the virtual environment
deactivate
