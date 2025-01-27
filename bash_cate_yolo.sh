#!/bin/bash
# Configuration files
CONFIG_FILE="config/config_cate.yaml"

FPS_VALUES=(3)
CAMERA_TYPES=(2 1 0)

# YOLO model sizes
YOLO_MODEL_SIZES=("n" "m" "x")

# Virtual environment directory
VENV_DIR="C:\Users\cater\anaconda3\envs\venv"

#Create and activate a virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

#Activate the virtual environment
source "$VENV_DIR/Scripts/activate"

#Install dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing requirements..."
  pip install --upgrade pip
  python.exe -m pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "requirements.txt not found. Please provide one."
  exit 1
fi

     
sed -i "s|extract_keypoints: .*|extract_keypoints: True|" "$CONFIG_FILE"


# Itera su YOLO model sizes
for YOLO_MODEL_SIZE in "${YOLO_MODEL_SIZES[@]}"; do
    for FPS in "${FPS_VALUES[@]}"; do
        for CAMERA_TYPE in "${CAMERA_TYPES[@]}"; do      
            echo "Running YOLO training with YOLO_MODEL_SIZE=$YOLO_MODEL_SIZE, FPS=$FPS, CAMERA_TYPE=$CAMERA_TYPE"
            # Aggiorna i parametri di YOLO nel file di configurazione
            sed -i "s|^\s*conv_backbone: .*|conv_backbone: \"YOLO\"|" "$CONFIG_FILE"      
            sed -i "s/^  model_size: .*/  model_size: \"$YOLO_MODEL_SIZE\"/" "$CONFIG_FILE"
            sed -i "s|^  processed_csv: .*|  processed_csv: dataset/keypoints_YOLO_${YOLO_MODEL_SIZE}_${FPS}.csv|" "$CONFIG_FILE"      
            sed -i "s/^  fps: .*/  fps: $FPS/" "$CONFIG_FILE"
            sed -i "/^train:/,/^val:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"      
            sed -i "/^val:/,/^test:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"
            sed -i "/^test:/,/^\s*$/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"      
            sed -i "s/^  resize_w: .*/  resize_w: 640/" "$CONFIG_FILE"
            sed -i "s/^  resize_h: .*/  resize_h: 480/" "$CONFIG_FILE"      
           
        
     
                # Update patient_ids for train, val, and test in the config file
            sed -i "/^train:/,/^val:/s/^  patient_ids: .*/  patient_ids: [123,234]/" "$CONFIG_FILE"
            sed -i "/^val:/,/^test:/s/^  patient_ids: .*/  patient_ids: [345]/" "$CONFIG_FILE"
            sed -i "/^test:/,/^$/s/^  patient_ids: .*/  patient_ids: [345]/" "$CONFIG_FILE"

        # Esegui lo script Python      
        python train_cate.py
        done  
    done
done