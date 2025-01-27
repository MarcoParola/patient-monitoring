#!/bin/bash

FPS_VALUES=(3)
CAMERA_TYPES=(2 1 0)

# YOLO model sizes
YOLO_MODEL_SIZES=("n" "m" "x")

# Update patient_ids for train, val, and test in the config file
    sed -i "s/^  patient_ids: .*/  patient_ids: [123,234]/" "$CONFIG_FILE"
    sed -i "/^val:/,/^test:/s/^  patient_ids: .*/  patient_ids: [345]/" "$CONFIG_FILE"

# Itera su YOLO model sizesfor YOLO_MODEL_SIZE in "${YOLO_MODEL_SIZES[@]}"; do
    for FPS in "${FPS_VALUES[@]}"; do
        for CAMERA_TYPE in "${CAMERA_TYPES[@]}"; do      
            echo "Running YOLO training with YOLO_MODEL_SIZE=$YOLO_MODEL_SIZE, FPS=$FPS, CAMERA_TYPE=$CAMERA_TYPE"
            # Aggiorna i parametri di YOLO nel file di configurazione
            sed -i "s|^\s*conv_backbone: .*|conv_backbone: \"YOLO\"|" "$CONFIG_FILE"      sed -i "s/^  model_size: .*/  model_size: \"$YOLO_MODEL_SIZE\"/" "$CONFIG_FILE"
            sed -i "s|^  processed_csv: .*|  processed_csv: dataset/keypoints/keypoints_YOLO_${YOLO_MODEL_SIZE}_${FPS}_${CAMERA_TYPE}.csv|" "$CONFIG_FILE"      sed -i "s/^  fps: .*/  fps: $FPS/" "$CONFIG_FILE"
            sed -i "/^train:/,/^val:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"      sed -i "/^val:/,/^test:/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"
            sed -i "/^test:/,/^\s*$/s/^  camera_type: .*/  camera_type: $CAMERA_TYPE/" "$CONFIG_FILE"      sed -i "s/^  resize_w: .*/  resize_w: 640/" "$CONFIG_FILE"
            sed -i "s/^  resize_h: .*/  resize_h: 480/" "$CONFIG_FILE"      
            # Imposta extract_keypoints      
            if [ "$first_iteration_yolo" = true ]; then
                # Prima iterazione del ciclo esterno, setta extract_keypoints a true       
                sed -i "s/^  extract_keypoints: .*/  extract_keypoints: true/" "$CONFIG_FILE"
                first_iteration_yolo=false      
            else
                # Iterazioni successive, setta extract_keypoints a false        
                sed -i "s/^  extract_keypoints: .*/  extract_keypoints: false/" "$CONFIG_FILE"
            fi
        # Esegui lo script Python      python train_cate.py
        done  
    done
done