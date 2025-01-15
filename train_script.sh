#!/bin/bash

# Lista dei modelli di privacy
models=("VIDEO_PRIVATIZER" "STYLEGAN2" "DEEP_PRIVACY2" "BLUR" "PIXELATE")

# Lista degli ID dei pazienti
ids=(123 234 345)

# Valori di fps per pose_dataset
fps_values=(5 10 15 20)

# Valori di camera_type (0 per 3 canali, 1 per 1 canale)
camera_values=(0 1)

# Genera tutte le permutazioni
for model in "${models[@]}"; do
    for id1 in "${ids[@]}"; do
        for id2 in "${ids[@]}"; do
            if [ "$id1" != "$id2" ]; then
                for id3 in "${ids[@]}"; do
                    if [ "$id3" != "$id1" ] && [ "$id3" != "$id2" ]; then
                        for fps in "${fps_values[@]}"; do
                            for camera_type in "${camera_values[@]}"; do
                                train_ids="$id1,$id2"
                                val_test_id="$id3"
                                echo python train.py train.patient_ids="[$train_ids]" val.patient_ids="[$val_test_id]" test.patient_ids="[$val_test_id]" privacy_model_type="$model" pose_dataset.fps="$fps" train.camera_type="$camera_type" val.camera_type="$camera_type" test.camera_type="$camera_type"
                                python train.py train.patient_ids="[$train_ids]" val.patient_ids="[$val_test_id]" test.patient_ids="[$val_test_id]" privacy_model_type="$model" pose_dataset.fps="$fps" train.camera_type="$camera_type" val.camera_type="$camera_type" test.camera_type="$camera_type"
                            done
                        done
                    fi
                done
            fi
        done
    done
done
