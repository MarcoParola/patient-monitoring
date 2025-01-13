#!/bin/bash

# Lista dei modelli di privacy
models=("VIDEO_PRIVATIZER" "STYLEGAN2" "DEEP_PRIVACY2" "BLUR" "PIXELATE")

# Lista degli ID dei pazienti
ids=(123 234 345)

# Valori di fps per pose_dataset
fps_values=(5 10 15 20)

# Valori di in_channels
in_channels_values=(1 3)

# Genera tutte le permutazioni
for model in "${models[@]}"; do
    for id1 in "${ids[@]}"; do
        for id2 in "${ids[@]}"; do
            if [ "$id1" != "$id2" ]; then
                for id3 in "${ids[@]}"; do
                    if [ "$id3" != "$id1" ] && [ "$id3" != "$id2" ]; then
                        for fps in "${fps_values[@]}"; do
                            for in_channels in "${in_channels_values[@]}"; do
                                if [ "$in_channels" -eq 3 ]; then
                                    camera_type=0
                                else
                                    camera_type=1
                                fi
                                train_ids="$id1,$id2"
                                val_test_id="$id3"
                                echo python train.py train.patient_ids="[$train_ids]" val.patient_ids="[$val_test_id]" test.patient_ids="[$val_test_id]" privacy_model_type="$model" pose_dataset.fps="$fps" in_channels="$in_channels" train.camera_type="$camera_type" val.camera_type="$camera_type" test.camera_type="$camera_type"
                                python train.py train.patient_ids="[$train_ids]" val.patient_ids="[$val_test_id]" test.patient_ids="[$val_test_id]" privacy_model_type="$model" pose_dataset.fps="$fps" in_channels="$in_channels" train.camera_type="$camera_type" val.camera_type="$camera_type" test.camera_type="$camera_type"
                            done
                        done
                    fi
                done
            fi
        done
    done
done
