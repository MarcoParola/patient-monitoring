python train.py dataset="action" privacy_model_type="DEEP_PRIVACY2" pose_dataset.fps=5 privacy_attributes.skin_color=3 output_dim=19 train.batch_size=2

python train.py privacy_model_type="DEEP_PRIVACY2" pose_dataset.fps=5 train.camera_type=0 val.camera_type=0 test.camera_type=0 train.batch_size=2
