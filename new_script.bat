python train.py dataset="action" privacy_model_type="VIDEO_PRIVATIZER" pose_dataset.fps=5 privacy_attributes.skin_color=3 output_dim=19
python train.py dataset="action" privacy_model_type="PIXELATE" pose_dataset.fps=5 privacy_attributes.skin_color=3 output_dim=19
python train.py dataset="action" privacy_model_type="BLUR" pose_dataset.fps=5 privacy_attributes.skin_color=3 output_dim=19
python train.py dataset="action" privacy_model_type="DEEP_PRIVACY2" pose_dataset.fps=5 privacy_attributes.skin_color=3 output_dim=19 train.batch_size=2
python train.py dataset="action" privacy_model_type="STYLEGAN2" pose_dataset.fps=5 privacy_attributes.skin_color=3 output_dim=19

python train.py privacy_model_type="VIDEO_PRIVATIZER" pose_dataset.fps=5
python train.py privacy_model_type="PIXELATE" pose_dataset.fps=5
python train.py privacy_model_type="BLUR" pose_dataset.fps=5
python train.py privacy_model_type="DEEP_PRIVACY2" pose_dataset.fps=5 train.batch_size=2
python train.py privacy_model_type="STYLEGAN2" pose_dataset.fps=5

python train.py privacy_model_type="VIDEO_PRIVATIZER" pose_dataset.fps=5 train.camera_type=1 val.camera_type=1 test.camera_type=1
python train.py privacy_model_type="PIXELATE" pose_dataset.fps=5 train.camera_type=1 val.camera_type=1 test.camera_type=1
python train.py privacy_model_type="BLUR" pose_dataset.fps=5 train.camera_type=1 val.camera_type=1 test.camera_type=1
python train.py privacy_model_type="DEEP_PRIVACY2" pose_dataset.fps=5 train.camera_type=1 val.camera_type=1 test.camera_type=1 train.batch_size=2
python train.py privacy_model_type="STYLEGAN2" pose_dataset.fps=5 train.camera_type=1 val.camera_type=1 test.camera_type=1