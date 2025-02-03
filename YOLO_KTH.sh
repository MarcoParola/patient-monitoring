python train_cate.py conv_backbone="YOLO" dataset="KTH" yolo.model_size="n" KTH_dataset.processed_csv="dataset/KTH_YOLO_n_5.csv" KTH_dataset.fps=5 KTH_dataset.resize_w=160 KTH_dataset.resize_h=128 train.batch_size=8 train.max_epochs=500 train.lr=0.000001 extract_keypoints=True output_dim=6 KTH_dataset.train=[11, 12, 13, 14, 15, 16, 17, 18] KTH_dataset.test=[19, 20, 21, 23, 24, 25, 1, 4] KTH_dataset.val=[22, 2, 3, 5, 6, 7, 10, 8, 9]

python train_cate.py conv_backbone="YOLO" dataset="KTH" yolo.model_size="m" KTH_dataset.processed_csv="dataset/KTH_YOLO_m_5.csv" KTH_dataset.fps=5 KTH_dataset.resize_w=160 KTH_dataset.resize_h=128 train.batch_size=8 train.max_epochs=500 train.lr=0.000001 extract_keypoints=True output_dim=6 KTH_dataset.train=[11, 12, 13, 14, 15, 16, 17, 18] KTH_dataset.test=[19, 20, 21, 23, 24, 25, 1, 4] KTH_dataset.val=[22, 2, 3, 5, 6, 7, 10, 8, 9]

python train_cate.py conv_backbone="YOLO" dataset="KTH" yolo.model_size="x" KTH_dataset.processed_csv="dataset/KTH_YOLO_x_5.csv" KTH_dataset.fps=5 KTH_dataset.resize_w=160 KTH_dataset.resize_h=128 train.batch_size=8 train.max_epochs=500 train.lr=0.000001 extract_keypoints=True output_dim=6 KTH_dataset.train=[11, 12, 13, 14, 15, 16, 17, 18] KTH_dataset.test=[19, 20, 21, 23, 24, 25, 1, 4] KTH_dataset.val=[22, 2, 3, 5, 6, 7, 10, 8, 9]
