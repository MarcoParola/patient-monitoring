python train_cate.py conv_backbone="CNN3D" output_dim=6 end="mlp" dataset="KTH" KTH_dataset.fps=5 KTH_dataset.resize_w=160 KTH_dataset.resize_h=120 train.batch_size=8 train.max_epochs=1000 train.lr=0.000001 KTH_dataset.path="dataset/KTH" KTH_dataset.csv_path="dataset/KTH_cut.csv" KTH_dataset.train=[11, 12, 13, 14, 15, 16, 17, 18] KTH_dataset.test=[19, 20, 21, 23, 24, 25, 1, 4] KTH_dataset.val=[22, 2, 3, 5, 6, 7, 10, 8, 9]
python train_cate.py conv_backbone="CNN3D" output_dim=6 end="mlp" dataset="KTH" KTH_dataset.fps=8 KTH_dataset.resize_w=160 KTH_dataset.resize_h=120 train.batch_size=8 train.max_epochs=1000 train.lr=0.000001 KTH_dataset.path="dataset/KTH" KTH_dataset.csv_path="dataset/KTH_cut.csv" KTH_dataset.train=[11, 12, 13, 14, 15, 16, 17, 18] KTH_dataset.test=[19, 20, 21, 23, 24, 25, 1, 4] KTH_dataset.val=[22, 2, 3, 5, 6, 7, 10, 8, 9]
python train_cate.py conv_backbone="CNN3D" output_dim=6 end="mlp" dataset="KTH" KTH_dataset.fps=10 KTH_dataset.resize_w=160 KTH_dataset.resize_h=120 train.batch_size=8 train.max_epochs=1000 train.lr=0.000001 KTH_dataset.path="dataset/KTH" KTH_dataset.csv_path="dataset/KTH_cut.csv" KTH_dataset.train=[11, 12, 13, 14, 15, 16, 17, 18] KTH_dataset.test=[19, 20, 21, 23, 24, 25, 1, 4] KTH_dataset.val=[22, 2, 3, 5, 6, 7, 10, 8, 9]

