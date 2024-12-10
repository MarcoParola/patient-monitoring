

def load_dataset(cfg):
    train, val, test = None, None, None
    if cfg.task == "classification":
        from src.datasets.pose.pose_dataset import PoseDatasetByPatients
        train = PoseDatasetByPatients(root=cfg.pose_dataset.path, csv_path=cfg.pose_dataset.csv_path, patient_ids=cfg.train.patient_ids, transform=None, camera_type=cfg.train.camera_type)
        val = PoseDatasetByPatients(root=cfg.pose_dataset.path, csv_path=cfg.pose_dataset.csv_path, patient_ids=cfg.val.patient_ids, transform=None, camera_type=cfg.val.camera_type)
        test = PoseDatasetByPatients(root=cfg.pose_dataset.path, csv_path=cfg.pose_dataset.csv_path, patient_ids=cfg.test.patient_ids, transform=None, camera_type=cfg.test.camera_type)
    elif cfg.task == "privacy":
        from src.datasets.pose.pose_dataset import PoseDatasetByPatientsPrivacy       
        
    return train, val, test