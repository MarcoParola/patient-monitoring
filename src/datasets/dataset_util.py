from torchvision import transforms

def load_dataset(cfg):
    """
    Carica i dataset di training, validation e test in base alla configurazione specificata.
    
    Args:
        cfg: Configurazione con informazioni su path, task e trasformazioni.

    Returns:
        Tuple contenente train, val, test dataset.
    """
    # Ottenere le trasformazioni per i dataset
    train_transform, val_transform, test_transform = get_transform(cfg.pose_dataset.resize, cfg.pose_dataset.fps)
    
    # Inizializzazione dataset
    train, val, test = None, None, None

    if cfg.task == "classification":
        from src.datasets.pose.pose_dataset import PoseDatasetByPatients

        train = PoseDatasetByPatients(
            root=cfg.pose_dataset.path,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.train.patient_ids,
            transform=train_transform,
            camera_type=cfg.train.camera_type
        )
        val = PoseDatasetByPatients(
            root=cfg.pose_dataset.path,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.val.patient_ids,
            transform=val_transform,
            camera_type=cfg.val.camera_type
        )
        test = PoseDatasetByPatients(
            root=cfg.pose_dataset.path,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.test.patient_ids,
            transform=test_transform,
            camera_type=cfg.test.camera_type
        )

    elif cfg.task == "privacy":
        from src.datasets.pose.pose_dataset import PoseDatasetByPatientsPrivacy

        train = PoseDatasetByPatientsPrivacy(
            root=cfg.pose_dataset.path,
            root_privacy=cfg.path_privacy,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.train.patient_ids,
            transform=train_transform,
            camera_type=cfg.train.camera_type
        )
        val = PoseDatasetByPatientsPrivacy(
            root=cfg.pose_dataset.path,
            root_privacy=cfg.path_privacy,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.val.patient_ids,
            transform=val_transform,
            camera_type=cfg.val.camera_type
        )
        test = PoseDatasetByPatientsPrivacy(
            root=cfg.pose_dataset.path,
            root_privacy=cfg.path_privacy,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.test.patient_ids,
            transform=test_transform,
            camera_type=cfg.test.camera_type
        )

    return train, val, test

class DownsampleFrames:
    def __init__(self, fps):
        self.fps = fps

    def __call__(self, video):
        # Funzione per sottocampionare i frame in base al frame per secondo
        step = max(1, len(video) // self.fps)
        return video[::step]

def get_transform(resize=None, fps=None):
    transform_list = []

    # Aggiungi ridimensionamento se specificato
    if resize is not None:
        transform_list.insert(0, transforms.Resize((resize, resize)))

    # Aggiungi sottocampionamento se fps è specificato
    if fps is not None:
        transform_list.insert(0, DownsampleFrames(fps))

    # Componi la trasformazione
    composed_transform = transforms.Compose(transform_list)
    return composed_transform, composed_transform, composed_transform
