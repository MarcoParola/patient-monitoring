from torchvision import transforms
import torch

def load_dataset(cfg):
    """
    Carica i dataset di training, validation e test in base alla configurazione specificata.
    
    Args:
        cfg: Configurazione con informazioni su path, task e trasformazioni.

    Returns:
        Tuple contenente train, val, test dataset.
    """
    # Ottenere le trasformazioni per i dataset
    train_transform, val_transform, test_transform = get_transform(cfg.pose_dataset.resize_h, cfg.pose_dataset.resize_w, cfg.pose_dataset.fps)
    
    # Inizializzazione dataset
    train, val, test = None, None, None

    if cfg.task == "classification":
        from src.datasets.pose.pose_dataset import PoseDatasetByPatients
        print("CGF",cfg.train.camera_type)
        print("CFG",cfg.test.camera_type)

        train = PoseDatasetByPatients(
            root=cfg.pose_dataset.path,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.train.patient_ids,
            transform=train_transform,
            camera_type=cfg.train.camera_type,
            pose_map=cfg.pose_map
        )
        val = PoseDatasetByPatients(
            root=cfg.pose_dataset.path,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.val.patient_ids,
            transform=val_transform,
            camera_type=cfg.val.camera_type,
            pose_map=cfg.pose_map
        )
        test = PoseDatasetByPatients(
            root=cfg.pose_dataset.path,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.test.patient_ids,
            transform=test_transform,
            camera_type=cfg.test.camera_type,
            pose_map=cfg.pose_map
        )

    elif cfg.task == "privacy":
        from src.datasets.pose.pose_dataset import PoseDatasetByPatientsPrivacy

        train = PoseDatasetByPatientsPrivacy(
            root=cfg.pose_dataset.path,
            root_privacy=cfg.path_privacy,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.train.patient_ids,
            transform=train_transform,
            camera_type=cfg.train.camera_type,
            pose_map=cfg.pose_map
        )
        val = PoseDatasetByPatientsPrivacy(
            root=cfg.pose_dataset.path,
            root_privacy=cfg.path_privacy,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.val.patient_ids,
            transform=val_transform,
            camera_type=cfg.val.camera_type,
            pose_map=cfg.pose_map
        )
        test = PoseDatasetByPatientsPrivacy(
            root=cfg.pose_dataset.path,
            root_privacy=cfg.path_privacy,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.test.patient_ids,
            transform=test_transform,
            camera_type=cfg.test.camera_type,
            pose_map=cfg.pose_map
        )

    return train, val, test

class SelectFrames:
    def __init__(self, fps):
        self.fps = fps

    def __call__(self, x):
        """
        Select frames uniformly distributed from the sequence.

        Args:
            x (torch.Tensor): Tensor of shape (C, T, H, W), where T is the number of frames.

        Returns:
            torch.Tensor: Tensor with the selected frames, of shape (C, self.fps, H, W).
        """
        num_frames = x.shape[1]
        if num_frames < self.fps:
            raise ValueError(f"The number of frames ({num_frames}) is less than the required frames ({self.fps}).")
        
        # Calculate uniformly distributed indices
        indices = torch.linspace(0, num_frames - 1, steps=self.fps).long()
        return x[:, indices, :, :]

def get_transform(resize_h, resize_w, fps):
    """
    Returns the transformations for the training, validation, and test datasets.

    Args:
        resize: Tuple with the new dimensions (height, width).
        fps: Number of desired frames.

    Returns:
        Tuple containing train_transform, val_transform, test_transform.
    """
    # Common transformations
    common_transforms = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        SelectFrames(fps)
    ])

    train_transform = common_transforms
    val_transform = common_transforms
    test_transform = common_transforms

    return train_transform, val_transform, test_transform


