from torchvision import transforms
import torch

def load_dataset(cfg):
    """
    Loads the training, validation, and test datasets based on the provided configuration.
    
    Args:
        cfg: Configuration with paths, task type, and transformations.

    Returns:
        Tuple containing train, validation, and test datasets.
    """
    # Get the transformations for the datasets
    train_transform, val_transform, test_transform = get_transform(cfg.pose_dataset.resize_height, cfg.pose_dataset.resize_width, cfg.pose_dataset.fps)
    
    # Initialize datasets
    train, val, test = None, None, None

    if cfg.task == "classification" and cfg.dataset == "pose":
        from src.datasets.pose.pose_dataset import PoseDatasetByPatients

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

    elif (cfg.task == "privacy" or cfg.task == "classification_privacy" or cfg.task == "privatizer") and cfg.dataset == "pose":
        from src.datasets.pose.pose_dataset import PoseDatasetByPatientsPrivacy

        train = PoseDatasetByPatientsPrivacy(
            root=cfg.pose_dataset.path,
            root_privacy=cfg.path_privacy,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.train.patient_ids,
            transform=train_transform,
            camera_type=cfg.train.camera_type,
            pose_map=cfg.pose_map,
            privacy_map=cfg.privacy_map
        )
        val = PoseDatasetByPatientsPrivacy(
            root=cfg.pose_dataset.path,
            root_privacy=cfg.path_privacy,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.val.patient_ids,
            transform=val_transform,
            camera_type=cfg.val.camera_type,
            pose_map=cfg.pose_map,
            privacy_map=cfg.privacy_map
        )
        test = PoseDatasetByPatientsPrivacy(
            root=cfg.pose_dataset.path,
            root_privacy=cfg.path_privacy,
            csv_path=cfg.pose_dataset.csv_path,
            patient_ids=cfg.test.patient_ids,
            transform=test_transform,
            camera_type=cfg.test.camera_type,
            pose_map=cfg.pose_map,
            privacy_map=cfg.privacy_map
        )

    elif cfg.dataset == "action":
        from torch.utils.data import random_split
        from src.datasets.action.hmdb_dataset import HMDBDatasetPrivacy

        torch.manual_seed(42)

        # Crea il dataset
        dataset = HMDBDatasetPrivacy(
            root=cfg.pose_dataset.path,
            csv_path=cfg.pose_dataset.csv_path,
            transform=test_transform,
            action_map=cfg.pose_map
        )

        dataset_size = len(dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.2 * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Dividi il dataset in modo riproducibile
        train, val, test = random_split(dataset, [train_size, val_size, test_size])

    return train, val, test

class SelectFrames:
    def __init__(self, fps):
        self.fps = fps

    def __call__(self, x):
        """
        Selects frames uniformly distributed from the sequence.

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

def get_transform(resize_height, resize_width, fps):
    """
    Returns the transformations for the training, validation, and test datasets.

    Args:
        resize_height: The height to resize the images to.
        resize_width: The width to resize the images to.
        fps: Number of desired frames.

    Returns:
        Tuple containing train_transform, val_transform, and test_transform.
    """
    # Common transformations
    common_transforms = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        SelectFrames(fps)
    ])

    train_transform = common_transforms
    val_transform = common_transforms
    test_transform = common_transforms

    return train_transform, val_transform, test_transform
