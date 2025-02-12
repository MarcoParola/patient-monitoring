import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.io

class HMDBDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, action_map=None, actions=None):
        """
        Initialize the dataset.

        Args:
            root: root directory of the dataset.
            csv_path: path to the main CSV file containing dataset information.
            transform: a transform function to apply to the videos (default: None).
            action_map: mappatura delle etichette per la conversione in valori numerici (default: None).
            actions: lista di azioni da considerare (default: None).
        """
        self.root = root
        self.transform = transform
        self.action_map = action_map  # Salva la mappatura nel dataset

        # Load the CSV and filter by patient_ids and camera_type
        self.data = pd.read_csv(csv_path, quotechar='"')  # Handle quoted strings properly
    
                

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Get the video tensor and label (body part) for the given index.

        Args:
            index: index of the data.

        Returns:
            A tuple containing:
                - video_tensor: the video data as a PyTorch tensor.
                - label: the body part as an integer label.
        """
        # Retrieve row from the filtered DataFrame
        row = self.data.iloc[index]

        # Extract video path and event
        video_id = row['video_id']
        action = row['action']

        # Construct the path to the video file
        video_path = os.path.join(self.root, f"{video_id}.avi")

        # Check if the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load video using torchvision.io.read_video (ignores audio)
        video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")

        # Permute dimensions to match PyTorch convention: C x T x H x W
        video_tensor = video.permute(3, 0, 1, 2).float() / 255.0  # From T x H x W x C to C x T x H x W
        
        # Apply transformation if specified
        if self.transform:
            video_tensor = self.transform(video_tensor)

        # Convert action to numeric label
        label = self.action_map[action]

        return video_tensor, label