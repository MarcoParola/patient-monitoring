import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.io
import json

class KTHDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, KTH_actions=None, actions=None):
        """
        Initialize the dataset.

        Args:
            root: root directory of the dataset.
            csv_path: path to the main CSV file containing dataset information.
            patient_ids: list of patient ids to include in this dataset.
            transform: a transform function to apply to the videos (default: None).
            camera_type: optional filter for camera type (default: None).
            pose_map: mappatura delle etichette per la conversione in valori numerici (default: None).
        """
        self.root = root
        self.transform = transform
        self.pose_map = KTH_actions  # Salva la mappatura nel dataset

        # Load the CSV and filter by patient_ids and camera_type
        self.data = pd.read_csv(csv_path, quotechar='"')  # Handle quoted strings properly
    
        if actions is not None:
            self.data = self.data[self.data['action'] in actions]
              
        

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
        video_id = row['video_name']
        event = row['class']

        # Construct the path to the video file
        video_path = os.path.join(self.root, event, f"{video_id}")
        print(video_path)

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
    

        # Convert label to numeric using the pose_map
        label = self.pose_map.get(event, -1)  # If label is not found, set it to -1

        return video_tensor, label
