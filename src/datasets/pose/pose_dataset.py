import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.io
import json  # For parsing JSON strings


class PoseDatasetByPatients(Dataset):
    def __init__(self, root, csv_path, patient_ids, transform=None, camera_type=None):
        """
        Initialize the dataset.

        Args:
            root: root directory of the dataset.
            csv_path: path to the main CSV file containing dataset information.
            patient_ids: list of patient ids to include in this dataset.
            transform: a transform function to apply to the videos (default: None).
            camera_type: optional filter for camera type (default: None).
        """
        self.root = root
        self.transform = transform

        # Load the CSV and filter by patient_ids and camera_type
        self.data = pd.read_csv(csv_path,quotechar='"')  # Handle quoted strings properly
       
        self.data = self.data[self.data['patient_id'].isin(patient_ids)]
        if camera_type is not None:
            self.data = self.data[self.data['camera_type'] == camera_type]

        # Convert the 'event' column from a JSON-like string to an actual Python object (list of dicts)
        self.data['event'] = self.data['event'].apply(lambda x: json.loads(x))

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Get the video tensor and event data for the given index.

        Args:
            index: index of the data.

        Returns:
            A tuple containing:
                - video_tensor: the video data as a PyTorch tensor.
                - event: parsed event information (list of dicts).
        """
        # Retrieve row from the filtered DataFrame
        row = self.data.iloc[index]

        # Extract video path and event
        video_id = row['video_id']
        event = row['event']  # Now event is a Python object (list of dicts)

        # Construct the path to the video file
        video_path = os.path.join(self.root, f"{video_id}.mp4")

        # Check if the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load video using torchvision.io.read_video (ignores audio)
        video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")

        # Permute dimensions to match PyTorch convention: C x T x H x W
        video_tensor = video.permute(3, 0, 1, 2)  # From T x H x W x C to C x T x H x W

        # Apply transformation if specified
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return video_tensor, event


# Test the PoseDatasetByPatients class
if __name__ == "__main__":
    # Parameters for the test
    root = "position"
    csv_path = "position.csv"
    patient_ids = [123]
    transform = None
    camera_type = 0

    # Create the dataset
    dataset = PoseDatasetByPatients(root=root, csv_path=csv_path, patient_ids=patient_ids, camera_type=camera_type)

    # Test dataset length
    assert len(dataset) > 0, "Dataset length should be greater than 0."
    print(f"Test __len__ passed! Dataset length: {len(dataset)}")

    # Test __getitem__
    video_tensor, event = dataset[0]
    assert video_tensor is not None, "Video tensor should not be None."
    assert event is not None, "Event data should not be None."
    print(f"Test __getitem__ passed! Video Shape: {video_tensor.shape}, Event: {event}")
