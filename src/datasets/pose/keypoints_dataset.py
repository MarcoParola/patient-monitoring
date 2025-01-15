import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.io

class PoseDatasetKeypoints(Dataset):
    def __init__(self, root, csv_path, patient_ids,camera_type, pose_map=None):
        """
        Initialize the dataset with frame_id and keypoints.

        Args:
            root: root directory of the dataset containing frames as images.
            csv_path: path to the CSV file containing frame information.
            patient_ids: list of patient ids to include in this dataset.
            transform: a transform function to apply to the images (default: None).
            pose_map: mapping for body part labels to numeric values (default: None).
        """
        self.root = root
        self.pose_map = pose_map
        
        # Load the CSV and filter by patient_ids
        self.data = pd.read_csv(csv_path, quotechar='"')
        self.data = self.data[self.data['patient_id'].isin(patient_ids)]
        if camera_type !="":
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
        Get the video tensor and label (body part) for the given index.

        Args:
            index: index of the data.

        Returns:
            A tuple containing:
                - keypoints_tensor: the keypoints data extracted as a PyTorch tensor.
                - label: the body part as an integer label.
        """
        # Retrieve row from the filtered DataFrame
        row = self.data.iloc[index]
        keypoints = eval(row["keypoints"]) # Convert the string to a Python list
        keypoints_tensor = torch.tensor(keypoints)  # Convert to a PyTorch tensor
        # Extract the label (body part) from the event
        event = row['event']
        if len(event) == 1 and 'body_part' in event[0]:
            label = event[0]['body_part']
        else:
            raise ValueError(f"Invalid event format: {event}")

        # Convert label to numeric using the pose_map
        label = self.pose_map.get(label, -1)  # If label is not found, set it to -1
        
        return keypoints_tensor, label