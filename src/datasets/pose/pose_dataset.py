import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.io
import json

class PoseDatasetByPatients(Dataset):
    def __init__(self, root, csv_path, patient_ids, transform=None, camera_type=None, pose_map=None):
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
        self.pose_map = pose_map  # Salva la mappatura nel dataset

        # Load the CSV and filter by patient_ids and camera_type
        self.data = pd.read_csv(csv_path, quotechar='"')  # Handle quoted strings properly
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
        event = row['event']

        # Construct the path to the video file
        video_path = os.path.join(self.root, f"{video_id}.mp4")

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

        # Extract the label (body part) from the event
        if len(event) == 1 and 'body_part' in event[0]:
            label = event[0]['body_part']
        else:
            raise ValueError(f"Invalid event format: {event}")

        # Convert label to numeric using the pose_map
        label = self.pose_map.get(label, -1)  # If label is not found, set it to -1

        return video_tensor, label

class PoseDatasetByPatientsPrivacy(PoseDatasetByPatients):
    def __init__(self, root_privacy, **kwargs):
        """
        Dataset that extends PoseDatasetByPatients by adding handling of privacy-related JSON files.

        Args:
            root_privacy (str): Directory containing privacy-related session.json files.
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        self.root_privacy = root_privacy
        self.patient_metadata = {}

        # Load privacy metadata from session.json files
        for patient_id in patient_ids:
            privacy_folder = os.path.join(root_privacy, str(patient_id))
            session_file = os.path.join(privacy_folder, "session.json")
            if os.path.exists(session_file):
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                    # Filter only relevant privacy attributes
                    filtered_metadata = {
                        "age": session_data.get("age"),
                        "skin_color": session_data.get("skin_color"),
                        "gender": session_data.get("gender") 
                    }
                    self.patient_metadata[patient_id] = filtered_metadata
            else:
                print(f"Warning: The session.json file for patient {patient_id} does not exist.")

    def __getitem__(self, index):
        """
        Get the video, CSV metadata, and privacy metadata associated with the patient.

        Args:
            index (int): Index of the sample.

        Returns:
            video (Tensor): Video data as a PyTorch tensor.
            csv_metadata (dict): Metadata from the CSV file.
            privacy_metadata (dict): Privacy-related metadata for the patient.
        """
        # Get video and CSV metadata using the base class method
        video_tensor, csv_metadata = super().__getitem__(index)

        # Extract the patient ID from the CSV metadata
        patient_id = self.data.iloc[index]['patient_id']

        # Retrieve privacy metadata for the given patient
        privacy_metadata = self.patient_metadata.get(patient_id, {})

        return video_tensor, (csv_metadata, privacy_metadata)

# Test the PoseDatasetByPatients class
if __name__ == "__main__":
    # Parameters for the test
    root = "dataset/position"
    csv_path = "dataset/position.csv"
    root_privacy = "data"
    patient_ids = [123, 234]
    transform = None
    camera_type = 0

    # Verifica dei percorsi
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Il file CSV non esiste: {csv_path}")
    if not os.path.exists(root):
        raise FileNotFoundError(f"La cartella dei video non esiste: {root}")
    for patient_id in patient_ids:
        session_file = os.path.join(root_privacy, str(patient_id), "session.json")
        if not os.path.exists(session_file):
            print(f"Avviso: Il file session.json non esiste per il paziente {patient_id}: {session_file}")

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

    # Inizializzazione del dataset esteso
    dataset = PoseDatasetByPatientsPrivacy(
        root=root,
        csv_path=csv_path,
        patient_ids=patient_ids,
        root_privacy=root_privacy,
        camera_type=camera_type,
    )

    # Creazione del DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Test del dataset
    print("\nTest del dataset:")
    for i, (video, metadata) in enumerate(dataloader):
        csv_metadata, privacy_metadata = metadata
        print(f"\nSample {i + 1}:")
        print(f"- Video shape: {video.shape}")  # Forma del tensore video
        print(f"- CSV Metadata: {csv_metadata}")  # Metadati dal CSV
        print(f"- Privacy Metadata: {privacy_metadata}")  # Metadati sulla privacy
