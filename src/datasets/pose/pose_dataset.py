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
    def __init__(self, root_privacy, privacy_map, **kwargs):
        """
        Dataset che estende PoseDatasetByPatients aggiungendo la gestione dei file JSON relativi alla privacy.

        Args:
            root_privacy (str): Directory contenente i file session.json con i metadati sulla privacy.
            privacy_map (dict): Mappatura per trasformare valori di genere e colore della pelle in numeri.
        """
        # Inizializza la classe madre
        super().__init__(**kwargs)

        self.root_privacy = root_privacy
        self.privacy_map = privacy_map
        self.patient_metadata = {}

        # Carica i metadati relativi alla privacy dai file session.json
        for patient_id in self.data['patient_id'].unique():
            privacy_folder = os.path.join(root_privacy, str(patient_id))
            session_file = os.path.join(privacy_folder, "session.json")
            if os.path.exists(session_file):
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                    # Filtra e salva i metadati sulla privacy
                    self.patient_metadata[patient_id] = {
                        "age": session_data.get("age"),
                        "skin_color": session_data.get("skin_color"),
                        "gender": session_data.get("gender")
                    }
            else:
                print(f"Warning: Il file session.json per il paziente {patient_id} non esiste.")

    def __getitem__(self, index):
        """
        Ottieni il video, l'evento dal CSV e i metadati sulla privacy associati al paziente.

        Args:
            index (int): Indice del campione.

        Returns:
            video_tensor (Tensor): Dati video come tensore PyTorch.
            event (dict): Evento associato al video.
            privacy_metadata (dict): Metadati relativi alla privacy.
        """
        # Ottieni video_tensor e label (l'evento) dalla classe madre
        video_tensor, event = super().__getitem__(index)

        # Ottieni il patient_id dal DataFrame
        patient_id = self.data.iloc[index]['patient_id']

        # Recupera i metadati sulla privacy per il paziente
        privacy_metadata = self.patient_metadata.get(patient_id, {})

        # Converti gender e skin_color in numeri usando privacy_map
        if "gender" in privacy_metadata:
            privacy_metadata["gender"] = torch.tensor([self.privacy_map.get(privacy_metadata["gender"], -1)], dtype=torch.int64)
        if "skin_color" in privacy_metadata:
            privacy_metadata["skin_color"] = torch.tensor([self.privacy_map.get(privacy_metadata["skin_color"], -1)], dtype=torch.int64)

        # Converte age in un tensor, se presente
        if "age" in privacy_metadata and privacy_metadata["age"] is not None:
            try:
                privacy_metadata["age"] = torch.tensor([int(privacy_metadata["age"])], dtype=torch.int64)
            except ValueError:
                privacy_metadata["age"] = torch.tensor([-1], dtype=torch.int64)  # Valore predefinito in caso di errore
        else:
            privacy_metadata["age"] = torch.tensor([-1], dtype=torch.int64)  # Default se non presente

        return video_tensor, (event, privacy_metadata)

if __name__ == "__main__":
    # Parameters for the test
    root = "dataset/position"
    csv_path = "dataset/position.csv"
    root_privacy = "data"
    patient_ids = [123, 234]
    transform = None
    camera_type = 0
    pose_map = {
        "Arm_Left": 0,
        "Arm_Right": 1,
        "Leg_Left": 2,
        "Leg_Right": 3,
        "Hand_Left": 4,
        "Hand_Right": 5,
        "Foot_Left": 6,
        "Foot_Right": 7,
        "Head": 8
    }
    privacy_map = {
        "white": 0, "brown": 1, "yellow": 2, "black": 3,
        "male": 0, "female": 1
    }

    # Test della classe PoseDatasetByPatients
    print("Testing PoseDatasetByPatients...")
    dataset_by_patients = PoseDatasetByPatients(
        root=root,
        csv_path=csv_path,
        patient_ids=patient_ids,
        transform=transform,
        camera_type=camera_type,
        pose_map=pose_map
    )

    dataloader_by_patients = DataLoader(dataset_by_patients, batch_size=2, shuffle=True)

    for video_tensor, label in dataloader_by_patients:
        print(f"Video tensor shape: {video_tensor.shape}")
        print(f"Label: {label}")
        break  # Esegui solo un batch per test

    # Test della classe PoseDatasetByPatientsPrivacy
    print("\nTesting PoseDatasetByPatientsPrivacy...")
    dataset_by_patients_privacy = PoseDatasetByPatientsPrivacy(
        root_privacy=root_privacy,
        privacy_map=privacy_map,
        root=root,
        csv_path=csv_path,
        patient_ids=patient_ids,
        transform=transform,
        camera_type=camera_type,
        pose_map=pose_map
    )

    dataloader_by_patients_privacy = DataLoader(dataset_by_patients_privacy, batch_size=2, shuffle=True)

    for video_tensor, (event, privacy_metadata) in dataloader_by_patients_privacy:
        print(f"Video tensor shape: {video_tensor.shape}")
        print(f"Event: {event}")
        print(f"Privacy metadata: {privacy_metadata}")
        break  # Esegui solo un batch per test
