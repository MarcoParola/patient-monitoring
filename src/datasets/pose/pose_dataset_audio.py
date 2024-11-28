from pose_dataset import PoseDatasetByPatients
import torch
import torch.utils.data as data
import os
import pandas as pd
import json
import torchvision.io
import torchaudio
from IPython.display import Audio
sample_rate = 32000

class PoseDatasetByPatientsWithAudio(PoseDatasetByPatients):
    def __getitem__(self, index):
        """
        Get the video tensor, audio tensor, and label for the given index.

        Args:
            index: index of the data.

        Returns:
            A tuple containing:
                - video_tensor: the video data as a PyTorch tensor.
                - audio_tensor: the audio data as a PyTorch tensor (1-channel).
                - label: the event label associated with the video.
        """
        # Retrieve row from the filtered DataFrame
        row = self.data.iloc[index]

        # Extract video path and event
        video_id = row['video_id']
        event_json_str = row['event']

     
        # Construct the path to the video file
        video_path = os.path.join(self.root, f"{video_id}.mp4")

        # Check if the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load video and audio using torchvision.io.read_video
        video, audio, _ = torchvision.io.read_video(video_path, pts_unit="sec")

        # Permute dimensions to match PyTorch convention: C x T x H x W
        video_tensor = video.permute(3, 0, 1, 2)  # From T x H x W x C to C x T x H x W

        # Ensure audio is mono (1 channel)
        if audio.shape[0] > 1:  # If multi-channel, average to mono
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio_tensor = audio  # Already a tensor

        # Apply transformation to video if specified
        if self.transform:
            video_tensor = self.transform(video_tensor)


        return video_tensor, audio_tensor


# Test della classe estesa PoseDatasetByPatientsWithAudio
if __name__ == "__main__":
    # Parametri per il test
    root = "position"
    csv_path = "position.csv"
    patient_ids = [123]
    transform = None
    camera_type = 0

    # Creazione del dataset esteso
    dataset_with_audio = PoseDatasetByPatientsWithAudio(root=root, csv_path=csv_path, patient_ids=patient_ids, camera_type=camera_type)

    # Test della lunghezza
    assert len(dataset_with_audio) > 0, "La lunghezza del dataset dovrebbe essere maggiore di 0."
    print(f"Test __len__ passato! Lunghezza del dataset: {len(dataset_with_audio)}")

    # # Test di __getitem__
    # video_tensor, audio_tensor = dataset_with_audio[15]
    
    # # Salva l'audio come file WAV
    # torchaudio.save('audio_temp.wav', audio_tensor, sample_rate)  # 44100 è il sample rate

    # Riproduci il file audio
    Audio('audio_temp.wav')
    assert video_tensor is not None, "Il tensore del video non dovrebbe essere None."
    assert audio_tensor is not None, "Il tensore dell'audio non dovrebbe essere None."
    print(f"Test __getitem__ passato! Video Shape: {video_tensor.shape}, Audio Shape: {audio_tensor.shape}")
