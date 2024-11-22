import torch
import torch.utils.data as data
import os



class PoseDatasetByPatients(data.Dataset):
    def __init__(self, root, video_path, patient_ids, transform=None):
        """ Intialize the dataset

        Args:
            root: root directory of the dataset
            patient_ids: list of patient ids that will be used for this dataset
            transform: a transform function to apply to the videos
        """
        self.root = root
        self.transform = transform
        patients = os.listdir(root)


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass