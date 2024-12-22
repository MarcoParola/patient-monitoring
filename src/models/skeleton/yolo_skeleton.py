import os
import torch
import pytorch_lightning as pl
from ultralytics import YOLO
import numpy as np
from src.models.skeleton.utils import *

class YOLOPoseAPI(pl.LightningModule):
    def __init__(self, model_path):
        """
        Inizializza il modulo Lightning per YOLO Pose.
        :param model_path: Percorso al file YOLO pre-addestrato (.pt).
        :param feature_dim: Dimensione dell'output del backbone.
        """
        super(YOLOPoseAPI, self).__init__()
        self.model = YOLO(model_path).requires_grad_(False)
        self.feature_dim = 17*3
        

    def forward(self, frame):
        """
        Esegue l'inferenza YOLO su un frame e restituisce i keypoint.
        :param frame: Frame dell'immagine in formato numpy array (BGR).
        :return: Tensor contenente i keypoint estratti.
        """
        frame = ensure_bgr(frame)  # Garantisce il formato BGR
        results = self.model.predict(frame)
        keypoints = self.extract_keypoints(results)
        print(keypoints.size())
        return keypoints

    def extract_keypoints(self, results):
        """
        Estrae i keypoint dai risultati YOLO in formato tensoriale.
        :param results: Risultati YOLO.
        :return: Tensor contenente i keypoint con shape (batch_size, num_keypoints, 2).
        """
        keypoints = []
        for result in results:
            if len(result.keypoints) > 0:
                kp = result.keypoints[:, :2] 
            else:
                kp = torch.zeros((17, 2)) 
            keypoints.append(kp)
        return torch.stack(keypoints)  

   

