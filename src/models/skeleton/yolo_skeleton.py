import os
import torch
import pytorch_lightning as pl
from ultralytics import YOLO
import numpy as np
from src.models.skeleton.utils import *

class YOLOPoseAPI(pl.LightningModule):
    def __init__(self, model_size="n"):
        """
        Inizializza il modulo Lightning per YOLO Pose.
        :param model_path: Percorso al file YOLO pre-addestrato (.pt).
        :param feature_dim: Dimensione dell'output del backbone.
        """
        super(YOLOPoseAPI, self).__init__()
        model_path = os.path.join(f"src/models/skeleton/yolo/yolo11{model_size}-pose.pt")
        self.model = YOLO(model_path)
        self.feature_dim = 17*2
        

    def forward(self, frame):
        """
        Esegue l'inferenza YOLO su un frame e restituisce i keypoint.
        :param frame: Frame dell'immagine in formato numpy array (BGR).
        :return: Tensor contenente i keypoint estratti.
        """
        frame = ensure_bgr(frame)  # Garantisce il formato BGR
        results = self.model.predict(frame)
        keypoints = self.extract_keypoints(results)
        
        return keypoints  # Appiattisce i keypoint in un tensore

    def extract_keypoints(self, results):
        """
        Estrae i keypoint dai risultati YOLO in formato tensoriale.
        :param results: Risultati YOLO.
        :return: Tensor contenente i keypoint con shape (batch_size, num_keypoints, 2).
        """
        keypoints_list = []  # Utilizza una lista per raccogliere i tensori dei keypoint
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints.has_visible:
                # Estrai i keypoints da result.keypoints.xy, che ha forma (1, 17, 2)
                kp = result.keypoints.xy[0, :, :]  # Estrai le coordinate (x, y) dei keypoints
            else:
                # Se non ci sono keypoints visibili, crea un array di zeri di forma (17, 2)
                kp = torch.zeros((17, 2))
            
            keypoints_list.append(kp)  # Aggiungi il tensore dei keypoint alla lista

        return torch.stack(keypoints_list)  # Combina i tensori in uno solo
