import cv2
import numpy as np
import os

def video_to_numpy(video_path):
        """
        Carica un video e lo converte in un array NumPy.
        :param video_path: Percorso del video.
        :return: Array NumPy contenente i frame del video.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames, dtype=np.uint8)

def ensure_bgr(frame):
    """
    Garantisce che un frame sia nel formato BGR a 3 canali.
    :param frame: Frame in formato HWC.
    :return: Frame convertito in formato BGR.
    """
    if len(frame.shape) == 2:  # Greyscale (H, W)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 1:  # Greyscale con un canale esplicito (H, W, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame
