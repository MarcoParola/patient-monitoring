import cv2
import pandas as pd
from pathlib import Path
import os
from src.models.skeleton.openpose.openpose_skeleton import OpenPoseAPI
from ..datasets.pose.keypoints_dataset import PoseDatasetKeypoints


def load_model(cfg):
    model = None
    if cfg.task == "classification":
        from src.models.pose_classifier import PoseClassifier
        model = PoseClassifier(input_shape=(cfg.train.batch_size, cfg.in_channels, cfg.depth, cfg.width, cfg.height), output_dim=cfg.output_dim, backbone=cfg.conv_backbone, freeze = cfg.freeze, detect_face=cfg.detect_face, detect_hands=cfg.detect_hands, fps=cfg.pose_dataset.fps)
    
    return model

def load_dataset_openpose(cfg):
    """
    Funzione per eseguire OpenPose sui video, estrarre keypoints e creare un dataset secondario.
    I video vengono ridimensionati e viene effettuato un undersampling prima di processarli con OpenPose.
    """
    
    # Lista per raccogliere i dati elaborati
    processed_data = []

    # Load the CSV containing video information
    csv_path = cfg['pose_dataset']['csv_path']
    df = pd.read_csv(csv_path)

    def preprocess_video(video_path, target_size=(224, 224), target_fps=30):
        """
        Preprocessa un video ridimensionandolo e regolando il numero di FPS.
        
        :param video_path: Percorso del video da preprocessare.
        :param target_size: Dimensione desiderata dei frame (larghezza, altezza).
        :param target_fps: Numero desiderato di FPS per il video preprocessato.
        :return: Percorso del video preprocessato.
        """
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Percorso del video preprocessato
        preprocessed_video_path = video_path.parent / f"{video_path.stem}_preprocessed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(preprocessed_video_path), fourcc, target_fps, target_size)

        # Calcola l'intervallo di campionamento dei frame
        frame_interval = round(original_fps / target_fps)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Scrive solo i frame a intervalli specificati
            if frame_idx % frame_interval == 0:
                resized_frame = cv2.resize(frame, target_size)
                out.write(resized_frame)

            frame_idx += 1

        cap.release()
        out.release()

        return preprocessed_video_path

    # Funzione per processare i dati (train, val, test)
    def process_data():
        for _, row in df.iterrows():
            video_id = row['video_id']
            patient_id = row['patient_id']
            camera_type = row['camera_type']   
            class_label = row['event']  # Class è direttamente la colonna "event"

            # Percorso del video originale
            video_path = Path(cfg['pose_dataset']['path'], f"{video_id}.mp4").resolve()

            # Preprocessa il video (ridimensiona e riduci FPS)
            preprocessed_video_path = preprocess_video(
                video_path,
                target_size=(cfg['pose_dataset']['resize_h'], cfg['pose_dataset']['resize_w']),
                target_fps=cfg['pose_dataset']['fps']
            )

            # Processa il video intero con OpenPose
            keypoints_data = OpenPoseAPI(video_path=preprocessed_video_path, detect_face=cfg['detect_face'], detect_hands=cfg['detect_hands']).keypoints
            # Salva i dati per ogni frame
            for frame_idx, keypoints in enumerate(keypoints_data):
                print(f"frame {frame_idx}, keypoints: {keypoints}")
                processed_data.append({
                    'video_id': video_id,
                    'patient_id': patient_id,
                    'camera_type': camera_type,
                    'frame': frame_idx,
                    'keypoints': keypoints,
                    'event': class_label
                })
                print(process_data)

            # Cancella il video preprocessato
            if preprocessed_video_path.exists():
                os.remove(preprocessed_video_path)
            return processed_data

    if(cfg.extract_keypoints and cfg.conv_backbone == "OpenPose"):
        processed_data = process_data()        
        # Salva i dati elaborati in un CSV
        processed_data_path = Path(cfg['pose_dataset']['processed_csv']).resolve()
        pd.DataFrame(processed_data).to_csv(processed_data_path, index=False)

    train = PoseDatasetKeypoints(
        root=cfg.pose_dataset.path,
        csv_path=cfg.pose_dataset.processed_csv,
        patient_ids=cfg.train.patient_ids,
        camera_type=cfg.train.camera_type,
        pose_map=cfg.pose_map
    )
    val = PoseDatasetKeypoints(
        root=cfg.pose_dataset.path,
        csv_path=cfg.pose_dataset.processed_csv,
        patient_ids=cfg.val.patient_ids,
        camera_type=cfg.val.camera_type,
        pose_map=cfg.pose_map
    )
    test = PoseDatasetKeypoints(
        root=cfg.pose_dataset.path,
        csv_path=cfg.pose_dataset.processed_csv,
        patient_ids=cfg.test.patient_ids,
        camera_type=cfg.test.camera_type,
        pose_map=cfg.pose_map
    )

    print(train)
    return train, val, test



def load_dataset_yolo(cfg):


    train = PoseDatasetKeypoints(
        root=cfg.pose_dataset.path,
        csv_path=cfg.pose_dataset.processed_csv,
        patient_ids=cfg.train.patient_ids,
        camera_type=cfg.train.camera_type,
        pose_map=cfg.pose_map
    )
    val = PoseDatasetKeypoints(
        root=cfg.pose_dataset.path,
        csv_path=cfg.pose_dataset.processed_csv,
        patient_ids=cfg.val.patient_ids,
        camera_type=cfg.val.camera_type,
        pose_map=cfg.pose_map
    )
    test = PoseDatasetKeypoints(
        root=cfg.pose_dataset.path,
        csv_path=cfg.pose_dataset.processed_csv,
        patient_ids=cfg.test.patient_ids,
        camera_type=cfg.test.camera_type,
        pose_map=cfg.pose_map
    )

    print(train)
    return train, val, test
