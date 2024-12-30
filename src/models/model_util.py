import cv2
import pandas as pd
from pathlib import Path
import os
from src.models.skeleton.openpose.openpose_skeleton import OpenPoseAPI
from ..datasets.pose.keypoints_dataset import PoseDatasetKeypoints
import torch
import torchvision.transforms as T
import torchvision.io
from src.models.skeleton.yolo_skeleton import YOLOPoseAPI

        

def load_model(cfg):
    model = None
    if cfg.task == "classification":
        from src.models.pose_classifier import PoseClassifier
        model = PoseClassifier(input_shape=(cfg.train.batch_size, cfg.in_channels, cfg.depth, cfg.width, cfg.height), output_dim=cfg.output_dim, backbone=cfg.conv_backbone, freeze = cfg.yolo.freeze, detect_face=cfg.openpose.detect_face, detect_hands=cfg.openpose.detect_hands, fps=cfg.pose_dataset.fps, model_size=cfg.yolo.model_size)
    
    return model

def load_dataset_openpose(cfg):
    """
    Funzione per eseguire OpenPose sui video, estrarre keypoints e creare un dataset secondario.
    I video vengono ridimensionati e viene effettuato un undersampling prima di processarli con OpenPose.
    """
    if cfg.task == "classification":
        # Lista per raccogliere i dati elaborati
        processed_data = []

        # Load the CSV containing video information
        csv_path = cfg['pose_dataset']['csv_path']
        df = pd.read_csv(csv_path)

    

        def preprocess_video(video_path, target_size=(224, 224), target_fps=30):
            """
            Preprocessa un video trasformandolo in tensore, ridimensionandolo e regolando il numero di FPS.
            
            :param video_path: Percorso del video da preprocessare.
            :param target_size: Dimensione desiderata dei frame (larghezza, altezza).
            :param target_fps: Numero desiderato di FPS per il video preprocessato.
            :return: Tensore PyTorch del video preprocessato con forma (C, T, H, W).
            """
            cap = cv2.VideoCapture(str(video_path))
            original_fps = cap.get(cv2.CAP_PROP_FPS)

            if not cap.isOpened():
                raise FileNotFoundError(f"Impossibile aprire il file video: {video_path}")

            # Calcola l'intervallo di campionamento dei frame
            frame_interval = round(original_fps / target_fps)

            # Trasformazioni da applicare ai frame
            transform = T.Compose([
                T.ToPILImage(),          # Converte i frame in immagini PIL
                T.Resize(target_size),   # Ridimensiona i frame
                T.ToTensor()             # Converte in tensore PyTorch
            ])

            frames = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Considera solo i frame a intervalli specificati
                if frame_idx % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converti in formato RGB
                    frame_tensor = transform(frame)  # Applica le trasformazioni
                    frames.append(frame_tensor)

                frame_idx += 1

            cap.release()

            # Combina i frame in un tensore con forma (C, T, H, W)
            if frames:
                video_tensor = torch.stack(frames, dim=1)  # Combina lungo la dimensione temporale
            else:
                raise ValueError("Nessun frame valido trovato nel video.")

            return video_tensor

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
    """
    Funzione per eseguire OpenPose sui video, estrarre keypoints e creare un dataset secondario.
    I video vengono ridimensionati e viene effettuato un undersampling prima di processarli con OpenPose.
    """
    if cfg.task == "classification":
        # Lista per raccogliere i dati elaborati
        processed_data = []

        # Load the CSV containing video information
        csv_path = cfg['pose_dataset']['csv_path']
        df = pd.read_csv(csv_path)

        def process_data():
            processed_data = []

            yolo_pose_api = YOLOPoseAPI(model_size="n")  # Inizializza YOLOPoseAPI

            for _, row in df.iterrows():
                video_id = row['video_id']
                patient_id = row['patient_id']
                camera_type = row['camera_type']
                class_label = row['event']  # Class è direttamente la colonna "event"

                # Percorso del video originale
                video_path = Path(cfg['pose_dataset']['path'], f"{video_id}.mp4").resolve()

                # Legge il video come tensore (T, H, W, C)
                video, _, metadata = torchvision.io.read_video(str(video_path), pts_unit="sec")
                video = video.permute(3, 0, 1, 2)  # Cambia la forma in (C, T, H, W)
                original_fps = metadata['video_fps']

                # Preprocessa il video (ridimensiona e regola FPS)
                transform = T.Compose([
                    T.Resize((cfg['pose_dataset']['resize_h'], cfg['pose_dataset']['resize_w'])),
                    T.ConvertImageDtype(torch.float32)  # Normalizza i pixel nell'intervallo [0, 1]
                ])
                frame_interval = round(original_fps / cfg['pose_dataset']['fps'])
                preprocessed_frames = []

                # Riduci i frame al target FPS e applica trasformazioni
                for frame_idx in range(0, video.shape[1], frame_interval):
                    frame = video[:, frame_idx]  # Seleziona il frame corrente (C, H, W)
                    preprocessed_frame = transform(frame)
                    preprocessed_frames.append(preprocessed_frame)

                video_tensor = torch.stack(preprocessed_frames, dim=1)  # Ricostruisci il video come (C, T, H, W)

                # Estrai i keypoint per ogni frame
                for frame_idx in range(video_tensor.shape[1]):  # Itera sui frame preprocessati
                    frame = video_tensor[:, frame_idx]  # Ottieni il frame corrente (C, H, W)
                    keypoints = yolo_pose_api(frame.unsqueeze(0))  # Passa il frame alla rete YOLOPoseAPI
                    
                    # Salva i dati per ogni frame
                    processed_data.append({
                        'video_id': video_id,
                        'patient_id': patient_id,
                        'camera_type': camera_type,
                        'frame': frame_idx,
                        'keypoints': keypoints.numpy().flatten().tolist(),  # Converte in lista
                        'event': class_label
                    })

            return processed_data


        if cfg.extract_keypoints:
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



def label_to_one_hot(label, num_classes=9):
    """
    Converte un'etichetta in un vettore one-hot.
    
    :param label: Il valore dell'etichetta (tra 1 e num_classes).
    :param num_classes: Il numero totale di classi (default 9).
    :return: Un tensore one-hot di dimensione (num_classes,).
    """
    # Inizializza il vettore con zeri
    one_hot = torch.zeros(num_classes)
    
    # Imposta il valore corrispondente all'etichetta a 1
    one_hot[label - 1] = 1  # label-1 perché l'indice parte da 0
    
    return one_hot