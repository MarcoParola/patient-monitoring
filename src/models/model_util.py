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
        model = PoseClassifier(input_shape=(cfg.train.batch_size, cfg.in_channels, cfg.depth, cfg.width, cfg.height), output_dim=cfg.output_dim, backbone=cfg.conv_backbone, detect_face=cfg.openpose.detect_face, detect_hands=cfg.openpose.detect_hands, fps=cfg.pose_dataset.fps, end=cfg.end)
    
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
            Preprocessa un video ridimensionandolo e regolando il numero di FPS.
            
            :param video_path: Percorso del video da preprocessare.
            :param target_size: Dimensione desiderata dei frame (larghezza, altezza).
            :param target_fps: Numero desiderato di FPS per il video preprocessato.
            :return: Percorso del video preprocessato.
            """
            cap = cv2.VideoCapture(str(video_path))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / original_fps
            
            # Percorso del video preprocessato
            preprocessed_video_path = video_path.parent / f"{video_path.stem}_preprocessed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(preprocessed_video_path), fourcc, target_fps, target_size)
            
            # Calcolo il numero totale di frame attesi
            target_total_frames = int(duration * target_fps)
            
            
            next_frame_time = 0  # Tempo del frame successivo da estrarre
            frame_idx = 0
            target_frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = frame_idx / original_fps
                if current_time >= next_frame_time:
                    resized_frame = cv2.resize(frame, target_size)
                    out.write(resized_frame)
                    target_frame_idx += 1
                    next_frame_time += 1 / target_fps

                    # Interrompo se raggiungo il numero desiderato di frame
                    if target_frame_idx >= target_total_frames:
                        break
                
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
                class_label = row['event'] 

                # Percorso del video originale
                video_path = Path(cfg['pose_dataset']['path'], f"{video_id}.mp4").resolve()
                
                # Preprocessa il video (ridimensiona e riduci FPS)
                preprocessed_video_path = preprocess_video(
                    video_path,
                    target_size=(cfg['pose_dataset']['resize_w'], cfg['pose_dataset']['resize_h']),
                    target_fps=cfg['pose_dataset']['fps']
                )
                            # Processa il video intero con OpenPose
                keypoints_data = OpenPoseAPI(video_path=preprocessed_video_path, detect_face=cfg['openpose']['detect_face'], detect_hands=cfg['openpose']['detect_hands']).keypoints

                # Crea una lista vuota per accumulare i keypoints per ogni video
                flat_keypoints = []

                # Salva i dati per ogni frame
                for frame_idx, keypoints in enumerate(keypoints_data):
                    #print(f"frame {frame_idx}")
                    # Aggiungi i keypoints del frame alla lista flat_keypoints
                    flat_keypoints.extend(keypoints)  # Unisce i keypoints del frame al vettore piatto

                # Dopo aver processato tutto il video, salva i dati
                processed_data.append({
                    'video_id': video_id,
                    'patient_id': patient_id,
                    'camera_type': camera_type,
                    'keypoints': flat_keypoints,  # Salva il vettore piatto dei keypoints
                    'event': class_label
                })
                

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

        def process_data(model_size="n"):
            processed_data = []

            yolo_pose_api = YOLOPoseAPI(model_size)  # Inizializza YOLOPoseAPI

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
                max_frames = int(5 * original_fps)
                video = video[:, :max_frames] if video.shape[1] > max_frames else video

                # Preprocessa il video (ridimensiona e regola FPS)
                transform = T.Compose([
                    T.Resize((cfg['pose_dataset']['resize_h'], cfg['pose_dataset']['resize_w'])),
                    T.ConvertImageDtype(torch.float32)  # Normalizza i pixel nell'intervallo [0, 1]
                ])
                frame_interval = round(original_fps / cfg['pose_dataset']['fps'])
                video_keypoints = []  # Vettore per accumulare tutti i keypoint del video
                for frame_idx in range(0, video.shape[1]-1, frame_interval):  # Itera sui frame con intervallo
                    frame = video[:, frame_idx]  # Seleziona il frame corrente (C, H, W)
                
                    # Applica trasformazioni al frame
                    preprocessed_frame = transform(frame)
                    
                    # Estrai i keypoint dal frame preprocessato
                    keypoints = yolo_pose_api(preprocessed_frame.unsqueeze(0))  # Passa il frame alla rete YOLOPoseAPI
                    print(keypoints.size())
                    # Concatenazione diretta dei keypoint
                    video_keypoints.extend(keypoints.flatten().tolist())
                    
                    
                    # Debug opzionale per verificare le dimensioni
                    print(f"Frame {frame_idx}: Keypoints Shape: {keypoints.shape}, Total Keypoints Shape: {pd.DataFrame(video_keypoints).shape}")

                # Salva i dati aggregati per il video
                processed_data.append({
                    'video_id': video_id,
                    'patient_id': patient_id,
                    'camera_type': camera_type,
                    'keypoints': video_keypoints,  # Vettore piatto di keypoint per tutto il video
                    'event': class_label
                })

            
            return processed_data


        if cfg.extract_keypoints:
            processed_data = process_data(cfg['yolo']['model_size'])
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

        print(train, val, test)
        return train, val, test



