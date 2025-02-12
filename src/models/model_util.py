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
from src.datasets.dataset_util import SelectFrames
import csv
import numpy as np
        

def load_model(cfg):
    model = None
    if cfg.task == "classification":
        from src.models.pose_classifier import PoseClassifier
        model = PoseClassifier(input_shape=(cfg.train.batch_size, cfg.in_channels, cfg.depth, cfg.width, cfg.height), output_dim=cfg.output_dim, backbone=cfg.conv_backbone, detect_face=cfg.openpose.detect_face, detect_hands=cfg.openpose.detect_hands, dataset=cfg.dataset, fps=cfg.pose_dataset.fps, end=cfg.end, lr = cfg.train.lr)
    
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
            target_total_frames = int(3 * target_fps)
            
            
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


        def process_kth_data(cfg):
            df = pd.read_csv(cfg['KTH_dataset']['csv_path'])
            processed_data = []
            
            for _, row in df.iterrows():
                video_name = row['video_name']
                class_label = row['class']
                person_id = row['person_id']
                
                # Percorso del video originale
                video_path = Path(cfg['KTH_dataset']['path'], video_name).resolve()
                
                # Preprocessa il video (ridimensiona e riduci FPS)
                preprocessed_video_path = preprocess_video(
                    video_path,
                    target_size=(cfg['KTH_dataset']['resize_w'], cfg['KTH_dataset']['resize_h']),
                    target_fps=cfg['KTH_dataset']['fps']
                )
                
                # Processa il video con OpenPose
                keypoints_data = OpenPoseAPI(video_path=preprocessed_video_path, detect_hands=cfg['openpose']['detect_hands'], detect_face=cfg['openpose']['detect_face']).keypoints
                
                # Crea una lista vuota per accumulare i keypoints per ogni video
                flat_keypoints = []
                
                # Salva i dati per ogni frame
                for frame_idx, keypoints in enumerate(keypoints_data):
                    flat_keypoints.extend(keypoints)  # Unisce i keypoints del frame al vettore piatto
                
                # Dopo aver processato tutto il video, salva i dati
                processed_data.append({
                    'video_name': video_name,
                    'class': class_label,
                    'person_id': person_id,
                    'keypoints': flat_keypoints  # Salva il vettore piatto dei keypoints
                })
                
                # Cancella il video preprocessato
                if preprocessed_video_path.exists():
                    os.remove(preprocessed_video_path)
            
            return processed_data
        

        if(cfg.extract_keypoints and cfg.conv_backbone == "OpenPose"):
            if(cfg.dataset == "KTH"):
                processed_data = process_kth_data(cfg)
                processed_data_path = Path(cfg['KTH_dataset']['processed_csv']).resolve()
                pd.DataFrame(processed_data).to_csv(processed_data_path, index=False)
            else:
                processed_data = process_data()        
                # Salva i dati elaborati in un CSV
                processed_data_path = Path(cfg['pose_dataset']['processed_csv']).resolve()
                pd.DataFrame(processed_data).to_csv(processed_data_path, index=False)
        
        if(cfg.dataset == "Custom"):
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

      
        def process_data(df, model_size="n"):
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
            

                # Crea un'istanza di SelectFrames con fps desiderato
                select_frames = SelectFrames(fps=3*cfg['pose_dataset']['fps'])

                # Seleziona i frame desiderati
                selected_frames = select_frames(video)

                # Preprocessa i frame (ridimensiona e regola FPS)
                transform = T.Compose([
                    T.Resize((cfg['pose_dataset']['resize_h'], cfg['pose_dataset']['resize_w'])),
                    T.ConvertImageDtype(torch.float32)  # Normalizza i pixel nell'intervallo [0, 1]
                ])
                
                video_keypoints = []  # Vettore per accumulare tutti i keypoint del video
                for frame in selected_frames.permute(1, 0, 2, 3):  # Itera sui frame selezionati (C, H, W)
                    # Applica trasformazioni al frame
                    preprocessed_frame = transform(frame)
                    
                    # Estrai i keypoint dal frame preprocessato
                    keypoints = yolo_pose_api(preprocessed_frame.unsqueeze(0))  # Passa il frame alla rete YOLOPoseAPI
                    
                    # Concatenazione diretta dei keypoint
                    video_keypoints.extend(keypoints.flatten().tolist())
                    
                    # Debug opzionale per verificare le dimensioni
                    print(f"Frame {frame}: Keypoints Shape: {keypoints.shape}, Total Keypoints Shape: {pd.DataFrame(video_keypoints).shape}")

                # Salva i dati aggregati per il video
                processed_data.append({
                    'video_id': video_id,
                    'patient_id': patient_id,
                    'camera_type': camera_type,
                    'keypoints': video_keypoints,  # Vettore piatto di keypoint per tutto il video
                    'event': class_label
                })

            return processed_data

        def process_data_KTH(df, model_size="n"):
            processed_data = []

            yolo_pose_api = YOLOPoseAPI(model_size)  # Initialize YOLOPoseAPI

            for _, row in df.iterrows():
                video_name = row['video_name']
                class_label = row['class']
                person_id = row['person_id']

                # Path to the original video
                video_path = Path(cfg['KTH_dataset']['path'], video_name).resolve()

                # Read the video as a tensor (T, H, W, C)
                video, _, metadata = torchvision.io.read_video(str(video_path), pts_unit="sec")
                video = video.permute(3, 0, 1, 2)  # Change shape to (C, T, H, W)

                # Create an instance of SelectFrames with the desired fps
                select_frames = SelectFrames(fps=3*cfg['KTH_dataset']['fps'])

                # Select the desired frames
                selected_frames = select_frames(video)

                # Preprocess the frames (resize and adjust FPS)
                transform = T.Compose([
                    T.Resize((cfg['KTH_dataset']['resize_h'], cfg['KTH_dataset']['resize_w'])),
                    T.ConvertImageDtype(torch.float32)  # Normalize pixels to the range [0, 1]
                ])

                video_keypoints = []  # List to accumulate all keypoints for the video
                for frame in selected_frames.permute(1, 0, 2, 3):  # Iterate through the selected frames (C, H, W)
                    # Apply transformations to the frame
                    preprocessed_frame = transform(frame)
                    
                    # Extract the keypoints from the preprocessed frame
                    keypoints = yolo_pose_api(preprocessed_frame.unsqueeze(0))  # Pass the frame to YOLOPoseAPI
                    
                    # Directly concatenate the keypoints
                    video_keypoints.extend(keypoints.flatten().tolist())

                    # Optional debug to check dimensions
                    print(f"Frame {frame}: Keypoints Shape: {keypoints.shape}, Total Keypoints Shape: {pd.DataFrame(video_keypoints).shape}")

                # Save the aggregated data for the video
                processed_data.append({
                    'video_name': video_name,
                    'person_id': person_id,
                    'keypoints': video_keypoints,  # Flattened keypoint vector for the entire video
                    'class': class_label
                })

            return processed_data


        if cfg.extract_keypoints:
            # Salva i dati elaborati in un CSV
            if cfg.dataset == "KTH":
                csv_path = cfg['KTH_dataset']['csv_path']
                df = pd.read_csv(csv_path)
                processed_data = process_data_KTH(df, cfg['yolo']['model_size'])
                processed_data_path = Path(cfg['KTH_dataset']['processed_csv']).resolve()
            else:
                csv_path = cfg['pose_dataset']['csv_path']
                df = pd.read_csv(csv_path)
                processed_data = process_data(df, cfg['yolo']['model_size'])
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



