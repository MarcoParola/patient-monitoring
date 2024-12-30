import os
import subprocess
import json
import torch 
from pathlib import Path

class OpenPoseAPI:
    def __init__(self, video_path= None, detect_face=False, detect_hands=False, fps=30 ):
         # Determinare il numero di feature in base ai parametri di rilevamento
        
        if video_path is not None:            
            self.keypoints = self.process_video_with_openpose(video_path, detect_face=detect_face, detect_hands=detect_hands)
            
   

    def process_video_with_openpose(self, video_path, openpose_path="bin/OpenPoseDemo.exe", 
                                    output_format="return", detect_face=False, detect_hands=False, 
                                    output_dir="output"):
        """
        Processa un video frame per frame con OpenPose, aggiungendo i parametri facoltativi per il rilevamento viso e mani.
        :param video_path: Percorso del video .mp4.
        :param openpose_path: Percorso del binario OpenPoseDemo.exe.
        :param output_format: Formato di output desiderato ('json', 'video', 'images').
        :param detect_face: Abilita il rilevamento del viso se True.
        :param detect_hands: Abilita il rilevamento delle mani se True.
        :param output_dir: Directory di destinazione per i risultati.
        :return: Lista di keypoints estratti dai file JSON.
        """
        # Creare la directory di output se non esiste
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Creare la lista dei comandi base
        command = [openpose_path]
        # Se si vogliono usare le immagini invece di un video
        if video_path:
            command.extend(["--video", video_path])

        # Utilizzare pathlib per costruire il percorso dell'output JSON
        json_output_dir = output_path / "output_jsons"
        json_output_dir.mkdir(parents=True, exist_ok=True)

        command.extend(["--write_json", str(json_output_dir)])
        command.extend(["--display", "0"]) 
        command.extend(["--render_pose", "0"])  
        
        # Aggiungere i parametri per il rilevamento del viso e delle mani
        if detect_face:
            command.append("--face")
        if detect_hands:
            command.append("--hand")

        os.chdir("src/models/skeleton/openpose")
        # # Esegui il comando nella directory specificata
        subprocess.run(command, check=True)
        os.chdir("../../../..")

        return self._extract_keypoints_from_json("src/models/skeleton/openpose/output/output_jsons", detect_face=detect_face, detect_hands=detect_hands)

    def _extract_keypoints_from_json(self, json_dir, detect_face=False, detect_hands=False):
        """
        Estrae i keypoints dai file JSON generati da OpenPose.
        :param json_dir: Directory contenente i file JSON.
        :return: Lista di keypoints estratti per ogni frame.
        """
        keypoints_list = []

        # Ottieni tutti i file JSON nella directory, ordinati per nome (assumendo che rappresentino l'ordine dei frame)
        json_files = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))

    

        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, "r") as f:
                data = json.load(f)

            # Estrai i keypoints della prima persona (se presente)
            
            keypoints = data["people"][0]["pose_keypoints_2d"]
            if detect_face:
                keypoints.extend(data["people"][0].get("face_keypoints_2d", []))
            if detect_hands:
                keypoints.extend(data["people"][0].get("hand_left_keypoints_2d", []))
                keypoints.extend(data["people"][0].get("hand_right_keypoints_2d", []))
            
            # Aggiungi il frame_id ai keypoints
            keypoints_list.append(keypoints)
            # Incrementa il frame_id per il prossimo frame
           
            
            # Cancella il file JSON dopo l'elaborazione
            os.remove(json_path)

        return keypoints_list

