import os
import subprocess
import json
import torch 
from pathlib import Path

class OpenPoseAPI:
    def __init__(self, video_path=None, detect_face=False, detect_hands=False, fps=1):
        """
        Inizializza l'API di OpenPose, calcola i keypoints totali in base al video e ai parametri.
        
        :param video_path: Percorso del video da processare.
        :param detect_face: Booleano per abilitare il rilevamento del viso.
        :param detect_hands: Booleano per abilitare il rilevamento delle mani.
        :param fps: Frame per secondo del video.
        """
        # Controlla la durata del video in secondi
        video_duration = 5  # Supponiamo che il video duri 5 secondi
        
        # Determina i keypoints per frame
        self.keypoints_per_frame = 25  # Corpo base (Body-25)
        if detect_hands:
            self.keypoints_per_frame += 21 * 2  # Mani (21 keypoints per mano)
        if detect_face:
            self.keypoints_per_frame += 70  # Viso
        
        # print(f"Keypoints per frame: {keypoints_per_frame}")
        # print(f"FPS: {fps}")
        # Calcola il numero totale di feature
        self.feature_dim = self.keypoints_per_frame * 2 * fps * video_duration 
        
        if video_path is not None:
            self.keypoints = self.process_video_with_openpose(video_path, detect_face=detect_face, detect_hands=detect_hands)
            self.feature_dim = len(self.keypoints)  # Sovrascrivi con la lunghezza effettiva dei keypoints estratti

   

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
        Estrae i keypoints dai file JSON generati da OpenPose senza la confidence.
        :param json_dir: Directory contenente i file JSON.
        :param detect_face: Flag per rilevare i keypoints del viso.
        :param detect_hands: Flag per rilevare i keypoints delle mani.
        :return: Lista di keypoints estratti per ogni frame.
        """
        keypoints_list = []

        # Ottieni tutti i file JSON nella directory, ordinati per nome (assumendo che rappresentino l'ordine dei frame)
        json_files = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))

        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, "r") as f:
                data = json.load(f)

             # Se il vettore `people` è vuoto, aggiungi un vettore di zeri
            if data["people"] == []:
                keypoints = [0] * (self.keypoints_per_frame * 2)
            else:
                # Estrai i keypoints della prima persona (se presente)
                keypoints = data["people"][0]["pose_keypoints_2d"]
                if detect_face:
                    keypoints.extend(data["people"][0].get("face_keypoints_2d", []))
                if detect_hands:
                    keypoints.extend(data["people"][0].get("hand_left_keypoints_2d", []))
                    keypoints.extend(data["people"][0].get("hand_right_keypoints_2d", []))

                # Rimuovi le confidence (prendi solo x e y)
                keypoints = [value for i, value in enumerate(keypoints) if i % 3 != 2]

            # Aggiungi i keypoints elaborati alla lista
            keypoints_list.append(keypoints)
            # Cancella il file JSON dopo l'elaborazione
            os.remove(json_path)


        return keypoints_list


