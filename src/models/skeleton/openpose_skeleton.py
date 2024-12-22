import os
import subprocess

class OpenPoseAPI:
    def __init__(self):
        self.feature_dim = None
    
    def process_video_with_openpose(self, video_path, openpose_path="openpose/bin/OpenPoseDemo.exe", 
                                    output_format="json", detect_face=False, detect_hands=False, 
                                    write_video=False, write_images=False, output_dir="output"):
        """
        Processa un video frame per frame con OpenPose, aggiungendo i parametri facoltativi per il rilevamento viso e mani.
        :param video_path: Percorso del video .mp4.
        :param openpose_path: Percorso del binario OpenPoseDemo.exe.
        :param output_format: Formato di output desiderato ('json', 'video', 'images').
        :param detect_face: Abilita il rilevamento del viso se True.
        :param detect_hands: Abilita il rilevamento delle mani se True.
        :param write_video: Abilita la scrittura del video di output.
        :param write_images: Abilita la scrittura delle immagini di output.
        :param output_dir: Directory di destinazione per i risultati.
        :return: Lista di percorsi dei file elaborati (json, video o immagini).
        """
        # Creare la directory di output se non esiste
        os.makedirs(output_dir, exist_ok=True)

        # Creare la lista dei comandi base
        command = [openpose_path]
        # Se si vogliono usare le immagini invece di un video
        if video_path:
            command.extend(["--video", video_path])

        # Aggiungere il formato di output
        if output_format == "json":
            command.extend(["--write_json", os.path.join(output_dir, "output_jsons")])
        elif output_format == "video":
            command.extend(["--write_video", os.path.join(output_dir, "result.avi")])
        elif output_format == "images":
            command.extend(["--write_images", os.path.join(output_dir, "output_images")])
        
        # Aggiungere i parametri per il rilevamento del viso e delle mani
        if detect_face:
            command.append("--face")
        if detect_hands:
            command.append("--hand")

        # Determinare il numero di feature in base ai parametri di rilevamento
        self.feature_dim = 18  # Numero base di keypoints per OpenPose senza viso né mani
        if detect_face:
            self.feature_dim += 5  # Aggiungi 5 keypoints per il viso
        if detect_hands:
            self.feature_dim += 21  # Aggiungi 21 keypoints per ogni mano (2 mani)

        # Esegui il comando OpenPose
        print(command)
        subprocess.run(command, check=True, shell=True)

        # Restituire i percorsi dei file elaborati in base al formato
        if output_format == "json":
            return [os.path.join(output_dir, "output_jsons")]
        elif output_format == "video":
            return [os.path.join(output_dir, "result.avi")]
        elif output_format == "images":
            return [os.path.join(output_dir, "output_images")]
