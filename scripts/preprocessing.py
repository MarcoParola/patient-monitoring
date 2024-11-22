import os
import json
import ffmpeg
from datetime import datetime

# Costanti e percorsi
VIDEO_DURATION = 30  # Durata di ciascun segmento video in secondi
BASE_FOLDER = "./data/"
OUTPUT_SUBFOLDER = "annotated_video"
RAW_VIDEO_SUBFOLDER = "raw_video"
EVENTS_SUBFILE = "events.json"

# Mappatura tra state_camera_mode e cartelle
CAMERA_MODE_MAPPING = {
    "day": "rgb",
    "night": "nir"
}

# Funzione per trovare i file video corrispondenti
def find_relevant_videos(timestamp, video_folder, video_duration):
    video_files = sorted(os.listdir(video_folder))
    relevant_videos = []

    print(f"Accedendo alla cartella: {video_folder}")
    print(f"Video trovati nella cartella: {video_files}")

    for video in video_files:
        if not video.endswith(".mp4"):
            continue

        try:
            # Estrai il timestamp di base e il numero del segmento
            base_time = video.split("_")[0] + "_" + video.split("_")[1]
            segment_index = int(video.split("_")[-1].split(".")[0])
            segment_start = (datetime.strptime(base_time, "%Y-%m-%d_%H-%M-%S").timestamp() 
                            + (segment_index - 1) * video_duration)
            segment_end = segment_start + video_duration

            # Controlla se il timestamp cade in questo segmento o nei 5 secondi precedenti
            if timestamp >= segment_start and timestamp <= segment_end + 5:
                relevant_videos.append((video, segment_start, segment_end))
        except (ValueError, IndexError):
            print(f"File non conforme: {video}. Ignorato.")
    
    return relevant_videos

# Processa i file per ciascun id_paziente
patient_ids = [folder for folder in os.listdir(BASE_FOLDER) if os.path.isdir(os.path.join(BASE_FOLDER, folder))]

for patient_id in patient_ids:
    events_file = os.path.join(BASE_FOLDER, patient_id, EVENTS_SUBFILE)
    raw_video_folder = os.path.join(BASE_FOLDER, patient_id, RAW_VIDEO_SUBFOLDER)
    output_folder = os.path.join(BASE_FOLDER, patient_id, OUTPUT_SUBFOLDER)

    if not os.path.exists(events_file):
        print(f"File eventi non trovato per il paziente {patient_id}.")
        continue

    with open(events_file, "r") as file:
        events = json.load(file)

    # Creazione directory di output per nir e rgb
    output_nir = os.path.join(output_folder, "nir")
    output_rgb = os.path.join(output_folder, "rgb")
    os.makedirs(output_nir, exist_ok=True)
    os.makedirs(output_rgb, exist_ok=True)

    for event in events:
        timestamp = event["timestamp"]
        camera_mode = event.get("state_camera_mode", "StateCameraMode.DAY").split(".")[-1].lower()

        # Mappa il valore del camera_mode al nome della cartella
        folder_name = CAMERA_MODE_MAPPING.get(camera_mode)
        if not folder_name:
            print(f"Modalità videocamera sconosciuta: {camera_mode}.")
            continue

        video_folder = os.path.join(raw_video_folder, folder_name)
        print(f"Controllando la cartella: {video_folder}")

        if not os.path.exists(video_folder):
            print(f"Cartella video {folder_name} non trovata per il paziente {patient_id}.")
            continue

        relevant_videos = find_relevant_videos(timestamp, video_folder, VIDEO_DURATION)

        if not relevant_videos:
            print(f"Nessun video trovato per l'evento con timestamp {timestamp} ({camera_mode}).")
            continue

        # Calcola i tempi di estrazione
        extract_start = timestamp - 5
        extract_end = timestamp
        output_filename = f"{int(timestamp)}.mp4"
        output_path = os.path.join(output_nir if folder_name == "nir" else output_rgb, output_filename)

        # Estrazione usando ffmpeg
        for video, segment_start, segment_end in relevant_videos:
            video_path = os.path.join(video_folder, video)

            try:
                # Usa ffmpeg per tagliare il segmento con audio
                ffmpeg.input(video_path, ss=max(0, extract_start - segment_start), t=extract_end - extract_start) \
                    .output(output_path, codec="copy") \
                    .run(overwrite_output=True)

                print(f"Salvato clip estratto con audio in {output_path}")
            except ffmpeg.Error as e:
                print(f"Errore durante l'estrazione di {video}: {e.stderr.decode()}")
