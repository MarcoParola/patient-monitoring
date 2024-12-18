import os
import json
import ffmpeg
import shutil
from datetime import datetime
import tempfile

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

def find_relevant_videos(timestamp, video_folder, video_duration):
    video_files = sorted(os.listdir(video_folder))
    relevant_videos = []

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

            # Controlla se il timestamp cade in questo segmento o nei 3 secondi precedenti o successivi
            if timestamp >= segment_start - 3 and timestamp <= segment_end + 3:
                relevant_videos.append((video, segment_start, segment_end))
        except (ValueError, IndexError):
            print(f"File non conforme: {video}. Ignorato.")
    
    return sorted(relevant_videos, key=lambda x: x[1])

def extract_video_segment(relevant_videos, video_folder, timestamp, output_path):
    def cleanup_temp_files(temp_files):
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Errore durante la rimozione del file temporaneo {temp_file}: {e}")

    if not relevant_videos:
        print(f"Nessun video rilevante trovato per l'estrazione.")
        return False

    temp_files = []
    try:
        if len(relevant_videos) == 1:
            # Se l'intervallo è in un singolo video
            video, segment_start, segment_end = relevant_videos[0]
            video_path = os.path.join(video_folder, video)
            
            ffmpeg.input(video_path, ss=max(0, timestamp - segment_start - 3), t=6) \
                .output(output_path, codec="copy") \
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            print(f"Salvato clip estratto con audio in {output_path}")
            return True

        elif len(relevant_videos) == 2:
            # Se l'intervallo è distribuito su due video
            video1, segment_start1, segment_end1 = relevant_videos[0]
            video2, segment_start2, segment_end2 = relevant_videos[1]
            
            video_path1 = os.path.join(video_folder, video1)
            video_path2 = os.path.join(video_folder, video2)
            
            # Utilizzare tempfile per creare percorsi sicuri
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Preparare i file temporanei per i sotto-segmenti
                temp_clip1 = os.path.join(tmpdirname, "part1.mp4")
                temp_clip2 = os.path.join(tmpdirname, "part2.mp4")
                temp_concat_list = os.path.join(tmpdirname, "concat_list.txt")
                
                # Estrai il segmento dal primo video
                first_segment_duration = segment_end1 - (timestamp - 3)
                ffmpeg.input(video_path1, ss=max(0, timestamp - segment_start1 - 3), t=first_segment_duration) \
                    .output(temp_clip1, codec="copy") \
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                
                # Estrai il segmento dal secondo video
                second_segment_duration = 6 - first_segment_duration
                ffmpeg.input(video_path2, ss=0, t=second_segment_duration) \
                    .output(temp_clip2, codec="copy") \
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                
                # Preparare il file di concatenazione
                with open(temp_concat_list, 'w') as f:
                    f.write(f"file '{temp_clip1}'\nfile '{temp_clip2}'")
                
                # Concatenare i video
                ffmpeg.input(temp_concat_list, format='concat', safe=0) \
                    .output(output_path, codec='copy') \
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            print(f"Salvato clip estratto con audio in {output_path}")
            return True

    except ffmpeg.Error as e:
        print(f"Errore ffmpeg durante l'estrazione: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Errore generico durante l'estrazione: {e}")
        return False

# Processa i file per ciascun id_paziente
def process_patient_videos():
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

        extracted_events = 0
        total_events = len(events)

        for event in events:
            timestamp = event["timestamp"]
            camera_mode = event.get("state_camera_mode", "StateCameraMode.DAY").split(".")[-1].lower()

            # Mappa il valore del camera_mode al nome della cartella
            folder_name = CAMERA_MODE_MAPPING.get(camera_mode)
            if not folder_name:
                print(f"Modalità videocamera sconosciuta: {camera_mode}.")
                continue

            video_folder = os.path.join(raw_video_folder, folder_name)

            if not os.path.exists(video_folder):
                print(f"Cartella video {folder_name} non trovata per il paziente {patient_id}.")
                continue

            relevant_videos = find_relevant_videos(timestamp, video_folder, VIDEO_DURATION)

            if not relevant_videos:
                print(f"Nessun video trovato per l'evento con timestamp {timestamp} ({camera_mode}).")
                continue

            # Calcola i tempi di estrazione
            output_filename = f"{int(timestamp)}.mp4"
            output_path = os.path.join(output_nir if folder_name == "nir" else output_rgb, output_filename)

            # Estrazione usando ffmpeg
            if extract_video_segment(relevant_videos, video_folder, timestamp, output_path):
                extracted_events += 1
            else:
                print(f"Impossibile estrarre il video per l'evento con timestamp {timestamp}")

        print(f"Paziente {patient_id}: Estratti {extracted_events} su {total_events} eventi")

    print("Estrazione Completata")

# Esegui il processo di estrazione
if __name__ == "__main__":
    process_patient_videos()