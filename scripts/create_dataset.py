import os
import json
import shutil
import pandas as pd

# Funzione per caricare e analizzare il file events.json
def load_events(patient_folder):
    events_file = os.path.join(patient_folder, 'events.json')
    with open(events_file, 'r') as f:
        events = json.load(f)
    return events

# Funzione per creare la struttura delle cartelle e i CSV
def create_structure(base_path):
    categories = ['position', 'verbal', 'motion', 'environment']
    csv_files = {category: [] for category in categories}

    # Creazione della cartella dataset nella stessa directory di data
    dataset_folder = os.path.join(os.path.dirname(base_path), 'dataset')
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    for category in categories:
        category_folder = os.path.join(dataset_folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
        
        # Creazione del CSV per la categoria
        csv_files[category] = []

    return dataset_folder, csv_files

# Funzione principale
def organize_videos(base_path):
    dataset_folder, csv_files = create_structure(base_path)
    video_id = 0

    # Esplora ogni cartella del paziente
    for patient_id in os.listdir(base_path):
        patient_folder = os.path.join(base_path, patient_id)
        if os.path.isdir(patient_folder):
            events = load_events(patient_folder)

            # Esplora i video per paziente
            annotated_video_folder = os.path.join(patient_folder, 'annotated_video')
            for event in events:
                timestamp = int(event['timestamp'])
                video_filename = f"{timestamp}.mp4"

                # Esplora le cartelle rgb e nir per il video
                for camera_type_folder, camera_type_value in [('rgb', 0), ('nir', 1)]:
                    video_path = os.path.join(annotated_video_folder, camera_type_folder, video_filename)
                    
                    if os.path.exists(video_path):  # Se il video esiste in questa cartella
                        # Determina le categorie in base agli eventi
                        categories = set()
                        if event['position']:
                            categories.add('position')
                        if event['verbal']:
                            categories.add('verbal')
                        if event['motion']:
                            categories.add('motion')
                        if event['environment']:
                            categories.add('environment')

                        # Copia il video nelle categorie corrispondenti
                        for category in categories:
                            category_folder = os.path.join(dataset_folder, category)
                            new_video_filename = f"{video_id}.mp4"
                            new_video_path = os.path.join(category_folder, new_video_filename)

                            # Copia il video nella nuova cartella
                            shutil.copy(video_path, new_video_path)

                            # Aggiungi informazioni nel CSV
                            csv_files[category].append({
                                'id_video': video_id,
                                'id_paziente': patient_id,
                                'evento': category,
                                'camera_type': camera_type_value
                            })

                        video_id += 1

    # Scrivi i CSV per ogni categoria
    for category, data in csv_files.items():
        df = pd.DataFrame(data)
        csv_path = os.path.join(dataset_folder, f"{category}.csv")
        df.to_csv(csv_path, index=False)

# Percorso base
base_path = 'data'
organize_videos(base_path)
