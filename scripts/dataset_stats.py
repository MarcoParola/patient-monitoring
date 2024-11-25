from datetime import datetime
import os
import json
from collections import Counter
from statistics import mean

# Percorso base dei dati
BASE_FOLDER = "./data/"

# Funzione per leggere i file JSON
def load_json(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Errore nel caricamento del file {file_path}: {e}")
        return None

# Funzione per calcolare le statistiche
def calculate_statistics(base_folder):
    stats = {
        "total_patients": 0,
        "total_events": 0,
        "camera_mode_counts": Counter(),
        "body_part_counts": Counter(),
        "average_glasgow": [],
        "room_type_counts": Counter(),
        "motion_event_count": 0,
        "verbal_event_count": 0,
        "environment_event_count": 0,
        "pathology_counts": Counter(),
    }
    
    # Elenco dei pazienti
    patient_ids = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    stats["total_patients"] = len(patient_ids)
    
    for patient_id in patient_ids:
        session_file = os.path.join(base_folder, patient_id, "session.json")
        events_file = os.path.join(base_folder, patient_id, "events.json")
        
        # Leggi il file session.json
        session_data = load_json(session_file)
        if session_data:
            stats["average_glasgow"].append(mean(session_data.get("glasgow", [])))
            stats["room_type_counts"][session_data.get("room_type", "Unknown")] += 1
            stats["pathology_counts"][session_data.get("pathology", "Unknown")] += 1
        
        # Leggi il file events.json
        events_data = load_json(events_file)
        if events_data:
            stats["total_events"] += len(events_data)
            
            for event in events_data:
                # Conta i tipi di camera
                camera_mode = event.get("state_camera_mode", "StateCameraMode.DAY").split(".")[-1]
                stats["camera_mode_counts"][camera_mode] += 1
                
                # Conta i body part
                for position in event.get("position", []):
                    body_part = position.get("body_part", "Unknown")
                    stats["body_part_counts"][body_part] += 1
                
                # Conta eventi con movimento, verbali e ambientali
                if event.get("motion"):
                    stats["motion_event_count"] += 1
                if event.get("verbal"):
                    stats["verbal_event_count"] += 1
                if event.get("environment"):
                    stats["environment_event_count"] += 1
    
    # Calcola la media dei Glasgow
    stats["average_glasgow"] = mean(stats["average_glasgow"]) if stats["average_glasgow"] else 0
    
    return stats

# Esecuzione e stampa del report
if __name__ == "__main__":
    statistics = calculate_statistics(BASE_FOLDER)
    
    print("=== Statistiche ===")
    print(f"Numero totale di pazienti: {statistics['total_patients']}")
    print(f"Numero totale di eventi: {statistics['total_events']}")
    print(f"Frequenza modalità camera: {dict(statistics['camera_mode_counts'])}")
    print(f"Frequenza parti del corpo: {dict(statistics['body_part_counts'])}")
    print(f"Media Glasgow tra tutti i pazienti: {statistics['average_glasgow']:.2f}")
    print(f"Distribuzione tipo di stanza: {dict(statistics['room_type_counts'])}")
    print(f"Distribuzione delle patologie: {dict(statistics['pathology_counts'])}")
    print(f"Numero eventi con movimento: {statistics['motion_event_count']}")
    print(f"Numero eventi verbali: {statistics['verbal_event_count']}")
    print(f"Numero eventi ambientali: {statistics['environment_event_count']}")
