import os
import json
from collections import Counter
from statistics import mean

# Base folder for data
BASE_FOLDER = "./data/"

# Function to load JSON files
def load_json(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Function to calculate statistics
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
        "skin_color_counts": Counter(),
        "gender_counts": Counter(),
        "age_distribution": [],
    }
    
    # List of patient directories
    patient_ids = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    stats["total_patients"] = len(patient_ids)
    
    for patient_id in patient_ids:
        session_file = os.path.join(base_folder, patient_id, "session.json")
        events_file = os.path.join(base_folder, patient_id, "events.json")
        
        # Read session.json
        session_data = load_json(session_file)
        if session_data:
            # Add average glasgow score
            if session_data.get("glasgow"):
                stats["average_glasgow"].append(mean(session_data["glasgow"]))
            stats["room_type_counts"][session_data.get("room_type", "Unknown")] += 1
            stats["pathology_counts"][session_data.get("pathology", "Unknown")] += 1
            
            # Add skin color to statistics
            skin_color = session_data.get("skin_color", "Unknown")
            stats["skin_color_counts"][skin_color] += 1
            
            # Add gender to statistics
            gender = session_data.get("gender", "Unknown")
            stats["gender_counts"][gender] += 1
            
            # Add age to statistics
            age = session_data.get("age")
            if age is not None:
                stats["age_distribution"].append(age)
        
        # Read events.json
        events_data = load_json(events_file)
        if events_data:
            stats["total_events"] += len(events_data)
            
            for event in events_data:
                # Count camera modes
                camera_mode = event.get("state_camera_mode", "StateCameraMode.DAY").split(".")[-1]
                stats["camera_mode_counts"][camera_mode] += 1
                
                # Count body parts
                for position in event.get("position", []):
                    body_part = position.get("body_part", "Unknown")
                    stats["body_part_counts"][body_part] += 1
                
                # Count motion, verbal, and environmental events
                if event.get("motion"):
                    stats["motion_event_count"] += 1
                if event.get("verbal"):
                    stats["verbal_event_count"] += 1
                if event.get("environment"):
                    stats["environment_event_count"] += 1
    
    # Calculate the average Glasgow score
    stats["average_glasgow"] = mean(stats["average_glasgow"]) if stats["average_glasgow"] else 0
    
    return stats

# Execution and report printing
if __name__ == "__main__":
    statistics = calculate_statistics(BASE_FOLDER)
    
    print("=== Statistics ===")
    print(f"Total number of patients: {statistics['total_patients']}")
    print(f"Total number of events: {statistics['total_events']}")
    print(f"Camera mode frequency: {dict(statistics['camera_mode_counts'])}")
    print(f"Body part frequency: {dict(statistics['body_part_counts'])}")
    print(f"Average Glasgow score across all patients: {statistics['average_glasgow']:.2f}")
    print(f"Room type distribution: {dict(statistics['room_type_counts'])}")
    print(f"Pathology distribution: {dict(statistics['pathology_counts'])}")
    print(f"Motion event count: {statistics['motion_event_count']}")
    print(f"Verbal event count: {statistics['verbal_event_count']}")
    print(f"Environmental event count: {statistics['environment_event_count']}")
    print(f"Skin color distribution: {dict(statistics['skin_color_counts'])}")
    print(f"Gender distribution: {dict(statistics['gender_counts'])}")
    print(f"Age distribution: {statistics['age_distribution']}")
