import os
import cv2
import face_recognition
import pickle
from config import DATASET_DIR, ENCODINGS_PATH

ENCODINGS_FILE = ENCODINGS_PATH

def encode_all_faces():

    known_encodings = []
    known_names = []
    
   
    
    print("Encoding faces...")
    
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset folder not found: {DATASET_DIR}")
        return

    # Loop through each student folder
    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        
        if not os.path.isdir(person_path):
            continue
        
        print(f"Processing : {person_name}")
        
        # Loop through each image
        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(person_path, img_name)
            
            # Read and convert image
            image = cv2.imread(img_path)
            if image is None:
                continue
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face and create encoding
            boxes = face_recognition.face_locations(rgb_image, model="hog")
            encodings = face_recognition.face_encodings(rgb_image, boxes)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
    
    # Save encodings to file
    if len(known_encodings) > 0:
        data = {"encodings": known_encodings, "names": known_names}
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"Successfully Encoded {len(known_encodings)} faces")
    else:
        print("[ERROR] No faces found")

if __name__ == "__main__":
    encode_all_faces()
