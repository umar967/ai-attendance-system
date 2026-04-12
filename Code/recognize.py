import cv2
import face_recognition
import pickle
from config import FACE_MATCH_TOLERANCE

class FaceRecognizer:
    # Initialize the recognizer with known encodings
    def __init__(self, encodings_file):        
        try:
            with open(encodings_file, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
            print(f"Loaded {len(self.known_encodings)} face encodings")
        except:
            print("Could not load encodings.pickle !")
            self.known_encodings = []
            self.known_names = []
    
    # Recognize a face from an image
    def recognize(self, face_image):
        if not self.known_encodings:
            return "Unknown"

        # Convert BGR to RGB
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        boxes = face_recognition.face_locations(rgb_face, model="hog")
        encodings = face_recognition.face_encodings(rgb_face, boxes)
        
        # If no face found
        if len(encodings) == 0:
            return "Unknown"
        
        # Get first face encoding
        face_encoding = encodings[0]
        
        face_distances = face_recognition.face_distance(
            self.known_encodings,
            face_encoding
        )
        best_match_index = face_distances.argmin()
        best_distance = face_distances[best_match_index]

        if best_distance <= FACE_MATCH_TOLERANCE:
            return self.known_names[best_match_index]

        return "Unknown"

if __name__ == "__main__":
    # Test the recognizer
    recognizer = FaceRecognizer("encodings.pickle")
    print(f"Ready to recognize {len(recognizer.known_names)} students")
