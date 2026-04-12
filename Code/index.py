from ultralytics import YOLO
from recognize import FaceRecognizer
from attendance import mark_single_attendance, finalize_attendance
import cv2
from config import YOLO_MODEL_PATH, ENCODINGS_PATH

# Load YOLO face detection model
yolo = YOLO(YOLO_MODEL_PATH)   

def start_attendance(subject):   

    # Load face recognizer
    recognizer = FaceRecognizer(ENCODINGS_PATH)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # One-time attendance per student
    marked_students = set()
       
    print(f"ATTENDANCE STARTED FOR: {subject}")
    print("[CONTROLS]")
    print("  Press 'q' - End and save attendance")
    print("  Press 's' - Take screenshot")
    
    # Attendance webcam loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform face detection
        results = yolo(frame, conf=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue
                
                # Recognize face
                name = recognizer.recognize(face)
                name = str(name)

                # Mark once
                if name != "Unknown" and name not in marked_students:
                    mark_single_attendance(subject, name)
                    marked_students.add(name)
                
                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if name == "Unknown":
                    cv2.putText(frame, "Not Enrolled", (x1, y1 - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Info panel
        cv2.putText(frame, f"Subject: {subject}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Present: {len(marked_students)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to end", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Press 's' to screenshot", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("YOLO Attendance System", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nEnding attendance...")
            break
        elif key == ord('s'):
            filename = f"screenshot_{subject}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    # Finalize attendance
    finalize_attendance(subject, marked_students)

    cap.release()
    cv2.destroyAllWindows()
