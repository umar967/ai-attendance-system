import cv2
import os
import time
from config import DATASET_DIR
from database import get_all_students, get_student_by_sap, initialize_database
from yolo_utils import get_face_yolo

# Get all students from dataset
def get_all_students_from_dataset():
    students = {}
    dataset_path = DATASET_DIR

    # Check if dataset folder exists
    if not os.path.exists(dataset_path):
        return students  # Return empty dict if no dataset folder

    # Loop through all folders in dataset
    for name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            info_file = os.path.join(folder_path, "info.txt")

            # Check if info.txt exists
            if os.path.exists(info_file):
                with open(info_file, "r") as f:
                    sap_id = f.readline().strip()  # Read first line and remove whitespace
                    students[name] = sap_id

    return students


def get_all_students_from_database():
    """Return enrolled students from MySQL as a name-to-SAP mapping."""
    students = {}
    for student in get_all_students():
        students[student["name"]] = student["sap_id"]
    return students


# Enroll Student
def enroll_student(student_name, images_count=10):
    model = get_face_yolo()
    cap = cv2.VideoCapture(0)

    save_dir = os.path.join(DATASET_DIR, student_name)
    os.makedirs(save_dir, exist_ok=True)

    print("Starting enrollment...")
    print("Make sure your face is clearly visible and well-lit")
    count = 0
    skipped = 0

    # Capture images
    while count < images_count:
        ret, frame = cap.read()
        if not ret:
            continue

        # Face detection
        results = model(frame, conf=0.5)

        # Save images with detected faces
        for r in results:
            if len(r.boxes) > 0:
                # Get the first face detection
                box = r.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Validate face size (minimum 100x100 pixels for quality)
                face_width = x2 - x1
                face_height = y2 - y1
                if face_width < 100 or face_height < 100:
                    skipped += 1
                    print(f"[SKIP] Face too small ({face_width}x{face_height}), move closer")
                    break

                count += 1
                img_path = os.path.join(save_dir, f"img{count}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"Captured image: {count}/{images_count} ({face_width}x{face_height}px)")
                time.sleep(0.8)
                break

        cv2.imshow("Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Enrollment completed! Captured {count} images (skipped {skipped} low-quality)")


# Enroll and Encode with Duplicate Checking
def enroll_and_encode(name, sap):
    if not initialize_database():
        print("[ERROR] MySQL database is not available.")
        return False
    
    # Check if SAP-ID is not empty
    if not sap or str(sap).strip() == "":
        print("\n" + "="*50)
        print("[ERROR] SAP-ID cannot be empty!")
        print("="*50 + "\n")
        return False

    existing_student = get_student_by_sap(sap)
    if existing_student:
        print(f"SAP-ID '{sap}' is already assigned!")
        print(f"This SAP-ID belongs to: {existing_student['name']}")
        print("Please use a different SAP-ID.")
        return False

    dataset_students = get_all_students_from_dataset()
    if sap in dataset_students.values():
        existing_name = next(
            (
                student_name
                for student_name, student_sap_id in dataset_students.items()
                if student_sap_id == sap
            ),
            "Unknown",
        )
        print(f"SAP-ID '{sap}' is already assigned in the dataset!")
        print(f"This SAP-ID belongs to: {existing_name}")
        print("Please use a different SAP-ID.")
        return False
    
    # Proceed with enrollment
    try:
        
        print(f"Enrolling student: {name}")
        print(f"SAP-ID: {sap}")
       
        
        # Perform enrollment (captures images)
        enroll_student(name)

        # Save student information to info.txt
        save_dir = os.path.join(DATASET_DIR, name)
        
        with open(os.path.join(save_dir, "info.txt"), "w") as f:
            f.write(f"{sap}\n")
        
        print(f"\n[SUCCESS] Student information saved!")

        # Run encoding (import here to avoid circular import at module load)
        print("\nEncoding faces...")
        from encode_faces import encode_all_faces
        encode_all_faces()
        
        # Final success message
        
        print("Student enrolled successfully!")
        print(f"Name: {name}")
        print(f"SAP-ID: {sap}")
        
        # Count total students
        updated_students = get_all_students_from_database()
        print(f"Total students enrolled: {len(updated_students)}")
        
        
        return True 
        
    except Exception as e:
        # Handle any errors during enrollment
        print("\n" + "="*50)
        print(f"[ERROR] Failed to enroll student: {e}")
        print("="*50 + "\n")
        
        # Clean up partially created folder if error occurs
        save_dir = os.path.join(DATASET_DIR, name)
        if os.path.exists(save_dir):
            import shutil
            try:
                shutil.rmtree(save_dir)
                print(f"[INFO] Cleaned up incomplete enrollment data.")
            except:
                pass
        
        return False


