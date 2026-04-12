import cv2
import os
from ultralytics import YOLO
import time
from config import DATASET_DIR

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


# Enroll Student 
def enroll_student(student_name, images_count=10):
   
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    save_dir = os.path.join(DATASET_DIR, student_name)
    os.makedirs(save_dir, exist_ok=True)

    print("Starting enrollment...")
    count = 0
    
    # Capture images
    while count < images_count:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Object detection
        results = model(frame, conf=0.5, classes=[0])
        
        # Save images with detected faces
        for r in results:
            if len(r.boxes) > 0:
                count += 1
                img_path = os.path.join(save_dir, f"img{count}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"Captured image : {count}")
                time.sleep(0.8)
                break

        cv2.imshow("Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Enrollment completed!")


# Enroll and Encode with Duplicate Checking
def enroll_and_encode(name, sap):
   
    all_students = get_all_students_from_dataset()
 
    if all_students:
        if sap in all_students.values():
            # SAP-ID already exists find which student has it
            existing_student = None
            for student_name, student_sap_id in all_students.items():
                if student_sap_id == sap:
                    existing_student = student_name
                    break
            
            # Print error message
            
            print(f"SAP-ID '{sap}' is already assigned!")
            print(f"This SAP-ID belongs to: {existing_student}")
            print("Please use a different SAP-ID.")
           
            return False  
    
   
    
    
    # Check if SAP-ID is not empty
    if not sap or str(sap).strip() == "":
        print("\n" + "="*50)
        print("[ERROR] SAP-ID cannot be empty!")
        print("="*50 + "\n")
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
        updated_students = get_all_students_from_dataset()
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


