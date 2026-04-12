import pandas as pd
import os
from datetime import datetime
from config import DATASET_DIR


def get_all_students_from_dataset():
    students = {}

    if not os.path.exists(DATASET_DIR):
        return students
    
    for student_name in os.listdir(DATASET_DIR):
        student_path = os.path.join(DATASET_DIR, student_name)
        
        if not os.path.isdir(student_path):
            continue
        
        # Read SAP-ID from info.txt
        info_file = os.path.join(student_path, "info.txt")
        with open(info_file, 'r') as f:
            sap_id = f.read()
            students[student_name] = sap_id
    
    return students

# Mark Single Attendance
def mark_single_attendance(subject, name):
  
    # Get SAP-ID from folder
    info_file = os.path.join(DATASET_DIR, name, "info.txt")
    sap_id = "Unknown"
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            sap_id = f.read().strip()
           
    
    print(f"{name} ({sap_id})")


def finalize_attendance(subject, present_students):
    # Get current date and time
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    
    # Create file path
    file_path = f"attendance_excel/{subject}.xlsx"
    os.makedirs("attendance_excel", exist_ok=True)
    
    # Get all enrolled students
    all_students = get_all_students_from_dataset()
    
    if not all_students:
        print(f"[WARNING] No students enrolled in {DATASET_DIR}!")
        return
    
    # Create attendance records for ALL students
    rows = []
    for name, sap_id in all_students.items():
        # Check if present or absent
        if name in present_students:
            attendance = 1
            status = "Present"
        else:
            attendance = 0
            status = "Absent"
        
        rows.append({
            "SAP-ID": sap_id,
            "Name": name,
            "Date": date,
            "Time": time,
            "Attendance": attendance
        })
    
    df_new = pd.DataFrame(rows)
    
    try:
        if os.path.exists(file_path):
            df_old = pd.read_excel(file_path, engine='openpyxl')
            
            # Check if already marked today
            if (df_old['Date'] == date).any():
                print(f"Attendance already marked for {subject} today")
                return
            
            # Append new records
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        
        # Save to Excel
        df.to_excel(file_path, index=False, engine='openpyxl')
        
        # Summary
        present_count = len(present_students)
        total_count = len(all_students)
        absent_count = total_count - present_count
        
       
        print(f"ATTENDANCE SAVED")
        
        print(f"Subject: {subject}")
        print(f"Date: {date}")
        print(f"Present: {present_count}/{total_count}")
        print(f"Absent: {absent_count}/{total_count}")
        print(f"File: {file_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save: {e}")
