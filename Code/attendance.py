import os
from datetime import datetime

import pandas as pd

from config import DATASET_DIR
from database import get_all_students, save_attendance_session


def get_all_students_from_dataset():
    """Read legacy dataset info.txt files as a fallback student source."""
    students = {}

    if not os.path.exists(DATASET_DIR):
        return students

    for student_name in os.listdir(DATASET_DIR):
        student_path = os.path.join(DATASET_DIR, student_name)
        if not os.path.isdir(student_path):
            continue

        info_file = os.path.join(student_path, "info.txt")
        if not os.path.exists(info_file):
            continue

        with open(info_file, "r") as f:
            students[student_name] = f.readline().strip()

    return students


def get_all_students_for_attendance():
    """Return enrolled students from MySQL, falling back to dataset metadata if needed."""
    database_students = get_all_students()
    if database_students:
        return [
            {"sap_id": student["sap_id"], "name": student["name"]}
            for student in database_students
        ], True

    dataset_students = get_all_students_from_dataset()
    return [
        {"sap_id": sap_id, "name": name}
        for name, sap_id in dataset_students.items()
    ], False


def normalize_present_students(present_students, all_students):
    """Convert a mixed set of names or SAP-IDs into a normalized SAP-ID set."""
    present_values = {str(value).strip() for value in present_students}
    name_to_sap = {student["name"]: student["sap_id"] for student in all_students}
    sap_ids = {student["sap_id"] for student in all_students}

    normalized = set()
    for value in present_values:
        if value in sap_ids:
            normalized.add(value)
        elif value in name_to_sap:
            normalized.add(name_to_sap[value])

    return normalized


# Mark Single Attendance
def mark_single_attendance(subject, name, sap_id=None):
    """Print one real-time attendance confirmation without finalizing the session."""
    display_sap = sap_id or "Unknown"
    print(f"{name} ({display_sap})")


def build_attendance_rows(all_students, present_sap_ids, date, time):
    """Build attendance rows for every enrolled student in one session."""
    rows = []
    for student in all_students:
        status = "Present" if student["sap_id"] in present_sap_ids else "Absent"
        rows.append(
            {
                "sap_id": student["sap_id"],
                "name": student["name"],
                "date": date,
                "time": time,
                "status": status,
                "attendance": 1 if status == "Present" else 0,
            }
        )
    return rows


def save_attendance_excel(file_path, rows):
    """Append one session's attendance rows to the subject Excel workbook."""
    df_new = pd.DataFrame(
        [
            {
                "SAP-ID": row["sap_id"],
                "Name": row["name"],
                "Date": row["date"],
                "Time": row["time"],
                "Attendance": row["attendance"],
                "Status": row["status"],
            }
            for row in rows
        ]
    )

    if os.path.exists(file_path):
        df_old = pd.read_excel(file_path, engine="openpyxl")
        if "Date" in df_old.columns and (df_old["Date"].astype(str) == rows[0]["date"]).any():
            return False
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(file_path, index=False, engine="openpyxl")
    return True


def safe_filename(value):
    """Convert a subject or timestamp string into a Windows-safe filename segment."""
    return "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in value
    )


def save_session_attendance_excel(file_path, rows):
    """Save a session-only attendance workbook for email attachments."""
    df = pd.DataFrame(
        [
            {
                "SAP-ID": row["sap_id"],
                "Name": row["name"],
                "Date": row["date"],
                "Time": row["time"],
                "Attendance": row["attendance"],
                "Status": row["status"],
            }
            for row in rows
        ]
    )
    df.to_excel(file_path, index=False, engine="openpyxl")


def finalize_attendance(subject, present_students):
    """Persist one completed attendance session to MySQL and the Excel workbook."""
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    file_path = f"attendance_excel/{subject}.xlsx"
    session_dir = "attendance_excel/sessions"
    session_file_path = os.path.join(
        session_dir,
        f"{safe_filename(subject)}_{safe_filename(date)}_{safe_filename(time)}.xlsx",
    )
    os.makedirs("attendance_excel", exist_ok=True)
    os.makedirs(session_dir, exist_ok=True)

    all_students, using_database = get_all_students_for_attendance()
    if not all_students:
        print(f"[WARNING] No students enrolled in MySQL or {DATASET_DIR}!")
        return {
            "subject": subject,
            "date": date,
            "time": time,
            "file_path": file_path,
            "session_file_path": session_file_path,
            "present_count": 0,
            "absent_count": 0,
            "total_count": 0,
            "absent_students": [],
            "rows": [],
            "saved": False,
        }

    present_sap_ids = normalize_present_students(present_students, all_students)
    rows = build_attendance_rows(all_students, present_sap_ids, date, time)

    if using_database:
        db_rows = [
            {"sap_id": row["sap_id"], "status": row["status"]}
            for row in rows
        ]
        save_attendance_session(subject, db_rows, date, time)
    else:
        print("[WARNING] Saving Excel only because MySQL students were not available.")

    try:
        excel_saved = save_attendance_excel(file_path, rows)
        save_session_attendance_excel(session_file_path, rows)
        if not excel_saved:
            print(f"Attendance already marked for {subject} today")
    except Exception as e:
        excel_saved = False
        print(f"[ERROR] Failed to save Excel attendance: {e}")

    present_count = len(present_sap_ids)
    total_count = len(all_students)
    absent_students = [
        student["name"]
        for student in all_students
        if student["sap_id"] not in present_sap_ids
    ]
    absent_count = total_count - present_count

    print("ATTENDANCE SAVED")
    print(f"Subject: {subject}")
    print(f"Date: {date}")
    print(f"Present: {present_count}/{total_count}")
    print(f"Absent: {absent_count}/{total_count}")
    print(f"File: {file_path}")

    return {
        "subject": subject,
        "date": date,
        "time": time,
        "file_path": file_path,
        "session_file_path": session_file_path,
        "present_count": present_count,
        "absent_count": absent_count,
        "total_count": total_count,
        "absent_students": absent_students,
        "rows": rows,
        "saved": excel_saved,
    }
