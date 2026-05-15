from datetime import datetime

import numpy as np

from config import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    mysql = None
    Error = Exception


FACE_ENCODING_SIZE = 128
ENCODING_DTYPE = np.float64
_DATABASE_READY = False


def initialize_database():
    """Create the attendai database and all tables required by the application."""
    global _DATABASE_READY
    if _DATABASE_READY:
        return True

    if mysql is None:
        print("[ERROR] mysql-connector-python is not installed.")
        return False

    try:
        server_connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        cursor = server_connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
        cursor.close()
        server_connection.close()

        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                sap_id VARCHAR(64) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                encoding LONGBLOB NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sap_id VARCHAR(64) NOT NULL,
                subject VARCHAR(255) NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                status VARCHAR(20) NOT NULL,
                INDEX idx_attendance_subject_date (subject, date),
                CONSTRAINT fk_attendance_student
                    FOREIGN KEY (sap_id) REFERENCES students(sap_id)
                    ON UPDATE CASCADE ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS phone_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL,
                time TIME NOT NULL,
                INDEX idx_phone_logs_datetime (date, time)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attention_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sap_id VARCHAR(64) NOT NULL,
                date DATE NOT NULL,
                subject VARCHAR(255) NOT NULL,
                drowsy_count INT NOT NULL DEFAULT 0,
                inattentive_count INT NOT NULL DEFAULT 0,
                aggressive_count INT NOT NULL DEFAULT 0,
                INDEX idx_attention_subject_date (subject, date),
                CONSTRAINT fk_attention_student
                    FOREIGN KEY (sap_id) REFERENCES students(sap_id)
                    ON UPDATE CASCADE ON DELETE CASCADE
            )
            """
        )
        ensure_attention_aggressive_column(cursor)
        connection.commit()
        cursor.close()
        connection.close()
        _DATABASE_READY = True
        return True
    except Error as error:
        print(f"[ERROR] Database initialization failed: {error}")
        return False


def ensure_attention_aggressive_column(cursor):
    """Add aggressive_count to older attention_logs tables when missing."""
    try:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'attention_logs'
              AND COLUMN_NAME = 'aggressive_count'
            """,
            (DB_NAME,),
        )
        column_exists = cursor.fetchone()[0] > 0
        if not column_exists:
            cursor.execute(
                """
                ALTER TABLE attention_logs
                ADD COLUMN aggressive_count INT NOT NULL DEFAULT 0
                """
            )
    except Error as error:
        print(f"[WARNING] Could not update attention_logs schema: {error}")


def get_connection():
    """Open a MySQL connection to the configured attendai database."""
    if mysql is None:
        raise RuntimeError("mysql-connector-python is not installed.")

    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )


def encode_numpy_array(encoding_array):
    """Convert one or more face encodings into bytes for MySQL LONGBLOB storage."""
    encodings = np.asarray(encoding_array, dtype=ENCODING_DTYPE)
    if encodings.ndim == 1:
        encodings = encodings.reshape(1, -1)

    if encodings.ndim != 2 or encodings.shape[1] != FACE_ENCODING_SIZE:
        raise ValueError("Face encodings must have shape (128,) or (n, 128).")

    return encodings.tobytes()


def decode_numpy_array(encoding_blob):
    """Convert a MySQL LONGBLOB back into one or more numpy face encodings."""
    if not encoding_blob:
        return np.empty((0, FACE_ENCODING_SIZE), dtype=ENCODING_DTYPE)

    encodings = np.frombuffer(encoding_blob, dtype=ENCODING_DTYPE)
    if encodings.size % FACE_ENCODING_SIZE != 0:
        raise ValueError("Stored encoding blob has an invalid size.")

    return encodings.reshape((-1, FACE_ENCODING_SIZE))


def save_student_encoding(sap_id, name, encoding_array):
    """Insert or update one student's name and face encoding bytes in MySQL."""
    try:
        initialize_database()
        encoding_blob = encode_numpy_array(encoding_array)
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO students (sap_id, name, encoding)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                encoding = VALUES(encoding)
            """,
            (str(sap_id).strip(), str(name).strip(), encoding_blob),
        )
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except (Error, RuntimeError, ValueError) as error:
        print(f"[ERROR] Failed to save student encoding: {error}")
        return False


def get_student_by_sap(sap_id):
    """Return one student row by SAP-ID, or None when it does not exist."""
    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT sap_id, name, encoding FROM students WHERE sap_id = %s",
            (str(sap_id).strip(),),
        )
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        return row
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to read student: {error}")
        return None


def get_all_students():
    """Read all enrolled students from MySQL ordered by student name."""
    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT sap_id, name, encoding FROM students ORDER BY name")
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return rows
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to read students: {error}")
        return []


def load_face_encodings():
    """Load all stored student encodings from MySQL for face recognition."""
    known_encodings = []
    known_names = []
    known_sap_ids = []

    for student in get_all_students():
        try:
            encodings = decode_numpy_array(student["encoding"])
        except ValueError as error:
            print(f"[SKIP] Invalid encoding for {student['name']}: {error}")
            continue

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(student["name"])
            known_sap_ids.append(student["sap_id"])

    return known_encodings, known_names, known_sap_ids


def attendance_exists_for_subject_date(subject, attendance_date):
    """Check whether a subject already has attendance rows for a date."""
    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM attendance WHERE subject = %s AND date = %s",
            (subject, attendance_date),
        )
        count = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return count > 0
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to check attendance date: {error}")
        return False


def save_attendance_session(subject, attendance_rows, attendance_date, attendance_time):
    """Save present and absent rows for one completed attendance session."""
    if attendance_exists_for_subject_date(subject, attendance_date):
        return False

    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor()
        cursor.executemany(
            """
            INSERT INTO attendance (sap_id, subject, date, time, status)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (
                    row["sap_id"],
                    subject,
                    attendance_date,
                    attendance_time,
                    row["status"],
                )
                for row in attendance_rows
            ],
        )
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to save attendance session: {error}")
        return False


def get_total_students():
    """Return the total number of enrolled students stored in MySQL."""
    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        total = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return total
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to count students: {error}")
        return 0


def get_low_attendance_students(threshold=75, subject=None):
    """Return students whose cumulative attendance percentage is below a threshold."""
    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        subject_filter = ""
        params = []
        if subject:
            subject_filter = "AND a.subject = %s"
            params.append(subject)

        cursor.execute(
            f"""
            SELECT
                s.sap_id,
                s.name,
                COUNT(a.id) AS total_classes,
                COALESCE(SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END), 0)
                    AS present_classes
            FROM students s
            LEFT JOIN attendance a
                ON a.sap_id = s.sap_id
                {subject_filter}
            GROUP BY s.sap_id, s.name
            ORDER BY s.name
            """,
            tuple(params),
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        at_risk = []
        for row in rows:
            total_classes = int(row["total_classes"] or 0)
            present_classes = int(row["present_classes"] or 0)
            percentage = (
                (present_classes / total_classes) * 100 if total_classes else 0
            )
            if percentage < threshold:
                at_risk.append(
                    {
                        "sap_id": row["sap_id"],
                        "name": row["name"],
                        "percentage": round(percentage, 2),
                    }
                )

        return at_risk
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to read low attendance students: {error}")
        return []


def get_attendance_percentages_by_subject(subject):
    """Return attendance percentages per student for one selected subject."""
    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT
                s.sap_id,
                s.name,
                COUNT(a.id) AS total_classes,
                COALESCE(SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END), 0)
                    AS present_classes
            FROM students s
            LEFT JOIN attendance a
                ON a.sap_id = s.sap_id
                AND a.subject = %s
            GROUP BY s.sap_id, s.name
            ORDER BY s.name
            """,
            (subject,),
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        percentages = []
        for row in rows:
            total_classes = int(row["total_classes"] or 0)
            present_classes = int(row["present_classes"] or 0)
            percentage = (
                (present_classes / total_classes) * 100 if total_classes else 0
            )
            percentages.append(
                {
                    "sap_id": row["sap_id"],
                    "name": row["name"],
                    "percentage": round(percentage, 2),
                    "total_classes": total_classes,
                    "present_classes": present_classes,
                }
            )

        return percentages
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to read attendance analytics: {error}")
        return []


def get_daily_attendance_trend(subject):
    """Return daily class attendance percentage trend for one subject."""
    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT
                date,
                COUNT(*) AS total_count,
                COALESCE(SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END), 0)
                    AS present_count
            FROM attendance
            WHERE subject = %s
            GROUP BY date
            ORDER BY date
            """,
            (subject,),
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        trend = []
        for row in rows:
            total_count = int(row["total_count"] or 0)
            present_count = int(row["present_count"] or 0)
            percentage = (present_count / total_count) * 100 if total_count else 0
            trend.append(
                {
                    "date": str(row["date"]),
                    "percentage": round(percentage, 2),
                    "present_count": present_count,
                    "total_count": total_count,
                }
            )

        return trend
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to read attendance trend: {error}")
        return []


def log_phone_detection(event_date=None, event_time=None):
    """Insert one phone detection event into the phone_logs table."""
    now = datetime.now()
    event_date = event_date or now.strftime("%Y-%m-%d")
    event_time = event_time or now.strftime("%H:%M:%S")

    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO phone_logs (date, time) VALUES (%s, %s)",
            (event_date, event_time),
        )
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to log phone detection: {error}")
        return False


def count_phone_logs_between(start_datetime, end_datetime):
    """Count phone detections logged between two datetime values."""
    start_datetime = start_datetime.replace(microsecond=0)
    end_datetime = end_datetime.replace(microsecond=0)

    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM phone_logs
            WHERE TIMESTAMP(date, time) BETWEEN %s AND %s
            """,
            (start_datetime, end_datetime),
        )
        count = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return int(count)
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to count phone logs: {error}")
        return 0


def save_attention_log(
    sap_id,
    attendance_date,
    subject,
    drowsy_count,
    inattentive_count,
    aggressive_count=0,
):
    """Store one student's drowsiness and inattention counts for a class session."""
    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO attention_logs
                (
                    sap_id,
                    date,
                    subject,
                    drowsy_count,
                    inattentive_count,
                    aggressive_count
                )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                sap_id,
                attendance_date,
                subject,
                int(drowsy_count),
                int(inattentive_count),
                int(aggressive_count),
            ),
        )
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to save attention log: {error}")
        return False


def get_attention_averages(subject, attendance_date=None):
    """Return average drowsiness and inattention counts for a class session."""
    try:
        initialize_database()
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        params = [subject]
        date_filter = ""
        if attendance_date:
            date_filter = "AND date = %s"
            params.append(attendance_date)

        cursor.execute(
            f"""
            SELECT
                COALESCE(AVG(drowsy_count), 0) AS avg_drowsy_count,
                COALESCE(AVG(inattentive_count), 0) AS avg_inattentive_count,
                COALESCE(AVG(aggressive_count), 0) AS avg_aggressive_count
            FROM attention_logs
            WHERE subject = %s
            {date_filter}
            """,
            tuple(params),
        )
        row = cursor.fetchone() or {}
        cursor.close()
        connection.close()
        return {
            "avg_drowsy_count": round(float(row.get("avg_drowsy_count") or 0), 2),
            "avg_inattentive_count": round(
                float(row.get("avg_inattentive_count") or 0), 2
            ),
            "avg_aggressive_count": round(
                float(row.get("avg_aggressive_count") or 0), 2
            ),
        }
    except (Error, RuntimeError) as error:
        print(f"[ERROR] Failed to read attention averages: {error}")
        return {
            "avg_drowsy_count": 0,
            "avg_inattentive_count": 0,
            "avg_aggressive_count": 0,
        }


if __name__ == "__main__":
    if initialize_database():
        print(f"Database '{DB_NAME}' is ready.")
    else:
        print(f"Database '{DB_NAME}' could not be initialized.")
