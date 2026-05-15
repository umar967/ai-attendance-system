import os
from datetime import datetime

import cv2
from ultralytics import YOLO

from attendance import finalize_attendance, mark_single_attendance
from attention import AttentionTracker
from config import (
    AGGRESSIVE_MOVEMENT_THRESHOLD,
    AGGRESSIVE_CONSEC_FRAMES,
    AGGRESSIVE_POSE_DELTA_THRESHOLD,
    ATTENTION_BASELINE_FRAMES,
    ATTENTION_CONSEC_FRAMES,
    COCO_MODEL_PATH,
    FACE_CROP_PADDING,
    HEAD_POSE_PITCH_THRESHOLD,
    HEAD_POSE_YAW_THRESHOLD,
    MONITOR_CROPS_DIR,
    PHONE_CLASS_ID,
    PHONE_MISSING_RESET_FRAMES,
)
from database import initialize_database, log_phone_detection, save_attention_log
from recognize import FaceRecognizer
from reporting import email_session_report, generate_session_report
from yolo_utils import get_face_yolo


yolo = get_face_yolo()
coco_yolo = None


def get_coco_model():
    """Lazily load the COCO YOLOv8n model used for mobile phone detection."""
    global coco_yolo
    if coco_yolo is None:
        coco_yolo = YOLO(COCO_MODEL_PATH)
    return coco_yolo


def clamp_box(box, frame_width, frame_height):
    """Clamp a YOLO bounding box so it stays inside the current frame."""
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))
    return x1, y1, x2, y2


def padded_box(box, frame_width, frame_height, padding_ratio=FACE_CROP_PADDING):
    """Expand a face crop slightly to improve recognition from turned faces."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    return clamp_box(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        frame_width,
        frame_height,
    )


def safe_name(value):
    """Create a filesystem-safe name segment for incident crops."""
    return "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in str(value)
    )


def save_incident_crop(frame, box, prefix, identifier):
    """Save a phone or behavior crop from the monitor feed for review."""
    os.makedirs(MONITOR_CROPS_DIR, exist_ok=True)
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{safe_name(prefix)}_{safe_name(identifier)}_{timestamp}.jpg"
    path = os.path.join(MONITOR_CROPS_DIR, filename)
    cv2.imwrite(path, crop)
    return path


class PhoneEventCounter:
    def __init__(self):
        """Count one phone event per visible phone session instead of every frame."""
        self.count = 0
        self.phone_visible = False
        self.missing_frames = PHONE_MISSING_RESET_FRAMES

    def update(self, frame, phone_model):
        """Detect phones, draw boxes, save crops, and log only new phone appearances."""
        results = phone_model(frame, conf=0.4, classes=[PHONE_CLASS_ID], verbose=False)
        frame_height, frame_width = frame.shape[:2]
        phone_boxes = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = clamp_box(box.xyxy[0], frame_width, frame_height)
                if x2 > x1 and y2 > y1:
                    phone_boxes.append((x1, y1, x2, y2))

        if phone_boxes:
            if not self.phone_visible:
                self.count += 1
                log_phone_detection()
                save_incident_crop(frame, phone_boxes[0], "phone", self.count)
            self.phone_visible = True
            self.missing_frames = 0
        else:
            self.missing_frames += 1
            if self.missing_frames >= PHONE_MISSING_RESET_FRAMES:
                self.phone_visible = False

        for box in phone_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "Phone Detected",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return self.count


def update_attention_counts(attention_counts, sap_id, attention_result):
    """Increment behavior counters for one recognized student."""
    if not sap_id or not attention_result:
        return

    attention_counts.setdefault(
        sap_id,
        {"drowsy": 0, "inattentive": 0, "aggressive": 0},
    )
    if attention_result.get("drowsy"):
        attention_counts[sap_id]["drowsy"] += 1
    if attention_result.get("distracted"):
        attention_counts[sap_id]["inattentive"] += 1
    if attention_result.get("aggressive"):
        attention_counts[sap_id]["aggressive"] += 1


def detect_aggressive_movement(sap_id, box, attention_result, behavior_state):
    """Flag only repeated, extreme movement as possible aggressive activity."""
    if not sap_id:
        return False

    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    current = {
        "center": center,
        "yaw": attention_result.get("yaw", 0) if attention_result else 0,
        "pitch": attention_result.get("pitch", 0) if attention_result else 0,
    }
    state = behavior_state.setdefault(sap_id, {})
    previous = state.get("motion")
    state["motion"] = current

    if not previous:
        state["aggressive_frames"] = 0
        return False

    movement = (
        (center[0] - previous["center"][0]) ** 2
        + (center[1] - previous["center"][1]) ** 2
    ) ** 0.5
    yaw_delta = abs(current["yaw"] - previous["yaw"])
    pitch_delta = abs(current["pitch"] - previous["pitch"])
    extreme_motion = (
        movement > AGGRESSIVE_MOVEMENT_THRESHOLD
        and (
            movement > AGGRESSIVE_MOVEMENT_THRESHOLD * 1.35
            or yaw_delta > AGGRESSIVE_POSE_DELTA_THRESHOLD
            or pitch_delta > AGGRESSIVE_POSE_DELTA_THRESHOLD
        )
    )
    if extreme_motion:
        state["aggressive_frames"] = state.get("aggressive_frames", 0) + 1
    else:
        state["aggressive_frames"] = max(state.get("aggressive_frames", 0) - 2, 0)

    return state["aggressive_frames"] >= AGGRESSIVE_CONSEC_FRAMES


def stabilize_attention_result(identity, attention_result, behavior_state):
    """Calibrate neutral head pose and require sustained distraction."""
    if not identity or not attention_result:
        return attention_result
    if attention_result.get("status") == "Attention unavailable":
        return attention_result

    state = behavior_state.setdefault(identity, {})
    attention_state = state.setdefault(
        "attention",
        {
            "baseline_frames": 0,
            "neutral_pitch": attention_result.get("pitch", 0),
            "neutral_yaw": attention_result.get("yaw", 0),
            "distracted_frames": 0,
        },
    )

    pitch = attention_result.get("pitch", 0)
    yaw = attention_result.get("yaw", 0)
    baseline_frames = attention_state["baseline_frames"]

    if baseline_frames < ATTENTION_BASELINE_FRAMES:
        next_count = baseline_frames + 1
        weight = 1 / next_count
        attention_state["neutral_pitch"] = (
            attention_state["neutral_pitch"] * (1 - weight)
        ) + (pitch * weight)
        attention_state["neutral_yaw"] = (
            attention_state["neutral_yaw"] * (1 - weight)
        ) + (yaw * weight)
        attention_state["baseline_frames"] = next_count
        pose_distracted = False
    else:
        pitch_delta = pitch - attention_state["neutral_pitch"]
        yaw_delta = yaw - attention_state["neutral_yaw"]
        pose_distracted = (
            abs(yaw_delta) > HEAD_POSE_YAW_THRESHOLD
            or abs(pitch_delta) > HEAD_POSE_PITCH_THRESHOLD
        )
        attention_result["pitch_delta"] = round(pitch_delta, 2)
        attention_result["yaw_delta"] = round(yaw_delta, 2)

        if not pose_distracted and not attention_result.get("drowsy"):
            attention_state["neutral_pitch"] = (
                attention_state["neutral_pitch"] * 0.98
            ) + (pitch * 0.02)
            attention_state["neutral_yaw"] = (
                attention_state["neutral_yaw"] * 0.98
            ) + (yaw * 0.02)

    if pose_distracted:
        attention_state["distracted_frames"] += 1
    else:
        attention_state["distracted_frames"] = max(
            attention_state["distracted_frames"] - 2,
            0,
        )

    attention_result["distracted"] = (
        attention_state["distracted_frames"] >= ATTENTION_CONSEC_FRAMES
    )
    if attention_result.get("drowsy"):
        attention_result["status"] = "Drowsy"
    elif attention_result["distracted"]:
        attention_result["status"] = "Distracted"
    else:
        attention_result["status"] = "Attentive"

    return attention_result


def draw_attendance_overlay(frame, box, recognition):
    """Draw attendance recognition with blink verification instructions."""
    x1, y1, x2, y2 = box
    name = recognition["name"]
    sap_id = recognition["sap_id"]
    live = recognition["live"]
    blink_waiting = recognition.get("blink_waiting", False)

    if not sap_id:
        color = (0, 0, 255)
        state_text = "Not Enrolled"
    elif live:
        color = (0, 255, 0)
        state_text = "Verified"
    elif blink_waiting:
        color = (0, 165, 255)
        state_text = "Open eyes"
    else:
        color = (0, 165, 255)
        state_text = "Blink to verify"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, name, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, state_text, (x1, max(20, y1 - 35)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_monitor_issue_overlay(frame, box, name, status):
    """Draw a monitor box only when a student is drowsy, inattentive, or aggressive."""
    x1, y1, x2, y2 = box
    colors = {
        "Drowsy": (0, 165, 255),
        "Distracted": (255, 180, 0),
        "Aggressive movement": (0, 0, 255),
    }
    color = colors.get(status, (0, 0, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{name}: {status}", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_info_panel(frame, subject, mode, present_count=0, phone_count=0):
    """Draw controls and counters on the camera feed."""
    cv2.putText(frame, f"Subject: {subject}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y = 60
    if mode == "attendance":
        cv2.putText(frame, f"Present: {present_count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30
    else:
        cv2.putText(frame, f"Phone Events: {phone_count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 30

    cv2.putText(frame, "Press 'q' to quit", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Press 's' for screenshot", (10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def process_attendance_faces(frame, recognizer, subject, marked_students):
    """Recognize students, require one quick blink, and mark attendance once."""
    results = yolo(frame, conf=0.5, verbose=False)
    frame_height, frame_width = frame.shape[:2]

    for result in results:
        for box in result.boxes:
            face_box = clamp_box(box.xyxy[0], frame_width, frame_height)
            x1, y1, x2, y2 = padded_box(face_box, frame_width, frame_height)
            if x2 <= x1 or y2 <= y1:
                continue

            face = frame[y1:y2, x1:x2]
            if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
                continue

            recognition = recognizer.recognize_with_details(face, check_liveness=True)
            if (
                recognition["sap_id"]
                and recognition["live"]
                and recognition["sap_id"] not in marked_students
            ):
                mark_single_attendance(subject, recognition["name"], recognition["sap_id"])
                marked_students.add(recognition["sap_id"])

            draw_attendance_overlay(frame, face_box, recognition)


def process_monitor_faces(
    frame,
    recognizer,
    attention_tracker,
    attention_counts,
    behavior_state,
    frame_index,
):
    """Monitor behavior only; never mark attendance from the class monitor."""
    results = yolo(frame, conf=0.5, verbose=False)
    frame_height, frame_width = frame.shape[:2]

    for result in results:
        for box in result.boxes:
            face_box = clamp_box(box.xyxy[0], frame_width, frame_height)
            x1, y1, x2, y2 = padded_box(face_box, frame_width, frame_height)
            if x2 <= x1 or y2 <= y1:
                continue

            face = frame[y1:y2, x1:x2]
            if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
                continue

            recognition = recognizer.recognize_with_details(face, check_liveness=False)
            sap_id = recognition["sap_id"]
            name = recognition["name"]
            attention_result = attention_tracker.analyze(face)
            attention_result = stabilize_attention_result(
                sap_id or name,
                attention_result,
                behavior_state,
            )
            aggressive = detect_aggressive_movement(
                sap_id,
                face_box,
                attention_result,
                behavior_state,
            )
            attention_result["aggressive"] = aggressive

            if aggressive:
                status = "Aggressive movement"
            else:
                status = attention_result["status"]

            if sap_id:
                update_attention_counts(attention_counts, sap_id, attention_result)

            issue_statuses = {"Drowsy", "Distracted", "Aggressive movement"}
            if status in issue_statuses:
                draw_monitor_issue_overlay(frame, face_box, name, status)
                state = behavior_state.setdefault(sap_id or name, {})
                last_crop_frame = state.get("last_crop_frame", -9999)
                last_status = state.get("last_status")
                if last_status != status or frame_index - last_crop_frame > 60:
                    save_incident_crop(frame, face_box, status, sap_id or name)
                    state["last_crop_frame"] = frame_index
                    state["last_status"] = status


def save_attention_counts(subject, attendance_date, attention_counts):
    """Persist per-student behavior counters collected during a monitor session."""
    for sap_id, counts in attention_counts.items():
        save_attention_log(
            sap_id,
            attendance_date,
            subject,
            counts.get("drowsy", 0),
            counts.get("inattentive", 0),
            counts.get("aggressive", 0),
        )


def send_session_report(
    subject,
    teacher_name,
    recipient_email,
    smtp_sender,
    smtp_password,
    finalize_result,
    session_start,
    session_end,
    report_callback,
):
    """Generate the AI report, email it, and notify Tkinter when available."""
    report, report_path, _ = generate_session_report(
        subject,
        teacher_name,
        finalize_result,
        session_start,
        session_end,
    )
    email_status = email_session_report(
        recipient_email,
        teacher_name,
        subject,
        report,
        report_path,
        finalize_result.get("session_file_path") or finalize_result.get("file_path"),
        smtp_sender,
        smtp_password,
    )

    if report_callback:
        report_callback(report, report_path, email_status)

    return report, report_path, email_status


def start_attendance(
    subject,
    teacher_name="",
    recipient_email="",
    smtp_sender="",
    smtp_password="",
    report_callback=None,
):
    """Start attendance, save Excel/MySQL rows, then email the AI session report."""
    initialize_database()
    recognizer = FaceRecognizer()

    def get_camera():
        # Try external camera first (index 0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            print("[INFO] External camera detected, using it.")
            return cap
        # Fall back to built-in webcam (index 1)
        print("[INFO] No external camera found, using built-in webcam.")
        return cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap = get_camera()
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return None

    marked_students = set()
    session_start = datetime.now()
    print(f"ATTENDANCE STARTED FOR: {subject}")
    print("[CONTROLS] Press 'q' to quit, 's' for screenshot")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            process_attendance_faces(frame, recognizer, subject, marked_students)
            draw_info_panel(
                frame,
                subject,
                mode="attendance",
                present_count=len(marked_students),
            )
            cv2.imshow("YOLO Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nEnding attendance...")
                break
            if key == ord("s"):
                filename = f"screenshot_{safe_name(subject)}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    finalize_result = finalize_attendance(subject, marked_students)
    finalize_result["session_mode"] = "attendance"
    session_end = datetime.now()
    send_session_report(
        subject,
        teacher_name,
        recipient_email,
        smtp_sender,
        smtp_password,
        finalize_result,
        session_start,
        session_end,
        report_callback,
    )
    return finalize_result


def start_class_monitor(
    subject,
    teacher_name="",
    recipient_email="",
    smtp_sender="",
    smtp_password="",
    report_callback=None,
):
    """Run behavior and phone monitoring only; attendance is never marked here."""
    initialize_database()
    recognizer = FaceRecognizer()
    phone_model = get_coco_model()
    phone_counter = PhoneEventCounter()
    attention_tracker = AttentionTracker()
    attention_counts = {}
    behavior_state = {}

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[INFO] External camera not available, falling back to built-in webcam.")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[ERROR] Could not open any camera.")
            return None

    session_start = datetime.now()
    frame_index = 0
    print(f"CLASS MONITOR STARTED FOR: {subject}")
    print("[CONTROLS] Press 'q' to quit, 's' for screenshot")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            phone_count = phone_counter.update(frame, phone_model)
            process_monitor_faces(
                frame,
                recognizer,
                attention_tracker,
                attention_counts,
                behavior_state,
                frame_index,
            )
            draw_info_panel(
                frame,
                subject,
                mode="monitor",
                phone_count=phone_count,
            )
            cv2.imshow("AttendAI Class Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nEnding class monitor...")
                break
            if key == ord("s"):
                filename = f"monitor_screenshot_{safe_name(subject)}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    session_end = datetime.now()
    attendance_date = session_end.strftime("%Y-%m-%d")
    save_attention_counts(subject, attendance_date, attention_counts)
    monitor_result = {
        "subject": subject,
        "date": attendance_date,
        "time": session_end.strftime("%H:%M:%S"),
        "file_path": None,
        "session_file_path": None,
        "present_count": 0,
        "absent_count": 0,
        "total_count": 0,
        "absent_students": [],
        "rows": [],
        "saved": False,
        "session_mode": "class monitor only",
    }
    send_session_report(
        subject,
        teacher_name,
        recipient_email,
        smtp_sender,
        smtp_password,
        monitor_result,
        session_start,
        session_end,
        report_callback,
    )
    return monitor_result
