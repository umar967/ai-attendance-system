import json
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

from config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    REPORTS_DIR,
)
from database import (
    count_phone_logs_between,
    get_attention_averages,
    get_low_attendance_students,
    get_total_students,
)

try:
    import requests
except ImportError:
    requests = None


REQUIRED_REPORT_KEYS = {
    "session_summary",
    "ai_model_summary",
    "at_risk_students",
    "behavioral_flags",
    "recommendation",
}


def build_session_report_data(subject, teacher_name, finalize_result, session_start, session_end):
    """Collect MySQL-backed attendance and behavior metrics for one class session."""
    attendance_date = finalize_result.get("date") or session_start.strftime("%Y-%m-%d")
    at_risk_students = get_low_attendance_students(threshold=75)
    phone_detection_count = count_phone_logs_between(session_start, session_end)
    attention_averages = get_attention_averages(subject, attendance_date)

    return {
        "teacher_name": teacher_name,
        "subject": subject,
        "date": attendance_date,
        "session_started": session_start.strftime("%Y-%m-%d %H:%M:%S"),
        "session_ended": session_end.strftime("%Y-%m-%d %H:%M:%S"),
        "total_students_enrolled": get_total_students(),
        "present_count": finalize_result.get("present_count", 0),
        "absent_count": finalize_result.get("absent_count", 0),
        "absent_students": finalize_result.get("absent_students", []),
        "students_below_75_percent": at_risk_students,
        "phone_detection_count": phone_detection_count,
        "average_drowsiness_count": attention_averages["avg_drowsy_count"],
        "average_inattention_count": attention_averages["avg_inattentive_count"],
        "average_aggressive_count": attention_averages["avg_aggressive_count"],
        "session_mode": finalize_result.get("session_mode", "attendance"),
    }


def build_ollama_prompt(report_data):
    """Create the JSON-only instruction prompt sent to the local Ollama model."""
    return f"""
You are generating a class session report in easy English.
Write like a helpful teacher assistant. Use short, clear sentences.
If session_mode is "class monitor only", do not say attendance was marked.
In that case, summarize only phone, drowsiness, inattention, and aggressive movement.
Return only a valid JSON object with exactly these keys:
session_summary: string
ai_model_summary: string
at_risk_students: list of sap_ids
behavioral_flags: string
recommendation: string

Use this session data:
{json.dumps(report_data, indent=2)}
"""


def fallback_report(report_data, reason):
    """Create a valid report object when Ollama is unavailable or returns bad JSON."""
    at_risk_sap_ids = [
        student["sap_id"]
        for student in report_data.get("students_below_75_percent", [])
    ]
    if report_data.get("session_mode") == "class monitor only":
        session_summary = (
            f"{report_data['subject']} class monitor session ended. "
            "Attendance was not marked in this mode. "
            f"Phone events counted: {report_data['phone_detection_count']}."
        )
    else:
        session_summary = (
            f"{report_data['subject']} session completed with "
            f"{report_data['present_count']} present and "
            f"{report_data['absent_count']} absent students. "
            f"AI generation fallback was used: {reason}"
        )

    return {
        "session_summary": session_summary,
        "ai_model_summary": (
            "The class report was generated from attendance, phone, drowsiness, "
            "and attention data. The summary is written in easy English."
        ),
        "at_risk_students": at_risk_sap_ids,
        "behavioral_flags": (
            f"Phone detections: {report_data['phone_detection_count']}. "
            f"Average drowsiness count: {report_data['average_drowsiness_count']}. "
            f"Average inattention count: {report_data['average_inattention_count']}. "
            f"Average aggressive movement count: "
            f"{report_data['average_aggressive_count']}."
        ),
        "recommendation": (
            "Review absent and at-risk students, reinforce device policy, "
            "and follow up with students showing repeated attention concerns."
        ),
    }


def normalize_report(report, report_data, reason):
    """Ensure generated reports always contain useful required fields."""
    fallback = fallback_report(report_data, reason)
    normalized = dict(report or {})

    for key in REQUIRED_REPORT_KEYS:
        value = normalized.get(key)
        if key == "at_risk_students":
            if not isinstance(value, list):
                normalized[key] = fallback[key]
            continue

        if value is None or str(value).strip() == "":
            normalized[key] = fallback[key]

    return normalized


def parse_json_response(response_text):
    """Parse a JSON object from the model response text, allowing minor wrapping text."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(response_text[start : end + 1])


def call_ollama_for_report(report_data):
    """Send session data to the local Ollama llama3.2:1b model and parse JSON output."""
    if requests is None:
        return fallback_report(report_data, "requests is not installed")

    prompt = build_ollama_prompt(report_data)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        response_text = payload.get("response", "").strip()
        if not response_text:
            return fallback_report(report_data, "Ollama returned an empty response")

        parsed = parse_json_response(response_text)
        if not REQUIRED_REPORT_KEYS.issubset(parsed.keys()):
            return fallback_report(report_data, "Ollama JSON missed required keys")
        return normalize_report(parsed, report_data, "Ollama returned an empty field")
    except Exception as error:
        print(f"[ERROR] Ollama report generation failed: {error}")
        return fallback_report(report_data, str(error))


def format_report_text(report, report_data):
    """Format the JSON report and source metrics into a readable txt file body."""
    return (
        "AttendAI Session Report\n"
        f"Subject: {report_data['subject']}\n"
        f"Teacher: {report_data['teacher_name']}\n"
        f"Date: {report_data['date']}\n\n"
        f"Session Summary:\n{report['session_summary']}\n\n"
        f"AI Model Summary:\n{report['ai_model_summary']}\n\n"
        f"At-Risk Students:\n{', '.join(map(str, report['at_risk_students'])) or 'None'}\n\n"
        f"Behavioral Flags:\n{report['behavioral_flags']}\n\n"
        f"Recommendation:\n{report['recommendation']}\n\n"
        "Source Metrics:\n"
        f"{json.dumps(report_data, indent=2)}\n"
    )


def save_report_file(report, report_data):
    """Save the generated AI report as a txt file under the reports folder."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_subject = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in report_data["subject"]
    )
    report_path = os.path.join(REPORTS_DIR, f"{safe_subject}_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(format_report_text(report, report_data))
    return report_path


def generate_session_report(subject, teacher_name, finalize_result, session_start, session_end):
    """Build, generate, parse, and save an AI report for a completed class session."""
    report_data = build_session_report_data(
        subject,
        teacher_name,
        finalize_result,
        session_start,
        session_end,
    )
    report = call_ollama_for_report(report_data)
    report = normalize_report(report, report_data, "Report field was empty")
    report_path = save_report_file(report, report_data)
    return report, report_path, report_data


def attach_file(message, file_path):
    """Attach a local file to an EmailMessage if the file exists."""
    if not file_path or not os.path.exists(file_path):
        return False

    with open(file_path, "rb") as file_handle:
        file_data = file_handle.read()

    filename = os.path.basename(file_path)
    if filename.lower().endswith(".xlsx"):
        maintype = "application"
        subtype = "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        maintype = "text"
        subtype = "plain"

    message.add_attachment(
        file_data,
        maintype=maintype,
        subtype=subtype,
        filename=filename,
    )
    return True


def email_session_report(
    recipient_email,
    teacher_name,
    subject,
    report,
    report_path,
    attendance_file_path,
    smtp_sender,
    smtp_password,
):
    """Email the AI report summary and attach the session Excel attendance workbook."""
    if not smtp_sender or not smtp_password:
        return "Email skipped: Gmail sender or app password was not provided."

    try:
        message = EmailMessage()
        message["From"] = smtp_sender
        message["To"] = recipient_email
        message["Subject"] = f"AttendAI Report - {subject}"
        message.set_content(
            f"Dear {teacher_name},\n\n"
            f"{report['session_summary']}\n\n"
            f"AI Model Summary:\n{report['ai_model_summary']}\n\n"
            f"Behavioral Flags:\n{report['behavioral_flags']}\n\n"
            f"Recommendation:\n{report['recommendation']}\n\n"
            "The report file is attached. The attendance workbook is attached "
            "when attendance was marked in this session.\n"
        )
        attach_file(message, attendance_file_path)
        attach_file(message, report_path)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(smtp_sender, smtp_password)
            smtp.send_message(message)
        return "Email sent successfully."
    except Exception as error:
        print(f"[ERROR] Email sending failed: {error}")
        return f"Email failed: {error}"
