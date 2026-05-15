import json
import threading
import tkinter as tk
from tkinter import messagebox

from analytics import open_analytics_window
from config import EMAIL_PASSWORD, EMAIL_SENDER
from enroll import enroll_and_encode
from index import start_attendance, start_class_monitor


def enroll():
    """Start student enrollment and MySQL face encoding in a background thread."""
    name = student_entry.get().strip()
    sap = sap_entry.get().strip()

    if name == "" or sap == "":
        messagebox.showwarning("Enroll Student", "Please enter SAP-ID and student name.")
        return

    threading.Thread(
        target=enroll_and_encode,
        args=(name, sap),
        daemon=True,
    ).start()


def start():
    """Start attendance and email the AI report after the session ends."""
    subject = subject_entry.get().strip()
    teacher_name = teacher_entry.get().strip()
    teacher_email = teacher_email_entry.get().strip()
    

    if subject == "" or teacher_name == "" or teacher_email == "":
        messagebox.showwarning(
            "Start Attendance",
            "Please enter subject, teacher name, and teacher email.",
        )
        return

    def report_callback(report, report_path, email_status):
        """Schedule report popup rendering safely on the Tkinter main thread."""
        root.after(0, lambda: show_report_popup(report, report_path, email_status))

    threading.Thread(
        target=start_attendance,
        args=(
            subject,
            teacher_name,
            teacher_email,
            EMAIL_SENDER,
            EMAIL_PASSWORD,
            report_callback,
        ),
        daemon=True,
    ).start()


def show_report_popup(report, report_path, email_status):
    """Display the generated AI report in a Tkinter popup after class monitor ends."""
    popup = tk.Toplevel(root)
    popup.title("AI Session Report")
    popup.geometry("650x520")

    text = tk.Text(popup, wrap=tk.WORD, padx=10, pady=10)
    text.pack(fill=tk.BOTH, expand=True)
    text.insert(tk.END, "AI Session Report\n\n")
    text.insert(tk.END, f"Session Summary:\n{report.get('session_summary', '')}\n\n")
    text.insert(tk.END, f"AI Model Summary:\n{report.get('ai_model_summary', '')}\n\n")
    text.insert(tk.END, f"Behavioral Flags:\n{report.get('behavioral_flags', '')}\n\n")
    text.insert(tk.END, f"Recommendation:\n{report.get('recommendation', '')}\n\n")
    text.insert(tk.END, "JSON:\n")
    text.insert(tk.END, json.dumps(report, indent=2))
    text.insert(tk.END, f"\n\nSaved report: {report_path}")
    text.insert(tk.END, f"\nEmail status: {email_status}")
    text.config(state=tk.DISABLED)

    tk.Button(popup, text="Close", command=popup.destroy).pack(pady=8)


def class_monitor():
    """Start the full class monitor flow with phone, attention, report, and email."""
    subject = subject_entry.get().strip()
    teacher_name = teacher_entry.get().strip()
    teacher_email = teacher_email_entry.get().strip()


    if subject == "" or teacher_name == "" or teacher_email == "":
        messagebox.showwarning(
            "Class Monitor",
            "Please enter subject, teacher name, and teacher email.",
        )
        return

    if EMAIL_SENDER == "" or EMAIL_PASSWORD == "":
        messagebox.showwarning(
            "Class Monitor",
            "Please enter sender Gmail and Gmail app password for email sending.",
        )
        return

    def report_callback(report, report_path, email_status):
        """Schedule report popup rendering safely on the Tkinter main thread."""
        root.after(0, lambda: show_report_popup(report, report_path, email_status))

    threading.Thread(
        target=start_class_monitor,
        args=(
            subject,
            teacher_name,
            teacher_email,
            EMAIL_SENDER,
            EMAIL_PASSWORD,
            report_callback,
        ),
        daemon=True,
    ).start()


def analytics():
    """Open the embedded matplotlib analytics dashboard for the selected subject."""
    open_analytics_window(root, subject_entry.get().strip())


# GUI Setup
root = tk.Tk()
root.title("AI Attendance System")
root.geometry("420x610")

tk.Label(root, text="SAP-ID").pack(pady=(12, 5))
sap_entry = tk.Entry(root, width=34)
sap_entry.pack()

tk.Label(root, text="Student Name").pack(pady=5)
student_entry = tk.Entry(root, width=34)
student_entry.pack()

tk.Button(root, text="Enroll Student", command=enroll, width=22).pack(pady=10)

tk.Label(root, text="Subject Name").pack(pady=(12, 5))
subject_entry = tk.Entry(root, width=34)
subject_entry.pack()

tk.Label(root, text="Teacher Name").pack(pady=5)
teacher_entry = tk.Entry(root, width=34)
teacher_entry.pack()

tk.Label(root, text="Teacher Email").pack(pady=5)
teacher_email_entry = tk.Entry(root, width=34)
teacher_email_entry.pack()



tk.Button(root, text="Start Attendance", command=start, width=22).pack(pady=(14, 5))
tk.Button(root, text="Class Monitor", command=class_monitor, width=22).pack(pady=5)
tk.Button(root, text="Analytics", command=analytics, width=22).pack(pady=5)
tk.Button(root, text="Exit", command=root.destroy, width=22).pack(pady=(16, 5))

root.mainloop()
