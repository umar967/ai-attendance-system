import tkinter as tk
import threading
from index import start_attendance
from enroll import enroll_and_encode


# Enroll Student 
def enroll():
    name = student_entry.get().strip()
    sap = sap_entry.get().strip()

    if name == "" or sap == "":
        return

    threading.Thread(
        target=enroll_and_encode,
        args=(name, sap),
        daemon=True
    ).start()

# Start Attendance 
def start():
    sub = subject_entry.get().strip()
    if sub == "":
        return

    threading.Thread(target=start_attendance, args=(sub,), daemon=True).start()


# GUI Setup

root = tk.Tk()
root.title("AI Attendance System")
root.geometry("300x300")

tk.Label(root, text="SAP-ID").pack(pady=5)
sap_entry = tk.Entry(root)
sap_entry.pack()

tk.Label(root, text="Student Name").pack(pady=5)
student_entry = tk.Entry(root)
student_entry.pack()

tk.Button(root, text="Enroll Student", command=enroll).pack(pady=10)

tk.Label(root, text="Subject Name").pack(pady=5)
subject_entry = tk.Entry(root)
subject_entry.pack()

tk.Button(root, text="Start Attendance", command=start).pack(pady=10)
tk.Button(root, text="Exit", command=root.destroy).pack(pady=5)

root.mainloop()
