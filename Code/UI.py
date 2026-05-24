import csv
import json
import os
import threading
from datetime import date, datetime
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from analytics import open_analytics_window
from config import (
    AGGRESSIVE_MOVEMENT_THRESHOLD,
    AGGRESSIVE_CONSEC_FRAMES,
    AGGRESSIVE_POSE_DELTA_THRESHOLD,
    ATTENTION_BASELINE_FRAMES,
    ATTENTION_CONSEC_FRAMES,
    COCO_MODEL_PATH,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_USER,
    EMAIL_PASSWORD,
    EMAIL_SENDER,
    FACE_CROP_PADDING,
    FACE_MATCH_TOLERANCE,
    HEAD_POSE_PITCH_THRESHOLD,
    HEAD_POSE_YAW_THRESHOLD,
    MONITOR_CROPS_DIR,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PHONE_CLASS_ID,
    PHONE_MISSING_RESET_FRAMES,
    load_theme,
    save_theme,
)
from database import (
    get_all_students,
    get_attendance_percentages_by_subject,
    get_connection,
    get_daily_attendance_trend,
    get_low_attendance_students,
    get_total_students,
    initialize_database,
)
from enroll import enroll_and_encode
from index import (
    get_coco_model,
    process_attendance_faces,
    process_monitor_faces,
    PhoneEventCounter,
    draw_info_panel,
    save_attention_counts,
    safe_name,
)
from attendance import finalize_attendance
from reporting import generate_session_report, email_session_report
from attention import AttentionTracker
from recognize import FaceRecognizer


APP_TITLE = "AttendAI Dashboard"
REFRESH_INTERVAL_MS = 30_000


THEMES = {
    "dark": {
        "bg": "#0F172A",
        "panel": "#111827",
        "card": "#1E293B",
        "card_alt": "#273449",
        "text": "#E5E7EB",
        "muted": "#94A3B8",
        "border": "#334155",
        "primary": "#3B82F6",
        "secondary": "#8B5CF6",
        "success": "#10B981",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "entry": "#0B1220",
        "table": "#111827",
        "table_alt": "#172033",
    },
    "light": {
        "bg": "#F8FAFC",
        "panel": "#E2E8F0",
        "card": "#FFFFFF",
        "card_alt": "#F1F5F9",
        "text": "#0F172A",
        "muted": "#475569",
        "border": "#CBD5E1",
        "primary": "#2563EB",
        "secondary": "#7C3AED",
        "success": "#059669",
        "warning": "#D97706",
        "danger": "#DC2626",
        "entry": "#FFFFFF",
        "table": "#FFFFFF",
        "table_alt": "#F8FAFC",
    },
}


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tip, text=self.text, style="Tooltip.TLabel")
        label.pack(ipadx=8, ipady=5)

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Content.TFrame")
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._update_width)

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _update_width(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)


class AttendAIDashboard:
    def __init__(self, root):
        self.root = root
        self.theme_name = load_theme()  # Load persisted theme
        self.colors = THEMES[self.theme_name]
        self.current_page = "Dashboard"
        self.nav_buttons = {}
        self.page_builders = {
            "Dashboard": self.build_dashboard_page,
            "Enrollment": self.build_enrollment_page,
            "Attendance": self.build_attendance_page,
            "Class Monitor": self.build_monitor_page,
            "Analytics": self.build_analytics_page,
            "Settings": self.build_settings_page,
        }
        self.table_sort_state = {}
        self.camera = None
        self.camera_label = None
        self.camera_running = False
        self.camera_job = None
        self.dashboard_job = None
        self.last_dashboard_stats = {}
        self.att_camera = None
        self.att_camera_running = False
        self.att_camera_job = None
        self.att_recognizer = None
        self.att_marked_students = set()
        self.att_session_start = None
        self.att_subject = ""
        self.att_teacher_name = ""
        self.att_teacher_email = ""
        self.att_camera_label = None
        self.att_view_job = None
        self.att_latest_frame = None
        self.att_new_frame = False
        self.att_target_size = None
        self.att_preview_camera = None
        self.att_preview_running = False
        self.att_preview_job = None
        self.mon_camera = None
        self.mon_camera_running = False
        self.mon_camera_job = None
        self.mon_recognizer = None
        self.mon_phone_counter = None
        self.mon_phone_model = None
        self.mon_attention_tracker = None
        self.mon_attention_counts = {}
        self.mon_behavior_state = {}
        self.mon_frame_index = 0
        self.mon_session_start = None
        self.mon_subject = ""
        self.mon_teacher_name = ""
        self.mon_teacher_email = ""
        self.mon_camera_label = None
        self.mon_phone_count_label = None
        self.mon_view_job = None
        self.mon_latest_frame = None
        self.mon_new_frame = False
        self.mon_target_size = None
        self.mon_latest_phone = 0
        self.mon_preview_camera = None
        self.mon_preview_running = False
        self.mon_preview_job = None
        self.mon_drowsy_label = None
        self.mon_distraction_label = None
        self.root.title(APP_TITLE)
        self.root.geometry("1240x780")
        self.root.minsize(1040, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.style = ttk.Style(self.root)
        self.configure_styles()
        self.build_shell()
        self.bind_shortcuts()
        self.show_page("Dashboard")

    def configure_styles(self):
        c = self.colors
        self.root.configure(bg=c["bg"])
        self.style.theme_use("clam")
        self.style.configure(".", font=("Segoe UI", 10), background=c["bg"], foreground=c["text"])
        self.style.configure("Shell.TFrame", background=c["bg"])
        self.style.configure("Sidebar.TFrame", background=c["panel"])
        self.style.configure("Content.TFrame", background=c["bg"])
        self.style.configure("Card.TFrame", background=c["card"], relief=tk.FLAT)
        self.style.configure("AltCard.TFrame", background=c["card_alt"], relief=tk.FLAT)
        self.style.configure("Header.TFrame", background=c["bg"])
        self.style.configure("TLabel", background=c["bg"], foreground=c["text"])
        self.style.configure("Muted.TLabel", background=c["bg"], foreground=c["muted"])
        self.style.configure("Card.TLabel", background=c["card"], foreground=c["text"])
        self.style.configure("CardMuted.TLabel", background=c["card"], foreground=c["muted"])
        self.style.configure("Metric.TLabel", background=c["card"], foreground=c["text"], font=("Segoe UI", 22, "bold"))
        self.style.configure("Title.TLabel", background=c["bg"], foreground=c["text"], font=("Segoe UI", 20, "bold"))
        self.style.configure("Section.TLabel", background=c["bg"], foreground=c["text"], font=("Segoe UI", 13, "bold"))
        self.style.configure("SidebarTitle.TLabel", background=c["panel"], foreground=c["text"], font=("Segoe UI", 15, "bold"))
        self.style.configure("SidebarMuted.TLabel", background=c["panel"], foreground=c["muted"])
        self.style.configure("Tooltip.TLabel", background=c["card_alt"], foreground=c["text"], relief=tk.SOLID, borderwidth=1)
        self.style.configure("TEntry", fieldbackground=c["entry"], foreground=c["text"], bordercolor=c["border"], insertcolor=c["text"])
        self.style.configure("TCombobox", fieldbackground=c["entry"], foreground=c["text"], bordercolor=c["border"], arrowcolor=c["text"])
        self.style.configure("TButton", background=c["card_alt"], foreground=c["text"], borderwidth=0, focusthickness=0, padding=(12, 8))
        self.style.map("TButton", background=[("active", c["border"])])
        self.style.configure("Primary.TButton", background=c["primary"], foreground="#FFFFFF")
        self.style.map("Primary.TButton", background=[("active", c["secondary"])])
        self.style.configure("Danger.TButton", background=c["danger"], foreground="#FFFFFF")
        self.style.map("Danger.TButton", background=[("active", "#B91C1C")])
        self.style.configure("Nav.TButton", background=c["panel"], foreground=c["muted"], anchor="w", padding=(16, 11))
        self.style.configure("ActiveNav.TButton", background=c["card"], foreground=c["text"], anchor="w", padding=(16, 11))
        self.style.map("Nav.TButton", background=[("active", c["card"])], foreground=[("active", c["text"])])
        self.style.configure("Treeview", background=c["table"], fieldbackground=c["table"], foreground=c["text"], rowheight=30, borderwidth=0)
        self.style.configure("Treeview.Heading", background=c["card_alt"], foreground=c["text"], font=("Segoe UI", 10, "bold"), relief=tk.FLAT)
        self.style.map("Treeview", background=[("selected", c["primary"])], foreground=[("selected", "#FFFFFF")])
        self.style.configure("Horizontal.TProgressbar", background=c["success"], troughcolor=c["card_alt"], bordercolor=c["border"])
        self.style.configure("TNotebook", background=c["bg"], borderwidth=0)
        self.style.configure("TNotebook.Tab", background=c["card_alt"], foreground=c["muted"], padding=(12, 8))
        self.style.map("TNotebook.Tab", background=[("selected", c["card"])], foreground=[("selected", c["text"])])
        if hasattr(self, "sidebar"):
            self.sidebar.configure(style="Sidebar.TFrame")
        for scroll in getattr(self, "scroll_frames", []):
            scroll.canvas.configure(bg=c["bg"])

    def build_shell(self):
        self.shell = ttk.Frame(self.root, style="Shell.TFrame")
        self.shell.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(self.shell, style="Sidebar.TFrame", width=230)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        brand = ttk.Label(self.sidebar, text="AttendAI", style="SidebarTitle.TLabel")
        brand.pack(anchor=tk.W, padx=18, pady=(22, 2))
        ttk.Label(self.sidebar, text="Attendance command center", style="SidebarMuted.TLabel").pack(anchor=tk.W, padx=18, pady=(0, 18))

        nav_items = [
            ("Dashboard", "Home overview"),
            ("Enrollment", "Register students"),
            ("Attendance", "Start attendance"),
            ("Class Monitor", "Behavior monitoring"),
            ("Analytics", "Reports and trends"),
            ("Settings", "Configuration"),
        ]
        for page, tooltip in nav_items:
            button = ttk.Button(
                self.sidebar,
                text=self.nav_text(page),
                style="Nav.TButton",
                command=lambda name=page: self.show_page(name),
            )
            button.pack(fill=tk.X, padx=12, pady=3)
            Tooltip(button, tooltip)
            self.nav_buttons[page] = button

        ttk.Frame(self.sidebar, style="Sidebar.TFrame").pack(fill=tk.BOTH, expand=True)
        self.status_label = ttk.Label(self.sidebar, text="Ready", style="SidebarMuted.TLabel", wraplength=190)
        self.status_label.pack(anchor=tk.W, padx=18, pady=(0, 18))

        self.main = ttk.Frame(self.shell, style="Shell.TFrame")
        self.main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.header = ttk.Frame(self.main, style="Header.TFrame", padding=(22, 16, 22, 8))
        self.header.pack(fill=tk.X)
        self.page_title = ttk.Label(self.header, text="", style="Title.TLabel")
        self.page_title.pack(side=tk.LEFT)
        self.header_actions = ttk.Frame(self.header, style="Header.TFrame")
        self.header_actions.pack(side=tk.RIGHT)

        self.theme_button = ttk.Button(self.header_actions, text=("Dark" if self.theme_name == "light" else "Light"), command=self.toggle_theme)
        self.theme_button.pack(side=tk.RIGHT, padx=(8, 0))
        Tooltip(self.theme_button, "Toggle dark and light mode")
        self.refresh_button = ttk.Button(self.header_actions, text="Refresh", command=self.refresh_current_page)
        self.refresh_button.pack(side=tk.RIGHT)
        Tooltip(self.refresh_button, "Reload data for the current page")

        self.page_host = ttk.Frame(self.main, style="Shell.TFrame", padding=(22, 8, 22, 22))
        self.page_host.pack(fill=tk.BOTH, expand=True)

    def bind_shortcuts(self):
        self.root.bind("<Escape>", lambda _event: self.close_topmost_popup())
        self.root.bind("<Control-Return>", lambda _event: self.primary_action())
        self.root.bind("<Control-r>", lambda _event: self.refresh_current_page())

    def nav_text(self, page):
        icons = {
            "Dashboard": "Home",
            "Enrollment": "+",
            "Attendance": "OK",
            "Class Monitor": "Live",
            "Analytics": "Chart",
            "Settings": "Cfg",
        }
        return f"{icons.get(page, '')}  {page}"

    def show_page(self, page):
        self.camera_label = None
        if self.att_camera_running:
            self.stop_embedded_attendance()
        if self.mon_camera_running:
            self.stop_embedded_monitor()
        self.stop_att_preview()
        self.stop_mon_preview()
        self.stop_camera_preview()
        if self.dashboard_job:
            self.root.after_cancel(self.dashboard_job)
            self.dashboard_job = None
        self.current_page = page
        self.page_title.config(text=page)
        for name, button in self.nav_buttons.items():
            button.configure(style="ActiveNav.TButton" if name == page else "Nav.TButton")
        for widget in self.page_host.winfo_children():
            widget.destroy()
        self.page_builders[page]()

    def refresh_current_page(self):
        self.show_page(self.current_page)

    def set_status(self, text):
        self.status_label.configure(text=text)

    def primary_action(self):
        if self.current_page == "Enrollment" and hasattr(self, "enroll_button"):
            self.enroll_button.invoke()
        elif self.current_page == "Attendance" and hasattr(self, "attendance_button"):
            self.attendance_button.invoke()
        elif self.current_page == "Class Monitor" and hasattr(self, "monitor_button"):
            self.monitor_button.invoke()

    def close_topmost_popup(self):
        popups = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        if popups:
            popups[-1].destroy()

    def make_scroll_page(self):
        scroll = ScrollFrame(self.page_host)
        scroll.pack(fill=tk.BOTH, expand=True)
        scroll.canvas.configure(bg=self.colors["bg"])
        return scroll.content

    def card(self, parent, padding=16):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=padding)
        return frame

    def metric_card(self, parent, title, value, detail="", accent=None):
        frame = self.card(parent)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=title, style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text=str(value), style="Metric.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 2))
        detail_label = ttk.Label(frame, text=detail, style="CardMuted.TLabel", wraplength=210)
        detail_label.grid(row=2, column=0, sticky="w")
        if accent:
            stripe = tk.Frame(frame, bg=accent, width=4)
            stripe.grid(row=0, column=1, rowspan=3, sticky="ns", padx=(14, 0))
        return frame

    def labeled_entry(self, parent, label, default="", show=None, width=28):
        wrapper = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(wrapper, text=label, style="CardMuted.TLabel").pack(anchor=tk.W)
        entry = ttk.Entry(wrapper, width=width, show=show)
        entry.insert(0, default)
        entry.pack(fill=tk.X, pady=(5, 0))
        
        # Redirect layout methods from entry to wrapper
        def custom_pack(**kwargs):
            wrapper.pack(**kwargs)
        def custom_grid(**kwargs):
            wrapper.grid(**kwargs)
        entry.pack = custom_pack
        entry.grid = custom_grid
        
        return entry

    def labeled_combobox(self, parent, label, values=None, default="", width=28):
        wrapper = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(wrapper, text=label, style="CardMuted.TLabel").pack(anchor=tk.W)
        combo = ttk.Combobox(wrapper, values=values or [], width=width)
        combo.set(default)
        combo.pack(fill=tk.X, pady=(5, 0))
        
        # Redirect layout methods from combo to wrapper
        def custom_pack(**kwargs):
            wrapper.pack(**kwargs)
        def custom_grid(**kwargs):
            wrapper.grid(**kwargs)
        combo.pack = custom_pack
        combo.grid = custom_grid
        
        return combo


    def build_dashboard_page(self):
        page = self.make_scroll_page()
        stats = self.load_dashboard_stats()
        self.last_dashboard_stats = stats

        metrics = ttk.Frame(page, style="Content.TFrame")
        metrics.pack(fill=tk.X)
        for index in range(4):
            metrics.columnconfigure(index, weight=1, uniform="metrics")
        cards = [
            ("Enrolled Students", stats["total_students"], "Students in MySQL", self.colors["primary"]),
            ("Classes Held", stats["total_classes"], "Unique subject/date sessions", self.colors["secondary"]),
            ("Average Attendance", f"{stats['average_attendance']:.1f}%", "Across all saved rows", self.colors["success"]),
            ("At Risk", stats["at_risk_count"], "Below 75% attendance", self.colors["danger"]),
        ]
        for col, item in enumerate(cards):
            self.metric_card(metrics, *item).grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0), pady=(0, 12))

        main_grid = ttk.Frame(page, style="Content.TFrame")
        main_grid.pack(fill=tk.BOTH, expand=True)
        main_grid.columnconfigure(0, weight=3)
        main_grid.columnconfigure(1, weight=2)

        chart_card = self.card(main_grid)
        chart_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=10)
        ttk.Label(chart_card, text="Attendance Distribution", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        self.draw_dashboard_charts(chart_card, stats)

        right = ttk.Frame(main_grid, style="Content.TFrame")
        right.grid(row=0, column=1, sticky="nsew", pady=10)
        today_card = self.card(right)
        today_card.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(today_card, text="Today by Subject", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        self.build_today_table(today_card, stats["today_summary"])

        risk_card = self.card(right)
        risk_card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(risk_card, text="Below 75%", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        self.build_risk_list(risk_card, stats["at_risk_students"])

        recent_card = self.card(page)
        recent_card.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(recent_card, text="Recent Activity", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 8))
        self.build_recent_table(recent_card, stats["recent_sessions"])

        self.dashboard_job = self.root.after(REFRESH_INTERVAL_MS, self.auto_refresh_dashboard)

    def auto_refresh_dashboard(self):
        self.dashboard_job = None
        if self.current_page == "Dashboard":
            self.refresh_current_page()

    def load_dashboard_stats(self):
        stats = {
            "total_students": 0,
            "total_classes": 0,
            "average_attendance": 0,
            "at_risk_count": 0,
            "today_summary": [],
            "recent_sessions": [],
            "distribution": {"Present": 0, "Absent": 0},
            "subject_percentages": [],
            "at_risk_students": [],
        }
        try:
            initialize_database()
            stats["total_students"] = get_total_students()
            stats["at_risk_students"] = get_low_attendance_students()
            stats["at_risk_count"] = len(stats["at_risk_students"])
            connection = get_connection()
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT COUNT(DISTINCT subject, date) AS total_classes FROM attendance")
            stats["total_classes"] = int((cursor.fetchone() or {}).get("total_classes") or 0)
            cursor.execute("SELECT status, COUNT(*) AS count FROM attendance GROUP BY status")
            for row in cursor.fetchall():
                stats["distribution"][row["status"]] = int(row["count"] or 0)
            total_rows = sum(stats["distribution"].values())
            if total_rows:
                stats["average_attendance"] = (stats["distribution"].get("Present", 0) / total_rows) * 100
            cursor.execute(
                """
                SELECT subject,
                       SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) AS present_count,
                       COUNT(*) AS total_count
                FROM attendance
                WHERE date = %s
                GROUP BY subject
                ORDER BY subject
                """,
                (date.today(),),
            )
            stats["today_summary"] = cursor.fetchall()
            cursor.execute(
                """
                SELECT subject, date, time,
                       SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) AS present_count,
                       SUM(CASE WHEN status = 'Absent' THEN 1 ELSE 0 END) AS absent_count,
                       COUNT(*) AS total_count
                FROM attendance
                GROUP BY subject, date, time
                ORDER BY date DESC, time DESC
                LIMIT 5
                """
            )
            stats["recent_sessions"] = cursor.fetchall()
            cursor.execute(
                """
                SELECT subject,
                       ROUND(100 * SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) / COUNT(*), 2) AS percentage
                FROM attendance
                GROUP BY subject
                ORDER BY percentage DESC
                LIMIT 5
                """
            )
            stats["subject_percentages"] = cursor.fetchall()
            cursor.close()
            connection.close()
            self.set_status("Dashboard data loaded")
        except Exception as error:
            self.set_status(f"Database offline: {error}")
        return stats

    def draw_dashboard_charts(self, parent, stats):
        c = self.colors
        figure = Figure(figsize=(7.6, 4.2), dpi=100, facecolor=c["card"])
        pie_axis = figure.add_subplot(121)
        bar_axis = figure.add_subplot(122)

        present = stats["distribution"].get("Present", 0)
        absent = stats["distribution"].get("Absent", 0)
        if present or absent:
            pie_axis.pie(
                [present, absent],
                labels=["Present", "Absent"],
                colors=[c["success"], c["danger"]],
                autopct="%1.0f%%",
                textprops={"color": c["text"], "fontsize": 9},
                wedgeprops={"linewidth": 1, "edgecolor": c["card"]},
            )
        else:
            pie_axis.text(0.5, 0.5, "No attendance yet", ha="center", va="center", color=c["muted"])
        pie_axis.set_title("Overall", color=c["text"], fontsize=11)

        subjects = [str(row["subject"]) for row in stats["subject_percentages"]]
        values = [float(row["percentage"] or 0) for row in stats["subject_percentages"]]
        if subjects:
            bar_axis.barh(subjects, values, color=c["primary"])
            bar_axis.set_xlim(0, 100)
            bar_axis.invert_yaxis()
        else:
            bar_axis.text(0.5, 0.5, "No subjects yet", ha="center", va="center", color=c["muted"])
        bar_axis.set_title("Top Subjects", color=c["text"], fontsize=11)
        bar_axis.tick_params(colors=c["muted"], labelsize=8)
        for spine in bar_axis.spines.values():
            spine.set_color(c["border"])
        bar_axis.set_facecolor(c["card"])
        pie_axis.set_facecolor(c["card"])
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def build_today_table(self, parent, rows):
        table = self.create_table(parent, ("subject", "present", "absent", "total"), height=5)
        self.setup_columns(table, {
            "subject": ("Subject", 150),
            "present": ("Present", 80),
            "absent": ("Absent", 80),
            "total": ("Total", 80),
        })
        if not rows:
            table.insert("", tk.END, values=("No sessions today", "-", "-", "-"))
        for row in rows:
            total = int(row["total_count"] or 0)
            present = int(row["present_count"] or 0)
            table.insert("", tk.END, values=(row["subject"], present, total - present, total))
        self.pack_table(table, fill=tk.X, pady=(8, 0))

    def build_risk_list(self, parent, rows):
        table = self.create_table(parent, ("sap_id", "name", "percentage"), height=7)
        self.setup_columns(table, {
            "sap_id": ("SAP-ID", 95),
            "name": ("Name", 170),
            "percentage": ("%", 70),
        })
        if not rows:
            table.insert("", tk.END, values=("-", "No at-risk students", "-"))
        for row in rows[:8]:
            table.insert("", tk.END, values=(row["sap_id"], row["name"], f"{row['percentage']:.1f}%"))
        self.pack_table(table, fill=tk.BOTH, expand=True, pady=(8, 0))

    def build_recent_table(self, parent, rows):
        table = self.create_table(parent, ("subject", "date", "time", "present", "absent", "total"), height=6)
        self.setup_columns(table, {
            "subject": ("Subject", 220),
            "date": ("Date", 130),
            "time": ("Time", 120),
            "present": ("Present", 100),
            "absent": ("Absent", 100),
            "total": ("Total", 100),
        })
        if not rows:
            table.insert("", tk.END, values=("No attendance sessions yet", "-", "-", "-", "-", "-"))
        for row in rows:
            table.insert(
                "",
                tk.END,
                values=(
                    row["subject"],
                    row["date"],
                    row["time"],
                    row["present_count"],
                    row["absent_count"],
                    row["total_count"],
                ),
            )
        self.pack_table(table, fill=tk.X, pady=(2, 0))

    def build_enrollment_page(self):
        page = self.make_scroll_page()
        layout = ttk.Frame(page, style="Content.TFrame")
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=2)
        layout.columnconfigure(1, weight=3)

        form = self.card(layout)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        ttk.Label(form, text="Student Enrollment", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        ttk.Label(form, text="Capture 10 face images and store the face encoding in MySQL.", style="CardMuted.TLabel", wraplength=330).pack(anchor=tk.W, pady=(2, 14))
        self.sap_entry = self.labeled_entry(form, "SAP-ID")
        self.sap_entry.pack(fill=tk.X, pady=7)
        self.student_entry = self.labeled_entry(form, "Student Name")
        self.student_entry.pack(fill=tk.X, pady=7)
        self.enroll_progress = ttk.Progressbar(form, maximum=10, value=0)
        self.enroll_progress.pack(fill=tk.X, pady=(10, 5))
        self.enroll_status = ttk.Label(form, text="Ready to enroll", style="CardMuted.TLabel")
        self.enroll_status.pack(anchor=tk.W)
        self.enroll_button = ttk.Button(form, text="Enroll Student", style="Primary.TButton", command=self.enroll)
        self.enroll_button.pack(fill=tk.X, pady=(14, 7))
        Tooltip(self.enroll_button, "Start webcam enrollment for this student")
        ttk.Button(form, text="Start Preview", command=self.start_camera_preview).pack(fill=tk.X, pady=3)
        ttk.Button(form, text="Stop Preview", command=self.stop_camera_preview).pack(fill=tk.X, pady=3)

        preview = self.card(layout)
        preview.grid(row=0, column=1, sticky="nsew", pady=(0, 12))
        ttk.Label(preview, text="Webcam Preview", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        self.camera_label = ttk.Label(preview, text="Camera preview is off", anchor=tk.CENTER, style="CardMuted.TLabel")
        self.camera_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0), ipady=120)

        students_card = self.card(page)
        students_card.pack(fill=tk.BOTH, expand=True)
        toolbar = ttk.Frame(students_card, style="Card.TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(toolbar, text="Enrolled Students", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        self.student_search = ttk.Entry(toolbar, width=30)
        self.student_search.pack(side=tk.RIGHT, padx=(8, 0))
        self.student_search.insert(0, "")
        self.student_search.bind("<KeyRelease>", lambda _event: self.load_students_table())
        ttk.Label(toolbar, text="Search", style="CardMuted.TLabel").pack(side=tk.RIGHT)
        self.students_table = self.create_table(students_card, ("sap_id", "name", "thumb"), height=9)
        self.setup_columns(self.students_table, {
            "sap_id": ("SAP-ID", 180),
            "name": ("Name", 280),
            "thumb": ("Dataset Preview", 260),
        })
        self.pack_table(self.students_table, fill=tk.BOTH, expand=True)
        self.attach_table_context_menu(
            self.students_table,
            [
                ("Copy selected row", lambda: self.copy_selected_row(self.students_table)),
                ("Export table CSV", lambda: self.export_table_csv(self.students_table)),
                ("Delete student", self.delete_selected_student),
            ],
        )
        buttons = ttk.Frame(students_card, style="Card.TFrame")
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Refresh List", command=self.load_students_table).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Delete Selected", style="Danger.TButton", command=self.delete_selected_student).pack(side=tk.LEFT, padx=8)
        self.load_students_table()

    def enroll(self):
        name = self.student_entry.get().strip()
        # Ensure any running camera preview is stopped before enrollment to avoid webcam conflicts
        self.stop_camera_preview()
        sap = self.sap_entry.get().strip()
        if not name or not sap:
            messagebox.showwarning("Enroll Student", "Please enter SAP-ID and student name.")
            return
        self.enroll_button.configure(state=tk.DISABLED)
        self.enroll_progress.configure(mode="indeterminate")
        self.enroll_progress.start(12)
        self.enroll_status.configure(text="Enrollment running. Watch the camera window and keep your face visible.")
        self.set_status(f"Enrolling {name}")

        def worker():
            success = enroll_and_encode(name, sap)
            self.root.after(0, lambda: self.finish_enrollment(success, name))

        threading.Thread(target=worker, daemon=True).start()

    def finish_enrollment(self, success, name):
        self.enroll_progress.stop()
        self.enroll_progress.configure(mode="determinate", value=10 if success else 0)
        self.enroll_button.configure(state=tk.NORMAL)
        self.enroll_status.configure(text="Enrollment completed" if success else "Enrollment failed. Check console for details.")
        self.set_status(f"{name} enrolled" if success else "Enrollment failed")
        self.load_students_table()
        if success:
            messagebox.showinfo("Enrollment", f"{name} was enrolled successfully.")

    def load_students_table(self):
        if not hasattr(self, "students_table"):
            return
        self.clear_table(self.students_table)
        query = self.student_search.get().strip().lower() if hasattr(self, "student_search") else ""
        rows = []
        try:
            rows = get_all_students()
        except Exception as error:
            self.set_status(f"Could not load students: {error}")
        for student in rows:
            name = str(student["name"])
            sap = str(student["sap_id"])
            if query and query not in name.lower() and query not in sap.lower():
                continue
            thumb = self.find_student_thumbnail(name)
            self.students_table.insert("", tk.END, values=(sap, name, thumb or "No thumbnail found"))
        if not self.students_table.get_children():
            self.students_table.insert("", tk.END, values=("-", "No students found", "-"))

    def find_student_thumbnail(self, name):
        folder = os.path.join("dataset", name)
        if not os.path.isdir(folder):
            return ""
        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(folder, filename)
        return ""

    def delete_selected_student(self):
        selected = self.students_table.selection()
        if not selected:
            messagebox.showwarning("Delete Student", "Please select a student first.")
            return
        values = self.students_table.item(selected[0], "values")
        sap_id, name = values[0], values[1]
        if sap_id == "-":
            return
        confirmed = messagebox.askyesno("Delete Student", f"Delete {name} ({sap_id}) from MySQL attendance system?")
        if not confirmed:
            return
        try:
            initialize_database()
            connection = get_connection()
            cursor = connection.cursor()
            cursor.execute("DELETE FROM students WHERE sap_id = %s", (sap_id,))
            connection.commit()
            cursor.close()
            connection.close()

            # Clean up the local dataset folder
            import shutil
            from config import DATASET_DIR
            folder = os.path.join(DATASET_DIR, name)
            if os.path.isdir(folder):
                try:
                    shutil.rmtree(folder)
                except Exception as e:
                    print(f"[WARNING] Could not delete dataset folder for {name}: {e}")

            self.load_students_table()
            self.set_status(f"Deleted {name}")
        except Exception as error:
            messagebox.showerror("Delete Student", f"Could not delete student:\n{error}")

    def start_camera_preview(self):
        if cv2 is None or Image is None or ImageTk is None:
            messagebox.showwarning("Camera Preview", "OpenCV and Pillow are required for embedded preview.")
            return
        if self.camera_running:
            return
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera = None
            messagebox.showerror("Camera Preview", "Could not open webcam.")
            return
        self.camera_running = True
        self.update_camera_preview()

    def update_camera_preview(self):
        if not self.camera_running or self.camera is None or self.camera_label is None:
            return
        ok, frame = self.camera.read()
        if ok:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = frame.shape[:2]
                target_width = max(self.camera_label.winfo_width(), 520)
                scale = min(target_width / width, 520 / height)
                # Correct resize dimensions tuple
                image = Image.fromarray(frame).resize((int(width * scale), int(height * scale)))
                photo = ImageTk.PhotoImage(image)
                if self.camera_label.winfo_exists():
                    self.camera_label.configure(image=photo, text="")
                    self.camera_label.image = photo
            except Exception as e:
                print(f"[WARNING] Camera preview update failed: {e}")
        # Schedule next frame update if still running
        if self.camera_running:
            self.camera_job = self.root.after(33, self.update_camera_preview)

    def stop_camera_preview(self):
        self.camera_running = False
        if self.camera_job:
            self.root.after_cancel(self.camera_job)
            self.camera_job = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        # camera_label may already be None (nulled in show_page) or the
        # underlying Tk widget may be destroyed — always guard with try/except.
        if self.camera_label is not None:
            try:
                if self.camera_label.winfo_exists():
                    self.camera_label.configure(image="", text="Camera preview is off")
                    self.camera_label.image = None
            except Exception:
                pass
            self.camera_label = None

    def toggle_theme(self):
        """Toggle between light and dark themes and refresh UI colors."""
        # Switch theme name
        self.theme_name = "dark" if self.theme_name == "light" else "light"
        self.colors = THEMES[self.theme_name]
        # Reconfigure all styles with new colors
        self.configure_styles()
        # Update theme button label to indicate opposite theme
        new_label = "Dark" if self.theme_name == "light" else "Light"
        self.theme_button.configure(text=new_label)
        # Persist the selected theme
        save_theme(self.theme_name)

    def build_attendance_page(self):
        page = self.make_scroll_page()
        layout = ttk.Frame(page, style="Content.TFrame")
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.columnconfigure(1, weight=2)

        form = self.card(layout)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ttk.Label(form, text="Attendance Session", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        ttk.Label(form, text="Face recognition with liveness check. Camera feed appears in this dashboard.", style="CardMuted.TLabel", wraplength=340).pack(anchor=tk.W, pady=(2, 14))
        self.attendance_subject = self.labeled_combobox(form, "Subject", self.get_subjects())
        self.attendance_subject.pack(fill=tk.X, pady=7)
        self.teacher_entry = self.labeled_entry(form, "Teacher Name")
        self.teacher_entry.pack(fill=tk.X, pady=7)
        self.teacher_email_entry = self.labeled_entry(form, "Teacher Email")
        self.teacher_email_entry.pack(fill=tk.X, pady=7)
        self.attendance_button = ttk.Button(form, text="Start Attendance", style="Primary.TButton", command=self.start_attendance_session)
        self.attendance_button.pack(fill=tk.X, pady=(14, 7))
        preview_btn_frame = ttk.Frame(form, style="Card.TFrame")
        preview_btn_frame.pack(fill=tk.X, pady=3)
        self.att_preview_start = ttk.Button(preview_btn_frame, text="Start Preview", command=self.start_att_preview)
        self.att_preview_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.att_preview_stop = ttk.Button(preview_btn_frame, text="Stop Preview", command=self.stop_att_preview)
        self.att_preview_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Button(form, text="Open Analytics Window", command=lambda: open_analytics_window(self.root, self.attendance_subject.get().strip())).pack(fill=tk.X, pady=3)
        ttk.Label(form, text="Tip: press q key or click Stop to finish and generate the report.", style="CardMuted.TLabel", wraplength=340).pack(anchor=tk.W, pady=(12, 0))

        right = ttk.Frame(layout, style="Content.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        cam_card = self.card(right)
        cam_card.grid(row=0, column=0, sticky="nsew")
        ttk.Label(cam_card, text="Camera Feed", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        self.att_camera_label = ttk.Label(cam_card, text="Camera is off. Use Start Preview to test or Start Attendance to begin.", anchor=tk.CENTER, style="CardMuted.TLabel")
        self.att_camera_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0), ipady=80)

        log_card = self.card(right, padding=8)
        log_card.grid(row=1, column=0, sticky="ew")
        ttk.Label(log_card, text="Session Log", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.attendance_log = tk.Text(log_card, height=6, wrap=tk.WORD, bg=self.colors["entry"], fg=self.colors["text"], insertbackground=self.colors["text"], relief=tk.FLAT)
        self.attendance_log.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.append_log(self.attendance_log, "Ready. Attendance results and report status will appear here.\n")

    def start_attendance_session(self):
        subject = self.attendance_subject.get().strip()
        teacher_name = self.teacher_entry.get().strip()
        teacher_email = self.teacher_email_entry.get().strip()
        if not subject or not teacher_name or not teacher_email:
            messagebox.showwarning("Start Attendance", "Please enter subject, teacher name, and teacher email.")
            return
        self.attendance_button.configure(state=tk.DISABLED)
        msg = f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting attendance for {subject}...\n"
        self.append_log(self.attendance_log, msg)
        self.set_status(f"Attendance running: {subject}")
        self.start_embedded_attendance(subject, teacher_name, teacher_email)

    def build_monitor_page(self):
        page = self.make_scroll_page()
        layout = ttk.Frame(page, style="Content.TFrame")
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.columnconfigure(1, weight=2)

        form = self.card(layout)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ttk.Label(form, text="Class Monitor", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        ttk.Label(form, text="Tracks phone, drowsiness, distraction, and aggressive movement.", style="CardMuted.TLabel", wraplength=340).pack(anchor=tk.W, pady=(2, 14))
        self.monitor_subject = self.labeled_combobox(form, "Subject", self.get_subjects())
        self.monitor_subject.pack(fill=tk.X, pady=7)
        self.monitor_teacher = self.labeled_entry(form, "Teacher Name")
        self.monitor_teacher.pack(fill=tk.X, pady=7)
        self.monitor_email = self.labeled_entry(form, "Teacher Email")
        self.monitor_email.pack(fill=tk.X, pady=7)
        self.monitor_button = ttk.Button(form, text="Start Monitor", style="Primary.TButton", command=self.start_monitor_session)
        self.monitor_button.pack(fill=tk.X, pady=(14, 7))
        preview_btn_frame = ttk.Frame(form, style="Card.TFrame")
        preview_btn_frame.pack(fill=tk.X, pady=3)
        self.mon_preview_start = ttk.Button(preview_btn_frame, text="Start Preview", command=self.start_mon_preview)
        self.mon_preview_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.mon_preview_stop = ttk.Button(preview_btn_frame, text="Stop Preview", command=self.stop_mon_preview)
        self.mon_preview_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Label(form, text="The report is emailed after you quit the camera session.", style="CardMuted.TLabel", wraplength=340).pack(anchor=tk.W, pady=(12, 0))

        right = ttk.Frame(layout, style="Content.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        cam_card = self.card(right)
        cam_card.grid(row=0, column=0, sticky="nsew")
        stats_row = ttk.Frame(cam_card, style="Card.TFrame")
        stats_row.pack(fill=tk.X)
        for i in range(3):
            stats_row.columnconfigure(i, weight=1, uniform="moncam")
        phone_card = self.card(stats_row, padding=8)
        phone_card.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        ttk.Label(phone_card, text="Phone Events", style="CardMuted.TLabel").pack(anchor=tk.W)
        self.mon_phone_count_label = ttk.Label(phone_card, text="0", style="Metric.TLabel")
        self.mon_phone_count_label.pack(anchor=tk.W)
        drowsy_card = self.card(stats_row, padding=8)
        drowsy_card.grid(row=0, column=1, sticky="nsew", padx=4)
        ttk.Label(drowsy_card, text="Drowsy", style="CardMuted.TLabel").pack(anchor=tk.W)
        self.mon_drowsy_label = ttk.Label(drowsy_card, text="0", style="Metric.TLabel")
        self.mon_drowsy_label.pack(anchor=tk.W)
        dist_card = self.card(stats_row, padding=8)
        dist_card.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        ttk.Label(dist_card, text="Distraction", style="CardMuted.TLabel").pack(anchor=tk.W)
        self.mon_distraction_label = ttk.Label(dist_card, text="0", style="Metric.TLabel")
        self.mon_distraction_label.pack(anchor=tk.W)

        ttk.Label(cam_card, text="Camera Feed", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.mon_camera_label = ttk.Label(cam_card, text="Camera is off. Use Start Preview to test or Start Monitor to begin.", anchor=tk.CENTER, style="CardMuted.TLabel")
        self.mon_camera_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0), ipady=80)

        log_card = self.card(right, padding=8)
        log_card.grid(row=1, column=0, sticky="ew")
        ttk.Label(log_card, text="Monitor Log", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.monitor_log = tk.Text(log_card, height=6, wrap=tk.WORD, bg=self.colors["entry"], fg=self.colors["text"], insertbackground=self.colors["text"], relief=tk.FLAT)
        self.monitor_log.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.append_log(self.monitor_log, "Ready. Behavior report status will appear here.\n")

    def start_monitor_session(self):
        subject = self.monitor_subject.get().strip()
        teacher_name = self.monitor_teacher.get().strip()
        teacher_email = self.monitor_email.get().strip()
        if not subject or not teacher_name or not teacher_email:
            messagebox.showwarning("Class Monitor", "Please enter subject, teacher name, and teacher email.")
            return
        if not EMAIL_SENDER or not EMAIL_PASSWORD:
            messagebox.showwarning("Class Monitor", "Please configure sender Gmail and app password.")
            return
        self.monitor_button.configure(state=tk.DISABLED)
        msg = f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting class monitor for {subject}...\n"
        self.append_log(self.monitor_log, msg)
        self.set_status(f"Monitor running: {subject}")
        self.start_embedded_monitor(subject, teacher_name, teacher_email)

    def build_analytics_page(self):
        page = self.make_scroll_page()
        controls = self.card(page)
        controls.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(controls, text="Analytics & Reports", style="Card.TLabel", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w", columnspan=6)
        ttk.Label(controls, text="Subject", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.analytics_subject = ttk.Combobox(controls, values=self.get_subjects(), width=26)
        self.analytics_subject.grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        if self.analytics_subject["values"]:
            self.analytics_subject.set(self.analytics_subject["values"][0])
        ttk.Label(controls, text="From Date", style="CardMuted.TLabel").grid(row=1, column=1, sticky="w", pady=(12, 0))
        self.from_date = ttk.Entry(controls, width=16)
        self.from_date.grid(row=2, column=1, sticky="w", padx=(0, 10), pady=(5, 0))
        ttk.Label(controls, text="To Date", style="CardMuted.TLabel").grid(row=1, column=2, sticky="w", pady=(12, 0))
        self.to_date = ttk.Entry(controls, width=16)
        self.to_date.grid(row=2, column=2, sticky="w", padx=(0, 10), pady=(5, 0))
        ttk.Button(controls, text="Load", style="Primary.TButton", command=self.load_analytics).grid(row=2, column=3, sticky="w", padx=(0, 8), pady=(5, 0))
        ttk.Button(controls, text="Export Table", command=lambda: self.export_table_csv(self.at_risk_table)).grid(row=2, column=4, sticky="w", padx=(0, 8), pady=(5, 0))
        ttk.Button(controls, text="Generate Report Preview", command=self.generate_preview_report).grid(row=2, column=5, sticky="w", pady=(5, 0))

        charts = self.card(page)
        charts.pack(fill=tk.BOTH, expand=True)
        self.analytics_chart_frame = charts
        self.at_risk_table = self.create_table(page, ("sap_id", "name", "percentage", "classes", "reason"), height=8)
        self.setup_columns(self.at_risk_table, {
            "sap_id": ("SAP-ID", 170),
            "name": ("Name", 260),
            "percentage": ("Attendance %", 140),
            "classes": ("Present/Total", 150),
            "reason": ("Reason", 340),
        })
        self.pack_table(self.at_risk_table, fill=tk.X, pady=(12, 0))
        self.attach_table_context_menu(
            self.at_risk_table,
            [
                ("Copy selected row", lambda: self.copy_selected_row(self.at_risk_table)),
                ("Export table CSV", lambda: self.export_table_csv(self.at_risk_table)),
            ],
        )
        if self.analytics_subject.get().strip():
            self.load_analytics()

    def load_analytics(self):
        subject = self.analytics_subject.get().strip()
        if not subject:
            messagebox.showwarning("Analytics", "Please enter a subject.")
            return
        attendance_rows = get_attendance_percentages_by_subject(subject)
        trend_rows = get_daily_attendance_trend(subject)
        self.draw_analytics_charts(attendance_rows, trend_rows)
        self.clear_table(self.at_risk_table)
        for row in attendance_rows:
            if row["percentage"] >= 75:
                continue
            reason = "No attendance yet" if row["total_classes"] == 0 else "Below 75% attendance"
            self.at_risk_table.insert(
                "",
                tk.END,
                values=(
                    row["sap_id"],
                    row["name"],
                    f"{row['percentage']:.2f}%",
                    f"{row['present_classes']}/{row['total_classes']}",
                    reason,
                ),
            )
        if not self.at_risk_table.get_children():
            self.at_risk_table.insert("", tk.END, values=("-", "No at-risk students", "-", "-", "All students are at or above 75%"))
        self.set_status(f"Analytics loaded for {subject}")

    def draw_analytics_charts(self, attendance_rows, trend_rows):
        for widget in self.analytics_chart_frame.winfo_children():
            widget.destroy()
        c = self.colors
        figure = Figure(figsize=(10, 5.2), dpi=100, facecolor=c["card"])
        bar_axis = figure.add_subplot(211)
        line_axis = figure.add_subplot(212)
        names = [row["name"] for row in attendance_rows]
        percentages = [row["percentage"] for row in attendance_rows]
        colors = [c["danger"] if value < 75 else c["primary"] for value in percentages]
        if names:
            bars = bar_axis.bar(names, percentages, color=colors)
            for bar, value in zip(bars, percentages):
                bar_axis.text(bar.get_x() + bar.get_width() / 2, min(value + 2, 98), f"{value:.0f}%", ha="center", color=c["text"], fontsize=8)
        else:
            bar_axis.text(0.5, 0.5, "No attendance data found", ha="center", va="center", color=c["muted"])
        bar_axis.axhline(75, color=c["warning"], linestyle="--", linewidth=1.2)
        bar_axis.set_ylim(0, 100)
        bar_axis.set_title("Per-Student Attendance", color=c["text"])
        bar_axis.tick_params(axis="x", labelrotation=20, colors=c["muted"], labelsize=8)
        bar_axis.tick_params(axis="y", colors=c["muted"])

        dates = [row["date"] for row in trend_rows]
        values = [row["percentage"] for row in trend_rows]
        if dates:
            line_axis.plot(dates, values, marker="o", color=c["success"], linewidth=2)
        else:
            line_axis.text(0.5, 0.5, "No daily trend yet", ha="center", va="center", color=c["muted"])
        line_axis.set_ylim(0, 100)
        line_axis.set_title("Daily Attendance Trend", color=c["text"])
        line_axis.tick_params(axis="x", labelrotation=20, colors=c["muted"], labelsize=8)
        line_axis.tick_params(axis="y", colors=c["muted"])

        for axis in (bar_axis, line_axis):
            axis.set_facecolor(c["card"])
            for spine in axis.spines.values():
                spine.set_color(c["border"])
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=self.analytics_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.analytics_figure = figure

    def generate_preview_report(self):
        subject = self.analytics_subject.get().strip() or "Selected Subject"
        report = {
            "session_summary": f"{subject} analytics preview is ready.",
            "ai_model_summary": "Use a completed attendance or monitor session to generate a full AI report with live source metrics.",
            "at_risk_students": [
                self.at_risk_table.item(item, "values")[0]
                for item in self.at_risk_table.get_children()
                if self.at_risk_table.item(item, "values")[0] != "-"
            ],
            "behavioral_flags": "Open Class Monitor to collect drowsiness, phone, distraction, and aggressive movement data.",
            "recommendation": "Review the at-risk table and follow up before the next attendance cycle.",
        }
        self.show_report_popup(report, "Preview only", "Email not sent from preview")

    def build_settings_page(self):
        page = self.make_scroll_page()
        grid = ttk.Frame(page, style="Content.TFrame")
        grid.pack(fill=tk.BOTH, expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        db_card = self.card(grid)
        db_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 12))
        ttk.Label(db_card, text="Database", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        self.db_host = self.labeled_entry(db_card, "Host", DB_HOST)
        self.db_host.pack(fill=tk.X, pady=6)
        self.db_user = self.labeled_entry(db_card, "User", DB_USER)
        self.db_user.pack(fill=tk.X, pady=6)
        self.db_password = self.labeled_entry(db_card, "Password", DB_PASSWORD, show="*")
        self.db_password.pack(fill=tk.X, pady=6)
        self.db_name = self.labeled_entry(db_card, "Database", DB_NAME)
        self.db_name.pack(fill=tk.X, pady=6)
        ttk.Button(db_card, text="Test Connection", style="Primary.TButton", command=self.test_database_connection).pack(fill=tk.X, pady=(12, 0))

        ai_card = self.card(grid)
        ai_card.grid(row=0, column=1, sticky="nsew", pady=(0, 12))
        ttk.Label(ai_card, text="AI & Email", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        self.ollama_url = self.labeled_entry(ai_card, "Ollama URL", OLLAMA_URL)
        self.ollama_url.pack(fill=tk.X, pady=6)
        self.ollama_model = self.labeled_entry(ai_card, "Ollama Model", OLLAMA_MODEL)
        self.ollama_model.pack(fill=tk.X, pady=6)
        self.sender_email = self.labeled_entry(ai_card, "Sender Email", EMAIL_SENDER)
        self.sender_email.pack(fill=tk.X, pady=6)
        self.sender_password = self.labeled_entry(ai_card, "App Password", EMAIL_PASSWORD, show="*")
        self.sender_password.pack(fill=tk.X, pady=6)

        recognition = self.card(grid)
        recognition.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(recognition, text="Recognition", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        self.tolerance_var = tk.DoubleVar(value=FACE_MATCH_TOLERANCE)
        ttk.Label(recognition, text="Face recognition tolerance", style="CardMuted.TLabel").pack(anchor=tk.W, pady=(12, 0))
        tolerance = ttk.Scale(recognition, from_=0.3, to=0.7, variable=self.tolerance_var)
        tolerance.pack(fill=tk.X, pady=6)
        self.liveness_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(recognition, text="Liveness detection", variable=self.liveness_var).pack(anchor=tk.W, pady=8)
        self.theme_var = tk.BooleanVar(value=self.theme_name == "dark")
        ttk.Checkbutton(recognition, text="Dark mode", variable=self.theme_var, command=self.toggle_theme_from_setting).pack(anchor=tk.W, pady=8)

        note = self.card(grid)
        note.grid(row=1, column=1, sticky="nsew")
        ttk.Label(note, text="Configuration Note", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        ttk.Label(
            note,
            text="These controls mirror current settings for testing and tuning in the UI. Permanent config edits are intentionally left out so the working backend files stay unchanged.",
            style="CardMuted.TLabel",
            wraplength=460,
        ).pack(anchor=tk.W, pady=(12, 0))
        ttk.Button(note, text="Reset Visible Defaults", command=self.refresh_current_page).pack(fill=tk.X, pady=(18, 0))

    def test_database_connection(self):
        try:
            ok = initialize_database()
            if not ok:
                raise RuntimeError("Database initialization returned false.")
            connection = get_connection()
            connection.close()
            messagebox.showinfo("Database", "Connection successful.")
            self.set_status("Database connection successful")
        except Exception as error:
            messagebox.showerror("Database", f"Connection failed:\n{error}")
            self.set_status("Database connection failed")

    def toggle_theme_from_setting(self):
        target = "dark" if self.theme_var.get() else "light"
        if target != self.theme_name:
            self.toggle_theme()

    def get_subjects(self):
        subjects = set()
        try:
            initialize_database()
            connection = get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT DISTINCT subject FROM attendance ORDER BY subject")
            subjects.update(row[0] for row in cursor.fetchall() if row[0])
            cursor.close()
            connection.close()
        except Exception:
            pass
        excel_dir = "attendance_excel"
        if os.path.isdir(excel_dir):
            for filename in os.listdir(excel_dir):
                if filename.lower().endswith(".xlsx"):
                    subjects.add(os.path.splitext(filename)[0])
        return sorted(subjects)

    def create_table(self, parent, columns, height=8):
        container = ttk.Frame(parent, style="Card.TFrame")
        table = ttk.Treeview(container, columns=columns, show="headings", height=height, selectmode="extended")
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)
        table._container = container
        table._scrollbar = scrollbar
        return table

    def pack_table(self, table, **pack_options):
        table._container.pack(**pack_options)
        table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        table._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_columns(self, table, columns):
        for column, (label, width) in columns.items():
            table.heading(column, text=label, command=lambda c=column, t=table: self.sort_table(t, c))
            table.column(column, width=width, anchor=tk.W)

    def clear_table(self, table):
        for item in table.get_children():
            table.delete(item)

    def sort_table(self, table, column):
        key = (str(table), column)
        reverse = not self.table_sort_state.get(key, False)
        self.table_sort_state[key] = reverse
        rows = [(table.set(item, column), item) for item in table.get_children("")]

        def normalize(value):
            text = str(value).replace("%", "")
            try:
                return float(text)
            except ValueError:
                return str(value).lower()

        rows.sort(key=lambda row: normalize(row[0]), reverse=reverse)
        for index, (_, item) in enumerate(rows):
            table.move(item, "", index)

    def attach_table_context_menu(self, table, actions):
        menu = tk.Menu(table, tearoff=False, bg=self.colors["card"], fg=self.colors["text"])
        for label, command in actions:
            menu.add_command(label=label, command=command)

        def show_menu(event):
            item = table.identify_row(event.y)
            if item:
                table.selection_set(item)
            menu.tk_popup(event.x_root, event.y_root)

        table.bind("<Button-3>", show_menu)

    def copy_selected_row(self, table):
        selected = table.selection()
        if not selected:
            return
        values = table.item(selected[0], "values")
        self.root.clipboard_clear()
        self.root.clipboard_append("\t".join(map(str, values)))
        self.set_status("Selected row copied")

    def export_table_csv(self, table):
        path = filedialog.asksaveasfilename(
            title="Export table",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        columns = table["columns"]
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            for item in table.get_children():
                writer.writerow(table.item(item, "values"))
        self.set_status(f"Exported {os.path.basename(path)}")

    def append_log(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.insert(tk.END, text)
        widget.see(tk.END)

    def show_report_popup(self, report, report_path, email_status):
        popup = tk.Toplevel(self.root)
        popup.title("AI Session Report")
        popup.geometry("760x580")
        popup.configure(bg=self.colors["bg"])
        frame = ttk.Frame(popup, style="Content.TFrame", padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="AI Session Report", style="Title.TLabel").pack(anchor=tk.W)
        text = tk.Text(frame, wrap=tk.WORD, bg=self.colors["entry"], fg=self.colors["text"], insertbackground=self.colors["text"], relief=tk.FLAT)
        text.pack(fill=tk.BOTH, expand=True, pady=(12, 10))
        text.insert(tk.END, f"Session Summary:\n{report.get('session_summary', '')}\n\n")
        text.insert(tk.END, f"AI Model Summary:\n{report.get('ai_model_summary', '')}\n\n")
        text.insert(tk.END, f"At-Risk Students:\n{', '.join(map(str, report.get('at_risk_students', []))) or 'None'}\n\n")
        text.insert(tk.END, f"Behavioral Flags:\n{report.get('behavioral_flags', '')}\n\n")
        text.insert(tk.END, f"Recommendation:\n{report.get('recommendation', '')}\n\n")
        text.insert(tk.END, "JSON:\n")
        text.insert(tk.END, json.dumps(report, indent=2))
        text.insert(tk.END, f"\n\nSaved report: {report_path}")
        text.insert(tk.END, f"\nEmail status: {email_status}")
        text.config(state=tk.DISABLED)
        ttk.Button(frame, text="Close", command=popup.destroy).pack(anchor=tk.E)

    # ── Embedded Attendance Camera (threaded for smooth UI) ────────────

    def start_embedded_attendance(self, subject, teacher_name, teacher_email):
        self.stop_att_preview()
        self.att_subject = subject
        self.att_teacher_name = teacher_name
        self.att_teacher_email = teacher_email
        self.att_marked_students = set()
        self.att_session_start = datetime.now()

        self.att_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.att_camera.isOpened():
            self.att_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.att_camera.isOpened():
            messagebox.showerror("Camera", "Could not open webcam.")
            self.attendance_button.configure(state=tk.NORMAL)
            return

        self.att_recognizer = FaceRecognizer()
        self.att_target_size = None
        self.att_latest_frame = None
        self.att_new_frame = False
        self.att_camera_running = True

        self.att_camera_label.configure(image="", text="Starting camera...")
        self.attendance_button.configure(text="Stop Attendance", style="Danger.TButton", command=self.stop_embedded_attendance)

        self.root.bind("<KeyPress-q>", self.att_key_q)
        self.root.bind("<KeyPress-s>", self.att_key_s)

        self.att_view_job = self.root.after(33, self._att_view_loop)
        threading.Thread(target=self._att_worker, daemon=True).start()

    def _att_worker(self):
        try:
            while self.att_camera_running and self.att_camera is not None:
                ret, frame = self.att_camera.read()
                if not ret:
                    break
                process_attendance_faces(frame, self.att_recognizer, self.att_subject, self.att_marked_students)
                draw_info_panel(frame, self.att_subject, mode="attendance", present_count=len(self.att_marked_students))
                self.att_latest_frame = frame
                self.att_new_frame = True
        except Exception:
            pass
        finally:
            self.root.after(0, self.stop_embedded_attendance)

    def _att_view_loop(self):
        if not self.att_camera_running:
            return
        try:
            if self.att_new_frame and self.att_latest_frame is not None:
                self.att_new_frame = False
                rgb = cv2.cvtColor(self.att_latest_frame, cv2.COLOR_BGR2RGB)
                fh, fw = rgb.shape[:2]
                if self.att_target_size is None:
                    self.root.update_idletasks()
                    lw = max(self.att_camera_label.winfo_width(), 400)
                    lh = max(self.att_camera_label.winfo_height(), 300)
                    scale = min(lw / fw, lh / fh)
                    self.att_target_size = (int(fw * scale), int(fh * scale))
                tw, th = self.att_target_size
                img = Image.fromarray(rgb).resize((tw, th))
                photo = ImageTk.PhotoImage(img)
                self.att_camera_label.configure(image=photo, text="")
                self.att_camera_label.image = photo
            self.att_view_job = self.root.after(33, self._att_view_loop)
        except Exception:
            pass

    def att_key_q(self, _event=None):
        self.stop_embedded_attendance()

    def att_key_s(self, _event=None):
        if self.att_latest_frame is not None:
            name = f"screenshot_{safe_name(self.att_subject)}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(name, self.att_latest_frame)
            self.append_log(self.attendance_log, f"Screenshot saved: {name}\n")

    def stop_embedded_attendance(self):
        self.att_camera_running = False
        if self.att_camera is not None:
            self.att_camera.release()
            self.att_camera = None
        self.att_latest_frame = None
        self.att_new_frame = False
        if self.att_view_job:
            self.root.after_cancel(self.att_view_job)
            self.att_view_job = None
        self.root.unbind("<KeyPress-q>")
        self.root.unbind("<KeyPress-s>")

        self.attendance_button.configure(text="Start Attendance", style="Primary.TButton", command=self.start_attendance_session)
        self.att_camera_label.configure(image="", text="Camera is off. Use Start Preview to test or Start Attendance to begin.")
        self.att_camera_label.image = None

        self.attendance_button.configure(state=tk.NORMAL)
        msg = f"\n[{datetime.now().strftime('%H:%M:%S')}] Camera stopped. Finalizing...\n"
        self.append_log(self.attendance_log, msg)
        self.set_status("Finalizing attendance session...")

        def finalize_worker():
            try:
                result = finalize_attendance(self.att_subject, self.att_marked_students)
                result["session_mode"] = "attendance"
                session_end = datetime.now()
                report, report_path, email_status = generate_session_report(
                    self.att_subject,
                    self.att_teacher_name,
                    result,
                    self.att_session_start,
                    session_end,
                )
                email_sent = email_session_report(
                    self.att_teacher_email,
                    self.att_teacher_name,
                    self.att_subject,
                    report,
                    report_path,
                    result.get("session_file_path") or result.get("file_path"),
                    EMAIL_SENDER,
                    EMAIL_PASSWORD,
                )
                self.root.after(0, lambda: self.show_report_popup(report, report_path, "Sent" if email_sent else "Failed"))
                self.root.after(0, lambda: self.append_log(
                    self.attendance_log,
                    f"Completed. Present: {result.get('present_count', 0)}, Absent: {result.get('absent_count', 0)}.\n"
                    f"Report: {report_path}\nEmail: {'Sent' if email_sent else 'Failed'}\n"))
                self.root.after(0, lambda: self.set_status(f"Attendance session finished: {self.att_subject}"))
            except Exception as e:
                self.root.after(0, lambda: self.append_log(self.attendance_log, f"Error finalizing: {e}\n"))
                self.root.after(0, lambda: self.set_status("Attendance finalization failed"))

        threading.Thread(target=finalize_worker, daemon=True).start()

    # ── Embedded Class Monitor Camera (threaded for smooth UI) ─────────

    def start_embedded_monitor(self, subject, teacher_name, teacher_email):
        self.stop_mon_preview()
        self.mon_subject = subject
        self.mon_teacher_name = teacher_name
        self.mon_teacher_email = teacher_email
        self.mon_session_start = datetime.now()
        self.mon_frame_index = 0
        self.mon_phone_counter = PhoneEventCounter()
        self.mon_attention_tracker = AttentionTracker()
        self.mon_attention_counts = {}
        self.mon_behavior_state = {}

        self.mon_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.mon_camera.isOpened():
            self.mon_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.mon_camera.isOpened():
            messagebox.showerror("Camera", "Could not open webcam.")
            self.monitor_button.configure(state=tk.NORMAL)
            return

        self.mon_recognizer = FaceRecognizer()
        self.mon_phone_model = get_coco_model()
        self.mon_target_size = None
        self.mon_latest_frame = None
        self.mon_new_frame = False
        self.mon_camera_running = True

        self.mon_camera_label.configure(image="", text="Starting camera...")
        self.monitor_button.configure(text="Stop Monitor", style="Danger.TButton", command=self.stop_embedded_monitor)

        self.root.bind("<KeyPress-q>", self.mon_key_q)
        self.root.bind("<KeyPress-s>", self.mon_key_s)

        self.mon_view_job = self.root.after(33, self._mon_view_loop)
        threading.Thread(target=self._mon_worker, daemon=True).start()

    def _mon_worker(self):
        try:
            while self.mon_camera_running and self.mon_camera is not None:
                ret, frame = self.mon_camera.read()
                if not ret:
                    break
                self.mon_frame_index += 1
                phone_count = self.mon_phone_counter.update(frame, self.mon_phone_model)
                process_monitor_faces(frame, self.mon_recognizer, self.mon_attention_tracker,
                    self.mon_attention_counts, self.mon_behavior_state, self.mon_frame_index)
                draw_info_panel(frame, self.mon_subject, mode="monitor", phone_count=phone_count)
                self.mon_latest_frame = frame
                self.mon_new_frame = True
                self.mon_latest_phone = phone_count
        except Exception:
            pass
        finally:
            self.root.after(0, self.stop_embedded_monitor)

    def _mon_view_loop(self):
        if not self.mon_camera_running:
            return
        try:
            if self.mon_new_frame and self.mon_latest_frame is not None:
                self.mon_new_frame = False
                rgb = cv2.cvtColor(self.mon_latest_frame, cv2.COLOR_BGR2RGB)
                fh, fw = rgb.shape[:2]
                if self.mon_target_size is None:
                    self.root.update_idletasks()
                    lw = max(self.mon_camera_label.winfo_width(), 400)
                    lh = max(self.mon_camera_label.winfo_height(), 300)
                    scale = min(lw / fw, lh / fh)
                    self.mon_target_size = (int(fw * scale), int(fh * scale))
                tw, th = self.mon_target_size
                img = Image.fromarray(rgb).resize((tw, th))
                photo = ImageTk.PhotoImage(img)
                self.mon_camera_label.configure(image=photo, text="")
                self.mon_camera_label.image = photo
                self.mon_phone_count_label.configure(text=str(self.mon_latest_phone))
                drowsy = sum(c["drowsy"] for c in self.mon_attention_counts.values())
                inattentive = sum(c["inattentive"] for c in self.mon_attention_counts.values())
                self.mon_drowsy_label.configure(text=str(drowsy))
                self.mon_distraction_label.configure(text=str(inattentive))
            self.mon_view_job = self.root.after(33, self._mon_view_loop)
        except Exception:
            pass

    def mon_key_q(self, _event=None):
        self.stop_embedded_monitor()

    def mon_key_s(self, _event=None):
        if self.mon_latest_frame is not None:
            name = f"monitor_screenshot_{safe_name(self.mon_subject)}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(name, self.mon_latest_frame)
            self.append_log(self.monitor_log, f"Screenshot saved: {name}\n")

    def stop_embedded_monitor(self):
        self.mon_camera_running = False
        if self.mon_camera is not None:
            self.mon_camera.release()
            self.mon_camera = None
        self.mon_latest_frame = None
        self.mon_new_frame = False
        if self.mon_view_job:
            self.root.after_cancel(self.mon_view_job)
            self.mon_view_job = None
        self.root.unbind("<KeyPress-q>")
        self.root.unbind("<KeyPress-s>")

        self.monitor_button.configure(text="Start Monitor", style="Primary.TButton", command=self.start_monitor_session)
        self.mon_camera_label.configure(image="", text="Camera is off. Use Start Preview to test or Start Monitor to begin.")
        self.mon_camera_label.image = None

        self.monitor_button.configure(state=tk.NORMAL)
        msg = f"\n[{datetime.now().strftime('%H:%M:%S')}] Camera stopped. Finalizing...\n"
        self.append_log(self.monitor_log, msg)
        self.set_status("Finalizing class monitor session...")

        def finalize_worker():
            try:
                session_end = datetime.now()
                attendance_date = session_end.strftime("%Y-%m-%d")
                save_attention_counts(self.mon_subject, attendance_date, self.mon_attention_counts)
                monitor_result = {
                    "subject": self.mon_subject,
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
                report, report_path, email_status = generate_session_report(
                    self.mon_subject,
                    self.mon_teacher_name,
                    monitor_result,
                    self.mon_session_start,
                    session_end,
                )
                email_sent = email_session_report(
                    self.mon_teacher_email,
                    self.mon_teacher_name,
                    self.mon_subject,
                    report,
                    report_path,
                    monitor_result.get("session_file_path") or monitor_result.get("file_path"),
                    EMAIL_SENDER,
                    EMAIL_PASSWORD,
                )
                self.root.after(0, lambda: self.show_report_popup(report, report_path, "Sent" if email_sent else "Failed"))
                self.root.after(0, lambda: self.append_log(
                    self.monitor_log,
                    f"Report: {report_path}\nEmail: {'Sent' if email_sent else 'Failed'}\n"))
                self.root.after(0, lambda: self.set_status(f"Class monitor finished: {self.mon_subject}"))
            except Exception as e:
                self.root.after(0, lambda: self.append_log(self.monitor_log, f"Error finalizing: {e}\n"))
                self.root.after(0, lambda: self.set_status("Monitor finalization failed"))

        threading.Thread(target=finalize_worker, daemon=True).start()

    # ── Attendance Preview (simple, no processing, same camera label) ──

    def start_att_preview(self):
        if self.att_preview_running or self.att_camera_running:
            return
        self.att_preview_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.att_preview_camera.isOpened():
            self.att_preview_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.att_preview_camera.isOpened():
            messagebox.showwarning("Camera", "Could not open webcam.")
            return
        self.att_preview_running = True
        self.att_preview_target = None
        self.att_camera_label.configure(image="", text="Starting preview...")
        self.att_preview_job = self.root.after(33, self.att_preview_loop)

    def att_preview_loop(self):
        if not self.att_preview_running or self.att_preview_camera is None:
            return
        try:
            ret, frame = self.att_preview_camera.read()
            if not ret:
                self.stop_att_preview()
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw = rgb.shape[:2]
            if self.att_preview_target is None:
                self.root.update_idletasks()
                lw = max(self.att_camera_label.winfo_width(), 400)
                lh = max(self.att_camera_label.winfo_height(), 300)
                scale = min(lw / fw, lh / fh)
                self.att_preview_target = (int(fw * scale), int(fh * scale))
            img = Image.fromarray(rgb).resize(self.att_preview_target)
            photo = ImageTk.PhotoImage(img)
            self.att_camera_label.configure(image=photo, text="")
            self.att_camera_label.image = photo
            self.att_preview_job = self.root.after(33, self.att_preview_loop)
        except Exception:
            self.stop_att_preview()

    def stop_att_preview(self):
        self.att_preview_running = False
        if self.att_preview_job:
            self.root.after_cancel(self.att_preview_job)
            self.att_preview_job = None
        if self.att_preview_camera is not None:
            self.att_preview_camera.release()
            self.att_preview_camera = None
        try:
            if not self.att_camera_running:
                self.att_camera_label.configure(image="", text="Camera is off. Use Start Preview to test or Start Attendance to begin.")
                self.att_camera_label.image = None
        except Exception:
            pass

    # ── Monitor Preview (simple, no processing, same camera label) ─────

    def start_mon_preview(self):
        if self.mon_preview_running or self.mon_camera_running:
            return
        self.mon_preview_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.mon_preview_camera.isOpened():
            self.mon_preview_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.mon_preview_camera.isOpened():
            messagebox.showwarning("Camera", "Could not open webcam.")
            return
        self.mon_preview_running = True
        self.mon_preview_target = None
        self.mon_camera_label.configure(image="", text="Starting preview...")
        self.mon_preview_job = self.root.after(33, self.mon_preview_loop)

    def mon_preview_loop(self):
        if not self.mon_preview_running or self.mon_preview_camera is None:
            return
        try:
            ret, frame = self.mon_preview_camera.read()
            if not ret:
                self.stop_mon_preview()
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw = rgb.shape[:2]
            if self.mon_preview_target is None:
                self.root.update_idletasks()
                lw = max(self.mon_camera_label.winfo_width(), 400)
                lh = max(self.mon_camera_label.winfo_height(), 300)
                scale = min(lw / fw, lh / fh)
                self.mon_preview_target = (int(fw * scale), int(fh * scale))
            img = Image.fromarray(rgb).resize(self.mon_preview_target)
            photo = ImageTk.PhotoImage(img)
            self.mon_camera_label.configure(image=photo, text="")
            self.mon_camera_label.image = photo
            self.mon_preview_job = self.root.after(33, self.mon_preview_loop)
        except Exception:
            self.stop_mon_preview()

    def stop_mon_preview(self):
        self.mon_preview_running = False
        if self.mon_preview_job:
            self.root.after_cancel(self.mon_preview_job)
            self.mon_preview_job = None
        if self.mon_preview_camera is not None:
            self.mon_preview_camera.release()
            self.mon_preview_camera = None
        try:
            if not self.mon_camera_running:
                self.mon_camera_label.configure(image="", text="Camera is off. Use Start Preview to test or Start Monitor to begin.")
                self.mon_camera_label.image = None
        except Exception:
            pass

    def close(self):
        self.att_camera_running = False
        if self.att_camera_job:
            self.root.after_cancel(self.att_camera_job)
        if self.att_camera is not None:
            self.att_camera.release()
        self.mon_camera_running = False
        if self.mon_camera_job:
            self.root.after_cancel(self.mon_camera_job)
        if self.mon_camera is not None:
            self.mon_camera.release()
        self.stop_att_preview()
        self.stop_mon_preview()
        if self.dashboard_job:
            self.root.after_cancel(self.dashboard_job)
            self.dashboard_job = None
        self.stop_camera_preview()
        self.root.destroy()


def main():
    root = tk.Tk()
    AttendAIDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
