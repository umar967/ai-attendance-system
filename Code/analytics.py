import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from database import (
    get_attendance_percentages_by_subject,
    get_daily_attendance_trend,
)


def clear_frame(frame):
    """Remove all child widgets from a Tkinter frame before redrawing analytics."""
    for widget in frame.winfo_children():
        widget.destroy()


def draw_charts(chart_frame, attendance_rows, trend_rows):
    """Draw the attendance bar chart and daily trend line chart inside Tkinter."""
    clear_frame(chart_frame)

    figure = Figure(figsize=(8, 5), dpi=100)
    bar_axis = figure.add_subplot(211)
    line_axis = figure.add_subplot(212)

    names = [row["name"] for row in attendance_rows]
    percentages = [row["percentage"] for row in attendance_rows]
    colors = ["#b23a48" if value < 75 else "#2f6f8f" for value in percentages]
    bars = bar_axis.bar(names, percentages, color=colors)
    bar_axis.axhline(75, color="red", linestyle="--", linewidth=1.5)
    bar_axis.set_ylim(0, 100)
    bar_axis.set_ylabel("Attendance %")
    bar_axis.set_title("Attendance Percentage Per Student")
    bar_axis.tick_params(axis="x", labelrotation=25)
    for bar, percentage in zip(bars, percentages):
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            max(percentage, 2),
            f"{percentage:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    dates = [row["date"] for row in trend_rows]
    trend_percentages = [row["percentage"] for row in trend_rows]
    line_axis.plot(dates, trend_percentages, marker="o", color="#4b8f5c")
    line_axis.set_ylim(0, 100)
    line_axis.set_ylabel("Class Attendance %")
    line_axis.set_title("Daily Attendance Trend")
    line_axis.tick_params(axis="x", labelrotation=25)

    figure.tight_layout()
    canvas = FigureCanvasTkAgg(figure, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def fill_at_risk_table(table, summary_label, attendance_rows):
    """Populate the at-risk table with students below the 75 percent threshold."""
    for item in table.get_children():
        table.delete(item)

    at_risk_rows = [row for row in attendance_rows if row["percentage"] < 75]
    summary_label.config(
        text=f"At-Risk Students Below 75%: {len(at_risk_rows)}"
    )

    if not at_risk_rows:
        table.insert(
            "",
            tk.END,
            values=("-", "No at-risk students", "-", "-", "All students are at or above 75%"),
        )
        return

    for row in at_risk_rows:
        reason = "No attendance yet"
        if row["total_classes"] > 0:
            reason = "Below 75% attendance"
        table.insert(
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


def load_subject_analytics(subject_var, chart_frame, table, summary_label):
    """Load MySQL analytics for the selected subject and refresh charts plus table."""
    subject = subject_var.get().strip()
    if not subject:
        messagebox.showwarning("Analytics", "Please enter a subject name.")
        return

    attendance_rows = get_attendance_percentages_by_subject(subject)
    trend_rows = get_daily_attendance_trend(subject)
    if not attendance_rows:
        messagebox.showinfo("Analytics", "No attendance data found for this subject.")
        clear_frame(chart_frame)
        fill_at_risk_table(table, summary_label, [])
        return

    draw_charts(chart_frame, attendance_rows, trend_rows)
    fill_at_risk_table(table, summary_label, attendance_rows)


def open_analytics_window(parent, default_subject=""):
    """Open a Tkinter analytics dashboard for a selected subject."""
    window = tk.Toplevel(parent)
    window.title("Attendance Analytics")
    window.geometry("950x750")

    controls = ttk.Frame(window, padding=10)
    controls.pack(fill=tk.X)

    ttk.Label(controls, text="Subject").pack(side=tk.LEFT, padx=(0, 8))
    subject_var = tk.StringVar(value=default_subject)
    subject_entry = ttk.Entry(controls, textvariable=subject_var, width=30)
    subject_entry.pack(side=tk.LEFT, padx=(0, 8))

    chart_frame = ttk.Frame(window, padding=10)
    chart_frame.pack(fill=tk.BOTH, expand=True)

    table_frame = ttk.Frame(window, padding=10)
    table_frame.pack(fill=tk.X)
    summary_label = ttk.Label(table_frame, text="At-Risk Students Below 75%: 0")
    summary_label.pack(anchor=tk.W)
    table = ttk.Treeview(
        table_frame,
        columns=("sap_id", "name", "percentage", "classes", "reason"),
        show="headings",
        height=6,
    )
    table.heading("sap_id", text="SAP-ID")
    table.heading("name", text="Name")
    table.heading("percentage", text="Attendance %")
    table.heading("classes", text="Present/Classes")
    table.heading("reason", text="Reason")
    table.column("sap_id", width=160)
    table.column("name", width=260)
    table.column("percentage", width=140)
    table.column("classes", width=140)
    table.column("reason", width=280)
    table.pack(fill=tk.X, pady=(5, 0))

    ttk.Button(
        controls,
        text="Load",
        command=lambda: load_subject_analytics(
            subject_var,
            chart_frame,
            table,
            summary_label,
        ),
    ).pack(side=tk.LEFT)

    if default_subject:
        window.after(
            100,
            lambda: load_subject_analytics(
                subject_var,
                chart_frame,
                table,
                summary_label,
            ),
        )
