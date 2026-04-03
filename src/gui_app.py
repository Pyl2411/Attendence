import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from src.mark_attendance import LABELS_FILE, MODEL_FILE


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES = "5"
ROLES = ["Manager", "Co Founder", "Employee", "Team Leader", "Trainee"]
DEFAULT_COMPANY_NAME = "Vickhardth Automation"
DEFAULT_LOGO_PATH = ROOT_DIR / "assets" / "logo.png"


class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vickhardth Automation - Smart Attendance")
        self.root.geometry("1000x700")
        self.root.configure(bg="#0f172a")
        self.is_busy = False
        self.logo_image = None

        self.name_var = tk.StringVar()
        self.mobile_var = tk.StringVar()
        self.employee_id_var = tk.StringVar()
        self.role_var = tk.StringVar(value=ROLES[2])
        self.company_var = tk.StringVar(value=DEFAULT_COMPANY_NAME)

        self._build_ui()
        self.root.after(1000, self._auto_start_attendance)

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#111827", padx=20, pady=14)
        top.pack(fill="x", padx=14, pady=(14, 8))

        left_brand = tk.Frame(top, bg="#111827")
        left_brand.pack(side="left", fill="x", expand=True)

        tk.Label(
            left_brand,
            text=DEFAULT_COMPANY_NAME.upper(),
            fg="#f8fafc",
            bg="#111827",
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left_brand,
            text="AI FACE RECOGNITION ATTENDANCE PORTAL",
            fg="#93c5fd",
            bg="#111827",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(4, 0))

        logo_box = tk.Frame(top, bg="#111827")
        logo_box.pack(side="right")
        self._render_logo(logo_box)

        content = tk.Frame(self.root, bg="#0f172a")
        content.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        form_card = tk.Frame(content, bg="#1f2937", padx=16, pady=16)
        form_card.pack(side="left", fill="y", padx=(0, 10))

        tk.Label(
            form_card, text="Employee Registration", bg="#1f2937", fg="#f9fafb",
            font=("Segoe UI", 14, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self._field(form_card, "Employee Name", self.name_var, 1)
        self._field(form_card, "Mobile Number", self.mobile_var, 2)
        self._field(form_card, "Employee ID", self.employee_id_var, 3)

        tk.Label(form_card, text="Role", bg="#1f2937", fg="#cbd5e1", font=("Segoe UI", 10)).grid(
            row=4, column=0, sticky="w", pady=6
        )
        role_menu = tk.OptionMenu(form_card, self.role_var, *ROLES)
        role_menu.config(width=26, bg="#0b1220", fg="#e2e8f0", highlightthickness=0)
        role_menu.grid(row=4, column=1, sticky="w", pady=6)

        btns = tk.Frame(form_card, bg="#1f2937")
        btns.grid(row=5, column=0, columnspan=2, pady=(16, 0), sticky="w")

        self.capture_btn = tk.Button(
            btns,
            text="Register & Auto Train",
            command=self.capture_employee,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=8,
        )
        self.capture_btn.pack(side="left", padx=(0, 8))

        log_card = tk.Frame(content, bg="#111827", padx=10, pady=10)
        log_card.pack(side="left", fill="both", expand=True)
        tk.Label(
            log_card, text="System Activity", bg="#111827", fg="#f8fafc",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 8))

        self.log_text = tk.Text(
            log_card,
            height=18,
            wrap="word",
            bg="#020617",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            font=("Consolas", 10),
        )
        self.log_text.pack(fill="both", expand=True)

        self._log("Ready. Fill employee details and click Register & Auto Train.")
        self._log("Attendance will start automatically when a trained model is available.")

    def _render_logo(self, parent):
        try:
            if not DEFAULT_LOGO_PATH.exists():
                raise FileNotFoundError(str(DEFAULT_LOGO_PATH))
            img = tk.PhotoImage(file=str(DEFAULT_LOGO_PATH))
            # Resize for compact header usage (smaller logo).
            max_w = 76
            max_h = 76
            x_factor = max(1, (img.width() + max_w - 1) // max_w)
            y_factor = max(1, (img.height() + max_h - 1) // max_h)
            factor = max(x_factor, y_factor)
            img = img.subsample(factor, factor)
            self.logo_image = img
            tk.Label(parent, image=self.logo_image, bg="#111827").pack(anchor="e")
        except Exception:
            tk.Label(
                parent,
                text="VA",
                fg="#93c5fd",
                bg="#111827",
                font=("Segoe UI", 18, "bold"),
            ).pack(anchor="e")

    def _field(self, parent, label, variable, row):
        tk.Label(parent, text=label, bg="#1f2937", fg="#cbd5e1", font=("Segoe UI", 10)).grid(
            row=row, column=0, sticky="w", pady=6
        )
        tk.Entry(
            parent,
            textvariable=variable,
            width=30,
            bg="#0b1220",
            fg="#f8fafc",
            insertbackground="#f8fafc",
            relief="flat",
        ).grid(row=row, column=1, sticky="w", pady=6)

    def _set_busy(self, busy):
        self.is_busy = busy
        state = "disabled" if busy else "normal"
        self.capture_btn.config(state=state)

    def _log(self, text):
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def _run_command(self, cmd, title, on_success=None):
        if self.is_busy:
            messagebox.showinfo("Please wait", "Another task is already running.")
            return

        def worker():
            succeeded = False
            self.root.after(0, lambda: self._set_busy(True))
            self.root.after(0, lambda: self._log(f"\n[{title}]"))
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    self.root.after(0, lambda l=line: self._log(l.rstrip()))
                return_code = process.wait()
                if return_code == 0:
                    succeeded = True
                    self.root.after(0, lambda: self._log(f"{title} completed successfully."))
                else:
                    self.root.after(
                        0, lambda: self._log(f"{title} failed (exit code {return_code}).")
                    )
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"Error: {exc}"))
            finally:
                self.root.after(0, lambda: self._set_busy(False))
                if succeeded and on_success:
                    self.root.after(100, on_success)

        threading.Thread(target=worker, daemon=True).start()

    def _auto_start_attendance(self):
        if self.is_busy:
            return
        if not MODEL_FILE.exists() or not LABELS_FILE.exists():
            self._log("Attendance is waiting for the first trained model.")
            return
        self._log("Opening attendance camera automatically.")
        self.start_attendance()

    def capture_employee(self):
        name = self.name_var.get().strip()
        mobile = self.mobile_var.get().strip()
        employee_id = self.employee_id_var.get().strip()
        role = self.role_var.get().strip()
        company_name = DEFAULT_COMPANY_NAME
        logo_path = DEFAULT_LOGO_PATH

        if not name:
            messagebox.showerror("Validation", "Please enter employee name.")
            return

        if not mobile.isdigit() or len(mobile) < 10:
            messagebox.showerror("Validation", "Please enter valid mobile number.")
            return

        if not employee_id:
            messagebox.showerror("Validation", "Please enter employee ID.")
            return

        if not role:
            messagebox.showerror("Validation", "Please select role.")
            return

        if not company_name:
            messagebox.showerror("Validation", "Please enter company name.")
            return

        cmd = [
            sys.executable,
            "src/capture_faces.py",
            "--name",
            name,
            "--mobile",
            mobile,
            "--employee-id",
            employee_id,
            "--role",
            role,
            "--company-name",
            company_name,
            "--logo-path",
            logo_path,
            "--samples",
            DEFAULT_SAMPLES,
        ]
        self._run_command(cmd, "Capture Faces", on_success=lambda: self.train_model(start_attendance=True))

    def train_model(self, start_attendance=False):
        cmd = [sys.executable, "src/train_model.py"]
        on_success = self.start_attendance if start_attendance else None
        self._run_command(cmd, "Train Model", on_success=on_success)

    def start_attendance(self):
        cmd = [sys.executable, "src/mark_attendance.py"]
        self._run_command(cmd, "Start Attendance")


def main():
    root = tk.Tk()
    AttendanceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
