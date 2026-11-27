"""Record RTSP video and display a live preview."""

import math
import os
import queue
import threading
from time import monotonic
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from urllib.parse import quote

# Configure native logging before loading OpenCV so RTSP credentials stay out of logs.
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
import cv2

CAMERAS = ("10.2.8.20", "10.2.8.21", "10.2.8.22")


def camera_url(address):
    password = quote("View!Rmutl@", safe="")
    return f"rtsp://view:{password}@{address}"


def stream_camera(source, destinations, stop, report, preview):
    cap = cv2.VideoCapture()
    writer = None
    count = 0
    started = None
    elapsed = -1
    path = None
    try:
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            8000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            5000,
        ]
        if not cap.open(source, cv2.CAP_FFMPEG, params):
            raise ValueError(
                "เปิดกล้องไม่ได้ กรุณาตรวจ IP, พอร์ต, พาธ, รหัสผ่าน และเครือข่าย"
            )
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not 0 < fps <= 240:
            fps = 25.0
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if stop.is_set():
                    break
                raise ValueError("สัญญาณกล้องขาดหรืออ่านภาพไม่ได้")
            if writer is None:
                try:
                    folder = destinations.get_nowait()
                except queue.Empty:
                    preview(frame)
                    continue
                path = folder / datetime.now().strftime("camera_%Y%m%d_%H%M%S_%f.mp4")
            h, w = frame.shape[:2]
            if writer is None:
                size = (w, h)
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
                )
                if not writer.isOpened():
                    raise ValueError("สร้างไฟล์ MP4 ไม่ได้ กรุณาตรวจโฟลเดอร์ปลายทาง")
            if (w, h) != size:
                raise ValueError("กล้องเปลี่ยนความละเอียด กรุณาเริ่มบันทึกใหม่")
            writer.write(frame)
            count += 1
            now = monotonic()
            if started is None:
                started = now
            seconds = int(now - started)
            if seconds != elapsed:
                elapsed = seconds
                duration = f"{seconds // 60:02d}:{seconds % 60:02d}"
                report(f"กำลังบันทึก {duration} | {count:,} เฟรม")
            preview(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return f"บันทึกแล้ว {duration} | {path}" if count else "ปิดภาพสดแล้ว"


def main():
    root = tk.Tk()
    root.title("บันทึกกล้อง RTSP")
    panel = ttk.Frame(root, padding=16)
    panel.pack(fill="both", expand=True)
    panel.columnconfigure(1, weight=1)

    host = tk.StringVar(value=CAMERAS[2])
    ttk.Label(panel, text="กล้อง RTSP").grid(row=0, column=0, sticky="w")
    camera = ttk.Combobox(panel, textvariable=host, values=CAMERAS, state="readonly")
    camera.grid(row=0, column=1, sticky="ew", pady=4)
    folder = tk.StringVar(value=r"E:\RMUTL\object DET 101\videos")
    ttk.Label(panel, text="โฟลเดอร์ปลายทาง").grid(row=1, column=0, sticky="w")
    ttk.Entry(panel, textvariable=folder, width=45).grid(
        row=1, column=1, sticky="ew", pady=4
    )

    def choose_folder():
        selected = filedialog.askdirectory()
        if selected:
            folder.set(selected)

    ttk.Button(panel, text="เลือกโฟลเดอร์", command=choose_folder).grid(row=1, column=2)
    status = tk.StringVar(value="พร้อมใช้งาน")
    events = queue.Queue()
    frames = deque(maxlen=1)
    preview_window = None
    preview_label = None
    stop = threading.Event()
    busy = closing = False
    destinations = None
    preview_record = None

    def begin_recording():
        if closing or (busy and stop.is_set()):
            return
        try:
            if not folder.get().strip():
                raise ValueError("กรุณาระบุโฟลเดอร์ปลายทาง")
            out = Path(folder.get().strip())
            out.mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as exc:
            messagebox.showerror("ตรวจสอบข้อมูล", str(exc))
            return
        if not busy:
            start_preview()
        destinations.put(out)
        start_button.configure(state="disabled")
        preview_record.configure(state="disabled")
        status.set("กำลังเริ่มบันทึก…")

    def start_preview():
        nonlocal busy, preview_window, preview_label, destinations, preview_record
        if busy or closing:
            return
        source = camera_url(host.get())
        destinations = queue.Queue()
        stop.clear()
        busy = True
        toggle(True)
        preview_window = tk.Toplevel(root)
        preview_window.title(f"ภาพสด {host.get()}")
        preview_window.geometry("900x720")
        preview_record = ttk.Button(
            preview_window, text="เริ่มบันทึก", command=begin_recording
        )
        preview_record.pack(side="bottom", pady=3)
        ttk.Label(preview_window, textvariable=status).pack(side="bottom")
        preview_label = ttk.Label(
            preview_window, text="กำลังเปิดภาพสด…", anchor="center"
        )
        preview_label.pack(fill="both", expand=True)
        preview_window.bind("<KeyPress-q>", lambda event: stop.set())
        preview_window.protocol("WM_DELETE_WINDOW", stop.set)
        preview_window.focus_set()
        status.set("กำลังเชื่อมต่อกล้อง…")

        def run():
            try:
                message = stream_camera(
                    source, destinations, stop, events.put, frames.append
                )
            except (ValueError, OSError) as exc:
                message = str(exc)
            except Exception:
                message = "ประมวลผลไม่สำเร็จ กรุณาตรวจสอบวิดีโอและโฟลเดอร์ปลายทาง"
            events.put(("done", message))

        threading.Thread(target=run, daemon=True).start()

    bar = ttk.Frame(panel)
    bar.grid(row=2, columnspan=3, sticky="w", pady=8)
    preview_button = ttk.Button(bar, text="ดูภาพสด", command=start_preview)
    preview_button.pack(side="left", padx=3)
    start_button = ttk.Button(bar, text="เริ่มบันทึก", command=begin_recording)
    stop_button = ttk.Button(bar, text="หยุด", command=stop.set)
    start_button.pack(side="left", padx=3)
    stop_button.pack(side="left", padx=3)
    ttk.Label(panel, textvariable=status, wraplength=510).grid(
        row=3, columnspan=3, sticky="w"
    )

    def toggle(active):
        camera.configure(state="disabled" if active else "readonly")
        preview_button.configure(state="disabled" if active else "normal")
        start_button.configure(state="normal")
        stop_button.configure(state="normal" if active else "disabled")

    def poll():
        nonlocal busy, preview_window
        if preview_window is not None and frames:
            frame = cv2.resize(frames.pop(), (900, 720))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = tk.PhotoImage(
                data=b"P6\n900 720\n255\n" + rgb.tobytes(), format="PPM"
            )
            preview_label.configure(image=photo)
            preview_label.image = photo
            if status.get() == "กำลังเชื่อมต่อกล้อง…":
                status.set("กำลังดูภาพสด")
        while not events.empty():
            event = events.get_nowait()
            if isinstance(event, str):
                status.set(event)
            else:
                if preview_window is not None:
                    preview_window.destroy()
                    preview_window = None
                    frames.clear()
                busy = False
                toggle(False)
                status.set(event[1])
        if closing and not busy:
            root.destroy()
        else:
            root.after(120, poll)

    def close():
        nonlocal closing
        closing = True
        stop.set()
        status.set("กำลังปิดไฟล์…")

    toggle(False)
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(120, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
