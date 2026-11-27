"""Thai desktop video-to-images utility."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from extractor import create_output_folder, extract, inspect_video, output_size


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('แยกวิดีโอเป็นภาพ')
        self.geometry('780x690')
        self.minsize(720, 650)
        self.option_add('*Font', ('Tahoma', 10))
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.result_folder = None
        self.selection = ()
        self.selection_label = None
        self.source = tk.StringVar()
        self.destination = tk.StringVar(value=str(Path.home() / 'Pictures' / 'Video Frames'))
        self.interval = tk.StringVar(value='30')
        self.resolution = tk.StringVar(value='ขนาดต้นฉบับ')
        self.width_value = tk.StringVar(value='1920')
        self.height_value = tk.StringVar(value='1080')
        self.keep = tk.BooleanVar(value=True)
        self.format = tk.StringVar(value='JPG')
        self.quality = tk.StringVar(value='95')
        self.info_text = tk.StringVar(value='เลือกวิดีโอเพื่อเริ่มต้น')
        self.status = tk.StringVar(value='พร้อมใช้งาน')
        self.build()
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.after(100, self.poll)

    def build(self):
        outer = ttk.Frame(self, padding=22)
        outer.pack(fill='both', expand=True)
        outer.columnconfigure(0, weight=1)
        ttk.Label(outer, text='แยกวิดีโอเป็นภาพ', font=('Tahoma', 20, 'bold')).grid(sticky='w')
        ttk.Label(outer, text='เลือกช่วงเวลาและขนาดภาพ แล้วบันทึกภาพลงในเครื่อง').grid(sticky='w', pady=(4, 18))
        self.controls = ttk.Frame(outer)
        self.controls.grid(sticky='ew')
        self.controls.columnconfigure(0, weight=1)
        self.path_row('ไฟล์วิดีโอ', self.source, self.choose_source, 0)
        ttk.Label(self.controls, textvariable=self.info_text, wraplength=680).grid(row=2, column=0, columnspan=2, sticky='w', pady=(3, 12))
        self.path_row('โฟลเดอร์บันทึกภาพ', self.destination, self.choose_destination, 3)
        settings = ttk.LabelFrame(self.controls, text='ตั้งค่าการแยกภาพ', padding=14)
        settings.grid(row=5, column=0, columnspan=2, sticky='ew', pady=16)
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text='บันทึกภาพทุก ๆ').grid(row=0, column=0, sticky='w', pady=5)
        interval_row = ttk.Frame(settings)
        interval_row.grid(row=0, column=1, sticky='w')
        ttk.Entry(interval_row, textvariable=self.interval, width=10).pack(side='left')
        ttk.Label(interval_row, text=' วินาที  (เช่น 0.5, 1, 30, 60)').pack(side='left')
        ttk.Label(settings, text='ความละเอียด').grid(row=1, column=0, sticky='w', pady=5)
        ttk.Combobox(settings, textvariable=self.resolution, state='readonly', width=31,
                     values=['ขนาดต้นฉบับ', '3840 × 2160 (4K)', '1920 × 1080 (Full HD)',
                             '1280 × 720 (HD)', '854 × 480', '640 × 360', 'กำหนดเอง']).grid(row=1, column=1, sticky='w')
        dimensions = ttk.Frame(settings)
        dimensions.grid(row=2, column=1, sticky='w', pady=5)
        ttk.Label(dimensions, text='กำหนดเอง: กว้าง ').pack(side='left')
        ttk.Entry(dimensions, textvariable=self.width_value, width=7).pack(side='left')
        ttk.Label(dimensions, text=' × สูง ').pack(side='left')
        ttk.Entry(dimensions, textvariable=self.height_value, width=7).pack(side='left')
        ttk.Label(dimensions, text=' px').pack(side='left')
        ttk.Checkbutton(settings, text='รักษาสัดส่วนภาพ (ให้อยู่ภายในขนาดที่เลือก)', variable=self.keep).grid(row=3, column=1, sticky='w', pady=5)
        ttk.Label(settings, text='ชนิดไฟล์').grid(row=4, column=0, sticky='w', pady=5)
        format_row = ttk.Frame(settings)
        format_row.grid(row=4, column=1, sticky='w')
        ttk.Combobox(format_row, textvariable=self.format, values=['JPG', 'PNG'], state='readonly', width=7).pack(side='left')
        ttk.Label(format_row, text='  คุณภาพ JPG (1–100): ').pack(side='left')
        ttk.Entry(format_row, textvariable=self.quality, width=6).pack(side='left')
        ttk.Label(outer, text='เริ่มที่วินาที 0 • แต่ละครั้งสร้างโฟลเดอร์ใหม่ • เพิ่มขนาดภาพได้ แต่รายละเอียดขึ้นกับต้นฉบับ',
                  wraplength=690).grid(sticky='w', pady=(0, 12))
        self.bar = ttk.Progressbar(outer, maximum=100)
        self.bar.grid(sticky='ew')
        ttk.Label(outer, textvariable=self.status, wraplength=690).grid(sticky='w', pady=10)
        buttons = ttk.Frame(outer)
        buttons.grid(sticky='ew')
        self.start_button = ttk.Button(buttons, text='เริ่มแยกภาพ', command=self.start)
        self.start_button.pack(side='left')
        self.cancel_button = ttk.Button(buttons, text='ยกเลิก', command=self.stop, state='disabled')
        self.cancel_button.pack(side='left', padx=8)
        self.open_button = ttk.Button(buttons, text='เปิดโฟลเดอร์ผลลัพธ์', command=self.open_result, state='disabled')
        self.open_button.pack(side='right')

    def path_row(self, label, variable, command, row):
        ttk.Label(self.controls, text=label).grid(row=row, column=0, sticky='w', pady=(3, 5))
        ttk.Entry(self.controls, textvariable=variable).grid(row=row + 1, column=0, sticky='ew')
        ttk.Button(self.controls, text='เลือก…', command=command).grid(row=row + 1, column=1, padx=(8, 0))

    def choose_source(self):
        paths = filedialog.askopenfilenames(title='เลือกหลายไฟล์ด้วย Ctrl หรือ Shift', filetypes=[
            ('วิดีโอ', '*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.mts *.m2ts'), ('ทุกไฟล์', '*.*')])
        if not paths:
            return
        self.selection = tuple(paths)
        self.selection_label = paths[0] if len(paths) == 1 else f'เลือก {len(paths)} ไฟล์'
        self.source.set(self.selection_label)
        if len(paths) > 1:
            self.info_text.set(', '.join(Path(p).name for p in paths[:3]) +
                               (' …' if len(paths) > 3 else ''))
        else:
            path = paths[0]
            self.info_text.set('กำลังอ่านข้อมูลวิดีโอ…')
            threading.Thread(target=self.probe, args=(path,), daemon=True).start()

    def probe(self, path):
        try:
            info = inspect_video(path)
            self.events.put(('info', path, f'{info.width} × {info.height} px | {info.fps:.2f} fps | ความยาวประมาณ {info.duration:.2f} วินาที'))
        except Exception as exc:
            self.events.put(('info', path, str(exc)))

    def choose_destination(self):
        path = filedialog.askdirectory(title='เลือกโฟลเดอร์บันทึกภาพ')
        if path:
            self.destination.set(path)

    def start(self):
        try:
            text = self.source.get().strip()
            sources = self.selection if text == self.selection_label else (text,)
            if not all(Path(path).is_file() for path in sources):
                raise ValueError('กรุณาเลือกไฟล์วิดีโอที่มีอยู่จริง')
            destination = self.destination.get().strip()
            if not destination:
                raise ValueError('กรุณาเลือกโฟลเดอร์บันทึกภาพ')
            interval = float(self.interval.get())
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError('ช่วงเวลาต้องมากกว่า 0 วินาที')
            selection = self.resolution.get()
            bounds = None
            if selection == 'กำหนดเอง':
                bounds = (int(self.width_value.get()), int(self.height_value.get()))
            elif selection != 'ขนาดต้นฉบับ':
                values = selection.split()
                bounds = (int(values[0]), int(values[2]))
            if bounds:
                output_size(1920, 1080, bounds, self.keep.get())
            quality = int(self.quality.get()) if self.format.get() == 'JPG' else 95
            if not 1 <= quality <= 100:
                raise ValueError('คุณภาพ JPG ต้องอยู่ระหว่าง 1–100')
        except (ValueError, OverflowError) as exc:
            messagebox.showerror('ตรวจสอบการตั้งค่า', str(exc))
            return
        self.cancel.clear()
        self.result_folder = None
        self.bar['value'] = 0
        self.status.set('กำลังเปิดวิดีโอ…')
        self.set_busy(True)
        self.open_button['state'] = 'disabled'
        args = dict(destination=destination, interval=interval, bounds=bounds,
                    keep_aspect=self.keep.get(), image_format=self.format.get().lower(), quality=quality)
        self.counts = {}
        self.finished = 0
        self.total_files = len(sources)
        threading.Thread(target=self.run_job, args=(sources, args), daemon=True).start()

    def run_job(self, sources, args):
        errors = []

        def convert(index, path):
            if self.cancel.is_set():
                return
            def progress(saved, total, seconds, folder):
                self.events.put(('progress', path, saved, total, folder))
            extract(path, **args, cancel=self.cancel, progress=progress,
                    batch_index=index if len(sources) > 1 else None)

        try:
            if len(sources) > 1:
                args = dict(args, destination=create_output_folder(args['destination'], 'batch'))
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = {pool.submit(convert, i, path): path for i, path in enumerate(sources, 1)}
                for job in as_completed(jobs):
                    try:
                        job.result()
                    except Exception as exc:
                        errors.append(f'{jobs[job]} — {exc}')
                    self.events.put(('finished',))
        except Exception as exc:
            errors.append(str(exc))
        self.events.put(('done', errors, args['destination']))

    def set_busy(self, busy):
        self.busy = busy
        def visit(widget):
            for child in widget.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Combobox)):
                    state = 'disabled' if busy else ('readonly' if isinstance(child, ttk.Combobox) else 'normal')
                    child.configure(state=state)
                visit(child)
        visit(self.controls)
        self.start_button['state'] = 'disabled' if busy else 'normal'
        self.cancel_button['state'] = 'normal' if busy else 'disabled'

    def poll(self):
        # Bound each poll so very short sampling intervals cannot starve the UI.
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind, *data = event
            if kind == 'info':
                if self.source.get() == data[0]:
                    self.info_text.set(data[1])
            elif kind == 'progress':
                path, saved, total, folder = data
                self.counts[path] = saved
                if self.total_files == 1:
                    self.result_folder = folder
                    self.bar['value'] = saved / total * 100
                if not self.cancel.is_set():
                    self.status.set(f'ประมวลผลแล้ว {self.finished}/{self.total_files} ไฟล์ | บันทึก {sum(self.counts.values()):,} ภาพ')
            elif kind == 'finished':
                self.finished += 1
                if self.total_files > 1 and not self.cancel.is_set():
                    self.bar['value'] = self.finished / self.total_files * 100
            elif kind == 'done':
                errors, destination = data
                self.set_busy(False)
                if self.total_files > 1 and self.counts:
                    self.result_folder = destination
                if self.result_folder:
                    self.open_button['state'] = 'normal'
                verb = 'ยกเลิกแล้ว' if self.cancel.is_set() else 'ประมวลผลเสร็จแล้ว'
                self.status.set(f'{verb} | บันทึก {sum(self.counts.values()):,} ภาพ | ผิดพลาด {len(errors)} ไฟล์')
                if errors:
                    messagebox.showerror('ไฟล์ที่แปลงไม่สำเร็จ', '\n'.join(errors))
        self.after(100, self.poll)

    def stop(self):
        self.cancel.set()
        self.status.set('กำลังยกเลิก… ภาพที่บันทึกแล้วจะยังอยู่')
        self.cancel_button['state'] = 'disabled'

    def open_result(self):
        if self.result_folder:
            try:
                os.startfile(self.result_folder)
            except OSError as exc:
                messagebox.showerror('เปิดโฟลเดอร์ไม่ได้', str(exc))

    def close(self):
        if self.busy:
            messagebox.showinfo('กำลังแยกภาพ', 'กรุณากดยกเลิก และรอให้หยุดก่อนปิดโปรแกรม')
            return
        self.destroy()


if __name__ == '__main__':
    App().mainloop()
