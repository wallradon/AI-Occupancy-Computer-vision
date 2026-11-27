"""Combine dated YOLO datasets without changing source files.

Run without arguments for the graphical interface.
CLI preview: python merge_yolo_datasets.py --preview
CLI merge:   python merge_yolo_datasets.py --merge --sets 00_2026-08-28 01_2026-08-31
Outputs are new, uniquely named directories under ROOT/merged_datasets.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = {0: "person"}


def scan_sets(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Folder not found: {root}")
    result = []
    for source in sorted(root.iterdir()):
        if not source.is_dir() or not re.match(r"^(?:[A-Za-z]-)?\d+_", source.name):
            continue
        images_dir = source / "images"
        images = sorted(
            (p for p in images_dir.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda p: p.name.lower(),
        ) if images_dir.is_dir() else []
        candidates = [
            images_dir / "yolo_labels", source / "labels",
            images_dir / "labels", images_dir,
        ]
        labels_dir = next(
            (p for p in candidates if p.is_dir() and any(p.glob("*.txt"))),
            None,
        )
        matched = sum(
            (labels_dir / (image.stem + ".txt")).is_file()
            for image in images
        ) if labels_dir else 0
        result.append({
            "name": source.name, "path": source, "images": images,
            "labels_dir": labels_dir, "matched": matched,
            "ready": bool(images) and matched == len(images),
        })
    return result


def validate_label(path):
    """Empty files represent explicitly labeled background images."""
    rows = path.read_text(encoding="utf-8-sig").splitlines()
    count = 0
    for line_no, line in enumerate(rows, 1):
        if not line.strip():
            continue
        parts = line.split()
        where = f"{path.name}, line {line_no}"
        if len(parts) != 5:
            raise ValueError(f"{where}: expected class cx cy width height")
        if parts[0] != "0":
            raise ValueError(f"{where}: only class 0 (person) is supported")
        try:
            cx, cy, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{where}: invalid number") from exc
        if not all(math.isfinite(v) and 0 <= v <= 1
                   for v in (cx, cy, width, height)):
            raise ValueError(f"{where}: coordinates must be finite and within 0..1")
        if width <= 0 or height <= 0:
            raise ValueError(f"{where}: width and height must be positive")
        tolerance = 0.000001
        if (cx - width / 2 < -tolerance
                or cy - height / 2 < -tolerance
                or cx + width / 2 > 1 + tolerance
                or cy + height / 2 > 1 + tolerance):
            raise ValueError(f"{where}: box extends beyond the image")
        count += 1
    return count


def prepare_merge(root, selected, copy_json=True):
    """Read and validate every selected source before creating output."""
    root = Path(root).resolve()
    available = {item["name"]: item for item in scan_sets(root)}
    if not selected:
        raise ValueError("Select at least one dataset.")
    if len(selected) != len(set(selected)):
        raise ValueError("A dataset was selected more than once.")
    records = []
    output_names = set()
    for name in selected:
        if name not in available:
            raise ValueError(f"Unknown dataset: {name}")
        item = available[name]
        if not item["ready"]:
            raise ValueError(
                f"{name}: {item['matched']}/{len(item['images'])} image-label pairs. "
                "Convert or finish missing labels first."
            )
        for image in item["images"]:
            if not image.resolve().is_relative_to(root):
                raise ValueError(f"Image resolves outside the root: {image}")
            stem = name + "__" + image.stem
            key = stem.casefold()
            if key in output_names:
                raise ValueError(f"Duplicate output label name: {stem}")
            output_names.add(key)
            label = item["labels_dir"] / (image.stem + ".txt")
            if not label.resolve().is_relative_to(root):
                raise ValueError(f"Label resolves outside the root: {label}")
            boxes = validate_label(label)
            annotation = image.with_suffix(".json")
            json_data = None
            if copy_json and annotation.is_file():
                if not annotation.resolve().is_relative_to(root):
                    raise ValueError(f"JSON resolves outside the root: {annotation}")
                json_data = json.loads(annotation.read_text(encoding="utf-8-sig"))
                json_data["imagePath"] = stem + image.suffix.lower()
            records.append({
                "dataset": name, "image": image, "label": label,
                "stem": stem, "suffix": image.suffix.lower(),
                "boxes": boxes, "json_data": json_data,
            })
    return records


def merge_datasets(root, selected, copy_json=True, progress=None):
    root = Path(root).resolve()
    records = prepare_merge(root, selected, copy_json)
    output_parent = root / "merged_datasets"
    if not output_parent.resolve().is_relative_to(root):
        raise ValueError("Output folder resolves outside the root.")
    output_parent.mkdir(exist_ok=True)
    output = output_parent / (
        datetime.now().strftime("merge_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    )
    output.mkdir()  # Always a new run; never overwrite an earlier dataset.
    images_out = output / "images"
    labels_out = output / "labels"
    images_out.mkdir()
    labels_out.mkdir()
    marker = output / "INCOMPLETE.txt"
    marker.write_text("Merge in progress. Do not train from this directory.\n",
                      encoding="utf-8")
    manifest = []
    hash_groups = {}
    json_count = 0
    for index, record in enumerate(records, 1):
        image_name = record["stem"] + record["suffix"]
        label_name = record["stem"] + ".txt"
        image_dest = images_out / image_name
        label_dest = labels_out / label_name
        shutil.copy2(record["image"], image_dest)
        shutil.copy2(record["label"], label_dest)
        if (image_dest.stat().st_size != record["image"].stat().st_size
                or label_dest.read_bytes() != record["label"].read_bytes()):
            raise IOError(f"Copy verification failed: {image_name}")
        if record["json_data"] is not None:
            image_dest.with_suffix(".json").write_text(
                json.dumps(record["json_data"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            json_count += 1
        # Equal image bytes stay in the dataset, but share a group in the manifest.
        # Keep each duplicate group together when creating train/val splits later.
        digest = hashlib.sha256(image_dest.read_bytes()).hexdigest()
        group = hash_groups.setdefault(digest, len(hash_groups) + 1)
        manifest.append({
            "dataset": record["dataset"],
            "source_image": str(record["image"]),
            "source_label": str(record["label"]),
            "image": "images/" + image_name,
            "label": "labels/" + label_name,
            "boxes": record["boxes"],
            "image_sha256": digest, "duplicate_group": group,
        })
        if progress:
            progress(index, len(records))
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (output / "classes.txt").write_text("person\n", encoding="utf-8")
    duplicate_counts = Counter(row["image_sha256"] for row in manifest)
    summary = {
        "status": "complete", "source_sets": list(selected),
        "images": len(manifest), "labels": len(manifest),
        "person_boxes": sum(row["boxes"] for row in manifest),
        "background_images": sum(row["boxes"] == 0 for row in manifest),
        "json_files_copied": json_count,
        "exact_duplicate_extra_images": sum(n - 1 for n in duplicate_counts.values()),
        "class_names": {"0": "person"}, "split": "not_yet_split",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.txt").write_text(
        "YOLO person dataset\n"
        "images/: copied images; optional X-AnyLabeling JSON files\n"
        "labels/: YOLO labels, class 0 = person\n"
        "manifest.csv: original dataset and file paths; duplicate groups\n"
        "summary.json: counts and selected source datasets\n\n"
        "Next: create train/val splits and data.yaml before training.\n"
        "Keep duplicate images and nearby video frames in the same split.\n"
        "The original source folders were not modified.\n"
        "If JSON was copied, YOLO labels remain the source used for this merge.\n",
        encoding="utf-8",
    )
    marker.unlink()
    return output, summary


def launch_gui(root):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    app = tk.Tk()
    app.title("รวมชุดข้อมูล YOLO — person = 0")
    app.geometry("900x620")
    root_text = tk.StringVar(value=str(root))
    copy_json = tk.BooleanVar(value=False)
    status = tk.StringVar(value="เลือกชุดข้อมูล แล้วกดตรวจสอบก่อนรวม")
    table_items = {}

    top = ttk.Frame(app, padding=12)
    top.pack(fill="x")
    ttk.Label(top, text="โฟลเดอร์หลัก").pack(anchor="w")
    ttk.Entry(top, textvariable=root_text).pack(side="left", fill="x", expand=True)

    def choose_root():
        path = filedialog.askdirectory(initialdir=root_text.get())
        if path:
            root_text.set(path)
            refresh()

    ttk.Button(top, text="เลือกโฟลเดอร์", command=choose_root).pack(side="left", padx=5)
    table = ttk.Treeview(
        app, columns=("images", "labels", "location", "status"),
        selectmode="extended",
    )
    table.heading("#0", text="ชุดข้อมูล")
    table.heading("images", text="ภาพ")
    table.heading("labels", text="คู่ label")
    table.heading("location", text="ตำแหน่ง labels")
    table.heading("status", text="สถานะ")
    table.column("#0", width=190)
    for column in ("images", "labels"):
        table.column(column, width=65, anchor="center")
    table.column("location", width=240)
    table.column("status", width=120)
    table.pack(fill="both", expand=True, padx=12, pady=8)
    ttk.Label(
        app, text="กด Ctrl หรือ Shift เพื่อเลือกหลายชุด • เลือกชุดที่มี label ครบให้อัตโนมัติ"
    ).pack(anchor="w", padx=12)
    ttk.Checkbutton(
        app, text="คัดลอก JSON สำหรับเปิดดูใน X-AnyLabeling ด้วย",
        variable=copy_json,
    ).pack(anchor="w", padx=12, pady=8)
    controls = ttk.Frame(app, padding=12)
    controls.pack(fill="x")
    bar = ttk.Progressbar(app, mode="determinate")
    bar.pack(fill="x", padx=12)
    ttk.Label(app, textvariable=status, wraplength=850).pack(
        anchor="w", padx=12, pady=12
    )
    busy = False

    def refresh():
        if busy:
            return
        try:
            items = scan_sets(Path(root_text.get()))
        except Exception as exc:
            messagebox.showerror("อ่านโฟลเดอร์ไม่ได้", str(exc))
            return
        table.delete(*table.get_children())
        table_items.clear()
        ready = []
        for item in items:
            location = (str(item["labels_dir"].relative_to(item["path"]))
                        if item["labels_dir"] else "-")
            row = table.insert("", "end", text=item["name"], values=(
                len(item["images"]), item["matched"], location,
                "พร้อม" if item["ready"] else "labels ไม่ครบ",
            ))
            table_items[row] = item["name"]
            if item["ready"]:
                ready.append(row)
        table.selection_set(ready)
        status.set(f"พบ {len(items)} ชุด • พร้อมรวม {len(ready)} ชุด")

    def select_all():
        if busy:
            return
        if not table.get_children():
            return
        table.selection_set(table.get_children())
        status.set(f"เลือกทั้งหมด {len(table.get_children())} ชุด")

    def select_ready():
        if busy:
            return
        ready = [row for row in table.get_children() if row in table_items and table_items[row] in {
            item["name"] for item in scan_sets(Path(root_text.get())) if item["ready"]
        }]
        table.selection_set(ready)
        status.set(f"เลือกที่พร้อมแล้ว {len(ready)} ชุด")

    def selected_sets():
        return [table_items[row] for row in table.selection()]

    def preview():
        if busy:
            return
        try:
            records = prepare_merge(
                Path(root_text.get()), selected_sets(), copy_json.get()
            )
            status.set(
                f"ตรวจผ่าน: {len(selected_sets())} ชุด • {len(records)} ภาพ • "
                f"{sum(r['boxes'] for r in records)} กรอบ person"
            )
            messagebox.showinfo(
                "ตรวจผ่าน",
                status.get() + "\n\nไฟล์รวมจะอยู่ใน merged_datasets ใต้โฟลเดอร์หลัก",
            )
        except Exception as exc:
            messagebox.showerror("ตรวจไม่ผ่าน", str(exc))

    def finish(result=None, error=None):
        nonlocal busy
        busy = False
        for button in action_buttons:
            button.configure(state="normal")
        if error:
            status.set("รวมไม่สำเร็จ: " + str(error))
            messagebox.showerror(
                "รวมไม่สำเร็จ",
                str(error) + "\nหากมีโฟลเดอร์ INCOMPLETE ให้เก็บไว้ตรวจสอบ "
                "และอย่าใช้เทรน",
            )
        else:
            output, summary = result
            status.set(str(output))
            messagebox.showinfo(
                "รวมเสร็จแล้ว",
                f"{summary['images']} ภาพ / {summary['person_boxes']} กรอบ\n\n"
                f"{output}\n\nขั้นต่อไป: แบ่ง train/val และสร้าง data.yaml",
            )

    def merge():
        nonlocal busy
        if busy:
            return
        chosen_root = Path(root_text.get())
        chosen_sets = selected_sets()
        chosen_json = copy_json.get()
        if not chosen_sets:
            messagebox.showerror("ยังไม่ได้เลือก", "เลือกอย่างน้อยหนึ่งชุดข้อมูล")
            return
        busy = True
        for button in action_buttons:
            button.configure(state="disabled")
        status.set("กำลังตรวจและคัดลอกไฟล์...")
        bar["value"] = 0

        def update(index, total):
            app.after(0, lambda: bar.configure(maximum=total, value=index))

        def worker():
            try:
                result = merge_datasets(chosen_root, chosen_sets, chosen_json, update)
            except Exception as exc:
                app.after(0, lambda error=exc: finish(error=error))
            else:
                app.after(0, lambda: finish(result=result))

        threading.Thread(target=worker, daemon=True).start()

    action_buttons = [
        ttk.Button(controls, text="สแกนใหม่", command=refresh),
        ttk.Button(controls, text="เลือกทั้งหมด", command=select_all),
        ttk.Button(controls, text="เลือกที่พร้อม", command=select_ready),
        ttk.Button(controls, text="ตรวจสอบ", command=preview),
        ttk.Button(controls, text="รวมชุดที่เลือก", command=merge),
    ]
    for button in action_buttons:
        button.pack(side="left", padx=5)

    def close():
        if busy:
            messagebox.showinfo("กำลังทำงาน", "รอให้รวมไฟล์เสร็จก่อนปิดหน้าต่าง")
        else:
            app.destroy()

    app.protocol("WM_DELETE_WINDOW", close)
    refresh()
    app.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="Validate only")
    mode.add_argument("--merge", action="store_true", help="Create a merged copy")
    parser.add_argument("--sets", nargs="+", help="Source dataset folder names")
    parser.add_argument("--json", dest="copy_json", action="store_true",
                        help="Copy X-AnyLabeling JSON files into the merged dataset")
    parser.add_argument("--no-json", dest="copy_json", action="store_false",
                        help="Do not copy JSON files (default)")
    parser.set_defaults(copy_json=False)
    args = parser.parse_args()
    if not args.preview and not args.merge:
        launch_gui(args.root)
        return
    selected = args.sets or [
        item["name"] for item in scan_sets(args.root) if item["ready"]
    ]
    if args.preview:
        records = prepare_merge(args.root, selected, args.copy_json)
        print(json.dumps({
            "selected_sets": selected, "images": len(records),
            "person_boxes": sum(r["boxes"] for r in records),
            "output_parent": str(args.root / "merged_datasets"),
            "json_copied": args.copy_json,
        }, ensure_ascii=False, indent=2))
    else:
        output, summary = merge_datasets(args.root, selected, args.copy_json)
        print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()