"""Prepare a merged YOLO dataset for Ultralytics object detection with auto-deduplication (Keep Newest Label).

CLI examples:
  python prepare_ultralytics.py
  python prepare_ultralytics.py "E:/path/to/merged_dataset"
  python prepare_ultralytics.py "E:/path/to/merged_dataset" --no-gui --val-fraction 0.2
  python prepare_ultralytics.py "E:/path/to/merged_dataset" --no-gui --per-set-val 01_base=20 02_fix=10

Features:
  - Deduplicates identical image SHA256 hashes by keeping only the NEWEST label/image version (based on st_mtime).
  - Provides GUI for adjusting per-folder Train/Val split percentages.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path resolves outside the input dataset: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_label(path):
    count = 0
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"{path.name}:{number}: expected class 0 and four coordinates")
        values = list(map(float, fields[1:]))
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f"{path.name}:{number}: invalid normalized coordinates")
        cx, cy, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError(f"{path.name}:{number}: empty bounding box")
        eps = 0.000001
        if (cx - width / 2 < -eps or cy - height / 2 < -eps
                or cx + width / 2 > 1 + eps or cy + height / 2 > 1 + eps):
            raise ValueError(f"{path.name}:{number}: box extends beyond the image")
        count += 1
    return count


def label_version(mtime_ns, dataset):
    date = re.search(r"\d{4}-\d{2}-\d{2}", dataset)
    return mtime_ns, date.group() if date else "", dataset.casefold()


def deduplicate_rows(rows, source, *, importing=False):
    """Find duplicate images by SHA256 and keep ONLY the version with the newest file modification time."""
    hash_to_row = {}
    duplicates_removed = 0
    processed_rows = []

    for row in rows:
        image = inside(source, row["image"])
        label = inside(source, row["label"]) if row["label"] or not importing else None
        if image.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Unexpected image type: {image.name}")
        if label and (label.suffix.lower() != ".txt" or image.stem.casefold() != label.stem.casefold()):
            raise ValueError(f"Unexpected label for {image.name}: {label.name}")
        boxes = check_label(label) if label else 0
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        mtime_ns = label.stat().st_mtime_ns if label else 0
        mtime = mtime_ns / 1_000_000_000

        row_data = dict(row)
        row_data["boxes"] = boxes
        row_data["image_sha256"] = digest
        row_data["_image"] = image
        row_data["_label"] = label
        row_data["_mtime_ns"] = mtime_ns
        row_data["_mtime_str"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        processed_rows.append(row_data)

    # คัดเลือกเฉพาะเวอร์ชันที่ mtime ล่าสุดเมื่อพบ SHA256 ซ้ำกัน
    for row in processed_rows:
        digest = row["image_sha256"]
        if digest in hash_to_row:
            duplicates_removed += 1
            existing = hash_to_row[digest]
            if label_version(row["_mtime_ns"], row["dataset"]) > label_version(existing["_mtime_ns"], existing["dataset"]):
                hash_to_row[digest] = row
        else:
            hash_to_row[digest] = row

    clean_rows = list(hash_to_row.values())

    # ตรวจสอบว่ารูปที่เหลืออยู่ไม่มีชื่อไฟล์ (stem) ซ้ำกัน
    if importing:
        return clean_rows, duplicates_removed
    names = set()
    for row in clean_rows:
        stem = row["_image"].stem.casefold()
        if stem in names:
            raise ValueError(f"Duplicate basename among non-identical images: {row['_image'].name}")
        names.add(stem)

    return clean_rows, duplicates_removed


def ordered_datasets(source, names):
    """Recover import order from saved metadata or legacy dataset history."""
    names = dict.fromkeys(names)
    current = source
    history = []
    visited = set()
    order = []
    while current not in visited:
        visited.add(current)
        try:
            summary = json.loads((current / "summary.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            summary = {}
        if "dataset_order" in summary:
            order = list(summary["dataset_order"])
            break
        if current == source:
            manifest_order = list(names)
        else:
            try:
                with (current / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
                    manifest_order = list(dict.fromkeys(row["dataset"] for row in csv.DictReader(f)))
            except (OSError, KeyError):
                manifest_order = []
        added = summary.get("additional_images_folder")
        if added:
            folder = Path(added)
            latest = folder.parent.name if folder.name.lower() == "images" else folder.name
            if latest in manifest_order:
                manifest_order.remove(latest)
                manifest_order.append(latest)
        history.append(manifest_order)
        parent = summary.get("input_merged_dataset")
        if not parent or not Path(parent).is_dir():
            break
        current = Path(parent).resolve()
    for previous in reversed(history):
        order.extend(name for name in previous if name not in order)
    return list(dict.fromkeys(name for name in [*order, *names] if name in names))


def inspect_manifest(source):
    """Read manifest.csv, resolve SHA256 duplicates by newest mtime, and return unique image rows."""
    source = Path(source).resolve()
    if (source / "INCOMPLETE.txt").exists():
        raise ValueError("The input merge is incomplete (INCOMPLETE.txt exists).")
    manifest_path = source / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.csv not found in {source}")
    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("The manifest contains no images.")

    clean_rows, duplicates_removed = deduplicate_rows(rows, source)
    for row in clean_rows:
        if not (row["_image"].is_relative_to(source / "images")
                and row["_label"].is_relative_to(source / "labels")):
            raise ValueError("Manifest paths must be within images/ and labels/.")
    counts = Counter(row["dataset"] for row in clean_rows)
    return source, clean_rows, dict(counts), duplicates_removed


def split_counts(total, val_pct, test_pct):
    if not (0 <= val_pct <= 100 and 0 <= test_pct <= 100 and val_pct + test_pct <= 100.000000001):
        raise ValueError("Val % และ Test % ต้องอยู่ระหว่าง 0–100 และรวมกันไม่เกิน 100%")
    val = round(total * (val_pct / 100))
    test = min(total - val, round(total * (test_pct / 100)))
    return total - val - test, val, test


def build_plan(source, per_set_val_pct=None, default_val_fraction=0.2, seed=42, preserve_existing=False, per_set_test_pct=None, default_test_fraction=0):
    """Validate files, keep newest duplicates, and assign train/val splits based on per-folder percentages."""
    source, clean_rows, counts, duplicates_removed = inspect_split_sources(source)

    if per_set_val_pct is None:
        per_set_val_pct = {ds: default_val_fraction * 100.0 for ds in counts}
    else:
        for ds in counts:
            if ds not in per_set_val_pct:
                per_set_val_pct[ds] = default_val_fraction * 100.0

    for ds, pct in per_set_val_pct.items():
        if not 0.0 <= pct <= 100.0:
            raise ValueError(f"Validation percentage for '{ds}' must be between 0 and 100.")

    # จัดกลุ่มรูปภาพที่ไม่ซ้ำตามโฟลเดอร์ และสุ่มแบ่ง Train/Val ตาม %
    rng = random.Random(seed)
    grouped_rows = defaultdict(list)
    for row in clean_rows:
        grouped_rows[row["dataset"]].append(row)

    for ds in sorted(grouped_rows):
        ds_rows = sorted(grouped_rows[ds], key=lambda r: r["_image"].name)
        rng.shuffle(ds_rows)
        test_pct = (per_set_test_pct or {}).get(ds, default_test_fraction * 100)
        train, val, test = split_counts(len(ds_rows), per_set_val_pct[ds], test_pct)
        remaining = {"val": val, "test": test, "train": train}
        unassigned = []
        for row in ds_rows:
            current = row.get("split")
            if preserve_existing and remaining.get(current, 0) > 0:
                remaining[current] -= 1
            else:
                unassigned.append(row)
        targets = (split for split, count in remaining.items() for _ in range(count))
        for row, split in zip(unassigned, targets):
            row["split"] = split

    train_total = sum(1 for r in clean_rows if r["split"] == "train")
    val_total = sum(1 for r in clean_rows if r["split"] == "val")
    if train_total == 0 or val_total == 0:
        raise ValueError(f"Both train and val must have at least 1 image (train={train_total}, val={val_total}).")

    return source, clean_rows, per_set_val_pct, duplicates_removed


def summarize(rows, per_set_val_pct, duplicates_removed=0, seed=42):
    result = {
        "images": len(rows),
        "person_boxes": sum(row["boxes"] for row in rows),
        "names": {"0": "person"},
        "split_policy": "per-folder percentage split with auto-deduplication (kept newest labels)",
        "random_seed": seed,
        "old_duplicate_images_removed": duplicates_removed,
        "exact_duplicate_images_across_splits": 0,
        "per_folder_val_percent": {k: round(v, 2) for k, v in sorted(per_set_val_pct.items())},
        "splits": {},
    }
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        ds_counts = Counter(row["dataset"] for row in subset)
        result["splits"][split] = {
            "images": len(subset),
            "labels": len(subset),
            "person_boxes": sum(row["boxes"] for row in subset),
            "source_sets": sorted(ds_counts.keys()),
            "source_set_counts": dict(sorted(ds_counts.items())),
        }
    return result


def prepare(source, per_set_val_pct=None, default_val_fraction=0.2, seed=42, preview=False, preserve_existing=False, per_set_test_pct=None, default_test_fraction=0):
    source, clean_rows, per_set_val_pct, duplicates_removed = build_plan(
        source=source,
        per_set_val_pct=per_set_val_pct,
        default_val_fraction=default_val_fraction,
        seed=seed,
        preserve_existing=preserve_existing, per_set_test_pct=per_set_test_pct,
        default_test_fraction=default_test_fraction,
    )
    summary = summarize(clean_rows, per_set_val_pct, duplicates_removed, seed=seed)
    summary["per_folder_test_percent"] = {ds: (per_set_test_pct or {}).get(ds, default_test_fraction * 100) for ds in per_set_val_pct}
    return write_dataset(source, clean_rows, summary, preview)


def discover_training_images(folder):
    images = Path(folder).resolve()
    if not images.is_dir():
        raise ValueError(f"ไม่พบโฟลเดอร์ {images}")
    pairs = []
    for image in sorted(images.rglob("*")):
        if image.suffix.lower() not in EXTENSIONS or not image.is_file():
            continue
        parent = image.parent
        txt = image.with_suffix(".txt").name
        candidates = [parent / "yolo_labels" / txt, parent / "labels" / txt, parent / txt]
        for ancestor in (parent, *parent.parents):
            if ancestor.name.lower() == "images":
                for label_dir in ("labels", "yolo_labels"):
                    candidates.append(ancestor.parent / label_dir / image.relative_to(ancestor).with_suffix(".txt"))
                break
            if ancestor == images.parent:
                break
        matches = list(dict.fromkeys(path for path in candidates if path.is_file()))
        if len(matches) > 1 and len({path.read_bytes() for path in matches}) > 1:
            raise ValueError(f"พบป้ายกำกับหลายไฟล์ที่ข้อมูลต่างกันสำหรับ {image}: {matches}")
        pairs.append((image, matches[0] if matches else None))
    if not pairs:
        raise ValueError(f"ไม่พบภาพใน {images} หรือโฟลเดอร์ย่อย (รองรับ {', '.join(sorted(EXTENSIONS))})")
    return pairs


def append_training_images(source, images, preview=False, *, missing="error", pairs=None):
    images = Path(images).resolve()
    pairs = discover_training_images(images) if pairs is None else pairs
    unlabeled = sum(label is None for _, label in pairs)
    if missing not in ("error", "skip", "background"):
        raise ValueError("Unknown missing-label policy")
    if unlabeled and missing == "error":
        raise ValueError(f"พบ {unlabeled} ภาพที่ไม่มี YOLO .txt กรุณาตีกรอบ หรือเลือก skip/background")
    source, rows, _, removed = inspect_manifest(source)
    if any(r.get("split") not in ("train", "val", "test") for r in rows):
        raise ValueError("Select a prepared dataset with train/val assignments.")
    dataset = images.parent.name if images.name.lower() == "images" else images.name
    by_hash = {r["image_sha256"]: r for r in rows}
    stems = {r["_image"].stem.casefold() for r in rows}
    additions = []
    paths = [image for image, _ in pairs] + [label for _, label in pairs if label]
    root = Path(os.path.commonpath([path.parent for path in paths]))
    for image, label in pairs:
        if label is None and missing == "skip":
            continue
        additions.append(dict(dataset=dataset, source_image=str(image), source_label=str(label) if label else "",
                              image=image.relative_to(root).as_posix(),
                              label=label.relative_to(root).as_posix() if label else ""))
    if not additions:
        raise ValueError("ภาพทั้งหมดรอตีกรอบ ยังไม่มีภาพพร้อมเพิ่มเข้า Train")
    incoming, batch_duplicates = deduplicate_rows(additions, root, importing=True)
    overlap = 0
    updated = 0
    for r in incoming:
        previous = by_hash.get(r["image_sha256"])
        if previous:
            overlap += 1
            if r["_label"] is not None and label_version(r["_mtime_ns"], r["dataset"]) > label_version(previous["_mtime_ns"], previous["dataset"]):
                r.update(split=previous["split"], _stem=previous["_image"].stem)
                by_hash[r["image_sha256"]] = r
                updated += 1
            continue
        stem = dataset + "__" + r["_image"].stem
        if stem.casefold() in stems:
            stem += "__" + r["image_sha256"]
        if stem.casefold() in stems:
            raise ValueError(f"Output name collision: {stem}")
        stems.add(stem.casefold())
        r.update(split="train", _stem=stem)
        by_hash[r["image_sha256"]] = r
    result = list(by_hash.values())
    if not {"train", "val"}.issubset(r["split"] for r in result):
        raise ValueError("Both train and val must contain images.")
    summary = summarize(result, {}, removed + batch_duplicates + overlap)
    summary.pop("random_seed")
    summary.pop("per_folder_val_percent")
    summary.update(split_policy="Preserve base splits and add unseen images to train",
                   additional_images_folder=str(images), incoming_images=len(pairs),
                   missing_labels=unlabeled, missing_label_policy=missing,
                   new_train_images=len(incoming) - overlap, matches_in_base=overlap,
                   updated_existing_labels=updated)
    return write_dataset(source, result, summary, preview)


def write_dataset(source, clean_rows, summary, preview):
    names = list(dict.fromkeys(row["dataset"] for row in clean_rows))
    added = summary.get("additional_images_folder")
    if added:
        folder = Path(added)
        latest = folder.parent.name if folder.name.lower() == "images" else folder.name
        if latest in names:
            names.remove(latest)
            names.append(latest)
    if added:
        with (source / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            base_order = ordered_datasets(source, (row["dataset"] for row in csv.DictReader(stream)))
        summary["dataset_order"] = list(dict.fromkeys(name for name in [*base_order, *names] if name in names))
    else:
        summary["dataset_order"] = ordered_datasets(source, names)
    if preview:
        return None, summary

    output = source.parent / (
        datetime.now().strftime("DATA-Ultralytics_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    )
    output.mkdir()
    marker = output / "INCOMPLETE.txt"
    marker.write_text("Preparation in progress. Do not train yet.\n", encoding="utf-8")

    for category in ("images", "labels"):
        for split in ("train", "val", "test"):
            if split == "test" and not summary["splits"]["test"]["images"]:
                continue
            (output / category / split).mkdir(parents=True)

    manifest = []
    for row in clean_rows:
        image = row["_image"]
        label = row["_label"]
        split = row["split"]
        stem = row.get("_stem", image.stem)
        image_rel = f"images/{split}/{stem}{image.suffix}"
        label_rel = f"labels/{split}/{stem}.txt"
        dest_image = output / image_rel
        dest_label = output / label_rel
        shutil.copy2(image, dest_image)
        if label is None:
            dest_label.write_bytes(b"")
            os.utime(dest_label, ns=(0, 0))
        else:
            shutil.copy2(label, dest_label)
        if hashlib.sha256(dest_image.read_bytes()).hexdigest() != row["image_sha256"]:
            raise IOError(f"Image verification failed: {dest_image.name}")
        if dest_label.read_bytes() != (label.read_bytes() if label else b""):
            raise IOError(f"Label verification failed: {dest_label.name}")

        annotation = image.with_suffix(".json")
        if annotation.is_file():
            data = json.loads(annotation.read_text(encoding="utf-8-sig"))
            data["imagePath"] = dest_image.name
            dest_image.with_suffix(".json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        saved = {key: value for key, value in row.items() if not key.startswith("_")}
        saved["merged_image"] = row["image"]
        saved["merged_label"] = row["label"]
        saved["image"] = image_rel
        saved["label"] = label_rel
        saved["last_modified"] = row["_mtime_str"]
        manifest.append(saved)

    test_line = "test: images/test\n" if summary["splits"]["test"]["images"] else ""
    yaml = (
        "# Ultralytics object detection dataset\n"
        "# Reference: https://docs.ultralytics.com/datasets/detect/\n"
        f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"{test_line}nc: 1\n"
        "names:\n"
        "  0: person\n"
    )
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    (output / "classes.txt").write_text("person\n", encoding="utf-8")

    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for item in manifest for key in item)))
        writer.writeheader()
        writer.writerows(manifest)

    summary["status"] = "complete"
    summary["input_merged_dataset"] = str(source)
    summary["output_dataset"] = str(output)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    training_command = (
        'yolo detect train model=yolo11n.pt '
        f'data="{output / "data.yaml"}" epochs=100 imgsz=640 batch=8 workers=0 '
        f'project="{output / "runs"}"'
    )
    (output / "README.txt").write_text(
        "Ultralytics YOLO person dataset\n\n"
        "images/train, images/val: images\n"
        "labels/train, labels/val: paired YOLO labels\n"
        "data.yaml: dataset paths and class 0 = person\n"
        "manifest.csv: source filenames and split assignments\n"
        "summary.json: split counts and policy\n\n"
        "Training command (requires an existing Ultralytics installation):\n"
        + training_command + "\n\n"
        "Source data was copied with duplicate removal (kept newest file versions).\n"
        f"Split policy: {summary['split_policy']}.\n"
        "If this directory is moved, update path in data.yaml.\n",
        encoding="utf-8",
    )
    marker.unlink()
    return output, summary


def import_status(summary):
    folder = Path(summary["additional_images_folder"])
    dataset = folder.parent.name if folder.name.lower() == "images" else folder.name
    train = summary["splits"]["train"]["source_set_counts"].get(dataset, 0)
    val = summary["splits"]["val"]["source_set_counts"].get(dataset, 0)
    test = summary["splits"].get("test", {}).get("source_set_counts", {}).get(dataset, 0)
    return (
        f"ชุดที่นำเข้าล่าสุด {dataset}  •  ใช้ใน Train {train} ภาพ / Val {val} ภาพ / Test {test} ภาพ\n"
        f"เพิ่มภาพใหม่ {summary['new_train_images']} ภาพ  •  "
        f"ใช้ label จาก {dataset} กับภาพเดิม {summary['updated_existing_labels']} ภาพ"
    )


def inspect_split_sources(source):
    source, rows, counts, removed = inspect_manifest(source)
    if source.parent.name.casefold() != "merged_datasets" or any(
        row.get("split") not in ("train", "val", "test") for row in rows
    ):
        return source, rows, counts, removed

    by_hash = {row["image_sha256"]: row for row in rows}
    winners = {}
    for directory in sorted(source.parent.parent.iterdir()):
        images = directory / "images"
        if not images.is_dir():
            continue
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in EXTENSIONS or not image.is_file():
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            base = by_hash.get(digest)
            if base is None:
                continue
            label = images / "yolo_labels" / (image.stem + ".txt")
            if not label.is_file():
                label = directory / "labels" / (image.stem + ".txt")
            if not label.is_file():
                continue
            version = label_version(label.stat().st_mtime_ns, directory.name)
            best = winners[digest][0] if digest in winners else label_version(base["_mtime_ns"], base["dataset"])
            if version > best:
                winners[digest] = version, directory, image, label

    stems = {row["_image"].stem.casefold() for h, row in by_hash.items() if h not in winners}
    for digest, (version, directory, image, label) in winners.items():
        stem = directory.name + "__" + image.stem
        while stem.casefold() in stems:
            stem += "__" + digest
        stems.add(stem.casefold())
        by_hash[digest] = dict(
            dataset=directory.name, source_image=str(image), source_label=str(label),
            image=image.relative_to(directory).as_posix(), label=label.relative_to(directory).as_posix(),
            split=by_hash[digest]["split"], boxes=check_label(label), image_sha256=digest,
            _image=image, _label=label, _stem=stem, _mtime_ns=version[0],
            _mtime_str=datetime.fromtimestamp(version[0] / 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S"),
        )
    rows = list(by_hash.values())
    counts = Counter(row["dataset"] for row in rows)
    return source, rows, dict(counts), removed + len(winners)


def format_percent(value):
    return f"{value:.12f}".rstrip("0").rstrip(".")


class DatasetSplitterGUI:
    def __init__(self, initial_source=None, default_val_pct=20):
        self.root = tk.Tk()
        self.root.title("Ultralytics YOLO Dataset Splitter (Auto-Deduplicate & Per-Folder GUI)")
        self.root.geometry("950x660")
        self.root.minsize(800, 520)

        self.source = None
        self.counts = {}
        self.duplicates_removed = 0
        self.folder_controls = {}
        self._syncing = False

        self.seed_var = tk.StringVar(value="42")
        self.import_status_var = tk.StringVar()
        self.source_var = tk.StringVar(value="ยังไม่ได้เลือกโฟลเดอร์")
        self.total_summary_var = tk.StringVar(value="กรุณาเลือกโฟลเดอร์ merge ที่มี manifest.csv")

        self._build_ui(default_val_pct)

        if initial_source:
            self.load_dataset(initial_source, default_val_pct)
        else:
            self.root.after(100, lambda: self.choose_folder(default_val_pct))

    def _build_ui(self, default_val_pct):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        # --- Top Frame ---
        top_frame = ttk.LabelFrame(self.root, text="1. โฟลเดอร์ชุดข้อมูลต้นทาง (Merged Dataset)", padding=10)
        top_frame.pack(fill="x", padx=12, pady=(10, 6))

        path_row = ttk.Frame(top_frame)
        path_row.pack(fill="x")
        ttk.Entry(path_row, textvariable=self.source_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ttk.Button(
            path_row,
            text="เลือกโฟลเดอร์...",
            command=lambda: self.choose_folder(default_val_pct),
        ).pack(side="right")

        ttk.Button(path_row, text="เพิ่มภาพเข้า Train จากโฟลเดอร์...", command=self.append_folder).pack(side="right", padx=4)

        ttk.Label(top_frame, textvariable=self.import_status_var, foreground="#1a5fb4").pack(anchor="w")

        preset_row = ttk.Frame(top_frame)
        preset_row.pack(fill="x", pady=(8, 0))
        ttk.Label(preset_row, text="ตั้งค่า Val (%) ทุกโฟลเดอร์พร้อมกัน:").pack(side="left")
        for pct in (0, 10, 20, 30, 50):
            label = f"{pct}% (Train ทั้งหมด)" if pct == 0 else f"{pct}%"
            ttk.Button(
                preset_row,
                text=label,
                command=lambda p=pct: self.apply_preset_all(p),
            ).pack(side="left", padx=4)

        ttk.Label(preset_row, text="Random Seed:").pack(side="right", padx=(10, 4))
        ttk.Entry(preset_row, textvariable=self.seed_var, width=8).pack(side="right")

        # --- Middle Frame ---
        mid_frame = ttk.LabelFrame(
            self.root,
            text="2. ปรับสัดส่วน Val / Test (%) แยกตามโฟลเดอร์ (คัดเฉพาะรูปภาพเวอร์ชันใหม่ล่าสุดอัตโนมัติ)",
            padding=10,
        )
        mid_frame.pack(fill="both", expand=True, padx=12, pady=6)

        header = ttk.Frame(mid_frame)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="ชื่อโฟลเดอร์ (dataset)", width=28, font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Label(header, text="ภาพที่ไม่ซ้ำ", width=12, anchor="center", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Label(header, text="สัดส่วน Val / Test (%)", width=30, anchor="center", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Label(header, text="หากแบ่งใหม่ (Train / Val / Test)", anchor="w", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(10, 0))

        ttk.Separator(mid_frame, orient="horizontal").pack(fill="x", pady=(0, 4))

        canvas_container = ttk.Frame(mid_frame)
        canvas_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width),
        )
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Bottom Frame ---
        bot_frame = ttk.LabelFrame(self.root, text="3. สรุปยอดรวมและดำเนินการ", padding=10)
        bot_frame.pack(fill="x", padx=12, pady=(6, 12))

        summary_label = ttk.Label(
            bot_frame,
            textvariable=self.total_summary_var,
            font=("Segoe UI", 10, "bold"),
            foreground="#1a5fb4",
        )
        summary_label.pack(anchor="w", pady=(0, 8))

        action_row = ttk.Frame(bot_frame)
        action_row.pack(fill="x")

        ttk.Label(
            action_row,
            text="✓ ระบบจะเลือกใช้ป้ายกรอบ (Label) เวอร์ชันที่แก้ไขล่าสุดโดยอัตโนมัติ",
            foreground="#2e7d32",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        btn_create = tk.Button(
            action_row,
            text="สร้างชุดข้อมูล Ultralytics YOLO",
            bg="#2e7d32",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=5,
            cursor="hand2",
            command=self.run_prepare,
        )
        btn_create.pack(side="right", padx=(8, 0))

        ttk.Button(
            action_row,
            text="ตรวจสอบความถูกต้อง (Preview)",
            command=self.run_preview,
        ).pack(side="right")

    def append_folder(self):
        if not self.source:
            messagebox.showinfo("เลือกชุดเดิม", "เลือกชุด Ultralytics เดิมก่อนเพิ่มภาพ")
            return
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ชุดข้อมูลหรือ images ที่จะเพิ่มเข้า Train",
                                         initialdir=self.source.parent.parent)
        if folder:
            try:
                pairs = discover_training_images(folder)
                count = sum(label is None for _, label in pairs)
                policy = "error"
                if count:
                    choice = messagebox.askyesnocancel(
                        "ภาพที่ไม่มีป้ายกำกับ",
                        f"พบภาพ {len(pairs)} ภาพ มี {count} ภาพที่ไม่มีไฟล์ YOLO .txt\n\n"
                        "Yes — ยืนยันว่าภาพที่ไม่มีป้ายกำกับทุกภาพไม่มีคน และเพิ่มเป็นภาพพื้นหลัง\n"
                        "No — เพิ่มเฉพาะภาพที่มีไฟล์ป้ายกำกับ\n"
                        "Cancel — กลับไปตรวจภาพหรือตีกรอบก่อน", default="cancel")
                    if choice is None:
                        return
                    if not choice and count == len(pairs):
                        messagebox.showinfo("รอตีกรอบ", "ยังไม่มีภาพที่มีไฟล์ YOLO .txt ให้เพิ่มเข้า Train")
                        return
                    policy = "background" if choice else "skip"
                output, summary = append_training_images(self.source, folder, missing=policy, pairs=pairs)
                self.load_dataset(output)
                messagebox.showinfo("รวมชุดข้อมูลสำเร็จ", f"{import_status(summary)}\n\n{output}")
            except Exception as exc:
                messagebox.showerror("รวมชุดข้อมูลไม่สำเร็จ", str(exc))

    def choose_folder(self, default_val_pct=20):
        initial = Path(__file__).resolve().parents[1] / "merged_datasets"
        if not os.path.isdir(initial):
            initial = os.getcwd()
        selected = filedialog.askdirectory(
            title="เลือกโฟลเดอร์ merge ที่มี manifest.csv",
            initialdir=initial,
        )
        if selected:
            self.load_dataset(selected, default_val_pct)

    def load_dataset(self, folder_path, default_val_pct=20):
        try:
            source, rows, counts, duplicates_removed = inspect_split_sources(folder_path)
            counts = {name: counts[name] for name in ordered_datasets(source, counts)}
            actual = Counter((row["dataset"], row.get("split")) for row in rows)
            summary_path = source / "summary.json"
            status = ""
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                if summary.get("additional_images_folder"):
                    status = import_status(summary)
        except Exception as exc:
            messagebox.showerror("โหลดข้อมูลไม่สำเร็จ", str(exc))
            return

        self.source = source
        self.counts = counts
        self.duplicates_removed = duplicates_removed
        self.source_var.set(str(source))
        self.import_status_var.set(status)

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.folder_controls.clear()

        self.canvas.yview_moveto(0)

        for ds, total in counts.items():
            row_frame = ttk.Frame(self.scrollable_frame, padding=(0, 4))
            row_frame.pack(fill="x")

            name = ds
            train = actual[ds, "train"]
            val = actual[ds, "val"]
            test = actual[ds, "test"]
            if train + val + test:
                name += f"\nในชุดนี้ Train {train} / Val {val} / Test {test}"
            ttk.Label(row_frame, text=name, width=28).pack(side="left")
            ttk.Label(row_frame, text=f"{total:,}", width=12, anchor="center").pack(side="left")

            pct = default_val_pct
            if train + val + test == total:
                pct = 100 * val / total
            val_var = tk.DoubleVar(value=pct)
            text_var = tk.StringVar(value=format_percent(pct))

            scale = ttk.Scale(
                row_frame,
                from_=0,
                to=100,
                orient="horizontal",
                length=85,
            )
            scale.set(pct)
            scale.pack(side="left", padx=(6, 6))

            entry = ttk.Entry(row_frame, textvariable=text_var, width=8, justify="right")
            entry.pack(side="left")
            ttk.Label(row_frame, text="%").pack(side="left", padx=(2, 10))

            test_var = tk.StringVar(value=format_percent(100 * test / total))
            ttk.Label(row_frame, text="Test %").pack(side="left")
            ttk.Entry(row_frame, textvariable=test_var, width=6).pack(side="left")
            info_label = ttk.Label(row_frame, text="")
            info_label.pack(side="left", padx=(6, 0))

            self.folder_controls[ds] = {
                "test_var": test_var,
                "val_var": val_var,
                "text_var": text_var,
                "scale": scale,
                "info_label": info_label,
                "total": total,
            }

            scale.configure(command=lambda v, d=ds: self._on_slider_change(d, v))
            test_var.trace_add("write", lambda *_: self.update_live_counts())
            text_var.trace_add("write", lambda *_, d=ds: self._on_entry_change(d))

        self.update_live_counts()

    def _on_slider_change(self, ds, raw_val):
        if self._syncing:
            return
        self._syncing = True
        try:
            pct = round(float(raw_val), 2)
            ctrl = self.folder_controls[ds]
            ctrl["val_var"].set(pct)
            ctrl["text_var"].set(format_percent(pct))
        finally:
            self._syncing = False
        self.update_live_counts()

    def _on_entry_change(self, ds):
        if self._syncing:
            return
        ctrl = self.folder_controls[ds]
        raw = ctrl["text_var"].get().strip()
        if not raw:
            return
        try:
            pct = float(raw)
            if not 0 <= pct <= 100:
                return
        except ValueError:
            return
        self._syncing = True
        try:
            ctrl["val_var"].set(pct)
            ctrl["scale"].set(pct)
        finally:
            self._syncing = False
        self.update_live_counts()

    def apply_preset_all(self, pct):
        if not self.folder_controls:
            return
        self._syncing = True
        try:
            for ds, ctrl in self.folder_controls.items():
                ctrl["val_var"].set(pct)
                ctrl["text_var"].set(format_percent(pct))
                ctrl["scale"].set(pct)
        finally:
            self._syncing = False
        self.update_live_counts()

    def update_live_counts(self):
        grand_total = 0
        grand_train = 0
        grand_val = 0

        for ds, ctrl in self.folder_controls.items():
            total = ctrl["total"]
            pct = ctrl["val_var"].get()
            try:
                train_cnt, val_cnt, test_cnt = split_counts(total, pct, float(ctrl["test_var"].get()))
            except ValueError:
                ctrl["info_label"].config(text="ตรวจสอบ Val % และ Test %")
                self.total_summary_var.set(f"ตรวจสอบสัดส่วนของ {ds}")
                return
            ctrl["info_label"].config(text=f"Train: {train_cnt:,}  |  Val: {val_cnt:,}  |  Test: {test_cnt:,}")
            grand_total += total
            grand_train += train_cnt
            grand_val += val_cnt

        if grand_total > 0:
            t_ratio = (grand_train / grand_total) * 100
            v_ratio = (grand_val / grand_total) * 100
            dup_msg = f"   (ตัดภาพเวอร์ชันเก่าออกอัตโนมัติ: {self.duplicates_removed:,} ภาพ)" if self.duplicates_removed > 0 else ""
            self.total_summary_var.set(
                f"ภาพที่ไม่ซ้ำกันทั้งหมด: {grand_total:,} ภาพ   |   "
                f"Train: {grand_train:,} ภาพ ({t_ratio:.1f}%)   |   "
                f"Val: {grand_val:,} ภาพ ({v_ratio:.1f}%)   |   Test: {grand_total-grand_train-grand_val:,} ภาพ\n"
                f"{dup_msg}"
            )

    def _collect_params(self):
        if not self.source or not self.folder_controls:
            raise ValueError("กรุณาเลือกโฟลเดอร์ชุดข้อมูลก่อน")
        try:
            seed = int(self.seed_var.get().strip())
        except ValueError:
            raise ValueError("Random Seed ต้องเป็นตัวเลขจำนวนเต็ม (เช่น 42)")

        per_set_val_pct = {}
        for ds, ctrl in self.folder_controls.items():
            raw = ctrl["text_var"].get().strip()
            try:
                pct = float(raw) if raw else float(ctrl["val_var"].get())
            except ValueError:
                raise ValueError(f"ค่า % ของโฟลเดอร์ '{ds}' ไม่ถูกต้อง")
            if not 0.0 <= pct <= 100.0:
                raise ValueError(f"ค่า % ของโฟลเดอร์ '{ds}' ต้องอยู่ระหว่าง 0 ถึง 100")
            per_set_val_pct[ds] = pct

        test_pcts = {ds: float(ctrl["test_var"].get()) for ds, ctrl in self.folder_controls.items()}
        return seed, per_set_val_pct, test_pcts

    def run_preview(self):
        try:
            seed, per_set_val_pct, test_pcts = self._collect_params()
            _, summary = prepare(
                source=self.source,
                per_set_val_pct=per_set_val_pct, per_set_test_pct=test_pcts,
                seed=seed,
                preserve_existing=True,
                preview=True,
            )
            tr = summary["splits"]["train"]
            va = summary["splits"]["val"]
            msg = (
                f"ตรวจสอบไฟล์ภาพและ Label ผ่านทั้งหมด!\n\n"
                f"• ภาพสุทธิที่ไม่ซ้ำกัน: {summary['images']:,} ภาพ\n"
                f"• ภาพเวอร์ชันเก่าที่ตัดออก: {summary['old_duplicate_images_removed']:,} ภาพ\n"
                f"• Train: {tr['images']:,} ภาพ ({tr['person_boxes']:,} boxes)\n"
                f"• Test: {summary['splits']['test']['images']:,} ภาพ\n"
                f"• Val: {va['images']:,} ภาพ ({va['person_boxes']:,} boxes)\n"
            )
            messagebox.showinfo("ผลการตรวจสอบ (Preview)", msg)
        except Exception as exc:
            messagebox.showerror("พบข้อผิดพลาด", str(exc))

    def run_prepare(self):
        try:
            seed, per_set_val_pct, test_pcts = self._collect_params()
            output, summary = prepare(
                source=self.source,
                per_set_val_pct=per_set_val_pct, per_set_test_pct=test_pcts,
                seed=seed,
                preserve_existing=True,
                preview=False,
            )
            tr = summary["splits"]["train"]
            va = summary["splits"]["val"]
            msg = (
                f"สร้างชุดข้อมูล Ultralytics สำเร็จแล้ว!\n\n"
                f"โฟลเดอร์ปลายทาง:\n{output}\n\n"
                f"• ภาพสุทธิที่ไม่ซ้ำกัน: {summary['images']:,} ภาพ\n"
                f"• ภาพเวอร์ชันเก่าที่ตัดออก: {summary['old_duplicate_images_removed']:,} ภาพ\n"
                f"• Train: {tr['images']:,} ภาพ ({tr['person_boxes']:,} boxes)\n"
                f"• Test: {summary['splits']['test']['images']:,} ภาพ\n"
                f"• Val: {va['images']:,} ภาพ ({va['person_boxes']:,} boxes)\n\n"
                f"ต้องการเปิดโฟลเดอร์ปลายทางเลยหรือไม่?"
            )
            if messagebox.askyesno("เสร็จสมบูรณ์", msg):
                if os.name == "nt":
                    os.startfile(output)
        except Exception as exc:
            messagebox.showerror("สร้างชุดข้อมูลไม่สำเร็จ", str(exc))

    def run(self):
        self.root.mainloop()


def parse_per_set_args(items):
    result = {}
    if not items:
        return None
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --per-set-val format '{item}'. Expected FOLDER=PERCENT")
        name, val = item.split("=", 1)
        result[name.strip()] = float(val.strip())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="Merged folder containing manifest.csv")
    parser.add_argument("--no-gui", action="store_true", help="Run in CLI mode without opening GUI")
    parser.add_argument("--preview", action="store_true", help="Validate without writing (CLI mode)")
    parser.add_argument("--test-fraction", type=float, default=0, help="Test fraction for CLI preparation (0-1)")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Default validation fraction (0-1)")
    parser.add_argument(
        "--per-set-val",
        nargs="+",
        help="Per-folder validation percentage, e.g. 01_base=20 02_fix=10",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling within folders")
    parser.add_argument("--add-images", type=Path, help="Append a labeled images folder to train, preserving base splits")
    parser.add_argument("--missing-labels", choices=("error", "skip", "background"), default="error")
    args = parser.parse_args()
    if args.add_images:
        if not args.source:
            parser.error("source is required with --add-images")
        output, summary = append_training_images(args.source, args.add_images, args.preview, missing=args.missing_labels)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if output:
            print(f"Dataset YAML: {output / 'data.yaml'}")
        return

    if not args.no_gui:
        app = DatasetSplitterGUI(
            initial_source=args.source,
            default_val_pct=int(round(args.val_fraction * 100)),
        )
        app.run()
        return

    if not args.source:
        parser.error("source is required when using --no-gui")

    per_set_val = parse_per_set_args(args.per_set_val)
    output, summary = prepare(
        source=args.source,
        per_set_val_pct=per_set_val,
        default_val_fraction=args.val_fraction, default_test_fraction=args.test_fraction,
        seed=args.seed,
        preview=args.preview,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if output:
        print(f"Dataset YAML: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()