"""Convert X-AnyLabeling rectangle annotations to Ultralytics YOLO labels."""

import json
from pathlib import Path

SOURCE_FOLDER = Path(r"E:\RMUTL\object DET 101\yolo\data-set\2026-11-24\batch_20261124_112311_819169_9x2ks3kx")
OUTPUT_FOLDER = SOURCE_FOLDER / "yolo_labels"
CLASSES = {"person": 0}


def main():
    if not SOURCE_FOLDER.is_dir():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_FOLDER}")

    json_files = sorted(SOURCE_FOLDER.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {SOURCE_FOLDER}")

    # Validate all annotations before writing any output files.
    plans = []
    total_boxes = 0
    for json_file in json_files:
        data = json.loads(json_file.read_text(encoding="utf-8-sig"))
        image_width = data["imageWidth"]
        image_height = data["imageHeight"]
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image dimensions: {json_file.name}")

        lines = []
        for shape in data["shapes"]:
            if shape["shape_type"] != "rectangle":
                raise ValueError(f"Non-rectangle annotation: {json_file.name}")
            if shape["label"] not in CLASSES:
                raise ValueError(
                    f"Unknown class {shape['label']!r}: {json_file.name}"
                )

            class_id = CLASSES[shape["label"]]
            points = shape["points"]
            if len(points) not in (2, 4):
                raise ValueError(f"Invalid rectangle points: {json_file.name}")

            x_min = max(0, min(image_width, min(p[0] for p in points)))
            y_min = max(0, min(image_height, min(p[1] for p in points)))
            x_max = max(0, min(image_width, max(p[0] for p in points)))
            y_max = max(0, min(image_height, max(p[1] for p in points)))

            if not (
                0 <= x_min < x_max <= image_width
                and 0 <= y_min < y_max <= image_height
            ):
                raise ValueError(f"Invalid bounding box: {json_file.name}")

            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # Include difficult annotations, as approved for this dataset.
            lines.append(
                f"{class_id} {x_center:.8f} {y_center:.8f} "
                f"{width:.8f} {height:.8f}"
            )

        plans.append((json_file.stem, lines))
        total_boxes += len(lines)

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    for stem, lines in plans:
        txt_file = OUTPUT_FOLDER / f"{stem}.txt"
        txt_file.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        print(f"{txt_file.name}: {len(lines)} boxes")

    print(f"Done: {len(plans)} label files, {total_boxes} person boxes.")
    print(f"Output: {OUTPUT_FOLDER}")
    print("Original JSON files were preserved.")


if __name__ == "__main__":
    main()