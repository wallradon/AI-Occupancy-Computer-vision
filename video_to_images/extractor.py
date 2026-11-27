"""Video frame extraction, independent of the desktop interface."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
import threading
from tempfile import mkdtemp

import cv2


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float


def inspect_video(source):
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError('เปิดวิดีโอไม่ได้ กรุณาตรวจสอบไฟล์หรือรูปแบบวิดีโอ')
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not all(math.isfinite(v) and v > 0 for v in (fps, count, width, height)):
            raise ValueError('อ่านข้อมูลความยาวหรือขนาดวิดีโอไม่ได้')
        return VideoInfo(width, height, fps, count / fps)
    finally:
        capture.release()


def output_size(width, height, bounds=None, keep_aspect=True):
    if bounds is None:
        return width, height
    target_w, target_h = bounds
    if not (1 <= target_w <= 16384 and 1 <= target_h <= 16384):
        raise ValueError('ความกว้างและความสูงต้องอยู่ระหว่าง 1–16384 พิกเซล')
    if not keep_aspect:
        return target_w, target_h
    ratio = min(target_w / width, target_h / height)
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def create_output_folder(destination, stem):
    parent = Path(destination)
    parent.mkdir(parents=True, exist_ok=True)
    name = stem + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return Path(mkdtemp(prefix=name + '_', dir=parent))


def extract(source, destination, interval=30, bounds=None, keep_aspect=True,
            image_format='jpg', quality=95, cancel=None, progress=None, *, batch_index=None):
    """Save the first frame at/after each target time using decoder seeking.

    Every run gets a new directory; existing images are never overwritten.
    Duration and seeking precision depend on the video's container and decoder.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('ช่วงเวลาต้องเป็นตัวเลขมากกว่า 0 วินาที')
    if image_format not in ('jpg', 'png'):
        raise ValueError('รองรับเฉพาะ JPG และ PNG')
    if not 1 <= quality <= 100:
        raise ValueError('คุณภาพ JPG ต้องอยู่ระหว่าง 1–100')
    info = inspect_video(source)
    output_size(info.width, info.height, bounds, keep_aspect)
    total = math.ceil(info.duration / interval)
    if batch_index is None:
        folder = create_output_folder(destination, Path(source).stem)
        prefix = ''
    else:
        folder = Path(destination)
        prefix = f'{batch_index:03d}_{Path(source).stem[:80]}_'
    capture = cv2.VideoCapture(str(source))
    saved = 0
    cancel = cancel or threading.Event()
    try:
        if not capture.isOpened():
            raise ValueError('เปิดวิดีโอไม่ได้')
        for index in range(total):
            if cancel.is_set():
                break
            seconds = index * interval
            if index and not capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000):
                raise RuntimeError(f'เลื่อนไปยังเวลา {seconds:.3f} วินาทีไม่ได้')
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'อ่านภาพที่เวลา {seconds:.3f} วินาทีไม่ได้ (บันทึกแล้ว {saved} ภาพ)')
            if cancel.is_set():
                break
            height, width = frame.shape[:2]
            size = output_size(width, height, bounds, keep_aspect)
            if size != (width, height):
                method = cv2.INTER_AREA if size[0] < width else cv2.INTER_CUBIC
                frame = cv2.resize(frame, size, interpolation=method)
            params = [cv2.IMWRITE_JPEG_QUALITY, quality] if image_format == 'jpg' else [cv2.IMWRITE_PNG_COMPRESSION, 3]
            ok, encoded = cv2.imencode('.' + image_format, frame, params)
            if not ok:
                raise RuntimeError('สร้างไฟล์ภาพไม่ได้')
            # Write via Python to support Thai and Unicode paths on Windows.
            filename = f'{prefix}frame_{index + 1:06d}_{seconds:012.3f}s.{image_format}'
            with (folder / filename).open('xb') as stream:
                stream.write(encoded.tobytes())
            saved += 1
            if progress:
                progress(saved, total, seconds, str(folder))
    finally:
        capture.release()
    return folder, saved, cancel.is_set()
