import argparse
import sys
import threading
import time

import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

from yolo_draw import draw_yolo_boxes


class SharedFrame:
    """Thread-safe holder for the latest captured frame."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

    def set(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
        self._new_frame_event.set()

    def get(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()


def capture_loop(cap: cv2.VideoCapture, shared: SharedFrame, stop_event: threading.Event):
    """Continuously reads from the physical camera into the shared buffer."""
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        shared.set(frame)


def make_processor(name: str, **kwargs):
    if name == "raw":
        return lambda f: f
    if name == "mirror":
        return lambda f: cv2.flip(f, 1)
    if name == "grayscale":
        def _gray(f):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return _gray
    if name == "yolo":
        weights_path = kwargs["weights_path"]
        conf = kwargs.get("conf", 0.8)
        box_color = kwargs.get("box_color", (0, 255, 0))
        return lambda f: draw_yolo_boxes(f, weights_path, conf=conf, box_color=box_color)
    raise ValueError(f"Unknown processor: {name}")


def output_loop(device_name: str, processor_name: str, processor_kwargs: dict, shared: SharedFrame,
                 width: int, height: int, fps: int, stop_event: threading.Event):
    """Runs one virtual camera output, pulling from the shared frame buffer."""
    processor = make_processor(processor_name, **processor_kwargs)

    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                  fmt=PixelFormat.BGR,
                                  backend="unitycapture",
                                  device=device_name) as vcam:
            print(f"[{device_name}] started ({processor_name} filter)")

            while not stop_event.is_set():
                frame = shared.get()
                if frame is None:
                    time.sleep(0.01)
                    continue

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                frame = processor(frame)
                vcam.send(frame)
                vcam.sleep_until_next_frame()

    except Exception as e:
        print(f"[{device_name}] ERROR: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Real camera -> multiple virtual cameras")
    parser.add_argument("--camera", type=int, default=0, help="Physical camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    # Map each registered Unity Capture device name to the filter applied for it.
    # Add/remove entries here to match how many devices you registered.
    OUTPUTS = {
        "Unity Video Capture": ("raw", {}),
        "Unity Video Capture #2": ("grayscale", {}),
        "Unity Video Capture #3": ("mirror", {}),
        "Unity Video Capture #4": ("yolo", {
            "weights_path": "runs/detect/runs/elephant/yolo26n/weights/best.pt",
            "conf": 0.4,
            "box_color": (0, 0, 255),
        })
    }
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Physical camera opened at {actual_w}x{actual_h}")

    shared = SharedFrame()
    stop_event = threading.Event()

    # Start the capture thread (single reader of the physical camera).
    cap_thread = threading.Thread(target=capture_loop, args=(cap, shared, stop_event), daemon=True)
    cap_thread.start()

    # Wait for the first frame before starting outputs.
    while shared.get() is None:
        time.sleep(0.05)

    # Start one output thread per virtual camera device.
    threads = []
    for device_name, (processor_name, processor_kwargs) in OUTPUTS.items():
        t = threading.Thread(
            target=output_loop,
            args=(device_name, processor_name, processor_kwargs, shared, actual_w, actual_h, args.fps, stop_event),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print(f"\n{len(OUTPUTS)} virtual cameras running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        cap_thread.join(timeout=2)
        for t in threads:
            t.join(timeout=2)
        cap.release()


if __name__ == "__main__":
    main()