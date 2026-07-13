import cv2
from ultralytics import YOLO

_model_cache = {}


def draw_yolo_boxes(frame, weights_path: str, conf: float = 0.4, box_color=(0, 255, 0)):
    """
    Runs YOLO detection on a frame and draws predicted boxes + labels on it.

    Args:
        frame: a cv2/numpy BGR image (H, W, 3)
        weights_path: path to a trained YOLO .pt file, e.g.
            "runs/elephant/yolo26n_4gb/weights/best.pt"
        conf: minimum confidence (0.0-1.0) for a detection to be shown
        box_color: BGR color tuple for the drawn boxes/labels

    Returns:
        A copy of the frame with detection boxes drawn on it.
    """
    if weights_path not in _model_cache:
        _model_cache[weights_path] = YOLO(weights_path)
    model = _model_cache[weights_path]

    results = model.predict(frame, conf=conf, verbose=False)[0]
    out = frame.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        confidence = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{model.names[cls_id]} {confidence:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    count = len(results.boxes)
    if count > 0:
        count_label = f"Elephants: {count}"
        font_scale = 2.0
        (tw, th), _ = cv2.getTextSize(count_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        margin = 10
        x2_box = out.shape[1] - margin
        x1_box = x2_box - tw - 12
        y1_box = margin
        y2_box = margin + th + 12

        cv2.rectangle(out, (x1_box, y1_box), (x2_box, y2_box), box_color, -1)
        cv2.putText(out, count_label, (x1_box + 6, y2_box - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    return out