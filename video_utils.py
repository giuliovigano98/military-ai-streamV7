import os
import cv2
from collections import Counter, defaultdict
from typing import Dict, Tuple

from model_utils import classify_equipment, predict_threat, draw_overlay

def process_video(input_path: str, mobilenet, cnn_tank, cnn_ifv,
                  output_path: str = "output.mp4",
                  sample_every_n_frames: int = 1) -> Tuple[str, Dict]:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Video non trovato: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {input_path}")

    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    equip_counter = Counter()
    threat_counter = Counter()
    threat_probs_sum = defaultdict(float)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if sample_every_n_frames > 1 and (frame_idx % sample_every_n_frames != 0):
            writer.write(frame)
            continue

        eq_label, eq_conf, p_ifv, p_tank = classify_equipment(frame, mobilenet)
        equip_counter[eq_label] += 1

        thr_label, thr_conf, low, med, high = predict_threat(frame, eq_label, cnn_tank, cnn_ifv)
        threat_counter[thr_label] += 1
        threat_probs_sum["Low"]    += float(low)
        threat_probs_sum["Medium"] += float(med)
        threat_probs_sum["High"]   += float(high)

        out_frame = draw_overlay(frame, eq_label, eq_conf, thr_label, thr_conf, low, med, high)
        writer.write(out_frame)

    cap.release()
    writer.release()

    total_frames = sum(equip_counter.values())
    if total_frames == 0:
        raise RuntimeError("Nessun frame letto. Video vuoto o corrotto?")

    final_report = {
        "total_frames": total_frames,
        "equipment_counts": dict(equip_counter),
        "threat_counts": dict(threat_counter),
        "avg_threat_probs": {k: (v / total_frames) for k, v in threat_probs_sum.items()}
    }
    return output_path, final_report

def draw_minimal_overlay(frame, box, label: str, conf: float):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    text = f"{label}  {int(max(0.0, min(1.0, conf)) * 100)}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    pad = 4
    chip = np.zeros((th + pad * 2, tw + pad * 2, 3), dtype=np.uint8)
    chip[:] = (30, 30, 30)
    cv2.putText(chip, text, (pad, th + pad - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y0 = max(0, y1 - (th + pad * 2 + 6))
    x0 = max(0, x1)
    h, w = chip.shape[:2]
    roi = frame[y0:y0 + h, x0:x0 + w]
    if roi.shape[:2] == (h, w):
        cv2.addWeighted(roi, 0.4, chip, 0.6, 0, roi)
    bar_w = int((x2 - x1) * max(0.0, min(1.0, conf)))
    yb = min(frame.shape[0] - 3, y2 + 6)
    cv2.rectangle(frame, (x1, yb), (x1 + bar_w, yb + 2), (255, 255, 255), -1)