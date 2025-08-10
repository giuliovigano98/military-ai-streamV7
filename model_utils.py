# model_utils.py
import os
import re
import cv2
import glob
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict

# ====== COSTANTI COMUNI ======
EQUIPMENT_CLASSES = ["IFV", "Tank"]          # ordine logico per MobileNet
THREAT_ORDER = ["Low", "Medium", "High"]     # ordine logico FISSO per l'app

# Silenzia TF un po'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Gestione GPU (se presente)
try:
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


# =========================
#   AUTO-DISCOVERY MODELLI
# =========================
def _scan_candidates() -> list:
    """Raccoglie tutti i file .keras/.h5 in ./, ./models, ../models."""
    candidates = []
    search_dirs = [
        os.getcwd(),
        os.path.join(os.getcwd(), "models"),
        os.path.abspath(os.path.join(os.getcwd(), "..", "models")),
    ]
    exts = ("*.keras", "*.h5")
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for ext in exts:
            for p in glob.glob(os.path.join(d, ext)):
                candidates.append(os.path.abspath(p))
    # unici mantenendo l'ordine
    return list(dict.fromkeys(candidates))


def _score_name(p: str, *tokens) -> int:
    """Punteggia il filename in base alla presenza di token."""
    name = os.path.basename(p).lower()
    score = 0
    for t in tokens:
        if t in name:
            score += 2
    if "model" in name:
        score += 1
    return score


def _try_load_shape_and_out(model_path: str):
    """Carica il modello per dedurre input (H,W,C) e output dim."""
    try:
        m = tf.keras.models.load_model(model_path, compile=False)
        shp_in = m.inputs[0].shape  # (None, H, W, C)
        H = int(shp_in[1]) if shp_in[1] is not None else -1
        W = int(shp_in[2]) if shp_in[2] is not None else -1
        C = int(shp_in[3]) if len(shp_in) >= 4 and shp_in[3] is not None else 1
        shp_out = m.outputs[0].shape
        out = int(shp_out[-1]) if shp_out[-1] is not None else -1
        return (H, W, C), out, m
    except Exception:
        return None, None, None


def discover_model_paths() -> Dict[str, str]:
    """
    Restituisce dict con percorsi {mobilenet, cnn_tank, cnn_ifv}.
    Strategia:
      1) Heuristics sul nome file
      2) Fallback: inferenza da input shape / output units
    """
    cands = _scan_candidates()
    if not cands:
        raise FileNotFoundError("Nessun modello (.keras/.h5) trovato in ./, ./models, ../models")

    # Heuristics sui nomi
    mobile_scores = [(p, _score_name(p, "mobilenet", "mv2")) for p in cands]
    tank_scores   = [(p, _score_name(p, "tank")) for p in cands]
    ifv_scores    = [(p, _score_name(p, "ifv")) for p in cands]

    mobile_guess = max(mobile_scores, key=lambda x: x[1])[0] if max(mobile_scores, key=lambda x: x[1])[1] > 0 else None
    tank_guess   = max(tank_scores,   key=lambda x: x[1])[0] if max(tank_scores,   key=lambda x: x[1])[1] > 0 else None
    ifv_guess    = max(ifv_scores,    key=lambda x: x[1])[0] if max(ifv_scores,    key=lambda x: x[1])[1] > 0 else None

    # Fallback: deduci da shape
    cache = {}
    def load_once(path):
        if path in cache:
            return cache[path]
        ish, out, m = _try_load_shape_and_out(path)
        cache[path] = (ish, out, m)
        return ish, out, m

    def is_mobilenet_like(ish, out):
        # tipicamente 224x224x3, output 2/1
        return ish and (ish[2] == 3) and (out in (2, 1))

    def is_threat_like(ish, out):
        # tipicamente 128x128x1 o simili, output 3/2/1
        return ish and (ish[2] in (1, 3)) and (out in (3, 2, 1))

    if mobile_guess is None:
        for p in cands:
            ish, out, _ = load_once(p)
            if is_mobilenet_like(ish, out):
                mobile_guess = p
                break

    remaining = [p for p in cands if p != mobile_guess]
    # prova a usare nome prima
    if tank_guess is None:
        tank_by_name = [p for p in remaining if re.search(r"tank", os.path.basename(p), re.I)]
        if tank_by_name:
            tank_guess = tank_by_name[0]
    if ifv_guess is None:
        ifv_by_name = [p for p in remaining if re.search(r"ifv", os.path.basename(p), re.I)]
        if ifv_by_name:
            ifv_guess = ifv_by_name[0]

    # se ancora mancano, prendi due threat-like a caso
    if tank_guess is None or ifv_guess is None:
        threats = []
        for p in remaining:
            ish, out, _ = load_once(p)
            if is_threat_like(ish, out):
                threats.append(p)
        threats = [p for p in threats if p not in (tank_guess, ifv_guess)]
        while (tank_guess is None or ifv_guess is None) and threats:
            p = threats.pop(0)
            if tank_guess is None:
                tank_guess = p
            elif ifv_guess is None:
                ifv_guess = p

    if not mobile_guess or not tank_guess or not ifv_guess:
        found = "\n".join(f"- {os.path.basename(p)}" for p in cands)
        raise RuntimeError(
            "Non sono riuscito ad assegnare tutti i modelli automaticamente.\n"
            "Trovati:\n" + found + "\n"
            "Rinomina i file includendo 'mobilenet', 'tank', 'ifv' oppure mettili in ./models."
        )

    return {"mobilenet": mobile_guess, "cnn_tank": tank_guess, "cnn_ifv": ifv_guess}


def load_keras_model(path: str) -> tf.keras.Model:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Modello non trovato: {path}")
    return tf.keras.models.load_model(path, compile=False)


def load_models(mobilenet_path: str = None, cnn_tank_path: str = None, cnn_ifv_path: str = None):
    """
    Carica i tre modelli. Se i percorsi mancano, usa auto-discovery.
    Ritorna: (mobilenet, cnn_tank, cnn_ifv, paths_dict)
    """
    if not (mobilenet_path and cnn_tank_path and cnn_ifv_path):
        paths = discover_model_paths()
        mobilenet_path = mobilenet_path or paths["mobilenet"]
        cnn_tank_path  = cnn_tank_path  or paths["cnn_tank"]
        cnn_ifv_path   = cnn_ifv_path   or paths["cnn_ifv"]

    mobilenet = load_keras_model(mobilenet_path)
    cnn_tank  = load_keras_model(cnn_tank_path)
    cnn_ifv   = load_keras_model(cnn_ifv_path)

    return mobilenet, cnn_tank, cnn_ifv, {
        "mobilenet": mobilenet_path,
        "cnn_tank": cnn_tank_path,
        "cnn_ifv": cnn_ifv_path
    }


# =========================
#   PREPROCESS DINAMICO
# =========================
def _model_input_hwC(model) -> Tuple[int, int, int]:
    """
    Ritorna (H, W, C) attesi dal modello (da model.input_shape).
    Se C non è specificato, assume 1.
    """
    shp = model.input_shape
    if isinstance(shp, list):
        shp = shp[0]
    H = int(shp[1]) if len(shp) > 1 and shp[1] is not None else 224
    W = int(shp[2]) if len(shp) > 2 and shp[2] is not None else 224
    C = int(shp[3]) if len(shp) > 3 and shp[3] is not None else 1
    return H, W, C


def preprocess_to_model(frame_bgr: np.ndarray, model) -> np.ndarray:
    """
    Adatta il frame alla input shape del modello, restituendo (1, H, W, C).
    - C==3: RGB; se (224,224,3) applica MobileNetV2 preprocess_input.
    - C==1: grayscale central-crop + resize + normalizzazione [0,1].
    - Altro: fallback ragionevole.
    """
    H, W, C = _model_input_hwC(model)

    if C == 3:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32)
        # euristica MobileNetV2
        if H == 224 and W == 224:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_pre
            x = mn_pre(x)
        x = np.expand_dims(x, 0)  # (1,H,W,3)
        return x

    if C == 1:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        md = min(h, w)
        sy = (h - md) // 2
        sx = (w - md) // 2
        gray = gray[sy:sy+md, sx:sx+md]
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
        x = gray.astype(np.float32) / 255.0
        x = np.expand_dims(x, -1)  # (H,W,1)
        x = np.expand_dims(x, 0)   # (1,H,W,1)
        return x

    # Fallback per C diverso
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    if C > 3:
        extra = np.zeros((H, W, C - 3), dtype=np.float32)
        x = np.concatenate([x, extra], axis=-1)
    elif C < 3:
        x = x[..., :1]
        x = np.repeat(x, C, axis=-1)
    x = np.expand_dims(x, 0)
    return x


# =========================
#        INFERENZA
# =========================
def classify_equipment(frame_bgr: np.ndarray, mobilenet: tf.keras.Model):
    """
    Restituisce (label_str, conf, p_ifv, p_tank).
    Robusto a output 1/2 unità (sigmoid/softmax).
    """
    x = preprocess_to_model(frame_bgr, mobilenet)
    probs = mobilenet.predict(x, verbose=0)
    probs = np.squeeze(probs).ravel()

    if probs.size == 1:
        p_tank = float(probs[0])
        p_ifv = 1.0 - p_tank
        out = np.array([p_ifv, p_tank], dtype=np.float32)
    elif probs.size >= 2:
        out = probs[:2].astype(np.float32)
    else:
        raise ValueError(f"Output inaspettato da mobilenet: shape {probs.shape}")

    idx = int(np.argmax(out))
    label = EQUIPMENT_CLASSES[idx]
    conf = float(out[idx])
    return label, conf, float(out[0]), float(out[1])


def infer_trained_order_for_3(_pred_vec: np.ndarray):
    """
    Se conosci l'ordine reale con cui hai addestrato il modello 3-classi,
    sostituisci e ritorna esattamente quella lista, es.: ['Low','Medium','High'].
    Di default assumiamo ordinamento alfabetico comune: ['High','Low','Medium'].
    """
    return ['High', 'Low', 'Medium']


def predict_threat(frame_bgr: np.ndarray, obj_class: str,
                   cnn_tank: tf.keras.Model, cnn_ifv: tf.keras.Model):
    """
    Output: (threat_label, threat_conf, low, med, high)
    - 3 classi: mappate via infer_trained_order_for_3
    - 2 classi: assume [Low, High], Medium=0
    - 1 classe: assume sigmoid=prob(High), Low=1-High, Medium=0
    """
    model = cnn_tank if obj_class == "Tank" else cnn_ifv
    x = preprocess_to_model(frame_bgr, model)
    pred = model.predict(x, verbose=0)
    pred = np.squeeze(pred).ravel().astype(np.float32)

    if pred.size == 3:
        trained_order = infer_trained_order_for_3(pred)
        probs_by_name = {cls: float(p) for cls, p in zip(trained_order, pred)}
        low  = probs_by_name.get("Low", 0.0)
        med  = probs_by_name.get("Medium", 0.0)
        high = probs_by_name.get("High", 0.0)
    elif pred.size == 2:
        # Assunzione: [Low, High]
        low, high = map(float, pred)
        med = 0.0
    elif pred.size == 1:
        high = float(pred[0])
        low = 1.0 - high
        med = 0.0
    else:
        raise ValueError(f"Output inatteso da threat model: shape {pred.shape}")

    probs = {"Low": low, "Medium": med, "High": high}
    threat = max(probs, key=probs.get)
    threat_conf = probs[threat]
    return threat, float(threat_conf), float(low), float(med), float(high)


# =========================
#        OVERLAY
# =========================
def draw_overlay(frame, eq_label, eq_conf, thr_label, thr_conf, low, med, high):
    """
    HUD in basso a destra:
    - Dimensioni aumentate
    - Testo leggibile
    - Nessuna sovrapposizione tra testo e barre
    """
    h, w = frame.shape[:2]

    # Scala generale
    SCALE = 0.85  # aumenta/diminuisci per regolare tutto proporzionalmente
    
    # Misure base scalate
    pad        = int(10 * SCALE)
    line_h     = int(20 * SCALE)  # più spazio verticale
    bar_w_max  = int(200 * SCALE)
    bar_h      = max(3, int(5 * SCALE))
    gap_after_text = int(25 * SCALE)  # spazio dopo il testo prima delle barre
    gap_between_bars = int(18 * SCALE)  # spazio tra barre

    # Font size in pixel
    title_px = max(14, int(0.01 * SCALE))
    text_px  = max(12, int(5 * SCALE))
    label_px = max(11, int(5 * SCALE))

    # Altezza e larghezza pannello
    panel_h = pad*2 + line_h*3 + gap_after_text + (bar_h * 3) + (gap_between_bars * 2)
    panel_w = pad*2 + bar_w_max

    # Posizione in basso a destra
    x2 = w - pad
    y2 = h - pad
    x1 = max(0, x2 - panel_w)
    y1 = max(0, y2 - panel_h)

    # Sfondo semi-trasparente
    hud = frame.copy()
    cv2.rectangle(hud, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(hud, 0.35, frame, 0.65, 0, frame)

    # Funzione per scrivere testo
    def put(x, y, text, px, color=(255, 255, 255), thick=1):
        try:
            _put_text(frame, x, y, text, font_px=px, color=color, thickness=thick)
        except NameError:
            scale = max(0.3, float(px) / 30.0)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, scale, color, thick, cv2.LINE_AA)

    # Cursor iniziale
    cursor_x = x1 + pad
    cursor_y = y1 + pad + line_h

    # Titolo
    put(cursor_x, cursor_y, "Threat Level Status", px=title_px)

    # Equipment
    cursor_y += line_h
    eq_txt = f"Equipment: {eq_label}  {int(eq_conf*100)}%"
    put(cursor_x, cursor_y, eq_txt, px=text_px, color=(230, 230, 230))

    # Threat
    cursor_y += line_h
    thr_txt = f"Threat: {thr_label}  {int(thr_conf*100)}%"
    put(cursor_x, cursor_y, thr_txt, px=text_px, color=(230, 230, 230))

    # Spazio extra prima delle barre
    cursor_y += gap_after_text

    # Barre
    def draw_white_bar(y, p, name):
        p = float(max(0.0, min(1.0, p)))
        filled = int(bar_w_max * p)
        cv2.rectangle(frame, (cursor_x, y), (cursor_x + bar_w_max, y + bar_h), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (cursor_x, y), (cursor_x + filled, y + bar_h), (255, 255, 255), -1)
        put(cursor_x, y - max(4, int(5 * SCALE)), name, px=label_px, color=(220, 220, 220))

    draw_white_bar(cursor_y, low, "Low")
    draw_white_bar(cursor_y + bar_h + gap_between_bars, med, "Medium")
    draw_white_bar(cursor_y + 2*(bar_h + gap_between_bars), high, "High")

    return frame
