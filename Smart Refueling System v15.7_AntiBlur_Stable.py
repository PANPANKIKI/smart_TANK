import os
import cv2
import re
import time
import json
import queue
import threading
import datetime
import io
import urllib.parse
import numpy as np
import pytesseract
import gspread
import gc
from PIL import Image, ImageDraw, ImageFont
from oauth2client.service_account import ServiceAccountCredentials

# ============================================================
#  Google Drive API
# ============================================================
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    DRIVE_AVAILABLE = True
except ImportError:
    DRIVE_AVAILABLE = False

# ============================================================
#  ลด log รบกวนจาก OpenCV / FFMPEG
# ============================================================
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

# ============================================================
#  CONFIG หลัก
# ============================================================
CREDS_FILE      = "service_account.json"
SHEET_KEY       = "1AMVQ650o1dsiGbdp2W2NoUp7nhc-J8k33nc4Fd0uUes"
DRIVE_FOLDER_ID = "1x3bEJHgQXhQCITnIuw6NHpTuEbp7MRxi"

PLATE_MODEL = "bestLicensePlate.onnx"
FUEL_MODEL  = "bestTANK.onnx"
ROI_FILE    = "roi_config.json"

# ค่าความมั่นใจขั้นต่ำ — ลดลงเพื่อจับรถจอดนิ่ง/ป้ายไกล
PLATE_MIN_CONF = 0.003
FUEL_MIN_CONF  = 0.003

# เงื่อนไขทะเบียน
PLATE_MIN_CHARS = 5       # สำรองไว้ใช้อ้างอิง — เกณฑ์จริงใช้ looks_like_plate() (5-7 หลัก, pattern XX-XXXX)
SAVE_COOLDOWN   = 12.0    # วินาที ป้องกันบันทึกซ้ำ

# เงื่อนไขมิเตอร์
OCR_WINDOW_SIZE  = 15
FUEL_MIN_VOTES   = 3
FUEL_STABLE_SEC  = 4.0
FUEL_MIN_CHARS   = 4      # OCR มิเตอร์ต้องได้ ≥ 4 ตัวเลข

# ขนาดหน้าจอแสดงผล
CELL_W, CELL_H = 640, 360

# เปิด/ปิด debug
DEBUG_UI    = True
DEBUG_PRINT = True

# ขนาดคิวส่งงาน cloud
CLOUD_QUEUE_MAX = 20

# สถานะการทำงาน
STATE_WAIT_PLATE = "WAIT_PLATE"
STATE_WAIT_FUEL  = "WAIT_FUEL"

# Google API scope
SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# กล้องแต่ละตัว
CAM_CONFIG = {
    "CAM1": {
        "user": "admin",
        "pass": "Pp_232323",
        "ip": "192.168.1.150",
        "path": "/stream1",
        "mode": "plate",
        "label": "CAM1",
        "station": "S1",
    },
    "CAM2": {
        "user": "admin",
        "pass": "Pp_282821",
        "ip": "192.168.1.48",
        "path": "/stream1",
        "mode": "fuel",
        "label": "TANK",
        "station": "S2",
    },
}

# ROI ค่าเริ่มต้น
DEFAULT_ROI = {
    "CAM1": [0.05, 0.20, 0.95, 0.75],
    "CAM2": [0.05, 0.40, 0.95, 0.95],
}
roi_cfg = dict(DEFAULT_ROI)

# ============================================================
#  Utility functions
# ============================================================
def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def file_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def dprint(*args):
    if DEBUG_PRINT:
        print("[DEBUG]", *args)

# ============================================================
#  สร้างสถานะเริ่มต้นของแต่ละสถานี
# ============================================================
def _new_station_state():
    return {
        "state":            STATE_WAIT_PLATE,
        "plate_text":       "",
        "plate_img":        None,
        "fuel_last_text":   "",
        "fuel_img":         None,
        "fuel_since":       0.0,
        "fuel_raw_history": [],
        "sheet_row_idx":    None,
        "last_save":        0.0,
        "lock":             threading.Lock(),
    }

station_state = {
    "S1": _new_station_state(),
    "S2": _new_station_state(),
}

# ============================================================
#  ตัวแปรเก็บภาพล่าสุด
# ============================================================
latest_frames = {k: None for k in CAM_CONFIG}
frame_locks   = {k: threading.Lock() for k in CAM_CONFIG}
crop_monitors = {k: None for k in CAM_CONFIG}
crop_locks    = {k: threading.Lock() for k in CAM_CONFIG}

# ============================================================
#  Queue สำหรับส่งงานขึ้น cloud
# ============================================================
cloud_queue = queue.Queue(maxsize=CLOUD_QUEUE_MAX)
_tls        = threading.local()
_font_cache = {}

# ============================================================
#  RTSP URL
# ============================================================
def rtsp_url(cam):
    c = CAM_CONFIG[cam]
    return f"rtsp://{c['user']}:{urllib.parse.quote_plus(c['pass'])}@{c['ip']}:554{c['path']}"

# ============================================================
#  โหลดฟอนต์
# ============================================================
def get_font(size=20):
    if size in _font_cache:
        return _font_cache[size]
    for p in [
        "/home/agentfuel/cam_fuel/TAHOMA.TTF",
        "/home/agentfuel/cam_fuel/TAHOMA.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _font_cache[size] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f

# ============================================================
#  วาดข้อความบนภาพ
# ============================================================
def put_text(img, text, pos, color=(0, 220, 120), size=20):
    if img is None:
        return img
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        ImageDraw.Draw(pil).text(pos, str(text), font=get_font(size), fill=tuple(color))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (int(color[2]), int(color[1]), int(color[0])), 2)
        return img

# ============================================================
#  ตัดภาพ ROI
# ============================================================
def get_roi_crop(img, rx):
    if img is None:
        return None
    H, W = img.shape[:2]
    x1 = int(rx[0] * W); y1 = int(rx[1] * H)
    x2 = int(rx[2] * W); y2 = int(rx[3] * H)
    c = img[y1:y2, x1:x2]
    return c if c.size > 0 else None

# ============================================================
#  โหลด/บันทึก ROI
# ============================================================
def load_roi(reset=False):
    global roi_cfg
    if reset and os.path.exists(ROI_FILE):
        os.remove(ROI_FILE)
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if k in roi_cfg and isinstance(v, list) and len(v) == 4:
                    roi_cfg[k] = v
        except Exception as e:
            print(f"[WARN] load_roi failed: {e}")

def save_roi():
    with open(ROI_FILE, "w", encoding="utf-8") as f:
        json.dump(roi_cfg, f, indent=2, ensure_ascii=False)

# ============================================================
#  โหลดโมเดล ONNX แบบ cache ตาม thread
# ============================================================
def get_net(model_path):
    attr = "net_" + os.path.basename(model_path).replace(".", "_")
    if not hasattr(_tls, attr):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        net = cv2.dnn.readNetFromONNX(model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        setattr(_tls, attr, net)
    return getattr(_tls, attr)

# ============================================================
#  คลาสอ่านกล้อง RTSP แบบ thread แยก
# ============================================================
class RealtimeCapture:
    def __init__(self, url, timeout_sec=8):
        self._url         = url
        self._timeout_sec = timeout_sec
        self._frame       = None
        self._lock        = threading.Lock()
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _open_cap(self):
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout_sec * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout_sec * 1000)
        except Exception:
            pass
        return cap

    def _run(self):
        retry_delay = 2
        while not self._stop.is_set():
            cap = None
            try:
                cap = self._open_cap()
                last_ok = time.time()
                retry_delay = 2
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        last_ok = time.time()
                        with self._lock:
                            self._frame = frame
                    elif time.time() - last_ok > self._timeout_sec:
                        break
                    time.sleep(0.01)
            except Exception:
                pass
            finally:
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                gc.collect()
            if not self._stop.is_set():
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

    def read(self):
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

# ============================================================
#  ทำความสะอาดข้อความ OCR
# ============================================================
def normalize_text(text, mode="plate"):
    if not text:
        return ""
    text = text.upper().strip()
    if mode == "plate":
        # ทะเบียนของระบบนี้เป็นตัวเลขล้วน รูปแบบ XX-XXXX ไม่มีตัวอักษรไทย/อังกฤษ
        text = text.replace("|", "1").replace("/", "1").replace("\\", "1")
        text = text.replace("O", "0").replace("I", "1").replace("Z", "2")
        text = text.replace("S", "5").replace("B", "8")
        text = re.sub(r"[^0-9\-]", "", text)
        return text
    text = text.replace("|", "1").replace("/", "1").replace("\\", "1").replace("O", "0")
    text = re.sub(r"[^0-9.]", "", text)
    if text.count(".") > 1:
        parts = text.split(".")
        text  = parts[0] + "." + "".join(parts[1:])
    return text

# ============================================================
#  ตรวจสอบว่าข้อความหน้าตาเหมือนทะเบียน XX-XXXX หรือไม่
# ============================================================
def looks_like_plate(text):
    """
    ทะเบียนรูปแบบจริงของระบบนี้: เลข 2 หลัก - เลข 4 หลัก เช่น 74-5710
    คืน True ถ้า digits-only ของ text มีความยาว 5-7 หลัก
    (กันกรณี OCR อ่านขาด/เกินมาเล็กน้อย)
    """
    digits = re.sub(r"[^0-9]", "", text)
    return 5 <= len(digits) <= 7

# ============================================================
#  preprocess OCR ทะเบียน — เพิ่ม upscale สำหรับป้ายไกล
# ============================================================
def preprocess_ocr_plate(img):
    if img is None:
        return None
    h, w = img.shape[:2]

    # ถ้าภาพเล็กมาก (ป้ายไกล) ให้ upscale มากขึ้น
    if min(h, w) < 40:
        scale = 6.0
    elif min(h, w) < 80:
        scale = 4.0
    elif min(h, w) < 120:
        scale = 3.0
    else:
        scale = 2.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    gray = clahe.apply(gray)
    gray = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 11)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((2, 2), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return th

# ============================================================
#  preprocess OCR มิเตอร์
# ============================================================
def preprocess_ocr_fuel(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 4.0 if min(h, w) < 90 else 3.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    th_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 31, 9)
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ratio_adapt = np.sum(th_adapt > 0) / th_adapt.size
    ratio_otsu  = np.sum(th_otsu  > 0) / th_otsu.size
    th = th_adapt if abs(ratio_adapt - 0.5) < abs(ratio_otsu - 0.5) else th_otsu
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return th

# ============================================================
#  OCR ทะเบียน — คืนค่าดีสุดเสมอ
# ============================================================
def ocr_plate(img):
    """
    รับ crop จาก YOLO → หาบริเวณสีเหลือง → ตัดแถบ THAILAND → OCR
    ทะเบียนของระบบนี้เป็นตัวเลขล้วน (XX-XXXX) จึงจำกัด whitelist เฉพาะตัวเลข
    และเลือกผลลัพธ์ที่ "ตรง pattern ทะเบียน" ไม่ใช่ผลลัพธ์ที่ "ยาวที่สุด"
    เพื่อกัน OCR อ่านมั่วเอาเส้น/ขอบ/ตัวอักษรจางๆ มาปนเป็นสตริงยาว
    """
    if img is None:
        return ""

    crop_for_ocr = img

    # ── หาบริเวณป้ายสีเหลือง ──────────────────────────────
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([10, 40, 60]), np.array([45, 255, 255]))
        ys, xs = np.where(mask > 0)
        if len(xs) > 50:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            h_img, w_img = img.shape[:2]
            pad_x = int((x2 - x1) * 0.05) + 3
            pad_y = int((y2 - y1) * 0.15) + 3
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(w_img, x2 + pad_x); y2 = min(h_img, y2 + pad_y)
            if (x2 - x1) > 20 and (y2 - y1) > 10:
                crop_for_ocr = img[y1:y2, x1:x2]
    except Exception:
        pass

    # ── ตัดแถบ THAILAND ด้านบน ────────────────────────────
    h, w = crop_for_ocr.shape[:2]
    y_s = int(h * 0.32); y_e = int(h * 0.78)
    top_crop = crop_for_ocr[y_s:y_e, :] if crop_for_ocr[y_s:y_e, :].size > 0 else crop_for_ocr

    th = preprocess_ocr_plate(top_crop)
    if th is None:
        return ""

    # whitelist เฉพาะตัวเลข + ขีด — ทะเบียนระบบนี้ไม่มีตัวอักษรเลย
    whitelist = "0123456789-"
    configs = [
        f"--psm 7  --oem 3 -c tessedit_char_whitelist={whitelist}",
        f"--psm 8  --oem 3 -c tessedit_char_whitelist={whitelist}",
        f"--psm 13 --oem 3 -c tessedit_char_whitelist={whitelist}",
    ]

    candidates = []
    for cfg in configs:
        raw    = pytesseract.image_to_string(th, lang="eng", config=cfg)
        result = normalize_text(raw, "plate")
        dprint(f"[OCR-PLATE] {cfg.split()[1]} raw='{raw.strip()}' -> '{result}'")
        if result:
            candidates.append(result)

    if not candidates:
        return ""

    # เลือกตัวที่ "ตรง pattern ทะเบียน" (5-7 หลัก) ก่อน
    valid = [c for c in candidates if looks_like_plate(c)]
    if valid:
        # ในกลุ่มที่ valid เลือกตัวที่ยาวที่สุด (ครบถ้วนที่สุด)
        return max(valid, key=len)

    # ไม่มีตัวไหน valid เลย → คืนตัวที่ยาวที่สุดแบบเดิม (เผื่อ caller อยากดู)
    # แต่ caller จะปฏิเสธอยู่ดีถ้าไม่ผ่าน PLATE_MIN_CHARS/looks_like_plate
    return max(candidates, key=len)

# ============================================================
#  OCR มิเตอร์
# ============================================================
def ocr_fuel_candidates(img):
    th = preprocess_ocr_fuel(img)
    if th is None:
        return []
    out = []
    configs = [
        "--psm 7  --oem 3 -c tessedit_char_whitelist=0123456789.",
        "--psm 8  --oem 3 -c tessedit_char_whitelist=0123456789.",
        "--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789.",
    ]
    for cfg in configs:
        raw = pytesseract.image_to_string(th, lang="eng", config=cfg)
        txt = normalize_text(raw, "fuel")
        if txt:
            out.append(txt)
    return out

def ocr_fuel(img):
    cands = ocr_fuel_candidates(img)
    if not cands:
        return ""
    counts = {}
    for x in cands:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get)

# ============================================================
#  YOLO postprocess
# ============================================================
def yolo_postprocess(frame, raw, conf_thres=0.003, nms_thres=0.45):
    BLOB_SIZE  = 640
    h, w       = frame.shape[:2]
    detections = []

    outs = [raw] if isinstance(raw, np.ndarray) else list(raw)
    for out in outs:
        arr = np.array(out, dtype=np.float32)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
        if arr.ndim != 2:
            continue

        num_cols = arr.shape[1]
        dprint(f"postprocess arr shape = {arr.shape}")

        if num_cols >= 5:
            raw_confs = arr[:, 4]
            dprint(
                f"conf stats: min={raw_confs.min():.4f} max={raw_confs.max():.4f} "
                f"mean={raw_confs.mean():.4f} "
                f">0.10={(raw_confs > 0.10).sum()} "
                f">0.01={(raw_confs > 0.01).sum()} "
                f">0.003={(raw_confs > 0.003).sum()}"
            )

        for row in arr:
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])

            if num_cols == 5:
                conf_raw = float(row[4])
                # raw output ของ YOLOv8 ส่งค่า 0-1 ตรงๆ ไม่ต้องทำ sigmoid
                conf   = conf_raw if 0.0 <= conf_raw <= 1.0 else 1.0 / (1.0 + np.exp(-conf_raw))
                cls_id = 0
            elif num_cols == 6:
                conf   = float(row[4]) * float(row[5])
                cls_id = 0
            elif num_cols == 85:
                obj_conf  = float(row[4])
                cls_scores = row[5:]
                cls_id    = int(np.argmax(cls_scores))
                conf      = obj_conf * float(cls_scores[cls_id])
            else:
                cls_scores = row[4:]
                cls_id     = int(np.argmax(cls_scores))
                conf       = float(cls_scores[cls_id])

            if conf < conf_thres:
                continue

            scale_x = w / BLOB_SIZE; scale_y = h / BLOB_SIZE
            cx_f = cx * scale_x; cy_f = cy * scale_y
            bw_f = bw * scale_x; bh_f = bh * scale_y

            left   = max(0, int(cx_f - bw_f / 2))
            top    = max(0, int(cy_f - bh_f / 2))
            width  = min(w - left, int(bw_f))
            height = min(h - top,  int(bh_f))

            if width < 4 or height < 4:
                continue

            detections.append([left, top, width, height, float(conf), cls_id])

    if not detections:
        dprint("yolo_postprocess: no detection above threshold")
        return None, 0.0, None

    boxes = [d[:4] for d in detections]
    confs = [d[4] for d in detections]
    idxs  = cv2.dnn.NMSBoxes(boxes, confs, conf_thres, nms_thres)

    candidates = detections if len(idxs) == 0 else [detections[i] for i in idxs.flatten().tolist()]

    def score(d):
        ar = d[2] / max(d[3], 1)
        ar_bonus = 0.15 if 1.4 <= ar <= 4.0 else 0.0
        return d[4] + ar_bonus

    best = max(candidates, key=score)
    dprint(f"yolo best: box={best[:4]} conf={best[4]:.4f} ar={best[2]/max(best[3],1):.2f}")
    return tuple(best[:4]), best[4], best[5]

# ============================================================
#  detect_inside_roi — ถ้า YOLO ไม่เจอ ลอง OCR ทั้ง ROI (fallback)
# ============================================================
def detect_inside_roi(frame_disp, rx, model_path, min_conf, cam_label=""):
    if frame_disp is None:
        return None, 0.0, None

    crop_img = get_roi_crop(frame_disp, rx)
    if crop_img is None or crop_img.size == 0:
        return None, 0.0, None

    net  = get_net(model_path)
    blob = cv2.dnn.blobFromImage(crop_img, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    raw = net.forward()

    dprint(f"[{cam_label}] YOLO output shape = {np.array(raw).shape}")

    box, conf, cls_id = yolo_postprocess(crop_img, raw, conf_thres=min_conf, nms_thres=0.45)
    dprint(f"[{cam_label}] detect result: box={box} conf={conf:.4f}")

    if box is None or conf < min_conf:
        return None, conf, None

    x, y, bw, bh = box
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(crop_img.shape[1], x + bw)
    y2 = min(crop_img.shape[0], y + bh)

    if x2 <= x1 or y2 <= y1:
        return None, conf, None

    sub = crop_img[y1:y2, x1:x2].copy()

    # วาดกรอบบนภาพแสดงผล
    H, W    = frame_disp.shape[:2]
    rx_x1   = int(rx[0] * W); rx_y1 = int(rx[1] * H)
    cv2.rectangle(frame_disp,
                  (rx_x1 + x1, rx_y1 + y1),
                  (rx_x1 + x2, rx_y1 + y2),
                  (0, 0, 255), 3)
    cv2.putText(frame_disp, f"{conf:.3f}",
                (rx_x1 + x1, rx_y1 + y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return sub, conf, (x1, y1, x2, y2)

# ============================================================
#  stable_vote สำหรับมิเตอร์
# ============================================================
def stable_vote(history, min_count=FUEL_MIN_VOTES):
    if not history:
        return ""
    numeric = []
    for v in history:
        try:
            numeric.append((float(v), v))
        except Exception:
            pass
    if not numeric:
        return ""
    vals = [n[0] for n in numeric]
    med  = float(np.median(vals))
    tol  = max(med * 0.20, 1.0)
    filtered = [v_str for v_f, v_str in numeric if abs(v_f - med) <= tol]
    if not filtered:
        filtered = [n[1] for n in numeric]
    counts = {}
    for v in filtered:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=counts.get)
    return best if counts[best] >= min_count else ""

# ============================================================
#  Google Sheet / Drive
# ============================================================
def init_google():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    sh    = gspread.authorize(creds).open_by_key(SHEET_KEY).sheet1
    drive_service = None
    if DRIVE_AVAILABLE and DRIVE_FOLDER_ID:
        try:
            from google.oauth2 import service_account
            sa_creds = service_account.Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPE)
            drive_service = build("drive", "v3", credentials=sa_creds)
        except Exception as e:
            print(f"[WARN] Drive init failed: {e}")
    return sh, drive_service

def upload_to_drive(drive_service, img_array, filename, folder_id, retries=3):
    if drive_service is None or img_array is None or not folder_id:
        return ""
    for attempt in range(retries):
        try:
            ok, buf = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return ""
            fh    = io.BytesIO(buf.tobytes())
            media = MediaIoBaseUpload(fh, mimetype="image/jpeg", resumable=False)
            r     = drive_service.files().create(
                        body={"name": filename, "parents": [folder_id]},
                        media_body=media,
                        fields="webViewLink"
                    ).execute()
            return r.get("webViewLink", "")
        except Exception as e:
            print(f"[WARN] Drive upload attempt {attempt+1}: {e}")
            time.sleep(1.5 ** attempt)
    return ""

# ============================================================
#  cloud_worker
# ============================================================
def cloud_worker(sh, drive_service):
    while True:
        task = cloud_queue.get()
        if task is None:
            cloud_queue.task_done()
            break
        try:
            action  = task["action"]
            station = task["station"]
            ts      = task["ts"]
            ts_safe = ts.replace("/", "-").replace(":", "-").replace(" ", "_")

            if action == "save_plate":
                txt = task.get("plate_text", "")
                img = task.get("plate_img")
                if img is not None:
                    cv2.imwrite(f"LOCAL_PLATE_{station}_{ts_safe}.jpg", img)
                upload_to_drive(drive_service, img, f"PLATE_{station}_{ts_safe}.jpg", DRIVE_FOLDER_ID)
                row_data = [ts, txt, "", "", "", ""]
                sh.append_row(row_data, value_input_option="USER_ENTERED")
                idx = len(sh.get_all_values())
                with station_state[station]["lock"]:
                    station_state[station]["sheet_row_idx"] = idx
                print(f"[SHEET] ✅ บันทึกทะเบียน '{txt}' แถว {idx}")

            elif action == "save_fuel":
                txt = task.get("fuel_text", "0")
                idx = task.get("row_idx")
                img = task.get("fuel_img")
                if img is not None:
                    cv2.imwrite(f"LOCAL_FUEL_{station}_{ts_safe}.jpg", img)
                upload_to_drive(drive_service, img, f"FUEL_{station}_{ts_safe}.jpg", DRIVE_FOLDER_ID)
                if idx:
                    sh.update(f"C{idx}", [[txt]], value_input_option="USER_ENTERED")
                print(f"[SHEET] ✅ บันทึกน้ำมัน '{txt}' แถว {idx}")

        except Exception as e:
            print(f"[ERROR] cloud_worker: {e}")
        finally:
            cloud_queue.task_done()
            time.sleep(0.3)

# ============================================================
#  Reset helpers
# ============================================================
def reset_station_to_plate(sid):
    ss = station_state[sid]
    ss["state"]            = STATE_WAIT_PLATE
    ss["plate_text"]       = ""
    ss["plate_img"]        = None
    ss["fuel_last_text"]   = ""
    ss["fuel_img"]         = None
    ss["fuel_since"]       = 0.0
    ss["fuel_raw_history"] = []
    ss["sheet_row_idx"]    = None
    ss["last_save"]        = time.time()
    gc.collect()

def reset_station_to_fuel(sid, plate_text, plate_img):
    ss = station_state[sid]
    ss["state"]            = STATE_WAIT_FUEL
    ss["plate_text"]       = plate_text
    ss["plate_img"]        = plate_img.copy() if plate_img is not None else None
    ss["fuel_last_text"]   = ""
    ss["fuel_img"]         = None
    ss["fuel_since"]       = time.time()
    ss["fuel_raw_history"] = []
    ss["sheet_row_idx"]    = None

# ============================================================
#  Thread CAM1 — ตรวจทะเบียน
# ============================================================
def cam_thread_plate():
    cam_key = "CAM1"
    sid     = CAM_CONFIG[cam_key]["station"]
    cap     = RealtimeCapture(rtsp_url(cam_key)).start()
    print(f"[{cam_key}] thread started")

    while True:
        try:
            ss = station_state[sid]
            with ss["lock"]:
                cur_state = ss["state"]

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            disp = frame.copy()
            rx   = roi_cfg[cam_key]
            H, W = frame.shape[:2]

            # วาดกรอบ ROI สีเขียว
            cv2.rectangle(disp,
                          (int(rx[0]*W), int(rx[1]*H)),
                          (int(rx[2]*W), int(rx[3]*H)),
                          (0, 255, 0), 2)

            # ── [หยุด CAM1] ตาม Flowchart ─────────────────────────
            if cur_state != STATE_WAIT_PLATE:
                disp = put_text(disp, "CAM1 STOPPED - WAITING FUEL",
                                (8, 10), (0, 255, 255), 20)
                with frame_locks[cam_key]:
                    latest_frames[cam_key] = disp.copy()
                time.sleep(0.05)
                continue

            # ── YOLO ────────────────────────────────────────────────
            best_crop, conf, box = detect_inside_roi(
                disp, rx, PLATE_MODEL, PLATE_MIN_CONF, cam_key)

            with crop_locks[cam_key]:
                crop_monitors[cam_key] = best_crop.copy() if best_crop is not None else None

            # ── บันทึก debug images ─────────────────────────────────
            if best_crop is not None:
                try:
                    cv2.imwrite("debug_plate_crop.jpg", best_crop)
                    dbg = best_crop
                    hsv_d = cv2.cvtColor(best_crop, cv2.COLOR_BGR2HSV)
                    mask_d = cv2.inRange(hsv_d, np.array([10, 40, 60]), np.array([45, 255, 255]))
                    ys_d, xs_d = np.where(mask_d > 0)
                    if len(xs_d) > 50:
                        x1d = max(0, xs_d.min() - 3); x2d = min(best_crop.shape[1], xs_d.max() + 3)
                        y1d = max(0, ys_d.min() - 5); y2d = min(best_crop.shape[0], ys_d.max() + 5)
                        if (x2d - x1d) > 20 and (y2d - y1d) > 10:
                            dbg = best_crop[y1d:y2d, x1d:x2d]
                    cv2.imwrite("debug_plate_yellow.jpg", dbg)
                    h_t = dbg.shape[0]
                    top_dbg = dbg[int(h_t*0.32):int(h_t*0.78), :] or dbg
                    cv2.imwrite("debug_plate_top.jpg", top_dbg)
                    th_d = preprocess_ocr_plate(top_dbg)
                    if th_d is not None:
                        cv2.imwrite("debug_plate_thresh.jpg", th_d)
                except Exception:
                    pass

            # ── Decision: พบป้ายหรือไม่ ────────────────────────────
            if best_crop is None:
                # YOLO ไม่เจอ → "ไม่พบป้ายทะเบียน"
                label = "[CAM1] ไม่พบป้ายทะเบียน"
                disp  = put_text(disp, label, (8, 10), (80, 80, 255), 20)
                with frame_locks[cam_key]:
                    latest_frames[cam_key] = disp.copy()
                continue

            # ── OCR ────────────────────────────────────────────────
            txt = ocr_plate(best_crop)
            digit_count = len(re.sub(r"[^0-9]", "", txt))
            dprint(f"[CAM1] OCR='{txt}' digits={digit_count} conf={conf:.4f}")

            # ── Decision: OCR ถูกต้องหรือไม่ (ต้องตรง pattern ทะเบียน XX-XXXX) ──
            if not looks_like_plate(txt):
                # อ่านไม่ถูกต้อง → แสดงผล loop ต่อ
                label = (f"[CAM1] OCR: '{txt}' ({digit_count} หลัก — ไม่ตรงรูปแบบทะเบียน)"
                         if txt else f"[CAM1] OCR ไม่ได้ผล  [{conf:.3f}]")
                disp  = put_text(disp, label, (8, 10), (0, 120, 255), 20)
                with frame_locks[cam_key]:
                    latest_frames[cam_key] = disp.copy()
                continue

            # ── OCR ตรง pattern ทะเบียน ──────────────────────────────
            # เช็ค cooldown ป้องกันบันทึกซ้ำ
            with ss["lock"]:
                since_last = time.time() - ss["last_save"]

            if since_last < SAVE_COOLDOWN:
                remain = SAVE_COOLDOWN - since_last
                disp   = put_text(disp,
                                  f"[CAM1] {txt}  cooldown {remain:.0f}s",
                                  (8, 10), (0, 200, 255), 20)
                with frame_locks[cam_key]:
                    latest_frames[cam_key] = disp.copy()
                continue

            # ── [หยุด CAM1] → บันทึก Sheet → Drive → ไป WAIT_FUEL ──
            plate_img = best_crop.copy()
            ts        = now_ts()

            with ss["lock"]:
                ss["last_save"] = time.time()

            # เปลี่ยน state ก่อน (CAM1 หยุด, CAM2 เริ่ม)
            reset_station_to_fuel(sid, txt, plate_img)

            # ส่งงาน cloud (บันทึก Sheet + Drive)
            try:
                cloud_queue.put_nowait({
                    "action":     "save_plate",
                    "station":    sid,
                    "ts":         ts,
                    "plate_text": txt,
                    "plate_img":  plate_img,
                })
            except queue.Full:
                print("[WARN] cloud_queue เต็ม ข้ามการบันทึกทะเบียน")

            print(f"[CAM1] ✅ บันทึกทะเบียน '{txt}'  {ts}")
            disp = put_text(disp,
                            f"[CAM1] ✅ SAVED: {txt}  {ts}",
                            (8, 10), (0, 255, 100), 22)
            with frame_locks[cam_key]:
                latest_frames[cam_key] = disp.copy()

        except Exception as e:
            print(f"[ERROR] cam_thread_plate: {e}")
            time.sleep(0.2)

# ============================================================
#  Thread CAM2 — ตรวจมิเตอร์
# ============================================================
def cam_thread_fuel():
    cam_key = "CAM2"
    sid     = CAM_CONFIG[cam_key]["station"]
    cap     = RealtimeCapture(rtsp_url(cam_key)).start()
    print(f"[{cam_key}] thread started")

    while True:
        try:
            ss = station_state[sid]
            with ss["lock"]:
                cur_state = ss["state"]

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            disp = frame.copy()
            rx   = roi_cfg[cam_key]
            H, W = frame.shape[:2]

            cv2.rectangle(disp,
                          (int(rx[0]*W), int(rx[1]*H)),
                          (int(rx[2]*W), int(rx[3]*H)),
                          (0, 255, 0), 2)

            # ── รอให้ CAM1 บันทึกทะเบียนก่อน ───────────────────────
            if cur_state != STATE_WAIT_FUEL:
                disp = put_text(disp, "WAITING PLATE", (8, 10), (255, 200, 0), 20)
                with frame_locks[cam_key]:
                    latest_frames[cam_key] = disp.copy()
                time.sleep(0.05)
                continue

            # ── YOLO ─────────────────────────────────────────────────
            best_crop, conf, box = detect_inside_roi(
                disp, rx, FUEL_MODEL, FUEL_MIN_CONF, cam_key)

            with crop_locks[cam_key]:
                crop_monitors[cam_key] = best_crop.copy() if best_crop is not None else None

            if best_crop is not None:
                try:
                    cv2.imwrite("debug_fuel_crop.jpg", best_crop)
                    th_d = preprocess_ocr_fuel(best_crop)
                    if th_d is not None:
                        cv2.imwrite("debug_fuel_thresh.jpg", th_d)
                except Exception:
                    pass

            # ── Decision: พบมิเตอร์หรือไม่ ──────────────────────────
            if best_crop is None:
                disp = put_text(disp, "[TANK] ไม่พบมิเตอร์", (8, 10), (80, 80, 255), 20)
                with frame_locks[cam_key]:
                    latest_frames[cam_key] = disp.copy()
                continue

            # ── OCR ──────────────────────────────────────────────────
            txt = ocr_fuel(best_crop)
            dprint(f"[CAM2] conf={conf:.4f} ocr='{txt}'")

            with ss["lock"]:
                if txt:
                    ss["fuel_raw_history"].append(txt)
                    if len(ss["fuel_raw_history"]) > OCR_WINDOW_SIZE:
                        ss["fuel_raw_history"].pop(0)

                    stable = stable_vote(ss["fuel_raw_history"])
                    if stable and stable != ss["fuel_last_text"]:
                        ss["fuel_last_text"] = stable
                        ss["fuel_since"]     = time.time()
                        ss["fuel_img"]       = best_crop.copy() if best_crop is not None else ss["fuel_img"]

                elapsed = time.time() - ss["fuel_since"] if ss["fuel_since"] > 0 else 0

                # ── Decision: ตัวเลขคงที่ + OCR ถูกต้อง ─────────────
                if elapsed >= FUEL_STABLE_SEC and ss["state"] == STATE_WAIT_FUEL:
                    final_fuel = ss["fuel_last_text"]

                    # Decision: OCR ถูกต้องหรือไม่ (≥ FUEL_MIN_CHARS ตัวเลข)
                    digits_only = re.sub(r"[^0-9]", "", final_fuel)
                    if len(digits_only) < FUEL_MIN_CHARS:
                        # ไม่ถูกต้อง → "ไม่พบปริมาณน้ำมัน"
                        disp = put_text(disp,
                                        f"[TANK] ไม่พบปริมาณน้ำมัน (OCR='{final_fuel}')",
                                        (8, 10), (0, 80, 255), 20)
                        # reset history เพื่อรอค่าใหม่
                        ss["fuel_raw_history"] = []
                        ss["fuel_last_text"]   = ""
                        ss["fuel_since"]       = 0.0
                        with frame_locks[cam_key]:
                            latest_frames[cam_key] = disp.copy()
                        continue

                    # OCR ผ่าน → [หยุด CAM2] → บันทึก
                    final_img  = (ss["fuel_img"].copy() if ss["fuel_img"] is not None
                                  else best_crop.copy())
                    target_row = ss["sheet_row_idx"]
                    ts         = now_ts()

                    try:
                        cloud_queue.put_nowait({
                            "action":    "save_fuel",
                            "station":   sid,
                            "ts":        ts,
                            "fuel_text": final_fuel,
                            "fuel_img":  final_img,
                            "row_idx":   target_row,
                        })
                    except queue.Full:
                        print("[WARN] cloud_queue เต็ม ข้ามการบันทึกน้ำมัน")

                    reset_station_to_plate(sid)
                    print(f"[CAM2] ✅ บันทึกน้ำมัน '{final_fuel}'  {ts}")
                    disp = put_text(disp,
                                    f"[TANK] ✅ SAVED: {final_fuel}  {ts}",
                                    (8, 10), (0, 255, 200), 20)
                    with frame_locks[cam_key]:
                        latest_frames[cam_key] = disp.copy()
                    continue

                current_fuel = ss["fuel_last_text"] or "รอมิเตอร์นิ่ง"

            color = (0, 255, 200) if elapsed >= FUEL_STABLE_SEC else (255, 200, 0)
            disp  = put_text(disp,
                             f"[TANK] {current_fuel}  ({elapsed:.1f}/{FUEL_STABLE_SEC:.0f}s)",
                             (8, 10), color, 20)

            with frame_locks[cam_key]:
                latest_frames[cam_key] = disp.copy()

        except Exception as e:
            print(f"[ERROR] cam_thread_fuel: {e}")
            time.sleep(0.2)

# ============================================================
#  ภาพว่าง
# ============================================================
def blank_cell(cam):
    c = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    return put_text(c, f"[{cam}] ไม่มีสัญญาณภาพ", (20, CELL_H // 2), (100, 100, 100), 20)

def blank_sub():
    return np.zeros((180, 320, 3), dtype=np.uint8) + 40

# ============================================================
#  ตั้ง ROI
# ============================================================
def setup_roi_interactively():
    for cam in ["CAM1", "CAM2"]:
        url = rtsp_url(cam)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ret, fr = cap.read()
        cap.release()
        if not ret or fr is None:
            print(f"[WARN] ROI setup: ไม่สามารถอ่านภาพจาก {cam}")
            continue

        H, W = fr.shape[:2]
        win  = f"ROI Setup: {cam}  (Enter=Save, S=Skip)"
        ds   = {"s": None, "e": None, "drag": False, "done": False}

        def mcb(ev, x, y, fl, p):
            if ev == cv2.EVENT_LBUTTONDOWN:
                ds.update(s=(x, y), e=(x, y), drag=True, done=False)
            elif ev == cv2.EVENT_MOUSEMOVE and ds["drag"]:
                ds["e"] = (x, y)
            elif ev == cv2.EVENT_LBUTTONUP:
                ds.update(e=(x, y), drag=False, done=True)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 850, 480)
        cv2.setMouseCallback(win, mcb)

        while True:
            d  = fr.copy()
            rx = roi_cfg[cam]
            cv2.rectangle(d,
                          (int(rx[0]*W), int(rx[1]*H)),
                          (int(rx[2]*W), int(rx[3]*H)),
                          (0, 255, 0), 2)
            if ds["s"] and ds["e"]:
                cv2.rectangle(d, ds["s"], ds["e"], (0, 0, 255), 3)
            cv2.imshow(win, d)
            k = cv2.waitKey(30) & 0xFF
            if k == 13 and ds["done"]:
                x1 = min(ds["s"][0], ds["e"][0]); y1 = min(ds["s"][1], ds["e"][1])
                x2 = max(ds["s"][0], ds["e"][0]); y2 = max(ds["s"][1], ds["e"][1])
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    roi_cfg[cam] = [x1/W, y1/H, x2/W, y2/H]
                    print(f"[ROI] {cam} = {roi_cfg[cam]}")
                break
            elif k in (ord('s'), ord('S'), 27):
                break
        cv2.destroyWindow(win)
    save_roi()

# ============================================================
#  main
# ============================================================
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_roi(reset=False)
    if not os.path.exists(ROI_FILE):
        setup_roi_interactively()

    sh, drive_service = init_google()
    threading.Thread(target=cloud_worker, args=(sh, drive_service), daemon=True).start()
    threading.Thread(target=cam_thread_plate, daemon=True).start()
    threading.Thread(target=cam_thread_fuel,  daemon=True).start()

    gc_timer = time.time()

    while True:
        row = []
        for k in ["CAM1", "CAM2"]:
            with frame_locks[k]:
                fr = latest_frames[k].copy() if latest_frames[k] is not None else None
            row.append(cv2.resize(fr, (CELL_W, CELL_H)) if fr is not None else blank_cell(k))

        grid = np.hstack(row)

        with station_state["S1"]["lock"]:
            pt1 = station_state["S1"]["plate_text"] or "-"
            st1 = station_state["S1"]["state"]
        with station_state["S2"]["lock"]:
            fu2 = station_state["S2"]["fuel_last_text"] or "-"
            st2 = station_state["S2"]["state"]

        status = (f"S1[{st1}] ทะเบียน={pt1}   "
                  f"S2[{st2}] มิเตอร์={fu2}   "
                  f"[Q=ปิดระบบ]")
        grid = put_text(grid, status, (8, grid.shape[0] - 28), (0, 0, 255), 18)
        cv2.imshow("Smart Refueling Dashboard", grid)

        if DEBUG_UI:
            subs = []
            for k in ["CAM1", "CAM2"]:
                with crop_locks[k]:
                    ci = crop_monitors[k].copy() if crop_monitors[k] is not None else None
                if ci is not None and ci.size > 0:
                    subs.append(cv2.resize(ci, (320, 180)))
                else:
                    b = blank_sub()
                    cv2.putText(b, f"{k} WAITING", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                    subs.append(b)
            cv2.imshow("Crop Monitor", np.hstack(subs))

        if time.time() - gc_timer > 30:
            gc.collect()
            gc_timer = time.time()

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cloud_queue.put(None)

if __name__ == "__main__":
    main()
