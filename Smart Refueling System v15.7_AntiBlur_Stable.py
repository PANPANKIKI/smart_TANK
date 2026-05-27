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
from PIL import Image, ImageDraw, ImageFont
from oauth2client.service_account import ServiceAccountCredentials

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    from google.oauth2.service_account import Credentials as SACredentials
    DRIVE_AVAILABLE = True
except ImportError:
    DRIVE_AVAILABLE = False

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

CREDS_FILE = "service_account.json"
SHEET_KEY = "1AMVQ650o1dsiGbdp2W2NoUp7nhc-J8k33nc4Fd0uUes"
DRIVE_FOLDER_ID = ""

PLATE_MODEL = "bestLicensePlate.onnx"
FUEL_MODEL = "bestTANK.onnx"
ROI_FILE = "roi_config.json"

PLATE_MIN_CONF = 0.30
FUEL_MIN_CONF = 0.25
PLATE_MIN_CHARS = 4
FUEL_STABLE_SEC = 5.0
SAVE_COOLDOWN = 12.0
OCR_CONFIRM_FRAMES = 6
OCR_WINDOW_SIZE = 12

CELL_W, CELL_H = 640, 360
DEBUG_UI = True

STATE_LPR_SCAN = "1_LPR_SCAN"
STATE_FUEL_SCAN = "5_FUEL_SCAN"

SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

CAM_CONFIG = {
    "CAM1":  {"user":"admin", "pass":"Pp_282829", "ip":"192.168.1.23",  "path":"/stream1", "mode":"plate", "label":"CAM1",  "station":"S1"},
    "CAM2":  {"user":"admin", "pass":"Pp_232323", "ip":"192.168.1.150", "path":"/stream1", "mode":"plate", "label":"CAM2",  "station":"S2"},
    "TAPO1": {"user":"Tapotank1", "pass":"pp_232222", "ip":"192.168.1.109", "path":"/stream1", "mode":"fuel",  "label":"TAPO1", "station":"S1"},
    "TAPO2": {"user":"Tapotank2", "pass":"pp_232121", "ip":"192.168.1.101", "path":"/stream1", "mode":"fuel",  "label":"TAPO2", "station":"S2"},
}

DEFAULT_ROI = {
    "CAM1":  [0.10, 0.35, 0.90, 0.95],
    "CAM2":  [0.05, 0.20, 0.90, 0.85],
    "TAPO1": [0.15, 0.30, 0.85, 0.90],
    "TAPO2": [0.15, 0.30, 0.85, 0.90],
}
roi_cfg = dict(DEFAULT_ROI)

def _new_station_state():
    return {
        "state": STATE_LPR_SCAN,
        "plate_text": "",
        "plate_img": None,
        "fuel_last_text": "",
        "fuel_img": None,
        "fuel_since": 0.0,
        "fuel_raw_history": [],
        "sheet_row_idx": None,
        "last_save": 0.0,
        "confirm_hits": 0,
        "confirm_value": "",
        "lock": threading.Lock()
    }

station_state = {"S1": _new_station_state(), "S2": _new_station_state()}
latest_frames = {k: None for k in CAM_CONFIG}
frame_locks = {k: threading.Lock() for k in CAM_CONFIG}
crop_monitors = {k: None for k in CAM_CONFIG}
crop_locks = {k: threading.Lock() for k in CAM_CONFIG}
cloud_queue = queue.Queue()
_tls = threading.local()
_font_cache = {}

def rtsp_url(cam):
    c = CAM_CONFIG[cam]
    return f"rtsp://{c['user']}:{urllib.parse.quote_plus(c['pass'])}@{c['ip']}:554{c['path']}"

def get_font(size=20):
    if size in _font_cache:
        return _font_cache[size]
    for p in [
        "/home/agentfuel/cam_fuel/TAHOMA.TTF",
        "/home/agentfuel/cam_fuel/TAHOMA.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _font_cache[size] = f
                return f
            except:
                pass
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f

def put_text(img, text, pos, color=(0,220,120), size=20):
    if img is None:
        return img
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        ImageDraw.Draw(pil).text(pos, str(text), font=get_font(size), fill=tuple(color))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (int(color[2]), int(color[1]), int(color[0])), 2)
        return img

def get_roi_crop(img, rx):
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = int(rx[0]*W), int(rx[1]*H), int(rx[2]*W), int(rx[3]*H)
    c = img[y1:y2, x1:x2]
    return c if c.size > 0 else None

def load_roi():
    global roi_cfg
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if k in roi_cfg and isinstance(v, list) and len(v) == 4:
                    roi_cfg[k] = v
            print("[ROI] loaded")
        except Exception as e:
            print(f"[ROI] load failed: {e}")

def save_roi():
    try:
        with open(ROI_FILE, "w", encoding="utf-8") as f:
            json.dump(roi_cfg, f, indent=2, ensure_ascii=False)
        print("[ROI] saved")
    except Exception as e:
        print(f"[ROI] save failed: {e}")

def get_net(model_path):
    attr = "net_" + os.path.basename(model_path).replace(".", "_")
    if not hasattr(_tls, attr):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        print(f"[ONNX] load: {model_path}")
        net = cv2.dnn.readNetFromONNX(model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        setattr(_tls, attr, net)
    return getattr(_tls, attr)

class RealtimeCapture:
    def __init__(self, url, cam_key, timeout_sec=8):
        self._url = url
        self._cam_key = cam_key
        self._timeout_sec = timeout_sec
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cap-{cam_key}")

    def start(self):
        self._thread.start()
        return self

    def _open_cap(self):
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout_sec * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout_sec * 1000)
        except:
            pass
        return cap

    def _run(self):
        while not self._stop.is_set():
            cap = None
            try:
                cap = self._open_cap()
                last_ok = time.time()
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        last_ok = time.time()
                        with self._lock:
                            self._frame = frame
                    elif time.time() - last_ok > self._timeout_sec:
                        break
                    time.sleep(0.01)
            except Exception as e:
                print(f"[CAP ERROR] {self._cam_key}: {e}")
            finally:
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
            if not self._stop.is_set():
                time.sleep(2)

    def read(self):
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

def normalize_text(text, mode="plate"):
    if not text:
        return ""
    text = text.upper().strip()
    if mode == "plate":
        text = re.sub(r"[^A-Z0-9ก-ฮ]", "", text)
        text = text.replace("O", "0").replace("I", "1").replace("Z", "2").replace("S", "5")
        return text
    text = text.replace("|", "1").replace("/", "1").replace("\\", "1").replace("O", "0")
    text = re.sub(r"[^0-9.]", "", text)
    if text.count(".") > 1:
        parts = text.split(".")
        text = parts[0] + "." + "".join(parts[1:])
    return text

def preprocess_ocr_plate(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 3.0 if min(h, w) < 120 else 2.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th

def preprocess_ocr_fuel(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 3.5 if min(h, w) < 90 else 2.5
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 9)
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

def ocr_plate(img):
    th = preprocess_ocr_plate(img)
    if th is None:
        return ""
    cfg = "--psm 7 --oem 3 -c preserve_interword_spaces=1"
    raw = pytesseract.image_to_string(th, lang="tha+eng", config=cfg)
    return normalize_text(raw, "plate")

def ocr_fuel_candidates(img):
    th = preprocess_ocr_fuel(img)
    if th is None:
        return []
    out = []
    configs = [
        "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.",
        "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789.",
        "--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789."
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

def box_from_yolo(raw, cw, ch):
    p = raw.squeeze()
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.shape[0] < p.shape[1]:
        p = p.T
    best_conf = 0.0
    best_box = None
    for row in p:
        if len(row) < 5:
            continue
        conf = float(row[4])
        if conf > best_conf:
            best_conf = conf
            best_box = row[:4]
    if best_box is None:
        return None, 0.0
    cx, cy, bw, bh = best_box
    x1 = max(0, int((cx - bw / 2) * cw / 640.0))
    y1 = max(0, int((cy - bh / 2) * ch / 640.0))
    x2 = min(cw, int((cx + bw / 2) * cw / 640.0))
    y2 = min(ch, int((cy + bh / 2) * ch / 640.0))
    if x2 <= x1 or y2 <= y1:
        return None, best_conf
    return (x1, y1, x2, y2), best_conf

def detect_inside_roi(frame_disp, rx, model_path, min_conf):
    if frame_disp is None:
        return None, 0.0, None
    crop_img = get_roi_crop(frame_disp, rx)
    if crop_img is None or crop_img.size == 0:
        return None, 0.0, None
    ch, cw = crop_img.shape[:2]
    H, W = frame_disp.shape[:2]
    net = get_net(model_path)
    blob = cv2.dnn.blobFromImage(crop_img, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    raw = net.forward()
    box, conf = box_from_yolo(raw, cw, ch)
    if box is None or conf < min_conf:
        return None, conf, None
    x1, y1, x2, y2 = box
    sub = crop_img[y1:y2, x1:x2].copy()
    rx_x1, rx_y1 = int(rx[0] * W), int(rx[1] * H)
    cv2.rectangle(frame_disp, (rx_x1 + x1, rx_y1 + y1), (rx_x1 + x2, rx_y1 + y2), (0, 0, 255), 3)
    return sub, conf, box

def stable_vote(history):
    if not history:
        return ""
    count = {}
    for v in history:
        count[v] = count.get(v, 0) + 1
    best = max(count, key=count.get)
    if count[best] >= max(3, len(history) // 2):
        return best
    return ""

def init_google():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    sh = gspread.authorize(creds).open_by_key(SHEET_KEY).sheet1
    drive_service = None
    if DRIVE_AVAILABLE and DRIVE_FOLDER_ID:
        try:
            sa = SACredentials.from_service_account_file(CREDS_FILE, scopes=SCOPE)
            drive_service = build("drive", "v3", credentials=sa)
        except Exception as e:
            print(f"[DRIVE WARNING] {e}")
    return sh, drive_service

def upload_to_drive(drive_service, img_array, filename, folder_id):
    if not drive_service or img_array is None or not folder_id:
        return ""
    try:
        ok, buf = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return ""
        fh = io.BytesIO(buf.tobytes())
        r = drive_service.files().create(
            body={"name": filename, "parents": [folder_id]},
            media_body=MediaIoBaseUpload(fh, mimetype="image/jpeg", resumable=False),
            fields="webViewLink"
        ).execute()
        return r.get("webViewLink", "")
    except Exception as e:
        print(f"[DRIVE ERROR] {e}")
        return ""

def cloud_worker(sh, drive_service):
    while True:
        task = cloud_queue.get()
        if task is None:
            cloud_queue.task_done()
            break
        try:
            action, station, ts = task["action"], task["station"], task["ts"]
            ts_safe = ts.replace("/", "-").replace(":", "-").replace(" ", "_")
            if action == "save_plate":
                txt = task.get("plate_text", "")
                upload_to_drive(drive_service, task.get("plate_img"), f"PLATE_{station}_{ts_safe}.jpg", DRIVE_FOLDER_ID)
                row_data = [ts, "", "", "", "", ""]
                if station == "S1":
                    row_data[1] = txt
                else:
                    row_data[2] = txt
                sh.append_row(row_data, value_input_option="USER_ENTERED")
                idx = len(sh.get_all_values())
                with station_state[station]["lock"]:
                    station_state[station]["sheet_row_idx"] = idx
                print(f"[SHEETS] plate {station} row={idx} txt={txt}")
            elif action == "save_fuel":
                txt = task.get("fuel_text", "0")
                idx = task.get("row_idx")
                upload_to_drive(drive_service, task.get("fuel_img"), f"FUEL_{station}_{ts_safe}.jpg", DRIVE_FOLDER_ID)
                if idx:
                    if station == "S1":
                        sh.update(f"D{idx}", [[txt]], value_input_option="USER_ENTERED")
                    else:
                        sh.update(f"E{idx}", [[txt]], value_input_option="USER_ENTERED")
                    print(f"[SHEETS] fuel {station} row={idx} txt={txt}")
        except Exception as e:
            print(f"[CLOUD ERROR] {e}")
        finally:
            cloud_queue.task_done()
            time.sleep(0.3)

def cam_thread(cam_key):
    cfg = CAM_CONFIG[cam_key]
    mode, label, sid = cfg["mode"], cfg["label"], cfg["station"]
    cap = RealtimeCapture(rtsp_url(cam_key), cam_key).start()
    while True:
        try:
            ret, frame = cap.read()
            ss = station_state[sid]
            with ss["lock"]:
                cur_state = ss["state"]
            should_process = (mode == "plate" and cur_state == STATE_LPR_SCAN) or (mode == "fuel" and cur_state == STATE_FUEL_SCAN)
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            disp = frame.copy()
            rx = roi_cfg[cam_key]
            H, W = frame.shape[:2]
            cv2.rectangle(disp, (int(rx[0]*W), int(rx[1]*H)), (int(rx[2]*W), int(rx[3]*H)), (0, 255, 0), 2)
            if not should_process:
                with frame_locks[cam_key]:
                    latest_frames[cam_key] = disp.copy()
                time.sleep(0.03)
                continue

            if mode == "plate":
                best_crop, conf, _ = detect_inside_roi(disp, rx, PLATE_MODEL, PLATE_MIN_CONF)
                with crop_locks[cam_key]:
                    crop_monitors[cam_key] = best_crop.copy() if best_crop is not None else None
                txt = ocr_plate(best_crop) if best_crop is not None else ""
                if len(txt) >= PLATE_MIN_CHARS:
                    with ss["lock"]:
                        if txt == ss["confirm_value"]:
                            ss["confirm_hits"] += 1
                        else:
                            ss["confirm_value"] = txt
                            ss["confirm_hits"] = 1
                        if ss["confirm_hits"] >= OCR_CONFIRM_FRAMES and (time.time() - ss["last_save"] >= SAVE_COOLDOWN):
                            ss["state"] = STATE_FUEL_SCAN
                            ss["plate_text"] = txt
                            ss["plate_img"] = best_crop.copy()
                            ss["fuel_last_text"] = ""
                            ss["fuel_since"] = time.time()
                            ss["fuel_raw_history"] = []
                            ss["sheet_row_idx"] = None
                            ss["last_save"] = time.time()
                            ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                            cloud_queue.put({
                                "action": "save_plate",
                                "station": sid,
                                "ts": ts,
                                "plate_text": txt,
                                "plate_img": best_crop.copy()
                            })
                            ss["confirm_hits"] = 0
                            ss["confirm_value"] = ""
                with ss["lock"]:
                    current_plate = ss["plate_text"] or "รอจับทะเบียน"
                disp = put_text(disp, f"[{label}] {current_plate}", (8, 10), (0, 220, 120), 20)

            elif mode == "fuel":
                best_crop, conf, _ = detect_inside_roi(disp, rx, FUEL_MODEL, FUEL_MIN_CONF)
                with crop_locks[cam_key]:
                    crop_monitors[cam_key] = best_crop.copy() if best_crop is not None else None
                txt = ocr_fuel(best_crop) if best_crop is not None else ""
                with ss["lock"]:
                    if txt:
                        ss["fuel_raw_history"].append(txt)
                        if len(ss["fuel_raw_history"]) > OCR_WINDOW_SIZE:
                            ss["fuel_raw_history"].pop(0)
                        stable = stable_vote(ss["fuel_raw_history"])
                        if stable and stable != ss["fuel_last_text"]:
                            ss["fuel_last_text"] = stable
                            ss["fuel_since"] = time.time()
                    elapsed = time.time() - ss["fuel_since"] if ss["fuel_since"] > 0 else 0
                    force_flush = elapsed >= FUEL_STABLE_SEC
                    if force_flush and ss["state"] == STATE_FUEL_SCAN:
                        final_fuel = ss["fuel_last_text"] if ss["fuel_last_text"] else "0"
                        final_img = ss["fuel_img"].copy() if ss["fuel_img"] is not None else (best_crop.copy() if best_crop is not None else None)
                        target_row = ss["sheet_row_idx"]
                        ss["state"] = STATE_LPR_SCAN
                        ss["plate_text"] = ""
                        ss["plate_img"] = None
                        ss["fuel_last_text"] = ""
                        ss["fuel_img"] = None
                        ss["fuel_since"] = 0.0
                        ss["fuel_raw_history"] = []
                        ss["confirm_hits"] = 0
                        ss["confirm_value"] = ""
                        ss["last_save"] = time.time()
                        ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        cloud_queue.put({
                            "action": "save_fuel",
                            "station": sid,
                            "ts": ts,
                            "fuel_text": final_fuel,
                            "fuel_img": final_img,
                            "row_idx": target_row
                        })
                        ss["sheet_row_idx"] = None
                    current_fuel = ss["fuel_last_text"] or "รอมิเตอร์นิ่ง"
                color = (0, 255, 200) if elapsed >= FUEL_STABLE_SEC else (255, 200, 0)
                disp = put_text(disp, f"[{label}] {current_fuel} ({elapsed:.1f}/{FUEL_STABLE_SEC:.0f}s)", (8, 10), color, 20)

            with frame_locks[cam_key]:
                latest_frames[cam_key] = disp.copy()
        except Exception as e:
            print(f"[CAM THREAD ERROR] {cam_key}: {e}")
            time.sleep(0.2)

_ds = {"s": None, "e": None, "drag": False, "done": False}
def _mcb(ev, x, y, fl, p):
    if ev == cv2.EVENT_LBUTTONDOWN:
        _ds.update(s=(x,y), e=(x,y), drag=True, done=False)
    elif ev == cv2.EVENT_MOUSEMOVE and _ds["drag"]:
        _ds["e"] = (x,y)
    elif ev == cv2.EVENT_LBUTTONUP:
        _ds.update(e=(x,y), drag=False, done=True)

def setup_roi_interactively():
    print("\n" + "="*60)
    print(" ROI SETUP")
    print(" Enter = save, S = skip, Esc = skip")
    print("="*60)
    for cam in ["CAM1", "CAM2", "TAPO1", "TAPO2"]:
        url = rtsp_url(cam)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ret, fr = cap.read()
        cap.release()
        if not ret or fr is None:
            print(f"[ROI WARNING] {cam} read fail")
            continue
        H, W = fr.shape[:2]
        win = f"ROI Setup Mode: {cam} (Enter=Save, S=Skip)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 850, 480)
        cv2.setMouseCallback(win, _mcb)
        _ds.update(s=None, e=None, drag=False, done=False)
        while True:
            d = fr.copy()
            rx = roi_cfg[cam]
            cv2.rectangle(d, (int(rx[0]*W), int(rx[1]*H)), (int(rx[2]*W), int(rx[3]*H)), (0, 255, 0), 2)
            if _ds["s"] and _ds["e"]:
                cv2.rectangle(d, _ds["s"], _ds["e"], (0, 0, 255), 3)
            cv2.imshow(win, d)
            k = cv2.waitKey(30) & 0xFF
            if k == 13 and _ds["done"]:
                x1 = min(_ds["s"][0], _ds["e"][0])
                y1 = min(_ds["s"][1], _ds["e"][1])
                x2 = max(_ds["s"][0], _ds["e"][0])
                y2 = max(_ds["s"][1], _ds["e"][1])
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    roi_cfg[cam] = [x1/W, y1/H, x2/W, y2/H]
                    print(f"[{cam}] saved: {roi_cfg[cam]}")
                break
            elif k in (ord('s'), ord('S'), 27):
                print(f"[{cam}] skip")
                break
        cv2.destroyWindow(win)
    save_roi()

def blank_cell(cam):
    c = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    return put_text(c, f"[{cam}] ไม่มีสัญญาณภาพ", (20, CELL_H//2), (100, 100, 100), 20)

def blank_sub():
    return np.zeros((180, 320, 3), dtype=np.uint8) + 40

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("\n" + "="*72)
    print(" Smart Refueling System v26.0")
    print("="*72)

    load_roi()
    setup_roi_interactively()

    try:
        sh, drive_service = init_google()
    except Exception as e:
        print(f"[FATAL ERROR] sheet auth failed: {e}")
        return

    threading.Thread(target=cloud_worker, args=(sh, drive_service), daemon=True, name="cloud-worker").start()
    for k in CAM_CONFIG:
        threading.Thread(target=cam_thread, args=(k,), daemon=True, name=f"cam-{k}").start()

    while True:
        row1, row2 = [], []
        for k in ["CAM1", "CAM2"]:
            with frame_locks[k]:
                fr = latest_frames[k].copy() if latest_frames[k] is not None else None
            row1.append(cv2.resize(fr, (CELL_W, CELL_H)) if fr is not None else blank_cell(k))
        for k in ["TAPO1", "TAPO2"]:
            with frame_locks[k]:
                fr = latest_frames[k].copy() if latest_frames[k] is not None else None
            row2.append(cv2.resize(fr, (CELL_W, CELL_H)) if fr is not None else blank_cell(k))

        grid = np.vstack([np.hstack(row1), np.hstack(row2)])

        status = []
        for sid in ("S1", "S2"):
            ss = station_state[sid]
            with ss["lock"]:
                st = ss["state"]
                pt = ss["plate_text"] or "-"
                fu = ss["fuel_last_text"] or "-"
            status.append(f"{sid}[{st}] ทะเบียน={pt} มิเตอร์={fu}")

        grid = np.ascontiguousarray(grid, dtype=np.uint8)
        grid = put_text(grid, " | ".join(status) + " [Q ปิดระบบ]", (8, grid.shape[0] - 28), (255, 255, 255), 18)
        cv2.imshow("Smart Refueling Dashboard", grid)

        if DEBUG_UI:
            subs = []
            for k in ["CAM1", "CAM2", "TAPO1", "TAPO2"]:
                with crop_locks[k]:
                    ci = crop_monitors[k].copy() if crop_monitors[k] is not None else None
                if ci is not None and ci.size > 0:
                    subs.append(cv2.resize(ci, (320, 180)))
                else:
                    b = blank_sub()
                    cv2.putText(b, f"{k} WAITING OBJECT", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                    subs.append(b)
            cv2.imshow("Crop Monitor", np.hstack(subs))

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
