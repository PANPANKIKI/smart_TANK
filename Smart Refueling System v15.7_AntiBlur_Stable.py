import os
import cv2
import re
import time
import json
import queue
import gspread
import threading
import datetime
import numpy as np
import pytesseract
import urllib.parse
from PIL import Image, ImageDraw, ImageFont
from oauth2client.service_account import ServiceAccountCredentials

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

# =========================================================
# CONFIG
# =========================================================
CREDS_FILE       = "service_account.json"
SHEET_KEY        = "1AMVQ650o1dsiGbdp2W2NoUp7nhc-J8k33nc4Fd0uUes"
ROI_FILE         = "roi_config.json"

# 🚀 ปรับตั้งค่าความเสถียร (หน่วงเพื่อเช็กให้แน่ใจว่าตัวเลขนิ่งจริง ไม่ใช่รถวิ่งผ่าน)
FUEL_STABLE_SEC = 5.0    
SAVE_COOLDOWN   = 30.0   # ขยายเวลาคูลดาวน์เพื่อแก้ปัญหา Google API 403
CELL_W, CELL_H  = 640, 360

print("[ONNX] กำลังโหลดโมเดลอัจฉริยะ เข้าสู่ OpenCV DNN...")
net = cv2.dnn.readNetFromONNX("best.onnx")

CAM_CONFIG = {
    "CAM1":  {"user":"admin",     "pass":"Pp_282829",  "ip":"192.168.1.23",
               "path":"/stream1", "mode":"plate", "label":"ทะเบียน 1"},
    "CAM2":  {"user":"admin",     "pass":"Pp_232323",  "ip":"192.168.1.150",
               "path":"/stream1", "mode":"plate", "label":"ทะเบียน 2"},
    "TAPO1": {"user":"Tapotank1", "pass":"pp_232222",  "ip":"192.168.1.109",
               "path":"/stream1", "mode":"fuel",  "label":"มิเตอร์ 1"},
    "TAPO2": {"user":"Tapotank2", "pass":"pp_232121",  "ip":"192.168.1.101",
               "path":"/stream1", "mode":"fuel",  "label":"มิเตอร์ 2"},
}

DEFAULT_ROI = {
    "CAM1":  [0.0, 0.4, 1.0, 1.0],
    "CAM2":  [0.0, 0.4, 1.0, 1.0],
    "TAPO1": [0.25, 0.65, 0.50, 0.85],
    "TAPO2": [0.55, 0.65, 0.80, 0.85],
}

# ใช้แค่สิทธิ์ Sheets ไม่พึ่งพา Google Drive อีกต่อไป
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets"]

# ระบบคิวเพื่อจัดการการส่งข้อมูลขึ้น Google Sheets ป้องกัน exit code -11
sheets_logging_queue = queue.Queue()

# =========================================================
# STATE
# =========================================================
roi_cfg       = dict(DEFAULT_ROI)
latest_frames = {k: None for k in CAM_CONFIG}
frame_locks   = {k: threading.Lock() for k in CAM_CONFIG}

plate_state = {k: {"text":"","img":None} for k in ("CAM1","CAM2")}

# ยกระดับโครงสร้างเก็บสถานะเพื่อคัดกรองปัญหามิเตอร์แกว่ง
fuel_state  = {k: {"text":"","img":None,"since":0,"saved":"","last_save":0, "lock_trigger": False}
               for k in ("TAPO1","TAPO2")}

crop_monitors = {k: None for k in CAM_CONFIG}
crop_locks    = {k: threading.Lock() for k in CAM_CONFIG}

# =========================================================
# REALTIME VIDEO CAPTURE
# =========================================================
class RealtimeVideoCapture:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.last_frame = None
        self.started = False
        self.lock = threading.Lock()

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def _update(self):
        while self.started:
            ret, frame = self.cap.read()
            # 📍 [FIXED] ตรวจสอบเฟรมว่างและกรณีเน็ตหลุด ป้องกัน Exception ระดับล่างแครชตัวโปรแกรม
            if not ret or frame is None:
                with self.lock:
                    self.last_frame = None
                time.sleep(1.0) # หน่วงเวลาเล็กน้อยเพื่อรอทำการเชื่อมต่อสายสัญญาณใหม่
                continue
                
            with self.lock:
                self.last_frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            return False, None

    def release(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

# =========================================================
# GRAPHICS FONT
# =========================================================
_font = None
def get_font(size=20):
    global _font
    if _font: return _font
    paths = [
        "/home/agentfuel/cam_fuel/TAHOMA.TTF", "/home/agentfuel/cam_fuel/TAHOMA.ttf",
        "/home/agentfuel/cam_fuel/TAHOMA BD.TTF", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for p in paths:
        if os.path.exists(p):
            try: _font = ImageFont.truetype(p, size); return _font
            except: pass
    _font = ImageFont.load_default()
    return _font

def put_text(img, text, pos, color=(0,220,120)):
    if img is None: return img
    try:
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        draw.text(pos, str(text), font=get_font(), fill=tuple(color))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (int(color[2]),int(color[1]),int(color[0])), 2)
        return img

# =========================================================
# ROI LOADING
# =========================================================
def load_roi():
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE) as f: roi_cfg.update(json.load(f))
            print(f"[ROI] โหลดพิกัดสำเร็จ")
        except: pass

def save_roi():
    with open(ROI_FILE,"w") as f: json.dump(roi_cfg,f,indent=2)

_ds = {"s":None,"e":None,"drag":False,"done":False}
def _mcb(ev,x,y,fl,p):
    if   ev==cv2.EVENT_LBUTTONDOWN: _ds.update(s=(x,y),e=(x,y),drag=True,done=False)
    elif ev==cv2.EVENT_MOUSEMOVE and _ds["drag"]: _ds["e"]=(x,y)
    elif ev==cv2.EVENT_LBUTTONUP:   _ds.update(e=(x,y),drag=False,done=True)

def setup_roi():
    print("\n[ROI] เริ่มต้นระบุขอบเขตกรอบกล้องตู้จ่ายน้ำมัน...")
    for cam in CAM_CONFIG:
        url = rtsp_url(cam)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ret,fr = cap.read(); cap.release()
        
        if not ret or fr is None:
            print(f"[WARNING] กล้อง {cam} ตัดการเชื่อมต่อชั่วคราว ไม่สามารถดึงภาพมาเซ็ตพิกัดขอบเขตได้")
            continue
            
        H,W = fr.shape[:2]
        win = f"ROI Setup: {cam}"
        cv2.namedWindow(win,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win,_mcb)
        _ds.update(s=None,e=None,drag=False,done=False)
        while True:
            d=fr.copy(); rx=roi_cfg[cam]
            cv2.rectangle(d,(int(rx[0]*W),int(rx[1]*H)),(int(rx[2]*W),int(rx[3]*H)),(255,80,0),2)
            if _ds["s"] and _ds["e"]: cv2.rectangle(d,_ds["s"],_ds["e"],(0,0,255),2)
            cv2.imshow(win,d)
            k=cv2.waitKey(30)&0xFF
            if k==13 and _ds["done"]:
                x1, y1 = min(_ds["s"][0],_ds["e"][0]), min(_ds["s"][1],_ds["e"][1])
                x2, y2 = max(_ds["s"][0],_ds["e"][0]), max(_ds["s"][1],_ds["e"][1])
                if (x2-x1)>15 and (y2-y1)>10:
                    roi_cfg[cam]=[x1/W,y1/H,x2/W,y2/H]
                break
            elif k in (ord('s'), ord('S'), ord('q')): break
        cv2.destroyWindow(win)
    save_roi()

def init_google():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE,SCOPE)
    gc    = gspread.authorize(creds)
    sh    = gc.open_by_key(SHEET_KEY).sheet1
    return sh

def rtsp_url(cam):
    c=CAM_CONFIG[cam]
    return f"rtsp://{c['user']}:{urllib.parse.quote_plus(c['pass'])}@{c['ip']}:554{c['path']}"

# =========================================================
# ONNX DETECTOR + ADVANCED ANTI-BLUR PROCESSOR
# =========================================================
def get_roi_crop(img, rx):
    if img is None: return None
    H, W = img.shape[:2]
    x1, y1 = int(rx[0]*W), int(rx[1]*H)
    x2, y2 = int(rx[2]*W), int(rx[3]*H)
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

def preprocess_ocr(img, mode):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ซูมคมชัดสูง 3 เท่า ปรับลดขนาด Noise รอบตัวเลขดิจิทัล
    resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    
    if mode == "fuel":
        blur = cv2.bilateralFilter(resized, 9, 75, 75)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        blur = cv2.GaussianBlur(resized, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return th

def process_onnx_ocr(cam_key, frame, rx, mode):
    try:
        if frame is None: return "", None
        crop_img = get_roi_crop(frame, rx)
        if crop_img is None or crop_img.size == 0: return "", frame
        
        ch, cw = crop_img.shape[:2]
        H, W = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(crop_img, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        
        preds = np.squeeze(preds)
        if len(preds.shape) == 2 and preds.shape[0] > preds.shape[1]:
            preds = preds.T
            
        best_crop = crop_img
        max_conf = 0.30  # ขยับเกณฑ์ความแม่นยำขึ้นเพื่อตัดเฟรมสั่นไหวทิ้ง
        best_box = None
        
        if len(preds.shape) == 2:
            for pred in preds:
                if len(pred) < 5: continue
                scores = pred[4:]
                conf = np.max(scores)
                if conf > max_conf:
                    max_conf = conf
                    best_box = pred[:4]
                    
        if best_box is not None and len(best_box) == 4:
            cx, cy, bw, bh = best_box
            bx1 = int((cx - bw/2) * (cw / 640))
            by1 = int((cy - bh/2) * (ch / 640))
            bx2 = int((cx + bw/2) * (cw / 640))
            by2 = int((cy + bh/2) * (ch / 640))
            
            bx1, bx2 = max(0, bx1), min(cw, bx2)
            by1, by2 = max(0, by1), min(ch, by2)
            
            sub_crop = crop_img[by1:by2, bx1:bx2]
            if sub_crop is not None and sub_crop.size > 0:
                best_crop = sub_crop
                
                rx_x1, rx_y1 = int(rx[0]*W), int(rx[1]*H)
                cv2.rectangle(frame, (int(rx_x1 + bx1), int(rx_y1 + by1)), (int(rx_x1 + bx2), int(rx_y1 + by2)), (0, 0, 255), 2)

        if best_crop is not None and best_crop.size > 0:
            with crop_locks[cam_key]:
                crop_monitors[cam_key] = best_crop.copy()

        th_img = preprocess_ocr(best_crop, mode)
        if th_img is None: return "", frame

        if mode == "plate":
            raw = pytesseract.image_to_string(th_img, lang="tha+eng", config="--psm 7 --oem 3")
            txt = re.sub(r'[^ก-ฮ๐-๙a-zA-Z0-9]', '', raw.strip())
            return txt, frame
        else:
            best_txt = ""
            for psm in (7, 11, 4):
                raw = pytesseract.image_to_string(
                    th_img, lang="eng",
                    config=f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789."
                )
                txt = "".join(c for c in raw.strip() if c.isdigit() or c == '.')
                if len(txt) > len(best_txt): best_txt = txt
                
            # กรองเศษขยะดิจิทัล: ถ้ายอดสแกนสั้นต่ำกว่า 2 หลัก หรือเป็นแค่จุดเดี่ยว ไม่ยอมรับค่า
            if len(best_txt) < 2: return "", frame
            
            if best_txt.count('.') > 1:
                parts = best_txt.split('.')
                best_txt = parts[0] + "." + "".join(parts[1:])
            return best_txt, frame
            
    except Exception:
        return "", frame

# =========================================================
# THREAD-SAFE QUEUE CLOUD LOGGING
# =========================================================
def google_sheets_worker_thread(sh):
    """ คอยตรวจสอบและโยนข้อมูลลง Google Sheets จากคิวอย่างปลอดภัย ป้องกันโปรแกรมเด้ง """
    while True:
        row_data = sheets_logging_queue.get()
        if row_data is None: break
        
        try:
            sh.append_row(row_data)
            print(f"   [SUCCESS] บันทึกข้อมูลคลาวด์ลง Sheets สำเร็จ!: {row_data}")
        except Exception as e:
            print(f"   [API DELAY WARNING] สิทธิ์ส่งถี่เกินขีดจำกัดคลาวด์ชั่วคราว: {e}")
            
        sheets_logging_queue.task_done()
        time.sleep(1) # ป้องกันการยิง Request ถี่เกินไป

def log_to_queue(station):
    """ เตรียมชุดข้อมูล (Text ล้วนๆ ไม่มีรูปภาพ) แล้วส่งเข้า Queue """
    pcam = "CAM1"  if station=="S1" else "CAM2"
    fcam = "TAPO1" if station=="S1" else "TAPO2"

    pt  = plate_state[pcam]["text"]
    ft  = fuel_state[fcam]["saved"]

    print(f"\n[CLOUD LOGGING] เตรียมส่งข้อมูลธุรกรรมตู้จ่าย {station} เข้าคิว...")

    ts  = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # โครงสร้างคอลัมน์: Date, Plate_CAM1, Plate_CAM2, Fuel_CAM1, Fuel_CAM2
    row = [ts, "", "", "", ""]

    if station=="S1":
        row[1] = pt if pt else "ไม่พบป้าย"
        row[3] = ft if ft else "0.00"
    else:
        row[2] = pt if pt else "ไม่พบป้าย"
        row[4] = ft if ft else "0.00"

    sheets_logging_queue.put(row)

    # คืนค่าสิทธิ์ล็อกเกอร์เพื่อให้รับรถคันใหม่ได้หลังจากเสร็จงาน
    fuel_state[fcam]["lock_trigger"] = False
    plate_state[pcam]["text"] = ""

# =========================================================
# CORE CAMERA STREAM ENGINE
# =========================================================
def cam_thread(cam):
    mode  = CAM_CONFIG[cam]["mode"]
    label = CAM_CONFIG[cam]["label"]

    while True:
        rt_cap = RealtimeVideoCapture(rtsp_url(cam)).start()
        
        while rt_cap.started:
            ret, frame = rt_cap.read()
            if not ret or frame is None: time.sleep(0.02); continue

            H, W = frame.shape[:2]
            rx = roi_cfg[cam]
            disp = frame.copy()
            cv2.rectangle(disp, (int(rx[0]*W), int(rx[1]*H)), (int(rx[2]*W), int(rx[3]*H)), (255,120,0), 2)

            txt, disp = process_onnx_ocr(cam, disp, rx, mode)

            if txt:
                crop_img = get_roi_crop(frame, rx)
                if mode == "plate":
                    with frame_locks[cam]:
                        plate_state[cam]["text"] = txt
                        plate_state[cam]["img"] = crop_img
                elif mode == "fuel":
                    with frame_locks[cam]:
                        fs = fuel_state[cam]
                        # 🚀 เงื่อนไขป้องกันภาพรถวิ่งผ่านเบลอ: ค่าตัวเลขสแกนต้องตรงกันเป๊ะกับเฟรมก่อนหน้า
                        if txt != fs["text"]:
                            fs["text"] = txt
                            fs["since"] = time.time()  # เริ่มนับ 1 ใหม่อีกครั้งทันทีที่ตัวเลขแกว่งสลับไปมา
                            fs["img"] = crop_img
                        else:
                            # ตัวเลขนิ่งสนิทแล้วเป็นระยะเวลาเท่าไหร่
                            elapsed = time.time() - fs["since"]
                            now = time.time()
                            since_save = now - fs["last_save"]
                            
                            # เช็กเงื่อนไข: นิ่งเกิน 5 วิ + ไม่ซ้ำค่าเดิม + พ้นระยะคูลดาวน์ API 30 วิ + ไม่มีงานค้าง
                            if (elapsed >= FUEL_STABLE_SEC and txt != fs["saved"] and 
                                since_save >= SAVE_COOLDOWN and not fs["lock_trigger"]):
                                
                                fs["saved"] = txt
                                fs["last_save"] = now
                                fs["lock_trigger"] = True # ล็อกการทำงานชั่วคราวห้ามสั่งเบิ้ลบันทึก
                                
                                sid = "S1" if cam == "TAPO1" else "S2"
                                # ✅ เรียกใช้งานระบบ Queue แทนการต่อ API ตรง
                                log_to_queue(sid)

            if disp is not None:
                ov = disp.copy()
                cv2.rectangle(ov, (0,0), (W,44), (8,8,8), -1)
                cv2.addWeighted(ov, 0.55, disp, 0.45, 0, disp)

                with frame_locks[cam]:
                    if mode == "plate":
                        ui = plate_state[cam]["text"] or "รอวัตถุ..."
                        disp = put_text(disp, f"[{label}] {ui}", (8,10), (0,220,120))
                    else:
                        fs = fuel_state[cam]
                        ui = fs['text'] or 'รอมิเตอร์น้ำมันนิ่งสนิท...'
                        disp = put_text(disp, f"[{label}] {ui}", (8,10), (0,220,120))

                with frame_locks[cam]:
                    latest_frames[cam] = disp.copy()

        rt_cap.release()
        time.sleep(3)

def blank_cell(cam):
    c = np.zeros((CELL_H,CELL_W,3), dtype=np.uint8)
    return put_text(c, f"[{cam}] รอภาพสตรีมสด...", (20,CELL_H//2-12), (100,100,100))

# =========================================================
# MAIN EXECUTIVE
# =========================================================
def main():
    print("\n"+"="*60)
    print("Smart Refueling System v15.7 [Strict Anti-Blur Engine]")
    print("="*60+"\n")

    load_roi()
    ans = input("ต้องการปรับแต่งกรอบพิกัดสแกนหน้าตู้ใหม่หรือไม่? (y/n): ").strip().lower()
    if ans == "y": setup_roi()

    sh = init_google()
    
    # สตาร์ท Worker ของระบบ Queue ทันที
    threading.Thread(target=google_sheets_worker_thread, args=(sh,), daemon=True).start()

    for k in CAM_CONFIG:
        threading.Thread(target=cam_thread, args=(k,), daemon=True).start()

    print("\n[*] บูทระบบตัวกรองป้องกันภาพเคลื่อนไหว และระงับ Drive API เรียบร้อยแล้ว...")

    while True:
        cells = []
        for k in ["CAM1","CAM2","TAPO1","TAPO2"]:
            with frame_locks[k]:
                fr = latest_frames[k].copy() if latest_frames[k] is not None else None
            cells.append(cv2.resize(fr, (CELL_W, CELL_H)) if fr is not None else blank_cell(k))

        grid = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])

        def _st(k):
            fs = fuel_state[k]
            if not fs["text"]: return f"{k}:รอนิ่ง"
            el = time.time() - fs["since"]
            return f"{k}:{fs['text']} ({el:.0f}s)"

        bar = (f"CAM1:{plate_state['CAM1']['text'] or 'รอวัตถุ'}  "
               f"CAM2:{plate_state['CAM2']['text'] or 'รอวัตถุ'}  |  "
               f"{_st('TAPO1')}  {_st('TAPO2')}  "
               f"[Q=ปิดระบบ]")
        
        grid = np.ascontiguousarray(grid, dtype=np.uint8)
        grid = put_text(grid, bar, (8, grid.shape[0]-28), (255,255,255))

        cv2.imshow("Smart Refueling v15 ONNX Dashboard", grid)

        sub_views = []
        for k in ["CAM1", "CAM2", "TAPO1", "TAPO2"]:
            with crop_locks[k]:
                c_img = crop_monitors[k].copy() if crop_monitors[k] is not None else None
            if c_img is not None and c_img.size > 0:
                sub_views.append(cv2.resize(c_img, (240, 135)))
            else:
                blank_sub = np.zeros((135, 240, 3), dtype=np.uint8) + 40
                cv2.putText(blank_sub, f"{k} Waiting...", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120), 1)
                sub_views.append(blank_sub)
        
        cv2.imshow("Model Target BBox Crops (Real-time)", np.vstack(sub_views))

        k2 = cv2.waitKey(30) & 0xFF
        if k2 == ord('q'): break

    cv2.destroyAllWindows()
    sheets_logging_queue.put(None) # แจ้งคิวให้หยุดรออย่างปลอดภัย
    print("[*] ปิดระบบเสร็จสิ้น")

if __name__ == "__main__":
    main()
