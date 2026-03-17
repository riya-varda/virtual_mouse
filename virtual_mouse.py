"""
╔══════════════════════════════════════════════════════════════════╗
║           GESTURE CONTROL SYSTEM  v2.0                          ║
║           Fixed for MediaPipe 0.10+  (Tasks API)                ║
║                                                                  ║
║  Gestures:                                                       ║
║   Index finger only      -> Move cursor                         ║
║   Index + pinch          -> Left click                          ║
║   Double pinch           -> Double click                        ║
║   Pinch + hold + move    -> Drag                                ║
║   Index+Mid + pinch      -> Right click                         ║
║   2 fingers vertical     -> Scroll up / down                    ║
║   3-finger swipe right   -> Next track                          ║
║   3-finger swipe left    -> Prev track                          ║
║   Open palm (all 5 up)   -> Play / Pause                        ║
║   Fist (hold 1s)         -> Screenshot                          ║
║   Thumb only up          -> Volume up                           ║
║   Pinky only up          -> Volume down                         ║
║   Rock sign (I + P)      -> Mute toggle                         ║
║   Ring only up           -> Brightness up                       ║
║   Ring + Pinky up        -> Brightness down                     ║
║                                                                  ║
║  Press ESC to quit                                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import pyautogui
import numpy as np
import time
import os
import platform
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ─────────────────────── MODEL DOWNLOAD ────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task (~9 MB) — one time only...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Download complete.")

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    camera_index:     int   = 0
    capture_w:        int   = 1280
    capture_h:        int   = 720
    target_fps:       int   = 60
    frame_pad:        int   = 130
    kalman_q:         float = 1e-3
    kalman_r:         float = 5e-2
    pinch_dist_px:    float = 36
    scroll_x_gap:     float = 28
    scroll_min_dy:    float = 0.035
    scroll_speed:     int   = 6
    swipe_min_dx:     float = 0.20
    swipe_max_time:   float = 0.55
    confirm_frames:   int   = 3
    click_cd:         float = 0.35
    gesture_cd:       float = 0.75
    volume_cd:        float = 0.28
    swipe_cd:         float = 0.90
    double_click_gap: float = 0.38
    detect_conf:      float = 0.80
    track_conf:       float = 0.75

CFG = Config()

# ══════════════════════════════════════════════════════════════════
#  KALMAN FILTER  — smooth, lag-free cursor
# ══════════════════════════════════════════════════════════════════

class Kalman1D:
    def __init__(self, q: float, r: float):
        self.q = q
        self.r = r
        self.x = 0.0
        self.p = 1.0

    def update(self, z: float) -> float:
        self.p += self.q
        k       = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x

class KalmanCursor:
    def __init__(self):
        self.kx = Kalman1D(CFG.kalman_q, CFG.kalman_r)
        self.ky = Kalman1D(CFG.kalman_q, CFG.kalman_r)
        self.sw, self.sh = pyautogui.size()

    def from_frame(self, lm_x, lm_y, fw, fh):
        sx = np.interp(lm_x, (CFG.frame_pad, fw - CFG.frame_pad), (0, self.sw))
        sy = np.interp(lm_y, (CFG.frame_pad, fh - CFG.frame_pad), (0, self.sh))
        return self.kx.update(sx), self.ky.update(sy)

# ══════════════════════════════════════════════════════════════════
#  HAND STATE
# ══════════════════════════════════════════════════════════════════

@dataclass
class HandState:
    lm:      List[Tuple[int, int]]
    nlm:     List[Tuple[float, float]]
    fu:      List[bool]              # [thumb, index, middle, ring, pinky]
    frame_w: int
    frame_h: int

    @property
    def index(self):  return self.lm[8]
    @property
    def middle(self): return self.lm[12]

    def dist(self, a: int, b: int) -> float:
        return float(np.hypot(self.lm[a][0]-self.lm[b][0],
                              self.lm[a][1]-self.lm[b][1]))

    @staticmethod
    def from_result(hand_landmarks, fw: int, fh: int) -> "HandState":
        lm  = [(int(l.x * fw), int(l.y * fh)) for l in hand_landmarks]
        nlm = [(l.x, l.y) for l in hand_landmarks]
        # Thumb uses x-axis (mirrored camera: right hand tip is LEFT of knuckle)
        thumb_up = lm[4][0] < lm[3][0]
        def fup(t, p): return lm[t][1] < lm[p][1]
        fu = [thumb_up, fup(8,6), fup(12,10), fup(16,14), fup(20,18)]
        return HandState(lm=lm, nlm=nlm, fu=fu, frame_w=fw, frame_h=fh)

# ══════════════════════════════════════════════════════════════════
#  GESTURE STATE MACHINE  — eliminates false triggers
# ══════════════════════════════════════════════════════════════════

class GSM:
    def __init__(self, cooldown: float):
        self.cooldown  = cooldown
        self.streak    = 0
        self.confirmed = False
        self.last_fire = 0.0

    def tick(self, active: bool) -> bool:
        now = time.time()
        if now - self.last_fire < self.cooldown:
            return False
        if active:
            self.streak += 1
            if self.streak >= CFG.confirm_frames and not self.confirmed:
                self.confirmed = True
                self.last_fire = now
                return True
        else:
            self.streak    = 0
            self.confirmed = False
        return False

# ══════════════════════════════════════════════════════════════════
#  GESTURE CONTROLLER
# ══════════════════════════════════════════════════════════════════

@dataclass
class GestureEvent:
    name:    str
    message: str
    color:   Tuple[int, int, int]

class GestureController:
    def __init__(self):
        self.cursor       = KalmanCursor()
        self.system       = platform.system()
        self.dragging     = False
        self.last_pinch_t = 0.0
        self.scroll_buf   = deque(maxlen=6)
        self.swipe_start_x: Optional[float] = None
        self.swipe_start_t: float           = 0.0
        self._last_swipe:   float           = 0.0
        self.cx = self.cursor.sw // 2
        self.cy = self.cursor.sh // 2

        self.gsm: Dict[str, GSM] = {
            "left_click":  GSM(CFG.click_cd),
            "right_click": GSM(CFG.click_cd),
            "play_pause":  GSM(CFG.gesture_cd),
            "screenshot":  GSM(2.0),
            "volume_up":   GSM(CFG.volume_cd),
            "volume_down": GSM(CFG.volume_cd),
            "mute":        GSM(CFG.gesture_cd),
            "bright_up":   GSM(CFG.gesture_cd),
            "bright_down": GSM(CFG.gesture_cd),
        }

    def _g(self, name, active): return self.gsm[name].tick(active)

    def _set_brightness(self, delta):
        if self.system == "Windows":
            try:
                import wmi
                o   = wmi.WMI(namespace="wmi")
                cur = o.WmiMonitorBrightness()[0].CurrentBrightness
                o.WmiMonitorBrightnessMethods()[0].WmiSetBrightness(
                    max(0, min(100, cur + delta)), 0)
            except Exception:
                pass

    def process(self, hs: HandState) -> List[GestureEvent]:
        ev: List[GestureEvent] = []
        T, I, M, R, P = hs.fu
        d_ti = hs.dist(4, 8)
        d_tm = hs.dist(4, 12)

        # 1. CURSOR (index only)
        if I and not M:
            cx, cy = self.cursor.from_frame(hs.index[0], hs.index[1], hs.frame_w, hs.frame_h)
            self.cx, self.cy = int(cx), int(cy)
            if d_ti < CFG.pinch_dist_px:
                if not self.dragging:
                    pyautogui.mouseDown()
                    self.dragging = True
                    ev.append(GestureEvent("drag_start", "DRAG START", (255,160,50)))
                pyautogui.moveTo(cx, cy)
            else:
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
                    ev.append(GestureEvent("drag_end", "DRAG RELEASED", (255,200,100)))
                pyautogui.moveTo(cx, cy)

        # 2. LEFT CLICK / DOUBLE CLICK (index + middle)
        elif I and M and not R:
            cx, cy = self.cursor.from_frame(hs.index[0], hs.index[1], hs.frame_w, hs.frame_h)
            self.cx, self.cy = int(cx), int(cy)
            pyautogui.moveTo(cx, cy)
            if self._g("left_click", d_ti < CFG.pinch_dist_px):
                now = time.time()
                if now - self.last_pinch_t < CFG.double_click_gap:
                    pyautogui.doubleClick()
                    ev.append(GestureEvent("dbl", "DOUBLE CLICK", (255,230,0)))
                else:
                    pyautogui.click()
                    ev.append(GestureEvent("lclick", "LEFT CLICK", (0,255,180)))
                self.last_pinch_t = now

        # 3. RIGHT CLICK
        if self._g("right_click", d_tm < CFG.pinch_dist_px and I and M):
            pyautogui.rightClick()
            ev.append(GestureEvent("rclick", "RIGHT CLICK", (255,80,80)))

        # 4. SCROLL
        if I and M and not R and not T:
            dx = abs(hs.lm[8][0] - hs.lm[12][0])
            dy = hs.lm[8][1] - hs.lm[12][1]
            if dx < CFG.scroll_x_gap:
                self.scroll_buf.append(dy / hs.frame_h)
                avg = float(np.mean(self.scroll_buf))
                if abs(avg) > CFG.scroll_min_dy:
                    pyautogui.scroll(CFG.scroll_speed if avg < 0 else -CFG.scroll_speed)

        # 5. SWIPE (3 fingers, lateral wrist motion)
        if I and M and R and not P:
            wx = hs.nlm[0][0]
            if self.swipe_start_x is None:
                self.swipe_start_x = wx
                self.swipe_start_t = time.time()
            else:
                delta   = wx - self.swipe_start_x
                elapsed = time.time() - self.swipe_start_t
                if (elapsed < CFG.swipe_max_time and
                        abs(delta) > CFG.swipe_min_dx and
                        time.time() - self._last_swipe > CFG.swipe_cd):
                    if delta > 0:
                        pyautogui.press("nexttrack")
                        ev.append(GestureEvent("next", "NEXT TRACK", (0,220,255)))
                    else:
                        pyautogui.press("prevtrack")
                        ev.append(GestureEvent("prev", "PREV TRACK", (0,220,255)))
                    self._last_swipe   = time.time()
                    self.swipe_start_x = None
        else:
            self.swipe_start_x = None

        # 6. PLAY / PAUSE (open palm)
        if self._g("play_pause", all(hs.fu)):
            pyautogui.press("space")
            ev.append(GestureEvent("pp", "PLAY / PAUSE", (255,255,80)))

        # 7. SCREENSHOT (fist, held)
        if self._g("screenshot", not any(hs.fu)):
            pyautogui.screenshot("gesture_screenshot.png")
            ev.append(GestureEvent("ss", "SCREENSHOT SAVED", (200,100,255)))

        # 8. VOLUME UP (thumb only)
        if self._g("volume_up", T and not I and not M and not R and not P):
            pyautogui.press("volumeup")
            ev.append(GestureEvent("vu", "VOLUME UP", (80,255,120)))

        # 9. VOLUME DOWN (pinky only)
        if self._g("volume_down", P and not I and not M and not R and not T):
            pyautogui.press("volumedown")
            ev.append(GestureEvent("vd", "VOLUME DOWN", (255,200,80)))

        # 10. MUTE (rock sign: index + pinky)
        if self._g("mute", I and P and not M and not R):
            pyautogui.press("volumemute")
            ev.append(GestureEvent("mute", "MUTED", (255,80,80)))

        # 11. BRIGHTNESS UP (ring only)
        if self._g("bright_up", R and not I and not M and not P):
            self._set_brightness(+10)
            ev.append(GestureEvent("bu", "BRIGHTNESS UP", (255,240,80)))

        # 12. BRIGHTNESS DOWN (ring + pinky)
        if self._g("bright_down", R and P and not I and not M):
            self._set_brightness(-10)
            ev.append(GestureEvent("bd", "BRIGHTNESS DOWN", (140,180,255)))

        return ev

# ══════════════════════════════════════════════════════════════════
#  RENDERER
# ══════════════════════════════════════════════════════════════════

GESTURE_MAP = [
    ("Index finger",       "Move cursor"),
    ("Index + pinch",      "Left click"),
    ("Double pinch",       "Double click"),
    ("Pinch + move",       "Drag"),
    ("I+M + pinch",        "Right click"),
    ("2 fingers vertical", "Scroll"),
    ("3-finger swipe R",   "Next track"),
    ("3-finger swipe L",   "Prev track"),
    ("Open palm",          "Play / Pause"),
    ("Fist (hold)",        "Screenshot"),
    ("Thumb only",         "Volume up"),
    ("Pinky only",         "Volume down"),
    ("Rock sign (I+P)",    "Mute"),
    ("Ring only",          "Brightness up"),
    ("Ring + Pinky",       "Brightness down"),
]

C_ACCENT = (0, 255, 180)
C_DIM    = (50, 70, 60)
C_GOLD   = (30, 210, 255)
C_WHITE  = (210, 220, 220)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

class Toast:
    def __init__(self, text, color, duration=1.4):
        self.text = text; self.color = color
        self.duration = duration; self.born = time.time()
    @property
    def alive(self): return time.time()-self.born < self.duration
    @property
    def alpha(self):
        return float(np.clip(1-(( time.time()-self.born)/self.duration)**3, 0, 1))

class Renderer:
    def __init__(self, fw, fh):
        self.fw = fw; self.fh = fh
        self.toasts: List[Toast] = []
        self.fps_buf = deque(maxlen=40)
        self.prev_t  = time.time()
        self._build_panel()

    def _build_panel(self):
        row_h = 19; pad = 10; pw = 308
        ph = len(GESTURE_MAP)*row_h + pad*2 + 30
        p  = np.zeros((ph, pw, 3), dtype=np.uint8)
        p[:] = (8,14,12)
        cv2.putText(p,"GESTURE REFERENCE",(pad,pad+14),cv2.FONT_HERSHEY_DUPLEX,0.47,C_GOLD,1)
        cv2.line(p,(pad,pad+20),(pw-pad,pad+20),C_DIM,1)
        for i,(g,a) in enumerate(GESTURE_MAP):
            y = pad+32+i*row_h
            cv2.putText(p,g,(pad,y),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_WHITE,1)
            cv2.putText(p,f"-> {a}",(170,y),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_ACCENT,1)
        self._panel=p; self._pw=pw; self._ph=ph

    def push(self, ev: GestureEvent):
        self.toasts.append(Toast(ev.message, ev.color))

    def draw(self, frame, hs: Optional[HandState], ctrl: GestureController, detected: bool):
        now = time.time()
        self.fps_buf.append(1.0/max(now-self.prev_t,1e-6))
        self.prev_t = now
        fps = int(np.mean(self.fps_buf))

        # Panel
        ph,pw = self._ph,self._pw
        y0,x0 = self.fh-ph-10,10
        roi   = frame[y0:y0+ph, x0:x0+pw]
        frame[y0:y0+ph, x0:x0+pw] = cv2.addWeighted(roi,0.15,self._panel,0.85,0)

        # Active zone
        p = CFG.frame_pad
        cv2.rectangle(frame,(p,p),(self.fw-p,self.fh-p),(30,55,45),1)
        L = 22
        for cx2,cy2,sx,sy in [(p,p,1,1),(self.fw-p,p,-1,1),(p,self.fh-p,1,-1),(self.fw-p,self.fh-p,-1,-1)]:
            cv2.line(frame,(cx2,cy2),(cx2+sx*L,cy2),C_ACCENT,2)
            cv2.line(frame,(cx2,cy2),(cx2,cy2+sy*L),C_ACCENT,2)

        cv2.putText(frame,"GESTURE CONTROL  v2.0",(10,30),cv2.FONT_HERSHEY_DUPLEX,0.6,C_ACCENT,1)

        # Status
        st = "HAND DETECTED" if detected else "SCANNING..."
        sc = C_ACCENT if detected else (80,80,80)
        tw = cv2.getTextSize(st,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)[0][0]
        bx = self.fw-tw-28
        cv2.circle(frame,(bx-10,22),5,sc,-1)
        cv2.putText(frame,st,(bx,27),cv2.FONT_HERSHEY_SIMPLEX,0.44,sc,1)
        cv2.putText(frame,f"FPS {fps}",(self.fw-80,self.fh-12),cv2.FONT_HERSHEY_SIMPLEX,0.42,(50,70,60),1)

        if hs is not None:
            # Finger indicators
            for j,(up,lb) in enumerate(zip(hs.fu,["T","I","M","R","P"])):
                cv2.putText(frame,lb,(self.fw//2-70+j*30,40),cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,C_ACCENT if up else C_DIM,2)

            # Pinch meter
            d = hs.dist(4,8)
            ratio = float(np.clip(1-d/(CFG.pinch_dist_px*2),0,1))
            ac = (0,int(ratio*255),int((1-ratio)*200))
            cx2,cy2 = 38,self.fh-42
            cv2.circle(frame,(cx2,cy2),17,C_DIM,2)
            if ratio>0.05:
                cv2.ellipse(frame,(cx2,cy2),(17,17),-90,0,int(360*ratio),ac,2)
            cv2.putText(frame,"PINCH",(cx2-19,cy2+30),cv2.FONT_HERSHEY_SIMPLEX,0.3,C_DIM,1)

            # Crosshair on index tip
            ix,iy = hs.index
            cv2.line(frame,(ix-14,iy),(ix+14,iy),C_ACCENT,1)
            cv2.line(frame,(ix,iy-14),(ix,iy+14),C_ACCENT,1)
            cv2.circle(frame,(ix,iy),4,C_ACCENT,-1)

            # Skeleton
            for a_i,b_i in CONNECTIONS:
                cv2.line(frame,hs.lm[a_i],hs.lm[b_i],(0,80,60),3)
                cv2.line(frame,hs.lm[a_i],hs.lm[b_i],(0,200,140),1)
            for i2,(x2,y2) in enumerate(hs.lm):
                r = 5 if i2 in (4,8,12,16,20) else 3
                cv2.circle(frame,(x2,y2),r+1,(0,60,40),-1)
                cv2.circle(frame,(x2,y2),r,(0,220,160),-1)

        if ctrl.dragging:
            cv2.putText(frame,"[ DRAGGING ]",(self.fw//2-65,self.fh-15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,160,50),2)

        # Toasts
        self.toasts = [t for t in self.toasts if t.alive]
        for i,toast in enumerate(reversed(self.toasts)):
            a2  = toast.alpha
            col = tuple(int(c*a2) for c in toast.color)
            y   = self.fh-70-i*38
            tw2 = cv2.getTextSize(toast.text,cv2.FONT_HERSHEY_DUPLEX,0.78,2)[0][0]
            tx  = self.fw//2-tw2//2
            ov  = frame.copy()
            cv2.rectangle(ov,(tx-12,y-26),(tx+tw2+12,y+6),(10,18,14),-1)
            cv2.addWeighted(ov,a2*0.7,frame,1-a2*0.7,0,frame)
            cv2.putText(frame,toast.text,(tx,y),cv2.FONT_HERSHEY_DUPLEX,0.78,col,2)

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=CFG.detect_conf,
        min_hand_presence_confidence=CFG.track_conf,
        min_tracking_confidence=CFG.track_conf,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CFG.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.capture_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.capture_h)
    cap.set(cv2.CAP_PROP_FPS,          CFG.target_fps)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ctrl = GestureController()
    rend = Renderer(fw, fh)

    print("Gesture Control v2.0 started — press ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(time.monotonic() * 1000)
        result   = landmarker.detect_for_video(mp_image, ts_ms)

        hs: Optional[HandState] = None
        events = []

        if result.hand_landmarks:
            hs     = HandState.from_result(result.hand_landmarks[0], fw, fh)
            events = ctrl.process(hs)

        rend.draw(frame, hs, ctrl, hs is not None)
        for ev in events:
            rend.push(ev)

        cv2.imshow("Gesture Control v2.0  [ESC to quit]", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if ctrl.dragging:
        pyautogui.mouseUp()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()