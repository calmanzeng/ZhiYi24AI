"""
Dashboard — 实时 HUD 渲染
从 Pipeline 回调中获取数据，渲染评分界面
"""
import cv2
import numpy as np
from collections import deque
import time


class Dashboard:
    """技能训练实时 HUD"""
    
    COLORS = {
        "bg": (30, 30, 35),
        "panel": (45, 45, 55),
        "green": (50, 255, 100),
        "red": (50, 80, 255),
        "yellow": (50, 255, 255),
        "white": (255, 255, 255),
        "gray": (160, 160, 170),
        "blue": (255, 180, 50),
    }
    
    def __init__(self, panel_width=280):
        self.pw = panel_width
        self.scores_hist = deque(maxlen=200)
        self.fps_q = deque(maxlen=30)
        self.lt = time.time()
    
    def tick(self):
        now = time.time()
        dt = now - self.lt
        self.lt = now
        if dt > 0:
            self.fps_q.append(1.0 / dt)
    
    def render(self, frame, metrics: dict, landmarks: dict,
               feedback: list, wrist_history: list) -> np.ndarray:
        """渲染完整 HUD"""
        h, w = frame.shape[:2]
        px = w - self.pw
        
        # 面板背景
        ov = frame.copy()
        cv2.rectangle(ov, (px, 0), (w, h), self.COLORS["panel"], -1)
        frame = cv2.addWeighted(frame, 0.7, ov, 0.3, 0)
        cv2.line(frame, (px, 0), (px, h), (80, 80, 90), 1)
        
        # 标题
        cv2.rectangle(frame, (px, 0), (w, 45), self.COLORS["bg"], -1)
        cv2.putText(frame, "AI Scorer", (px + 10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS["white"], 2)
        
        fps = np.mean(self.fps_q) if self.fps_q else 0
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 70, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS["gray"], 1)
        
        if not metrics:
            cv2.putText(frame, "等待操作...", (px + 20, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLORS["yellow"], 2)
            return frame
        
        # ---- 分数 ----
        overall = metrics.get("overall", {}).get("value", 0)
        sc = self.COLORS["green"] if overall >= 80 else (
            self.COLORS["yellow"] if overall >= 60 else self.COLORS["red"]
        )
        cv2.putText(frame, str(overall), (px + 10, 85),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, sc, 3)
        cv2.putText(frame, "/100", (px + 90, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS["white"], 1)
        
        # 等级
        grade = "A" if overall >= 85 else ("B" if overall >= 70 else (
            "C" if overall >= 55 else "D"))
        cv2.rectangle(frame, (px + 145, 53), (px + 185, 83), sc, -1)
        cv2.putText(frame, grade, (px + 153, 76),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, self.COLORS["bg"], 2)
        
        # ---- 指标表 ----
        y = 110
        cv2.line(frame, (px + 5, y), (w - 5, y), (60, 60, 70), 1)
        y += 15
        
        metric_order = [
            ("compression_rate", "CPM", "CPM"),
            ("rhythm_consistency", "一致性", "%"),
            ("depth_index", "深度", ""),
            ("compression_count", "次数", ""),
        ]
        
        for key, label, unit in metric_order:
            if key in metrics:
                val = metrics[key].get("value", 0)
                score = metrics[key].get("score", 0)
                color = self.COLORS["green"] if score >= 0.8 else (
                    self.COLORS["yellow"] if score >= 0.5 else self.COLORS["white"]
                )
                cv2.putText(frame, label, (px + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS["gray"], 1)
                cv2.putText(frame, f"{val:.1f} {unit}" if unit else f"{val:.3f}",
                            (px + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 24
        
        # ---- 反馈 ----
        y += 5
        cv2.line(frame, (px + 5, y), (w - 5, y), (60, 60, 70), 1)
        y += 15
        
        for fb in feedback:
            sev = fb.get("severity", "")
            fc = self.COLORS["green"] if "success" in sev else (
                self.COLORS["yellow"] if "warning" in sev else self.COLORS["red"]
            )
            cv2.putText(frame, fb["message"], (px + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, fc, 1)
            y += 18
        
        # ---- 趋势图 ----
        if len(self.scores_hist) > 3:
            y += 10
            self._draw_trend(frame, px + 10, y, self.pw - 20, 40, list(self.scores_hist))
        
        # ---- 手腕轨迹 ----
        if wrist_history and len(wrist_history) > 3:
            y += 50
            self._draw_trend(frame, px + 10, y, self.pw - 20, 35,
                             list(wrist_history), self.COLORS["yellow"])
        
        # ---- 骨架 ----
        if landmarks:
            self._draw_skeleton(frame, landmarks)
        
        return frame
    
    def _draw_trend(self, frame, x, y, w, h, data, color=None):
        if color is None:
            color = self.COLORS["green"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 60, 70), 1)
        d = np.array(data)
        if len(d) < 2: return
        dmin, dmax = np.min(d), np.max(d)
        if dmax == dmin: dmax = dmin + 1
        pts = d[-min(200, len(d)):]
        for i in range(len(pts)-1):
            x1 = x + int(i/max(len(pts)-1, 1)*w)
            y1 = y + h - int((pts[i]-dmin)/(dmax-dmin)*h)
            x2 = x + int((i+1)/max(len(pts)-1, 1)*w)
            y2 = y + h - int((pts[i+1]-dmin)/(dmax-dmin)*h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 1)
    
    def _draw_skeleton(self, frame, landmarks):
        h, w = frame.shape[:2]
        def pt(name):
            lm = landmarks.get(name, {})
            return (int(lm.get("x", 0) * w), int(lm.get("y", 0) * h))
        
        # Arms
        for wi, ei, si in [("left_wrist", "left_elbow", "left_shoulder"),
                            ("right_wrist", "right_elbow", "right_shoulder")]:
            if all(k in landmarks for k in [wi, ei, si]):
                cv2.line(frame, pt(si), pt(ei), self.COLORS["green"], 2)
                cv2.line(frame, pt(ei), pt(wi), self.COLORS["green"], 2)
                cv2.circle(frame, pt(wi), 8, self.COLORS["yellow"], -1)
        
        # Shoulders
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            cv2.line(frame, pt("left_shoulder"), pt("right_shoulder"),
                     self.COLORS["green"], 2)
        
        # Wrist center
        if "left_wrist" in landmarks and "right_wrist" in landmarks:
            wx = (landmarks["left_wrist"]["x"] + landmarks["right_wrist"]["x"]) / 2
            wy = (landmarks["left_wrist"]["y"] + landmarks["right_wrist"]["y"]) / 2
            cv2.circle(frame, (int(wx*w), int(wy*h)), 12, (0, 255, 255), 2)
    
    def update_score(self, score):
        if score is not None:
            self.scores_hist.append(score)
