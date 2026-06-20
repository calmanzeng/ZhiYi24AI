"""
venipuncture Scoring Plugin
"""
import numpy as np
from typing import Dict, List, Any, Optional
class SkillScorer:
    def __init__(self, config=None):
        self.config = config or {}
        self.skill_name = "venipuncture"
        self.landmark_history = {}
        self.feature_cache = {}
    def extract_features(self, landmarks):
        f = {}

        lw = self._lm(landmarks, "left_wrist"); rw = self._lm(landmarks, "right_wrist")
        le = self._lm(landmarks, "left_elbow"); re = self._lm(landmarks, "right_elbow")
        rs = self._lm(landmarks, "right_shoulder"); ls = self._lm(landmarks, "left_shoulder")
        f["needle_angle"] = self._a(rs, re, rw)
        f["dom_x"] = rw[0]; f["dom_y"] = rw[1]
        if "dom_x" in self.landmark_history and len(self.landmark_history["dom_x"]) >= 10:
            xh = self.landmark_history["dom_x"][-15:]; yh = self.landmark_history["dom_y"][-15:]
            f["hand_stability"] = float(np.sqrt(np.var(xh)+np.var(yh)))
        if "left_wrist" in self.landmark_history:
            lt = self.landmark_history["left_wrist"][-15:]
            if len(lt) >= 5:
                dp = [np.sqrt((lt[i][0]-lt[i-1][0])**2+(lt[i][1]-lt[i-1][1])**2) for i in range(1, len(lt))]
                f["smoothness"] = float(1.0/(1.0+np.var(dp)*10)) if dp else 0.8
        if "right_wrist" in self.landmark_history and "left_wrist" in self.landmark_history:
            rt = self.landmark_history["right_wrist"][-10:]; lt = self.landmark_history["left_wrist"][-10:]
            if len(rt)>=5 and len(lt)>=5:
                rv = np.var([p[0] for p in rt])+np.var([p[1] for p in rt])
                lv = np.var([p[0] for p in lt])+np.var([p[1] for p in lt])
                f["fine_motor"] = float(1.0/(1.0+rv+lv))
    
        for k,v in f.items():
            self.landmark_history.setdefault(k,[]).append(v)
            if len(self.landmark_history[k]) > 300: self.landmark_history[k] = self.landmark_history[k][-300:]
        self.feature_cache = f; return f
    def _lm(self, lm, name):
        if name in lm:
            v = lm[name]
            if isinstance(v, (list, tuple)): return (v[0], v[1], v[2], v[3] if len(v) > 3 else 1.0)
        return (0,0,0,0)
    def _d(self, a, b):
        return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))
    def _a(self, a, b, c):
        ba = (a[0]-b[0], a[1]-b[1], a[2]-b[2]); bc = (c[0]-b[0], c[1]-b[1], c[2]-b[2])
        dot = ba[0]*bc[0]+ba[1]*bc[1]+ba[2]*bc[2]
        mag = np.sqrt(ba[0]**2+ba[1]**2+ba[2]**2)*np.sqrt(bc[0]**2+bc[1]**2+bc[2]**2)
        if mag < 1e-6: return 90
        return np.degrees(np.arccos(np.clip(dot/mag,-1,1)))
    def compute_metrics(self, features):
        m = {}

        ideal_angle = 30  # 静脉穿刺理想角度
        dev = abs(features.get("needle_angle", 30)-ideal_angle)
        m["angle_score"] = round(max(40,100-dev*1.5), 1)
        m["stability"] = round(max(40,100-features.get("hand_stability",0)*40), 1)
        m["smoothness"] = round(min(100,features.get("smoothness",0.8)*100), 1)
        m["precision"] = round(min(100,features.get("fine_motor",0.7)*100), 1)
    
        return m
    def get_feedback(self, metrics, features):
        fb = [{"type":"overall","message":f"总分：{min(100,max(0,sum(m*0.25 for m in metrics.values()))):.0f}/100","severity":"info"}]

        na = features.get("needle_angle", 30)
        if na<15 or na>45: fb.append({"type":"warning","message":f"进针角度({na:.0f})偏移理想值30度","severity":"warning","metric":"angle_score"})
        if features.get("hand_stability",0) > 1.5: fb.append({"type":"warning","message":"手部不够稳定","severity":"warning","metric":"stability"})
        if features.get("fine_motor",0.7) < 0.5: fb.append({"type":"info","message":"精细控制有提升空间","severity":"info","metric":"precision"})
    
        return fb if len(fb)>1 else fb+[{"type":"info","message":"操作完成","severity":"info"}]
    def reset(self):
        self.landmark_history = {}; self.feature_cache = {}
