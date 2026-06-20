"""
bone_marrow_puncture Scoring Plugin
"""
import numpy as np
from typing import Dict, List, Any, Optional
class SkillScorer:
    def __init__(self, config=None):
        self.config = config or {}
        self.skill_name = "bone_marrow_puncture"
        self.landmark_history = {}
        self.feature_cache = {}
    def extract_features(self, landmarks):
        f = {}

        ls = self._lm(landmarks, "left_shoulder"); rs = self._lm(landmarks, "right_shoulder")
        le = self._lm(landmarks, "left_elbow"); re = self._lm(landmarks, "right_elbow")
        lw = self._lm(landmarks, "left_wrist"); rw = self._lm(landmarks, "right_wrist")
        lh = self._lm(landmarks, "left_hip"); rh = self._lm(landmarks, "right_hip")
        mid_hip = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2)
        f["insertion_angle"] = self._a(mid_hip, rs, rw)
        f["dom_x"] = rw[0]; f["dom_y"] = rw[1]
        if "dom_x" in self.landmark_history and len(self.landmark_history["dom_x"]) >= 10:
            xh = self.landmark_history["dom_x"][-15:]; yh = self.landmark_history["dom_y"][-15:]
            f["hand_stability"] = float(np.sqrt(np.var(xh)+np.var(yh)))
        if "dom_y" in self.landmark_history and len(self.landmark_history["dom_y"]) >= 10:
            s = np.std(self.landmark_history["dom_y"][-10:])
            f["steady_pressure"] = float(1.0/(1.0+s*20))  # lower std = more steady
        if "left_wrist" in self.landmark_history:
            lh_t = self.landmark_history["left_wrist"][-15:]
            if len(lh_t) >= 5:
                f["wrist_range"] = float(max(p[1] for p in lh_t)-min(p[1] for p in lh_t))
    
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

        dev = abs(features.get("insertion_angle", 90)-90)
        m["angle_score"] = round(max(0,100-dev*1.5), 1)
        m["stability"] = round(max(40,100-features.get("hand_stability",0)*30), 1)
        m["pressure_score"] = round(features.get("steady_pressure",0.8)*100, 1)
        wr = features.get("wrist_range", 0.05)
        m["smoothness"] = round(85 if 0.02<=wr<=0.08 else (60 if wr>0.1 else 75), 1)
    
        return m
    def get_feedback(self, metrics, features):
        fb = [{"type":"overall","message":f"总分：{min(100,max(0,sum(m*0.25 for m in metrics.values()))):.0f}/100","severity":"info"}]

        ia = features.get("insertion_angle", 90)
        if ia<75 or ia>105: fb.append({"type":"critical","message":"骨穿应垂直进针（90度）","severity":"critical","metric":"angle_score"})
        if features.get("hand_stability",0) > 2.0: fb.append({"type":"critical","message":"手部抖动明显，影响穿刺安全性","severity":"critical","metric":"stability"})
        if features.get("steady_pressure",0.8) < 0.5: fb.append({"type":"warning","message":"推进力度不均匀，应持续稳定施压","severity":"warning","metric":"pressure_score"})
    
        return fb if len(fb)>1 else fb+[{"type":"info","message":"操作完成","severity":"info"}]
    def reset(self):
        self.landmark_history = {}; self.feature_cache = {}
