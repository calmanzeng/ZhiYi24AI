"""
oxygen_therapy Scoring Plugin
"""
import numpy as np
from typing import Dict, List, Any, Optional
class SkillScorer:
    def __init__(self, config=None):
        self.config = config or {}
        self.skill_name = "oxygen_therapy"
        self.landmark_history = {}
        self.feature_cache = {}
    def extract_features(self, landmarks):
        f = {}

        nose = self._lm(landmarks, "nose")
        lw = self._lm(landmarks, "left_wrist"); rw = self._lm(landmarks, "right_wrist")
        ls = self._lm(landmarks, "left_shoulder"); rs = self._lm(landmarks, "right_shoulder")
        le = self._lm(landmarks, "left_elbow"); re = self._lm(landmarks, "right_elbow")
        f["hand_face_distance"]=float(np.sqrt((rw[0]-nose[0])**2+(rw[1]-nose[1])**2))
        f["dom_x"]=rw[0]; f["dom_y"]=rw[1]
        if "dom_x" in self.landmark_history and len(self.landmark_history["dom_x"])>=10:
            xh=self.landmark_history["dom_x"][-15:]; yh=self.landmark_history["dom_y"][-15:]
            f["hand_stability"]=float(np.sqrt(np.var(xh)+np.var(yh)))
        if "right_wrist" in self.landmark_history:
            rt=self.landmark_history["right_wrist"][-10:]
            if len(rt)>=3:
                dp=[np.sqrt((rt[i][0]-rt[i-1][0])**2+(rt[i][1]-rt[i-1][1])**2) for i in range(1,len(rt))]
                f["movement_smoothness"]=float(1.0/(1.0+np.var(dp)*10)) if dp else 0.8
        f["elbow_angle"]=self._a(rs, re, rw)
    
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

        hfd=features.get("hand_face_distance",0.3)
        m["positioning"]=round(90 if 0.1<=hfd<=0.4 else (70 if hfd<0.1 else 60),1)
        m["stability"]=round(max(40,100-features.get("hand_stability",0)*30),1)
        m["smoothness"]=round(features.get("movement_smoothness",0.8)*100,1)
        ea=features.get("elbow_angle",90)
        m["posture"]=round(85 if 60<=ea<=120 else 65,1)
    
        return m
    def get_feedback(self, metrics, features):
        fb = [{"type":"overall","message":f"总分：{min(100,max(0,sum(m*0.25 for m in metrics.values()))):.0f}/100","severity":"info"}]

        hfd=features.get("hand_face_distance",0.3)
        if hfd>0.5: fb.append({"type":"info","message":"手离面部太远，调整操作距离","severity":"info","metric":"positioning"})
        if features.get("hand_stability",0)>1.5: fb.append({"type":"warning","message":"手部稳定性不足","severity":"warning","metric":"stability"})
    
        return fb if len(fb)>1 else fb+[{"type":"info","message":"操作完成","severity":"info"}]
    def reset(self):
        self.landmark_history = {}; self.feature_cache = {}
