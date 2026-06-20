"""
fracture_splinting Scoring Plugin
"""
import numpy as np
from typing import Dict, List, Any, Optional
class SkillScorer:
    def __init__(self, config=None):
        self.config = config or {}
        self.skill_name = "fracture_splinting"
        self.landmark_history = {}
        self.feature_cache = {}
    def extract_features(self, landmarks):
        f = {}

        lw = self._lm(landmarks, "left_wrist"); rw = self._lm(landmarks, "right_wrist")
        le = self._lm(landmarks, "left_elbow"); re = self._lm(landmarks, "right_elbow")
        lk = self._lm(landmarks, "left_knee"); rk = self._lm(landmarks, "right_knee")
        f["limb_stability"] = 0.0
        if "right_wrist" in self.landmark_history:
            rt = self.landmark_history["right_wrist"][-15:]
            if len(rt)>=5:
                xv=np.var([p[0] for p in rt]); yv=np.var([p[1] for p in rt])
                f["limb_stability"]=float(xv+yv)
        f["hand_distance"]=float(np.sqrt((lw[0]-rw[0])**2+(lw[1]-rw[1])**2))
        if "hand_distance" in self.landmark_history and len(self.landmark_history["hand_distance"])>=20:
            hd=self.landmark_history["hand_distance"][-20:]
            zc=sum(1 for i in range(1,len(hd)) if (hd[i]-np.mean(hd))*(hd[i-1]-np.mean(hd))<0)
            f["wrapping_pattern"]=min(1.0,zc/6) if zc>0 else 0
        if "right_wrist" in self.landmark_history:
            rt=self.landmark_history["right_wrist"][-15:]
            if len(rt)>=5:
                rng=max(p[0] for p in rt)-min(p[0] for p in rt)
                f["movement_control"]=float(rng)
    
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

        m["stability"]=round(max(40,100-features.get("limb_stability",0)*30),1)
        m["wrapping"]=round(features.get("wrapping_pattern",0.7)*100,1)
        mc=features.get("movement_control",0.1)
        m["control"]=round(90 if mc<0.15 else (75 if mc<0.3 else 60),1)
        hd=features.get("hand_distance",0.3)
        m["smoothness"]=round(85 if 0.2<=hd<=0.5 else 70,1)
    
        return m
    def get_feedback(self, metrics, features):
        fb = [{"type":"overall","message":f"总分：{min(100,max(0,sum(m*0.25 for m in metrics.values()))):.0f}/100","severity":"info"}]

        if features.get("limb_stability",0)>1.5: fb.append({"type":"critical","message":"患肢移动过大，制动不充分","severity":"critical","metric":"stability"})
        if features.get("wrapping_pattern",0.7)<0.4: fb.append({"type":"warning","message":"缠绕模式不规律","severity":"warning","metric":"wrapping"})
    
        return fb if len(fb)>1 else fb+[{"type":"info","message":"操作完成","severity":"info"}]
    def reset(self):
        self.landmark_history = {}; self.feature_cache = {}
