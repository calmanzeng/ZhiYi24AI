"""
dressing_change Scoring Plugin
"""
import numpy as np
from typing import Dict, List, Any, Optional
class SkillScorer:
    def __init__(self, config=None):
        self.config = config or {}
        self.skill_name = "dressing_change"
        self.landmark_history = {}
        self.feature_cache = {}
    def extract_features(self, landmarks):
        f = {}

        lw = self._lm(landmarks, "left_wrist"); rw = self._lm(landmarks, "right_wrist")
        ls = self._lm(landmarks, "left_shoulder"); rs = self._lm(landmarks, "right_shoulder")
        le = self._lm(landmarks, "left_elbow"); re = self._lm(landmarks, "right_elbow")
        f["hand_distance"]=float(np.sqrt((lw[0]-rw[0])**2+(lw[1]-rw[1])**2))
        f["arm_elevation"]=((lw[1]-ls[1])+(rw[1]-rs[1]))/2
        if "left_wrist" in self.landmark_history and "right_wrist" in self.landmark_history:
            lt=self.landmark_history["left_wrist"][-30:]; rt=self.landmark_history["right_wrist"][-30:]
            if len(lt)>=10 and len(rt)>=10:
                ax=[p[0] for p in lt]+[p[0] for p in rt]; ay=[p[1] for p in lt]+[p[1] for p in rt]
                f["movement_scope"]=float((max(ax)-min(ax))*(max(ay)-min(ay)))
        if "left_wrist" in self.landmark_history:
            lt=self.landmark_history["left_wrist"][-10:]
            if len(lt)>=3:
                dp=[np.sqrt((lt[i][0]-lt[i-1][0])**2+(lt[i][1]-lt[i-1][1])**2) for i in range(1,len(lt))]
                f["smoothness"]=float(1.0/(1.0+np.var(dp)*10)) if dp else 0.8
    
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

        hd=features.get("hand_distance",0.3)
        m["sterile"]=round(90 if 0.2<=hd<=0.6 else (70 if hd<0.2 else 75),1)
        ae=features.get("arm_elevation",0.05)
        m["technique"]=round(85 if ae>0 else (70 if ae>-0.05 else 50),1)
        m["smoothness"]=round(features.get("smoothness",0.8)*100,1)
        ms=features.get("movement_scope",0.2)
        m["efficiency"]=round(85 if ms<0.3 else 65,1)
    
        return m
    def get_feedback(self, metrics, features):
        fb = [{"type":"overall","message":f"总分：{min(100,max(0,sum(m*0.25 for m in metrics.values()))):.0f}/100","severity":"info"}]

        if features.get("arm_elevation",0.05)<0: fb.append({"type":"critical","message":"手臂位置过低，注意无菌操作","severity":"critical","metric":"sterile"})
        if features.get("hand_distance",0.3)<0.15: fb.append({"type":"warning","message":"双手间距过近","severity":"warning","metric":"sterile"})
    
        return fb if len(fb)>1 else fb+[{"type":"info","message":"操作完成","severity":"info"}]
    def reset(self):
        self.landmark_history = {}; self.feature_cache = {}
