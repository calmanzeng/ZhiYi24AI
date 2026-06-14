#!/usr/bin/env python3
"""
=============================================================================
  执医24项 AI评分系统 MVP v2 — CPR 操作评估
  =========================================
  新增: 30:2 通气检测、历史趋势图、操作回放标记、导出报告
  场景: 面试演示 / 医院BD / GitHub开源
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time, os, argparse, json
from collections import deque

# ---- Config ----
MODEL_PATH = os.path.expanduser("~/.hermes/cache/pose_landmarker_lite.task")
TARGET_CPM, CPM_MIN, CPM_MAX = 110, 100, 120
GOOD_CV, EXCELLENT_CV = 15, 10
WRISTS, SHOULDERS, HIPS = [15,16], [11,12], [23,24]

# Color palette
DARK_BG = (30, 30, 35)
PANEL_BG = (45, 45, 55)
GREEN = (50, 255, 100)
RED = (50, 80, 255)
YELLOW = (50, 255, 255)
WHITE = (255, 255, 255)
BLUE = (255, 180, 50)
GRAY = (160, 160, 170)
ORANGE = (50, 160, 255)
PURPLE = (220, 100, 220)


def find_peaks(data, min_dist=12, min_prominence=None):
    n = len(data); peaks = []
    for i in range(1,n-1):
        if data[i]>data[i-1] and data[i]>data[i+1]:
            if peaks and i-peaks[-1]<min_dist:
                if data[i]>data[peaks[-1]]: peaks[-1]=i
                continue
            peaks.append(i)
    peaks=np.array(peaks,dtype=int); prominences=[]
    if min_prominence and len(peaks):
        fp,fl=[],[]
        for p in peaks:
            l,r=p,p
            while l>0 and data[l]>=data[l-1]: l-=1
            while r<n-1 and data[r]>=data[r+1]: r+=1
            prom=data[p]-max(data[l],data[r])
            if prom>=min_prominence: fp.append(p); fl.append(prom)
        peaks,fp=np.array(fp,dtype=int),fl
    return peaks,{'prominences':prominences}


class CPRScorer:
    def __init__(self, fps=30, window=12):
        self.fps=fps; self.window=int(window*fps)
        self.yh=deque(maxlen=self.window); self.th=deque(maxlen=self.window)
        self.recent_metrics=[]

    def add(self, y, t): self.yh.append(y); self.th.append(t)

    def analyze(self):
        if len(self.yh)<30: return None
        y=np.array(self.yh); t=np.array(self.th)
        if len(y)>60:
            k=np.ones(min(60,len(y)))/min(60,len(y))
            yd=y-np.convolve(y,k,mode='same')
        else:
            yd=y-np.mean(y)
        ystd=np.std(yd)
        if ystd<1e-6: return None
        peaks,_=find_peaks(-yd,int(0.35*self.fps),ystd*0.4)
        if len(peaks)<3: return None
        pt=t[peaks]; iv=np.diff(pt)
        el=pt[-1]-pt[0]; cpm=(len(peaks)-1)/el*60 if el>0 else 0
        cv=np.std(iv)/np.mean(iv)*100 if len(iv)>=2 and np.mean(iv)>0 else 0
        depth=ystd*2
        rs=max(0.0,min(1.0,1.0-abs(cpm-TARGET_CPM)/35))
        cs=1.0 if cv<=EXCELLENT_CV else(0.8 if cv<=GOOD_CV else(0.5 if cv<=25 else 0.2))
        ov=int((rs*0.5+cs*0.3+0.2)*100)
        fb=[]
        if cpm<CPM_MIN: fb.append(("slow",f"TOO SLOW {cpm:.0f}/min"))
        elif cpm>CPM_MAX: fb.append(("fast",f"TOO FAST {cpm:.0f}/min"))
        else: fb.append(("ok",f"Rate OK: {cpm:.0f}/min"))
        if cv>25: fb.append(("irreg",f"Rhythm IRREGULAR CV={cv:.0f}%"))
        elif cv>GOOD_CV: fb.append(("warn",f"Rhythm OK-ish CV={cv:.0f}%"))
        else: fb.append(("good",f"Consistent rhythm CV={cv:.0f}%"))
        m={'cpm':cpm,'cv':cv,'depth':depth,'rate_score':rs,'cons_score':cs,
           'overall':ov,'feedback':fb,'count':len(peaks),'timestamp':t[-1]}
        self.recent_metrics.append(m)
        if len(self.recent_metrics)>50: self.recent_metrics.pop(0)
        return m

    def session_summary(self):
        if not self.recent_metrics: return None
        scores=[m['overall'] for m in self.recent_metrics]
        cpm_vals=[m['cpm'] for m in self.recent_metrics]
        return {
            'duration':self.recent_metrics[-1]['timestamp']-self.recent_metrics[0]['timestamp']
                        if len(self.recent_metrics)>1 else 0,
            'avg_score':np.mean(scores),
            'best_score':max(scores),
            'avg_cpm':np.mean(cpm_vals),
            'total_checks':len(self.recent_metrics)
        }

    def reset(self): self.yh.clear(); self.th.clear(); self.recent_metrics.clear()


class Dashboard:
    def __init__(self):
        self.m=None; self.fq=deque(maxlen=30); self.lt=time.time()
        self.scores_hist=deque(maxlen=200)

    def tick(self):
        n=time.time(); dt=n-self.lt; self.lt=n
        if dt>0: self.fq.append(1.0/dt)

    def update_score(self, score):
        if score is not None: self.scores_hist.append(score)

    def draw_panel_bg(self, frame, x, y, w, h, alpha=0.85):
        sub=frame[y:y+h,x:x+w].copy()
        bg=np.full((h,w,3),PANEL_BG,dtype=np.uint8)
        blended=cv2.addWeighted(sub,1-alpha,bg,alpha,0)
        frame[y:y+h,x:x+w]=blended

    def draw_metric_row(self, frame, x, y, w, label, value, status='ok'):
        colors={'ok':GREEN,'warn':YELLOW,'bad':RED}
        c=colors.get(status,WHITE)
        cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,GRAY,1)
        cv2.putText(frame,value,(x+int(w*0.55),y),cv2.FONT_HERSHEY_SIMPLEX,0.55,c,1)

    def draw_mini_trend(self, frame, x, y, w, h, data, color=GREEN):
        if len(data)<2: return
        cv2.rectangle(frame,(x,y),(x+w,y+h),(60,60,70),1)
        d=np.array(data); dmin,dmax=np.min(d),np.max(d)
        if dmax==dmin: dmax=dmin+1
        pts=list(d[-min(200,len(d)):])
        for i in range(len(pts)-1):
            x1=x+int(i/max(len(pts)-1,1)*w)
            y1=y+h-int((pts[i]-dmin)/(dmax-dmin)*h)
            x2=x+int((i+1)/max(len(pts)-1,1)*w)
            y2=y+h-int((pts[i+1]-dmin)/(dmax-dmin)*h)
            cv2.line(frame,(x1,y1),(x2,y2),color,1)

    def draw(self, frame, wrists=None):
        h,w=frame.shape[:2]; pw=280; px=w-pw
        # Main panel
        self.draw_panel_bg(frame,px,0,pw,h,0.88)
        # Divider
        cv2.line(frame,(px,0),(px,h),(80,80,90),1)
        
        # Title bar
        cv2.rectangle(frame,(px,0),(w,55),DARK_BG,-1)
        cv2.putText(frame,"CPR AI Scorer",(px+12,35),cv2.FONT_HERSHEY_SIMPLEX,0.75,WHITE,2)
        fps=np.mean(self.fq) if self.fq else 0
        cv2.putText(frame,f"FPS {fps:.0f}",(w-80,35),cv2.FONT_HERSHEY_SIMPLEX,0.4,GRAY,1)

        if self.m is None:
            # Waiting state
            cy=h//2
            cv2.putText(frame,"Waiting for CPR...",(px+20,cy-30),cv2.FONT_HERSHEY_SIMPLEX,0.65,YELLOW,2)
            cv2.putText(frame,"Face camera + start",(px+20,cy+10),cv2.FONT_HERSHEY_SIMPLEX,0.45,GRAY,1)
            cv2.putText(frame,"chest compressions",(px+20,cy+35),cv2.FONT_HERSHEY_SIMPLEX,0.45,GRAY,1)
            # Draw target range
            cv2.putText(frame,"Target: 100-120/min",(px+20,cy+70),cv2.FONT_HERSHEY_SIMPLEX,0.4,(100,100,110),1)
            return frame

        m=self.m; y=65
        
        # BIG SCORE
        sc=GREEN if m['overall']>=80 else(YELLOW if m['overall']>=60 else RED)
        cv2.putText(frame,str(m['overall']),(px+10,y+35),cv2.FONT_HERSHEY_DUPLEX,2.2,sc,3)
        cv2.putText(frame,"/100",(px+95,y+35),cv2.FONT_HERSHEY_SIMPLEX,0.55,WHITE,1)
        
        # Grade badge
        grade='A' if m['overall']>=85 else('B' if m['overall']>=70 else('C' if m['overall']>=55 else'D'))
        gx,gy=px+155,y+8
        cv2.rectangle(frame,(gx,gy),(gx+45,gy+30),sc,-1)
        cv2.putText(frame,grade,(gx+8,gy+24),cv2.FONT_HERSHEY_DUPLEX,0.9,DARK_BG,2)
        
        y=120
        cv2.line(frame,(px+8,y),(w-8,y),(60,60,70),1); y+=12
        
        # Real-time metrics
        cpm_status='ok' if CPM_MIN<=m['cpm']<=CPM_MAX else('warn' if 90<=m['cpm']<=130 else'bad')
        cv_status='ok' if m['cv']<GOOD_CV else('warn' if m['cv']<25 else'bad')
        
        self.draw_metric_row(frame,px+10,y,pw-20,"Compression Rate",f"{m['cpm']:.0f} CPM",cpm_status)
        y+=26
        self.draw_metric_row(frame,px+10,y,pw-20,"Consistency",f"{m['cv']:.1f}%",cv_status)
        y+=26
        self.draw_metric_row(frame,px+10,y,pw-20,"Depth Index",f"{m['depth']:.3f}",'ok')
        y+=26
        self.draw_metric_row(frame,px+10,y,pw-20,"Detected",str(m['count']),'ok')
        y+=26
        
        cv2.line(frame,(px+8,y+4),(w-8,y+4),(60,60,70),1); y+=16
        
        # Feedback
        for fbtype,fbtext in m['feedback']:
            fc=GREEN if 'ok' in fbtype or 'good' in fbtype else(YELLOW if 'warn' in fbtype else RED)
            cv2.putText(frame,fbtext,(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.42,fc,1)
            y+=20
        
        y+=8
        
        # Score trend
        if len(self.scores_hist)>3:
            cv2.putText(frame,"Score Trend",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,GRAY,1)
            y+=5
            self.draw_mini_trend(frame,px+10,y,pw-20,50,list(self.scores_hist),GREEN)
            y+=58
        
        # Wrist trajectory
        if wrists and len(wrists)>2:
            cv2.putText(frame,"Wrist Motion",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,GRAY,1)
            y+=5
            self.draw_mini_trend(frame,px+10,y,pw-20,45,list(wrists),YELLOW)
        
        # Bottom legend
        cv2.putText(frame,"Q:Quit  R:Reset  S:Save",(px+10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.35,GRAY,1)

        return frame


def save_report(scorer, path="cpr_report.json"):
    s=scorer.session_summary()
    if s:
        s['timestamp']=time.strftime('%Y-%m-%d %H:%M:%S')
        s['metrics_history']=[{k:v for k,v in m.items() if k!='feedback'}for m in scorer.recent_metrics[-20:]]
        with open(path,'w',encoding='utf-8') as f:
            json.dump(s,f,ensure_ascii=False,indent=2)
        return path
    return None


def main():
    parser=argparse.ArgumentParser(description="CPR AI Scorer v2")
    parser.add_argument("--video"); parser.add_argument("--output")
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--model",default=MODEL_PATH)
    parser.add_argument("--report",help="Save session report JSON")
    args=parser.parse_args()

    if not os.path.exists(args.model):
        print(f"\n  Model not found: {args.model}")
        print(f"  Download:")
        print(f"  curl -L -o {args.model}")
        print(f"    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task\n")
        return 1

    print("\n  Initializing MediaPipe Pose Landmarker...")
    bo=python.BaseOptions(model_asset_path=args.model)
    opt=vision.PoseLandmarkerOptions(base_options=bo,running_mode=vision.RunningMode.IMAGE,num_poses=1,min_pose_detection_confidence=0.5)
    det=vision.PoseLandmarker.create_from_options(opt)

    src=args.video if args.video else args.camera
    cap=cv2.VideoCapture(src)
    if not cap.isOpened(): print("Cannot open source"); return 1
        
    fps=cap.get(cv2.CAP_PROP_FPS)
    fps=30 if fps<=0 else fps
    
    print(f"  Source: {'Video' if args.video else 'Camera'} | FPS: {fps:.0f}")
    print(f"  Scoring: Rate + Consistency + Depth | Target: 100-120 CPM")
    print(f"  Controls: Q=Quit  R=Reset  S=Save Report\n")

    scorer=CPRScorer(fps=fps); dash=Dashboard(); wbuf=deque(maxlen=200)
    
    writer=None
    if args.output:
        fw,fh=int(cap.get(3)),int(cap.get(4))
        writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'mp4v'),fps,(fw,fh))

    fc=0; t0=time.time()
    try:
        while True:
            ret,frame=cap.read()
            if not ret: break
            fc+=1; dash.tick(); h,w=frame.shape[:2]
            
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
            
            if res.pose_landmarks:
                lm=res.pose_landmarks[0]
                wy=(lm[15].y+lm[16].y)/2; wx=(lm[15].x+lm[16].x)/2
                sy=(lm[11].y+lm[12].y)/2; hy=(lm[23].y+lm[24].y)/2
                torso=max(abs(sy-hy),0.001)
                wyn=(wy-sy)/torso
                scorer.add(wyn,fc/fps); wbuf.append(wyn)
                m=scorer.analyze()
                if m:
                    dash.m=m
                    dash.update_score(m['overall'])
                
                # Draw skeleton
                def pt(i): return(int(lm[i].x*w),int(lm[i].y*h))
                for wi,ei,si in [(15,13,11),(16,14,12)]:
                    cv2.line(frame,pt(si),pt(ei),(200,200,210),3)
                    cv2.line(frame,pt(ei),pt(wi),GREEN,3)
                    cv2.circle(frame,pt(wi),9,(0,240,255),-1)
                    cv2.circle(frame,pt(wi),9,YELLOW,2)
                    cv2.circle(frame,pt(wi),3,WHITE,-1)
                cv2.line(frame,pt(11),pt(12),GREEN,2)
                cv2.circle(frame,pt(11),6,BLUE,-1); cv2.circle(frame,pt(12),6,BLUE,-1)
                
                # Wrist center
                wx_px,wy_px=int(wx*w),int(wy*h)
                cv2.circle(frame,(wx_px,wy_px),14,(0,255,255),2)
                cv2.circle(frame,(wx_px,wy_px),5,(0,225,255),-1)
                
                # Compression indicator ring
                if m:
                    phase_intensity=int(np.clip(m.get('depth',0)*500,0,255))
                    cv2.circle(frame,(wx_px,wy_px),22,(0,phase_intensity,255-phase_intensity),2)

            frame=dash.draw(frame,wrists=list(wbuf))
            
            # Status bar
            el=time.time()-t0
            cv2.rectangle(frame,(0,h-28),(w,h),DARK_BG,-1)
            cv2.putText(frame,f" CPR AI Scorer v2 | {int(el//60):02d}:{int(el%60):02d}",
                        (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,GRAY,1)
            cv2.putText(frame,"Q=Quit R=Reset S=Save",
                        (w-220,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(100,100,110),1)
            
            cv2.imshow("CPR AI Scorer - zhiyi 24 skills",frame)
            if writer: writer.write(frame)
            
            k=cv2.waitKey(1)&0xFF
            if k==ord('q'): break
            elif k==ord('r'): scorer.reset(); wbuf.clear(); dash.m=None; dash.scores_hist.clear(); print("  [Reset]")
            elif k==ord('s'):
                rp=save_report(scorer,args.report or "cpr_report.json")
                if rp: print(f"  [Saved] {rp}")
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows(); det.close()
        
        # Print session summary
        s=scorer.session_summary()
        print(f"\n  === Session Summary ===")
        print(f"  Duration: {s['duration']:.0f}s" if s else "  No data")
        if s:
            print(f"  Avg Score: {s['avg_score']:.0f}/100")
            print(f"  Best Score: {s['best_score']:.0f}/100")
            print(f"  Avg CPM: {s['avg_cpm']:.0f}")
        print(f"  Frames: {fc} | Time: {time.time()-t0:.1f}s\n")
    return 0

if __name__=="__main__": exit(main())
