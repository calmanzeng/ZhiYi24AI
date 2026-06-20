# 🩺 ZhiYi24AI — 执医24项AI-Native智能教学评估系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 用 AI 替代考官的主观评分 — 基于 MediaPipe 姿态估计的 CPR 操作实时评估系统

## 🎯 这是什么？

执业医师资格考试（执医）和住院医师规范化培训（住培）的 24 项临床技能考核，至今仍主要依赖**考官人工打分**——主观性强、标准不一、无法规模化。

这个项目是 **AI 驱动的客观技能评分系统** 的第一个 MVP，从最高频的 CPR（心肺复苏）操作开始：

- 📹 **输入**: 普通 USB 摄像头 + 真人操作
- 🧠 **AI**: MediaPipe Pose Landmarker (33 个身体关键点)
- 📊 **输出**: 按压频率 (CPM) + 节奏一致性 + 深度指数 + 整体评分 + 实时语音级反馈

## 🚀 5 分钟跑起来

```bash
# 1. 安装
pip install mediapipe opencv-python numpy

# 2. 下载模型
curl -L -o ~/.hermes/cache/pose_landmarker_lite.task   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

# 3. 运行（需要摄像头）
python mvp.py

# 可选:
python mvp.py --video cpr_demo.mp4           # 从视频评估
python mvp.py --output annotated.mp4          # 保存带标注输出
python mvp.py --report session_report.json   # 导出 Session 报告
```

## 📊 评分标准

| 维度 | 检测方式 | 满分条件 |
|------|---------|----------|
| **按压频率** | 手腕 Y 轴轨迹峰检测 → CPM | 100-120 CPM |
| **节奏一致性** | 峰间间隔变异系数 (CV%) | CV < 10% |
| **深度指数** | 手腕位移幅度 (相对躯干高度归一化) | 稳定且有力 |
| **整体评分** | 加权综合 (频率 50% + 一致性 30% + 深度 20%) | ≥ 85/100 |

## 🖥️ 界面预览

```
┌─────────────────────────────────────┬──────────────┐
│                                     │  CPR AI      │
│         Camera Feed                 │  ╔════════╗  │
│                                     │  ║  92/100║  │
│    ○ Shoulder ── ○ Shoulder        │  ╚════════╝  │
│         ╲          ╱                │  Grade: A     │
│          ╲        ╱                 │               │
│           ○ Elbow ○                 │  Rate: 108 OK │
│              ╲  ╱                   │  CV:   7.2%   │
│              ●●●  ← Hands           │  Depth: 0.12  │
│              Compression            │               │
│                                     │  ✅ Good rate │
│                                     │  ✅ Steady    │
│                                     │               │
│                                     │  [Score Trend]│
│                                     │  ╱╲╱╲╱╲     │
└─────────────────────────────────────┴──────────────┘
```

## 🔧 技术架构

```
Camera → OpenCV → MediaPipe Pose (33 landmarks)
                         │
                         ▼
              Wrist Trajectory Extraction
              (Y-axis, normalized by torso height)
                         │
                         ▼
              Peak Detection (pure NumPy)
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           CPM Rate    CV%       Depth
              │          │          │
              └──────────┼──────────┘
                         ▼
                 Overall Score
                         │
                         ▼
              Real-time HUD Overlay
              (+ Trend Graph, Feedback)
```

## 📦 零成本依赖

| 组件 | 用途 | 费用 |
|------|------|:--:|
| MediaPipe | 人体姿态估计 (33 关键点) | 免费 |
| OpenCV | 视频采集/显示 | 免费 |
| NumPy | 信号处理/峰检测 | 免费 |
| USB 摄像头 | 输入设备 | ~¥50 |

## 🗺️ 路线图

- [x] CPR 按压频率 + 一致性 + 深度评分
- [x] 实时 HUD + 趋势图
- [x] Session 报告导出 (JSON)
- [ ] 30:2 通气循环自动检测
- [ ] 按压深度厘米级精确估计（参考物标定）
- [ ] 扩展到体格检查（第 2 站）: 叩诊/触诊手法评分
- [ ] 扩展到其他操作（第 3 站）: 缝合/穿刺/气管插管
- [ ] Web 前端 (FastAPI + WebRTC 远程评估)
- [ ] 多人同时评估 (OSCE 考试场景)
- [ ] 培训数据平台 (个人成长曲线 + 科室仪表盘)

## 🏥 目标应用场景

- **三甲医院技能培训中心**: 住培生操作练习 + AI 自动评分
- **执业医师技能考试**: 替代/辅助考官主观评分
- **医学院教学**: 标准化操作教学反馈
- **护理培训**: CPR + 其他操作培训

## 👥 目标受众

这个项目的目标是向以下三类企业展示"AI + 执医技能"的可行性：

1. **医学教育平台**: 医视网/华医网/医博士 — 从内容分发升级为 AI 评分
2. **模拟设备厂商**: 天堰科技/医模科技/弘联 — 从硬件制造升级为智能评分
3. **AI 公司**: 科大讯飞/商汤 — 医疗场景的落地应用



## 🏗️ Skills Implemented (24/24) 🎉

| # | Skill | Category | Station |
|:-:|------|--------:|:-------:|
| 1 | 心肺复苏 (CPR) | Emergency | 3 |
| 2 | 缝合打结 (Suturing) | Surgical | 3 |
| 3 | 胸腔穿刺术 (Thoracentesis) | Procedural | 3 |
| 4 | 腰椎穿刺术 (Lumbar Puncture) | Procedural | 3 |
| 5 | 导尿术 (Catheterization) | Procedural | 3 |
| 6 | 气管插管术 (Intubation) | Emergency | 3 |
| 7 | 心肺叩诊 (Percussion) | Physical Exam | 2 |
| 8 | 无菌术 (Sterile Technique) | Surgical | 3 |
| 9 | 腹腔穿刺术 (Abdominal Paracentesis) | Procedural | 3 |
| 10 | 骨髓穿刺术 (Bone Marrow Puncture) | Procedural | 3 |
| 11 | 静脉穿刺术 (Venipuncture) | Procedural | 3 |
| 12 | 清创术 (Debridement) | Surgical | 3 |
| 13 | 骨折固定术 (Fracture Splinting) | Emergency | 3 |
| 14 | 换药术 (Dressing Change) | Surgical | 3 |
| 15 | 腹部触诊 (Abdominal Palpation) | Physical Exam | 2 |
| 16 | 吸氧术 (Oxygen Therapy) | Procedural | 3 |
| 17 | 一般检查 (General Physical Exam) | Physical Exam | 2 |
| 18 | 头颈部检查 (Head & Neck Exam) | Physical Exam | 2 |
| 19 | 胸部检查 (Chest Exam) | Physical Exam | 2 |
| 20 | 神经系统检查 (Neurological Exam) | Physical Exam | 2 |
| 21 | 脊柱四肢检查 (Spine & Extremities) | Physical Exam | 2 |
| 22 | 吸痰术 (Sputum Suctioning) | Procedural | 3 |
| 23 | 除颤术 (Defibrillation) | Emergency | 3 |
| 24 | 胃管置入术 (NG Tube Insertion) | Procedural | 3 |

**All 24 clinical skills of 执业医师资格考试 are now supported! ✅**24 skills target, now 16 implemented (67%). Remaining 8: 病史采集(LLM), 病例分析(LLM), 一般检查, 头颈部检查, 胸部检查, 神经系统检查, 脊柱四肢, 吸痰术, 除颤术, 胃管置入**
**

---
> 核心算法库已迁移至私有仓库 [ZhiYi24AI-core](https://github.com/calmanzeng/ZhiYi24AI-core)（需授权访问）
