# 🩺 执医24项 AI 评分系统 — CPR 操作评估 MVP

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **此仓库为个人研究项目，目前缺乏足够的医疗操作数据源来精进模型算法。**
> 如有任何交流意见、合作意向或数据资源，欢迎邮件联系：**keyneszeng@gmail.com**



> 用 AI 替代考官的主观评分 — 基于 MediaPipe 姿态估计的 CPR 操作实时评估系统。
>
> **纯 Python + 普通 USB 摄像头，零专用硬件，4 小时搭建。**

## 🎬 30秒演示

![CPR AI Scorer Demo](demo.gif)

*正常 CPR → 故意加速 → 评分实时变化 → 恢复正常*

## 🎯 这是什么？

执业医师资格考试（执医）和住院医师规范化培训（住培）的 24 项临床技能考核，至今仍主要依赖**考官人工打分**——主观性强、标准不一、无法规模化。

这个项目是 **AI 驱动的客观技能评分系统** 的第一个 MVP，从最高频的 CPR（心肺复苏）操作开始：

- 📹 **输入**: 普通 USB 摄像头 (720p) + 真人操作
- 🧠 **AI**: MediaPipe Pose Landmarker (33 个身体关键点) → 手腕轨迹分析
- 📊 **输出**: 按压频率 (CPM) + 节奏一致性 (CV%) + 深度指数 + 整体评分 (0-100) + 实时反馈

## 🚀 3 分钟跑起来

```bash
# 1. 安装依赖
pip install mediapipe opencv-python numpy

# 2. 下载 AI 模型 (5.5 MB)
curl -L -o ~/.hermes/cache/pose_landmarker_lite.task   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

# 3. 运行
python mvp.py

# 可选:
python mvp.py --video cpr_demo.mp4           # 从视频评估
python mvp.py --output annotated.mp4          # 保存带标注输出
python mvp.py --report session_report.json   # 导出 Session 报告
```

**快捷键**: Q 退出 | R 重置评分 | S 保存报告

## 📊 评分标准

| 维度 | 检测方式 | 满分条件 |
|------|---------|----------|
| **按压频率** | 手腕 Y 轴轨迹峰检测 → CPM | 100-120 CPM (target: 110) |
| **节奏一致性** | 峰间间隔变异系数 (CV%) | CV < 10% |
| **深度指数** | 手腕位移幅度 (躯干高度归一化) | 稳定且有力 |
| **整体评分** | 频率(50%) + 一致性(30%) + 深度(20%) | ≥ 85/100 → A 级 |

## 🖥️ 界面

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
│              ●●●  ← Hands          │  Depth: 0.12  │
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
USB Camera → OpenCV → MediaPipe Pose (33 landmarks)
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

## 📦 零成本

| 组件 | 用途 | 费用 |
|------|------|:--:|
| MediaPipe | 人体姿态估计 (33 关键点) | 免费 |
| OpenCV | 视频采集/显示/输出 | 免费 |
| NumPy | 信号处理/峰检测 | 免费 |
| USB 摄像头 | 输入设备 | ~¥50 |
| **总计** | | **¥50** |

## 🗺️ 路线图

- [x] CPR 按压频率 + 一致性 + 深度评分
- [x] 实时 HUD + 趋势图 + 等级评定
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

## 💼 商业潜力

| 产品形态 | 当前价格 | AI 升级后 | 增量 |
|---------|:---:|:---:|:---:|
| OSCE 考站 (硬件) | 50万/套 | 80-120万/套 | +60-140% |
| 住培管理系统 (软件) | 10-20万/年 | 30-50万/年 | +150% |
| AI 评分 SaaS | 0 | 15-30万/院/年 | 全新收入 |

> 目标企业: 天堰科技 (300314) / 医模科技 (300331) / 弘联医学 / 科大讯飞
> 详见 [TARGET_COMPANIES.md](TARGET_COMPANIES.md)

## 📄 License

MIT — 开源免费，商用友好

---

**Made as a job-seeking / BD demo tool. 4 hours from idea to working MVP.**
