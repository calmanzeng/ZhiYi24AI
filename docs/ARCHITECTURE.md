# 执医24项 AI-Native 技能训练框架 — 架构设计

## 核心设计理念

> **声明式技能定义 + 可插拔 AI 引擎 + 自动数据回流**

任何人都可以通过编写 `skill.yaml` + `scoring.py` 来贡献新技能。
框架负责：视频采集、AI推理、数据管道、UI渲染、数据回流。
开发者只定义：关键点、特征提取、评分规则、反馈逻辑。

## 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│                  CLI / API / GUI Layer                   │
├─────────────────────────────────────────────────────────┤
│                    Framework Core                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │Skill     │  │Pipeline  │  │Data Hub              │  │
│  │Registry  │  │Manager   │  │(local + sync)        │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   AI Backend Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │MediaPipe │  │YOLO      │  │Custom ONNX/TFLite    │  │
│  │Pose/Hand │  │Object    │  │Models                │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                  Input / Output Layer                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │Camera    │  │Video File│  │RTSP / Network        │  │
│  │(USB/IP)  │  │          │  │Stream (future)       │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 目录结构

```
cpr-ai-scorer/
├── hermes_skills/                # 框架核心包
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py             # Pipeline 引擎
│   │   ├── registry.py           # 技能注册表
│   │   ├── data_hub.py           # 数据存储 + 回流
│   │   └── dashboard.py          # UI 渲染
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── pose_landmarker.py    # MediaPipe 姿态
│   │   └── models.py             # 模型管理
│   └── io/
│       ├── __init__.py
│       └── video_source.py       # 摄像头/视频/网络流
│
├── skills/                       # 技能插件目录
│   ├── cpr/
│   │   ├── skill.yaml
│   │   └── scoring.py
│   ├── abdominal_palpation/      # 下一个技能
│   │   ├── skill.yaml
│   │   └── scoring.py
│   └── ...
│
├── tests/                        # 测试
├── docs/                         # 文档
├── mvp.py                        # 旧版（保留兼容）
├── demo.py                       # 演示入口
└── README.md
```
