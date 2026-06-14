#!/usr/bin/env python3
"""
执医24项 AI-Native 框架 — 入口
================================
用法:
  python main.py                      # 默认运行 CPR
  python main.py --skill cpr          # 指定技能
  python main.py --list               # 列出所有技能
  python main.py --skill cpr --video test.mp4 --output out.mp4
"""
import sys
import os
import argparse

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(__file__))

from hermes_skills.core.engine import SkillPipeline
from hermes_skills.core.registry import SkillRegistry
from hermes_skills.core.dashboard import Dashboard


def main():
    parser = argparse.ArgumentParser(description="执医24项 AI-Native 技能训练框架")
    parser.add_argument("--skill", type=str, default="cpr", help="技能名称")
    parser.add_argument("--list", action="store_true", help="列出所有技能")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, help="输出带标注的视频")
    parser.add_argument("--headless", action="store_true", help="无窗口模式")
    parser.add_argument("--skills-dir", type=str, default="./skills")
    args = parser.parse_args()
    
    # 列出技能
    if args.list:
        registry = SkillRegistry(args.skills_dir)
        skills = registry.list_skills()
        print(f"\n 已注册技能 ({len(skills)}):\n")
        for s in skills:
            icon = "✅" if s["has_scorer"] else "⚠️"
            print(f"  {icon} {s['name']:<25} {s['display_name']:<25} [{s['category']}]")
        print()
        return 0
    
    print(f"\n  执医24项 AI-Native 框架 v0.1.0")
    print(f"  技能: {args.skill}\n")
    
    # 创建 Pipeline
    pipe = SkillPipeline(args.skill, skills_dir=args.skills_dir)
    
    # Dashboard
    dash = Dashboard()
    
    def render_callback(frame, metrics, landmarks, feedback, wrist_hist):
        dash.tick()
        if metrics:
            ov = metrics.get("overall", {}).get("value")
            dash.update_score(ov)
        return dash.render(frame, metrics, landmarks, feedback, wrist_hist)
    
    def metrics_callback(metrics):
        ov = metrics.get("overall", {}).get("value", 0)
        cpm = metrics.get("compression_rate", {}).get("value", 0)
        cv = metrics.get("rhythm_consistency", {}).get("value", 0)
        print(f"\r  分数:{ov:3d}  频率:{cpm:3.0f}CPM  一致性:{cv:4.1f}%  ",
              end="", flush=True)
    
    # 运行
    try:
        sid = pipe.run(
            camera=args.camera,
            video_path=args.video,
            headless=args.headless,
            on_frame=render_callback,
            on_metrics=metrics_callback if args.headless else None,
        )
        
        stats = pipe.data_hub.get_stats(args.skill)
        print(f"\n\n  === 训练统计 ===")
        print(f"  总次数: {stats['total_sessions']}")
        print(f"  平均分: {stats['avg_score']}")
        print(f"  最高分: {stats['best_score']}")
        print(f"  总时长: {stats['total_duration']} 分钟\n")
        
    except KeyboardInterrupt:
        print("\n  退出")
    except Exception as e:
        print(f"\n  ❌ {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
